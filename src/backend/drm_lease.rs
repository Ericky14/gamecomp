//! DRM lease backend — DRM output via a leased fd, Wayland input from host.
//!
//! This hybrid backend is used when gamecomp is launched by a parent compositor
//! (e.g., cosmic-comp) that grants a DRM lease for the primary display. The
//! lease fd provides exclusive access to a connector, CRTC, and plane set —
//! enabling VRR, tearing control, and direct scanout with zero compositor hops.
//!
//! Input arrives via a Wayland client connection to the parent compositor's
//! `wl_seat`. This separates concerns: the parent retains input device ownership
//! (libinput/libseat) while gamecomp owns the display pipeline.
//!
//! The backend decomposes into:
//! - **DRM output** — reuses the same atomic modesetting logic as [`DrmBackend`]
//!   but operates on a lease fd instead of a directly-opened device.
//! - **Wayland input thread** — connects to the host compositor, binds `wl_seat`,
//!   and forwards keyboard/pointer events via channel.

use std::collections::HashMap;
use std::os::unix::io::{AsFd, BorrowedFd, OwnedFd, RawFd};

use anyhow::{Context, bail};
use drm::Device as DrmDeviceTrait;
use drm::buffer::PlanarBuffer;
use drm::control::{
    AtomicCommitFlags, Device as ControlDevice, FbCmd2Flags, Mode, PlaneType, connector, crtc,
    plane, property,
};
use drm_fourcc::{DrmFormat, DrmFourcc, DrmModifier};
use tracing::{debug, info, warn};

use super::drm::{GbmOutputBuffer, InFormatsBlobHeader, InFormatsModifier};
use super::{
    Backend, BackendCaps, BackendError, ConnectorInfo, DmaBuf, DmaBufPlane, FlipResult, Framebuffer,
};
use super::wayland::WaylandEvent;

// ─── DRM device wrappers for lease fd ───────────────────────────────

/// Owns the DRM lease file descriptor.
struct LeaseDevice(OwnedFd);

impl AsFd for LeaseDevice {
    #[inline(always)]
    fn as_fd(&self) -> BorrowedFd<'_> {
        self.0.as_fd()
    }
}

impl DrmDeviceTrait for LeaseDevice {}
impl ControlDevice for LeaseDevice {}

/// Borrowed reference to the lease device fd.
struct LeaseRef<'a>(BorrowedFd<'a>);

impl<'a> AsFd for LeaseRef<'a> {
    #[inline(always)]
    fn as_fd(&self) -> BorrowedFd<'_> {
        self.0
    }
}

impl DrmDeviceTrait for LeaseRef<'_> {}
impl ControlDevice for LeaseRef<'_> {}

// ─── Cached DRM property handles ────────────────────────────────────

#[derive(Debug, Default)]
struct PlaneProps {
    fb_id: Option<property::Handle>,
    crtc_id: Option<property::Handle>,
    crtc_x: Option<property::Handle>,
    crtc_y: Option<property::Handle>,
    crtc_w: Option<property::Handle>,
    crtc_h: Option<property::Handle>,
    src_x: Option<property::Handle>,
    src_y: Option<property::Handle>,
    src_w: Option<property::Handle>,
    src_h: Option<property::Handle>,
    in_fence_fd: Option<property::Handle>,
}

#[derive(Debug, Default)]
struct CrtcProps {
    active: Option<property::Handle>,
    mode_id: Option<property::Handle>,
    vrr_enabled: Option<property::Handle>,
}

#[derive(Debug, Default)]
struct ConnectorProps {
    crtc_id: Option<property::Handle>,
    vrr_capable: Option<property::Handle>,
}

#[derive(Debug)]
struct PlaneState {
    handle: plane::Handle,
    kind: PlaneType,
    props: PlaneProps,
    formats: Vec<DrmFormat>,
    possible_crtcs: drm::control::CrtcListFilter,
}

#[derive(Debug)]
struct OutputState {
    connector: connector::Handle,
    crtc: crtc::Handle,
    mode: Mode,
    connector_props: ConnectorProps,
    crtc_props: CrtcProps,
    primary_plane: plane::Handle,
    vrr_capable: bool,
    vrr_enabled: bool,
    active: bool,
}

// ─── DRM Lease Backend ──────────────────────────────────────────────

/// DRM lease backend: DRM output via lease fd + Wayland input from host.
pub struct DrmLeaseBackend {
    /// DRM lease file descriptor.
    fd: OwnedFd,
    /// All planes discovered on the lease.
    planes: HashMap<plane::Handle, PlaneState>,
    /// Active output (single connector from the lease).
    output: Option<OutputState>,
    /// Connector info exposed to the compositor.
    connector_info: Vec<ConnectorInfo>,
    /// Backend capabilities.
    caps: BackendCaps,
    /// Scanout formats from the primary plane.
    scanout_formats: Vec<DrmFormat>,
    /// Page flip pending flag.
    flip_pending: bool,
    /// First atomic commit needs ALLOW_MODESET.
    needs_modeset: bool,
    /// Host Wayland display name for input (e.g., "wayland-0").
    host_wayland_display: Option<String>,
    /// Channel for receiving input events from the Wayland input thread.
    input_rx: Option<std::sync::mpsc::Receiver<WaylandEvent>>,
    /// Handle to the Wayland input thread.
    _input_thread: Option<std::thread::JoinHandle<()>>,
}

impl DrmLeaseBackend {
    /// Create a new DRM lease backend from an inherited lease fd.
    ///
    /// The fd must be a valid DRM lease obtained from the parent compositor
    /// via `drmModeCreateLease()`. It grants access to a specific connector,
    /// CRTC, and set of planes.
    pub fn new(lease_fd: OwnedFd, host_wayland_display: Option<String>) -> Self {
        Self {
            fd: lease_fd,
            planes: HashMap::new(),
            output: None,
            connector_info: Vec::new(),
            caps: BackendCaps::default(),
            scanout_formats: Vec::new(),
            flip_pending: false,
            needs_modeset: true,
            host_wayland_display,
            input_rx: None,
            _input_thread: None,
        }
    }

    /// Discover planes available on the lease.
    fn enumerate_planes(&mut self) -> anyhow::Result<()> {
        let dev = LeaseRef(self.fd.as_fd());
        let plane_handles = dev
            .plane_handles()
            .context("failed to enumerate planes on lease fd")?;

        for &handle in plane_handles.as_slice() {
            let plane = match dev.get_plane(handle) {
                Ok(p) => p,
                Err(_) => continue, // Plane not part of the lease.
            };

            let props = match dev.get_properties(handle) {
                Ok(p) => p,
                Err(_) => continue,
            };

            let mut kind = PlaneType::Overlay;
            let mut plane_props = PlaneProps::default();
            for (&prop_handle, &value) in &props {
                let prop_info = dev.get_property(prop_handle).ok();
                if let Some(info) = prop_info {
                    match info.name().to_str().unwrap_or("") {
                        "type" => {
                            kind = match value {
                                1 => PlaneType::Primary,
                                2 => PlaneType::Cursor,
                                _ => PlaneType::Overlay,
                            };
                        }
                        "FB_ID" => plane_props.fb_id = Some(prop_handle),
                        "CRTC_ID" => plane_props.crtc_id = Some(prop_handle),
                        "CRTC_X" => plane_props.crtc_x = Some(prop_handle),
                        "CRTC_Y" => plane_props.crtc_y = Some(prop_handle),
                        "CRTC_W" => plane_props.crtc_w = Some(prop_handle),
                        "CRTC_H" => plane_props.crtc_h = Some(prop_handle),
                        "SRC_X" => plane_props.src_x = Some(prop_handle),
                        "SRC_Y" => plane_props.src_y = Some(prop_handle),
                        "SRC_W" => plane_props.src_w = Some(prop_handle),
                        "SRC_H" => plane_props.src_h = Some(prop_handle),
                        "IN_FENCE_FD" => plane_props.in_fence_fd = Some(prop_handle),
                        _ => {}
                    }
                }
            }

            let formats: Vec<_> = plane
                .formats()
                .iter()
                .map(|&f| DrmFormat {
                    code: DrmFourcc::try_from(f).unwrap_or(DrmFourcc::Argb8888),
                    modifier: DrmModifier::Linear,
                })
                .collect();

            let possible_crtcs = plane.possible_crtcs();

            debug!(
                ?handle,
                ?kind,
                num_formats = formats.len(),
                "lease: discovered plane"
            );

            self.planes.insert(
                handle,
                PlaneState {
                    handle,
                    kind,
                    props: plane_props,
                    formats,
                    possible_crtcs,
                },
            );
        }

        info!(count = self.planes.len(), "lease: discovered planes");
        Ok(())
    }

    /// Discover the leased connector and CRTC.
    fn enumerate_output(&mut self) -> anyhow::Result<()> {
        let dev = LeaseRef(self.fd.as_fd());
        let resources = dev
            .resource_handles()
            .context("failed to get DRM resources on lease fd")?;

        for &conn_handle in resources.connectors() {
            let conn = match dev.get_connector(conn_handle, false) {
                Ok(c) => c,
                Err(_) => continue, // Not part of lease.
            };

            // Find the CRTC (the lease should include exactly one).
            let encoder = conn.current_encoder().and_then(|e| dev.get_encoder(e).ok());
            let crtc_handle = encoder
                .and_then(|e| e.crtc())
                .or_else(|| resources.crtcs().first().copied());

            let Some(crtc_handle) = crtc_handle else {
                warn!(?conn_handle, "lease: no CRTC for connector");
                continue;
            };

            let mode = conn
                .modes()
                .iter()
                .find(|m| {
                    m.mode_type()
                        .contains(drm::control::ModeTypeFlags::PREFERRED)
                })
                .or_else(|| conn.modes().first())
                .copied();

            let Some(mode) = mode else {
                warn!(?conn_handle, "lease: no modes available");
                continue;
            };

            // Cache connector properties.
            let mut conn_props = ConnectorProps::default();
            if let Ok(props) = dev.get_properties(conn_handle) {
                for (&prop_handle, &_value) in &props {
                    if let Ok(info) = dev.get_property(prop_handle) {
                        match info.name().to_str().unwrap_or("") {
                            "CRTC_ID" => conn_props.crtc_id = Some(prop_handle),
                            "vrr_capable" => conn_props.vrr_capable = Some(prop_handle),
                            _ => {}
                        }
                    }
                }
            }

            // Cache CRTC properties.
            let mut crtc_props = CrtcProps::default();
            if let Ok(props) = dev.get_properties(crtc_handle) {
                for (&prop_handle, &_value) in &props {
                    if let Ok(info) = dev.get_property(prop_handle) {
                        match info.name().to_str().unwrap_or("") {
                            "ACTIVE" => crtc_props.active = Some(prop_handle),
                            "MODE_ID" => crtc_props.mode_id = Some(prop_handle),
                            "VRR_ENABLED" => crtc_props.vrr_enabled = Some(prop_handle),
                            _ => {}
                        }
                    }
                }
            }

            let vrr_capable = conn_props.vrr_capable.is_some();

            // Find primary plane compatible with this CRTC.
            let primary_plane = self
                .planes
                .values()
                .find(|p| {
                    p.kind == PlaneType::Primary
                        && resources
                            .filter_crtcs(p.possible_crtcs)
                            .contains(&crtc_handle)
                })
                .map(|p| p.handle);

            let Some(primary_plane) = primary_plane else {
                warn!(?crtc_handle, "lease: no primary plane for CRTC");
                continue;
            };

            let conn_name = format!("{:?}-{}", conn.interface(), conn.interface_id());
            let physical_size_mm = conn.size().unwrap_or((0, 0));

            info!(
                connector = %conn_name,
                mode = ?mode,
                vrr_capable,
                "lease: discovered output"
            );

            self.connector_info.push(ConnectorInfo {
                handle: conn_handle,
                crtc: crtc_handle,
                name: conn_name,
                mode,
                physical_size_mm,
                vrr_enabled: false,
            });

            self.output = Some(OutputState {
                connector: conn_handle,
                crtc: crtc_handle,
                mode,
                connector_props: conn_props,
                crtc_props,
                primary_plane,
                vrr_capable,
                vrr_enabled: false,
                active: false,
            });

            // Leases grant exactly one connector — done.
            break;
        }

        if self.output.is_none() {
            bail!("no usable output in DRM lease");
        }

        // Collect scanout formats.
        if let Some(ref output) = self.output
            && let Some(plane) = self.planes.get(&output.primary_plane)
        {
            self.scanout_formats = plane.formats.clone();
        }

        Ok(())
    }

    /// Spawn the Wayland input thread that connects to the host compositor.
    fn start_input_thread(&mut self) -> anyhow::Result<()> {
        let display_name = self
            .host_wayland_display
            .clone()
            .or_else(|| std::env::var("WAYLAND_DISPLAY").ok())
            .context("no host Wayland display for DRM lease input")?;

        let (tx, rx) = std::sync::mpsc::channel::<WaylandEvent>();
        self.input_rx = Some(rx);

        let thread = std::thread::Builder::new()
            .name("gamecomp-lease-input".to_string())
            .spawn(move || {
                if let Err(e) = run_input_thread(&display_name, tx) {
                    warn!(?e, "lease input thread exited with error");
                }
            })
            .context("failed to spawn lease input thread")?;

        self._input_thread = Some(thread);
        info!("lease Wayland input thread started");
        Ok(())
    }

    /// Query primary plane IN_FORMATS modifiers (same logic as DrmBackend).
    pub fn query_primary_plane_modifiers(&self, format: DrmFourcc) -> Vec<u64> {
        let dev = LeaseRef(self.fd.as_fd());

        let primary_plane = match self.output.as_ref() {
            Some(o) => o.primary_plane,
            None => return Vec::new(),
        };

        let props = match dev.get_properties(primary_plane) {
            Ok(p) => p,
            Err(_) => return Vec::new(),
        };

        let mut in_formats_blob_id: Option<u64> = None;
        for (&prop_handle, &value) in &props {
            if let Ok(info) = dev.get_property(prop_handle)
                && info.name().to_str() == Ok("IN_FORMATS")
            {
                in_formats_blob_id = Some(value);
                break;
            }
        }

        let Some(blob_id) = in_formats_blob_id else {
            return Vec::new();
        };

        let blob_data = match dev.get_property_blob(blob_id) {
            Ok(d) => d,
            Err(_) => return Vec::new(),
        };

        if blob_data.len() < std::mem::size_of::<InFormatsBlobHeader>() {
            return Vec::new();
        }

        // SAFETY: blob_data is a kernel DRM blob with drm_format_modifier_blob
        // layout. We verified the minimum size above. All fields are POD u32.
        let header = unsafe { &*(blob_data.as_ptr() as *const InFormatsBlobHeader) };
        let target_fourcc = format as u32;
        let mut modifiers = Vec::new();

        for i in 0..header.count_modifiers {
            let mod_offset = header.modifiers_offset as usize
                + (i as usize) * std::mem::size_of::<InFormatsModifier>();
            if mod_offset + std::mem::size_of::<InFormatsModifier>() > blob_data.len() {
                break;
            }
            // SAFETY: bounds-checked above; InFormatsModifier is repr(C) POD.
            let mod_entry =
                unsafe { &*(blob_data.as_ptr().add(mod_offset) as *const InFormatsModifier) };

            for j in 0..64u32 {
                if mod_entry.formats & (1u64 << j) != 0 {
                    let fmt_idx = (j + mod_entry.offset) as usize;
                    let fmt_byte_offset = header.formats_offset as usize + fmt_idx * 4;
                    if fmt_byte_offset + 4 > blob_data.len() {
                        continue;
                    }
                    // SAFETY: bounds-checked above; format codes are u32.
                    let fourcc =
                        unsafe { *(blob_data.as_ptr().add(fmt_byte_offset) as *const u32) };
                    if fourcc == target_fourcc {
                        modifiers.push(mod_entry.modifier);
                    }
                }
            }
        }

        info!(count = modifiers.len(), "lease: queried IN_FORMATS modifiers");
        modifiers
    }

    /// Drain input events from the Wayland input thread.
    pub fn drain_input(&self) -> Vec<WaylandEvent> {
        let Some(ref rx) = self.input_rx else {
            return Vec::new();
        };
        let mut events = Vec::new();
        while let Ok(event) = rx.try_recv() {
            events.push(event);
        }
        events
    }

    /// Allocate GBM output buffers (same logic as DrmBackend).
    pub fn allocate_gbm_output_buffers(
        &mut self,
        count: usize,
        width: u32,
        height: u32,
        modifiers: &[u64],
    ) -> anyhow::Result<Vec<GbmOutputBuffer>> {
        use gbm::{BufferObjectFlags, Device as GbmDevice};
        use std::os::unix::io::AsRawFd;

        let gbm: GbmDevice<LeaseRef> = GbmDevice::new(LeaseRef(self.fd.as_fd()))
            .context("failed to create GBM device on lease fd")?;

        let drm_modifiers: Vec<DrmModifier> =
            modifiers.iter().map(|&m| DrmModifier::from(m)).collect();

        let mut outputs = Vec::with_capacity(count);

        for i in 0..count {
            let bo = gbm
                .create_buffer_object_with_modifiers2::<()>(
                    width,
                    height,
                    gbm::Format::Xrgb8888,
                    drm_modifiers.iter().copied(),
                    BufferObjectFlags::SCANOUT | BufferObjectFlags::RENDERING,
                )
                .with_context(|| format!("GBM buffer alloc failed for lease output {i}"))?;

            let bo_modifier = bo.modifier();
            let bo_stride = bo.stride();
            let bo_offset = bo.offset(0);
            let bo_fd = bo
                .fd()
                .with_context(|| format!("failed to export GBM fd for lease output {i}"))?;

            let dev = LeaseRef(self.fd.as_fd());
            let flags = if bo_modifier != DrmModifier::Invalid {
                FbCmd2Flags::MODIFIERS
            } else {
                FbCmd2Flags::empty()
            };
            let fb_handle = dev
                .add_planar_framebuffer(&bo, flags)
                .with_context(|| format!("add_planar_framebuffer failed for lease output {i}"))?;

            let fb = Framebuffer {
                handle: fb_handle,
                format: DrmFourcc::Xrgb8888,
                modifier: bo_modifier,
                size: (width, height),
            };

            let raw_fd = bo_fd.as_raw_fd();
            let dmabuf = DmaBuf {
                width,
                height,
                format: DrmFourcc::Xrgb8888,
                modifier: bo_modifier,
                planes: vec![DmaBufPlane {
                    fd: raw_fd,
                    offset: bo_offset,
                    stride: bo_stride,
                }],
            };

            // Forget the GBM BO to prevent its destructor from freeing the
            // GEM handle. The DRM FB and DMA-BUF fd keep the backing memory alive.
            std::mem::forget(bo);

            outputs.push(GbmOutputBuffer {
                fb,
                dmabuf,
                _fd: bo_fd,
            });
        }

        info!(
            count = outputs.len(),
            width,
            height,
            "lease: allocated GBM output buffers"
        );
        Ok(outputs)
    }

    /// Force a modeset on the next present.
    pub fn force_modeset(&mut self) {
        self.needs_modeset = true;
        self.flip_pending = false;
    }

    /// Destroy a framebuffer handle.
    pub fn destroy_framebuffer(&self, fb: drm::control::framebuffer::Handle) {
        let dev = LeaseRef(self.fd.as_fd());
        if let Err(e) = dev.destroy_framebuffer(fb) {
            warn!(?fb, ?e, "lease: failed to destroy framebuffer");
        }
    }
}

// ─── Backend trait implementation ───────────────────────────────────

impl Backend for DrmLeaseBackend {
    fn init(&mut self) -> anyhow::Result<()> {
        info!("initializing DRM lease backend");

        {
            let dev = LeaseRef(self.fd.as_fd());
            dev.set_client_capability(drm::ClientCapability::Atomic, true)
                .context("lease fd: kernel does not support atomic modesetting")?;
            dev.set_client_capability(drm::ClientCapability::UniversalPlanes, true)
                .context("lease fd: kernel does not support universal planes")?;
        }

        self.enumerate_planes()?;
        self.enumerate_output()?;

        self.caps.modifiers = true;
        self.caps.vrr = self
            .output
            .as_ref()
            .is_some_and(|o| o.vrr_capable);
        self.caps.explicit_sync = self.planes.values().any(|p| p.props.in_fence_fd.is_some());

        info!(?self.caps, "lease: backend capabilities");

        // Start input thread to receive keyboard/pointer from host compositor.
        self.start_input_thread()?;

        Ok(())
    }

    fn connectors(&self) -> &[ConnectorInfo] {
        &self.connector_info
    }

    fn capabilities(&self) -> BackendCaps {
        self.caps
    }

    fn scanout_formats(&self) -> &[DrmFormat] {
        &self.scanout_formats
    }

    fn import_dmabuf(&mut self, dmabuf: &DmaBuf) -> anyhow::Result<Framebuffer> {
        let dev = LeaseRef(self.fd.as_fd());

        let mut gem_handles: [Option<drm::buffer::Handle>; 4] = [None; 4];
        let mut pitches = [0u32; 4];
        let mut offsets = [0u32; 4];
        for (i, plane) in dmabuf.planes.iter().enumerate().take(4) {
            // SAFETY: The DMA-BUF fd is valid for the lifetime of the DmaBuf.
            let borrowed_fd = unsafe { BorrowedFd::borrow_raw(plane.fd) };
            let gem = dev
                .prime_fd_to_buffer(borrowed_fd)
                .context("PRIME_FD_TO_HANDLE failed on lease fd")?;
            gem_handles[i] = Some(gem);
            pitches[i] = plane.stride;
            offsets[i] = plane.offset;
        }

        let use_modifiers = dmabuf.modifier != DrmModifier::Invalid;

        struct ImportBuffer {
            size: (u32, u32),
            format: DrmFourcc,
            modifier: Option<DrmModifier>,
            handles: [Option<drm::buffer::Handle>; 4],
            pitches: [u32; 4],
            offsets: [u32; 4],
        }
        impl PlanarBuffer for ImportBuffer {
            fn size(&self) -> (u32, u32) {
                self.size
            }
            fn format(&self) -> DrmFourcc {
                self.format
            }
            fn modifier(&self) -> Option<DrmModifier> {
                self.modifier
            }
            fn pitches(&self) -> [u32; 4] {
                self.pitches
            }
            fn handles(&self) -> [Option<drm::buffer::Handle>; 4] {
                self.handles
            }
            fn offsets(&self) -> [u32; 4] {
                self.offsets
            }
        }

        let buf = ImportBuffer {
            size: (dmabuf.width, dmabuf.height),
            format: dmabuf.format,
            modifier: if use_modifiers {
                Some(dmabuf.modifier)
            } else {
                None
            },
            handles: gem_handles,
            pitches,
            offsets,
        };

        let flags = if use_modifiers {
            FbCmd2Flags::MODIFIERS
        } else {
            FbCmd2Flags::empty()
        };

        let fb = dev
            .add_planar_framebuffer(&buf, flags)
            .context("failed to create framebuffer on lease fd")?;

        // Close GEM handles (deduplicate).
        for gem in gem_handles.iter().filter_map(|h| *h) {
            let already_closed = gem_handles
                .iter()
                .take(
                    gem_handles
                        .iter()
                        .position(|h| *h == Some(gem))
                        .unwrap_or(0),
                )
                .any(|h| *h == Some(gem));
            if !already_closed {
                let _ = dev.close_buffer(gem);
            }
        }

        Ok(Framebuffer {
            handle: fb,
            format: dmabuf.format,
            modifier: dmabuf.modifier,
            size: (dmabuf.width, dmabuf.height),
        })
    }

    fn try_direct_scanout(&mut self, fb: &Framebuffer) -> anyhow::Result<bool> {
        let dev = LeaseRef(self.fd.as_fd());
        let output = match self.output.as_ref() {
            Some(o) => o,
            None => return Ok(false),
        };
        let plane = match self.planes.get(&output.primary_plane) {
            Some(p) => p,
            None => return Ok(false),
        };

        let (mode_w, mode_h) = (output.mode.size().0 as u64, output.mode.size().1 as u64);

        let mut req = drm::control::atomic::AtomicModeReq::new();
        if let Some(prop) = plane.props.fb_id {
            req.add_property(
                output.primary_plane,
                prop,
                drm::control::property::Value::Framebuffer(Some(fb.handle)),
            );
        }
        if let Some(prop) = plane.props.crtc_id {
            req.add_property(
                output.primary_plane,
                prop,
                drm::control::property::Value::CRTC(Some(output.crtc)),
            );
        }
        let src_w = (fb.size.0 as u64) << 16;
        let src_h = (fb.size.1 as u64) << 16;
        if let Some(prop) = plane.props.src_x {
            req.add_property(output.primary_plane, prop, drm::control::property::Value::UnsignedRange(0));
        }
        if let Some(prop) = plane.props.src_y {
            req.add_property(output.primary_plane, prop, drm::control::property::Value::UnsignedRange(0));
        }
        if let Some(prop) = plane.props.src_w {
            req.add_property(output.primary_plane, prop, drm::control::property::Value::UnsignedRange(src_w));
        }
        if let Some(prop) = plane.props.src_h {
            req.add_property(output.primary_plane, prop, drm::control::property::Value::UnsignedRange(src_h));
        }
        if let Some(prop) = plane.props.crtc_x {
            req.add_property(output.primary_plane, prop, drm::control::property::Value::SignedRange(0));
        }
        if let Some(prop) = plane.props.crtc_y {
            req.add_property(output.primary_plane, prop, drm::control::property::Value::SignedRange(0));
        }
        if let Some(prop) = plane.props.crtc_w {
            req.add_property(output.primary_plane, prop, drm::control::property::Value::UnsignedRange(mode_w));
        }
        if let Some(prop) = plane.props.crtc_h {
            req.add_property(output.primary_plane, prop, drm::control::property::Value::UnsignedRange(mode_h));
        }

        match dev.atomic_commit(AtomicCommitFlags::TEST_ONLY, req) {
            Ok(()) => Ok(true),
            Err(_) => Ok(false),
        }
    }

    fn present(&mut self, fb: &Framebuffer) -> anyhow::Result<FlipResult> {
        let dev = LeaseRef(self.fd.as_fd());
        let output = match self.output.as_ref() {
            Some(o) => o,
            None => return Ok(FlipResult::Failed(BackendError::NoOutput.into())),
        };
        let plane = match self.planes.get(&output.primary_plane) {
            Some(p) => p,
            None => return Ok(FlipResult::Failed(BackendError::NoPrimaryPlane.into())),
        };

        let src_w = (fb.size.0 as u64) << 16;
        let src_h = (fb.size.1 as u64) << 16;
        let dst_w = fb.size.0 as u64;
        let dst_h = fb.size.1 as u64;

        let mut req = drm::control::atomic::AtomicModeReq::new();

        if self.needs_modeset {
            let mode_blob = dev
                .create_property_blob(&output.mode)
                .context("failed to create mode blob for lease present")?;

            if let Some(prop) = output.connector_props.crtc_id {
                req.add_property(
                    output.connector,
                    prop,
                    drm::control::property::Value::CRTC(Some(output.crtc)),
                );
            }
            if let Some(prop) = output.crtc_props.active {
                req.add_property(
                    output.crtc,
                    prop,
                    drm::control::property::Value::Boolean(true),
                );
            }
            if let Some(prop) = output.crtc_props.mode_id {
                req.add_property(
                    output.crtc,
                    prop,
                    drm::control::property::Value::Blob(mode_blob.into()),
                );
            }
        }

        // Primary plane.
        if let Some(prop) = plane.props.fb_id {
            req.add_property(
                output.primary_plane,
                prop,
                drm::control::property::Value::Framebuffer(Some(fb.handle)),
            );
        }
        if let Some(prop) = plane.props.crtc_id {
            req.add_property(
                output.primary_plane,
                prop,
                drm::control::property::Value::CRTC(Some(output.crtc)),
            );
        }
        if let Some(prop) = plane.props.src_x {
            req.add_property(output.primary_plane, prop, drm::control::property::Value::UnsignedRange(0));
        }
        if let Some(prop) = plane.props.src_y {
            req.add_property(output.primary_plane, prop, drm::control::property::Value::UnsignedRange(0));
        }
        if let Some(prop) = plane.props.src_w {
            req.add_property(output.primary_plane, prop, drm::control::property::Value::UnsignedRange(src_w));
        }
        if let Some(prop) = plane.props.src_h {
            req.add_property(output.primary_plane, prop, drm::control::property::Value::UnsignedRange(src_h));
        }
        if let Some(prop) = plane.props.crtc_x {
            req.add_property(output.primary_plane, prop, drm::control::property::Value::SignedRange(0));
        }
        if let Some(prop) = plane.props.crtc_y {
            req.add_property(output.primary_plane, prop, drm::control::property::Value::SignedRange(0));
        }
        if let Some(prop) = plane.props.crtc_w {
            req.add_property(output.primary_plane, prop, drm::control::property::Value::UnsignedRange(dst_w));
        }
        if let Some(prop) = plane.props.crtc_h {
            req.add_property(output.primary_plane, prop, drm::control::property::Value::UnsignedRange(dst_h));
        }

        if self.needs_modeset {
            match dev.atomic_commit(AtomicCommitFlags::ALLOW_MODESET, req) {
                Ok(()) => {
                    self.needs_modeset = false;
                    if let Some(ref mut o) = self.output {
                        o.active = true;
                    }
                    info!("lease: initial modeset+flip committed");
                    Ok(FlipResult::DirectScanout)
                }
                Err(e) => {
                    warn!(?e, "lease: initial modeset+flip failed");
                    Ok(FlipResult::Failed(e.into()))
                }
            }
        } else {
            let flags = AtomicCommitFlags::PAGE_FLIP_EVENT | AtomicCommitFlags::NONBLOCK;
            match dev.atomic_commit(flags, req) {
                Ok(()) => {
                    self.flip_pending = true;
                    Ok(FlipResult::Queued)
                }
                Err(e) => {
                    warn!(?e, "lease: atomic commit failed");
                    Ok(FlipResult::Failed(e.into()))
                }
            }
        }
    }

    fn drm_fd(&self) -> Option<RawFd> {
        use std::os::unix::io::AsRawFd;
        Some(self.fd.as_raw_fd())
    }

    fn handle_page_flip(&mut self) -> anyhow::Result<Option<u64>> {
        let dev = LeaseRef(self.fd.as_fd());
        let events = dev
            .receive_events()
            .context("failed to receive DRM events on lease fd")?;

        let mut vblank_ns: Option<u64> = None;
        for event in events {
            if let drm::control::Event::PageFlip(flip) = event {
                vblank_ns = Some(flip.duration.as_nanos() as u64);
            }
        }

        self.flip_pending = false;
        Ok(vblank_ns)
    }

    fn set_vrr(&mut self, enabled: bool) -> anyhow::Result<()> {
        let dev = LeaseRef(self.fd.as_fd());
        let output = self.output.as_mut().context("no output for VRR")?;

        if !output.vrr_capable {
            bail!("lease: connector does not support VRR");
        }

        let prop = output
            .crtc_props
            .vrr_enabled
            .context("VRR_ENABLED property not found on lease")?;

        let mut req = drm::control::atomic::AtomicModeReq::new();
        req.add_property(
            output.crtc,
            prop,
            drm::control::property::Value::Boolean(enabled),
        );

        dev.atomic_commit(AtomicCommitFlags::ALLOW_MODESET, req)
            .context("failed to set VRR on lease")?;

        output.vrr_enabled = enabled;
        info!(enabled, "lease: VRR state changed");
        Ok(())
    }
}

// ─── Wayland input thread ───────────────────────────────────────────

/// Minimal Wayland client that connects to the host compositor for input only.
///
/// Binds `wl_seat` → `wl_keyboard` + `wl_pointer` and forwards all input
/// events to the main thread via channel. No surfaces are created — this
/// connection exists solely for input reception.
fn run_input_thread(
    display_name: &str,
    tx: std::sync::mpsc::Sender<WaylandEvent>,
) -> anyhow::Result<()> {
    use wayland_client::{Connection, Dispatch, QueueHandle};
    use wayland_client::protocol::{wl_registry, wl_seat, wl_keyboard, wl_pointer};

    struct InputState {
        tx: std::sync::mpsc::Sender<WaylandEvent>,
        seat: Option<wl_seat::WlSeat>,
        keyboard: Option<wl_keyboard::WlKeyboard>,
        pointer: Option<wl_pointer::WlPointer>,
    }

    impl Dispatch<wl_registry::WlRegistry, ()> for InputState {
        fn event(
            state: &mut Self,
            registry: &wl_registry::WlRegistry,
            event: wl_registry::Event,
            _data: &(),
            _conn: &Connection,
            qh: &QueueHandle<Self>,
        ) {
            if let wl_registry::Event::Global {
                name,
                interface,
                version,
            } = event
                && interface == "wl_seat"
            {
                let seat = registry.bind::<wl_seat::WlSeat, _, _>(name, version.min(5), qh, ());
                state.seat = Some(seat);
            }
        }
    }

    impl Dispatch<wl_seat::WlSeat, ()> for InputState {
        fn event(
            state: &mut Self,
            seat: &wl_seat::WlSeat,
            event: wl_seat::Event,
            _data: &(),
            _conn: &Connection,
            qh: &QueueHandle<Self>,
        ) {
            if let wl_seat::Event::Capabilities { capabilities } = event {
                let caps = wl_seat::Capability::from_bits_truncate(capabilities.into());
                if caps.contains(wl_seat::Capability::Keyboard) && state.keyboard.is_none() {
                    state.keyboard = Some(seat.get_keyboard(qh, ()));
                }
                if caps.contains(wl_seat::Capability::Pointer) && state.pointer.is_none() {
                    state.pointer = Some(seat.get_pointer(qh, ()));
                }
            }
        }
    }

    impl Dispatch<wl_keyboard::WlKeyboard, ()> for InputState {
        fn event(
            state: &mut Self,
            _proxy: &wl_keyboard::WlKeyboard,
            event: wl_keyboard::Event,
            _data: &(),
            _conn: &Connection,
            _qh: &QueueHandle<Self>,
        ) {
            match event {
                wl_keyboard::Event::Key {
                    key, state: key_state, ..
                } => {
                    let pressed = key_state == wayland_client::WEnum::Value(wl_keyboard::KeyState::Pressed);
                    let _ = state.tx.send(WaylandEvent::Key { key, pressed });
                }
                wl_keyboard::Event::Modifiers {
                    mods_depressed,
                    mods_latched,
                    mods_locked,
                    group,
                    ..
                } => {
                    let _ = state.tx.send(WaylandEvent::Modifiers {
                        mods_depressed,
                        mods_latched,
                        mods_locked,
                        group,
                    });
                }
                wl_keyboard::Event::Keymap { format, fd, size } => {
                    let _ = state.tx.send(WaylandEvent::Keymap {
                        format: format.into(),
                        fd,
                        size,
                    });
                }
                _ => {}
            }
        }
    }

    impl Dispatch<wl_pointer::WlPointer, ()> for InputState {
        fn event(
            state: &mut Self,
            _proxy: &wl_pointer::WlPointer,
            event: wl_pointer::Event,
            _data: &(),
            _conn: &Connection,
            _qh: &QueueHandle<Self>,
        ) {
            match event {
                wl_pointer::Event::Motion {
                    surface_x,
                    surface_y,
                    ..
                } => {
                    let _ = state.tx.send(WaylandEvent::PointerMotion {
                        x: surface_x,
                        y: surface_y,
                    });
                }
                wl_pointer::Event::Button {
                    button,
                    state: btn_state,
                    ..
                } => {
                    let pressed = btn_state == wayland_client::WEnum::Value(wl_pointer::ButtonState::Pressed);
                    let _ = state.tx.send(WaylandEvent::PointerButton { button, pressed });
                }
                wl_pointer::Event::Axis { axis, value, .. } => {
                    let (dx, dy) = if axis == wayland_client::WEnum::Value(wl_pointer::Axis::HorizontalScroll) {
                        (value, 0.0)
                    } else {
                        (0.0, value)
                    };
                    let _ = state.tx.send(WaylandEvent::Scroll { dx, dy });
                }
                _ => {}
            }
        }
    }

    let conn = Connection::connect_to_env()
        .or_else(|_| {
            // SAFETY: called before any threads are spawned in this input thread.
            unsafe { std::env::set_var("WAYLAND_DISPLAY", display_name) };
            Connection::connect_to_env()
        })
        .context("failed to connect to host Wayland for input")?;

    let display = conn.display();
    let mut event_queue = conn.new_event_queue();
    let qh = event_queue.handle();

    let mut input_state = InputState {
        tx,
        seat: None,
        keyboard: None,
        pointer: None,
    };

    let _registry = display.get_registry(&qh, ());

    // Initial roundtrip to discover globals.
    event_queue
        .roundtrip(&mut input_state)
        .context("initial roundtrip failed")?;

    // Second roundtrip for seat capabilities.
    event_queue
        .roundtrip(&mut input_state)
        .context("seat capabilities roundtrip failed")?;

    info!("lease input: connected to host Wayland, dispatching events");

    // Main dispatch loop.
    loop {
        match event_queue.blocking_dispatch(&mut input_state) {
            Ok(_) => {}
            Err(e) => {
                // Connection closed by host — normal during shutdown.
                info!(?e, "lease input: host connection closed");
                break;
            }
        }
    }

    Ok(())
}
