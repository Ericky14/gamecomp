//! DRM/KMS backend for direct hardware display output.
//!
//! This backend opens a GPU device via libseat session management,
//! discovers CRTCs/connectors/planes via the DRM API, and presents
//! frames using atomic modesetting. It supports:
//!
//! - Direct scanout (zero-copy client buffer → hardware plane)
//! - Plane assignment via `DRM_MODE_ATOMIC_TEST_ONLY` probing
//! - VRR (variable refresh rate) via CRTC properties
//! - Explicit sync via DRM syncobj (when available)
//!
//! The DRM backend runs exclusively on the render thread. The main thread
//! never touches DRM state — it sends [`FrameInfo`] via channel and receives
//! flip completion events back.

use std::collections::HashMap;
use std::os::unix::io::{AsFd, BorrowedFd, OwnedFd, RawFd};
use std::path::PathBuf;

use anyhow::{Context, bail};
use drm::Device as DrmDeviceTrait;
use drm::buffer::PlanarBuffer;
use drm::control::{
    AtomicCommitFlags, Device as ControlDevice, FbCmd2Flags, Mode, PlaneType, connector, crtc,
    plane, property,
};
use drm_fourcc::{DrmFormat, DrmFourcc, DrmModifier};
use tracing::{debug, info, warn};

use super::{
    Backend, BackendCaps, BackendError, ConnectorInfo, DmaBuf, DmaBufPlane, FlipResult, Framebuffer,
};

/// A GBM-allocated output buffer with a pre-created DRM framebuffer.
///
/// Allocated via GBM with native GEM handles, these buffers bypass the
/// `PRIME_FD_TO_HANDLE` path that causes tiling metadata corruption on
/// NVIDIA. The DMA-BUF fd is suitable for Vulkan import.
pub struct GbmOutputBuffer {
    /// DRM framebuffer created from GBM's native GEM handle.
    pub fb: Framebuffer,
    /// DMA-BUF descriptor for importing into Vulkan.
    pub dmabuf: DmaBuf,
    /// Owned fd keeping the DMA-BUF alive. The `dmabuf.planes[*].fd`
    /// raw fds borrow this — must outlive all Vulkan imports.
    _fd: std::os::unix::io::OwnedFd,
}

/// Header of the DRM IN_FORMATS property blob (`drm_format_modifier_blob`).
///
/// This kernel structure describes the format+modifier pairs a DRM plane
/// supports for scanout. Parsed to discover which modifiers the display
/// controller can accept.
#[repr(C)]
struct InFormatsBlobHeader {
    _version: u32,
    _flags: u32,
    count_formats: u32,
    formats_offset: u32,
    count_modifiers: u32,
    modifiers_offset: u32,
}

/// Per-modifier entry in the IN_FORMATS blob (`drm_format_modifier`).
#[repr(C)]
struct InFormatsModifier {
    /// Bitmask of which formats (by index) this modifier applies to.
    formats: u64,
    /// Offset into the formats array for this modifier's format range.
    offset: u32,
    _pad: u32,
    /// The DRM modifier value.
    modifier: u64,
}

/// State for a single DRM plane (primary, overlay, or cursor).
#[derive(Debug)]
struct PlaneState {
    handle: plane::Handle,
    kind: PlaneType,
    /// Property handles cached at init for fast atomic commits.
    props: PlaneProps,
    /// Formats supported by this plane.
    formats: Vec<DrmFormat>,
    /// Filter describing which CRTCs this plane can be bound to.
    possible_crtcs: drm::control::CrtcListFilter,
}

/// Cached DRM property handles for a plane.
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

/// Cached DRM property handles for a CRTC.
#[derive(Debug, Default)]
struct CrtcProps {
    active: Option<property::Handle>,
    mode_id: Option<property::Handle>,
    vrr_enabled: Option<property::Handle>,
    out_fence_ptr: Option<property::Handle>,
}

/// Cached DRM property handles for a connector.
#[derive(Debug, Default)]
struct ConnectorProps {
    crtc_id: Option<property::Handle>,
    vrr_capable: Option<property::Handle>,
    hdr_output_metadata: Option<property::Handle>,
}

/// State for a CRTC + connector + planes combination.
#[derive(Debug)]
struct OutputState {
    connector: connector::Handle,
    crtc: crtc::Handle,
    mode: Mode,
    connector_props: ConnectorProps,
    crtc_props: CrtcProps,
    primary_plane: plane::Handle,
    overlay_planes: Vec<plane::Handle>,
    cursor_plane: Option<plane::Handle>,
    vrr_capable: bool,
    vrr_enabled: bool,
    active: bool,
}

/// The DRM/KMS display backend.
pub struct DrmBackend {
    /// DRM device file descriptor.
    fd: OwnedFd,
    /// Path to the DRM device node (e.g., `/dev/dri/card0`).
    device_path: PathBuf,
    /// GBM device for buffer allocation.
    gbm: Option<gbm::Device<std::fs::File>>,
    /// All discovered planes and their cached state.
    planes: HashMap<plane::Handle, PlaneState>,
    /// Active output configurations.
    outputs: Vec<OutputState>,
    /// Connector info exposed to the compositor.
    connector_info: Vec<ConnectorInfo>,
    /// Backend capabilities.
    caps: BackendCaps,
    /// Supported scanout formats for the primary plane.
    scanout_formats: Vec<DrmFormat>,
    /// Whether a page flip is currently pending.
    flip_pending: bool,
    /// The first atomic commit that assigns an FB to the primary plane
    /// is effectively a modeset and needs `ALLOW_MODESET`. This flag is
    /// cleared after the first successful flip.
    needs_modeset: bool,
}

/// Wrapper to implement `drm::Device` on an `OwnedFd`.
struct DrmDevice(OwnedFd);

impl AsFd for DrmDevice {
    fn as_fd(&self) -> BorrowedFd<'_> {
        self.0.as_fd()
    }
}

impl DrmDeviceTrait for DrmDevice {}
impl ControlDevice for DrmDevice {}

/// Borrowed reference to a DRM device, for temporary control operations.
///
/// Unlike `DrmDevice`, this borrows only the fd rather than owning it.
/// By constructing via `DrmRef(self.fd.as_fd())`, we avoid borrowing the
/// entire `DrmBackend`, preventing borrow checker conflicts when other
/// fields need simultaneous access.
struct DrmRef<'a>(BorrowedFd<'a>);

impl<'a> AsFd for DrmRef<'a> {
    fn as_fd(&self) -> BorrowedFd<'_> {
        self.0
    }
}

impl DrmDeviceTrait for DrmRef<'_> {}
impl ControlDevice for DrmRef<'_> {}

impl DrmBackend {
    /// Create a new DRM backend for the given device path.
    pub fn new(device_path: PathBuf, fd: OwnedFd) -> Self {
        Self {
            fd,
            device_path,
            gbm: None,
            planes: HashMap::new(),
            outputs: Vec::new(),
            connector_info: Vec::new(),
            caps: BackendCaps::default(),
            scanout_formats: Vec::new(),
            flip_pending: false,
            needs_modeset: true,
        }
    }

    /// Discover planes and cache their properties.
    fn enumerate_planes(&mut self) -> anyhow::Result<()> {
        let dev = DrmRef(self.fd.as_fd());
        let plane_handles = dev
            .plane_handles()
            .context("failed to enumerate DRM planes")?;

        for &handle in plane_handles.as_slice() {
            let plane = dev.get_plane(handle).context("failed to get plane info")?;

            let props = dev
                .get_properties(handle)
                .context("failed to get plane properties")?;

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

            // Collect supported formats.
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
                ?possible_crtcs,
                num_formats = formats.len(),
                "discovered plane"
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

        info!(count = self.planes.len(), "discovered DRM planes");
        Ok(())
    }

    /// Query the modifiers supported by the primary plane's IN_FORMATS blob
    /// for the given DRM fourcc format code.
    ///
    /// These modifiers represent what the display controller can scan out.
    /// Pass them to `VulkanBlitter` to create output images the hardware can
    /// display directly, avoiding LINEAR-only fallback that produces noise
    /// on NVIDIA.
    pub fn query_primary_plane_modifiers(&self, format: DrmFourcc) -> Vec<u64> {
        let dev = DrmRef(self.fd.as_fd());

        // Find the primary plane for the first output.
        let primary_plane = match self.outputs.first() {
            Some(output) => output.primary_plane,
            None => return Vec::new(),
        };

        // Get the plane's properties to find IN_FORMATS.
        let props = match dev.get_properties(primary_plane) {
            Ok(p) => p,
            Err(e) => {
                warn!(?e, "failed to get plane properties for IN_FORMATS");
                return Vec::new();
            }
        };

        // Find the IN_FORMATS property handle and its blob ID value.
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
            info!("IN_FORMATS property not found on primary plane");
            return Vec::new();
        };

        // Read the blob data.
        let blob_data = match dev.get_property_blob(blob_id) {
            Ok(d) => d,
            Err(e) => {
                warn!(?e, "failed to read IN_FORMATS blob");
                return Vec::new();
            }
        };

        if blob_data.len() < std::mem::size_of::<InFormatsBlobHeader>() {
            warn!(size = blob_data.len(), "IN_FORMATS blob too small");
            return Vec::new();
        }

        // Parse the drm_format_modifier_blob header.
        // SAFETY: blob_data is a kernel DRM blob with drm_format_modifier_blob
        // layout. We verified the minimum size above. All fields are POD u32.
        let header = unsafe { &*(blob_data.as_ptr() as *const InFormatsBlobHeader) };

        let target_fourcc = format as u32;
        let mut modifiers = Vec::new();

        // Walk the modifier entries and collect modifiers for our target format.
        for i in 0..header.count_modifiers {
            let mod_offset = header.modifiers_offset as usize
                + (i as usize) * std::mem::size_of::<InFormatsModifier>();
            if mod_offset + std::mem::size_of::<InFormatsModifier>() > blob_data.len() {
                break;
            }
            // SAFETY: bounds-checked above; InFormatsModifier is repr(C) POD.
            let mod_entry =
                unsafe { &*(blob_data.as_ptr().add(mod_offset) as *const InFormatsModifier) };

            // Check each format bit in this modifier's bitmask.
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

        info!(
            count = modifiers.len(),
            format = format!("{:?}", format),
            "queried primary plane IN_FORMATS modifiers"
        );
        for m in &modifiers {
            debug!(modifier = format!("0x{:016x}", m), "plane scanout modifier");
        }

        modifiers
    }

    /// Discover connectors and CRTCs, build output configurations.
    fn enumerate_outputs(&mut self) -> anyhow::Result<()> {
        let dev = DrmRef(self.fd.as_fd());
        let resources = dev
            .resource_handles()
            .context("failed to get DRM resources")?;

        for &conn_handle in resources.connectors() {
            let conn = match dev.get_connector(conn_handle, false) {
                Ok(c) => c,
                Err(e) => {
                    warn!(?conn_handle, ?e, "failed to get connector, skipping");
                    continue;
                }
            };

            if conn.state() != connector::State::Connected {
                debug!(?conn_handle, "connector not connected, skipping");
                continue;
            }

            // Find a CRTC for this connector.
            let encoder = conn.current_encoder().and_then(|e| dev.get_encoder(e).ok());
            let crtc_handle = encoder
                .map(|e| e.crtc().unwrap())
                .or_else(|| resources.crtcs().first().copied());

            let Some(crtc_handle) = crtc_handle else {
                warn!(?conn_handle, "no CRTC available for connector");
                continue;
            };

            // Select the preferred mode or the first available.
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
                warn!(?conn_handle, "no modes available for connector");
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
                            "HDR_OUTPUT_METADATA" => {
                                conn_props.hdr_output_metadata = Some(prop_handle)
                            }
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
                            "OUT_FENCE_PTR" => crtc_props.out_fence_ptr = Some(prop_handle),
                            _ => {}
                        }
                    }
                }
            }

            // Check VRR capability.
            let vrr_capable = conn_props.vrr_capable.is_some();

            // Filter planes by possible_crtcs — each plane is only compatible
            // with specific CRTCs. Using an incompatible plane causes EINVAL
            // on atomic commit.
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
                warn!(?crtc_handle, "no compatible primary plane found for CRTC");
                continue;
            };

            debug!(
                ?crtc_handle,
                ?primary_plane,
                "selected primary plane for CRTC"
            );

            // Collect overlay and cursor planes compatible with this CRTC.
            let overlay_planes: Vec<_> = self
                .planes
                .values()
                .filter(|p| {
                    p.kind == PlaneType::Overlay
                        && resources
                            .filter_crtcs(p.possible_crtcs)
                            .contains(&crtc_handle)
                })
                .map(|p| p.handle)
                .collect();

            let cursor_plane = self
                .planes
                .values()
                .find(|p| {
                    p.kind == PlaneType::Cursor
                        && resources
                            .filter_crtcs(p.possible_crtcs)
                            .contains(&crtc_handle)
                })
                .map(|p| p.handle);

            let conn_name = format!("{:?}-{}", conn.interface(), conn.interface_id());

            let physical_size_mm = conn.size().unwrap_or((0, 0));

            info!(
                connector = %conn_name,
                mode = ?mode,
                vrr_capable,
                "discovered output"
            );

            self.outputs.push(OutputState {
                connector: conn_handle,
                crtc: crtc_handle,
                mode,
                connector_props: conn_props,
                crtc_props,
                primary_plane,
                overlay_planes,
                cursor_plane,
                vrr_capable,
                vrr_enabled: false,
                active: false,
            });

            self.connector_info.push(ConnectorInfo {
                handle: conn_handle,
                crtc: crtc_handle,
                name: conn_name,
                mode,
                physical_size_mm,
                vrr_enabled: false,
            });
        }

        if self.outputs.is_empty() {
            bail!("no connected displays found");
        }

        // Collect scanout formats from the primary plane.
        if let Some(output) = self.outputs.first()
            && let Some(plane) = self.planes.get(&output.primary_plane)
        {
            self.scanout_formats = plane.formats.clone();
        }

        Ok(())
    }

    /// Blank all displays except the chosen output by disabling their
    /// CRTCs and connectors.
    ///
    /// The chosen output's CRTC is NOT enabled here — that happens in the
    /// first `present()` call which combines the modeset with the primary
    /// plane assignment in a single atomic commit. This avoids a
    /// double-modeset that NVIDIA's DRM driver may reject.
    fn modeset_first_output(&mut self) -> anyhow::Result<()> {
        let dev = DrmRef(self.fd.as_fd());

        let output = self.outputs.first_mut().context("no outputs to modeset")?;
        let active_connector = output.connector;
        let active_crtc = output.crtc;

        // Build atomic request to disable all OTHER connectors and CRTCs.
        let mut req = drm::control::atomic::AtomicModeReq::new();
        let mut has_disable = false;

        let resources = dev
            .resource_handles()
            .context("failed to get DRM resources for blanking")?;

        for &conn_handle in resources.connectors() {
            if conn_handle == active_connector {
                continue; // Will be enabled in first present().
            }
            if let Ok(props) = dev.get_properties(conn_handle) {
                for (&prop_handle, _) in &props {
                    if let Ok(info) = dev.get_property(prop_handle)
                        && info.name().to_str() == Ok("CRTC_ID")
                    {
                        req.add_property(
                            conn_handle,
                            prop_handle,
                            drm::control::property::Value::CRTC(None),
                        );
                        has_disable = true;
                        debug!(?conn_handle, "disabling connector");
                    }
                }
            }
        }

        for &crtc_handle in resources.crtcs() {
            if crtc_handle == active_crtc {
                continue; // Will be enabled in first present().
            }
            if let Ok(props) = dev.get_properties(crtc_handle) {
                for (&prop_handle, _) in &props {
                    if let Ok(info) = dev.get_property(prop_handle) {
                        match info.name().to_str() {
                            Ok("ACTIVE") => {
                                req.add_property(
                                    crtc_handle,
                                    prop_handle,
                                    drm::control::property::Value::Boolean(false),
                                );
                                has_disable = true;
                            }
                            Ok("MODE_ID") => {
                                req.add_property(
                                    crtc_handle,
                                    prop_handle,
                                    drm::control::property::Value::Blob(0),
                                );
                            }
                            _ => {}
                        }
                    }
                }
            }
            debug!(?crtc_handle, "disabling CRTC");
        }

        if has_disable {
            dev.atomic_commit(AtomicCommitFlags::ALLOW_MODESET, req)
                .context("atomic blanking commit failed")?;
        }

        info!(
            connector = %self.connector_info[0].name,
            mode = ?output.mode,
            ?active_crtc,
            "blanked other displays (chosen output will be enabled on first present)"
        );

        Ok(())
    }
}

impl Backend for DrmBackend {
    fn init(&mut self) -> anyhow::Result<()> {
        info!(path = %self.device_path.display(), "initializing DRM backend");

        // Set client capability for atomic modesetting.
        {
            let dev = DrmRef(self.fd.as_fd());
            dev.set_client_capability(drm::ClientCapability::Atomic, true)
                .context("kernel does not support atomic modesetting")?;
            dev.set_client_capability(drm::ClientCapability::UniversalPlanes, true)
                .context("kernel does not support universal planes")?;
        }

        self.enumerate_planes()?;
        self.enumerate_outputs()?;

        // Detect capabilities.
        self.caps.modifiers = true; // Assume modifier support with atomic.
        self.caps.vrr = self.outputs.iter().any(|o| o.vrr_capable);
        self.caps.explicit_sync = self.planes.values().any(|p| p.props.in_fence_fd.is_some());

        info!(?self.caps, "DRM backend capabilities");

        self.modeset_first_output()?;

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
        let dev = DrmRef(self.fd.as_fd());

        debug!(
            width = dmabuf.width,
            height = dmabuf.height,
            format = ?dmabuf.format,
            modifier = ?dmabuf.modifier,
            num_planes = dmabuf.planes.len(),
            "importing DMA-BUF"
        );

        // Import DMA-BUF fds as GEM handles via PRIME_FD_TO_HANDLE.
        let mut gem_handles: [Option<drm::buffer::Handle>; 4] = [None; 4];
        let mut pitches = [0u32; 4];
        let mut offsets = [0u32; 4];
        for (i, plane) in dmabuf.planes.iter().enumerate().take(4) {
            // SAFETY: The DMA-BUF fd is valid for the lifetime of the DmaBuf.
            let borrowed_fd = unsafe { BorrowedFd::borrow_raw(plane.fd) };
            let gem = dev
                .prime_fd_to_buffer(borrowed_fd)
                .context("PRIME_FD_TO_HANDLE failed")?;
            debug!(
                plane_idx = i,
                fd = plane.fd,
                stride = plane.stride,
                offset = plane.offset,
                gem = ?gem,
                "imported DMA-BUF plane"
            );
            gem_handles[i] = Some(gem);
            pitches[i] = plane.stride;
            offsets[i] = plane.offset;
        }

        // Decide FB creation path based on modifier.
        //
        // DRM_FORMAT_MOD_INVALID: Use the legacy path (drmModeAddFB2 without
        //   DRM_MODE_FB_MODIFIERS). The buffer's memory layout
        //   is row-major linear, which the legacy path expects.
        //
        // Any concrete modifier (including LINEAR=0): Use the modifier-aware
        //   path (drmModeAddFB2WithModifiers with DRM_MODE_FB_MODIFIERS flag).
        //   Required by NVIDIA's DRM driver for modifier-aware allocations.
        let use_modifiers = dmabuf.modifier != DrmModifier::Invalid;

        // Wrapper implementing PlanarBuffer for add_planar_framebuffer.
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

        info!(
            use_modifiers,
            modifier = ?dmabuf.modifier,
            format = ?dmabuf.format,
            width = dmabuf.width,
            height = dmabuf.height,
            gem_h0 = ?gem_handles[0],
            pitch0 = pitches[0],
            offset0 = offsets[0],
            "creating DRM framebuffer"
        );

        let fb = dev
            .add_planar_framebuffer(&buf, flags)
            .context("failed to import DMA-BUF as framebuffer")?;

        info!(
            fb_id = ?fb,
            format = ?dmabuf.format,
            modifier = ?dmabuf.modifier,
            width = dmabuf.width,
            height = dmabuf.height,
            "DRM framebuffer created successfully"
        );

        // Close GEM handles immediately after FB creation — the kernel's
        // framebuffer object internally holds a reference to the GEM
        // objects, keeping the backing memory alive.
        for gem in gem_handles.iter().filter_map(|h| *h) {
            // Deduplicate: GEM handles aren't ref-counted per the kernel
            // API. Two DMA-BUFs may return the same handle, and we must
            // not double-close them.
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
        let dev = DrmRef(self.fd.as_fd());
        let output = match self.outputs.first() {
            Some(o) => o,
            None => return Ok(false),
        };

        let plane = match self.planes.get(&output.primary_plane) {
            Some(p) => p,
            None => return Ok(false),
        };

        let mode = output.mode;
        let (mode_w, mode_h) = (mode.size().0 as u64, mode.size().1 as u64);

        // Build atomic request: assign fb to primary plane.
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

        // Source rect (16.16 fixed point).
        let src_w = (fb.size.0 as u64) << 16;
        let src_h = (fb.size.1 as u64) << 16;
        if let Some(prop) = plane.props.src_x {
            req.add_property(
                output.primary_plane,
                prop,
                drm::control::property::Value::UnsignedRange(0),
            );
        }
        if let Some(prop) = plane.props.src_y {
            req.add_property(
                output.primary_plane,
                prop,
                drm::control::property::Value::UnsignedRange(0),
            );
        }
        if let Some(prop) = plane.props.src_w {
            req.add_property(
                output.primary_plane,
                prop,
                drm::control::property::Value::UnsignedRange(src_w),
            );
        }
        if let Some(prop) = plane.props.src_h {
            req.add_property(
                output.primary_plane,
                prop,
                drm::control::property::Value::UnsignedRange(src_h),
            );
        }

        // Dest rect.
        if let Some(prop) = plane.props.crtc_x {
            req.add_property(
                output.primary_plane,
                prop,
                drm::control::property::Value::SignedRange(0),
            );
        }
        if let Some(prop) = plane.props.crtc_y {
            req.add_property(
                output.primary_plane,
                prop,
                drm::control::property::Value::SignedRange(0),
            );
        }
        if let Some(prop) = plane.props.crtc_w {
            req.add_property(
                output.primary_plane,
                prop,
                drm::control::property::Value::UnsignedRange(mode_w),
            );
        }
        if let Some(prop) = plane.props.crtc_h {
            req.add_property(
                output.primary_plane,
                prop,
                drm::control::property::Value::UnsignedRange(mode_h),
            );
        }

        // Test-only commit to check if direct scanout works.
        match dev.atomic_commit(AtomicCommitFlags::TEST_ONLY, req) {
            Ok(()) => {
                debug!("direct scanout test passed");
                Ok(true)
            }
            Err(_) => {
                debug!("direct scanout test failed, will composite");
                Ok(false)
            }
        }
    }

    fn present(&mut self, fb: &Framebuffer) -> anyhow::Result<FlipResult> {
        let dev = DrmRef(self.fd.as_fd());
        let output = match self.outputs.first() {
            Some(o) => o,
            None => return Ok(FlipResult::Failed(BackendError::NoOutput.into())),
        };

        let plane = match self.planes.get(&output.primary_plane) {
            Some(p) => p,
            None => return Ok(FlipResult::Failed(BackendError::NoPrimaryPlane.into())),
        };

        let mode = output.mode;
        let (mode_w, mode_h) = (mode.size().0 as u64, mode.size().1 as u64);

        // Source rect must match dest rect — most DRM drivers (including
        // NVIDIA) do NOT support plane scaling on the primary plane.
        // Always present at display resolution; the Vulkan compositor must
        // scale the client buffer to match before we get here.
        //
        // For now, use the framebuffer dimensions as both src and dst.
        // If the FB is smaller than the display, the extra pixels are
        // undefined (dark/garbage). This avoids EINVAL from the kernel
        // when src != dst and scaling isn't supported.
        let src_w = (fb.size.0 as u64) << 16;
        let src_h = (fb.size.1 as u64) << 16;
        let dst_w = fb.size.0 as u64;
        let dst_h = fb.size.1 as u64;

        debug!(
            fb_w = fb.size.0,
            fb_h = fb.size.1,
            mode_w,
            mode_h,
            needs_modeset = self.needs_modeset,
            primary_plane = ?output.primary_plane,
            crtc = ?output.crtc,
            connector = ?output.connector,
            "presenting framebuffer"
        );

        let mut req = drm::control::atomic::AtomicModeReq::new();

        // When this is the first flip after modeset, we must include
        // the CRTC and connector properties in the same atomic request.
        // The kernel requires a complete atomic state when transitioning
        // from "no plane FB" to "plane FB assigned".
        if self.needs_modeset {
            let mode_blob = dev
                .create_property_blob(&output.mode)
                .context("failed to create mode blob for present")?;

            // Connector → CRTC.
            if let Some(prop) = output.connector_props.crtc_id {
                req.add_property(
                    output.connector,
                    prop,
                    drm::control::property::Value::CRTC(Some(output.crtc)),
                );
            }
            // CRTC active + mode.
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
            debug!("first flip: including CRTC+connector in atomic request");
        }

        // Assign framebuffer to primary plane.
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
            req.add_property(
                output.primary_plane,
                prop,
                drm::control::property::Value::UnsignedRange(0),
            );
        }
        if let Some(prop) = plane.props.src_y {
            req.add_property(
                output.primary_plane,
                prop,
                drm::control::property::Value::UnsignedRange(0),
            );
        }
        if let Some(prop) = plane.props.src_w {
            req.add_property(
                output.primary_plane,
                prop,
                drm::control::property::Value::UnsignedRange(src_w),
            );
        }
        if let Some(prop) = plane.props.src_h {
            req.add_property(
                output.primary_plane,
                prop,
                drm::control::property::Value::UnsignedRange(src_h),
            );
        }
        if let Some(prop) = plane.props.crtc_x {
            req.add_property(
                output.primary_plane,
                prop,
                drm::control::property::Value::SignedRange(0),
            );
        }
        if let Some(prop) = plane.props.crtc_y {
            req.add_property(
                output.primary_plane,
                prop,
                drm::control::property::Value::SignedRange(0),
            );
        }
        if let Some(prop) = plane.props.crtc_w {
            req.add_property(
                output.primary_plane,
                prop,
                drm::control::property::Value::UnsignedRange(dst_w),
            );
        }
        if let Some(prop) = plane.props.crtc_h {
            req.add_property(
                output.primary_plane,
                prop,
                drm::control::property::Value::UnsignedRange(dst_h),
            );
        }

        if self.needs_modeset {
            // The first commit that assigns an FB to the primary plane is
            // effectively a modeset. The kernel's atomic helper rejects
            // NONBLOCK and PAGE_FLIP_EVENT on modeset commits (returns
            // EINVAL). Use a synchronous ALLOW_MODESET commit for the
            // initial flip, then switch to async flips for subsequent
            // frames.
            match dev.atomic_commit(AtomicCommitFlags::ALLOW_MODESET, req) {
                Ok(()) => {
                    self.needs_modeset = false;
                    self.outputs.first_mut().unwrap().active = true;
                    info!("initial modeset+flip committed (synchronous)");
                    Ok(FlipResult::DirectScanout)
                }
                Err(e) => {
                    warn!(
                        ?e,
                        plane = ?output.primary_plane,
                        crtc = ?output.crtc,
                        connector = ?output.connector,
                        possible_crtcs = ?plane.possible_crtcs,
                        fb = ?fb.handle,
                        fb_format = ?fb.format,
                        fb_modifier = ?fb.modifier,
                        "initial modeset+flip commit failed"
                    );
                    Ok(FlipResult::Failed(e.into()))
                }
            }
        } else {
            let flags = AtomicCommitFlags::PAGE_FLIP_EVENT | AtomicCommitFlags::NONBLOCK;
            match dev.atomic_commit(flags, req) {
                Ok(()) => {
                    self.flip_pending = true;
                    debug!("atomic commit queued (page flip pending)");
                    Ok(FlipResult::Queued)
                }
                Err(e) => {
                    warn!(?e, "atomic commit failed");
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
        let dev = DrmRef(self.fd.as_fd());
        let events = dev
            .receive_events()
            .context("failed to receive DRM events")?;

        let mut vblank_ns: Option<u64> = None;
        for event in events {
            if let drm::control::Event::PageFlip(flip) = event {
                // flip.duration is the absolute vblank timestamp from the kernel
                // (typically CLOCK_MONOTONIC when DRM_CAP_TIMESTAMP_MONOTONIC is set).
                vblank_ns = Some(flip.duration.as_nanos() as u64);
            }
        }

        self.flip_pending = false;

        Ok(vblank_ns)
    }

    fn set_vrr(&mut self, enabled: bool) -> anyhow::Result<()> {
        let dev = DrmRef(self.fd.as_fd());
        let output = self.outputs.first_mut().context("no output for VRR")?;

        if !output.vrr_capable {
            bail!("connector does not support VRR");
        }

        let prop = output
            .crtc_props
            .vrr_enabled
            .context("VRR_ENABLED property not found")?;

        let mut req = drm::control::atomic::AtomicModeReq::new();
        req.add_property(
            output.crtc,
            prop,
            drm::control::property::Value::Boolean(enabled),
        );

        dev.atomic_commit(AtomicCommitFlags::ALLOW_MODESET, req)
            .context("failed to set VRR")?;

        output.vrr_enabled = enabled;
        info!(enabled, "VRR state changed");

        Ok(())
    }
}

impl DrmBackend {
    /// Destroy a framebuffer handle.
    ///
    /// The kernel will release the underlying GEM references when the FB
    /// is no longer in use by the display controller. Callers must ensure
    /// the FB is not currently being scanned out (wait for page flip first).
    /// Force a modeset on the next present (e.g., after VT switch resume).
    ///
    /// When the session is re-enabled after a VT switch, the display controller
    /// state is lost. The next present must include CRTC+connector state with
    /// `DRM_MODE_ALLOW_MODESET` to re-establish the display pipeline.
    pub fn force_modeset(&mut self) {
        self.needs_modeset = true;
        self.flip_pending = false;
    }

    pub fn destroy_framebuffer(&self, fb: drm::control::framebuffer::Handle) {
        let dev = DrmRef(self.fd.as_fd());
        if let Err(e) = dev.destroy_framebuffer(fb) {
            warn!(?fb, ?e, "failed to destroy framebuffer");
        }
    }

    /// Allocate GBM-backed output buffers for Vulkan composition.
    ///
    /// Creates `count` GBM buffer objects with the given modifiers, creates
    /// DRM framebuffers from their native GEM handles (no PRIME import),
    /// and returns DMA-BUF descriptors for importing into Vulkan.
    ///
    /// This bypasses `PRIME_FD_TO_HANDLE` entirely — GBM allocates using
    /// the kernel DRM driver's native allocation path, so the resulting
    /// GEM handles carry correct tiling metadata. On NVIDIA, this is the
    /// only reliable way to create scanout buffers that are both renderable
    /// by Vulkan (via DMA-BUF import) and displayable by the DRM plane.
    pub fn allocate_gbm_output_buffers(
        &mut self,
        count: usize,
        width: u32,
        height: u32,
        modifiers: &[u64],
    ) -> anyhow::Result<Vec<GbmOutputBuffer>> {
        use gbm::{BufferObjectFlags, Device as GbmDevice};
        use std::os::unix::io::AsRawFd;

        let gbm: GbmDevice<DrmRef> = GbmDevice::new(DrmRef(self.fd.as_fd()))
            .context("failed to create GBM device for output allocation")?;

        info!(
            backend = gbm.backend_name(),
            count,
            width,
            height,
            modifier_count = modifiers.len(),
            "allocating GBM output buffers"
        );

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
                .with_context(|| format!("GBM buffer allocation failed for output {i}"))?;

            let bo_modifier = bo.modifier();
            let bo_stride = bo.stride();
            let bo_offset = bo.offset(0);

            // Export DMA-BUF fd (GBM dups internally — we get an independent OwnedFd).
            let bo_fd = bo
                .fd()
                .with_context(|| format!("failed to export GBM buffer fd for output {i}"))?;

            info!(
                i,
                modifier = format!("0x{:016x}", u64::from(bo_modifier)),
                stride = bo_stride,
                offset = bo_offset,
                fd = bo_fd.as_raw_fd(),
                plane_count = bo.plane_count(),
                "GBM output buffer allocated"
            );

            // Create DRM FB directly from the GBM buffer object.
            // This uses native GEM handles — no PRIME_FD_TO_HANDLE.
            let dev = DrmRef(self.fd.as_fd());
            let flags = if bo_modifier != DrmModifier::Invalid {
                FbCmd2Flags::MODIFIERS
            } else {
                FbCmd2Flags::empty()
            };
            let fb_handle = dev
                .add_planar_framebuffer(&bo, flags)
                .with_context(|| format!("add_planar_framebuffer failed for output {i}"))?;

            info!(
                i,
                fb = ?fb_handle,
                modifier = format!("0x{:016x}", u64::from(bo_modifier)),
                "DRM framebuffer created from GBM (native GEM, no PRIME)"
            );

            let fb = Framebuffer {
                handle: fb_handle,
                format: DrmFourcc::Xrgb8888,
                modifier: bo_modifier,
                size: (width, height),
            };

            // The DMA-BUF fd is an independent dup from GBM. We store the
            // raw fd for the DmaBuf descriptor — Vulkan will dup it again
            // when importing, so this fd must stay alive until import.
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

            outputs.push(GbmOutputBuffer {
                fb,
                dmabuf,
                _fd: bo_fd,
            });

            // Forget the GBM buffer object to prevent its destructor from
            // freeing the underlying GEM buffer. The DRM framebuffer holds
            // a kernel reference to the GEM object, keeping the memory
            // alive. The DMA-BUF fd we exported is an independent dup.
            std::mem::forget(bo);
        }

        // Drop GBM device — all BOs are forgotten and DRM FBs + DMA-BUF
        // fds keep the backing memory alive. Releases the DrmRef borrow.
        drop(gbm);

        info!(count = outputs.len(), "GBM output buffers ready");
        Ok(outputs)
    }
}
