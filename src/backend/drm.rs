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

use super::{Backend, BackendCaps, BackendError, ConnectorInfo, DmaBuf, FlipResult, Framebuffer};

/// State for a single DRM plane (primary, overlay, or cursor).
#[derive(Debug)]
struct PlaneState {
    handle: plane::Handle,
    kind: PlaneType,
    /// Property handles cached at init for fast atomic commits.
    props: PlaneProps,
    /// Formats supported by this plane.
    formats: Vec<DrmFormat>,
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
            let formats = plane
                .formats()
                .iter()
                .map(|&f| DrmFormat {
                    code: DrmFourcc::try_from(f).unwrap_or(DrmFourcc::Argb8888),
                    modifier: DrmModifier::Linear,
                })
                .collect();

            self.planes.insert(
                handle,
                PlaneState {
                    handle,
                    kind,
                    props: plane_props,
                    formats,
                },
            );
        }

        info!(count = self.planes.len(), "discovered DRM planes");
        Ok(())
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

            // Find primary plane for this CRTC.
            let primary_plane = self
                .planes
                .values()
                .find(|p| p.kind == PlaneType::Primary)
                .map(|p| p.handle);

            let Some(primary_plane) = primary_plane else {
                warn!(?crtc_handle, "no primary plane found for CRTC");
                continue;
            };

            // Collect overlay and cursor planes.
            let overlay_planes: Vec<_> = self
                .planes
                .values()
                .filter(|p| p.kind == PlaneType::Overlay)
                .map(|p| p.handle)
                .collect();

            let cursor_plane = self
                .planes
                .values()
                .find(|p| p.kind == PlaneType::Cursor)
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

    /// Set the initial mode on the first output via atomic commit.
    fn modeset_first_output(&mut self) -> anyhow::Result<()> {
        let dev = DrmRef(self.fd.as_fd());

        let output = self.outputs.first_mut().context("no outputs to modeset")?;

        // Create a mode blob.
        let mode_blob = dev
            .create_property_blob(&output.mode)
            .context("failed to create mode blob")?;

        // Build atomic request.
        let mut req = drm::control::atomic::AtomicModeReq::new();

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

        dev.atomic_commit(AtomicCommitFlags::ALLOW_MODESET, req)
            .context("atomic modeset commit failed")?;

        output.active = true;
        info!(
            connector = %self.connector_info[0].name,
            mode = ?output.mode,
            "modeset complete"
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
            gem_handles[i] = Some(gem);
            pitches[i] = plane.stride;
            offsets[i] = plane.offset;
        }

        // Wrapper implementing PlanarBuffer for add_planar_framebuffer.
        struct ImportBuffer {
            size: (u32, u32),
            format: DrmFourcc,
            modifier: DrmModifier,
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
                Some(self.modifier)
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
            modifier: dmabuf.modifier,
            handles: gem_handles,
            pitches,
            offsets,
        };

        let fb = dev
            .add_planar_framebuffer(&buf, FbCmd2Flags::MODIFIERS)
            .context("failed to import DMA-BUF as framebuffer")?;

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
        let src_w = (fb.size.0 as u64) << 16;
        let src_h = (fb.size.1 as u64) << 16;

        let mut req = drm::control::atomic::AtomicModeReq::new();

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

        // Non-blocking commit with page flip event.
        match dev.atomic_commit(
            AtomicCommitFlags::PAGE_FLIP_EVENT | AtomicCommitFlags::NONBLOCK,
            req,
        ) {
            Ok(()) => {
                self.flip_pending = true;
                Ok(FlipResult::Queued)
            }
            Err(e) => Ok(FlipResult::Failed(e.into())),
        }
    }

    fn drm_fd(&self) -> Option<RawFd> {
        use std::os::unix::io::AsRawFd;
        Some(self.fd.as_raw_fd())
    }

    fn handle_page_flip(&mut self) -> anyhow::Result<()> {
        let dev = DrmRef(self.fd.as_fd());
        // Process DRM events (page flip completion).
        dev.receive_events()
            .context("failed to receive DRM events")?;
        self.flip_pending = false;
        Ok(())
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
