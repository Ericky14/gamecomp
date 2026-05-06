//! `wp_linux_drm_syncobj_manager_v1` — explicit DRM synchronization protocol.
//!
//! Allows clients (especially XWayland) to attach DRM syncobj timeline acquire
//! and release points to surface commits. The compositor waits on the acquire
//! point before reading the buffer (GPU rendering must be done) and signals
//! the release point when the buffer is no longer in use (display page flip
//! completed).
//!
//! This replaces the implicit `wl_buffer.release` mechanism for DMA-BUF
//! buffers, providing correct synchronization even when the client's buffer
//! is used directly for display scanout (zero-copy flip).

use std::os::unix::io::{AsFd, BorrowedFd, OwnedFd};
use std::sync::Arc;

use parking_lot::Mutex;
use tracing::{debug, info, trace, warn};
use wayland_protocols::wp::linux_drm_syncobj::v1::server::{
    wp_linux_drm_syncobj_manager_v1::{self, WpLinuxDrmSyncobjManagerV1},
    wp_linux_drm_syncobj_surface_v1::{self, WpLinuxDrmSyncobjSurfaceV1},
    wp_linux_drm_syncobj_timeline_v1::{self, WpLinuxDrmSyncobjTimelineV1},
};
use wayland_server::protocol::wl_surface::WlSurface;
use wayland_server::{Client, DataInit, Dispatch, DisplayHandle, GlobalDispatch, New, Resource};

use crate::wayland::WaylandState;

// ─── DRM syncobj kernel primitives ──────────────────────────────────

/// A DRM syncobj handle imported from a client timeline fd.
///
/// The handle is local to the compositor's DRM device and must be
/// destroyed when no longer needed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SyncobjHandle(pub u32);

/// Ref-counted DRM syncobj handle + device.
///
/// The kernel syncobj is destroyed when the last `Arc<SyncobjGuard>` drops.
/// Both `TimelineData` (Wayland protocol lifetime) and `SyncPoint` (in-flight
/// buffer lifetime) hold an `Arc`, so the handle stays valid until all
/// references are released.
pub struct SyncobjGuard {
    /// Imported DRM syncobj handle (compositor-local).
    pub handle: SyncobjHandle,
    /// DRM device for cleanup.
    device: SyncobjDevice,
}

impl Drop for SyncobjGuard {
    fn drop(&mut self) {
        self.device.destroy(self.handle);
    }
}

impl std::fmt::Debug for SyncobjGuard {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SyncobjGuard")
            .field("handle", &self.handle)
            .finish()
    }
}

/// A timeline point on a DRM syncobj — (handle, point) pair.
///
/// Holds an `Arc<SyncobjGuard>` to keep the kernel syncobj alive
/// while the compositor has in-flight references (e.g., pending
/// acquire/release on buffers sent to the render thread).
#[derive(Debug, Clone)]
pub struct SyncPoint {
    /// Ref-counted guard keeping the kernel syncobj alive.
    guard: Arc<SyncobjGuard>,
    /// Timeline point value.
    pub point: u64,
}

impl SyncPoint {
    /// The DRM syncobj handle.
    #[inline(always)]
    #[must_use]
    pub fn handle(&self) -> SyncobjHandle {
        self.guard.handle
    }
}

/// DRM device reference for syncobj operations.
///
/// Wraps a borrowed DRM fd and implements the kernel ioctls needed for
/// explicit sync: import timeline fd → handle, signal timeline points,
/// and register eventfd notifications.
#[derive(Clone)]
pub struct SyncobjDevice {
    fd: Arc<OwnedFd>,
}

impl SyncobjDevice {
    /// Create a new syncobj device from an owned DRM fd.
    pub fn new(fd: OwnedFd) -> Self {
        Self { fd: Arc::new(fd) }
    }

    /// Import a client's DRM syncobj timeline fd into a local handle.
    pub fn import_timeline(&self, timeline_fd: BorrowedFd<'_>) -> std::io::Result<SyncobjHandle> {
        use drm::control::Device;
        let dev = DrmRef(self.fd.as_fd());
        let handle = dev.fd_to_syncobj(timeline_fd, false)?;
        let raw: u32 = handle.into();
        Ok(SyncobjHandle(raw))
    }

    /// Signal a timeline point — tells the client the buffer is released.
    pub fn timeline_signal(&self, handle: SyncobjHandle, point: u64) -> std::io::Result<()> {
        use drm::control::Device;
        let dev = DrmRef(self.fd.as_fd());
        let h = to_handle(handle.0)?;
        dev.syncobj_timeline_signal(&[h], &[point])
    }

    /// Check if the acquire fence has been signaled (non-blocking).
    ///
    /// Uses `syncobj_timeline_query` (same as gamescope's `drmSyncobjQuery`)
    /// to read the current signaled point of the timeline. Returns `true`
    /// if the signaled point is >= the target point, meaning the GPU has
    /// finished the work associated with this fence.
    ///
    /// This is preferred over `syncobj_timeline_wait(timeout=0)` because
    /// it is a pure query with no wait semantics — identical to what
    /// gamescope uses in its fast path.
    pub fn is_acquire_ready(&self, handle: SyncobjHandle, point: u64) -> bool {
        use drm::control::Device;
        let dev = DrmRef(self.fd.as_fd());
        let Ok(h) = to_handle(handle.0) else {
            return false;
        };
        let mut signaled_points = [0u64];
        if dev
            .syncobj_timeline_query(&[h], &mut signaled_points, false)
            .is_ok()
        {
            signaled_points[0] >= point
        } else {
            false
        }
    }

    /// Export the fence at a timeline point as a sync file fd.
    ///
    /// Used to pass the acquire fence to Vulkan for GPU-side waiting.
    pub fn export_sync_file(
        &self,
        handle: SyncobjHandle,
        point: u64,
    ) -> std::io::Result<OwnedFd> {
        use drm::control::Device;
        let dev = DrmRef(self.fd.as_fd());
        // Transfer the timeline point to a temporary binary syncobj,
        // then export as sync file.
        let tmp = dev.create_syncobj(false)?;
        let h = to_handle(handle.0)?;
        dev.syncobj_timeline_transfer(h, tmp, point, 0)?;
        let fd = dev.syncobj_to_fd(tmp, true)?;
        dev.destroy_syncobj(tmp)?;
        Ok(fd)
    }

    /// Destroy a syncobj handle.
    pub fn destroy(&self, handle: SyncobjHandle) {
        use drm::control::Device;
        let dev = DrmRef(self.fd.as_fd());
        if let Ok(h) = to_handle(handle.0) {
            let _ = dev.destroy_syncobj(h);
        }
    }

    /// Test whether the kernel supports syncobj_eventfd (required for
    /// non-blocking acquire fence waiting).
    pub fn supports_eventfd(&self) -> bool {
        use drm::control::Device;
        let dev = DrmRef(self.fd.as_fd());
        // Create a dummy syncobj and try to register an eventfd on it.
        let Ok(tmp) = dev.create_syncobj(true) else {
            return false;
        };
        let efd = match rustix::event::eventfd(0, rustix::event::EventfdFlags::NONBLOCK) {
            Ok(fd) => fd,
            Err(_) => {
                let _ = dev.destroy_syncobj(tmp);
                return false;
            }
        };
        let result = dev.syncobj_eventfd(tmp, 0, efd.as_fd(), false);
        let _ = dev.destroy_syncobj(tmp);
        result.is_ok()
    }

    /// Create an eventfd that fires when the acquire fence at the given
    /// timeline point is signaled (GPU rendering complete).
    ///
    /// The returned `OwnedFd` becomes readable when the fence signals.
    /// Register it with an event loop (e.g., calloop) for non-blocking
    /// waiting instead of busy-polling `is_acquire_ready()`.
    ///
    /// Uses `DRM_IOCTL_SYNCOBJ_EVENTFD` with `wait_available = false`
    /// (wait for signal, not just materialization).
    pub fn acquire_eventfd(
        &self,
        handle: SyncobjHandle,
        point: u64,
    ) -> std::io::Result<OwnedFd> {
        use drm::control::Device;
        let dev = DrmRef(self.fd.as_fd());
        let h = to_handle(handle.0)?;
        let efd = rustix::event::eventfd(0, rustix::event::EventfdFlags::NONBLOCK)
            .map_err(std::io::Error::other)?;
        dev.syncobj_eventfd(h, point, efd.as_fd(), false)?;
        Ok(efd)
    }
}

/// Newtype wrapper for raw DRM fd → `drm::control::Device` trait.
struct DrmRef<'a>(BorrowedFd<'a>);

impl<'a> std::os::unix::io::AsFd for DrmRef<'a> {
    fn as_fd(&self) -> BorrowedFd<'_> {
        self.0
    }
}

impl<'a> drm::Device for DrmRef<'a> {}
impl<'a> drm::control::Device for DrmRef<'a> {}

/// Convert a raw u32 handle to a typed syncobj Handle.
#[inline(always)]
fn to_handle(raw: u32) -> std::io::Result<drm::control::syncobj::Handle> {
    drm::control::from_u32::<drm::control::syncobj::Handle>(raw)
        .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidInput, "invalid syncobj handle (0)"))
}

// ─── Per-timeline user data ─────────────────────────────────────────

/// User data on `wp_linux_drm_syncobj_timeline_v1`.
///
/// Holds an `Arc<SyncobjGuard>` so the kernel syncobj stays alive as
/// long as either the Wayland timeline object or any in-flight
/// `SyncPoint` references it.
pub struct TimelineData {
    /// Ref-counted guard owning the kernel syncobj handle.
    pub guard: Arc<SyncobjGuard>,
}

// ─── Per-surface sync state ─────────────────────────────────────────

/// Pending acquire/release points set on a surface before commit.
///
/// Double-buffered: filled by `set_acquire_point` / `set_release_point`,
/// consumed on `wl_surface.commit`.
#[derive(Default, Clone)]
pub struct SyncState {
    /// Acquire point — compositor waits for this before reading the buffer.
    pub acquire: Option<SyncPoint>,
    /// Release point — compositor signals this when done with the buffer.
    pub release: Option<SyncPoint>,
}

/// User data on `wp_linux_drm_syncobj_surface_v1`.
pub struct SurfaceSyncData {
    /// The `wl_surface` this sync object is bound to.
    pub surface: WlSurface,
    /// XWayland server index (from `SurfaceData`).
    pub server_index: u32,
    /// Pending sync state (double-buffered, consumed on commit).
    pub pending: Mutex<SyncState>,
}

// ─── Global data ────────────────────────────────────────────────────

/// Global user data for the syncobj manager.
pub struct SyncobjGlobalData {
    /// DRM device for syncobj operations. `None` in nested mode
    /// (explicit sync requires DRM).
    pub device: Option<SyncobjDevice>,
}

// ─── Protocol dispatch implementations ──────────────────────────────

impl GlobalDispatch<WpLinuxDrmSyncobjManagerV1, SyncobjGlobalData> for WaylandState {
    fn bind(
        _state: &mut Self,
        _dh: &DisplayHandle,
        _client: &Client,
        resource: New<WpLinuxDrmSyncobjManagerV1>,
        data: &SyncobjGlobalData,
        data_init: &mut DataInit<'_, Self>,
    ) {
        let device = data.device.clone();
        data_init.init(resource, device);
        info!("wp_linux_drm_syncobj_manager_v1: bound");
    }
}

impl Dispatch<WpLinuxDrmSyncobjManagerV1, Option<SyncobjDevice>> for WaylandState {
    fn request(
        state: &mut Self,
        _client: &Client,
        _resource: &WpLinuxDrmSyncobjManagerV1,
        request: wp_linux_drm_syncobj_manager_v1::Request,
        data: &Option<SyncobjDevice>,
        _dh: &DisplayHandle,
        data_init: &mut DataInit<'_, Self>,
    ) {
        match request {
            wp_linux_drm_syncobj_manager_v1::Request::GetSurface { id, surface } => {
                let surface_id = surface.id().protocol_id();
                let server_index = surface
                    .data::<super::SurfaceData>()
                    .map_or(u32::MAX, |d| d.server_index);
                let key = (surface_id, server_index);
                // If the surface already has a sync object, the protocol
                // spec says this is an error. However, when multiple
                // XWayland servers share the same wl_surface ID space,
                // surface IDs can be reused. Replace the stale entry
                // rather than panicking.
                if state.syncobj_surfaces.contains_key(&key) {
                    warn!(
                        surface_id,
                        server_index,
                        "syncobj: surface already has a sync object, replacing"
                    );
                    state.syncobj_surfaces.remove(&key);
                }
                let sync_surface = data_init.init(
                    id,
                    SurfaceSyncData {
                        surface: surface.clone(),
                        server_index,
                        pending: Mutex::new(SyncState::default()),
                    },
                );
                state.syncobj_surfaces.insert(key, sync_surface);
                debug!(surface_id, server_index, "syncobj: created surface sync");
            }
            wp_linux_drm_syncobj_manager_v1::Request::ImportTimeline { id, fd } => {
                let Some(device) = data else {
                    warn!("syncobj: import_timeline called but no DRM device");
                    return;
                };
                match device.import_timeline(fd.as_fd()) {
                    Ok(handle) => {
                        let guard = Arc::new(SyncobjGuard {
                            handle,
                            device: device.clone(),
                        });
                        data_init.init(id, TimelineData { guard });
                        debug!(?handle, "syncobj: imported timeline");
                    }
                    Err(e) => {
                        warn!(?e, "syncobj: failed to import timeline fd");
                    }
                }
            }
            wp_linux_drm_syncobj_manager_v1::Request::Destroy => {}
            _ => {}
        }
    }
}

impl Dispatch<WpLinuxDrmSyncobjTimelineV1, TimelineData> for WaylandState {
    fn request(
        _state: &mut Self,
        _client: &Client,
        _resource: &WpLinuxDrmSyncobjTimelineV1,
        request: wp_linux_drm_syncobj_timeline_v1::Request,
        _data: &TimelineData,
        _dh: &DisplayHandle,
        _data_init: &mut DataInit<'_, Self>,
    ) {
        #[allow(clippy::single_match)]
        match request {
            wp_linux_drm_syncobj_timeline_v1::Request::Destroy => {
                trace!("syncobj: timeline destroyed");
            }
            _ => {}
        }
    }
}

impl Dispatch<WpLinuxDrmSyncobjSurfaceV1, SurfaceSyncData> for WaylandState {
    fn request(
        state: &mut Self,
        _client: &Client,
        _resource: &WpLinuxDrmSyncobjSurfaceV1,
        request: wp_linux_drm_syncobj_surface_v1::Request,
        data: &SurfaceSyncData,
        _dh: &DisplayHandle,
        _data_init: &mut DataInit<'_, Self>,
    ) {
        match request {
            wp_linux_drm_syncobj_surface_v1::Request::SetAcquirePoint {
                timeline,
                point_hi,
                point_lo,
            } => {
                let point = ((point_hi as u64) << 32) | (point_lo as u64);
                let timeline_data = timeline.data::<TimelineData>().unwrap();
                let mut pending = data.pending.lock();
                pending.acquire = Some(SyncPoint {
                    guard: Arc::clone(&timeline_data.guard),
                    point,
                });
                trace!(point, handle = ?timeline_data.guard.handle, "syncobj: set acquire point");
            }
            wp_linux_drm_syncobj_surface_v1::Request::SetReleasePoint {
                timeline,
                point_hi,
                point_lo,
            } => {
                let point = ((point_hi as u64) << 32) | (point_lo as u64);
                let timeline_data = timeline.data::<TimelineData>().unwrap();
                let mut pending = data.pending.lock();
                pending.release = Some(SyncPoint {
                    guard: Arc::clone(&timeline_data.guard),
                    point,
                });
                trace!(point, handle = ?timeline_data.guard.handle, "syncobj: set release point");
            }
            wp_linux_drm_syncobj_surface_v1::Request::Destroy => {
                let surface_id = data.surface.id().protocol_id();
                let key = (surface_id, data.server_index);
                state.syncobj_surfaces.remove(&key);
                debug!(surface_id, server_index = data.server_index, "syncobj: surface sync destroyed");
            }
            _ => {}
        }
    }
}

/// Extract and consume the pending sync state for a surface (called at commit time).
///
/// Returns `None` if the surface has no syncobj, or `Some(SyncState)` if
/// acquire/release points were set. The pending state is cleared.
pub fn take_pending_sync(state: &WaylandState, surface_id: u32, server_index: u32) -> Option<SyncState> {
    let sync_surface = state.syncobj_surfaces.get(&(surface_id, server_index))?;
    let sync_data = sync_surface.data::<SurfaceSyncData>()?;
    let mut pending = sync_data.pending.lock();
    let sync = SyncState {
        acquire: pending.acquire.take(),
        release: pending.release.take(),
    };
    if sync.acquire.is_some() || sync.release.is_some() {
        Some(sync)
    } else {
        None
    }
}
