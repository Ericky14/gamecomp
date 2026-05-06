//! `wl_compositor`, `wl_surface`, `wl_region`, and `wl_callback` dispatch.

use parking_lot::Mutex;
use std::os::unix::io::AsFd;
use std::sync::atomic::Ordering;

use tracing::{debug, trace};
use wayland_server::protocol::{
    wl_callback::{self, WlCallback},
    wl_compositor::{self, WlCompositor},
    wl_region::{self, WlRegion},
    wl_shm,
    wl_surface::{self, WlSurface},
};
use wayland_server::{Client, DataInit, Dispatch, DisplayHandle, GlobalDispatch, New, Resource};

use super::{BufferData, CommittedBuffer, CommittedDmaBufPlane, SurfaceData, sync_dma_buf_fence};
use crate::wayland::WaylandState;

/// Classification of a `wl_surface.commit` for the commit gate.
///
/// Determines how the compositor handles the committed buffer:
/// - `App`: staged for fullscreen presentation (FPS-limited)
/// - `Overlay` / `ExternalOverlay`: composited on top of the app
/// - `PassThrough`: accepted to keep the client's Present chain alive,
///   but NOT staged for display
///
/// **Invariant**: every path that is NOT `App` must signal any pending
/// DRM syncobj release point immediately. The buffer is never composited,
/// so the compositor must tell the client's GPU it can reuse the buffer.
/// Failure to signal causes buffer pool exhaustion and a dead Present chain.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CommitTarget {
    App,
    Overlay,
    ExternalOverlay,
    PassThrough,
}

/// Input parameters for commit classification, decoupled from Wayland
/// protocol objects so the logic is unit-testable.
#[derive(Debug, Clone, Copy)]
pub struct CommitGateInput {
    pub surface_id: u32,
    pub surface_srv: u32,
    pub focused_id: u32,
    pub focused_srv: u32,
    pub overlay_id: u32,
    pub overlay_srv: u32,
    pub ext_overlay_id: u32,
    pub ext_overlay_srv: u32,
}

/// Pure function: classify a surface commit.
///
/// Returns `Some(target)` if the commit should be accepted, or `None` if
/// rejected (surface doesn't match any known target).
///
/// **Critical rules tested by unit tests:**
/// - Server 0 surfaces are always accepted (at least `PassThrough`).
/// - Non-server-0, non-matching surfaces are rejected.
/// - Overlay/ExternalOverlay match takes priority over PassThrough.
#[must_use]
pub fn classify_commit(input: &CommitGateInput) -> Option<CommitTarget> {
    if input.focused_id != 0
        && input.surface_id == input.focused_id
        && input.surface_srv == input.focused_srv
    {
        Some(CommitTarget::App)
    } else if input.overlay_id != 0
        && input.surface_id == input.overlay_id
        && input.surface_srv == input.overlay_srv
    {
        Some(CommitTarget::Overlay)
    } else if input.ext_overlay_id != 0
        && input.surface_id == input.ext_overlay_id
        && input.surface_srv == input.ext_overlay_srv
    {
        Some(CommitTarget::ExternalOverlay)
    } else if input.surface_srv == 0 {
        Some(CommitTarget::PassThrough)
    } else {
        None
    }
}

impl GlobalDispatch<WlCompositor, ()> for WaylandState {
    fn bind(
        _state: &mut Self,
        _dh: &DisplayHandle,
        _client: &Client,
        resource: New<WlCompositor>,
        _data: &(),
        data_init: &mut DataInit<'_, Self>,
    ) {
        data_init.init(resource, ());
    }
}

impl Dispatch<WlCompositor, ()> for WaylandState {
    fn request(
        state: &mut Self,
        _client: &Client,
        _resource: &WlCompositor,
        request: wl_compositor::Request,
        _data: &(),
        _dh: &DisplayHandle,
        data_init: &mut DataInit<'_, Self>,
    ) {
        match request {
            wl_compositor::Request::CreateSurface { id } => {
                debug!("wl_compositor: create_surface");
                let server_index = state
                    .xwayland_client_map
                    .get(&_client.id())
                    .copied()
                    .unwrap_or(u32::MAX);
                let surface = data_init.init(
                    id,
                    SurfaceData {
                        attached_buffer: Mutex::new(None),
                        is_cursor: std::sync::atomic::AtomicBool::new(false),
                        hotspot_x: std::sync::atomic::AtomicI32::new(0),
                        hotspot_y: std::sync::atomic::AtomicI32::new(0),
                        server_index,
                        pending_feedbacks: Mutex::new(Vec::new()),
                    },
                );
                // Track all client surfaces for per-client focus enter.
                state.client_surfaces.push(surface);
            }
            wl_compositor::Request::CreateRegion { id } => {
                data_init.init(id, ());
            }
            _ => {}
        }
    }
}

impl Dispatch<WlSurface, SurfaceData> for WaylandState {
    fn request(
        state: &mut Self,
        _client: &Client,
        _surface: &WlSurface,
        request: wl_surface::Request,
        data: &SurfaceData,
        _dh: &DisplayHandle,
        data_init: &mut DataInit<'_, Self>,
    ) {
        match request {
            wl_surface::Request::Attach { buffer, x: _, y: _ } => {
                // Store the buffer reference. We'll read pixels on commit.
                *data.attached_buffer.lock() = buffer;
            }
            wl_surface::Request::Damage { .. } | wl_surface::Request::DamageBuffer { .. } => {
                // Mark surface as damaged for next frame.
            }
            wl_surface::Request::Frame { callback } => {
                let cb = data_init.init(callback, ());
                let surface_id = _surface.id().protocol_id();
                state.push_surface_callback(surface_id, data.server_index, cb);
            }
            wl_surface::Request::Commit => {
                // Cursor surfaces: read SHM pixels and forward to host
                // compositor instead of staging as application frames.
                if data.is_cursor.load(Ordering::Relaxed) {
                    let attached = data.attached_buffer.lock();
                    if let Some(ref buffer) = *attached
                        && let Some(BufferData::Shm(shm)) = buffer.data::<BufferData>()
                    {
                        let len = (shm.stride * shm.height) as usize;
                        if let Some(pixels) = shm.pool.lock().read_pixels(shm.offset as usize, len)
                        {
                            // Detect fully-transparent cursor images (e.g.,
                            // GTK/Flutter `SystemMouseCursors.none` sends a
                            // blank cursor surface instead of null). In
                            // ARGB8888, alpha is byte 3 of each 4-byte pixel
                            // on little-endian.
                            let is_argb = shm.format == wl_shm::Format::Argb8888;
                            let all_transparent = is_argb
                                && pixels.len() >= 4
                                && pixels[3..].iter().step_by(4).all(|&a| a == 0);
                            if all_transparent {
                                if let Some(ref tx) = state.cursor_tx {
                                    let _ = tx.send(crate::backend::wayland::CursorUpdate::Hide);
                                }
                            } else if state.cursor_user_moved {
                                let hotspot_x = data.hotspot_x.load(Ordering::Relaxed);
                                let hotspot_y = data.hotspot_y.load(Ordering::Relaxed);
                                if let Some(ref tx) = state.cursor_tx {
                                    let _ = tx.send(crate::backend::wayland::CursorUpdate::Image {
                                        pixels,
                                        width: shm.width as u32,
                                        height: shm.height as u32,
                                        hotspot_x,
                                        hotspot_y,
                                    });
                                }
                            }
                        }
                    } else {
                        // No buffer attached — hide cursor.
                        if let Some(ref tx) = state.cursor_tx {
                            let _ = tx.send(crate::backend::wayland::CursorUpdate::Hide);
                        }
                    }
                    // Fire cursor surface's frame callbacks so the client
                    // can keep animating the cursor.
                    let cid = _surface.id().protocol_id();
                    state.fire_surface_pending(cid, data.server_index);
                    return;
                }

                // Classify this surface: app, overlay, external overlay, or reject.
                // Classify this surface commit: app, overlay, external overlay,
                // or "pass-through" (accepted but not staged).
                //
                // Gamescope accepts ALL commits from ALL windows — it never
                // rejects or gates any surface. This keeps every client's
                // XWayland Present chain alive continuously, so overlays
                // resume instantly when (re)activated.
                //
                // We match this by accepting all commits from server 0
                // (the platform / Grid server) even when it's not focused.
                // Server 0 surfaces are either the overlay or the platform
                // app — both must keep their Present chains alive.
                let target = {
                    let input = CommitGateInput {
                        surface_id: _surface.id().protocol_id(),
                        surface_srv: data.server_index,
                        focused_id: state.focused_wl_surface_id.load(Ordering::Relaxed),
                        focused_srv: state.focused_server_index.load(Ordering::Relaxed),
                        overlay_id: state.overlay_wl_surface_id.load(Ordering::Relaxed),
                        overlay_srv: state.overlay_server_index.load(Ordering::Relaxed),
                        ext_overlay_id: state
                            .external_overlay_wl_surface_id
                            .load(Ordering::Relaxed),
                        ext_overlay_srv: state
                            .external_overlay_server_index
                            .load(Ordering::Relaxed),
                    };
                    classify_commit(&input)
                };

                let Some(target) = target else {
                    // Log rejection with full gate state for debugging
                    // overlay issues. Only at debug level for non-app
                    // surfaces that might be overlays.
                    let surface_id = _surface.id().protocol_id();
                    let srv = data.server_index;
                    let overlay_id = state.overlay_wl_surface_id.load(Ordering::Relaxed);
                    if overlay_id != 0
                        || surface_id != state.focused_wl_surface_id.load(Ordering::Relaxed)
                    {
                        debug!(
                            surface_id,
                            server_index = srv,
                            focused_id = state.focused_wl_surface_id.load(Ordering::Relaxed),
                            focused_srv = state.focused_server_index.load(Ordering::Relaxed),
                            overlay_id,
                            overlay_srv = state.overlay_server_index.load(Ordering::Relaxed),
                            "commit gate: rejected (potential overlay miss)"
                        );
                    } else {
                        trace!(surface_id, server_index = srv, "commit gate: rejected");
                    }
                    // Signal syncobj release so the client's GPU driver
                    // doesn't starve waiting for a fence that will never
                    // come. Same rationale as the PassThrough path.
                    if let Some(sync) =
                        super::drm_syncobj::take_pending_sync(state, surface_id, srv)
                        && let Some(rp) = sync.release
                        && let Some(ref device) = state.syncobj_device
                    {
                        let _ = device.timeline_signal(rp.handle(), rp.point);
                    }
                    // Release the buffer so the client doesn't starve.
                    let attached = data.attached_buffer.lock();
                    if let Some(ref buffer) = *attached {
                        buffer.release();
                    }
                    // Defer frame callbacks to vblank so the rejected
                    // client keeps its frame callback cycle alive.
                    // Without this, a wl_surface.frame() from a rejected
                    // commit sits in the surface's pending callbacks
                    // indefinitely — the client never gets wl_callback.done
                    // and stops rendering. Using defer (not immediate fire)
                    // matches gamescope's vblank-gated flush_frame_done.
                    let rej_id = _surface.id().protocol_id();
                    state.defer_surface_callbacks(rej_id, data.server_index);
                    return;
                };

                // PassThrough: accept the commit to keep the client's
                // Present chain alive, but don't stage the buffer.
                // Signal syncobj release immediately, defer callbacks
                // to vblank (matching gamescope's vblank-gated flush).
                if matches!(target, CommitTarget::PassThrough) {
                    let surface_id = _surface.id().protocol_id();
                    trace!(
                        surface_id,
                        server_index = data.server_index,
                        "commit gate: accepted pass-through (not staged)"
                    );
                    // Consume pending syncobj and signal release immediately.
                    // PassThrough buffers are never composited, so the
                    // compositor is "done" with them right away. Without
                    // signaling, the client's GPU driver waits forever for
                    // the release fence, exhausting XWayland's buffer pool
                    // after 2-3 commits and killing the Present chain.
                    if let Some(sync) =
                        super::drm_syncobj::take_pending_sync(state, surface_id, data.server_index)
                        && let Some(rp) = sync.release
                        && let Some(ref device) = state.syncobj_device
                    {
                        let _ = device.timeline_signal(rp.handle(), rp.point);
                    }
                    // Defer callbacks to vblank instead of firing
                    // immediately. Gamescope gates frame callbacks on
                    // BOTH receivedDoneCommit AND unlockedForFrameCallback
                    // (next vblank). Firing immediately collapses the
                    // commit→callback round-trip to zero, causing
                    // XWayland's Present chain to go idle between
                    // vblanks. Deferring keeps a pending vblank wait
                    // in the Present chain at all times.
                    state.defer_surface_callbacks(surface_id, data.server_index);
                    // Track the wl_buffer so wl_buffer.release() fires
                    // when the buffer is no longer needed. Do NOT call
                    // buffer.release() here — held_buffers handles
                    // release via release_stale_buffers(). Releasing
                    // here AND pushing to held_buffers is a double-
                    // release protocol violation.
                    let attached = data.attached_buffer.lock();
                    if let Some(ref buffer) = *attached {
                        state.held_buffers.push(buffer.clone());
                    }
                    return;
                }

                state.frame_seq += 1;
                let target_name = match target {
                    CommitTarget::App => "app",
                    CommitTarget::Overlay => "overlay",
                    CommitTarget::ExternalOverlay => "external_overlay",
                    CommitTarget::PassThrough => unreachable!(),
                };
                if matches!(
                    target,
                    CommitTarget::Overlay | CommitTarget::ExternalOverlay
                ) {
                    debug!(
                        frame_seq = state.frame_seq,
                        surface_id = _surface.id().protocol_id(),
                        server_index = data.server_index,
                        target = target_name,
                        "commit gate: accepted overlay"
                    );
                } else {
                    trace!(
                        frame_seq = state.frame_seq,
                        surface_id = _surface.id().protocol_id(),
                        server_index = data.server_index,
                        target = target_name,
                        "commit gate: accepted"
                    );
                }

                // Read pixels from the attached buffer and stage for presentation.
                let attached = data.attached_buffer.lock();
                if let Some(ref buffer) = *attached {
                    let buf_data_opt = buffer.data::<BufferData>();
                    if let Some(buf_data) = buf_data_opt {
                        match buf_data {
                            BufferData::DmaBuf(dmabuf) => {
                                // Check for explicit sync points on this surface.
                                let sync = super::drm_syncobj::take_pending_sync(
                                    state,
                                    _surface.id().protocol_id(),
                                    data.server_index,
                                );
                                let has_explicit_sync = sync
                                    .as_ref()
                                    .is_some_and(|s| s.acquire.is_some() && s.release.is_some());

                                // On the nested Wayland path, wait for the
                                // client GPU to finish writing to the DMA-BUF
                                // before forwarding to the host compositor.
                                // On DRM, the render thread handles sync
                                // asynchronously — skip the blocking wait.
                                // With explicit sync, the acquire fence replaces
                                // implicit sync entirely.
                                if !has_explicit_sync
                                    && !state.drm_mode
                                    && let Some(first_plane) = dmabuf.planes.first()
                                {
                                    sync_dma_buf_fence(&first_plane.fd);
                                }
                                // Zero-copy path: dup the fds and forward metadata.
                                let planes: Vec<CommittedDmaBufPlane> = dmabuf
                                    .planes
                                    .iter()
                                    .filter_map(|p| {
                                        rustix::io::dup(p.fd.as_fd()).ok().map(|fd| {
                                            CommittedDmaBufPlane {
                                                fd,
                                                offset: p.offset,
                                                stride: p.stride,
                                            }
                                        })
                                    })
                                    .collect();
                                let release_point = sync.as_ref().and_then(|s| s.release.clone());
                                let acquire_point = sync.as_ref().and_then(|s| s.acquire.clone());
                                let committed = CommittedBuffer::DmaBuf {
                                    planes,
                                    width: dmabuf.width as u32,
                                    height: dmabuf.height as u32,
                                    format: dmabuf.format,
                                    modifier: dmabuf.modifier,
                                    acquire_point,
                                    release_point,
                                };
                                match target {
                                    CommitTarget::App => {
                                        let now = crate::wayland::monotonic_ns();
                                        let since_cb = if state.last_callback_fire_ns > 0 {
                                            (now - state.last_callback_fire_ns) / 1_000_000
                                        } else {
                                            0
                                        };
                                        trace!(
                                            since_last_callback_ms = since_cb,
                                            held_count = state.held_buffers.len(),
                                            has_explicit_sync,
                                            "commit: app frame staged (time since last callback fire)"
                                        );
                                        // If a previously-staged frame hasn't been forwarded yet,
                                        // its presentation feedbacks must be discarded — it will
                                        // never reach the screen.
                                        crate::wayland::protocols::presentation::discard_staged(
                                            state,
                                        );
                                        // Signal the old staged buffer's release point so the
                                        // client can reclaim it. Without this, coalesced frames
                                        // permanently lock client buffers.
                                        if let Some(mut old_staged) = state.staged_buffer.take()
                                            && let Some(ref device) = state.syncobj_device
                                        {
                                            old_staged.signal_release(device);
                                        }
                                        // Move this surface's pending presentation feedbacks
                                        // onto the staged buffer.
                                        let mut pending = data.pending_feedbacks.lock();
                                        state.staged_feedbacks.append(&mut *pending);
                                        drop(pending);
                                        state.staged_buffer = Some(committed);
                                        state.staged_buffer_server_index = data.server_index;
                                        state.staged_at_ns = now;
                                        state.commit_count += 1;
                                        let sid = _surface.id().protocol_id();
                                        state.defer_surface_callbacks(sid, data.server_index);
                                        // With explicit sync, the release point in the
                                        // CommittedBuffer tells the render thread when
                                        // to signal the syncobj timeline. We still push
                                        // to held_buffers as a fallback for wl_buffer.release.
                                        state.held_buffers.push(buffer.clone());
                                    }
                                    CommitTarget::Overlay => {
                                        // Signal old overlay's release point so the
                                        // client can recycle the previous buffer.
                                        if let Some(mut old) = state.staged_overlay_buffer.take()
                                            && let Some(ref device) = state.syncobj_device
                                        {
                                            old.signal_release(device);
                                        }
                                        state.staged_overlay_buffer = Some(committed);
                                        // Stage a dup as primary content ONLY when
                                        // no game is focused. When a game IS running,
                                        // let game commits drive the pipeline and the
                                        // overlay goes through staged_overlay_buffer
                                        // → dup_or_clear_overlay() in forward_staged_frame.
                                        //
                                        // Without this guard, the overlay's acquire fence
                                        // ends up in frame.app, blocking the render thread
                                        // on a slow overlay fence and starving the game
                                        // (attempt 6 failure: 5-second fence stall → black screen).
                                        let has_focused_app =
                                            state.focused_wl_surface_id.load(Ordering::Relaxed)
                                                != 0;
                                        if !has_focused_app
                                            && let Some(ref buf) = state.staged_overlay_buffer
                                            && let Ok(dup) = buf.try_dup()
                                        {
                                            let now = crate::wayland::monotonic_ns();
                                            state.staged_buffer = Some(dup);
                                            state.staged_buffer_server_index = data.server_index;
                                            state.staged_at_ns = now;
                                            state.commit_count += 1;
                                        }
                                        // Defer overlay callbacks to vblank instead
                                        // of firing immediately. Gamescope gates
                                        // frame callbacks on receivedDoneCommit AND
                                        // unlockedForFrameCallback (next vblank).
                                        // Immediate fire collapses the commit→callback
                                        // round-trip to zero, causing XWayland's
                                        // Present chain to idle between vblanks and
                                        // Flutter to stop committing.
                                        let ov_id = _surface.id().protocol_id();
                                        state.defer_surface_callbacks(ov_id, data.server_index);
                                        // Track the wl_buffer so wl_buffer.release()
                                        // is sent when the buffer is no longer needed.
                                        // Without this, XWayland's buffer pool is
                                        // exhausted after 2-3 commits and Grid stops
                                        // rendering (PresentIdleNotify never fires).
                                        state.held_buffers.push(buffer.clone());
                                    }
                                    CommitTarget::ExternalOverlay => {
                                        // Signal old external overlay's release point.
                                        if let Some(mut old) =
                                            state.staged_external_overlay_buffer.take()
                                            && let Some(ref device) = state.syncobj_device
                                        {
                                            old.signal_release(device);
                                        }
                                        state.staged_external_overlay_buffer = Some(committed);
                                        let has_focused_app =
                                            state.focused_wl_surface_id.load(Ordering::Relaxed)
                                                != 0;
                                        if !has_focused_app
                                            && let Some(ref buf) =
                                                state.staged_external_overlay_buffer
                                            && let Ok(dup) = buf.try_dup()
                                        {
                                            let now = crate::wayland::monotonic_ns();
                                            state.staged_buffer = Some(dup);
                                            state.staged_buffer_server_index = data.server_index;
                                            state.staged_at_ns = now;
                                            state.commit_count += 1;
                                        }
                                        let eov_id = _surface.id().protocol_id();
                                        state.defer_surface_callbacks(eov_id, data.server_index);
                                        state.held_buffers.push(buffer.clone());
                                    }
                                    // PassThrough handled by early return above.
                                    CommitTarget::PassThrough => unreachable!(),
                                }
                            }
                            BufferData::Shm(shm) => {
                                // CPU-copy fallback for SHM buffers.
                                let offset = shm.offset as usize;
                                let len = (shm.stride * shm.height) as usize;
                                let pool = shm.pool.lock();
                                if let Some(pixels) = pool.read_pixels(offset, len) {
                                    trace!(
                                        width = shm.width,
                                        height = shm.height,
                                        pixel_bytes = pixels.len(),
                                        "commit: staging SHM frame for FPS-limited forwarding"
                                    );
                                    let committed = CommittedBuffer::Shm {
                                        pixels,
                                        width: shm.width as u32,
                                        height: shm.height as u32,
                                        stride: shm.stride as u32,
                                    };
                                    match target {
                                        CommitTarget::App => {
                                            let now = crate::wayland::monotonic_ns();
                                            state.staged_at_ns = now;
                                            state.commit_count += 1;
                                            crate::wayland::protocols::presentation::discard_staged(
                                                state,
                                            );
                                            if let Some(mut old_staged) = state.staged_buffer.take()
                                                && let Some(ref device) = state.syncobj_device
                                            {
                                                old_staged.signal_release(device);
                                            }
                                            let mut pending = data.pending_feedbacks.lock();
                                            state.staged_feedbacks.append(&mut *pending);
                                            drop(pending);
                                            state.staged_buffer = Some(committed);
                                            state.staged_buffer_server_index = data.server_index;
                                            let sid = _surface.id().protocol_id();
                                            state.defer_surface_callbacks(sid, data.server_index);
                                            state.held_buffers.push(buffer.clone());
                                        }
                                        CommitTarget::Overlay => {
                                            if let Some(mut old) =
                                                state.staged_overlay_buffer.take()
                                                && let Some(ref device) = state.syncobj_device
                                            {
                                                old.signal_release(device);
                                            }
                                            state.staged_overlay_buffer = Some(committed);
                                            if let Some(ref buf) = state.staged_overlay_buffer
                                                && let Ok(dup) = buf.try_dup()
                                            {
                                                let now = crate::wayland::monotonic_ns();
                                                state.staged_buffer = Some(dup);
                                                state.staged_buffer_server_index =
                                                    data.server_index;
                                                state.staged_at_ns = now;
                                                state.commit_count += 1;
                                            }
                                            let ov_id = _surface.id().protocol_id();
                                            state.fire_surface_pending(ov_id, data.server_index);
                                        }
                                        CommitTarget::ExternalOverlay => {
                                            if let Some(mut old) =
                                                state.staged_external_overlay_buffer.take()
                                                && let Some(ref device) = state.syncobj_device
                                            {
                                                old.signal_release(device);
                                            }
                                            state.staged_external_overlay_buffer = Some(committed);
                                            if let Some(ref buf) =
                                                state.staged_external_overlay_buffer
                                                && let Ok(dup) = buf.try_dup()
                                            {
                                                let now = crate::wayland::monotonic_ns();
                                                state.staged_buffer = Some(dup);
                                                state.staged_buffer_server_index =
                                                    data.server_index;
                                                state.staged_at_ns = now;
                                                state.commit_count += 1;
                                            }
                                            let eov_id = _surface.id().protocol_id();
                                            state.fire_surface_pending(eov_id, data.server_index);
                                        }
                                        // PassThrough handled by early return above.
                                        CommitTarget::PassThrough => unreachable!(),
                                    }
                                }
                            }
                        }
                    } else {
                        debug!("commit: buffer has no BufferData");
                    }
                } else {
                    trace!("commit: no attached buffer");
                }
            }
            wl_surface::Request::SetBufferScale { .. }
            | wl_surface::Request::SetBufferTransform { .. }
            | wl_surface::Request::Offset { .. } => {}
            wl_surface::Request::Destroy => {
                // Discard any pending presentation feedbacks so the
                // client's wp_presentation_feedback objects don't leak.
                crate::wayland::protocols::presentation::discard_surface_pending(data);
            }
            _ => {}
        }
    }
}

impl Dispatch<WlRegion, ()> for WaylandState {
    fn request(
        _state: &mut Self,
        _client: &Client,
        _resource: &WlRegion,
        _request: wl_region::Request,
        _data: &(),
        _dh: &DisplayHandle,
        _data_init: &mut DataInit<'_, Self>,
    ) {
        // Handle Add, Subtract, Destroy — stub for now.
    }
}

impl Dispatch<WlCallback, ()> for WaylandState {
    fn request(
        _state: &mut Self,
        _client: &Client,
        _resource: &WlCallback,
        _request: wl_callback::Request,
        _data: &(),
        _dh: &DisplayHandle,
        _data_init: &mut DataInit<'_, Self>,
    ) {
        // wl_callback has no requests — only the `done` event.
    }
}

#[cfg(test)]
#[path = "compositor_tests.rs"]
mod tests;
