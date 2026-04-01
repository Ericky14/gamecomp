//! `wl_compositor`, `wl_surface`, `wl_region`, and `wl_callback` dispatch.

use parking_lot::Mutex;
use std::os::unix::io::AsFd;
use std::sync::atomic::Ordering;

use tracing::{debug, trace};
use wayland_server::protocol::{
    wl_callback::{self, WlCallback},
    wl_compositor::{self, WlCompositor},
    wl_region::{self, WlRegion},
    wl_surface::{self, WlSurface},
};
use wayland_server::{Client, DataInit, Dispatch, DisplayHandle, GlobalDispatch, New, Resource};

use super::{BufferData, CommittedBuffer, CommittedDmaBufPlane, SurfaceData, sync_dma_buf_fence};
use crate::wayland::WaylandState;

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
                state.pending_frame_callbacks.push(cb);
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
                    // Fire cursor surface's frame callbacks so the client
                    // can keep animating the cursor.
                    state.fire_frame_callbacks();
                    return;
                }

                // Only the focused window's surface may present. Reject
                // commits from other surfaces. Release their attached
                // buffer immediately so the client can recycle it —
                // otherwise the client exhausts all swapchain images
                // and blocks permanently on vkAcquireNextImage.
                {
                    let focused_id = state.focused_wl_surface_id.load(Ordering::Relaxed);
                    let focused_srv = state.focused_server_index.load(Ordering::Relaxed);
                    let surface_id = _surface.id().protocol_id();
                    let surface_srv = data.server_index;
                    if focused_id == 0 || surface_id != focused_id || surface_srv != focused_srv {
                        trace!(
                            surface_id,
                            surface_srv, focused_id, focused_srv, "commit gate: rejected"
                        );
                        // Release the buffer so the client doesn't starve.
                        let attached = data.attached_buffer.lock();
                        if let Some(ref buffer) = *attached {
                            buffer.release();
                        }
                        return;
                    }
                }

                state.frame_seq += 1;
                trace!(
                    frame_seq = state.frame_seq,
                    surface_id = _surface.id().protocol_id(),
                    server_index = data.server_index,
                    "commit gate: accepted"
                );

                // Read pixels from the attached buffer and send to presenter.
                let attached = data.attached_buffer.lock();
                if let Some(ref buffer) = *attached {
                    let buf_data_opt = buffer.data::<BufferData>();
                    if let Some(buf_data) = buf_data_opt {
                        match buf_data {
                            BufferData::DmaBuf(dmabuf) => {
                                // Ensure GPU has finished writing to the DMA-BUF
                                // before forwarding. Export an implicit sync fence
                                // and wait on it. This prevents the host compositor
                                // from reading an incomplete render.
                                if let Some(first_plane) = dmabuf.planes.first() {
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
                                let committed = CommittedBuffer::DmaBuf {
                                    planes,
                                    width: dmabuf.width as u32,
                                    height: dmabuf.height as u32,
                                    format: dmabuf.format,
                                    modifier: dmabuf.modifier,
                                };
                                // Stage the buffer for FPS-limited forwarding.
                                // The main loop forwards the staged buffer to
                                // the render thread when the FPS limiter allows.
                                // If the client is committing faster than the
                                // target FPS, the previous staged buffer is
                                // silently dropped (its dup'd fds are closed).
                                state.staged_buffer = Some(committed);
                                state.staged_buffer_server_index = data.server_index;
                                state.defer_frame_callbacks();
                                // Hold the wl_buffer — do NOT release it.
                                // By withholding release, the client cannot
                                // recycle this buffer. Once all client buffers
                                // are held, the client blocks. The main loop
                                // releases held buffers when the FPS limiter
                                // fires, creating real backpressure.
                                state.held_buffers.push(buffer.clone());
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
                                    state.staged_buffer = Some(committed);
                                    state.staged_buffer_server_index = data.server_index;
                                    state.defer_frame_callbacks();
                                    state.held_buffers.push(buffer.clone());
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
            wl_surface::Request::Destroy => {}
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
