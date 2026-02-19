//! `wl_shm` and `wl_shm_pool` dispatch, plus SHM buffer infrastructure.

use std::os::unix::io::{AsRawFd, IntoRawFd};
use std::sync::Arc;

use parking_lot::Mutex;

use tracing::{debug, info, warn};
use wayland_server::protocol::{
    wl_shm::{self, WlShm},
    wl_shm_pool::{self, WlShmPool},
};
use wayland_server::{Client, DataInit, Dispatch, DisplayHandle, GlobalDispatch, New, WEnum};

use super::BufferData;
use crate::wayland::WaylandState;

// ─── SHM buffer infrastructure ─────────────────────────────────────

/// Shared mmap of a wl_shm_pool's file descriptor.
///
/// The pool can be resized, so we store the fd and re-mmap on resize.
/// Multiple `WlBuffer` objects reference the same pool.
pub struct ShmPoolData {
    /// Raw pointer to the mmap'd region.
    ptr: *mut u8,
    /// Current mapped size.
    size: usize,
    /// File descriptor (kept open for resize).
    fd: std::os::unix::io::RawFd,
}

// SAFETY: ShmPoolData is only accessed from the main thread.
unsafe impl Send for ShmPoolData {}

impl ShmPoolData {
    /// Create a new SHM pool mapping from an fd.
    pub(super) fn new(fd: std::os::unix::io::OwnedFd, size: usize) -> Option<Arc<Mutex<Self>>> {
        let raw_fd = fd.as_raw_fd();
        // SAFETY: Valid fd and size from the client's CreatePool request.
        let ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                size,
                libc::PROT_READ,
                libc::MAP_SHARED,
                raw_fd,
                0,
            )
        };
        if ptr == libc::MAP_FAILED {
            warn!("failed to mmap SHM pool");
            return None;
        }
        // Leak the OwnedFd so it stays open — we need it for resize.
        let leaked_fd = fd.into_raw_fd();
        Some(Arc::new(Mutex::new(Self {
            ptr: ptr as *mut u8,
            size,
            fd: leaked_fd,
        })))
    }

    /// Resize the pool mapping.
    fn resize(&mut self, new_size: usize) {
        if new_size <= self.size {
            return;
        }
        // SAFETY: Remapping a larger region from the same fd.
        unsafe {
            libc::munmap(self.ptr as *mut libc::c_void, self.size);
        }
        let ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                new_size,
                libc::PROT_READ,
                libc::MAP_SHARED,
                self.fd,
                0,
            )
        };
        if ptr != libc::MAP_FAILED {
            self.ptr = ptr as *mut u8;
            self.size = new_size;
        } else {
            warn!("failed to remap SHM pool");
        }
    }

    /// Read buffer pixels from this pool.
    ///
    /// Returns a copy of the pixel data.
    pub fn read_pixels(&self, offset: usize, len: usize) -> Option<Vec<u8>> {
        if offset + len > self.size {
            return None;
        }
        // SAFETY: offset + len is within the mapped region.
        let slice = unsafe { std::slice::from_raw_parts(self.ptr.add(offset), len) };
        Some(slice.to_vec())
    }
}

impl Drop for ShmPoolData {
    fn drop(&mut self) {
        // SAFETY: Valid mmap pointer and size.
        unsafe {
            libc::munmap(self.ptr as *mut libc::c_void, self.size);
            libc::close(self.fd);
        }
    }
}

/// Per-buffer metadata stored as user data on `WlBuffer`.
pub struct ShmBufferData {
    /// Reference to the parent pool's mmap.
    pub pool: Arc<Mutex<ShmPoolData>>,
    /// Byte offset into the pool.
    pub offset: i32,
    /// Buffer width in pixels.
    pub width: i32,
    /// Buffer height in pixels.
    pub height: i32,
    /// Stride in bytes.
    pub stride: i32,
    /// Pixel format.
    pub format: wl_shm::Format,
}

// ─── Dispatch impls ─────────────────────────────────────────────────

impl GlobalDispatch<WlShm, ()> for WaylandState {
    fn bind(
        _state: &mut Self,
        _dh: &DisplayHandle,
        _client: &Client,
        resource: New<WlShm>,
        _data: &(),
        data_init: &mut DataInit<'_, Self>,
    ) {
        let shm = data_init.init(resource, ());
        // Mandatory formats.
        shm.format(wl_shm::Format::Argb8888);
        shm.format(wl_shm::Format::Xrgb8888);
    }
}

impl Dispatch<WlShm, ()> for WaylandState {
    fn request(
        _state: &mut Self,
        _client: &Client,
        _shm: &WlShm,
        request: wl_shm::Request,
        _data: &(),
        _dh: &DisplayHandle,
        data_init: &mut DataInit<'_, Self>,
    ) {
        if let wl_shm::Request::CreatePool { id, fd, size } = request {
            info!(size, "wl_shm: creating pool");
            let pool_data = ShmPoolData::new(fd, size as usize);
            match pool_data {
                Some(pd) => {
                    data_init.init(id, pd);
                    debug!(size, "wl_shm: created pool");
                }
                None => {
                    warn!("wl_shm: failed to mmap pool, creating stub");
                    // Create with a dummy — will fail on buffer read.
                    data_init.init(
                        id,
                        Arc::new(Mutex::new(ShmPoolData {
                            ptr: std::ptr::null_mut(),
                            size: 0,
                            fd: -1,
                        })),
                    );
                }
            }
        }
    }
}

impl Dispatch<WlShmPool, Arc<Mutex<ShmPoolData>>> for WaylandState {
    fn request(
        _state: &mut Self,
        _client: &Client,
        _pool: &WlShmPool,
        request: wl_shm_pool::Request,
        data: &Arc<Mutex<ShmPoolData>>,
        _dh: &DisplayHandle,
        data_init: &mut DataInit<'_, Self>,
    ) {
        match request {
            wl_shm_pool::Request::CreateBuffer {
                id,
                offset,
                width,
                height,
                stride,
                format,
            } => {
                let fmt = match format {
                    WEnum::Value(f) => f,
                    _ => wl_shm::Format::Argb8888,
                };
                let buf_data = BufferData::Shm(ShmBufferData {
                    pool: data.clone(),
                    offset,
                    width,
                    height,
                    stride,
                    format: fmt,
                });
                data_init.init(id, buf_data);
                info!(width, height, stride, offset, "wl_shm_pool: created buffer");
            }
            wl_shm_pool::Request::Resize { size } => {
                let mut pool = data.lock();
                pool.resize(size as usize);
            }
            wl_shm_pool::Request::Destroy => {}
            _ => {}
        }
    }
}
