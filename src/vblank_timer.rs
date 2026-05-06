//! Free-running VBlank timer using `timerfd`.
//!
//! Unlike DRM page-flip events (which only fire when a frame is presented),
//! this timer fires at the display refresh rate regardless of frame
//! production. This provides a stable clock for frame callback delivery,
//! matching gamescope's `CVBlankTimer` architecture.
//!
//! The timer is pollable via its raw fd, so it integrates with `ppoll`.

use std::os::unix::io::{AsRawFd, RawFd};

use tracing::{debug, warn};

/// A timerfd-based periodic VBlank timer.
pub struct VBlankTimer {
    fd: RawFd,
    /// Interval in nanoseconds between ticks.
    interval_ns: u64,
}

impl VBlankTimer {
    /// Create a new VBlank timer. Does NOT start ticking until [`arm`] is called.
    pub fn new() -> Option<Self> {
        // SAFETY: Creating a timerfd with CLOCK_MONOTONIC. The fd is owned
        // by this struct and closed on drop.
        let fd =
            unsafe { libc::timerfd_create(libc::CLOCK_MONOTONIC, libc::TFD_NONBLOCK | libc::TFD_CLOEXEC) };
        if fd < 0 {
            warn!("failed to create timerfd for VBlank timer");
            return None;
        }
        Some(Self { fd, interval_ns: 0 })
    }

    /// Arm the timer to fire periodically at the given refresh rate (Hz).
    pub fn arm(&mut self, refresh_hz: u32) {
        if refresh_hz == 0 {
            return;
        }
        let interval_ns = 1_000_000_000u64 / refresh_hz as u64;
        self.interval_ns = interval_ns;
        let sec = interval_ns / 1_000_000_000;
        let nsec = interval_ns % 1_000_000_000;

        let spec = libc::itimerspec {
            it_interval: libc::timespec {
                tv_sec: sec as i64,
                tv_nsec: nsec as i64,
            },
            it_value: libc::timespec {
                tv_sec: sec as i64,
                tv_nsec: nsec as i64,
            },
        };

        // SAFETY: fd is a valid timerfd, spec is on the stack, null oldvalue.
        let ret = unsafe { libc::timerfd_settime(self.fd, 0, &spec, std::ptr::null_mut()) };
        if ret < 0 {
            warn!("failed to arm timerfd");
        } else {
            debug!(
                refresh_hz,
                interval_ms = interval_ns as f64 / 1_000_000.0,
                "VBlank timer armed"
            );
        }
    }

    /// Read and acknowledge pending timer expirations.
    ///
    /// Returns the number of elapsed ticks since the last read.
    /// Returns 0 if no ticks pending (non-blocking).
    #[inline(always)]
    pub fn read_ticks(&self) -> u64 {
        let mut buf: u64 = 0;
        // SAFETY: fd is a valid timerfd, buf is 8 bytes on the stack.
        let ret = unsafe {
            libc::read(
                self.fd,
                &mut buf as *mut u64 as *mut libc::c_void,
                std::mem::size_of::<u64>(),
            )
        };
        if ret == std::mem::size_of::<u64>() as isize {
            buf
        } else {
            0
        }
    }

    /// Get the raw fd for use in `ppoll`/`epoll`.
    #[inline(always)]
    pub fn raw_fd(&self) -> RawFd {
        self.fd
    }

    /// Get the interval in nanoseconds.
    #[inline(always)]
    pub fn interval_ns(&self) -> u64 {
        self.interval_ns
    }
}

impl AsRawFd for VBlankTimer {
    fn as_raw_fd(&self) -> RawFd {
        self.fd
    }
}

impl Drop for VBlankTimer {
    fn drop(&mut self) {
        // SAFETY: fd is a valid timerfd owned by this struct.
        unsafe {
            libc::close(self.fd);
        }
    }
}
