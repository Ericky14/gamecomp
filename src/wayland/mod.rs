//! Wayland server and protocol handling.
//!
//! Creates the Wayland display, binds a listening socket, and registers
//! protocol globals. Runs on the main thread as part of the calloop event loop.
//!
//! The server implements the minimal set of protocols needed for a single-app
//! fullscreen compositor:
//! - `wl_compositor` / `wl_surface` — surface management
//! - `wl_shm` — shared memory buffers
//! - `wl_seat` — input device multiplexing
//! - `wl_output` — display information
//! - `xdg_shell` — window management (xdg_toplevel for fullscreen)
//! - `wp_linux_dmabuf_v1` — zero-copy DMA-BUF buffer sharing
//! - `wp_presentation_time` — frame timing feedback

pub mod atoms;
pub mod protocols;
pub mod window_tracker;
pub mod xwayland;

use std::os::unix::io::RawFd;

use anyhow::Context;
use tracing::info;
use wayland_server::protocol::wl_callback::WlCallback;
use wayland_server::{Display, ListeningSocket};

use crate::backend::ConnectorInfo;
use crate::wayland::protocols::CommittedBuffer;

/// Per-client data stored with each Wayland client connection.
struct ClientData;

impl wayland_server::backend::ClientData for ClientData {
    fn initialized(&self, _client_id: wayland_server::backend::ClientId) {}
    fn disconnected(
        &self,
        _client_id: wayland_server::backend::ClientId,
        _reason: wayland_server::backend::DisconnectReason,
    ) {
        info!("Wayland client disconnected");
    }
}

/// State for the Wayland server.
///
/// Owned exclusively by the main thread. Never shared with render or XWM threads.
pub struct WaylandServer {
    /// The Wayland display.
    display: Display<WaylandState>,
    /// Listening socket for client connections.
    listener: ListeningSocket,
    /// Socket name (e.g., "wayland-1").
    socket_name: String,
    /// Whether the server is running.
    running: bool,
    /// Registered protocol globals.
    _globals: Option<protocols::Globals>,
}

/// Per-client state stored in the Wayland display.
pub struct WaylandState {
    /// Connected output information.
    pub outputs: Vec<ConnectorInfo>,
    /// Current pointer position.
    pub pointer_x: f64,
    pub pointer_y: f64,
    /// Currently focused surface (if any).
    pub focused_surface: Option<u32>,
    /// Frame sequence counter.
    pub frame_seq: u64,
    /// Output resolution for configure events.
    pub output_width: u32,
    pub output_height: u32,
    /// Serial counter for configure events.
    serial: u32,
    /// Pending frame callbacks to fire on next present.
    pub pending_frame_callbacks: Vec<WlCallback>,
    /// Deferred frame callbacks — withheld by the FPS limiter until it's
    /// time for the client to render the next frame. Moved here from
    /// `pending_frame_callbacks` on commit; fired by the main loop when
    /// the limiter allows.
    pub deferred_frame_callbacks: Vec<WlCallback>,
    /// Set to `true` on each `wl_surface.commit` that has a buffer.
    /// Cleared after deferred callbacks are fired.
    pub has_pending_commit: bool,
    /// Channel to send committed frames to the wayland backend for presentation.
    pub frame_channel: Option<std::sync::mpsc::Sender<CommittedBuffer>>,
    /// Staged committed buffer awaiting FPS-limited forwarding.
    ///
    /// On `wl_surface.commit`, the latest buffer is staged here instead of
    /// being sent directly to the render thread. The main loop forwards it
    /// when the FPS limiter allows. If the client commits faster than the
    /// target FPS, intermediate frames are dropped (overwritten) — only the
    /// most recent buffer is ever forwarded.
    pub staged_buffer: Option<CommittedBuffer>,
    /// Held `wl_buffer` objects that have NOT been released back to the client.
    ///
    /// By withholding `wl_buffer.release`, we prevent the client from
    /// recycling its buffer pool. Once all client-side buffers are held,
    /// the client blocks on `vkAcquireNextImage` (or equivalent). The main
    /// loop releases exactly **one** buffer per FPS tick via
    /// [`release_one_buffer`], keeping the client's frame count in
    /// lockstep with the compositor's display rate.
    pub held_buffers: Vec<wayland_server::protocol::wl_buffer::WlBuffer>,
    /// Host compositor's DMA-BUF format→modifier map. Populated by the wayland
    /// backend's event thread after connecting to the host. Used by the dmabuf
    /// module to advertise formats that allow zero-copy forwarding to the host.
    pub host_dmabuf_formats:
        std::sync::Arc<parking_lot::Mutex<std::collections::HashMap<u32, Vec<u64>>>>,
}

impl WaylandState {
    /// Create initial state with the given outputs.
    pub fn new(outputs: Vec<ConnectorInfo>, width: u32, height: u32) -> Self {
        Self {
            outputs,
            pointer_x: 0.0,
            pointer_y: 0.0,
            focused_surface: None,
            frame_seq: 0,
            output_width: width,
            output_height: height,
            serial: 1,
            pending_frame_callbacks: Vec::new(),
            deferred_frame_callbacks: Vec::new(),
            has_pending_commit: false,
            frame_channel: None,
            staged_buffer: None,
            held_buffers: Vec::new(),
            host_dmabuf_formats: std::sync::Arc::new(parking_lot::Mutex::new(
                std::collections::HashMap::new(),
            )),
        }
    }

    /// Get the output resolution.
    #[inline(always)]
    pub fn output_resolution(&self) -> (u32, u32) {
        (self.output_width, self.output_height)
    }

    /// Get the next serial number for protocol events.
    pub fn next_serial(&mut self) -> u32 {
        let s = self.serial;
        self.serial = self.serial.wrapping_add(1);
        s
    }

    /// Move pending frame callbacks to the deferred queue.
    ///
    /// Called on `wl_surface.commit`. The deferred callbacks will be
    /// released later by [`fire_deferred_callbacks`] when the FPS
    /// limiter allows.
    pub fn defer_frame_callbacks(&mut self) {
        self.deferred_frame_callbacks
            .append(&mut self.pending_frame_callbacks);
        self.has_pending_commit = true;
    }

    /// Fire all pending frame callbacks immediately (no FPS limiting).
    ///
    /// Use this when FPS limiting is disabled, or in code paths that
    /// should not be throttled (e.g., initial frame, cursor updates).
    pub fn fire_frame_callbacks(&mut self) {
        let now_ms = monotonic_ms();
        for cb in self.pending_frame_callbacks.drain(..) {
            cb.done(now_ms);
        }
    }

    /// Fire deferred frame callbacks. Called by the main loop when the
    /// FPS limiter says it's time to release.
    ///
    /// Returns `true` if any callbacks were actually fired.
    pub fn fire_deferred_callbacks(&mut self) -> bool {
        if self.deferred_frame_callbacks.is_empty() {
            return false;
        }
        let now_ms = monotonic_ms();
        for cb in self.deferred_frame_callbacks.drain(..) {
            cb.done(now_ms);
        }
        self.has_pending_commit = false;
        true
    }

    /// Release the oldest held `wl_buffer` back to the client.
    ///
    /// Called by the main loop on each FPS limiter tick.d    ///
    /// Returns `true` if a buffer was released.
    #[inline(always)]
    pub fn release_one_buffer(&mut self) -> bool {
        if self.held_buffers.is_empty() {
            return false;
        }
        // Release the oldest (FIFO order) so buffers cycle predictably.
        let buf = self.held_buffers.remove(0);
        buf.release();
        true
    }

    /// Release **all** held `wl_buffer` objects back to the client.
    ///
    /// Used only during shutdown / cleanup — NOT during normal frame
    /// pacing. For frame pacing, use [`release_one_buffer`].
    pub fn release_all_buffers(&mut self) {
        for buf in self.held_buffers.drain(..) {
            buf.release();
        }
    }
}

/// Get the current CLOCK_MONOTONIC time in nanoseconds.
#[inline(always)]
pub fn monotonic_ns() -> u64 {
    let mut ts = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    // SAFETY: Valid pointer to timespec.
    unsafe { libc::clock_gettime(libc::CLOCK_MONOTONIC, &mut ts) };
    ts.tv_sec as u64 * 1_000_000_000 + ts.tv_nsec as u64
}

/// Get the current CLOCK_MONOTONIC time as a `u32` millisecond timestamp
/// suitable for `wl_callback.done`.
#[inline(always)]
fn monotonic_ms() -> u32 {
    let ns = monotonic_ns();
    ((ns / 1_000_000) & 0xFFFF_FFFF) as u32
}

impl WaylandServer {
    /// Create a new Wayland server with protocol globals registered.
    pub fn new(
        _outputs: Vec<ConnectorInfo>,
        output_width: u32,
        output_height: u32,
    ) -> anyhow::Result<Self> {
        let display: Display<WaylandState> =
            Display::new().context("failed to create Wayland display")?;

        // Register protocol globals.
        let dh = display.handle();
        let globals = protocols::register_globals(&dh, output_width, output_height);

        // Bind listening socket.
        let listener =
            ListeningSocket::bind_auto("wayland", 0..33).context("failed to add Wayland socket")?;
        let socket_name = listener
            .socket_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_else(|| "wayland-0".to_string());

        info!(socket = %socket_name, "Wayland server listening");

        Ok(Self {
            display,
            listener,
            socket_name,
            running: false,
            _globals: Some(globals),
        })
    }

    /// Get the Wayland socket name for clients to connect to.
    pub fn socket_name(&self) -> &str {
        &self.socket_name
    }

    /// Get the Wayland display fd for polling.
    pub fn display_fd(&self) -> RawFd {
        // The display fd is used by calloop to know when clients send requests.
        use std::os::unix::io::{AsFd, AsRawFd};
        self.display.as_fd().as_raw_fd()
    }

    /// Dispatch pending Wayland events.
    ///
    /// Called by the calloop event loop when the display fd is readable.
    pub fn dispatch(&mut self, state: &mut WaylandState) -> anyhow::Result<()> {
        self.display
            .dispatch_clients(state)
            .context("failed to dispatch Wayland clients")?;
        Ok(())
    }

    /// Flush outgoing events to all clients.
    pub fn flush(&mut self) {
        self.display.flush_clients().ok();
    }

    /// Get the display handle (for registering globals).
    pub fn display_handle(&self) -> wayland_server::DisplayHandle {
        self.display.handle()
    }

    /// Mark the server as running.
    pub fn start(&mut self) {
        self.running = true;
        info!("Wayland server started");
    }

    /// Stop the server.
    pub fn stop(&mut self) {
        self.running = false;
        info!("Wayland server stopped");
    }

    /// Whether the server is currently running.
    pub fn is_running(&self) -> bool {
        self.running
    }

    /// Accept a pending client connection from the listening socket.
    ///
    /// Returns `Some(stream)` if a client was waiting, `None` otherwise.
    pub fn accept(&self) -> Option<std::os::unix::net::UnixStream> {
        match self.listener.accept() {
            Ok(opt) => opt,
            Err(e) => {
                tracing::warn!(?e, "failed to accept Wayland client");
                None
            }
        }
    }

    /// Insert an accepted client stream into the display.
    pub fn insert_client(
        &mut self,
        stream: std::os::unix::net::UnixStream,
        _state: &mut WaylandState,
    ) -> anyhow::Result<()> {
        let mut dh = self.display.handle();
        dh.insert_client(stream, std::sync::Arc::new(ClientData))
            .context("failed to insert Wayland client")?;
        info!("accepted new Wayland client");
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wayland_state_new() {
        let state = WaylandState::new(Vec::new(), 1920, 1080);
        assert!(state.outputs.is_empty());
        assert_eq!(state.pointer_x, 0.0);
        assert_eq!(state.pointer_y, 0.0);
        assert!(state.focused_surface.is_none());
        assert_eq!(state.frame_seq, 0);
    }

    #[test]
    fn wayland_state_pointer_position() {
        let mut state = WaylandState::new(Vec::new(), 1920, 1080);
        state.pointer_x = 500.0;
        state.pointer_y = 300.0;
        assert!((state.pointer_x - 500.0).abs() < f64::EPSILON);
        assert!((state.pointer_y - 300.0).abs() < f64::EPSILON);
    }

    #[test]
    fn wayland_state_focus() {
        let mut state = WaylandState::new(Vec::new(), 1920, 1080);
        assert!(state.focused_surface.is_none());
        state.focused_surface = Some(42);
        assert_eq!(state.focused_surface, Some(42));
    }

    #[test]
    fn wayland_server_creates_and_listens() {
        let server = WaylandServer::new(Vec::new(), 1920, 1080).unwrap();
        let socket = server.socket_name();
        assert!(socket.starts_with("wayland-"));
    }

    #[test]
    fn wayland_server_start_stop() {
        let mut server = WaylandServer::new(Vec::new(), 1920, 1080).unwrap();
        assert!(!server.is_running());
        server.start();
        assert!(server.is_running());
        server.stop();
        assert!(!server.is_running());
    }

    #[test]
    fn wayland_state_serial_increment() {
        let mut state = WaylandState::new(Vec::new(), 1920, 1080);
        let s1 = state.next_serial();
        let s2 = state.next_serial();
        assert_eq!(s1 + 1, s2);
    }

    #[test]
    fn wayland_state_output_resolution() {
        let state = WaylandState::new(Vec::new(), 2560, 1440);
        assert_eq!(state.output_resolution(), (2560, 1440));
    }
}
