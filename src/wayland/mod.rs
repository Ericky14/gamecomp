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
use wayland_server::protocol::wl_keyboard::WlKeyboard;
use wayland_server::protocol::wl_output::WlOutput;
use wayland_server::protocol::wl_pointer::{self, WlPointer};
use wayland_server::protocol::wl_surface::WlSurface;
use wayland_server::{Display, ListeningSocket, Resource};

use wayland_protocols::xdg::shell::server::xdg_toplevel::XdgToplevel;

use crate::backend::ConnectorInfo;
use crate::backend::wayland::CursorUpdate;
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
    /// Client's `wl_keyboard` objects — used to forward key events.
    pub keyboards: Vec<WlKeyboard>,
    /// Client's `wl_pointer` objects — used to forward pointer events.
    pub pointers: Vec<WlPointer>,
    /// All client surfaces — used for focus enter/leave events.
    /// Multiple clients (XWayland, Flutter) may each create a surface.
    pub client_surfaces: Vec<WlSurface>,
    /// Surfaces that have an xdg_toplevel role. Used preferentially for
    /// focus enter — cursor and subsurfaces are ignored.
    pub toplevel_surfaces: Vec<WlSurface>,
    /// Client IDs that have already received `wl_keyboard.enter`.
    pub keyboard_entered_clients: std::collections::HashSet<wayland_server::backend::ClientId>,
    /// Client IDs that have already received `wl_pointer.enter`.
    pub pointer_entered_clients: std::collections::HashSet<wayland_server::backend::ClientId>,
    /// Bound `wl_output` objects — used to send mode updates on resize.
    pub bound_outputs: Vec<WlOutput>,
    /// Active `xdg_toplevel` objects — used to re-configure on output resize.
    pub toplevels: Vec<XdgToplevel>,
    /// Channel to send cursor image updates to the host compositor thread.
    pub cursor_tx: Option<std::sync::mpsc::Sender<CursorUpdate>>,
    /// Steam integration mode. When true, only the focused window's
    /// surface is allowed to present (gated by `focused_wl_surface_id`).
    pub steam_mode: bool,
    /// Wayland protocol object ID of the focused window's surface. Written
    /// by XWM threads, read by the commit handler to gate presentation.
    /// 0 means no surface is focused (all commits rejected in steam mode).
    pub focused_wl_surface_id: std::sync::Arc<std::sync::atomic::AtomicU32>,
    /// XWayland server index of the focused surface. Used together with
    /// `focused_wl_surface_id` to uniquely identify the focused surface
    /// across multiple XWayland servers (protocol_id is per-client).
    pub focused_server_index: std::sync::Arc<std::sync::atomic::AtomicU32>,
    /// Maps Wayland ClientId → XWayland server index. Populated during
    /// XWayland spawn so the commit handler can determine which server
    /// a surface belongs to.
    pub xwayland_client_map: std::collections::HashMap<wayland_server::backend::ClientId, u32>,
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
            keyboards: Vec::new(),
            pointers: Vec::new(),
            client_surfaces: Vec::new(),
            toplevel_surfaces: Vec::new(),
            keyboard_entered_clients: std::collections::HashSet::new(),
            pointer_entered_clients: std::collections::HashSet::new(),
            bound_outputs: Vec::new(),
            toplevels: Vec::new(),
            cursor_tx: None,
            steam_mode: false,
            focused_wl_surface_id: std::sync::Arc::new(std::sync::atomic::AtomicU32::new(0)),
            focused_server_index: std::sync::Arc::new(std::sync::atomic::AtomicU32::new(u32::MAX)),
            xwayland_client_map: std::collections::HashMap::new(),
        }
    }

    /// Get the output resolution.
    #[inline(always)]
    pub fn output_resolution(&self) -> (u32, u32) {
        (self.output_width, self.output_height)
    }

    /// Update the advertised output resolution and notify bound clients.
    ///
    /// Sends updated `wl_output.mode` + `wl_output.done` to all bound
    /// output objects so clients re-configure at the new size.
    pub fn update_output_resolution(&mut self, width: u32, height: u32) {
        if width == self.output_width && height == self.output_height {
            return;
        }
        info!(
            old_w = self.output_width,
            old_h = self.output_height,
            new_w = width,
            new_h = height,
            "updating Wayland output resolution"
        );
        self.output_width = width;
        self.output_height = height;

        // Notify all bound wl_output objects.
        self.bound_outputs.retain(|o| o.is_alive());
        for output in &self.bound_outputs {
            output.mode(
                wayland_server::protocol::wl_output::Mode::Current
                    | wayland_server::protocol::wl_output::Mode::Preferred,
                width as i32,
                height as i32,
                60_000,
            );
            output.done();
        }

        // Re-configure all toplevels so clients resize their buffers.
        self.toplevels.retain(|t| t.is_alive());
        let states = protocols::xdg_shell::activated_fullscreen_states();
        let serial = self.next_serial();
        for toplevel in &self.toplevels {
            toplevel.configure(width as i32, height as i32, states.clone());
            if let Some(td) = toplevel.data::<protocols::xdg_shell::XdgToplevelData>() {
                td.xdg_surface.configure(serial);
            }
        }
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

    /// Send keyboard and pointer enter events to each client's own surface.
    ///
    /// Called every main loop iteration. For each keyboard/pointer, finds the
    /// surface belonging to the same Wayland client and sends enter if not
    /// already sent. Prefers toplevel surfaces (xdg_toplevel role) so that
    /// native Wayland clients like Flutter/GTK receive enter on the correct
    /// window surface rather than a cursor or subsurface.
    pub fn send_focus_enter(&mut self) {
        if self.client_surfaces.is_empty() && self.toplevel_surfaces.is_empty() {
            return;
        }

        // Clean up dead surfaces.
        self.client_surfaces.retain(|s| s.is_alive());
        self.toplevel_surfaces.retain(|s| s.is_alive());

        // Helper: find the best surface for a client. Prefer toplevel
        // surfaces; fall back to client_surfaces (bare surfaces from
        // XWayland that never create an xdg_toplevel).
        let find_surface =
            |cid: &wayland_server::backend::ClientId, toplevel: &[WlSurface]| -> Option<usize> {
                // First try toplevel_surfaces.
                for (i, s) in toplevel.iter().enumerate() {
                    if s.client().map(|c| c.id()).as_ref() == Some(cid) {
                        // Return a sentinel: index >= toplevel.len() not needed,
                        // we'll use a tag to distinguish.
                        return Some(i);
                    }
                }
                None
            };

        // ── Keyboard enters ────────────────────────────────────────
        // (keyboard_index, surface_ref) where surface_ref encodes
        // which list the surface is in: true=toplevel, false=client.
        let mut kb_enters: Vec<(usize, usize, bool)> = Vec::new();
        for (ki, kb) in self.keyboards.iter().enumerate() {
            if !kb.is_alive() {
                continue;
            }
            let Some(kb_client) = kb.client() else {
                continue;
            };
            let kb_cid = kb_client.id();
            if self.keyboard_entered_clients.contains(&kb_cid) {
                continue;
            }
            if let Some(si) = find_surface(&kb_cid, &self.toplevel_surfaces) {
                kb_enters.push((ki, si, true));
            } else {
                // Fall back to client_surfaces (XWayland).
                for (si, surface) in self.client_surfaces.iter().enumerate() {
                    if surface.client().map(|c| c.id()) == Some(kb_cid.clone()) {
                        kb_enters.push((ki, si, false));
                        break;
                    }
                }
            }
        }

        for (ki, si, is_toplevel) in &kb_enters {
            let serial = self.next_serial();
            let surface = if *is_toplevel {
                &self.toplevel_surfaces[*si]
            } else {
                &self.client_surfaces[*si]
            };
            let kb = &self.keyboards[*ki];
            kb.enter(serial, surface, vec![]);
            if let Some(c) = kb.client() {
                self.keyboard_entered_clients.insert(c.id());
            }
        }

        // ── Pointer enters ─────────────────────────────────────────
        let mut ptr_enters: Vec<(usize, usize, bool)> = Vec::new();
        for (pi, ptr) in self.pointers.iter().enumerate() {
            if !ptr.is_alive() {
                continue;
            }
            let Some(ptr_client) = ptr.client() else {
                continue;
            };
            let ptr_cid = ptr_client.id();
            if self.pointer_entered_clients.contains(&ptr_cid) {
                continue;
            }
            if let Some(si) = find_surface(&ptr_cid, &self.toplevel_surfaces) {
                ptr_enters.push((pi, si, true));
            } else {
                for (si, surface) in self.client_surfaces.iter().enumerate() {
                    if surface.client().map(|c| c.id()) == Some(ptr_cid.clone()) {
                        ptr_enters.push((pi, si, false));
                        break;
                    }
                }
            }
        }

        let cx = (self.output_width as f64) / 2.0;
        let cy = (self.output_height as f64) / 2.0;
        for (pi, si, is_toplevel) in &ptr_enters {
            let serial = self.next_serial();
            let surface = if *is_toplevel {
                &self.toplevel_surfaces[*si]
            } else {
                &self.client_surfaces[*si]
            };
            let ptr = &self.pointers[*pi];
            ptr.enter(serial, surface, cx, cy);
            ptr.frame();
            if let Some(c) = ptr.client() {
                self.pointer_entered_clients.insert(c.id());
            }
        }
    }

    /// Forward a keyboard key event to the focused client.
    ///
    /// `key` is the raw Linux evdev keycode. The Wayland `wl_keyboard.key`
    /// event sends evdev keycodes directly — the XKB keymap handles the
    /// evdev→keysym translation on the client side.
    pub fn send_key(&mut self, key: u32, pressed: bool, time_ms: u32) {
        let serial = self.next_serial();
        let state = if pressed {
            wayland_server::protocol::wl_keyboard::KeyState::Pressed
        } else {
            wayland_server::protocol::wl_keyboard::KeyState::Released
        };
        for kb in &self.keyboards {
            if kb.is_alive() {
                kb.key(serial, time_ms, key, state);
            }
        }
    }

    /// Forward keyboard modifier state to the focused client.
    ///
    /// Must be sent after keymap and after key events that change
    /// modifier state (Shift, Ctrl, Alt, etc.).
    pub fn send_modifiers(
        &mut self,
        mods_depressed: u32,
        mods_latched: u32,
        mods_locked: u32,
        group: u32,
    ) {
        let serial = self.next_serial();
        for kb in &self.keyboards {
            if kb.is_alive() {
                kb.modifiers(serial, mods_depressed, mods_latched, mods_locked, group);
            }
        }
    }

    /// Forward an XKB keymap to the focused client.
    ///
    /// Sends `wl_keyboard.keymap` with the given format, fd, and size.
    /// Used in nested mode to forward the host compositor's keymap
    /// instead of a hardcoded one.
    pub fn send_keymap(&mut self, format: u32, fd: std::os::unix::io::OwnedFd, size: u32) {
        use std::os::unix::io::AsFd;
        use wayland_server::protocol::wl_keyboard::KeymapFormat;
        let fmt = if format == 1 {
            KeymapFormat::XkbV1
        } else {
            KeymapFormat::NoKeymap
        };
        for kb in &self.keyboards {
            if kb.is_alive() {
                kb.keymap(fmt, fd.as_fd(), size);
            }
        }
    }

    /// Forward a pointer motion event to the focused client.
    ///
    /// Accumulates relative deltas (DRM mode evdev). Tracks position in
    /// output space, clamped to `output_width × output_height`.
    pub fn send_pointer_motion(&mut self, dx: f64, dy: f64, time_ms: u32) {
        self.pointer_x = (self.pointer_x + dx).clamp(0.0, self.output_width as f64 - 1.0);
        self.pointer_y = (self.pointer_y + dy).clamp(0.0, self.output_height as f64 - 1.0);
        for ptr in &self.pointers {
            if ptr.is_alive() {
                ptr.motion(time_ms, self.pointer_x, self.pointer_y);
                ptr.frame();
            }
        }
    }

    /// Forward an absolute pointer position to the focused client.
    ///
    /// Used in nested mode where the host backend has already mapped
    /// host surface-local coordinates to client buffer coordinates.
    /// Coordinates are clamped to `output_width × output_height`.
    pub fn send_pointer_motion_absolute(&mut self, x: f64, y: f64, time_ms: u32) {
        let w = self.output_width.max(1) as f64;
        let h = self.output_height.max(1) as f64;
        self.pointer_x = x.clamp(0.0, w - 1.0);
        self.pointer_y = y.clamp(0.0, h - 1.0);
        for ptr in &self.pointers {
            if ptr.is_alive() {
                ptr.motion(time_ms, self.pointer_x, self.pointer_y);
                ptr.frame();
            }
        }
    }

    /// Forward a pointer button event to the focused client.
    pub fn send_pointer_button(&mut self, button: u32, pressed: bool, time_ms: u32) {
        let serial = self.next_serial();
        let state = if pressed {
            wl_pointer::ButtonState::Pressed
        } else {
            wl_pointer::ButtonState::Released
        };
        for ptr in &self.pointers {
            if ptr.is_alive() {
                ptr.button(serial, time_ms, button, state);
                ptr.frame();
            }
        }
    }

    /// Forward a scroll event to the focused client.
    pub fn send_pointer_axis(&mut self, dx: f64, dy: f64, time_ms: u32) {
        for ptr in &self.pointers {
            if ptr.is_alive() {
                if dy.abs() > f64::EPSILON {
                    ptr.axis(time_ms, wl_pointer::Axis::VerticalScroll, dy);
                }
                if dx.abs() > f64::EPSILON {
                    ptr.axis(time_ms, wl_pointer::Axis::HorizontalScroll, dx);
                }
                ptr.frame();
            }
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
    ///
    /// Returns the `ClientId` of the newly inserted client so the caller
    /// can associate it with an XWayland server index.
    pub fn insert_client(
        &mut self,
        stream: std::os::unix::net::UnixStream,
        _state: &mut WaylandState,
    ) -> anyhow::Result<wayland_server::backend::ClientId> {
        let mut dh = self.display.handle();
        let client = dh
            .insert_client(stream, std::sync::Arc::new(ClientData))
            .context("failed to insert Wayland client")?;
        let client_id = client.id();
        info!("accepted new Wayland client");
        Ok(client_id)
    }
}

#[cfg(test)]
#[path = "mod_tests.rs"]
mod tests;
