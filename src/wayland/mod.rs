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
use tracing::{debug, info};
use wayland_server::protocol::wl_callback::WlCallback;
use wayland_server::protocol::wl_keyboard::WlKeyboard;
use wayland_server::protocol::wl_output::WlOutput;
use wayland_server::protocol::wl_pointer::{self, WlPointer};
use wayland_server::protocol::wl_surface::WlSurface;
use wayland_server::{Display, ListeningSocket, Resource};

use wayland_protocols::wp::pointer_constraints::zv1::server::zwp_locked_pointer_v1::ZwpLockedPointerV1;
use wayland_protocols::wp::relative_pointer::zv1::server::zwp_relative_pointer_v1::ZwpRelativePointerV1;
use wayland_protocols::xdg::shell::server::xdg_toplevel::XdgToplevel;

use crate::backend::ConnectorInfo;
use crate::backend::wayland::CursorUpdate;
use crate::wayland::protocols::{CommittedBuffer, CommittedFrame};

/// Per-client data stored with each Wayland client connection.
struct ClientData;

impl wayland_server::backend::ClientData for ClientData {
    fn initialized(&self, _client_id: wayland_server::backend::ClientId) {}
    fn disconnected(
        &self,
        _client_id: wayland_server::backend::ClientId,
        _reason: wayland_server::backend::DisconnectReason,
    ) {
        debug!("Wayland client disconnected");
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
    /// Output refresh rate in mHz (e.g. 144000 for 144Hz). Advertised
    /// to clients via `wl_output.mode`. Updated when the DRM/host display
    /// refresh rate is detected.
    pub output_refresh_mhz: u32,
    /// Serial counter for configure events.
    serial: u32,
    /// Per-surface pending frame callbacks.
    ///
    /// Key: `(wl_surface protocol_id, server_index)`.
    /// Callbacks are pushed here on `wl_surface.frame()`.
    /// Moved to `surface_deferred_callbacks` on commit for App surfaces
    /// (FPS-limited path), or fired immediately for overlays / cursors /
    /// rejected commits.
    pub surface_callbacks: std::collections::HashMap<(u32, u32), Vec<WlCallback>>,
    /// Per-surface deferred frame callbacks.
    ///
    /// Key: `(wl_surface protocol_id, server_index)`.
    /// Moved here from `surface_callbacks` on App commit; fired by the
    /// main loop on each vblank tick alongside buffer releases. This
    /// matches gamescope's `receivedDoneCommit` + `unlockedForFrameCallback`
    /// gate: callbacks fire on the first vblank after commit, AT THE SAME
    /// TIME as `wl_buffer.release`. The synchronous delivery is critical —
    /// XWayland's Present extension uses both events together to calibrate
    /// its MSC timing, and desynchronizing them causes 250ms hiccups on
    /// idle→active transitions.
    pub surface_deferred_callbacks: std::collections::HashMap<(u32, u32), Vec<WlCallback>>,
    /// Channel to send committed frames to the wayland backend for presentation.
    pub frame_channel: Option<std::sync::mpsc::Sender<CommittedFrame>>,
    /// Staged committed buffer awaiting FPS-limited forwarding.
    ///
    /// On `wl_surface.commit`, the latest buffer is staged here instead of
    /// being sent directly to the render thread. The main loop forwards it
    /// when the FPS limiter allows. If the client commits faster than the
    /// target FPS, intermediate frames are dropped (overwritten) — only the
    /// most recent buffer is ever forwarded.
    pub staged_buffer: Option<CommittedBuffer>,
    /// Server index of the client that staged `staged_buffer`.
    /// Used by the main loop to detect when the newly-focused client
    /// has committed its first frame (commit-based presentation switch).
    pub staged_buffer_server_index: u32,
    /// Held `wl_buffer` objects that have NOT been released back to the client.
    ///
    /// By withholding `wl_buffer.release`, we prevent the client from
    /// recycling its buffer pool. Once all client-side buffers are held,
    /// the client blocks on `vkAcquireNextImage` (or equivalent). The main
    /// loop releases exactly **one** buffer per FPS tick via
    /// [`release_one_buffer`], keeping the client's frame count in
    /// lockstep with the compositor's display rate.
    pub held_buffers: Vec<wayland_server::protocol::wl_buffer::WlBuffer>,
    /// Presentation feedback callbacks attached to the current `staged_buffer`.
    /// Moved from `SurfaceData.pending_feedbacks` on commit. If a newer commit
    /// replaces the staged buffer before forwarding, these are `discarded()`.
    pub staged_feedbacks: Vec<
        wayland_protocols::wp::presentation_time::server::wp_presentation_feedback::WpPresentationFeedback,
    >,
    /// Presentation feedback callbacks for the buffer currently in the
    /// display pipeline (forwarded to render thread, awaiting page flip).
    /// Sent `presented(...)` when the page flip completes.
    pub inflight_feedbacks: Vec<
        wayland_protocols::wp::presentation_time::server::wp_presentation_feedback::WpPresentationFeedback,
    >,
    /// Monotonic sequence counter for presentation events (MSC).
    pub presentation_sequence: u64,
    /// Number of `wp_presentation.feedback` requests received (ever).
    /// Observable indicator that a client actually uses the protocol.
    pub presentation_requests: u64,
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
    /// Map of currently-entered keyboard `(client, resource id)` →
    /// `(surface client, surface id)` it was entered on.
    ///
    /// Wayland protocol IDs are *per-client*, so the key must include the
    /// client to avoid collisions when a disconnected client's IDs are
    /// reused by a new client.
    pub entered_keyboards: std::collections::HashMap<
        (wayland_server::backend::ClientId, u32),
        (wayland_server::backend::ClientId, u32),
    >,
    pub entered_pointers: std::collections::HashMap<
        (wayland_server::backend::ClientId, u32),
        (wayland_server::backend::ClientId, u32),
    >,
    /// Bound `wl_output` objects — used to send mode updates on resize.
    pub bound_outputs: Vec<WlOutput>,
    /// Active `xdg_toplevel` objects — used to re-configure on output resize.
    pub toplevels: Vec<XdgToplevel>,
    /// Channel to send cursor image updates to the host compositor thread.
    pub cursor_tx: Option<std::sync::mpsc::Sender<CursorUpdate>>,
    /// Wayland protocol object ID of the focused window's surface. Written
    /// by XWM threads, read by the commit handler to gate presentation.
    /// 0 means no surface is focused (all commits rejected).
    pub focused_wl_surface_id: std::sync::Arc<std::sync::atomic::AtomicU32>,
    /// XWayland server index of the focused surface. Used together with
    /// `focused_wl_surface_id` to uniquely identify the focused surface
    /// across multiple XWayland servers (protocol_id is per-client).
    pub focused_server_index: std::sync::Arc<std::sync::atomic::AtomicU32>,
    /// Overlay surface protocol ID. Written by XWM when overlay focus
    /// changes. The commit handler accepts commits from this surface
    /// alongside the focused app surface.
    pub overlay_wl_surface_id: std::sync::Arc<std::sync::atomic::AtomicU32>,
    /// Overlay server index.
    pub overlay_server_index: std::sync::Arc<std::sync::atomic::AtomicU32>,
    /// External overlay surface protocol ID.
    pub external_overlay_wl_surface_id: std::sync::Arc<std::sync::atomic::AtomicU32>,
    /// External overlay server index.
    pub external_overlay_server_index: std::sync::Arc<std::sync::atomic::AtomicU32>,
    /// Steam overlay input focus mode (from `STEAM_INPUT_FOCUS`).
    /// 0=none, 1=overlay grabs all input, 2=overlay grabs pointer only
    /// (keyboard stays with the app — gamescope-compatible).
    pub overlay_input_focus_mode: std::sync::Arc<std::sync::atomic::AtomicU32>,
    /// External overlay input focus mode (same semantics as above).
    pub external_overlay_input_focus_mode: std::sync::Arc<std::sync::atomic::AtomicU32>,
    /// Staged overlay buffer (Steam overlay). Forwarded alongside
    /// the app buffer when compositing.
    pub staged_overlay_buffer: Option<CommittedBuffer>,
    /// Staged external overlay buffer (MangoHud, etc.).
    pub staged_external_overlay_buffer: Option<CommittedBuffer>,
    /// Overlay opacity (0.0–1.0, from _NET_WM_OPACITY).
    pub overlay_opacity: std::sync::Arc<parking_lot::Mutex<f32>>,
    /// External overlay opacity (0.0–1.0).
    pub external_overlay_opacity: std::sync::Arc<parking_lot::Mutex<f32>>,
    /// Maps Wayland ClientId → XWayland server index. Populated during
    /// XWayland spawn so the commit handler can determine which server
    /// a surface belongs to.
    pub xwayland_client_map: std::collections::HashMap<wayland_server::backend::ClientId, u32>,
    /// Focused app ID for native Wayland clients (non-XWayland).
    /// Set when a non-XWayland client creates an xdg_toplevel.
    /// Uses 769 (STEAMAPP_PLAYTRON_APP_ID) by default for Grid compatibility.
    pub native_focused_app_id: std::sync::Arc<std::sync::atomic::AtomicU32>,
    /// Focused surface protocol ID for native Wayland clients.
    /// Set to the wl_surface protocol_id when a non-XWayland toplevel is created.
    pub native_focused_surface_id: std::sync::Arc<std::sync::atomic::AtomicU32>,
    /// Client's `zwp_relative_pointer_v1` objects — used to forward raw
    /// motion deltas even while the pointer is locked.
    pub relative_pointers: Vec<ZwpRelativePointerV1>,
    /// Whether the pointer is currently locked by a client via
    /// `zwp_pointer_constraints_v1.lock_pointer`. While locked,
    /// `wl_pointer.motion` events are suppressed and only
    /// `relative_motion` events are emitted.
    pub pointer_locked: bool,
    /// The active locked pointer object (if any). Stored so we can send
    /// `unlocked` when the lock is released.
    pub locked_pointer: Option<ZwpLockedPointerV1>,
    /// Whether the user has moved the mouse since startup. Like gamescope,
    /// the cursor stays hidden until the first real pointer motion event.
    pub cursor_user_moved: bool,
    /// DRM direct scanout mode. When true, the commit handler skips
    /// synchronous DMA-BUF fence waits — the render thread handles
    /// implicit sync asynchronously via `poll_dmabuf_ready()`.
    pub drm_mode: bool,
    /// DRM syncobj device for explicit sync operations.
    /// Set in DRM mode when the kernel supports `syncobj_eventfd`.
    pub syncobj_device: Option<protocols::SyncobjDevice>,
    /// Map of `(surface_id, server_index)` → `WpLinuxDrmSyncobjSurfaceV1`.
    /// Keyed by both IDs because different XWayland servers can allocate
    /// the same `wl_surface` protocol ID.
    pub syncobj_surfaces: std::collections::HashMap<
        (u32, u32),
        wayland_protocols::wp::linux_drm_syncobj::v1::server::wp_linux_drm_syncobj_surface_v1::WpLinuxDrmSyncobjSurfaceV1,
    >,
    /// Monotonic timestamp (ns) of the last deferred-callback fire.
    /// Used for diagnosing frame pacing: the delta from this to the
    /// next commit arrival reveals client render time + IPC latency.
    pub last_callback_fire_ns: u64,
    /// Monotonic timestamp (ns) when the current `staged_buffer` was set.
    /// Measures how long a frame sits in the staging slot before forwarding.
    pub staged_at_ns: u64,
    /// Counter for total commits accepted (for stats).
    pub commit_count: u64,
}

impl WaylandState {
    /// Create initial state with the given outputs.
    pub fn new(outputs: Vec<ConnectorInfo>, width: u32, height: u32) -> Self {
        Self {
            outputs,
            pointer_x: width as f64 / 2.0,
            pointer_y: height as f64 / 2.0,
            focused_surface: None,
            frame_seq: 0,
            output_width: width,
            output_height: height,
            output_refresh_mhz: 60_000,
            serial: 1,
            surface_callbacks: std::collections::HashMap::new(),
            surface_deferred_callbacks: std::collections::HashMap::new(),
            frame_channel: None,
            staged_buffer: None,
            staged_buffer_server_index: u32::MAX,
            held_buffers: Vec::new(),
            staged_feedbacks: Vec::new(),
            inflight_feedbacks: Vec::new(),
            presentation_sequence: 0,
            presentation_requests: 0,
            host_dmabuf_formats: std::sync::Arc::new(parking_lot::Mutex::new(
                std::collections::HashMap::new(),
            )),
            keyboards: Vec::new(),
            pointers: Vec::new(),
            client_surfaces: Vec::new(),
            toplevel_surfaces: Vec::new(),
            entered_keyboards: std::collections::HashMap::new(),
            entered_pointers: std::collections::HashMap::new(),
            bound_outputs: Vec::new(),
            toplevels: Vec::new(),
            cursor_tx: None,
            focused_wl_surface_id: std::sync::Arc::new(std::sync::atomic::AtomicU32::new(0)),
            focused_server_index: std::sync::Arc::new(std::sync::atomic::AtomicU32::new(u32::MAX)),
            overlay_wl_surface_id: std::sync::Arc::new(std::sync::atomic::AtomicU32::new(0)),
            overlay_server_index: std::sync::Arc::new(std::sync::atomic::AtomicU32::new(u32::MAX)),
            external_overlay_wl_surface_id: std::sync::Arc::new(std::sync::atomic::AtomicU32::new(
                0,
            )),
            external_overlay_server_index: std::sync::Arc::new(std::sync::atomic::AtomicU32::new(
                u32::MAX,
            )),
            overlay_input_focus_mode: std::sync::Arc::new(std::sync::atomic::AtomicU32::new(0)),
            external_overlay_input_focus_mode: std::sync::Arc::new(
                std::sync::atomic::AtomicU32::new(0),
            ),
            staged_overlay_buffer: None,
            staged_external_overlay_buffer: None,
            overlay_opacity: std::sync::Arc::new(parking_lot::Mutex::new(0.0)),
            external_overlay_opacity: std::sync::Arc::new(parking_lot::Mutex::new(0.0)),
            xwayland_client_map: std::collections::HashMap::new(),
            native_focused_app_id: std::sync::Arc::new(std::sync::atomic::AtomicU32::new(0)),
            native_focused_surface_id: std::sync::Arc::new(std::sync::atomic::AtomicU32::new(0)),
            relative_pointers: Vec::new(),
            pointer_locked: false,
            locked_pointer: None,
            cursor_user_moved: false,
            drm_mode: false,
            syncobj_device: None,
            syncobj_surfaces: std::collections::HashMap::new(),
            last_callback_fire_ns: 0,
            staged_at_ns: 0,
            commit_count: 0,
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

        // Rescale pointer position proportionally so it stays at the
        // same relative location (e.g., center stays center).
        if self.output_width > 0 && self.output_height > 0 {
            self.pointer_x = self.pointer_x * width as f64 / self.output_width as f64;
            self.pointer_y = self.pointer_y * height as f64 / self.output_height as f64;
        }

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
                self.output_refresh_mhz as i32,
            );
            output.done();
        }

        // Re-configure all toplevels so clients resize their buffers.
        self.reconfigure_toplevels();
    }

    /// Update the advertised output refresh rate and re-send `wl_output.mode`
    /// to all bound outputs so clients update their internal frame timing
    /// (e.g. Flutter syncs to the new vsync interval).
    pub fn update_output_refresh(&mut self, refresh_mhz: u32) {
        if refresh_mhz == 0 || refresh_mhz == self.output_refresh_mhz {
            return;
        }
        info!(
            old_mhz = self.output_refresh_mhz,
            new_mhz = refresh_mhz,
            "updating advertised wl_output refresh rate"
        );
        self.output_refresh_mhz = refresh_mhz;
        self.bound_outputs.retain(|o| o.is_alive());
        for output in &self.bound_outputs {
            output.mode(
                wayland_server::protocol::wl_output::Mode::Current
                    | wayland_server::protocol::wl_output::Mode::Preferred,
                self.output_width as i32,
                self.output_height as i32,
                refresh_mhz as i32,
            );
            output.done();
        }
    }

    /// Re-configure all toplevels (Activated + Fullscreen).
    ///
    /// Called when focus changes so the newly-focused client receives a
    /// `configure` event and responds with a `commit`, restarting its
    /// frame callback cycle. Without this, a client that lost focus and
    /// had its frame callbacks drained by cursor commits would never
    /// learn it should start rendering again.
    pub fn reconfigure_toplevels(&mut self) {
        self.toplevels.retain(|t| t.is_alive());
        let states = protocols::xdg_shell::activated_fullscreen_states();
        let serial = self.next_serial();
        for toplevel in &self.toplevels {
            toplevel.configure(
                self.output_width as i32,
                self.output_height as i32,
                states.clone(),
            );
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

    /// Push a frame callback for a specific surface.
    ///
    /// Called from `wl_surface::Request::Frame`. Keyed by
    /// `(protocol_id, server_index)` so each surface's callbacks are
    /// isolated — matching gamescope's per-window callback model.
    pub fn push_surface_callback(&mut self, surface_id: u32, server_index: u32, cb: WlCallback) {
        self.surface_callbacks
            .entry((surface_id, server_index))
            .or_default()
            .push(cb);
    }

    /// Move a surface's pending callbacks to the deferred queue.
    ///
    /// Called on `wl_surface.commit` for the focused App surface.
    /// Deferred callbacks are released on the next vblank tick by
    /// [`fire_all_deferred_callbacks`].
    pub fn defer_surface_callbacks(&mut self, surface_id: u32, server_index: u32) {
        let key = (surface_id, server_index);
        if let Some(mut pending) = self.surface_callbacks.remove(&key) {
            self.surface_deferred_callbacks
                .entry(key)
                .or_default()
                .append(&mut pending);
        }
    }

    /// Fire a single surface's pending callbacks immediately.
    ///
    /// Used for overlays, cursors, and rejected commits — surfaces that
    /// should not be FPS-limited.
    pub fn fire_surface_pending(&mut self, surface_id: u32, server_index: u32) {
        if let Some(cbs) = self.surface_callbacks.remove(&(surface_id, server_index)) {
            let now_ms = monotonic_ms();
            for cb in cbs {
                cb.done(now_ms);
            }
        }
    }

    /// Fire all frame callbacks (pending and deferred) for ALL surfaces,
    /// and release all held buffers.
    ///
    /// Used on VT resume, host focus regain, and other recovery paths
    /// to unblock all clients.
    pub fn fire_all_callbacks(&mut self) {
        let now_ms = monotonic_ms();
        for (_, cbs) in self.surface_callbacks.drain() {
            for cb in cbs {
                cb.done(now_ms);
            }
        }
        for (_, cbs) in self.surface_deferred_callbacks.drain() {
            for cb in cbs {
                cb.done(now_ms);
            }
        }
        // Release held buffers so the client can reuse them immediately.
        for buf in self.held_buffers.drain(..) {
            buf.release();
        }
    }

    /// Fire all pending and deferred callbacks for ALL surfaces, then
    /// release stale held buffers.
    ///
    /// Called on each vblank tick. Matches gamescope's `paint_all` where
    /// both `flush_frame_done` (callbacks) and buffer unlocks happen in
    /// the same vblank iteration. Delivering `wl_callback.done` and
    /// `wl_buffer.release` together is critical — XWayland's Present
    /// extension uses both events to calibrate its MSC timing.
    /// Desynchronizing them (e.g., releasing buffers immediately on
    /// commit but deferring callbacks to vblank) causes 250ms hiccups
    /// when Flutter resumes from idle.
    ///
    /// Returns `true` if any callbacks were fired.
    pub fn fire_all_surface_callbacks(&mut self) -> bool {
        let has_any = !self.surface_callbacks.is_empty()
            || !self.surface_deferred_callbacks.is_empty();
        if !has_any {
            return false;
        }
        let now_ms = monotonic_ms();
        for (_, cbs) in self.surface_callbacks.drain() {
            for cb in cbs {
                cb.done(now_ms);
            }
        }
        for (_, cbs) in self.surface_deferred_callbacks.drain() {
            for cb in cbs {
                cb.done(now_ms);
            }
        }

        // Release stale held buffers AT THE SAME TIME as callbacks.
        // Gamescope releases old buffers (via commit_t destructor) during
        // paint_all, synchronized with frame callback delivery. Releasing
        // buffers here instead of in forward_staged_frame() ensures
        // wl_buffer.release and wl_callback.done arrive in the same
        // Wayland flush, keeping XWayland's Present timing coherent.
        self.release_stale_buffers();

        true
    }

    /// Fire all pending + deferred callbacks for a specific server.
    ///
    /// Used when waking a server (e.g., overlay activation wakes
    /// server 0) without disturbing other servers' callback rhythm.
    pub fn fire_server_callbacks(&mut self, server_index: u32) {
        let now_ms = monotonic_ms();
        let pending_keys: Vec<_> = self
            .surface_callbacks
            .keys()
            .filter(|k| k.1 == server_index)
            .copied()
            .collect();
        for key in pending_keys {
            if let Some(cbs) = self.surface_callbacks.remove(&key) {
                for cb in cbs {
                    cb.done(now_ms);
                }
            }
        }
        let deferred_keys: Vec<_> = self
            .surface_deferred_callbacks
            .keys()
            .filter(|k| k.1 == server_index)
            .copied()
            .collect();
        for key in deferred_keys {
            if let Some(cbs) = self.surface_deferred_callbacks.remove(&key) {
                for cb in cbs {
                    cb.done(now_ms);
                }
            }
        }
    }

    /// Total number of pending callbacks across all surfaces.
    #[must_use]
    pub fn pending_callback_count(&self) -> usize {
        self.surface_callbacks.values().map(Vec::len).sum()
    }

    /// Total number of deferred callbacks across all surfaces.
    #[must_use]
    pub fn deferred_callback_count(&self) -> usize {
        self.surface_deferred_callbacks.values().map(Vec::len).sum()
    }

    /// Whether any surface has pending or deferred callbacks.
    #[must_use]
    pub fn has_surface_callbacks(&self) -> bool {
        self.surface_callbacks.values().any(|v| !v.is_empty())
            || self.surface_deferred_callbacks.values().any(|v| !v.is_empty())
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

    /// Release every held `wl_buffer` *except* the two most recently
    /// committed ones.
    ///
    /// Called when forwarding a staged frame: any older held buffers must
    /// have come from commits the render thread silently coalesced (it
    /// drains the channel keeping only the latest frame). Without this,
    /// those wl_buffers stay held forever — the client runs out of pool
    /// slots and frame rate drops.
    ///
    /// We keep TWO buffers because in the DRM direct-scanout path, the
    /// client's DMA-BUF *is* the display scanout buffer. When a new
    /// frame is committed, the previous frame is still being scanned out
    /// until the new frame's page flip completes. Keeping two ensures the
    /// active scanout buffer is never released prematurely.
    ///
    /// Returns the number of buffers released.
    pub fn release_stale_buffers(&mut self) -> usize {
        if self.held_buffers.len() <= 2 {
            return 0;
        }
        // Keep the two newest (last two pushed).
        let keep_new = self.held_buffers.pop();
        let keep_prev = self.held_buffers.pop();
        let count = self.held_buffers.len();
        for buf in self.held_buffers.drain(..) {
            buf.release();
        }
        if let Some(buf) = keep_prev {
            self.held_buffers.push(buf);
        }
        if let Some(buf) = keep_new {
            self.held_buffers.push(buf);
        }
        count
    }

    /// Compute the client that should currently receive keyboard focus.
    ///
    /// Compute the (client, surface_id) currently entitled to keyboard focus.
    ///
    /// Priority:
    ///   external_overlay (mode > 0 && mode != 2)
    ///     > steam_overlay (mode > 0 && mode != 2)
    ///     > focused app
    ///
    /// Mode 2 means "pointer only" — keyboard stays with the app.
    fn target_keyboard_focus(&self) -> Option<(wayland_server::backend::ClientId, u32)> {
        use std::sync::atomic::Ordering::Relaxed;
        let ext_id = self.external_overlay_wl_surface_id.load(Relaxed);
        let ext_srv = self.external_overlay_server_index.load(Relaxed);
        let ext_mode = self.external_overlay_input_focus_mode.load(Relaxed);
        let ovl_id = self.overlay_wl_surface_id.load(Relaxed);
        let ovl_srv = self.overlay_server_index.load(Relaxed);
        let ovl_mode = self.overlay_input_focus_mode.load(Relaxed);
        let app_srv = self.focused_server_index.load(Relaxed);
        let app_id = self.focused_wl_surface_id.load(Relaxed);

        if ext_id != 0 && ext_mode > 0 && ext_mode != 2 {
            return self
                .find_xwayland_client_by_server(ext_srv)
                .map(|c| (c, ext_id));
        }
        if ovl_id != 0 && ovl_mode > 0 && ovl_mode != 2 {
            return self
                .find_xwayland_client_by_server(ovl_srv)
                .map(|c| (c, ovl_id));
        }
        if app_id != 0 {
            return self
                .find_xwayland_client_by_server(app_srv)
                .map(|c| (c, app_id));
        }
        None
    }

    /// Compute the (client, surface_id) currently entitled to pointer focus.
    /// Same as keyboard but mode 2 still grabs pointer.
    fn target_pointer_focus(&self) -> Option<(wayland_server::backend::ClientId, u32)> {
        use std::sync::atomic::Ordering::Relaxed;
        let ext_id = self.external_overlay_wl_surface_id.load(Relaxed);
        let ext_srv = self.external_overlay_server_index.load(Relaxed);
        let ext_mode = self.external_overlay_input_focus_mode.load(Relaxed);
        let ovl_id = self.overlay_wl_surface_id.load(Relaxed);
        let ovl_srv = self.overlay_server_index.load(Relaxed);
        let ovl_mode = self.overlay_input_focus_mode.load(Relaxed);
        let app_srv = self.focused_server_index.load(Relaxed);
        let app_id = self.focused_wl_surface_id.load(Relaxed);

        if ext_id != 0 && ext_mode > 0 {
            return self
                .find_xwayland_client_by_server(ext_srv)
                .map(|c| (c, ext_id));
        }
        if ovl_id != 0 && ovl_mode > 0 {
            return self
                .find_xwayland_client_by_server(ovl_srv)
                .map(|c| (c, ovl_id));
        }
        if app_id != 0 {
            return self
                .find_xwayland_client_by_server(app_srv)
                .map(|c| (c, app_id));
        }
        None
    }

    /// Reverse-lookup: find the XWayland client owning the given server index.
    fn find_xwayland_client_by_server(
        &self,
        server_idx: u32,
    ) -> Option<wayland_server::backend::ClientId> {
        if server_idx == u32::MAX {
            return None;
        }
        self.xwayland_client_map
            .iter()
            .find(|(_, v)| **v == server_idx)
            .map(|(k, _)| k.clone())
    }

    /// Find a specific surface by protocol ID belonging to the given client.
    ///
    /// Critical: clients (especially XWayland) create many surfaces — one
    /// per X11 window, plus cursor surfaces and DnD icons. Picking any
    /// surface for the client is wrong: cursor surfaces have role `cursor`
    /// and sending `wl_pointer.enter` on them produces broken behavior
    /// (default X-shaped cursor, no input). The XWM publishes the actual
    /// focused window's `wl_surface.protocol_id()` via `focused_wl_surface_id`,
    /// and we must use exactly that.
    fn find_surface_by_id(
        &self,
        cid: &wayland_server::backend::ClientId,
        surface_id: u32,
    ) -> Option<WlSurface> {
        for s in self
            .toplevel_surfaces
            .iter()
            .chain(self.client_surfaces.iter())
        {
            if s.is_alive()
                && s.id().protocol_id() == surface_id
                && s.client().map(|c| c.id()).as_ref() == Some(cid)
            {
                return Some(s.clone());
            }
        }
        None
    }

    /// Update keyboard and pointer focus to match the current focus state.
    ///
    /// Called every main loop iteration. Per-resource sweep: any
    /// pointer/keyboard belonging to the focused client that hasn't yet
    /// received `enter` gets one (provided we can find a focus surface);
    /// any resource currently entered whose client is no longer focused
    /// receives `leave`. Per-resource tracking ensures new resources
    /// created by the focused client (or after a surface becomes
    /// available) are picked up automatically.
    // ClientId carries an `Arc<AtomicBool>` aliveness flag (interior
    // mutability), but its identity for hashing/equality is the underlying
    // wl_client pointer — stable for the lifetime we care about.
    #[allow(clippy::mutable_key_type)]
    pub fn update_input_focus(&mut self) {
        self.client_surfaces.retain(|s| s.is_alive());
        self.toplevel_surfaces.retain(|s| s.is_alive());
        self.keyboards.retain(|k| k.is_alive());
        self.pointers.retain(|p| p.is_alive());

        // Prune entries whose resource no longer exists. Wayland protocol
        // IDs are recycled per client, so stale entries would collide with
        // freshly-created resources sharing the same id.
        let live_kb_keys: std::collections::HashSet<(wayland_server::backend::ClientId, u32)> =
            self.keyboards
                .iter()
                .filter_map(|k| k.client().map(|c| (c.id(), k.id().protocol_id())))
                .collect();
        self.entered_keyboards
            .retain(|k, _| live_kb_keys.contains(k));
        let live_ptr_keys: std::collections::HashSet<(wayland_server::backend::ClientId, u32)> =
            self.pointers
                .iter()
                .filter_map(|p| p.client().map(|c| (c.id(), p.id().protocol_id())))
                .collect();
        self.entered_pointers
            .retain(|k, _| live_ptr_keys.contains(k));

        let target_kb = self.target_keyboard_focus();
        let target_ptr = self.target_pointer_focus();

        // ── Keyboard sweep ──────────────────────────────────────────
        // Compute desired enter/leave; apply enters last so the new resource
        // sees the correct (current) target.
        let mut kb_leaves: Vec<(usize, wayland_server::backend::ClientId, u32)> = Vec::new();
        let mut kb_enters: Vec<usize> = Vec::new();
        for (i, kb) in self.keyboards.iter().enumerate() {
            let kb_cid = match kb.client() {
                Some(c) => c.id(),
                None => continue,
            };
            let key = (kb_cid.clone(), kb.id().protocol_id());
            let is_target = target_kb.as_ref().is_some_and(|(c, _)| c == &kb_cid);
            match self.entered_keyboards.get(&key) {
                Some((entered_cid, entered_sid)) => {
                    let target_sid = target_kb.as_ref().map(|(_, s)| *s);
                    if !is_target || target_sid != Some(*entered_sid) {
                        kb_leaves.push((i, entered_cid.clone(), *entered_sid));
                    }
                }
                None => {
                    if is_target {
                        kb_enters.push(i);
                    }
                }
            }
        }
        for (i, cid, sid) in kb_leaves {
            if let Some(surf) = self.find_surface_by_id(&cid, sid) {
                let serial = self.next_serial();
                self.keyboards[i].leave(serial, &surf);
            }
            if let Some(c) = self.keyboards[i].client() {
                let key = (c.id(), self.keyboards[i].id().protocol_id());
                self.entered_keyboards.remove(&key);
            }
        }
        if let Some((target_cid, target_sid)) = target_kb.clone()
            && let Some(surf) = self.find_surface_by_id(&target_cid, target_sid)
        {
            for i in kb_enters {
                // Only the focus target client can have its keyboards entered.
                let Some(c) = self.keyboards[i].client() else {
                    continue;
                };
                if c.id() != target_cid {
                    continue;
                }
                let key = (c.id(), self.keyboards[i].id().protocol_id());
                let serial = self.next_serial();
                self.keyboards[i].enter(serial, &surf, vec![]);
                self.entered_keyboards
                    .insert(key, (target_cid.clone(), target_sid));
            }
        }

        // ── Pointer sweep ──────────────────────────────────────────
        let mut ptr_leaves: Vec<(usize, wayland_server::backend::ClientId, u32)> = Vec::new();
        let mut ptr_enters: Vec<usize> = Vec::new();
        for (i, ptr) in self.pointers.iter().enumerate() {
            let ptr_cid = match ptr.client() {
                Some(c) => c.id(),
                None => continue,
            };
            let key = (ptr_cid.clone(), ptr.id().protocol_id());
            let is_target = target_ptr.as_ref().is_some_and(|(c, _)| c == &ptr_cid);
            match self.entered_pointers.get(&key) {
                Some((entered_cid, entered_sid)) => {
                    let target_sid = target_ptr.as_ref().map(|(_, s)| *s);
                    if !is_target || target_sid != Some(*entered_sid) {
                        ptr_leaves.push((i, entered_cid.clone(), *entered_sid));
                    }
                }
                None => {
                    if is_target {
                        ptr_enters.push(i);
                    }
                }
            }
        }
        for (i, cid, sid) in ptr_leaves {
            if let Some(surf) = self.find_surface_by_id(&cid, sid) {
                let serial = self.next_serial();
                self.pointers[i].leave(serial, &surf);
                self.pointers[i].frame();
            }
            if let Some(c) = self.pointers[i].client() {
                let key = (c.id(), self.pointers[i].id().protocol_id());
                self.entered_pointers.remove(&key);
            }
        }
        let cx = (self.output_width as f64) / 2.0;
        let cy = (self.output_height as f64) / 2.0;
        if let Some((target_cid, target_sid)) = target_ptr.clone()
            && let Some(surf) = self.find_surface_by_id(&target_cid, target_sid)
        {
            for i in ptr_enters {
                let Some(c) = self.pointers[i].client() else {
                    continue;
                };
                if c.id() != target_cid {
                    continue;
                }
                let key = (c.id(), self.pointers[i].id().protocol_id());
                let serial = self.next_serial();
                self.pointers[i].enter(serial, &surf, cx, cy);
                self.pointers[i].frame();
                self.entered_pointers
                    .insert(key, (target_cid.clone(), target_sid));
            }
        }
    }

    /// Whether the given keyboard has currently been sent `enter`.
    #[inline]
    fn keyboard_is_focused(&self, kb: &WlKeyboard) -> bool {
        let Some(cid) = kb.client().map(|c| c.id()) else {
            return false;
        };
        self.entered_keyboards
            .contains_key(&(cid, kb.id().protocol_id()))
    }

    /// Whether the given pointer has currently been sent `enter`.
    #[inline]
    fn pointer_is_focused(&self, ptr: &WlPointer) -> bool {
        let Some(cid) = ptr.client().map(|c| c.id()) else {
            return false;
        };
        self.entered_pointers
            .contains_key(&(cid, ptr.id().protocol_id()))
    }

    /// Forward a keyboard key event to the focused client.
    ///
    /// `key` is the raw Linux evdev keycode. The Wayland `wl_keyboard.key`
    /// event sends evdev keycodes directly — the XKB keymap handles the
    /// evdev→keysym translation on the client side.
    pub fn send_key(&mut self, key: u32, pressed: bool, time_ms: u32) {
        if self.entered_keyboards.is_empty() {
            return;
        }
        let serial = self.next_serial();
        let state = if pressed {
            wayland_server::protocol::wl_keyboard::KeyState::Pressed
        } else {
            wayland_server::protocol::wl_keyboard::KeyState::Released
        };
        for kb in &self.keyboards {
            if kb.is_alive() && self.keyboard_is_focused(kb) {
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
        if self.entered_keyboards.is_empty() {
            return;
        }
        let serial = self.next_serial();
        for kb in &self.keyboards {
            if kb.is_alive() && self.keyboard_is_focused(kb) {
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
    ///
    /// When the pointer is locked, `wl_pointer.motion` is suppressed but
    /// `zwp_relative_pointer_v1.relative_motion` is still emitted so the
    /// client receives raw deltas (essential for FPS camera controls).
    pub fn send_pointer_motion(&mut self, dx: f64, dy: f64, time_ms: u32) {
        self.cursor_user_moved = true;
        // Always send relative motion (unaffected by lock).
        self.send_relative_motion(dx, dy, time_ms);

        if self.pointer_locked {
            // Protocol spec: no wl_pointer.motion while locked.
            return;
        }

        self.pointer_x = (self.pointer_x + dx).clamp(0.0, self.output_width as f64 - 1.0);
        self.pointer_y = (self.pointer_y + dy).clamp(0.0, self.output_height as f64 - 1.0);
        if self.entered_pointers.is_empty() {
            return;
        }
        for ptr in &self.pointers {
            if ptr.is_alive() && self.pointer_is_focused(ptr) {
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
        self.cursor_user_moved = true;
        let w = self.output_width.max(1) as f64;
        let h = self.output_height.max(1) as f64;

        let old_x = self.pointer_x;
        let old_y = self.pointer_y;
        self.pointer_x = x.clamp(0.0, w - 1.0);
        self.pointer_y = y.clamp(0.0, h - 1.0);

        // Emit relative motion from the delta of absolute positions.
        let rdx = self.pointer_x - old_x;
        let rdy = self.pointer_y - old_y;
        if rdx.abs() > f64::EPSILON || rdy.abs() > f64::EPSILON {
            self.send_relative_motion(rdx, rdy, time_ms);
        }

        if self.pointer_locked {
            return;
        }

        if self.entered_pointers.is_empty() {
            return;
        }
        for ptr in &self.pointers {
            if ptr.is_alive() && self.pointer_is_focused(ptr) {
                ptr.motion(time_ms, self.pointer_x, self.pointer_y);
                ptr.frame();
            }
        }
    }

    /// Forward a pointer button event to the focused client.
    pub fn send_pointer_button(&mut self, button: u32, pressed: bool, time_ms: u32) {
        if self.entered_pointers.is_empty() {
            return;
        }
        let serial = self.next_serial();
        let state = if pressed {
            wl_pointer::ButtonState::Pressed
        } else {
            wl_pointer::ButtonState::Released
        };
        for ptr in &self.pointers {
            if ptr.is_alive() && self.pointer_is_focused(ptr) {
                ptr.button(serial, time_ms, button, state);
                ptr.frame();
            }
        }
    }

    /// Forward a scroll event to the focused client.
    pub fn send_pointer_axis(&mut self, dx: f64, dy: f64, time_ms: u32) {
        if self.entered_pointers.is_empty() {
            return;
        }
        for ptr in &self.pointers {
            if ptr.is_alive() && self.pointer_is_focused(ptr) {
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

    /// Send `relative_motion` to all bound `zwp_relative_pointer_v1` objects.
    ///
    /// These events carry raw (unaccelerated) dx/dy deltas and are emitted
    /// regardless of pointer lock state. `time_ms` is split into the
    /// u64-microsecond hi/lo halves expected by the protocol.
    fn send_relative_motion(&mut self, dx: f64, dy: f64, time_ms: u32) {
        if self.relative_pointers.is_empty() {
            return;
        }
        let time_us = time_ms as u64 * 1000;
        let hi = (time_us >> 32) as u32;
        let lo = time_us as u32;
        self.relative_pointers.retain(|rp| rp.is_alive());
        for rp in &self.relative_pointers {
            rp.relative_motion(hi, lo, dx, dy, dx, dy);
        }
    }

    /// Activate pointer lock — hide hardware cursor and suppress
    /// `wl_pointer.motion` events.
    pub fn lock_pointer(&mut self, locked_ptr: ZwpLockedPointerV1) {
        debug!("pointer lock activated");
        self.pointer_locked = true;

        // Send `locked` event to notify the client.
        locked_ptr.locked();
        self.locked_pointer = Some(locked_ptr);

        // Hide the hardware cursor.
        if let Some(ref tx) = self.cursor_tx {
            let _ = tx.send(CursorUpdate::Hide);
        }
    }

    /// Deactivate pointer lock — restore cursor visibility.
    pub fn unlock_pointer(&mut self, resource: &ZwpLockedPointerV1) {
        if self.locked_pointer.as_ref() == Some(resource) {
            debug!("pointer lock deactivated");
            resource.unlocked();
            self.locked_pointer = None;
            self.pointer_locked = false;
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
        let globals = protocols::register_globals(&dh, output_width, output_height, None);

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

    /// Register the `wp_linux_drm_syncobj_manager_v1` global after DRM init.
    ///
    /// Must be called before XWayland starts so the global is visible.
    pub fn register_syncobj_global(&self, device: protocols::SyncobjDevice) {
        let dh = self.display.handle();
        use wayland_protocols::wp::linux_drm_syncobj::v1::server::wp_linux_drm_syncobj_manager_v1::WpLinuxDrmSyncobjManagerV1;
        dh.create_global::<WaylandState, WpLinuxDrmSyncobjManagerV1, protocols::SyncobjGlobalData>(
            1,
            protocols::SyncobjGlobalData {
                device: Some(device),
            },
        );
        info!("wp_linux_drm_syncobj_manager_v1: registered (late)");
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
        debug!("accepted new Wayland client");
        Ok(client_id)
    }
}

#[cfg(test)]
#[path = "mod_tests.rs"]
mod tests;
