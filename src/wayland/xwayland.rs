//! XWayland integration and X11 window management.
//!
//! Launches the XWayland server process, connects to it via x11rb for
//! X11 window management, and handles window classification, focus
//! determination, and runtime atom-based control.
//!
//! The XWM runs on a dedicated thread (`gamecomp-xwm`) and communicates
//! with the main thread via `calloop::channel`. It sends window lifecycle
//! and focus events, and receives compositor commands.
//!
//! # Atom-based runtime control
//!
//! External controllers (Steam, scripts, MangoHud) set properties on the
//! root window to control compositor behavior at runtime. The XWM watches
//! for `PropertyNotify` events and translates them into `XwmEvent`s.
//!
//! # Multi-display
//!
//! Each XWayland server instance gets a unique `display_id`. The compositor
//! can spawn additional instances via the `GAMECOMP_CREATE_XWAYLAND_SERVER`
//! atom or `XwmCommand::CreateXWaylandServer`.

use std::process::Child;

use anyhow::Context;
use tracing::{debug, error, info};
use x11rb::connection::Connection;
use x11rb::wrapper::ConnectionExt as WrapperConnectionExt;

use super::atoms::Atoms;
use super::window_tracker::{FocusState, WindowRole, WindowTracker};

/// X11 atom identifier (re-exported for convenience).
type X11Atom = u32;

/// Events sent from the XWM thread to the main thread.
#[derive(Debug)]
pub enum XwmEvent {
    /// A window has been mapped (made visible).
    WindowMapped {
        /// X11 window ID.
        window_id: u32,
        /// Window title.
        title: String,
        /// Requested size (may be overridden by compositor).
        width: u32,
        height: u32,
        /// Window role classification.
        role: WindowRole,
        /// Steam AppID (0 if not a game).
        app_id: u32,
    },
    /// A window has been unmapped (hidden or destroyed).
    WindowUnmapped { window_id: u32 },
    /// A window has been destroyed.
    WindowDestroyed { window_id: u32 },
    /// A window requests fullscreen mode.
    WindowFullscreen { window_id: u32 },
    /// A window's title has changed.
    TitleChanged { window_id: u32, title: String },
    /// Focus has changed — includes the full focus state for the compositor.
    FocusChanged(FocusState),

    // --- Atom-driven runtime control events ---
    /// FPS limit changed via `GAMECOMP_FPS_LIMIT` atom.
    FpsLimitChanged(u32),
    /// VRR enable/disable changed via `GAMECOMP_VRR_ENABLED` atom.
    VrrChanged(bool),
    /// Low-latency mode changed via `GAMECOMP_LOW_LATENCY` atom.
    LowLatencyChanged(bool),
    /// Allow-tearing changed via `GAMECOMP_ALLOW_TEARING` atom.
    AllowTearingChanged(bool),
    /// Scaling filter changed via `GAMECOMP_SCALING_FILTER` atom.
    ScalingFilterChanged(u32),
    /// FSR sharpness changed via `GAMECOMP_FSR_SHARPNESS` atom.
    FsrSharpnessChanged(u32),
    /// HDR enable/disable changed via `GAMECOMP_HDR_ENABLED` atom.
    HdrChanged(bool),
    /// Force composition mode (no direct scanout) via `GAMECOMP_COMPOSITE_FORCE`.
    CompositeForceChanged(bool),
    /// Screenshot requested via `GAMECOMP_REQUEST_SCREENSHOT` atom.
    ScreenshotRequested(u32),
    /// Request to create a new XWayland server via atom.
    CreateXWaylandServerRequested(u32),
    /// Request to destroy an XWayland server via atom.
    DestroyXWaylandServerRequested(u32),
}

/// Commands sent from the main thread to the XWM thread.
#[derive(Debug)]
pub enum XwmCommand {
    /// Set the output resolution for XWayland.
    SetResolution { width: u32, height: u32 },
    /// Close a window.
    CloseWindow { window_id: u32 },
    /// Focus a specific window by ID.
    FocusWindow { window_id: u32 },
    /// Focus a window by Steam AppID.
    FocusAppId { app_id: u32 },
    /// Update a feedback atom value on the root window.
    SetFeedback { atom_name: FeedbackAtom, value: u32 },
    /// Shutdown the XWM.
    Shutdown,
}

/// Feedback atoms that the compositor can write back to the root window.
#[derive(Debug, Clone, Copy)]
pub enum FeedbackAtom {
    VrrCapable,
    VrrInUse,
    HdrSupported,
    FsrActive,
    DisplayRefreshRate,
    CursorVisible,
}

/// XWayland server state.
///
/// Manages the XWayland child process and X11 window manager connection.
pub struct XWaylandServer {
    /// XWayland child process.
    process: Option<Child>,
    /// X11 display number (e.g., ":1").
    display_number: i32,
    /// Whether XWayland is ready (has sent the SIGUSR1 signal).
    ready: bool,
}

impl XWaylandServer {
    /// Create a new XWayland server (not yet started).
    pub fn new() -> Self {
        Self {
            process: None,
            display_number: -1,
            ready: false,
        }
    }

    /// Get the X11 display string (e.g., ":1").
    pub fn display_string(&self) -> String {
        format!(":{}", self.display_number)
    }

    /// Whether XWayland is ready to accept connections.
    pub fn is_ready(&self) -> bool {
        self.ready
    }
}

impl Drop for XWaylandServer {
    fn drop(&mut self) {
        if let Some(mut child) = self.process.take() {
            info!("shutting down XWayland");
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

/// Run the X11 window manager event loop.
///
/// This function runs on the `gamecomp-xwm` thread. It connects to XWayland
/// via x11rb, interns all atoms, sets up SubstructureRedirect and
/// PropertyChangeMask for window management and atom-based control, and
/// processes X11 events in a loop.
///
/// # Arguments
///
/// * `display` — X11 display string (e.g., ":1")
/// * `event_tx` — Channel sender for XWM events to the main thread
/// * `cmd_rx` — Channel receiver for commands from the main thread
pub fn run_xwm(
    display: &str,
    event_tx: &calloop::channel::Sender<XwmEvent>,
    cmd_rx: &std::sync::mpsc::Receiver<XwmCommand>,
    output_width: u32,
    output_height: u32,
) -> anyhow::Result<()> {
    let x11_display = display;
    info!(
        x11_display,
        output_width, output_height, "starting X11 window manager"
    );

    // Connect to XWayland.
    let (conn, screen_num) =
        x11rb::connect(Some(display)).context("failed to connect to XWayland")?;

    let screen = &conn.setup().roots[screen_num];
    let root = screen.root;

    info!(root, screen_num, "connected to XWayland");

    // Intern all atoms in a single batched round-trip.
    let atoms = Atoms::intern(&conn).context("failed to intern X11 atoms")?;

    // Initialize window tracker.
    let mut tracker = WindowTracker::new();

    // Register as window manager (SubstructureRedirect on root).
    // Also subscribe to PropertyChangeMask so we receive PropertyNotify
    // events when external controllers set atoms on the root window.
    use x11rb::protocol::xproto::{ChangeWindowAttributesAux, ConnectionExt, EventMask};
    conn.change_window_attributes(
        root,
        &ChangeWindowAttributesAux::new().event_mask(
            EventMask::SUBSTRUCTURE_REDIRECT
                | EventMask::SUBSTRUCTURE_NOTIFY
                | EventMask::PROPERTY_CHANGE,
        ),
    )
    .context("failed to set SubstructureRedirect -- is another WM running?")?
    .check()
    .context("another window manager is already running")?;

    conn.flush().context("failed to flush X11 connection")?;

    // Enable Composite redirect so XWayland creates Wayland surfaces for
    // each X11 window. Without this, XWayland renders internally but never
    // submits frames to our Wayland server.
    use x11rb::protocol::composite::{ConnectionExt as CompositeExt, Redirect};
    conn.composite_redirect_subwindows(root, Redirect::MANUAL)
        .context("failed to redirect subwindows")?
        .check()
        .context("Composite extension not available")?;

    // Publish our PID on the root window for identification.
    publish_pid(&conn, root, &atoms);

    info!("XWM registered as window manager with atom support");

    // Event loop.
    while let ControlFlow::Continue = process_command(cmd_rx, &conn, root, &atoms, &mut tracker) {
        // --- Poll for X11 events (non-blocking) ---
        match conn.poll_for_event() {
            Ok(Some(event)) => {
                handle_x11_event(&conn, &atoms, &mut tracker, event_tx, root, event);
            }
            Ok(None) => {
                // No events pending. Brief sleep to avoid busy-waiting.
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
            Err(e) => {
                error!(?e, "X11 connection error");
                return Err(e.into());
            }
        }

        // --- Re-evaluate focus if dirty ---
        if tracker.is_focus_dirty() && tracker.determine_focus() {
            publish_focus_feedback(&conn, root, &atoms, &tracker, event_tx);
        }
    }

    info!("XWM event loop exited");
    Ok(())
}

/// Loop control flow returned by command processing.
enum ControlFlow {
    Continue,
    Break,
}

/// Publish the compositor PID on the root window for identification.
fn publish_pid<C: Connection>(conn: &C, root: u32, atoms: &Atoms) {
    use x11rb::protocol::xproto::{AtomEnum, PropMode};
    let pid = std::process::id();
    let _ = conn.change_property32(
        PropMode::REPLACE,
        root,
        atoms.pid,
        AtomEnum::CARDINAL,
        &[pid],
    );
    let _ = conn.flush();
}

/// Process one pending command from the main thread.
///
/// Returns `ControlFlow::Break` if the event loop should exit.
fn process_command<C: Connection>(
    cmd_rx: &std::sync::mpsc::Receiver<XwmCommand>,
    conn: &C,
    root: u32,
    atoms: &Atoms,
    tracker: &mut WindowTracker,
) -> ControlFlow {
    use x11rb::protocol::xproto::{AtomEnum, PropMode};
    match cmd_rx.try_recv() {
        Ok(XwmCommand::Shutdown) => {
            info!("XWM received shutdown command");
            return ControlFlow::Break;
        }
        Ok(XwmCommand::SetResolution { width, height }) => {
            debug!(width, height, "XWM: resolution change requested");
            // TODO: Use RANDR to resize the root window.
        }
        Ok(XwmCommand::CloseWindow { window_id }) => {
            debug!(window_id, "XWM: close window requested");
            // TODO: Send WM_DELETE_WINDOW or XDestroyWindow.
        }
        Ok(XwmCommand::FocusWindow { window_id }) => {
            debug!(window_id, "XWM: focus window requested");
            tracker.set_requested_window(Some(window_id));
        }
        Ok(XwmCommand::FocusAppId { app_id }) => {
            debug!(app_id, "XWM: focus AppID requested");
            tracker.set_requested_app_ids(vec![app_id]);
        }
        Ok(XwmCommand::SetFeedback { atom_name, value }) => {
            let atom_id = match atom_name {
                FeedbackAtom::VrrCapable => atoms.vrr_capable,
                FeedbackAtom::VrrInUse => atoms.vrr_in_use,
                FeedbackAtom::HdrSupported => atoms.hdr_supported,
                FeedbackAtom::FsrActive => atoms.fsr_active,
                FeedbackAtom::DisplayRefreshRate => atoms.display_refresh_rate,
                FeedbackAtom::CursorVisible => atoms.cursor_visible,
            };
            let _ = conn.change_property32(
                PropMode::REPLACE,
                root,
                atom_id,
                AtomEnum::CARDINAL,
                &[value],
            );
            let _ = conn.flush();
        }
        Err(std::sync::mpsc::TryRecvError::Empty) => {}
        Err(std::sync::mpsc::TryRecvError::Disconnected) => {
            info!("XWM command channel disconnected, shutting down");
            return ControlFlow::Break;
        }
    }
    ControlFlow::Continue
}

/// Handle a single X11 event — map, unmap, destroy, configure, or property change.
fn handle_x11_event<C: Connection>(
    conn: &C,
    atoms: &Atoms,
    tracker: &mut WindowTracker,
    event_tx: &calloop::channel::Sender<XwmEvent>,
    root: u32,
    event: x11rb::protocol::Event,
) {
    use x11rb::protocol::xproto::{
        ChangeWindowAttributesAux, ConfigureWindowAux, ConnectionExt, EventMask,
    };
    match event {
        x11rb::protocol::Event::MapRequest(e) => {
            debug!(window = e.window, "MapRequest");

            // Subscribe to property changes on this window so we
            // detect when the client sets classification atoms.
            let _ = conn.change_window_attributes(
                e.window,
                &ChangeWindowAttributesAux::new().event_mask(
                    EventMask::PROPERTY_CHANGE
                        | EventMask::SUBSTRUCTURE_NOTIFY
                        | EventMask::FOCUS_CHANGE,
                ),
            );

            // Map the window at the client's requested size.
            let _ = conn.map_window(e.window);
            let _ = conn.flush();

            // Read geometry.
            let (w, h) = match conn.get_geometry(e.window) {
                Ok(cookie) => match cookie.reply() {
                    Ok(geo) => (geo.width as u32, geo.height as u32),
                    Err(_) => (0, 0),
                },
                Err(_) => (0, 0),
            };

            // Read initial properties for classification.
            let role = classify_window(conn, atoms, e.window);
            let app_id = read_u32_prop(conn, e.window, atoms.steam_game);

            // Register in tracker.
            let win = tracker.add_window(e.window);
            win.mapped = true;
            win.width = w;
            win.height = h;
            win.role = role;
            win.app_id = app_id;

            info!(
                window = e.window,
                width = w,
                height = h,
                ?role,
                app_id,
                "mapped window"
            );

            let _ = event_tx.send(XwmEvent::WindowMapped {
                window_id: e.window,
                title: String::new(), // TODO: Read _NET_WM_NAME.
                width: w,
                height: h,
                role,
                app_id,
            });
        }

        x11rb::protocol::Event::UnmapNotify(e) => {
            debug!(window = e.window, "UnmapNotify");
            tracker.unmap_window(e.window);
            let _ = event_tx.send(XwmEvent::WindowUnmapped {
                window_id: e.window,
            });
        }

        x11rb::protocol::Event::DestroyNotify(e) => {
            debug!(window = e.window, "DestroyNotify");
            tracker.remove_window(e.window);
            let _ = event_tx.send(XwmEvent::WindowDestroyed {
                window_id: e.window,
            });
        }

        x11rb::protocol::Event::ConfigureRequest(e) => {
            // Pass through the client's requested geometry verbatim.
            // The compositor handles scaling at present
            // time — the game renders at its preferred size and the
            // viewport stretches the content to fill the host window.
            debug!(
                window = e.window,
                x = e.x,
                y = e.y,
                width = e.width,
                height = e.height,
                "ConfigureRequest"
            );
            let aux = ConfigureWindowAux::from_configure_request(&e).border_width(0);
            let _ = conn.configure_window(e.window, &aux);
            let _ = conn.flush();

            tracker.configure_window(
                e.window,
                e.x as i32,
                e.y as i32,
                e.width as u32,
                e.height as u32,
            );
        }

        x11rb::protocol::Event::PropertyNotify(e) => {
            handle_property_notify(conn, atoms, tracker, event_tx, root, e.window, e.atom);
        }

        _ => {
            // Ignore other events for now.
        }
    }
}

/// Write focus feedback atoms to the root window and notify the main thread.
fn publish_focus_feedback<C: Connection>(
    conn: &C,
    root: u32,
    atoms: &Atoms,
    tracker: &WindowTracker,
    event_tx: &calloop::channel::Sender<XwmEvent>,
) {
    use x11rb::protocol::xproto::{AtomEnum, PropMode};
    let focus = *tracker.focus();
    debug!(?focus, "focus changed");

    let _ = conn.change_property32(
        PropMode::REPLACE,
        root,
        atoms.focused_app,
        AtomEnum::CARDINAL,
        &[focus.focused_app_id],
    );

    if let Some(win_id) = focus.app {
        let _ = conn.change_property32(
            PropMode::REPLACE,
            root,
            atoms.focused_window,
            AtomEnum::CARDINAL,
            &[win_id],
        );
    }

    let focusable_ids = tracker.focusable_app_ids();
    let _ = conn.change_property32(
        PropMode::REPLACE,
        root,
        atoms.focusable_apps,
        AtomEnum::CARDINAL,
        &focusable_ids,
    );

    let triplets = tracker.focusable_window_triplets();
    let _ = conn.change_property32(
        PropMode::REPLACE,
        root,
        atoms.focusable_windows,
        AtomEnum::CARDINAL,
        &triplets,
    );

    let _ = conn.flush();
    let _ = event_tx.send(XwmEvent::FocusChanged(focus));
}

/// Handle a PropertyNotify event on a window or the root.
///
/// Reads the changed property and updates the window tracker or sends
/// runtime control events to the main thread.
fn handle_property_notify<C: Connection>(
    conn: &C,
    atoms: &Atoms,
    tracker: &mut WindowTracker,
    event_tx: &calloop::channel::Sender<XwmEvent>,
    root: u32,
    window: u32,
    atom: X11Atom,
) {
    let is_root = window == root;

    // --- Window properties (set on individual windows) ---
    if !is_root {
        if atom == atoms.steam_game {
            let app_id = read_u32_prop(conn, window, atoms.steam_game);
            tracker.set_app_id(window, app_id);
            debug!(window, app_id, "STEAM_GAME changed");
        } else if atom == atoms.steam_overlay {
            let val = read_u32_prop(conn, window, atoms.steam_overlay);
            if val > 0 {
                tracker.set_role(window, WindowRole::Overlay);
            }
            debug!(window, val, "STEAM_OVERLAY changed");
        } else if atom == atoms.steam_bigpicture {
            let val = read_u32_prop(conn, window, atoms.steam_bigpicture);
            if val > 0 {
                tracker.set_role(window, WindowRole::PlatformClient);
            }
            debug!(window, val, "STEAM_BIGPICTURE changed");
        } else if atom == atoms.external_overlay {
            let val = read_u32_prop(conn, window, atoms.external_overlay);
            if val > 0 {
                tracker.set_role(window, WindowRole::ExternalOverlay);
            }
            debug!(window, val, "GAMECOMP_EXTERNAL_OVERLAY changed");
        } else if atom == atoms.net_wm_opacity {
            let raw = read_u32_prop(conn, window, atoms.net_wm_opacity);
            let opacity = (raw as f64 / u32::MAX as f64) as f32;
            tracker.set_opacity(window, opacity);
        } else if atom == atoms.steam_input_focus {
            let mode = read_u32_prop(conn, window, atoms.steam_input_focus);
            if let Some(win) = tracker.get_mut(window) {
                win.input_focus_mode = mode;
            }
        }
        return;
    }

    // --- Root window properties (runtime control atoms) ---
    if atom == atoms.focus_appid {
        let ids = read_u32_list_prop(conn, root, atoms.focus_appid);
        debug!(?ids, "GAMECOMP_BASELAYER_APPID changed");
        tracker.set_requested_app_ids(ids);
    } else if atom == atoms.focus_window {
        let id = read_u32_prop(conn, root, atoms.focus_window);
        debug!(id, "GAMECOMP_BASELAYER_WINDOW changed");
        tracker.set_requested_window(if id > 0 { Some(id) } else { None });
    } else if atom == atoms.fps_limit {
        let val = read_u32_prop(conn, root, atoms.fps_limit);
        debug!(val, "GAMECOMP_FPS_LIMIT changed");
        let _ = event_tx.send(XwmEvent::FpsLimitChanged(val));
    } else if atom == atoms.vrr_enabled {
        let val = read_u32_prop(conn, root, atoms.vrr_enabled);
        debug!(val, "GAMECOMP_VRR_ENABLED changed");
        let _ = event_tx.send(XwmEvent::VrrChanged(val > 0));
    } else if atom == atoms.low_latency {
        let val = read_u32_prop(conn, root, atoms.low_latency);
        debug!(val, "GAMECOMP_LOW_LATENCY changed");
        let _ = event_tx.send(XwmEvent::LowLatencyChanged(val > 0));
    } else if atom == atoms.allow_tearing {
        let val = read_u32_prop(conn, root, atoms.allow_tearing);
        debug!(val, "GAMECOMP_ALLOW_TEARING changed");
        let _ = event_tx.send(XwmEvent::AllowTearingChanged(val > 0));
    } else if atom == atoms.scaling_filter {
        let val = read_u32_prop(conn, root, atoms.scaling_filter);
        debug!(val, "GAMECOMP_SCALING_FILTER changed");
        let _ = event_tx.send(XwmEvent::ScalingFilterChanged(val));
    } else if atom == atoms.fsr_sharpness {
        let val = read_u32_prop(conn, root, atoms.fsr_sharpness);
        debug!(val, "GAMECOMP_FSR_SHARPNESS changed");
        let _ = event_tx.send(XwmEvent::FsrSharpnessChanged(val));
    } else if atom == atoms.hdr_enabled {
        let val = read_u32_prop(conn, root, atoms.hdr_enabled);
        debug!(val, "GAMECOMP_HDR_ENABLED changed");
        let _ = event_tx.send(XwmEvent::HdrChanged(val > 0));
    } else if atom == atoms.composite_force {
        let val = read_u32_prop(conn, root, atoms.composite_force);
        debug!(val, "GAMECOMP_COMPOSITE_FORCE changed");
        let _ = event_tx.send(XwmEvent::CompositeForceChanged(val > 0));
    } else if atom == atoms.request_screenshot {
        let val = read_u32_prop(conn, root, atoms.request_screenshot);
        debug!(val, "GAMECOMP_REQUEST_SCREENSHOT");
        let _ = event_tx.send(XwmEvent::ScreenshotRequested(val));
    } else if atom == atoms.create_xwayland_server {
        let val = read_u32_prop(conn, root, atoms.create_xwayland_server);
        info!(id = val, "GAMECOMP_CREATE_XWAYLAND_SERVER requested");
        let _ = event_tx.send(XwmEvent::CreateXWaylandServerRequested(val));
    } else if atom == atoms.destroy_xwayland_server {
        let val = read_u32_prop(conn, root, atoms.destroy_xwayland_server);
        info!(id = val, "GAMECOMP_DESTROY_XWAYLAND_SERVER requested");
        let _ = event_tx.send(XwmEvent::DestroyXWaylandServerRequested(val));
    }
}

/// Classify a window's role by reading its initial properties.
fn classify_window<C: Connection>(conn: &C, atoms: &Atoms, window: u32) -> WindowRole {
    if read_u32_prop(conn, window, atoms.steam_overlay) > 0 {
        return WindowRole::Overlay;
    }
    if read_u32_prop(conn, window, atoms.external_overlay) > 0 {
        return WindowRole::ExternalOverlay;
    }
    if read_u32_prop(conn, window, atoms.steam_bigpicture) > 0 {
        return WindowRole::PlatformClient;
    }
    WindowRole::App
}

/// Read a single `u32` property from an X11 window.
///
/// Returns `0` if the property doesn't exist or can't be read.
fn read_u32_prop<C: Connection>(conn: &C, window: u32, atom: X11Atom) -> u32 {
    use x11rb::protocol::xproto::{AtomEnum, ConnectionExt};
    let reply = conn.get_property(false, window, atom, AtomEnum::CARDINAL, 0, 1);
    match reply {
        Ok(cookie) => match cookie.reply() {
            Ok(prop) if prop.value_len == 1 && prop.format == 32 => {
                let vals: &[u32] = bytemuck_or_manual_u32(&prop.value);
                vals.first().copied().unwrap_or(0)
            }
            _ => 0,
        },
        Err(_) => 0,
    }
}

/// Read a list of `u32` properties from an X11 window.
///
/// Returns an empty vec if the property doesn't exist.
fn read_u32_list_prop<C: Connection>(conn: &C, window: u32, atom: X11Atom) -> Vec<u32> {
    use x11rb::protocol::xproto::{AtomEnum, ConnectionExt};
    let reply = conn.get_property(false, window, atom, AtomEnum::CARDINAL, 0, 256);
    match reply {
        Ok(cookie) => match cookie.reply() {
            Ok(prop) if prop.format == 32 && !prop.value.is_empty() => {
                bytemuck_or_manual_u32(&prop.value).to_vec()
            }
            _ => Vec::new(),
        },
        Err(_) => Vec::new(),
    }
}

/// Reinterpret a `&[u8]` as `&[u32]` (little-endian, X11 property data).
///
/// X11 properties with format=32 are stored as arrays of 32-bit values.
fn bytemuck_or_manual_u32(bytes: &[u8]) -> &[u32] {
    if !bytes.len().is_multiple_of(4) || bytes.is_empty() {
        return &[];
    }
    // SAFETY: bytes slice length is a multiple of 4, and X11 properties
    // with format=32 are always properly aligned by the X server.
    unsafe { std::slice::from_raw_parts(bytes.as_ptr().cast::<u32>(), bytes.len() / 4) }
}
