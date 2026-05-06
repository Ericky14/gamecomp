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
//! can spawn additional instances via the `GAMESCOPE_CREATE_XWAYLAND_SERVER`
//! atom or `XwmCommand::CreateXWaylandServer`.

use std::process::Child;

use anyhow::Context;
use tracing::{debug, error, info, trace};
use x11rb::connection::Connection;
use x11rb::wrapper::ConnectionExt as WrapperConnectionExt;

use super::atoms::Atoms;
use super::window_tracker::{FocusState, WindowRole, WindowTracker};

/// X11 atom identifier (re-exported for convenience).
type X11Atom = u32;

/// Events sent from the XWM thread to the main thread.
///
/// Each event carries its `server_index` so the main thread knows which
/// XWayland server produced it.
#[derive(Debug)]
pub enum XwmEvent {
    /// A window has been mapped (made visible).
    WindowMapped {
        /// XWayland server index (0 = platform, 1+ = game).
        server_index: u32,
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
    WindowUnmapped { server_index: u32, window_id: u32 },
    /// A window has been destroyed.
    WindowDestroyed { server_index: u32, window_id: u32 },
    /// A window requests fullscreen mode.
    WindowFullscreen { server_index: u32, window_id: u32 },
    /// A window's title has changed.
    TitleChanged {
        server_index: u32,
        window_id: u32,
        title: String,
    },
    /// Focus has changed — includes the full focus state for the compositor.
    FocusChanged(u32, FocusState),

    // --- Atom-driven runtime control events ---
    /// FPS limit changed via `GAMESCOPE_FPS_LIMIT` atom.
    FpsLimitChanged(u32),
    /// VRR enable/disable changed via `GAMESCOPE_VRR_ENABLED` atom.
    VrrChanged(bool),
    /// Low-latency mode changed via `GAMESCOPE_LOW_LATENCY` atom.
    LowLatencyChanged(bool),
    /// Allow-tearing changed via `GAMESCOPE_ALLOW_TEARING` atom.
    AllowTearingChanged(bool),
    /// Scaling filter changed via `GAMESCOPE_SCALING_FILTER` atom.
    ScalingFilterChanged(u32),
    /// FSR sharpness changed via `GAMESCOPE_FSR_SHARPNESS` atom.
    FsrSharpnessChanged(u32),
    /// HDR enable/disable changed via `GAMESCOPE_HDR_ENABLED` atom.
    HdrChanged(bool),
    /// Force composition mode (no direct scanout) via `GAMESCOPE_COMPOSITE_FORCE`.
    CompositeForceChanged(bool),
    /// Screenshot requested via `GAMESCOPECTRL_REQUEST_SCREENSHOT` atom.
    ScreenshotRequested(u32),
    /// Request to create a new XWayland server via atom.
    CreateXWaylandServerRequested(u32),
    /// Request to destroy an XWayland server via atom.
    DestroyXWaylandServerRequested(u32),
    /// Resolution change for a specific server via `GAMESCOPE_XWAYLAND_MODE_CONTROL`.
    /// Fields: `(target_server_idx, width, height, flags)`.
    XWaylandModeControl {
        target_server: u32,
        width: u32,
        height: u32,
        flags: u32,
    },
    /// Focus display changed via `GAMESCOPE_FOCUS_DISPLAY`.
    FocusDisplayChanged(u32),
    /// Baselayer AppID list changed via `GAMESCOPECTRL_BASELAYER_APPID`.
    /// The main loop uses this to prefer servers whose focused AppID
    /// matches a requested ID (cross-server focus control).
    BaselayerAppIdsChanged(Vec<u32>),
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
    /// Update the global focused app/window on this server's root window.
    SetGlobalFocus { app_id: u32 },
    /// Send Expose events to all mapped windows to force a full repaint.
    /// Used after VT switch to restart the client's frame clock.
    RefreshWindows,
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
#[allow(clippy::too_many_arguments)]
pub fn run_xwm(
    display: &str,
    event_tx: &calloop::channel::Sender<XwmEvent>,
    cmd_rx: &std::sync::mpsc::Receiver<XwmCommand>,
    server_index: u32,
    focused_app_id: &std::sync::atomic::AtomicU32,
    focused_wl_surface_id: &std::sync::atomic::AtomicU32,
    output_width: u32,
    output_height: u32,
    steam_mode: bool,
) -> anyhow::Result<()> {
    let x11_display = display;
    info!(
        x11_display,
        server_index, output_width, output_height, "starting X11 window manager"
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

    // Register as a composite manager so GTK/GDK clients know RGBA
    // (Depth 32) visuals are usable. Without this, toolkits fall back to
    // Depth 24 (XRGB) and overlays lose their alpha channel.
    register_composite_manager(&conn, screen_num, root)
        .context("failed to register as composite manager")?;

    // Publish our PID on the root window for identification.
    publish_pid(&conn, root, &atoms);

    // Publish the server index on the root window so gamescope-dbus
    // and other ecosystem tools can discover this XWayland instance.
    {
        use x11rb::protocol::xproto::{AtomEnum, PropMode};
        let _ = conn.change_property32(
            PropMode::REPLACE,
            root,
            atoms.xwayland_server_id,
            AtomEnum::CARDINAL,
            &[server_index],
        );

        // Mark the primary server (index 0) with GAMESCOPE_KEYBOARD_FOCUS_DISPLAY
        if server_index == 0 {
            let _ = conn.change_property32(
                PropMode::REPLACE,
                root,
                atoms.keyboard_focus_display,
                AtomEnum::CARDINAL,
                &[server_index],
            );
        }

        let _ = conn.flush();
    }

    info!("XWM registered as window manager with atom support");

    // Mutable output dimensions — updated via SetResolution commands
    // when the host window resizes.
    let mut output_width = output_width;
    let mut output_height = output_height;

    // Event loop.
    while let ControlFlow::Continue = process_command(
        cmd_rx,
        &conn,
        root,
        &atoms,
        &mut tracker,
        &mut output_width,
        &mut output_height,
    ) {
        // --- Poll for X11 events (non-blocking) ---
        match conn.poll_for_event() {
            Ok(Some(event)) => {
                handle_x11_event(
                    &conn,
                    &atoms,
                    &mut tracker,
                    event_tx,
                    root,
                    event,
                    server_index,
                    output_width,
                    output_height,
                    steam_mode,
                );
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
            publish_focus_feedback(
                &conn,
                root,
                &atoms,
                &tracker,
                event_tx,
                server_index,
                focused_app_id,
                focused_wl_surface_id,
            );
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

/// Register as a composite manager on the given screen.
///
/// Creates a small 1×1 window, claims the `_NET_WM_CM_S<screen>` selection,
/// and sets `_NET_SUPPORTING_WM_CHECK` on the root window. GTK/GDK uses
/// this to decide whether RGBA (Depth 32) visuals are available — without
/// it, clients fall back to Depth 24 (XRGB) and lose alpha transparency.
fn register_composite_manager<C: Connection>(
    conn: &C,
    screen_num: usize,
    root: u32,
) -> anyhow::Result<()> {
    use x11rb::COPY_DEPTH_FROM_PARENT;
    use x11rb::protocol::xproto::{ConnectionExt, WindowClass};

    // Intern the per-screen composite manager atom.
    let atom_name = format!("_NET_WM_CM_S{screen_num}");
    let cm_atom = conn
        .intern_atom(false, atom_name.as_bytes())
        .context("failed to send InternAtom for _NET_WM_CM")?
        .reply()
        .context("failed to intern _NET_WM_CM atom")?
        .atom;

    // Create a small off-screen window to own the selection.
    let wid = conn
        .generate_id()
        .context("failed to generate X11 window ID")?;
    conn.create_window(
        COPY_DEPTH_FROM_PARENT,
        wid,
        root,
        0,
        0,
        1,
        1,
        0,
        WindowClass::INPUT_ONLY,
        0, // CopyFromParent visual
        &Default::default(),
    )
    .context("failed to create composite manager window")?
    .check()
    .context("CreateWindow failed")?;

    // Claim the _NET_WM_CM_S<screen> selection. GDK checks this to
    // decide whether RGBA (Depth 32) visuals are usable.
    conn.set_selection_owner(wid, cm_atom, x11rb::CURRENT_TIME)
        .context("failed to set selection owner")?
        .check()
        .context("SetSelectionOwner failed")?;

    conn.flush()
        .context("failed to flush after CM registration")?;

    info!(atom_name, wid, "registered as composite manager");

    Ok(())
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
    output_width: &mut u32,
    output_height: &mut u32,
) -> ControlFlow {
    use x11rb::protocol::xproto::{AtomEnum, ConfigureWindowAux, ConnectionExt, PropMode};
    match cmd_rx.try_recv() {
        Ok(XwmCommand::Shutdown) => {
            info!("XWM received shutdown command");
            return ControlFlow::Break;
        }
        Ok(XwmCommand::SetResolution { width, height }) => {
            if width != *output_width || height != *output_height {
                info!(
                    old_w = *output_width,
                    old_h = *output_height,
                    new_w = width,
                    new_h = height,
                    "XWM: output resolution changed"
                );
                *output_width = width;
                *output_height = height;

                // Reconfigure non-fixed-size windows to fill the new
                // output. Fixed-size windows (WM_NORMAL_HINTS min==max)
                // keep their current dimensions.
                for win_id in tracker.mapped_windows() {
                    let should_resize = tracker.get(win_id).is_some_and(|w| !w.size_hints_fixed);
                    if should_resize {
                        let _ = conn.configure_window(
                            win_id,
                            &ConfigureWindowAux::new()
                                .x(0)
                                .y(0)
                                .width(width)
                                .height(height),
                        );
                    }
                }
                let _ = conn.flush();
            }
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
        Ok(XwmCommand::SetGlobalFocus { app_id }) => {
            // Override the root window's GAMESCOPE_FOCUSED_APP with the
            // global arbiter result. This is needed when a native Wayland
            // client wins focus — the local XWM has no X11 window for it.
            let _ = conn.change_property32(
                PropMode::REPLACE,
                root,
                atoms.focused_app,
                AtomEnum::CARDINAL,
                &[app_id],
            );
            let _ = conn.flush();
        }
        Ok(XwmCommand::RefreshWindows) => {
            use x11rb::protocol::xproto::{ConnectionExt as _, ExposeEvent};
            for win_id in tracker.mapped_windows() {
                if let Some(win) = tracker.get(win_id) {
                    let event = ExposeEvent {
                        response_type: x11rb::protocol::xproto::EXPOSE_EVENT,
                        sequence: 0,
                        window: win_id,
                        x: 0,
                        y: 0,
                        width: win.width as u16,
                        height: win.height as u16,
                        count: 0,
                    };
                    // Use NO_EVENT so the event is delivered to the
                    // window owner regardless of their event-mask
                    // selections. Modern DRI3/Present clients (Flutter)
                    // don't select ExposureMask, so EXPOSURE would
                    // silently drop the event.
                    let _ = conn.send_event(
                        false,
                        win_id,
                        x11rb::protocol::xproto::EventMask::NO_EVENT,
                        event,
                    );
                }
            }
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
#[allow(clippy::too_many_arguments)]
fn handle_x11_event<C: Connection>(
    conn: &C,
    atoms: &Atoms,
    tracker: &mut WindowTracker,
    event_tx: &calloop::channel::Sender<XwmEvent>,
    root: u32,
    event: x11rb::protocol::Event,
    server_index: u32,
    output_width: u32,
    output_height: u32,
    steam_mode: bool,
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

            // Check WM_NORMAL_HINTS for fixed-size windows (e.g., glxgears).
            let (size_fixed, hint_w, hint_h) =
                read_size_hints(conn, e.window, atoms.wm_normal_hints);

            // Read initial _NET_WM_STATE to check for fullscreen.
            let states = read_atom_list_prop(conn, e.window, atoms.net_wm_state);
            let wants_fullscreen = states.contains(&atoms.net_wm_state_fullscreen);

            // Determine map size. Gaming compositor policy: App and
            // PlatformClient windows fill the output so the client
            // renders at the correct resolution. Fixed-size windows
            // (WM_NORMAL_HINTS min==max) are left alone.
            let (w, h) = if size_fixed {
                // Fixed-size window — ensure position is 0,0 and use
                // the hint dimensions.
                let _ = conn.configure_window(e.window, &ConfigureWindowAux::new().x(0).y(0));
                (hint_w, hint_h)
            } else {
                // Non-fixed window — fill the output so the client
                // renders at display resolution. This covers both
                // explicit fullscreen and normal app windows.
                let _ = conn.configure_window(
                    e.window,
                    &ConfigureWindowAux::new()
                        .x(0)
                        .y(0)
                        .width(output_width)
                        .height(output_height),
                );
                (output_width, output_height)
            };
            let _ = conn.map_window(e.window);
            let _ = conn.flush();

            // Read initial properties for classification.
            let role = classify_window(conn, atoms, e.window);
            // Resolve AppID:
            // 1. STEAM_GAME atom is authoritative if set.
            // 2. In steam mode, walk the process tree to find the AppID
            // 3. In non-steam mode, use the window ID as a synthetic AppID
            //    so every window is always identifiable and focusable.
            let steam_id = read_u32_prop(conn, e.window, atoms.steam_game);
            let app_id = if steam_id != 0 {
                steam_id
            } else if steam_mode {
                let pid = get_client_pid(conn, e.window);
                if pid != 0 {
                    let resolved = get_appid_from_pid(pid);
                    debug!(
                        window = e.window,
                        pid,
                        resolved_app_id = resolved,
                        "PID-based AppID resolution"
                    );
                    resolved
                } else {
                    0
                }
            } else {
                e.window
            };

            // Register in tracker.
            let win = tracker.add_window(e.window);
            win.mapped = true;
            win.width = w;
            win.height = h;
            win.role = role;
            win.app_id = app_id;
            win.size_hints_fixed = size_fixed;
            win.wants_fullscreen = wants_fullscreen;
            if size_fixed {
                win.requested_width = hint_w;
                win.requested_height = hint_h;
            }

            // Apply initial _NET_WM_STATE flags (already read above).
            win.skip_taskbar = states.contains(&atoms.net_wm_state_skip_taskbar);
            win.skip_pager = states.contains(&atoms.net_wm_state_skip_pager);

            // Read WM_TRANSIENT_FOR parent.
            win.transient_for = read_window_prop(conn, e.window, atoms.wm_transient_for);

            info!(
                window = e.window,
                width = w,
                height = h,
                ?role,
                app_id,
                "mapped window"
            );

            let _ = event_tx.send(XwmEvent::WindowMapped {
                server_index,
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
                server_index,
                window_id: e.window,
            });
        }

        x11rb::protocol::Event::DestroyNotify(e) => {
            debug!(window = e.window, "DestroyNotify");
            tracker.remove_window(e.window);
            let _ = event_tx.send(XwmEvent::WindowDestroyed {
                server_index,
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

        x11rb::protocol::Event::ClientMessage(e) => {
            // XWayland sends WL_SURFACE_ID as a ClientMessage to associate
            // an X11 window with its Wayland wl_surface protocol object ID.
            if e.type_ == atoms.wl_surface_id {
                let wl_id = e.data.as_data32()[0];
                trace!(window = e.window, wl_id, "WL_SURFACE_ID ClientMessage");
                tracker.set_wl_surface_id(e.window, wl_id);
            } else if e.type_ == atoms.net_wm_state {
                // EWMH _NET_WM_STATE change request.
                // data32: [action, first_atom, second_atom, source, 0]
                // action: 0=Remove, 1=Add, 2=Toggle
                let d = e.data.as_data32();
                let action = d[0];
                let prop_atoms = [d[1], d[2]];
                let mut fullscreen_changed = false;
                if let Some(win) = tracker.get_mut(e.window) {
                    for &pa in &prop_atoms {
                        if pa == 0 {
                            continue;
                        }
                        if pa == atoms.net_wm_state_fullscreen {
                            match action {
                                0 => win.wants_fullscreen = false,
                                1 => win.wants_fullscreen = true,
                                2 => win.wants_fullscreen = !win.wants_fullscreen,
                                _ => {}
                            }
                            fullscreen_changed = true;
                        } else if pa == atoms.net_wm_state_skip_taskbar {
                            match action {
                                0 => win.skip_taskbar = false,
                                1 => win.skip_taskbar = true,
                                2 => win.skip_taskbar = !win.skip_taskbar,
                                _ => {}
                            }
                        } else if pa == atoms.net_wm_state_skip_pager {
                            match action {
                                0 => win.skip_pager = false,
                                1 => win.skip_pager = true,
                                2 => win.skip_pager = !win.skip_pager,
                                _ => {}
                            }
                        }
                    }
                }
                if fullscreen_changed {
                    // Resize to match new fullscreen state.
                    let is_fs = tracker.get(e.window).is_some_and(|w| w.wants_fullscreen);
                    if is_fs {
                        let _ = conn.configure_window(
                            e.window,
                            &ConfigureWindowAux::new()
                                .x(0)
                                .y(0)
                                .width(output_width)
                                .height(output_height),
                        );
                        let _ = conn.flush();
                    }
                    tracker.mark_focus_dirty();
                }
            }
        }

        _ => {
            // Ignore other events for now.
        }
    }
}

/// Write focus feedback atoms to the root window, set X11 input focus,
/// and notify the main thread.
#[allow(clippy::too_many_arguments)]
fn publish_focus_feedback<C: Connection>(
    conn: &C,
    root: u32,
    atoms: &Atoms,
    tracker: &WindowTracker,
    event_tx: &calloop::channel::Sender<XwmEvent>,
    server_index: u32,
    focused_app_id: &std::sync::atomic::AtomicU32,
    focused_wl_surface_id: &std::sync::atomic::AtomicU32,
) {
    use x11rb::protocol::xproto::{AtomEnum, ConnectionExt, InputFocus, PropMode};
    let focus = *tracker.focus();
    debug!(?focus, "focus changed");

    // Update per-server atomics. The main loop aggregates all servers
    // and picks the winner (first with app_id != 0).
    focused_app_id.store(focus.focused_app_id, std::sync::atomic::Ordering::Relaxed);
    focused_wl_surface_id.store(
        focus.focused_wl_surface_id,
        std::sync::atomic::Ordering::Relaxed,
    );

    // Set X11 input focus based on overlay input_focus_mode priority.
    // Gamescope priority: external_overlay > steam_overlay > app.
    // Only overlays with input_focus_mode > 0 grab input focus.
    let focus_target = if focus.external_overlay_input_focus_mode > 0 {
        focus.external_overlay
    } else if focus.overlay_input_focus_mode > 0 {
        focus.overlay
    } else {
        focus.app
    };
    if let Some(win_id) = focus_target {
        let _ = conn.set_input_focus(InputFocus::NONE, win_id, x11rb::CURRENT_TIME);
    }

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
    let _ = event_tx.send(XwmEvent::FocusChanged(server_index, focus));
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
            } else {
                tracker.set_role(window, WindowRole::App);
            }
            debug!(window, val, "STEAM_OVERLAY changed");
        } else if atom == atoms.steam_bigpicture {
            let val = read_u32_prop(conn, window, atoms.steam_bigpicture);
            if val > 0 {
                tracker.set_role(window, WindowRole::PlatformClient);
            } else {
                tracker.set_role(window, WindowRole::App);
            }
            debug!(window, val, "STEAM_BIGPICTURE changed");
        } else if atom == atoms.external_overlay {
            let val = read_u32_prop(conn, window, atoms.external_overlay);
            if val > 0 {
                tracker.set_role(window, WindowRole::ExternalOverlay);
            } else {
                tracker.set_role(window, WindowRole::App);
            }
            debug!(window, val, "GAMESCOPE_EXTERNAL_OVERLAY changed");
        } else if atom == atoms.net_wm_opacity {
            let raw = read_u32_prop(conn, window, atoms.net_wm_opacity);
            let opacity = (raw as f64 / u32::MAX as f64) as f32;
            tracker.set_opacity(window, opacity);
        } else if atom == atoms.steam_input_focus {
            let mode = read_u32_prop(conn, window, atoms.steam_input_focus);
            if let Some(win) = tracker.get_mut(window) {
                win.input_focus_mode = mode;
            }
        } else if atom == atoms.wl_surface_id {
            let wl_id = read_u32_prop(conn, window, atoms.wl_surface_id);
            trace!(window, wl_id, "WL_SURFACE_ID set");
            tracker.set_wl_surface_id(window, wl_id);
        } else if atom == atoms.net_wm_state {
            // Read _NET_WM_STATE list and update skip_taskbar/skip_pager.
            update_net_wm_state_flags(conn, atoms, tracker, window);
        } else if atom == atoms.wm_transient_for {
            // Read WM_TRANSIENT_FOR (window type atom, stored as WINDOW).
            let parent = read_window_prop(conn, window, atoms.wm_transient_for);
            if let Some(win) = tracker.get_mut(window) {
                win.transient_for = parent;
            }
        }
        return;
    }

    // --- Root window properties (runtime control atoms) ---
    if atom == atoms.focus_appid {
        let ids = read_u32_list_prop(conn, root, atoms.focus_appid);
        debug!(?ids, "GAMESCOPECTRL_BASELAYER_APPID changed");
        tracker.set_requested_app_ids(ids.clone());
        let _ = event_tx.send(XwmEvent::BaselayerAppIdsChanged(ids));
    } else if atom == atoms.focus_window {
        let id = read_u32_prop(conn, root, atoms.focus_window);
        debug!(id, "GAMESCOPECTRL_BASELAYER_WINDOW changed");
        tracker.set_requested_window(if id > 0 { Some(id) } else { None });
    } else if atom == atoms.fps_limit {
        let val = read_u32_prop(conn, root, atoms.fps_limit);
        debug!(val, "GAMESCOPE_FPS_LIMIT changed");
        let _ = event_tx.send(XwmEvent::FpsLimitChanged(val));
    } else if atom == atoms.vrr_enabled {
        let val = read_u32_prop(conn, root, atoms.vrr_enabled);
        debug!(val, "GAMESCOPE_VRR_ENABLED changed");
        let _ = event_tx.send(XwmEvent::VrrChanged(val > 0));
    } else if atom == atoms.low_latency {
        let val = read_u32_prop(conn, root, atoms.low_latency);
        debug!(val, "GAMESCOPE_LOW_LATENCY changed");
        let _ = event_tx.send(XwmEvent::LowLatencyChanged(val > 0));
    } else if atom == atoms.allow_tearing {
        let val = read_u32_prop(conn, root, atoms.allow_tearing);
        debug!(val, "GAMESCOPE_ALLOW_TEARING changed");
        let _ = event_tx.send(XwmEvent::AllowTearingChanged(val > 0));
    } else if atom == atoms.scaling_filter {
        let val = read_u32_prop(conn, root, atoms.scaling_filter);
        debug!(val, "GAMESCOPE_SCALING_FILTER changed");
        let _ = event_tx.send(XwmEvent::ScalingFilterChanged(val));
    } else if atom == atoms.fsr_sharpness {
        let val = read_u32_prop(conn, root, atoms.fsr_sharpness);
        debug!(val, "GAMESCOPE_FSR_SHARPNESS changed");
        let _ = event_tx.send(XwmEvent::FsrSharpnessChanged(val));
    } else if atom == atoms.hdr_enabled {
        let val = read_u32_prop(conn, root, atoms.hdr_enabled);
        debug!(val, "GAMESCOPE_HDR_ENABLED changed");
        let _ = event_tx.send(XwmEvent::HdrChanged(val > 0));
    } else if atom == atoms.composite_force {
        let val = read_u32_prop(conn, root, atoms.composite_force);
        debug!(val, "GAMESCOPE_COMPOSITE_FORCE changed");
        let _ = event_tx.send(XwmEvent::CompositeForceChanged(val > 0));
    } else if atom == atoms.request_screenshot {
        let val = read_u32_prop(conn, root, atoms.request_screenshot);
        debug!(val, "GAMESCOPECTRL_REQUEST_SCREENSHOT");
        let _ = event_tx.send(XwmEvent::ScreenshotRequested(val));
    } else if atom == atoms.create_xwayland_server {
        let val = read_u32_prop(conn, root, atoms.create_xwayland_server);
        info!(id = val, "GAMESCOPE_CREATE_XWAYLAND_SERVER requested");
        let _ = event_tx.send(XwmEvent::CreateXWaylandServerRequested(val));
    } else if atom == atoms.destroy_xwayland_server {
        let val = read_u32_prop(conn, root, atoms.destroy_xwayland_server);
        info!(id = val, "GAMESCOPE_DESTROY_XWAYLAND_SERVER requested");
        let _ = event_tx.send(XwmEvent::DestroyXWaylandServerRequested(val));
    } else if atom == atoms.xwayland_mode_control {
        let vals = read_u32_list_prop(conn, root, atoms.xwayland_mode_control);
        if vals.len() >= 3 {
            let target = vals[0];
            let width = vals[1];
            let height = vals[2];
            let flags = vals.get(3).copied().unwrap_or(0);
            info!(
                target_server = target,
                width, height, flags, "GAMESCOPE_XWAYLAND_MODE_CONTROL changed"
            );
            let _ = event_tx.send(XwmEvent::XWaylandModeControl {
                target_server: target,
                width,
                height,
                flags,
            });
        }
    } else if atom == atoms.focus_display {
        let val = read_u32_prop(conn, root, atoms.focus_display);
        info!(display = val, "GAMESCOPE_FOCUS_DISPLAY changed");
        let _ = event_tx.send(XwmEvent::FocusDisplayChanged(val));
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

/// Read an atom-list property (type ATOM, format 32) from an X11 window.
fn read_atom_list_prop<C: Connection>(conn: &C, window: u32, atom: X11Atom) -> Vec<u32> {
    use x11rb::protocol::xproto::{AtomEnum, ConnectionExt};
    let reply = conn.get_property(false, window, atom, AtomEnum::ATOM, 0, 256);
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

/// Read a WINDOW-type property (format 32, single value) from an X11 window.
fn read_window_prop<C: Connection>(conn: &C, window: u32, atom: X11Atom) -> u32 {
    use x11rb::protocol::xproto::{AtomEnum, ConnectionExt};
    let reply = conn.get_property(false, window, atom, AtomEnum::WINDOW, 0, 1);
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

/// Size hint flags from `WM_SIZE_HINTS`.
const P_MIN_SIZE: u32 = 1 << 4; // 16
const P_MAX_SIZE: u32 = 1 << 5; // 32

/// Read `WM_NORMAL_HINTS` and return `(is_fixed, width, height)`.
///
/// A window is "fixed size" when `PMinSize` and `PMaxSize` are both set
/// and `min_width == max_width && min_height == max_height`.
fn read_size_hints<C: Connection>(
    conn: &C,
    window: u32,
    wm_normal_hints_atom: X11Atom,
) -> (bool, u32, u32) {
    use x11rb::protocol::xproto::ConnectionExt;

    // WM_SIZE_HINTS is type WM_SIZE_HINTS (not CARDINAL). Use AnyPropertyType (0).
    let reply = conn.get_property(false, window, wm_normal_hints_atom, 0u32, 0, 18);
    let Ok(cookie) = reply else {
        return (false, 0, 0);
    };
    let Ok(prop) = cookie.reply() else {
        return (false, 0, 0);
    };
    if prop.format != 32 || prop.value_len < 9 {
        return (false, 0, 0);
    }

    let vals = bytemuck_or_manual_u32(&prop.value);
    if vals.len() < 9 {
        return (false, 0, 0);
    }

    let flags = vals[0];
    // Offsets: min_width=5, min_height=6, max_width=7, max_height=8
    let min_w = vals[5];
    let min_h = vals[6];
    let max_w = vals[7];
    let max_h = vals[8];

    let has_min = flags & P_MIN_SIZE != 0;
    let has_max = flags & P_MAX_SIZE != 0;
    let is_fixed = has_min
        && has_max
        && min_w > 0
        && min_h > 0
        && max_w > 0
        && max_h > 0
        && min_w == max_w
        && min_h == max_h;

    if is_fixed {
        (true, max_w, max_h)
    } else {
        (false, 0, 0)
    }
}

/// Query the PID of the client that created an X11 window via XRes.
///
/// Returns `0` if the PID cannot be determined.
fn get_client_pid<C: Connection>(conn: &C, window: u32) -> u32 {
    use x11rb::protocol::res::{ClientIdMask, ClientIdSpec, ConnectionExt as ResExt};
    let spec = ClientIdSpec {
        client: window,
        mask: ClientIdMask::LOCAL_CLIENT_PID,
    };
    let Ok(cookie) = conn.res_query_client_ids(&[spec]) else {
        return 0;
    };
    let Ok(reply) = cookie.reply() else {
        return 0;
    };
    for id_value in &reply.ids {
        if let Some(&pid) = id_value.value.first()
            && pid > 0
        {
            return pid;
        }
    }
    0
}

/// Read `_NET_WM_STATE` and update skip_taskbar / skip_pager on the tracked window.
fn update_net_wm_state_flags<C: Connection>(
    conn: &C,
    atoms: &super::atoms::Atoms,
    tracker: &mut super::window_tracker::WindowTracker,
    window: u32,
) {
    let states = read_atom_list_prop(conn, window, atoms.net_wm_state);
    let skip_taskbar = states.contains(&atoms.net_wm_state_skip_taskbar);
    let skip_pager = states.contains(&atoms.net_wm_state_skip_pager);
    let wants_fullscreen = states.contains(&atoms.net_wm_state_fullscreen);
    if let Some(win) = tracker.get_mut(window) {
        win.skip_taskbar = skip_taskbar;
        win.skip_pager = skip_pager;
        if win.wants_fullscreen != wants_fullscreen {
            win.wants_fullscreen = wants_fullscreen;
            tracker.mark_focus_dirty();
        }
    }
}

/// Resolve an AppID from a process ID by walking the parent chain.
///
/// Looks for a `reaper` process with `SteamLaunch AppId=<N>` in its
/// command line (Steam's convention). Returns the numeric `<N>`.
///
/// Returns 0 if no reaper is found in the ancestry.
fn get_appid_from_pid(pid: u32) -> u32 {
    let mut next_pid = pid;

    loop {
        // Read /proc/<pid>/stat to get the process name and parent PID.
        let stat_path = format!("/proc/{next_pid}/stat");
        let stat_contents = match std::fs::read_to_string(&stat_path) {
            Ok(s) => s,
            Err(_) => break,
        };

        // Parse process name (between parentheses) and parent PID.
        let open_paren = match stat_contents.find('(') {
            Some(i) => i,
            None => break,
        };
        let close_paren = match stat_contents.rfind(')') {
            Some(i) => i,
            None => break,
        };
        let proc_name = &stat_contents[open_paren + 1..close_paren];

        // Parse parent PID: field after ") <state> <ppid>"
        let after_paren = &stat_contents[close_paren + 1..];
        let fields: Vec<&str> = after_paren.split_whitespace().collect();
        // fields[0] = state, fields[1] = ppid
        let parent_pid: u32 = fields.get(1).and_then(|s| s.parse().ok()).unwrap_or(0);

        if proc_name == "reaper" {
            // Read cmdline for Steam's reaper convention.
            let cmdline_path = format!("/proc/{next_pid}/cmdline");
            if let Ok(cmdline_bytes) = std::fs::read(&cmdline_path) {
                let mut steam_launch = false;
                for arg in cmdline_bytes.split(|b| *b == 0) {
                    let arg_str = std::str::from_utf8(arg).unwrap_or("");
                    if arg_str == "SteamLaunch" {
                        steam_launch = true;
                    } else if let Some(rest) = arg_str.strip_prefix("AppId=")
                        && steam_launch
                        && let Ok(app_id) = rest.parse::<u32>()
                        && app_id != 0
                    {
                        return app_id;
                    }
                    if arg_str == "--" {
                        break;
                    }
                }
            }
        }

        if parent_pid == 0 {
            break;
        }
        next_pid = parent_pid;
    }

    0
}
