//! X11 atom definitions and registry.
//!
//! Defines all custom atoms used for runtime compositor control. Atoms are interned once at XWM startup
//! and stored for the lifetime of the connection.
//!
//! # Design
//!
//! Atoms are organized by category:
//! - **Window properties**: set on individual X11 windows to classify them
//!   (game, overlay, notification, etc.)
//! - **Root properties**: set on the root window by external controllers
//!   (Steam, MangoHud, scripts) to control compositor behavior at runtime
//! - **Feedback properties**: written by the compositor back to the root window
//!   so controllers can read current state
//!
//! The naming convention follows gamescope:
//! - `GAMECOMP_*` — compositor-specific atoms
//! - `STEAM_*` — Steam compatibility atoms (same names as gamescope for
//!   drop-in compatibility)

use anyhow::Context;
use tracing::info;
use x11rb::connection::Connection;
use x11rb::protocol::xproto::ConnectionExt;

/// All interned atoms, grouped by category.
///
/// Created once per XWM connection via [`Atoms::intern`].
#[derive(Debug)]
pub struct Atoms {
    // --- Standard atoms ---
    /// `WL_SURFACE_ID` — links X11 window to its Wayland wl_surface.
    pub wl_surface_id: u32,
    /// `_NET_WM_NAME` — window title (UTF-8).
    pub net_wm_name: u32,
    /// `_NET_WM_PID` — window process ID.
    pub net_wm_pid: u32,
    /// `_NET_WM_STATE` — window state list.
    pub net_wm_state: u32,
    /// `_NET_WM_STATE_FULLSCREEN` — fullscreen state flag.
    pub net_wm_state_fullscreen: u32,
    /// `_NET_WM_WINDOW_TYPE` — window type.
    pub net_wm_window_type: u32,
    /// `_NET_WM_WINDOW_TYPE_NORMAL` — normal window type.
    pub net_wm_window_type_normal: u32,
    /// `_NET_WM_WINDOW_TYPE_DIALOG` — dialog window type.
    pub net_wm_window_type_dialog: u32,
    /// `_NET_WM_WINDOW_TYPE_POPUP_MENU` — popup menu window type.
    pub net_wm_window_type_popup: u32,
    /// `_NET_WM_OPACITY` — window opacity (0–0xFFFFFFFF).
    pub net_wm_opacity: u32,
    /// `WM_DELETE_WINDOW` — close window protocol atom.
    pub wm_delete_window: u32,
    /// `WM_PROTOCOLS` — supported protocols list.
    pub wm_protocols: u32,
    /// `UTF8_STRING` — UTF-8 string type atom.
    pub utf8_string: u32,

    // --- Window classification atoms (set on individual windows) ---
    /// `STEAM_GAME` — Steam AppID for game windows.
    pub steam_game: u32,
    /// `STEAM_OVERLAY` — marks a window as a Steam overlay.
    pub steam_overlay: u32,
    /// `STEAM_BIGPICTURE` — marks a window as the Steam client.
    pub steam_bigpicture: u32,
    /// `STEAM_INPUT_FOCUS` — input focus mode (0=normal, 1=overlay grabs).
    pub steam_input_focus: u32,
    /// `GAMECOMP_EXTERNAL_OVERLAY` — external overlay (MangoHud, etc.).
    pub external_overlay: u32,

    // --- Focus control atoms (set on root by Steam/controller) ---
    /// `GAMECOMP_BASELAYER_APPID` — array of AppIDs to focus.
    pub focus_appid: u32,
    /// `GAMECOMP_BASELAYER_WINDOW` — specific window ID to focus.
    pub focus_window: u32,
    /// `GAMECOMP_FOCUS_DISPLAY` — which display receives focus.
    pub focus_display: u32,

    // --- Resolution / display atoms (set on root) ---
    /// `GAMECOMP_XWAYLAND_MODE_CONTROL` — `[server_idx, width, height, flags]`.
    pub xwayland_mode_control: u32,
    /// `GAMECOMP_FORCE_WINDOWS_FULLSCREEN` — force all windows fullscreen.
    pub force_windows_fullscreen: u32,

    // --- FPS / performance atoms (set on root) ---
    /// `GAMECOMP_FPS_LIMIT` — runtime FPS cap.
    pub fps_limit: u32,
    /// `GAMECOMP_VRR_ENABLED` — enable variable refresh rate.
    pub vrr_enabled: u32,
    /// `GAMECOMP_LOW_LATENCY` — enable low-latency mode.
    pub low_latency: u32,
    /// `GAMECOMP_ALLOW_TEARING` — allow tearing for VRR.
    pub allow_tearing: u32,

    // --- Scaling / filter atoms (set on root) ---
    /// `GAMECOMP_SCALING_FILTER` — scaling filter selection.
    pub scaling_filter: u32,
    /// `GAMECOMP_FSR_SHARPNESS` — FSR sharpness (0–20).
    pub fsr_sharpness: u32,
    /// `GAMECOMP_SHARPNESS` — general sharpness (0–20).
    pub sharpness: u32,

    // --- HDR atoms (set on root) ---
    /// `GAMECOMP_HDR_ENABLED` — enable HDR output.
    pub hdr_enabled: u32,
    /// `GAMECOMP_SDR_ON_HDR_BRIGHTNESS` — SDR content brightness in HDR mode.
    pub sdr_on_hdr_brightness: u32,

    // --- Compositor debug atoms (set on root) ---
    /// `GAMECOMP_COMPOSITE_FORCE` — force full composition (no direct scanout).
    pub composite_force: u32,
    /// `GAMECOMP_COMPOSITE_DEBUG` — debug visualization flags.
    pub composite_debug: u32,

    // --- Multi-display atoms (set on root) ---
    /// `GAMECOMP_XWAYLAND_SERVER_ID` — published on each server's root window
    /// so external clients can identify which XWayland server they are on.
    pub xwayland_server_id: u32,
    /// `GAMECOMP_CREATE_XWAYLAND_SERVER` — request new XWayland server.
    pub create_xwayland_server: u32,
    /// `GAMECOMP_CREATE_XWAYLAND_SERVER_FEEDBACK` — feedback after dynamic server
    /// creation. Value is `"<identifier> <server_id> <display_name>"`.
    pub create_xwayland_server_feedback: u32,
    /// `GAMECOMP_DESTROY_XWAYLAND_SERVER` — destroy XWayland server by ID.
    pub destroy_xwayland_server: u32,

    // --- Feedback atoms (written by compositor to root for controllers to read) ---
    /// `GAMECOMP_FOCUSED_APP` — currently focused AppID.
    pub focused_app: u32,
    /// `GAMECOMP_FOCUSED_WINDOW` — currently focused window ID.
    pub focused_window: u32,
    /// `GAMECOMP_FOCUSABLE_APPS` — list of all focusable AppIDs.
    pub focusable_apps: u32,
    /// `GAMECOMP_FOCUSABLE_WINDOWS` — triplets [windowID, appID, pid].
    pub focusable_windows: u32,
    /// `GAMECOMP_VRR_CAPABLE` — whether display supports VRR.
    pub vrr_capable: u32,
    /// `GAMECOMP_VRR_IN_USE` — whether VRR is currently active.
    pub vrr_in_use: u32,
    /// `GAMECOMP_HDR_SUPPORTED` — whether display supports HDR.
    pub hdr_supported: u32,
    /// `GAMECOMP_FSR_ACTIVE` — whether FSR is currently active.
    pub fsr_active: u32,
    /// `GAMECOMP_DISPLAY_REFRESH_RATE` — current display refresh rate.
    pub display_refresh_rate: u32,
    /// `GAMECOMP_CURSOR_VISIBLE` — cursor visibility state.
    pub cursor_visible: u32,
    /// `GAMECOMP_PID` — compositor process ID.
    pub pid: u32,

    // --- Screenshot ---
    /// `GAMECOMP_REQUEST_SCREENSHOT` — trigger screenshot.
    pub request_screenshot: u32,
}

impl Atoms {
    /// Intern all atoms on the given X11 connection.
    ///
    /// Sends all `InternAtom` requests in a single batch for efficiency,
    /// then collects all replies. This is much faster than interning
    /// one-by-one (one round-trip vs N round-trips).
    pub fn intern<C: Connection>(conn: &C) -> anyhow::Result<Self> {
        let names: &[&str] = &[
            // Standard
            "WL_SURFACE_ID",
            "_NET_WM_NAME",
            "_NET_WM_PID",
            "_NET_WM_STATE",
            "_NET_WM_STATE_FULLSCREEN",
            "_NET_WM_WINDOW_TYPE",
            "_NET_WM_WINDOW_TYPE_NORMAL",
            "_NET_WM_WINDOW_TYPE_DIALOG",
            "_NET_WM_WINDOW_TYPE_POPUP_MENU",
            "_NET_WM_OPACITY",
            "WM_DELETE_WINDOW",
            "WM_PROTOCOLS",
            "UTF8_STRING",
            // Window classification
            "STEAM_GAME",
            "STEAM_OVERLAY",
            "STEAM_BIGPICTURE",
            "STEAM_INPUT_FOCUS",
            "GAMECOMP_EXTERNAL_OVERLAY",
            // Focus control
            "GAMECOMP_BASELAYER_APPID",
            "GAMECOMP_BASELAYER_WINDOW",
            "GAMECOMP_FOCUS_DISPLAY",
            // Resolution / display
            "GAMECOMP_XWAYLAND_MODE_CONTROL",
            "GAMECOMP_FORCE_WINDOWS_FULLSCREEN",
            // FPS / performance
            "GAMECOMP_FPS_LIMIT",
            "GAMECOMP_VRR_ENABLED",
            "GAMECOMP_LOW_LATENCY",
            "GAMECOMP_ALLOW_TEARING",
            // Scaling / filter
            "GAMECOMP_SCALING_FILTER",
            "GAMECOMP_FSR_SHARPNESS",
            "GAMECOMP_SHARPNESS",
            // HDR
            "GAMECOMP_HDR_ENABLED",
            "GAMECOMP_SDR_ON_HDR_BRIGHTNESS",
            // Debug
            "GAMECOMP_COMPOSITE_FORCE",
            "GAMECOMP_COMPOSITE_DEBUG",
            // Multi-display
            "GAMECOMP_XWAYLAND_SERVER_ID",
            "GAMECOMP_CREATE_XWAYLAND_SERVER",
            "GAMECOMP_CREATE_XWAYLAND_SERVER_FEEDBACK",
            "GAMECOMP_DESTROY_XWAYLAND_SERVER",
            // Feedback
            "GAMECOMP_FOCUSED_APP",
            "GAMECOMP_FOCUSED_WINDOW",
            "GAMECOMP_FOCUSABLE_APPS",
            "GAMECOMP_FOCUSABLE_WINDOWS",
            "GAMECOMP_VRR_CAPABLE",
            "GAMECOMP_VRR_IN_USE",
            "GAMECOMP_HDR_SUPPORTED",
            "GAMECOMP_FSR_ACTIVE",
            "GAMECOMP_DISPLAY_REFRESH_RATE",
            "GAMECOMP_CURSOR_VISIBLE",
            "GAMECOMP_PID",
            // Screenshot
            "GAMECOMP_REQUEST_SCREENSHOT",
        ];

        // Send all requests in one batch (pipelined, single flush).
        let cookies: Vec<_> = names
            .iter()
            .map(|name| conn.intern_atom(false, name.as_bytes()))
            .collect::<Result<Vec<_>, _>>()
            .context("failed to send InternAtom requests")?;

        // Collect all replies.
        let atoms: Vec<u32> = cookies
            .into_iter()
            .map(|cookie| cookie.reply().map(|r| r.atom))
            .collect::<Result<Vec<_>, _>>()
            .context("failed to collect InternAtom replies")?;

        info!(count = atoms.len(), "interned X11 atoms");

        Ok(Self {
            // Standard
            wl_surface_id: atoms[0],
            net_wm_name: atoms[1],
            net_wm_pid: atoms[2],
            net_wm_state: atoms[3],
            net_wm_state_fullscreen: atoms[4],
            net_wm_window_type: atoms[5],
            net_wm_window_type_normal: atoms[6],
            net_wm_window_type_dialog: atoms[7],
            net_wm_window_type_popup: atoms[8],
            net_wm_opacity: atoms[9],
            wm_delete_window: atoms[10],
            wm_protocols: atoms[11],
            utf8_string: atoms[12],
            // Window classification
            steam_game: atoms[13],
            steam_overlay: atoms[14],
            steam_bigpicture: atoms[15],
            steam_input_focus: atoms[16],
            external_overlay: atoms[17],
            // Focus control
            focus_appid: atoms[18],
            focus_window: atoms[19],
            focus_display: atoms[20],
            // Resolution / display
            xwayland_mode_control: atoms[21],
            force_windows_fullscreen: atoms[22],
            // FPS / performance
            fps_limit: atoms[23],
            vrr_enabled: atoms[24],
            low_latency: atoms[25],
            allow_tearing: atoms[26],
            // Scaling / filter
            scaling_filter: atoms[27],
            fsr_sharpness: atoms[28],
            sharpness: atoms[29],
            // HDR
            hdr_enabled: atoms[30],
            sdr_on_hdr_brightness: atoms[31],
            // Debug
            composite_force: atoms[32],
            composite_debug: atoms[33],
            // Multi-display
            xwayland_server_id: atoms[34],
            create_xwayland_server: atoms[35],
            create_xwayland_server_feedback: atoms[36],
            destroy_xwayland_server: atoms[37],
            // Feedback
            focused_app: atoms[38],
            focused_window: atoms[39],
            focusable_apps: atoms[40],
            focusable_windows: atoms[41],
            vrr_capable: atoms[42],
            vrr_in_use: atoms[43],
            hdr_supported: atoms[44],
            fsr_active: atoms[45],
            display_refresh_rate: atoms[46],
            cursor_visible: atoms[47],
            pid: atoms[48],
            // Screenshot
            request_screenshot: atoms[49],
        })
    }
}
