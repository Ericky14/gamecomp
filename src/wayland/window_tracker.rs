//! X11 window tracking and focus determination.
//!
//! Tracks all X11 windows managed by the XWM, classifies them by role
//! (game, overlay, notification, etc.), and determines which window
//! should receive focus.
//!
//! # Design
//!
//! Each `TrackedWindow` stores the X11 window ID, classification, geometry,
//! and per-window properties read from atoms. The `FocusState` holds the
//! current focus decisions — one slot per window role.
//!
//! Focus determination runs whenever:
//! - A window is mapped, unmapped, or destroyed
//! - A property changes on a window or the root (e.g., `GAMESCOPECTRL_BASELAYER_APPID`)
//! - The compositor requests a focus change via `XwmCommand`

use std::collections::HashMap;

/// Classification of an X11 window's role.
///
/// Determines compositing layer order and focus priority.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum WindowRole {
    /// A normal application or game window (composited at `LAYER_APP`).
    #[default]
    App,
    /// Steam overlay (Shift+Tab) — composited above the app.
    Overlay,
    /// External overlay (MangoHud, etc.) — composited on top of everything.
    ExternalOverlay,
    /// Override-redirect popup/dropdown — composited above the focused app.
    Popup,
    /// The platform client window (e.g., storefront launcher).
    PlatformClient,
}

/// Per-window tracked state in the XWM.
#[derive(Debug, Clone)]
pub struct TrackedWindow {
    /// X11 window ID.
    pub id: u32,
    /// Window role classification.
    pub role: WindowRole,
    /// Steam AppID (0 = not a Steam game).
    pub app_id: u32,
    /// Process ID of the window's client.
    pub pid: u32,
    /// Window title.
    pub title: String,
    /// Current geometry.
    pub x: i32,
    pub y: i32,
    pub width: u32,
    pub height: u32,
    /// Opacity (0.0–1.0, from `_NET_WM_OPACITY`).
    pub opacity: f32,
    /// Whether the window is currently mapped (visible).
    pub mapped: bool,
    /// Whether this is an override-redirect window.
    pub override_redirect: bool,
    /// Monotonic sequence number for ordering (higher = more recent).
    pub map_sequence: u64,
    /// Whether the window wants fullscreen.
    pub wants_fullscreen: bool,
    /// Input focus mode (from `STEAM_INPUT_FOCUS`).
    /// 0 = normal, 1 = overlay grabs input, 2 = overlay grabs but keyboard stays.
    pub input_focus_mode: u32,
    /// Wayland surface protocol ID (from `WL_SURFACE_ID` X11 property).
    /// Set by XWayland when mapping an X11 window to a wl_surface.
    /// 0 = not yet associated.
    pub wl_surface_id: u32,
    /// Whether `_NET_WM_STATE_SKIP_TASKBAR` is set.
    pub skip_taskbar: bool,
    /// Whether `_NET_WM_STATE_SKIP_PAGER` is set.
    pub skip_pager: bool,
    /// Whether this window is a system tray icon.
    pub is_systray_icon: bool,
    /// Parent window (from `WM_TRANSIENT_FOR`). 0 = not transient.
    pub transient_for: u32,
}

impl TrackedWindow {
    /// Create a new tracked window with defaults.
    pub fn new(id: u32) -> Self {
        Self {
            id,
            role: WindowRole::App,
            app_id: 0,
            pid: 0,
            title: String::new(),
            x: 0,
            y: 0,
            width: 0,
            height: 0,
            opacity: 1.0,
            mapped: false,
            override_redirect: false,
            map_sequence: 0,
            wants_fullscreen: false,
            input_focus_mode: 0,
            wl_surface_id: 0,
            skip_taskbar: false,
            skip_pager: false,
            is_systray_icon: false,
            transient_for: 0,
        }
    }

    /// Whether this window is focusable (visible, non-trivial size, app/platform role).
    ///
    /// Excludes 1x1 windows, override-redirect,
    /// system tray icons, and skip-taskbar+skip-pager non-fullscreen windows.
    #[inline(always)]
    pub fn is_focusable(&self) -> bool {
        self.mapped
            && self.width > 1
            && self.height > 1
            && !self.override_redirect
            && !self.is_systray_icon
            && !(self.skip_taskbar && self.skip_pager && !self.wants_fullscreen)
            && matches!(self.role, WindowRole::App | WindowRole::PlatformClient)
    }
}

/// Current focus state — one slot per window role.
#[derive(Debug, Clone, Copy, Default)]
pub struct FocusState {
    /// The primary game/app window receiving input.
    pub app: Option<u32>,
    /// Steam overlay window (e.g., Shift+Tab).
    pub overlay: Option<u32>,
    /// External overlay window (MangoHud, etc.).
    pub external_overlay: Option<u32>,
    /// Override-redirect popup/dropdown on top of focused app.
    pub popup: Option<u32>,
    /// The AppID of the currently focused game.
    pub focused_app_id: u32,
    /// Wayland surface protocol ID of the focused app window.
    /// Used by the compositor to gate buffer presentation.
    pub focused_wl_surface_id: u32,
}

/// Tracks all X11 windows and determines focus.
pub struct WindowTracker {
    /// All tracked windows, keyed by X11 window ID.
    windows: HashMap<u32, TrackedWindow>,
    /// Current focus state.
    focus: FocusState,
    /// Monotonic counter for ordering window maps.
    next_sequence: u64,
    /// AppID(s) that an external controller (Steam) wants focused.
    /// Set via `GAMESCOPECTRL_BASELAYER_APPID` atom on the root window.
    requested_app_ids: Vec<u32>,
    /// Specific window ID that an external controller wants focused.
    /// Set via `GAMESCOPECTRL_BASELAYER_WINDOW` atom on the root window.
    requested_window: Option<u32>,
    /// Whether focus needs re-evaluation.
    focus_dirty: bool,
}

impl WindowTracker {
    /// Create a new empty window tracker.
    pub fn new() -> Self {
        Self {
            windows: HashMap::new(),
            focus: FocusState::default(),
            next_sequence: 0,
            requested_app_ids: Vec::new(),
            requested_window: None,
            focus_dirty: false,
        }
    }

    /// Whether a window is a valid focus candidate.
    ///
    /// All focusable, mapped windows qualify. Every window gets an AppID
    /// (either from `STEAM_GAME` or a synthetic ID from the X11 window ID),
    /// so `app_id` is always > 0.
    #[inline(always)]
    fn is_focus_candidate(&self, w: &TrackedWindow) -> bool {
        w.is_focusable()
    }

    /// Register a new window (on MapRequest, before mapping).
    pub fn add_window(&mut self, id: u32) -> &mut TrackedWindow {
        let seq = self.next_sequence;
        self.next_sequence += 1;
        let win = self.windows.entry(id).or_insert_with(|| {
            let mut w = TrackedWindow::new(id);
            w.map_sequence = seq;
            w
        });
        self.focus_dirty = true;
        win
    }

    /// Mark a window as mapped.
    pub fn map_window(&mut self, id: u32, width: u32, height: u32) {
        if let Some(win) = self.windows.get_mut(&id) {
            win.mapped = true;
            win.width = width;
            win.height = height;
            self.focus_dirty = true;
        }
    }

    /// Mark a window as unmapped.
    pub fn unmap_window(&mut self, id: u32) {
        if let Some(win) = self.windows.get_mut(&id) {
            win.mapped = false;
            self.focus_dirty = true;
        }
    }

    /// Remove a window entirely (DestroyNotify).
    pub fn remove_window(&mut self, id: u32) {
        self.windows.remove(&id);
        self.focus_dirty = true;
    }

    /// Update a window's geometry.
    pub fn configure_window(&mut self, id: u32, x: i32, y: i32, width: u32, height: u32) {
        if let Some(win) = self.windows.get_mut(&id) {
            win.x = x;
            win.y = y;
            win.width = width;
            win.height = height;
        }
    }

    /// Set the AppID for a window (from `STEAM_GAME` property).
    pub fn set_app_id(&mut self, id: u32, app_id: u32) {
        if let Some(win) = self.windows.get_mut(&id) {
            win.app_id = app_id;
            self.focus_dirty = true;
        }
    }

    /// Set the window role (from atom properties).
    pub fn set_role(&mut self, id: u32, role: WindowRole) {
        if let Some(win) = self.windows.get_mut(&id) {
            win.role = role;
            self.focus_dirty = true;
        }
    }

    /// Set the opacity for a window (from `_NET_WM_OPACITY`).
    pub fn set_opacity(&mut self, id: u32, opacity: f32) {
        if let Some(win) = self.windows.get_mut(&id) {
            win.opacity = opacity;
        }
    }

    /// Set the Wayland surface protocol ID (from `WL_SURFACE_ID` X11 property).
    pub fn set_wl_surface_id(&mut self, id: u32, wl_surface_id: u32) {
        if let Some(win) = self.windows.get_mut(&id) {
            win.wl_surface_id = wl_surface_id;
            self.focus_dirty = true;
        }
    }

    /// Set the externally requested focus target(s).
    pub fn set_requested_app_ids(&mut self, app_ids: Vec<u32>) {
        self.requested_app_ids = app_ids;
        self.focus_dirty = true;
    }

    /// Set the externally requested focus window.
    pub fn set_requested_window(&mut self, window_id: Option<u32>) {
        self.requested_window = window_id;
        self.focus_dirty = true;
    }

    /// Get a tracked window by ID.
    pub fn get(&self, id: u32) -> Option<&TrackedWindow> {
        self.windows.get(&id)
    }

    /// Get a mutable reference to a tracked window by ID.
    pub fn get_mut(&mut self, id: u32) -> Option<&mut TrackedWindow> {
        self.windows.get_mut(&id)
    }

    /// Current focus state.
    pub fn focus(&self) -> &FocusState {
        &self.focus
    }

    /// Whether focus needs re-evaluation.
    pub fn is_focus_dirty(&self) -> bool {
        self.focus_dirty
    }

    /// Determine focus based on current window state.
    ///
    /// 1. Collect all focusable app windows
    /// 2. If an external controller requested a specific window/app, prefer that
    /// 3. Otherwise, pick the most recently mapped focusable window
    /// 4. Pick the best overlay, external overlay, and notification windows
    ///
    /// Returns `true` if focus changed.
    pub fn determine_focus(&mut self) -> bool {
        self.focus_dirty = false;

        let old_focus = self.focus;

        // Reset focus state — each slot is recomputed from scratch.
        self.focus = FocusState::default();

        // --- Pick the primary app window ---
        // If a specific window is requested, use it.
        if let Some(req_id) = self.requested_window {
            if self
                .windows
                .get(&req_id)
                .is_some_and(|w| w.mapped && w.width > 1 && w.height > 1)
            {
                self.focus.app = Some(req_id);
            }
        } else if !self.requested_app_ids.is_empty() {
            // Find the best window matching one of the requested AppIDs.
            self.focus.app = self
                .windows
                .values()
                .filter(|w| {
                    self.is_focus_candidate(w) && self.requested_app_ids.contains(&w.app_id)
                })
                .max_by_key(|w| w.map_sequence)
                .map(|w| w.id);
        }

        // Fallback: most recently mapped focus candidate.
        if self.focus.app.is_none() {
            self.focus.app = self
                .windows
                .values()
                .filter(|w| self.is_focus_candidate(w))
                .max_by_key(|w| w.map_sequence)
                .map(|w| w.id);
        }

        // Update focused AppID and Wayland surface ID.
        let focused_win = self.focus.app.and_then(|id| self.windows.get(&id));
        self.focus.focused_app_id = focused_win.map_or(0, |w| w.app_id);
        self.focus.focused_wl_surface_id = focused_win.map_or(0, |w| w.wl_surface_id);

        // --- Pick overlay window (highest opacity mapped overlay) ---
        self.focus.overlay = self
            .windows
            .values()
            .filter(|w| w.mapped && w.role == WindowRole::Overlay)
            .max_by(|a, b| {
                a.opacity
                    .partial_cmp(&b.opacity)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|w| w.id);

        // --- Pick external overlay ---
        self.focus.external_overlay = self
            .windows
            .values()
            .filter(|w| w.mapped && w.role == WindowRole::ExternalOverlay)
            .max_by_key(|w| w.map_sequence)
            .map(|w| w.id);

        // --- Pick popup/override-redirect ---
        self.focus.popup = self
            .windows
            .values()
            .filter(|w| w.mapped && w.role == WindowRole::Popup)
            .max_by_key(|w| w.map_sequence)
            .map(|w| w.id);

        // Return whether focus changed — including app_id and surface_id
        // updates that happen asynchronously (STEAM_GAME, WL_SURFACE_ID
        // arrive after MapRequest).
        self.focus.app != old_focus.app
            || self.focus.focused_app_id != old_focus.focused_app_id
            || self.focus.focused_wl_surface_id != old_focus.focused_wl_surface_id
            || self.focus.overlay != old_focus.overlay
            || self.focus.external_overlay != old_focus.external_overlay
            || self.focus.popup != old_focus.popup
    }

    /// Build the list of all focusable AppIDs (for feedback atom).
    pub fn focusable_app_ids(&self) -> Vec<u32> {
        let mut ids: Vec<u32> = self
            .windows
            .values()
            .filter(|w| self.is_focus_candidate(w) && w.app_id > 0)
            .map(|w| w.app_id)
            .collect();
        ids.sort_unstable();
        ids.dedup();
        ids
    }

    /// Build focusable window triplets `[windowID, appID, pid]` (for feedback atom).
    pub fn focusable_window_triplets(&self) -> Vec<u32> {
        let mut triplets = Vec::new();
        for w in self.windows.values().filter(|w| self.is_focus_candidate(w)) {
            triplets.push(w.id);
            triplets.push(w.app_id);
            triplets.push(w.pid);
        }
        triplets
    }
}

#[cfg(test)]
#[path = "window_tracker_tests.rs"]
mod tests;
