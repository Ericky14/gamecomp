use super::*;

#[test]
fn empty_tracker_has_no_focus() {
    let mut tracker = WindowTracker::new();
    tracker.determine_focus();
    assert!(tracker.focus().app.is_none());
}

#[test]
fn single_mapped_window_gets_focus() {
    let mut tracker = WindowTracker::new();
    tracker.add_window(42);
    tracker.map_window(42, 1920, 1080);
    tracker.determine_focus();
    assert_eq!(tracker.focus().app, Some(42));
}

#[test]
fn most_recent_window_wins_focus() {
    let mut tracker = WindowTracker::new();
    tracker.add_window(1);
    tracker.map_window(1, 1920, 1080);
    tracker.add_window(2);
    tracker.map_window(2, 1920, 1080);
    tracker.determine_focus();
    assert_eq!(tracker.focus().app, Some(2));
}

#[test]
fn requested_app_id_takes_priority() {
    let mut tracker = WindowTracker::new();

    tracker.add_window(1);
    tracker.map_window(1, 1920, 1080);
    tracker.set_app_id(1, 100);

    tracker.add_window(2);
    tracker.map_window(2, 1920, 1080);
    tracker.set_app_id(2, 200);

    // Request AppID 100 (window 1), even though window 2 is newer.
    tracker.set_requested_app_ids(vec![100]);
    tracker.determine_focus();
    assert_eq!(tracker.focus().app, Some(1));
    assert_eq!(tracker.focus().focused_app_id, 100);
}

#[test]
fn overlay_classification() {
    let mut tracker = WindowTracker::new();

    // Game window.
    tracker.add_window(1);
    tracker.map_window(1, 1920, 1080);

    // Overlay window.
    tracker.add_window(2);
    tracker.map_window(2, 1920, 1080);
    tracker.set_role(2, WindowRole::Overlay);

    tracker.determine_focus();
    assert_eq!(tracker.focus().app, Some(1));
    assert_eq!(tracker.focus().overlay, Some(2));
}

#[test]
fn unmapped_window_loses_focus() {
    let mut tracker = WindowTracker::new();
    tracker.add_window(1);
    tracker.map_window(1, 1920, 1080);
    tracker.determine_focus();
    assert_eq!(tracker.focus().app, Some(1));

    tracker.unmap_window(1);
    tracker.determine_focus();
    assert!(tracker.focus().app.is_none());
}

#[test]
fn focusable_app_ids_deduped() {
    let mut tracker = WindowTracker::new();

    tracker.add_window(1);
    tracker.map_window(1, 1920, 1080);
    tracker.set_app_id(1, 100);

    tracker.add_window(2);
    tracker.map_window(2, 800, 600);
    tracker.set_app_id(2, 100); // Same AppID.

    let ids = tracker.focusable_app_ids();
    assert_eq!(ids, vec![100]);
}

#[test]
fn window_with_app_id_gets_focus() {
    let mut tracker = WindowTracker::new();

    // Window with STEAM_GAME atom — gets focus.
    tracker.add_window(1);
    tracker.map_window(1, 1920, 1080);
    tracker.set_app_id(1, 769);

    tracker.determine_focus();
    assert_eq!(tracker.focus().app, Some(1));
    assert_eq!(tracker.focus().focused_app_id, 769);
}

#[test]
fn window_without_app_id_is_focusable() {
    let mut tracker = WindowTracker::new();

    // Window without STEAM_GAME — still focusable (app_id assigned
    // from X11 window ID by the XWM, but tracker sees app_id=0
    // for windows that haven't been classified yet).
    tracker.add_window(1);
    tracker.map_window(1, 1920, 1080);

    tracker.determine_focus();
    assert_eq!(tracker.focus().app, Some(1));
}

#[test]
fn overlay_tracks_wl_surface_id() {
    let mut tracker = WindowTracker::new();

    // Game window.
    tracker.add_window(1);
    tracker.map_window(1, 1920, 1080);
    tracker.set_wl_surface_id(1, 100);

    // Overlay window with wl_surface_id.
    tracker.add_window(2);
    tracker.map_window(2, 1920, 1080);
    tracker.set_role(2, WindowRole::Overlay);
    tracker.set_wl_surface_id(2, 200);

    tracker.determine_focus();
    assert_eq!(tracker.focus().overlay, Some(2));
    assert_eq!(tracker.focus().overlay_wl_surface_id, 200);
}

#[test]
fn external_overlay_tracks_wl_surface_id() {
    let mut tracker = WindowTracker::new();

    tracker.add_window(1);
    tracker.map_window(1, 1920, 1080);

    tracker.add_window(2);
    tracker.map_window(2, 800, 600);
    tracker.set_role(2, WindowRole::ExternalOverlay);
    tracker.set_wl_surface_id(2, 300);

    tracker.determine_focus();
    assert_eq!(tracker.focus().external_overlay, Some(2));
    assert_eq!(tracker.focus().external_overlay_wl_surface_id, 300);
}

#[test]
fn overlay_opacity_tracked_in_focus_state() {
    let mut tracker = WindowTracker::new();

    tracker.add_window(1);
    tracker.map_window(1, 1920, 1080);

    tracker.add_window(2);
    tracker.map_window(2, 1920, 1080);
    tracker.set_role(2, WindowRole::Overlay);
    tracker.set_opacity(2, 0.75);

    tracker.determine_focus();
    assert_eq!(tracker.focus().overlay_opacity, 0.75);
}

#[test]
fn overlay_input_focus_mode_tracked() {
    let mut tracker = WindowTracker::new();

    tracker.add_window(1);
    tracker.map_window(1, 1920, 1080);

    tracker.add_window(2);
    tracker.map_window(2, 1920, 1080);
    tracker.set_role(2, WindowRole::Overlay);
    if let Some(win) = tracker.get_mut(2) {
        win.input_focus_mode = 1;
    }

    tracker.determine_focus();
    assert_eq!(tracker.focus().overlay_input_focus_mode, 1);
}

#[test]
fn highest_opacity_overlay_wins() {
    let mut tracker = WindowTracker::new();

    tracker.add_window(1);
    tracker.map_window(1, 1920, 1080);

    // Two overlays — higher opacity wins.
    tracker.add_window(2);
    tracker.map_window(2, 1920, 1080);
    tracker.set_role(2, WindowRole::Overlay);
    tracker.set_opacity(2, 0.5);

    tracker.add_window(3);
    tracker.map_window(3, 1920, 1080);
    tracker.set_role(3, WindowRole::Overlay);
    tracker.set_opacity(3, 0.9);

    tracker.determine_focus();
    assert_eq!(tracker.focus().overlay, Some(3));
    assert_eq!(tracker.focus().overlay_opacity, 0.9);
}

#[test]
fn focus_change_detected_on_overlay_surface_id_update() {
    let mut tracker = WindowTracker::new();

    tracker.add_window(1);
    tracker.map_window(1, 1920, 1080);

    tracker.add_window(2);
    tracker.map_window(2, 1920, 1080);
    tracker.set_role(2, WindowRole::Overlay);

    let changed = tracker.determine_focus();
    assert!(changed);

    // Same state — no change.
    tracker.mark_focus_dirty();
    let changed = tracker.determine_focus();
    assert!(!changed);

    // Update overlay wl_surface_id — should detect change.
    tracker.set_wl_surface_id(2, 999);
    let changed = tracker.determine_focus();
    assert!(changed);
    assert_eq!(tracker.focus().overlay_wl_surface_id, 999);
}
