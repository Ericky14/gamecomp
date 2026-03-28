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
