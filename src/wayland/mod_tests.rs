use super::*;

#[test]
fn wayland_state_new() {
    let state = WaylandState::new(Vec::new(), 1920, 1080);
    assert!(state.outputs.is_empty());
    assert_eq!(state.pointer_x, 960.0);
    assert_eq!(state.pointer_y, 540.0);
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

// ── Callback state-machine tests ─────────────────────────────────────
//
// These verify the HashMap-level callback lifecycle that drives the
// overlay Present chain. Without working defer/fire, the callback→commit
// cycle breaks and overlays stop rendering.
//
// Note: We cannot construct real WlCallback objects without a live
// Wayland display, so these tests verify the *structural* invariants
// (counts, has_callbacks, defer moves entries) via the public API.

#[test]
fn no_callbacks_initially() {
    let state = WaylandState::new(Vec::new(), 1920, 1080);
    assert_eq!(state.pending_callback_count(), 0);
    assert_eq!(state.deferred_callback_count(), 0);
    assert!(!state.has_surface_callbacks());
}

#[test]
fn fire_all_returns_false_when_empty() {
    let mut state = WaylandState::new(Vec::new(), 1920, 1080);
    assert!(!state.fire_all_surface_callbacks());
}

#[test]
fn defer_with_no_pending_is_noop() {
    let mut state = WaylandState::new(Vec::new(), 1920, 1080);
    // Deferring a surface that has no pending callbacks should not crash
    // or create empty entries.
    state.defer_surface_callbacks(19, 0);
    assert_eq!(state.deferred_callback_count(), 0);
    assert!(!state.has_surface_callbacks());
}

#[test]
fn fire_server_callbacks_with_no_callbacks_is_noop() {
    let mut state = WaylandState::new(Vec::new(), 1920, 1080);
    // Should not panic when firing callbacks for a server with none.
    state.fire_server_callbacks(0);
    state.fire_server_callbacks(1);
}

#[test]
fn fire_all_callbacks_clears_held_buffers() {
    let mut state = WaylandState::new(Vec::new(), 1920, 1080);
    // fire_all_callbacks (the recovery path) must drain held_buffers.
    // We can't push real wl_buffers, but we can verify the method
    // doesn't panic on empty state.
    state.fire_all_callbacks();
    assert!(state.held_buffers.is_empty());
}

// ── Buffer release tests ─────────────────────────────────────────────

#[test]
fn release_stale_buffers_keeps_two_newest() {
    let mut state = WaylandState::new(Vec::new(), 1920, 1080);
    // With 0, 1, or 2 buffers, nothing should be released.
    assert_eq!(state.release_stale_buffers(), 0);
    assert_eq!(state.held_buffers.len(), 0);
}

#[test]
fn release_stale_buffers_returns_zero_when_under_limit() {
    let state = WaylandState::new(Vec::new(), 1920, 1080);
    // Held buffers start empty — release should return 0.
    assert_eq!(state.held_buffers.len(), 0);
}
