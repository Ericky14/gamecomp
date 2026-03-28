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
