use super::*;

fn make_server(index: u32, app_id: u32, surface_id: u32) -> ServerFocusState {
    ServerFocusState {
        index,
        focused_app_id: Arc::new(AtomicU32::new(app_id)),
        focused_wl_surface_id: Arc::new(AtomicU32::new(surface_id)),
    }
}

#[test]
fn no_servers_yields_no_focus() {
    let mut arbiter = FocusArbiter::new(0);
    let result = arbiter.update(&[]);
    assert_eq!(result.app_id, 0);
    assert_eq!(result.server_index, u32::MAX);
}

#[test]
fn single_server_with_app_gets_focus() {
    let servers = [make_server(0, 769, 10)];
    let mut arbiter = FocusArbiter::new(1);
    let result = arbiter.update(&servers);
    assert_eq!(result.app_id, 769);
    assert_eq!(result.surface_id, 10);
    assert_eq!(result.server_index, 0);
    assert!(result.changed);
}

#[test]
fn stealer_wins_over_existing() {
    let servers = [make_server(0, 769, 10), make_server(1, 0, 0)];
    let mut arbiter = FocusArbiter::new(2);

    // First tick: server 0 wins.
    let r = arbiter.update(&servers);
    assert_eq!(r.server_index, 0);

    // Server 1 gets a new app — it steals focus.
    servers[1].focused_app_id.store(100, Ordering::Relaxed);
    servers[1]
        .focused_wl_surface_id
        .store(20, Ordering::Relaxed);
    let r = arbiter.update(&servers);
    assert_eq!(r.app_id, 100);
    assert_eq!(r.server_index, 1);
    assert!(r.changed);
}

#[test]
fn current_winner_retained_when_no_stealer() {
    let servers = [make_server(0, 769, 10), make_server(1, 100, 20)];
    let mut arbiter = FocusArbiter::new(2);

    // First tick: both are new, last stealer (server 1) wins.
    let r = arbiter.update(&servers);
    assert_eq!(r.server_index, 1);

    // No changes — server 1 retained.
    let r = arbiter.update(&servers);
    assert_eq!(r.server_index, 1);
    assert!(!r.changed);
}

#[test]
fn fallback_when_winner_loses_focus() {
    let servers = [make_server(0, 769, 10), make_server(1, 100, 20)];
    let mut arbiter = FocusArbiter::new(2);
    arbiter.update(&servers);

    // Server 1 loses focus.
    servers[1].focused_app_id.store(0, Ordering::Relaxed);
    let r = arbiter.update(&servers);
    assert_eq!(r.server_index, 0);
    assert_eq!(r.app_id, 769);
    assert!(r.changed);
}

#[test]
fn baselayer_overrides_stealer() {
    let servers = [make_server(0, 769, 10), make_server(1, 100, 20)];
    let mut arbiter = FocusArbiter::new(2);
    arbiter.update(&servers);

    // Set baselayer to prefer app 769 (server 0).
    arbiter.baselayer_app_ids = vec![769];
    let r = arbiter.update(&servers);
    assert_eq!(r.server_index, 0);
    assert_eq!(r.app_id, 769);
}

#[test]
fn clearing_baselayer_triggers_recompete() {
    let servers = [make_server(0, 769, 10), make_server(1, 100, 20)];
    let mut arbiter = FocusArbiter::new(2);

    // Baselayer pins server 0.
    arbiter.baselayer_app_ids = vec![769];
    arbiter.update(&servers);
    assert_eq!(arbiter.prev_server_index, 0);

    // Clear baselayer — prev_server_app_ids reset forces recompete.
    arbiter.baselayer_app_ids.clear();
    arbiter.prev_server_app_ids.fill(0);
    let r = arbiter.update(&servers);
    // Both servers look "new" — last stealer (server 1) wins.
    assert_eq!(r.server_index, 1);
    assert!(r.changed);
}
