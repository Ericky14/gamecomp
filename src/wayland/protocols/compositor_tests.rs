use super::*;

// ── CommitTarget classification tests ────────────────────────────────

fn gate(input: CommitGateInput) -> Option<CommitTarget> {
    classify_commit(&input)
}

#[allow(clippy::too_many_arguments)]
fn input(
    surface_id: u32,
    surface_srv: u32,
    focused_id: u32,
    focused_srv: u32,
    overlay_id: u32,
    overlay_srv: u32,
    ext_overlay_id: u32,
    ext_overlay_srv: u32,
) -> CommitGateInput {
    CommitGateInput {
        surface_id,
        surface_srv,
        focused_id,
        focused_srv,
        overlay_id,
        overlay_srv,
        ext_overlay_id,
        ext_overlay_srv,
    }
}

#[test]
fn focused_app_matches() {
    assert_eq!(
        gate(input(11, 1, 11, 1, 0, 0, 0, 0)),
        Some(CommitTarget::App)
    );
}

#[test]
fn overlay_matches() {
    // Game focused on server 1, overlay on server 0.
    assert_eq!(
        gate(input(19, 0, 11, 1, 19, 0, 0, 0)),
        Some(CommitTarget::Overlay)
    );
}

#[test]
fn external_overlay_matches() {
    assert_eq!(
        gate(input(25, 0, 11, 1, 0, 0, 25, 0)),
        Some(CommitTarget::ExternalOverlay)
    );
}

#[test]
fn server0_unfocused_is_passthrough() {
    // Surface 19 on server 0, game focused on server 1, no overlay set.
    // Must be PassThrough — never rejected.
    assert_eq!(
        gate(input(19, 0, 11, 1, 0, 0, 0, 0)),
        Some(CommitTarget::PassThrough)
    );
}

#[test]
fn server0_always_accepted_even_with_no_focus() {
    // Nothing focused at all. Server 0 surface must still be accepted.
    assert_eq!(
        gate(input(5, 0, 0, u32::MAX, 0, 0, 0, 0)),
        Some(CommitTarget::PassThrough)
    );
}

#[test]
fn non_server0_unfocused_rejected() {
    // Surface from server 2, game focused on server 1, no overlay.
    assert_eq!(gate(input(30, 2, 11, 1, 0, 0, 0, 0)), None);
}

#[test]
fn overlay_takes_priority_over_passthrough_on_server0() {
    // Surface IS the overlay AND on server 0.
    // Must resolve to Overlay, not PassThrough.
    assert_eq!(
        gate(input(19, 0, 11, 1, 19, 0, 0, 0)),
        Some(CommitTarget::Overlay)
    );
}

#[test]
fn app_takes_priority_over_passthrough_on_server0() {
    // Surface IS the focused app AND on server 0 (Grid is focused).
    assert_eq!(
        gate(input(11, 0, 11, 0, 0, 0, 0, 0)),
        Some(CommitTarget::App)
    );
}

#[test]
fn focused_id_zero_means_no_app() {
    // focused_id == 0 → no app focused. Surface on server 1 rejected.
    assert_eq!(gate(input(11, 1, 0, 1, 0, 0, 0, 0)), None);
}

// ── Overlay regression: server 0 must NEVER be rejected ──────────────
//
// This is the core invariant that caused 16 failed attempts. If server 0
// commits are rejected, their syncobj release points are never signaled,
// killing the XWayland Present chain.

#[test]
fn regression_server0_never_rejected_during_game_focus() {
    // Game on server 1 focused, no overlay active.
    // Surface 19 on server 0 (Grid's main window) must be PassThrough.
    let result = gate(input(19, 0, 11, 1, 0, 0, 0, 0));
    assert!(
        result.is_some(),
        "server 0 surface must NEVER be rejected — syncobj release points would be lost"
    );
}

#[test]
fn regression_server0_never_rejected_with_overlay_on_different_surface() {
    // Game on server 1, overlay is surface 19 on server 0.
    // A DIFFERENT server 0 surface (e.g. surface 5, a popup) must still
    // be accepted as PassThrough, not rejected.
    let result = gate(input(5, 0, 11, 1, 19, 0, 0, 0));
    assert!(
        result.is_some(),
        "all server 0 surfaces must be accepted — either as Overlay or PassThrough"
    );
    assert_eq!(result, Some(CommitTarget::PassThrough));
}

#[test]
fn regression_all_server0_surfaces_accepted_exhaustive() {
    // Test with many different surface IDs on server 0.
    // Every single one must be accepted (Some), never rejected (None).
    for surface_id in [1, 5, 11, 19, 25, 42, 100, 999] {
        for &(foc_id, foc_srv) in &[(11, 1), (0, u32::MAX), (19, 0)] {
            let result = gate(input(surface_id, 0, foc_id, foc_srv, 0, 0, 0, 0));
            assert!(
                result.is_some(),
                "server 0 surface {} rejected with focus=({},{})",
                surface_id,
                foc_id,
                foc_srv
            );
        }
    }
}

// ── CommitTarget determines syncobj handling ─────────────────────────
//
// Document the invariant: PassThrough and rejected paths must signal
// release points. App/Overlay/ExternalOverlay paths stage the buffer
// and the render thread signals release on present.

#[test]
fn passthrough_and_rejected_require_immediate_release_signal() {
    // This is a documentation test — the actual signaling happens in the
    // Dispatch impl. But we verify the classification is correct so the
    // right code path runs.
    //
    // PassThrough path: signals syncobj release immediately
    // Rejected (None): signals syncobj release immediately
    // App/Overlay/ExtOverlay: release handled by render thread

    // Server 0 non-overlay → PassThrough → must signal release
    assert_eq!(
        gate(input(19, 0, 11, 1, 0, 0, 0, 0)),
        Some(CommitTarget::PassThrough),
    );

    // Server 2 non-matching → None → must signal release
    assert_eq!(gate(input(30, 2, 11, 1, 0, 0, 0, 0)), None);
}
