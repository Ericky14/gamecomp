//! Tests for keyboard hotkey detection and modifier tracking.

use super::*;

fn make_key_event(code: u16, value: i32) -> InputEvent {
    InputEvent {
        tv_sec: 0,
        tv_usec: 0,
        type_: EV_KEY,
        code,
        value,
    }
}

/// Helper: process a key event and return collected actions.
fn process(km: &mut KeyboardMonitor, code: u16, value: i32) -> Vec<KeyAction> {
    let mut actions = Vec::new();
    km.process_event(&make_key_event(code, value), &mut actions);
    actions
}

/// Helper: check that processing a key event produces a specific VT switch.
fn expect_vt_switch(km: &mut KeyboardMonitor, code: u16, vt: i32) {
    let actions = process(km, code, 1);
    assert!(
        actions.contains(&KeyAction::SwitchVt(vt)),
        "expected SwitchVt({vt}), got {actions:?}"
    );
}

/// Helper: check that processing a key event does NOT produce a VT switch.
fn expect_no_vt_switch(km: &mut KeyboardMonitor, code: u16, value: i32) {
    let actions = process(km, code, value);
    assert!(
        !actions.iter().any(|a| matches!(a, KeyAction::SwitchVt(_))),
        "unexpected VT switch in {actions:?}"
    );
}

#[test]
fn detects_ctrl_alt_f1() {
    let mut km = KeyboardMonitor::new();
    // Press Ctrl.
    expect_no_vt_switch(&mut km, KEY_LEFTCTRL, 1);
    // Press Alt.
    expect_no_vt_switch(&mut km, KEY_LEFTALT, 1);
    // Press F1.
    expect_vt_switch(&mut km, KEY_F1, 1);
}

#[test]
fn detects_ctrl_alt_f12() {
    let mut km = KeyboardMonitor::new();
    process(&mut km, KEY_RIGHTCTRL, 1);
    process(&mut km, KEY_RIGHTALT, 1);
    expect_vt_switch(&mut km, KEY_F12, 12);
}

#[test]
fn no_action_without_modifiers() {
    let mut km = KeyboardMonitor::new();
    expect_no_vt_switch(&mut km, KEY_F1, 1);
}

#[test]
fn no_action_ctrl_only() {
    let mut km = KeyboardMonitor::new();
    process(&mut km, KEY_LEFTCTRL, 1);
    expect_no_vt_switch(&mut km, KEY_F1, 1);
}

#[test]
fn no_action_on_release() {
    let mut km = KeyboardMonitor::new();
    process(&mut km, KEY_LEFTCTRL, 1);
    process(&mut km, KEY_LEFTALT, 1);
    // Release F1 (value=0) should NOT trigger VT switch.
    expect_no_vt_switch(&mut km, KEY_F1, 0);
}

#[test]
fn modifier_release_clears_state() {
    let mut km = KeyboardMonitor::new();
    process(&mut km, KEY_LEFTCTRL, 1);
    process(&mut km, KEY_LEFTALT, 1);
    // Release Ctrl.
    process(&mut km, KEY_LEFTCTRL, 0);
    // F1 should no longer trigger.
    expect_no_vt_switch(&mut km, KEY_F1, 1);
}

#[test]
fn repeat_events_ignored_for_modifiers() {
    let mut km = KeyboardMonitor::new();
    process(&mut km, KEY_LEFTCTRL, 1);
    process(&mut km, KEY_LEFTALT, 1);
    // Repeat event (value=2) on Ctrl — should not clear state.
    process(&mut km, KEY_LEFTCTRL, 2);
    expect_vt_switch(&mut km, KEY_F1, 1);
}

#[test]
fn mixed_left_right_modifiers() {
    let mut km = KeyboardMonitor::new();
    process(&mut km, KEY_LEFTCTRL, 1);
    process(&mut km, KEY_RIGHTALT, 1);
    expect_vt_switch(&mut km, KEY_F1 + 2, 3);
}

#[test]
fn reset_modifiers_clears_stuck_state() {
    let mut km = KeyboardMonitor::new();
    // Simulate Ctrl+Alt held (as would happen before VT switch).
    process(&mut km, KEY_LEFTCTRL, 1);
    process(&mut km, KEY_LEFTALT, 1);
    assert!(km.ctrl_held);
    assert!(km.alt_held);

    // Session restore resets modifiers.
    km.reset_modifiers();
    assert!(!km.ctrl_held);
    assert!(!km.alt_held);
    assert!(!km.left_ctrl);
    assert!(!km.left_alt);

    // F1 alone should NOT trigger VT switch after reset.
    expect_no_vt_switch(&mut km, KEY_F1, 1);

    // But re-pressing Ctrl+Alt+F1 should work.
    process(&mut km, KEY_LEFTCTRL, 1);
    process(&mut km, KEY_LEFTALT, 1);
    expect_vt_switch(&mut km, KEY_F1, 1);
}

#[test]
fn replace_devices_clears_old() {
    let mut km = KeyboardMonitor::new();
    // Initially no devices.
    assert!(km.devices.is_empty());

    // replace_devices with empty vec stays empty.
    km.replace_devices(vec![]);
    assert!(km.devices.is_empty());
}

#[test]
fn key_events_emitted_for_all_presses() {
    let mut km = KeyboardMonitor::new();
    let actions = process(&mut km, KEY_F1, 1);
    // Should emit a Key action for the F1 press.
    assert!(actions.iter().any(|a| matches!(
        a,
        KeyAction::Key {
            key: 59,
            pressed: true,
            ..
        }
    )));
}

#[test]
fn key_events_emitted_for_releases() {
    let mut km = KeyboardMonitor::new();
    let actions = process(&mut km, KEY_LEFTCTRL, 0);
    assert!(actions.iter().any(|a| matches!(
        a,
        KeyAction::Key {
            key: 29,
            pressed: false,
            ..
        }
    )));
}

#[test]
fn repeat_events_not_forwarded() {
    let mut km = KeyboardMonitor::new();
    // Repeat event (value=2) should produce no actions.
    let actions = process(&mut km, KEY_F1, 2);
    assert!(actions.is_empty());
}
