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

#[test]
fn detects_ctrl_alt_f1() {
    let mut km = KeyboardMonitor::new();
    // Press Ctrl.
    assert!(km.process_event(&make_key_event(KEY_LEFTCTRL, 1)).is_none());
    // Press Alt.
    assert!(km.process_event(&make_key_event(KEY_LEFTALT, 1)).is_none());
    // Press F1.
    let action = km.process_event(&make_key_event(KEY_F1, 1));
    assert_eq!(action, Some(KeyAction::SwitchVt(1)));
}

#[test]
fn detects_ctrl_alt_f12() {
    let mut km = KeyboardMonitor::new();
    km.process_event(&make_key_event(KEY_RIGHTCTRL, 1));
    km.process_event(&make_key_event(KEY_RIGHTALT, 1));
    let action = km.process_event(&make_key_event(KEY_F12, 1));
    assert_eq!(action, Some(KeyAction::SwitchVt(12)));
}

#[test]
fn no_action_without_modifiers() {
    let mut km = KeyboardMonitor::new();
    let action = km.process_event(&make_key_event(KEY_F1, 1));
    assert!(action.is_none());
}

#[test]
fn no_action_ctrl_only() {
    let mut km = KeyboardMonitor::new();
    km.process_event(&make_key_event(KEY_LEFTCTRL, 1));
    let action = km.process_event(&make_key_event(KEY_F1, 1));
    assert!(action.is_none());
}

#[test]
fn no_action_on_release() {
    let mut km = KeyboardMonitor::new();
    km.process_event(&make_key_event(KEY_LEFTCTRL, 1));
    km.process_event(&make_key_event(KEY_LEFTALT, 1));
    // Release F1 (value=0) should NOT trigger.
    let action = km.process_event(&make_key_event(KEY_F1, 0));
    assert!(action.is_none());
}

#[test]
fn modifier_release_clears_state() {
    let mut km = KeyboardMonitor::new();
    km.process_event(&make_key_event(KEY_LEFTCTRL, 1));
    km.process_event(&make_key_event(KEY_LEFTALT, 1));
    // Release Ctrl.
    km.process_event(&make_key_event(KEY_LEFTCTRL, 0));
    // F1 should no longer trigger.
    let action = km.process_event(&make_key_event(KEY_F1, 1));
    assert!(action.is_none());
}

#[test]
fn repeat_events_ignored_for_modifiers() {
    let mut km = KeyboardMonitor::new();
    km.process_event(&make_key_event(KEY_LEFTCTRL, 1));
    km.process_event(&make_key_event(KEY_LEFTALT, 1));
    // Repeat event (value=2) on Ctrl — should not clear state.
    km.process_event(&make_key_event(KEY_LEFTCTRL, 2));
    let action = km.process_event(&make_key_event(KEY_F1, 1));
    assert_eq!(action, Some(KeyAction::SwitchVt(1)));
}

#[test]
fn mixed_left_right_modifiers() {
    let mut km = KeyboardMonitor::new();
    km.process_event(&make_key_event(KEY_LEFTCTRL, 1));
    km.process_event(&make_key_event(KEY_RIGHTALT, 1));
    let action = km.process_event(&make_key_event(KEY_F1 + 2, 1));
    assert_eq!(action, Some(KeyAction::SwitchVt(3)));
}

#[test]
fn reset_modifiers_clears_stuck_state() {
    let mut km = KeyboardMonitor::new();
    // Simulate Ctrl+Alt held (as would happen before VT switch).
    km.process_event(&make_key_event(KEY_LEFTCTRL, 1));
    km.process_event(&make_key_event(KEY_LEFTALT, 1));
    assert!(km.ctrl_held);
    assert!(km.alt_held);

    // Session restore resets modifiers.
    km.reset_modifiers();
    assert!(!km.ctrl_held);
    assert!(!km.alt_held);
    assert!(!km.left_ctrl);
    assert!(!km.left_alt);

    // F1 alone should NOT trigger VT switch after reset.
    let action = km.process_event(&make_key_event(KEY_F1, 1));
    assert!(action.is_none());

    // But re-pressing Ctrl+Alt+F1 should work.
    km.process_event(&make_key_event(KEY_LEFTCTRL, 1));
    km.process_event(&make_key_event(KEY_LEFTALT, 1));
    let action = km.process_event(&make_key_event(KEY_F1, 1));
    assert_eq!(action, Some(KeyAction::SwitchVt(1)));
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
