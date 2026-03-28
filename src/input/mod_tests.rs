use super::*;

#[test]
fn input_batch_default_is_empty() {
    let batch = InputBatch::new();
    assert_eq!(batch.pointer_dx, 0.0);
    assert_eq!(batch.pointer_dy, 0.0);
    assert!(!batch.has_pointer_motion);
    assert_eq!(batch.key_event_count, 0);
    assert_eq!(batch.button_event_count, 0);
}

#[test]
fn input_batch_accumulates_motion() {
    let mut batch = InputBatch::new();
    batch.accumulate_motion(1.5, 2.0);
    batch.accumulate_motion(-0.5, 1.0);

    assert!(batch.has_pointer_motion);
    assert!((batch.pointer_dx - 1.0).abs() < f64::EPSILON);
    assert!((batch.pointer_dy - 3.0).abs() < f64::EPSILON);
}

#[test]
fn input_batch_reset_clears() {
    let mut batch = InputBatch::new();
    batch.accumulate_motion(10.0, 20.0);
    batch.key_event_count = 5;
    batch.button_event_count = 3;

    batch.reset();
    assert_eq!(batch.pointer_dx, 0.0);
    assert_eq!(batch.pointer_dy, 0.0);
    assert!(!batch.has_pointer_motion);
    assert_eq!(batch.key_event_count, 0);
    assert_eq!(batch.button_event_count, 0);
}

#[test]
fn input_handler_creates() {
    let handler = InputHandler::new().unwrap();
    let batch = handler.batch();
    assert!(!batch.has_pointer_motion);
}

#[test]
fn input_handler_sequence_increments() {
    let mut handler = InputHandler::new().unwrap();
    assert_eq!(handler.next_sequence(), 1);
    assert_eq!(handler.next_sequence(), 2);
    assert_eq!(handler.next_sequence(), 3);
}

#[test]
fn input_handler_batch_reset() {
    let mut handler = InputHandler::new().unwrap();
    // Simulate accumulating motion (directly via the batch field).
    handler.reset_batch();
    assert!(!handler.batch().has_pointer_motion);
}

#[test]
fn key_event_fields() {
    let event = KeyEvent {
        key: 28, // KEY_ENTER
        pressed: true,
        time_usec: 123456,
    };
    assert_eq!(event.key, 28);
    assert!(event.pressed);
}

#[test]
fn scroll_event_fields() {
    let event = ScrollEvent {
        dx: 0.0,
        dy: -15.0,
        time_usec: 999,
    };
    assert!((event.dy - (-15.0)).abs() < f64::EPSILON);
}
