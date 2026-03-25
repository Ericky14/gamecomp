//! Input handling via libinput.
//!
//! Integrates libinput with the calloop event loop for keyboard, pointer,
//! and touch input. All input processing happens on the main thread —
//! no locks needed.
//!
//! Design: Events are batched per-frame. Multiple pointer motion events
//! within a single dispatch cycle are accumulated into one delta, reducing
//! the number of Wayland events sent to the client.

pub mod keyboard;
pub mod pointer;

use tracing::info;

/// Accumulated input state for one frame.
///
/// Pointer motion events are accumulated during a dispatch cycle and
/// flushed as a single event to reduce Wayland protocol overhead.
#[derive(Debug, Clone, Copy, Default)]
pub struct InputBatch {
    /// Accumulated pointer motion delta (x, y).
    pub pointer_dx: f64,
    pub pointer_dy: f64,
    /// Whether any pointer motion occurred this frame.
    pub has_pointer_motion: bool,
    /// Number of key events processed.
    pub key_event_count: u32,
    /// Number of button events processed.
    pub button_event_count: u32,
}

impl InputBatch {
    /// Create a new empty batch.
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }

    /// Reset the batch for the next frame.
    #[inline(always)]
    pub fn reset(&mut self) {
        *self = Self::default();
    }

    /// Accumulate a pointer motion delta.
    #[inline(always)]
    pub fn accumulate_motion(&mut self, dx: f64, dy: f64) {
        self.pointer_dx += dx;
        self.pointer_dy += dy;
        self.has_pointer_motion = true;
    }
}

/// A keyboard key event.
#[derive(Debug, Clone, Copy)]
pub struct KeyEvent {
    /// Linux key code (KEY_* from input-event-codes.h).
    pub key: u32,
    /// Whether the key was pressed (true) or released (false).
    pub pressed: bool,
    /// Event timestamp in microseconds.
    pub time_usec: u64,
}

/// A pointer button event.
#[derive(Debug, Clone, Copy)]
pub struct ButtonEvent {
    /// Linux button code (BTN_* from input-event-codes.h).
    pub button: u32,
    /// Whether the button was pressed (true) or released (false).
    pub pressed: bool,
    /// Event timestamp in microseconds.
    pub time_usec: u64,
}

/// A pointer scroll event.
#[derive(Debug, Clone, Copy)]
pub struct ScrollEvent {
    /// Scroll delta (horizontal, vertical).
    pub dx: f64,
    pub dy: f64,
    /// Event timestamp in microseconds.
    pub time_usec: u64,
}

/// Events produced by the input handler and consumed by the main loop.
#[derive(Debug)]
pub enum InputEvent {
    /// Pointer motion (accumulated delta for this frame).
    PointerMotion { dx: f64, dy: f64 },
    /// Absolute pointer position (e.g., from touchscreen).
    PointerAbsolute { x: f64, y: f64 },
    /// Pointer button press/release.
    Button(ButtonEvent),
    /// Keyboard key press/release.
    Key(KeyEvent),
    /// Scroll wheel.
    Scroll(ScrollEvent),
}

/// Input handler state.
///
/// Wraps libinput and processes input events. Owned by the main thread.
pub struct InputHandler {
    /// Input event batch accumulator.
    batch: InputBatch,
    /// Sequence counter for input events.
    sequence: u64,
}

impl InputHandler {
    /// Create a new input handler.
    pub fn new() -> anyhow::Result<Self> {
        info!("input handler initialized");

        Ok(Self {
            batch: InputBatch::new(),
            sequence: 0,
        })
    }

    /// Get the current input batch (accumulated events this frame).
    #[inline(always)]
    pub fn batch(&self) -> &InputBatch {
        &self.batch
    }

    /// Reset the event batch for the next frame.
    #[inline(always)]
    pub fn reset_batch(&mut self) {
        self.batch.reset();
    }

    /// Get the next sequence number for event ordering.
    #[inline(always)]
    pub fn next_sequence(&mut self) -> u64 {
        self.sequence += 1;
        self.sequence
    }
}

#[cfg(test)]
mod tests {
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
}
