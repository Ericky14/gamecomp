use super::*;

#[test]
fn frame_info_is_copy() {
    fn assert_copy<T: Copy>() {}
    assert_copy::<FrameInfo>();
    assert_copy::<Layer>();
}

#[test]
fn default_frame_is_skip() {
    let frame = FrameInfo::default();
    assert_eq!(frame.mode, CompositionMode::Skip);
    assert_eq!(frame.layer_count, 0);
}

#[test]
fn single_layer_is_direct_scanout() {
    let frame = FrameInfo::single_layer(1920, 1080, 0, DrmFourcc::Argb8888);
    assert!(frame.is_direct_scanout_candidate());
    assert_eq!(frame.mode, CompositionMode::DirectScanout);
}

#[test]
fn active_layers_respects_count() {
    let mut frame = FrameInfo {
        layer_count: 2,
        ..Default::default()
    };
    frame.layers[0].active = true;
    frame.layers[1].active = true;
    assert_eq!(frame.active_layers().len(), 2);
}

#[test]
fn frame_info_size_is_small() {
    // Ensure FrameInfo stays small enough to send via channel efficiently.
    let size = std::mem::size_of::<FrameInfo>();
    assert!(size < 512, "FrameInfo is {} bytes, should be < 512", size);
}
