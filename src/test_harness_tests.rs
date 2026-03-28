use super::*;

#[test]
fn mock_backend_init() {
    let mut backend = MockBackend::new(1920, 1080);
    assert!(!backend.initialized);
    backend.init().unwrap();
    assert!(backend.initialized);
    assert_eq!(backend.connectors().len(), 1);
    assert_eq!(backend.connectors()[0].name, "MOCK-1");
}

#[test]
fn mock_backend_present_records() {
    let mut backend = MockBackend::new(1920, 1080);
    backend.init().unwrap();

    let fb = Framebuffer {
        // SAFETY: Mock handle is never used for real DRM operations.
        handle: unsafe { std::mem::transmute::<u32, drm::control::framebuffer::Handle>(1u32) },
        format: DrmFourcc::Argb8888,
        modifier: DrmModifier::Linear,
        size: (1920, 1080),
    };
    backend.present(&fb).unwrap();
    backend.present(&fb).unwrap();

    assert_eq!(backend.present_count(), 2);
    assert_eq!(backend.frame_count, 2);
}

#[test]
fn mock_backend_scanout_tracking() {
    let mut backend = MockBackend::new(1920, 1080);
    backend.init().unwrap();
    backend.direct_scanout_succeeds = true;

    let fb = Framebuffer {
        // SAFETY: Mock handle is never used for real DRM operations.
        handle: unsafe { std::mem::transmute::<u32, drm::control::framebuffer::Handle>(1u32) },
        format: DrmFourcc::Argb8888,
        modifier: DrmModifier::Linear,
        size: (1920, 1080),
    };

    assert!(backend.try_direct_scanout(&fb).unwrap());
    backend.direct_scanout_succeeds = false;
    assert!(!backend.try_direct_scanout(&fb).unwrap());

    assert_eq!(backend.scanout_attempt_count(), 2);
    assert_eq!(backend.scanout_success_count(), 1);
}

#[test]
fn mock_backend_reset() {
    let mut backend = MockBackend::new(1920, 1080);
    backend.init().unwrap();
    let fb = Framebuffer {
        // SAFETY: Mock handle is never used for real DRM operations.
        handle: unsafe { std::mem::transmute::<u32, drm::control::framebuffer::Handle>(1u32) },
        format: DrmFourcc::Argb8888,
        modifier: DrmModifier::Linear,
        size: (1920, 1080),
    };
    backend.present(&fb).unwrap();
    assert_eq!(backend.present_count(), 1);

    backend.reset();
    assert_eq!(backend.present_count(), 0);
    assert_eq!(backend.frame_count, 0);
}

#[test]
fn test_compositor_direct_scanout() {
    let mut comp = TestCompositor::new();
    let frame = FrameBuilder::new(1920, 1080).with_fullscreen_app(0).build();

    assert!(frame.is_direct_scanout_candidate());
    comp.submit_frame(frame).unwrap();
    comp.assert_frame_count(1);
}

#[test]
fn test_compositor_composite_with_overlay() {
    let mut comp = TestCompositor::new();
    let frame = FrameBuilder::new(1920, 1080)
        .with_fullscreen_app(0)
        .with_overlay(1, 100, 100, 300, 200, 0.8)
        .build();

    assert!(!frame.is_direct_scanout_candidate());
    assert_eq!(frame.mode, CompositionMode::Composite);
    comp.submit_frame(frame).unwrap();
    comp.assert_frame_count(1);
}

#[test]
fn test_compositor_run_frames() {
    let mut comp = TestCompositor::new();
    let results = comp.run_frames(60, |seq| {
        FrameBuilder::new(1920, 1080)
            .with_fullscreen_app(0)
            .with_seq(seq)
            .build()
    });
    assert_eq!(results.len(), 60);
    comp.assert_frame_count(60);
}

#[test]
fn frame_builder_auto_detects_mode() {
    // Single fullscreen = DirectScanout.
    let frame = FrameBuilder::new(1920, 1080).with_fullscreen_app(0).build();
    assert_eq!(frame.mode, CompositionMode::DirectScanout);

    // With overlay = Composite.
    let frame = FrameBuilder::new(1920, 1080)
        .with_fullscreen_app(0)
        .with_cursor(1, 500, 500)
        .build();
    assert_eq!(frame.mode, CompositionMode::Composite);

    // Empty = Skip.
    let frame = FrameBuilder::new(1920, 1080).build();
    assert_eq!(frame.mode, CompositionMode::Skip);
}

#[test]
fn frame_builder_layer_assertions() {
    let frame = FrameBuilder::new(1920, 1080)
        .with_fullscreen_app(0)
        .with_overlay(1, 0, 0, 200, 100, 0.5)
        .build();

    // App at LAYER_APP (1) + overlay at LAYER_OVERLAY (2) → layer_count = 3.
    assert_layer_count(&frame, 3);
    assert_layer_active(&frame, 1); // LAYER_APP
    assert_layer_active(&frame, 2); // LAYER_OVERLAY
    assert_layer_fullscreen(&frame, 1); // App covers full output
}

#[test]
fn timing_helper_works() {
    let (result, ns) = measure_ns(|| 42);
    assert_eq!(result, 42);
    assert!(ns < 1_000_000_000); // Less than 1 second.
}

#[test]
fn vrf_mode_configuration() {
    let mut backend = MockBackend::new(1920, 1080);
    backend.init().unwrap();
    assert!(!backend.vrr_enabled);
    backend.set_vrr(true).unwrap();
    assert!(backend.vrr_enabled);
}
