use super::*;

#[test]
fn wayland_backend_creates_with_defaults() {
    let backend = WaylandBackend::new_default();
    assert_eq!(backend.width, 1280);
    assert_eq!(backend.height, 720);
    assert!(backend.connectors().is_empty()); // Not yet initialized.
}

#[test]
fn wayland_backend_custom_config() {
    let config = WaylandConfig {
        width: 1920,
        height: 1080,
        title: "test".to_string(),
        fullscreen: true,
        use_vulkan: true,
        host_wayland_display: None,
        committed_frame_rx: None,
        cursor_rx: None,
        detected_refresh_mhz: Arc::new(AtomicU32::new(0)),
        host_dmabuf_formats: Arc::new(parking_lot::Mutex::new(std::collections::HashMap::new())),
    };
    let backend = WaylandBackend::new(config);
    assert_eq!(backend.width, 1920);
    assert_eq!(backend.height, 1080);
}

#[test]
fn wayland_backend_no_direct_scanout() {
    let mut backend = WaylandBackend::new_default();
    let _ = backend.init();
    let fb = Framebuffer {
        handle: unsafe { std::mem::transmute::<u32, drm::control::framebuffer::Handle>(1u32) },
        format: DrmFourcc::Argb8888,
        modifier: DrmModifier::Linear,
        size: (1280, 720),
    };
    // Nested mode never supports direct scanout.
    assert!(!backend.try_direct_scanout(&fb).unwrap());
}

#[test]
fn wayland_backend_caps_are_limited() {
    let mut backend = WaylandBackend::new_default();
    let _ = backend.init();
    let caps = backend.capabilities();
    assert!(!caps.vrr);
    assert!(!caps.hdr);
    assert!(!caps.tearing);
    assert!(!caps.explicit_sync);
}

#[test]
fn wayland_backend_present_increments_frame() {
    let mut backend = WaylandBackend::new_default();
    let _ = backend.init();
    let fb = Framebuffer {
        handle: unsafe { std::mem::transmute::<u32, drm::control::framebuffer::Handle>(1u32) },
        format: DrmFourcc::Argb8888,
        modifier: DrmModifier::Linear,
        size: (1280, 720),
    };
    let result = backend.present(&fb);
    assert!(matches!(result.unwrap(), FlipResult::Queued));
    assert_eq!(backend.frame_count, 1);
}
