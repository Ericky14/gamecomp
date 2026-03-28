use super::*;

#[test]
fn headless_backend_creates() {
    let backend = HeadlessBackend::new(1920, 1080, 60);
    assert_eq!(backend.width, 1920);
    assert_eq!(backend.height, 1080);
    assert_eq!(backend.refresh_hz, 60);
    assert!(backend.connectors().is_empty());
}

#[test]
fn headless_backend_init() {
    let mut backend = HeadlessBackend::new(1280, 720, 144);
    backend.init().unwrap();

    assert_eq!(backend.connectors().len(), 1);
    assert_eq!(backend.connectors()[0].name, "HEADLESS-1");
    assert_eq!(backend.scanout_formats().len(), 4);
}

#[test]
fn headless_backend_no_vrr() {
    let mut backend = HeadlessBackend::new(800, 600, 30);
    backend.init().unwrap();
    let caps = backend.capabilities();
    assert!(!caps.vrr);
    assert!(!caps.hdr);
    assert!(!caps.tearing);
}

#[test]
fn headless_backend_always_scanout() {
    let mut backend = HeadlessBackend::new(1920, 1080, 60);
    backend.init().unwrap();

    let fb = Framebuffer {
        // SAFETY: Handle is never passed to DRM operations.
        handle: unsafe { std::mem::transmute::<u32, drm::control::framebuffer::Handle>(1u32) },
        format: DrmFourcc::Argb8888,
        modifier: DrmModifier::Linear,
        size: (1920, 1080),
    };
    // Headless always reports direct scanout success.
    assert!(backend.try_direct_scanout(&fb).unwrap());
}

#[test]
fn headless_backend_present_increments() {
    let mut backend = HeadlessBackend::new(1920, 1080, 60);
    backend.init().unwrap();

    let fb = Framebuffer {
        // SAFETY: Handle is never passed to DRM operations.
        handle: unsafe { std::mem::transmute::<u32, drm::control::framebuffer::Handle>(1u32) },
        format: DrmFourcc::Argb8888,
        modifier: DrmModifier::Linear,
        size: (1920, 1080),
    };
    backend.present(&fb).unwrap();
    backend.present(&fb).unwrap();
    assert_eq!(backend.frame_count, 2);
}

#[test]
fn headless_backend_no_drm_fd() {
    let backend = HeadlessBackend::new(1920, 1080, 60);
    assert!(backend.drm_fd().is_none());
}

#[test]
fn headless_backend_import_dmabuf() {
    let mut backend = HeadlessBackend::new(1920, 1080, 60);
    backend.init().unwrap();

    let dmabuf = DmaBuf {
        width: 1920,
        height: 1080,
        format: DrmFourcc::Argb8888,
        modifier: DrmModifier::Linear,
        planes: vec![],
    };
    let fb = backend.import_dmabuf(&dmabuf).unwrap();
    assert_eq!(fb.format, DrmFourcc::Argb8888);
    assert_eq!(fb.size, (1920, 1080));
}
