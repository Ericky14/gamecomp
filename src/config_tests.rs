use super::*;

#[test]
fn default_config_values() {
    let config = Config::default();
    assert_eq!(config.backend, BackendKind::Drm);
    assert!(config.vrr);
    assert!(!config.hdr);
    assert_eq!(config.upscale, UpscaleMode::None);
    assert_eq!(config.xwayland_count, 1);
    assert_eq!(config.red_zone_us, 1500);
    assert!(config.child_command.is_none());
    assert!(config.resolution.is_none());
}

#[test]
fn parse_backend_headless() {
    let args = vec!["gamecomp".into(), "--backend".into(), "headless".into()];
    let config = Config::from_args(args.into_iter());
    assert_eq!(config.backend, BackendKind::Headless);
}

#[test]
fn parse_backend_wayland() {
    let args = vec!["gamecomp".into(), "--backend".into(), "wayland".into()];
    let config = Config::from_args(args.into_iter());
    assert_eq!(config.backend, BackendKind::Wayland);
}

#[test]
fn parse_wayland_shorthand() {
    let args = vec!["gamecomp".into(), "--wayland".into()];
    let config = Config::from_args(args.into_iter());
    assert_eq!(config.backend, BackendKind::Wayland);
}

#[test]
fn parse_resolution() {
    let args = vec![
        "gamecomp".into(),
        "--width".into(),
        "1920".into(),
        "--height".into(),
        "1080".into(),
    ];
    let config = Config::from_args(args.into_iter());
    assert_eq!(config.resolution, Some((1920, 1080)));
}

#[test]
fn parse_refresh_rate() {
    let args = vec!["gamecomp".into(), "--refresh-rate".into(), "144".into()];
    let config = Config::from_args(args.into_iter());
    assert_eq!(config.refresh_rate, Some(144));
}

#[test]
fn parse_vrr_flags() {
    let args = vec!["gamecomp".into(), "--no-vrr".into()];
    let config = Config::from_args(args.into_iter());
    assert!(!config.vrr);

    let args = vec!["gamecomp".into(), "--no-vrr".into(), "--vrr".into()];
    let config = Config::from_args(args.into_iter());
    assert!(config.vrr);
}

#[test]
fn parse_upscale_modes() {
    for (arg, expected) in [
        ("fsr", UpscaleMode::Fsr),
        ("nis", UpscaleMode::Nis),
        ("cas", UpscaleMode::Cas),
        ("none", UpscaleMode::None),
    ] {
        let args = vec!["gamecomp".into(), "--upscale".into(), arg.into()];
        let config = Config::from_args(args.into_iter());
        assert_eq!(config.upscale, expected, "failed for upscale mode: {}", arg);
    }
}

#[test]
fn parse_child_command_after_separator() {
    let args = vec![
        "gamecomp".into(),
        "--".into(),
        "my-game".into(),
        "--fullscreen".into(),
    ];
    let config = Config::from_args(args.into_iter());
    assert_eq!(
        config.child_command,
        Some("my-game --fullscreen".to_string())
    );
}

#[test]
fn parse_output_selector() {
    let args = vec!["gamecomp".into(), "--output".into(), "HDMI-A-1".into()];
    let config = Config::from_args(args.into_iter());
    assert_eq!(config.preferred_output, Some("HDMI-A-1".to_string()));
}

#[test]
fn parse_log_level() {
    let args = vec!["gamecomp".into(), "--log".into(), "debug".into()];
    let config = Config::from_args(args.into_iter());
    assert_eq!(config.log_level, "debug");
}

#[test]
fn apply_file_overrides() {
    let mut config = Config::default();
    let file = ConfigFile {
        backend: Some("nested".into()),
        width: Some(3840),
        height: Some(2160),
        refresh_rate: Some(120),
        vrr: Some(false),
        hdr: Some(true),
        upscale: Some("fsr".into()),
        ..Default::default()
    };
    config.apply_file(&file);
    assert_eq!(config.backend, BackendKind::Wayland);
    assert_eq!(config.resolution, Some((3840, 2160)));
    assert_eq!(config.refresh_rate, Some(120));
    assert!(!config.vrr);
    assert!(config.hdr);
    assert_eq!(config.upscale, UpscaleMode::Fsr);
}

#[test]
fn backend_kind_default_is_drm() {
    assert_eq!(BackendKind::default(), BackendKind::Drm);
}

#[test]
fn upscale_mode_default_is_none() {
    assert_eq!(UpscaleMode::default(), UpscaleMode::None);
}
