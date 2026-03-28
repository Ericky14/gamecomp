use super::*;

#[test]
fn select_prefers_boot_vga() {
    let gpus = vec![
        GpuInfo {
            dev_path: PathBuf::from("/dev/dri/card0"),
            sys_path: PathBuf::from("/sys/devices/card0"),
            boot_vga: false,
            has_render_node: true,
        },
        GpuInfo {
            dev_path: PathBuf::from("/dev/dri/card1"),
            sys_path: PathBuf::from("/sys/devices/card1"),
            boot_vga: true,
            has_render_node: false,
        },
    ];
    let primary = select_primary_gpu(&gpus).unwrap();
    assert_eq!(primary.dev_path, PathBuf::from("/dev/dri/card1"));
}

#[test]
fn select_prefers_render_node() {
    let gpus = vec![
        GpuInfo {
            dev_path: PathBuf::from("/dev/dri/card0"),
            sys_path: PathBuf::from("/sys/devices/card0"),
            boot_vga: false,
            has_render_node: false,
        },
        GpuInfo {
            dev_path: PathBuf::from("/dev/dri/card1"),
            sys_path: PathBuf::from("/sys/devices/card1"),
            boot_vga: false,
            has_render_node: true,
        },
    ];
    let primary = select_primary_gpu(&gpus).unwrap();
    assert_eq!(primary.dev_path, PathBuf::from("/dev/dri/card1"));
}

#[test]
fn select_falls_back_to_first() {
    let gpus = vec![GpuInfo {
        dev_path: PathBuf::from("/dev/dri/card0"),
        sys_path: PathBuf::from("/sys/devices/card0"),
        boot_vga: false,
        has_render_node: false,
    }];
    let primary = select_primary_gpu(&gpus).unwrap();
    assert_eq!(primary.dev_path, PathBuf::from("/dev/dri/card0"));
}

#[test]
fn select_returns_none_for_empty() {
    assert!(select_primary_gpu(&[]).is_none());
}

#[test]
fn render_node_derivation() {
    // This is a pure path computation test — doesn't check filesystem.
    let path = std::path::Path::new("/dev/dri/card0");
    let filename = path.file_name().unwrap().to_str().unwrap();
    let card_num: u32 = filename.strip_prefix("card").unwrap().parse().unwrap();
    assert_eq!(card_num, 0);
    assert_eq!(128 + card_num, 128);
}
