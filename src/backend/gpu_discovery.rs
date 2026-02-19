//! GPU device discovery via udev.
//!
//! Scans the system for DRM GPU devices and selects the primary GPU
//! for presentation. Supports boot_vga detection and seat filtering.

use std::ffi::OsString;
use std::path::PathBuf;

use anyhow::Context;
use tracing::{debug, info, warn};

/// Discovered GPU device information.
#[derive(Debug, Clone)]
pub struct GpuInfo {
    /// Device node path (e.g., `/dev/dri/card0`).
    pub dev_path: PathBuf,
    /// System path in sysfs.
    pub sys_path: PathBuf,
    /// Whether this is the boot GPU (`boot_vga=1` in PCI parent).
    pub boot_vga: bool,
    /// Whether a render node exists for this GPU.
    pub has_render_node: bool,
}

/// Discover all DRM GPU devices for the given seat.
///
/// Scans udev for `drm` subsystem devices matching `card[0-9]*` and
/// filters by seat assignment (defaults to `seat0`).
pub fn discover_gpus(seat: &str) -> anyhow::Result<Vec<GpuInfo>> {
    let mut enumerator = udev::Enumerator::new().context("failed to create udev enumerator")?;
    enumerator
        .match_subsystem("drm")
        .context("failed to filter drm subsystem")?;
    enumerator
        .match_sysname("card[0-9]*")
        .context("failed to filter card sysname")?;

    let devices = enumerator
        .scan_devices()
        .context("failed to scan udev devices")?;

    let mut gpus = Vec::new();
    for device in devices {
        // Filter by seat.
        let device_seat = device
            .property_value("ID_SEAT")
            .map(|v| v.to_os_string())
            .unwrap_or_else(|| OsString::from("seat0"));
        if device_seat != seat {
            debug!(
                path = ?device.devnode(),
                device_seat = ?device_seat,
                expected_seat = seat,
                "skipping GPU on different seat"
            );
            continue;
        }

        let Some(dev_path) = device.devnode().map(PathBuf::from) else {
            continue;
        };

        let sys_path = device.syspath().to_path_buf();

        // Check PCI parent for boot_vga attribute.
        let boot_vga = if let Some(pci) = device
            .parent_with_subsystem(std::path::Path::new("pci"))
            .ok()
            .flatten()
        {
            pci.attribute_value("boot_vga")
                .is_some_and(|v| v == "1")
        } else {
            false
        };

        // Check if a render node exists (renderD*).
        let has_render_node = render_node_for(&dev_path).is_some();

        let gpu = GpuInfo {
            dev_path,
            sys_path,
            boot_vga,
            has_render_node,
        };
        debug!(?gpu, "discovered GPU");
        gpus.push(gpu);
    }

    info!(count = gpus.len(), seat, "GPU discovery complete");
    Ok(gpus)
}

/// Select the primary GPU from a list of discovered GPUs.
///
/// Priority:
/// 1. Boot GPU (`boot_vga=1`)
/// 2. GPU with a render node
/// 3. First discovered GPU
pub fn select_primary_gpu(gpus: &[GpuInfo]) -> Option<&GpuInfo> {
    if gpus.is_empty() {
        warn!("no GPUs discovered");
        return None;
    }

    // Prefer boot_vga GPU.
    if let Some(gpu) = gpus.iter().find(|g| g.boot_vga) {
        info!(path = %gpu.dev_path.display(), "selected boot GPU as primary");
        return Some(gpu);
    }

    // Prefer GPU with render node.
    if let Some(gpu) = gpus.iter().find(|g| g.has_render_node) {
        info!(
            path = %gpu.dev_path.display(),
            "selected GPU with render node as primary"
        );
        return Some(gpu);
    }

    // Fallback to first.
    let gpu = &gpus[0];
    info!(
        path = %gpu.dev_path.display(),
        "selected first GPU as primary (no boot_vga or render node)"
    );
    Some(gpu)
}

/// Find the render node (e.g., `/dev/dri/renderD128`) for a card node.
///
/// Derives it from the card number: card0 → renderD128, card1 → renderD129, etc.
fn render_node_for(card_path: &std::path::Path) -> Option<PathBuf> {
    let filename = card_path.file_name()?.to_str()?;
    let card_num: u32 = filename.strip_prefix("card")?.parse().ok()?;
    let render_path = card_path
        .parent()?
        .join(format!("renderD{}", 128 + card_num));
    if render_path.exists() {
        Some(render_path)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
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
}
