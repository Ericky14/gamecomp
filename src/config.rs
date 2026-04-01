//! Configuration for the compositor.
//!
//! Supports two configuration sources merged with precedence:
//! 1. CLI arguments (highest priority)
//! 2. TOML config file (`$XDG_CONFIG_HOME/gamecomp/config.toml`)
//!
//! All configuration values have sensible defaults. The compositor can run
//! with zero configuration.

use std::path::PathBuf;

use serde::Deserialize;
use tracing::info;

/// Complete compositor configuration.
#[derive(Debug, Clone)]
pub struct Config {
    /// Display backend to use.
    pub backend: BackendKind,
    /// Output resolution override (width, height). None = use native mode.
    pub resolution: Option<(u32, u32)>,
    /// Refresh rate override in Hz. None = use native rate.
    pub refresh_rate: Option<u32>,
    /// Preferred output connector name (e.g., "eDP-1"). None = auto-select.
    pub preferred_output: Option<String>,
    /// Whether to enable VRR if the display supports it.
    pub vrr: bool,
    /// Whether to enable HDR if the display supports it.
    pub hdr: bool,
    /// Upscaling mode.
    pub upscale: UpscaleMode,
    /// Number of XWayland instances to spawn.
    pub xwayland_count: u32,

    /// Frame pacer red zone in microseconds.
    pub red_zone_us: u64,
    /// Game render resolution (`-w`×`-h`). This is the resolution
    /// advertised to Wayland/XWayland clients — the size the game
    /// actually renders at. When set, it stays fixed regardless of
    /// window resizing (the compositor scales to the output).
    /// If `None`, falls back to `resolution`.
    pub game_resolution: Option<(u32, u32)>,
    /// FPS limit. 0 = match display refresh rate (no explicit cap).
    pub fps_limit: u32,
    /// Cursor hide delay in milliseconds. 0 = never hide.
    pub cursor_hide_delay_ms: u64,
    /// Steam mode: use PID-based AppID resolution (gamescope `-e`).
    /// When enabled, windows get their AppID by walking the parent
    /// process chain for a Steam reaper process (`SteamLaunch AppId=N`).
    /// Without this, AppID falls back to the X11 window ID.
    pub steam_mode: bool,
    /// Child command to launch inside the compositor.
    pub child_command: Option<String>,
    /// Stats pipe path. None = disabled.
    pub stats_pipe: Option<PathBuf>,
    /// Log level (RUST_LOG format).
    pub log_level: String,
}

/// Which display backend to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BackendKind {
    /// Direct DRM/KMS output (production).
    #[default]
    Drm,
    /// Headless (no output, for CI/robotics).
    Headless,
    /// Wayland mode (run inside another Wayland compositor as a window).
    Wayland,
}

/// Upscaling algorithm.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum UpscaleMode {
    /// No upscaling — render at native resolution.
    #[default]
    None,
    /// AMD FidelityFX Super Resolution 1.0.
    Fsr,
    /// NVIDIA Image Scaling.
    Nis,
    /// Contrast Adaptive Sharpening (sharpening only, no upscale).
    Cas,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            backend: BackendKind::Drm,
            resolution: None,
            refresh_rate: None,
            preferred_output: None,
            vrr: true,
            hdr: false,
            upscale: UpscaleMode::None,
            xwayland_count: 1,
            red_zone_us: 1500,
            game_resolution: None,
            fps_limit: 0,
            cursor_hide_delay_ms: 3000,
            steam_mode: true,
            child_command: None,
            stats_pipe: None,
            log_level: "info".to_string(),
        }
    }
}

impl Config {
    /// Build a `WaylandConfig` from the compositor config.
    ///
    /// Maps `--width`/`--height` (resolution) to the Wayland backend window size.
    pub fn to_wayland_config(&self) -> crate::backend::wayland::WaylandConfig {
        let (width, height) = self.resolution.unwrap_or((1280, 720));
        crate::backend::wayland::WaylandConfig {
            width,
            height,
            title: "gamecomp".to_string(),
            fullscreen: false,
            use_vulkan: true,
            host_wayland_display: None,
            committed_frame_rx: None,
            cursor_rx: None,
            detected_refresh_mhz: std::sync::Arc::new(std::sync::atomic::AtomicU32::new(0)),
            host_dmabuf_formats: std::sync::Arc::new(parking_lot::Mutex::new(
                std::collections::HashMap::new(),
            )),
        }
    }
}

/// TOML config file structure.
#[derive(Debug, Deserialize, Default)]
struct ConfigFile {
    #[serde(default)]
    backend: Option<String>,
    #[serde(default)]
    width: Option<u32>,
    #[serde(default)]
    height: Option<u32>,
    #[serde(default)]
    refresh_rate: Option<u32>,
    #[serde(default)]
    preferred_output: Option<String>,
    #[serde(default)]
    vrr: Option<bool>,
    #[serde(default)]
    hdr: Option<bool>,
    #[serde(default)]
    upscale: Option<String>,
    #[serde(default)]
    xwayland_count: Option<u32>,
    #[serde(default)]
    red_zone_us: Option<u64>,
    #[serde(default)]
    fps_limit: Option<u32>,
    #[serde(default)]
    cursor_hide_delay_ms: Option<u64>,
    #[serde(default)]
    stats_pipe: Option<String>,
    #[serde(default)]
    log_level: Option<String>,
    #[serde(default)]
    steam_mode: Option<bool>,
}

impl Config {
    /// Parse configuration from CLI arguments and optional config file.
    pub fn from_args(args: impl Iterator<Item = String>) -> Self {
        let mut config = Self::default();
        let args: Vec<String> = args.collect();

        // Try loading config file first (lowest priority).
        if let Some(config_dir) = dirs_config_path() {
            let config_path = config_dir.join("gamecomp").join("config.toml");
            if config_path.exists() {
                match std::fs::read_to_string(&config_path) {
                    Ok(contents) => {
                        if let Ok(file_config) = toml::from_str::<ConfigFile>(&contents) {
                            config.apply_file(&file_config);
                            info!(path = %config_path.display(), "loaded config file");
                        }
                    }
                    Err(e) => {
                        tracing::warn!(
                            path = %config_path.display(),
                            error = %e,
                            "failed to read config file"
                        );
                    }
                }
            }
        }

        // Parse CLI args (highest priority).
        let mut i = 1; // Skip argv[0].
        let mut backend_explicit = false;
        while i < args.len() {
            match args[i].as_str() {
                "--backend" | "-b" => {
                    if let Some(val) = args.get(i + 1) {
                        config.backend = match val.as_str() {
                            "headless" => BackendKind::Headless,
                            "nested" | "wayland" | "x11" => BackendKind::Wayland,
                            _ => BackendKind::Drm,
                        };
                        backend_explicit = true;
                        i += 1;
                    }
                }
                "--nested" | "--wayland" | "-n" => {
                    config.backend = BackendKind::Wayland;
                    backend_explicit = true;
                }
                "--output-width" | "--width" | "-W" => {
                    if let Some(val) = args.get(i + 1).and_then(|v| v.parse().ok()) {
                        let h = config.resolution.map_or(0, |(_, h)| h);
                        config.resolution = Some((val, h));
                        i += 1;
                    }
                }
                "--output-height" | "--height" | "-H" => {
                    if let Some(val) = args.get(i + 1).and_then(|v| v.parse().ok()) {
                        let w = config.resolution.map_or(0, |(w, _)| w);
                        config.resolution = Some((w, val));
                        i += 1;
                    }
                }
                "--nested-width" | "-w" => {
                    if let Some(val) = args.get(i + 1).and_then(|v| v.parse().ok()) {
                        let h = config.game_resolution.map_or(0, |(_, h)| h);
                        config.game_resolution = Some((val, h));
                        i += 1;
                    }
                }
                "--nested-height" | "-h" => {
                    if let Some(val) = args.get(i + 1).and_then(|v| v.parse().ok()) {
                        let w = config.game_resolution.map_or(0, |(w, _)| w);
                        config.game_resolution = Some((w, val));
                        i += 1;
                    }
                }
                "--refresh-rate" | "-r" => {
                    if let Some(val) = args.get(i + 1).and_then(|v| v.parse().ok()) {
                        config.refresh_rate = Some(val);
                        i += 1;
                    }
                }
                "--output" | "-o" => {
                    if let Some(val) = args.get(i + 1) {
                        config.preferred_output = Some(val.clone());
                        i += 1;
                    }
                }
                "--vrr" => config.vrr = true,
                "--no-vrr" => config.vrr = false,
                "--hdr" => config.hdr = true,
                "--no-hdr" => config.hdr = false,
                "--upscale" => {
                    if let Some(val) = args.get(i + 1) {
                        config.upscale = match val.as_str() {
                            "fsr" => UpscaleMode::Fsr,
                            "nis" => UpscaleMode::Nis,
                            "cas" => UpscaleMode::Cas,
                            _ => UpscaleMode::None,
                        };
                        i += 1;
                    }
                }
                "--xwayland-count" => {
                    if let Some(val) = args.get(i + 1).and_then(|v| v.parse().ok()) {
                        config.xwayland_count = val;
                        i += 1;
                    }
                }
                "--fps-limit" => {
                    if let Some(val) = args.get(i + 1).and_then(|v| v.parse().ok()) {
                        config.fps_limit = val;
                        i += 1;
                    }
                }
                "--stats-pipe" => {
                    if let Some(val) = args.get(i + 1) {
                        config.stats_pipe = Some(PathBuf::from(val));
                        i += 1;
                    }
                }
                "--log" => {
                    if let Some(val) = args.get(i + 1) {
                        config.log_level = val.clone();
                        i += 1;
                    }
                }
                "-e" | "--steam" => config.steam_mode = true,
                "--no-steam" => config.steam_mode = false,
                "--" => {
                    // Everything after `--` is the child command.
                    let child_args: Vec<_> = args[i + 1..].to_vec();
                    if !child_args.is_empty() {
                        config.child_command = Some(child_args.join(" "));
                    }
                    break;
                }
                _ => {
                    // If no -- separator, treat remaining args as child command.
                    if !args[i].starts_with('-') {
                        config.child_command = Some(args[i..].join(" "));
                        break;
                    }
                }
            }
            i += 1;
        }

        // Auto-detect Wayland backend: if the user didn't explicitly choose a
        // backend and we're running inside a Wayland or X11 session, switch
        // to the Wayland backend automatically.
        if !backend_explicit
            && (std::env::var_os("WAYLAND_DISPLAY").is_some()
                || std::env::var_os("DISPLAY").is_some())
        {
            info!("auto-detected wayland backend from WAYLAND_DISPLAY / DISPLAY");
            config.backend = BackendKind::Wayland;
        }

        config
    }

    /// Apply values from a config file (lower priority than CLI).
    fn apply_file(&mut self, file: &ConfigFile) {
        if let Some(ref b) = file.backend {
            self.backend = match b.as_str() {
                "headless" => BackendKind::Headless,
                "nested" | "wayland" | "x11" => BackendKind::Wayland,
                _ => BackendKind::Drm,
            };
        }
        if let (Some(w), Some(h)) = (file.width, file.height) {
            self.resolution = Some((w, h));
        }
        if let Some(r) = file.refresh_rate {
            self.refresh_rate = Some(r);
        }
        if file.preferred_output.is_some() {
            self.preferred_output = file.preferred_output.clone();
        }
        if let Some(v) = file.vrr {
            self.vrr = v;
        }
        if let Some(h) = file.hdr {
            self.hdr = h;
        }
        if let Some(ref u) = file.upscale {
            self.upscale = match u.as_str() {
                "fsr" => UpscaleMode::Fsr,
                "nis" => UpscaleMode::Nis,
                "cas" => UpscaleMode::Cas,
                _ => UpscaleMode::None,
            };
        }
        if let Some(x) = file.xwayland_count {
            self.xwayland_count = x;
        }
        if let Some(rz) = file.red_zone_us {
            self.red_zone_us = rz;
        }
        if let Some(fl) = file.fps_limit {
            self.fps_limit = fl;
        }
        if let Some(cd) = file.cursor_hide_delay_ms {
            self.cursor_hide_delay_ms = cd;
        }
        if let Some(ref sp) = file.stats_pipe {
            self.stats_pipe = Some(PathBuf::from(sp));
        }
        if let Some(ref ll) = file.log_level {
            self.log_level = ll.clone();
        }
        if let Some(sm) = file.steam_mode {
            self.steam_mode = sm;
        }
    }
}

/// Get the XDG config directory path.
fn dirs_config_path() -> Option<PathBuf> {
    std::env::var_os("XDG_CONFIG_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".config")))
}

#[cfg(test)]
#[path = "config_tests.rs"]
mod tests;
