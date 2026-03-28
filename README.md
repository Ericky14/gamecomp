# Gamecomp

High-performance single-app fullscreen Wayland compositor for gaming and robotics.

## Features

- **Direct scanout** — Zero-copy, zero-GPU presentation when possible (DRM backend)
- **Zero-copy DMA-BUF forwarding** — Forward client buffers to host compositor without GPU work (Wayland backend)
- **Vulkan compute compositor** — Fast shader-based composition when overlays or scaling are needed
- **Aspect-preserving letterboxing** — Subsurface architecture with automatic pillarbox/letterbox
- **XWayland** — Full X11 application support with per-server XWM threads
- **VRR support** — Variable refresh rate for compatible displays
- **Wayland backend** — Run inside another Wayland compositor for development/testing
- **Multi-XWayland** — Multiple XWayland servers with independent focus tracking and lifecycle management
- **Focus gating** — 4-phase cross-server focus arbitration (baselayer → stealer → retention → fallback). Only the focused server's clients receive frame callbacks — zero GPU waste on unfocused servers.
- **Baselayer control** — `GAMECOMP_BASELAYER_APPID` X11 atom for cross-server focus pinning
- **Frame pacing** — Adaptive VBlank scheduling with rolling peak draw-time tracking, compositing floor, and FPS limiting
- **Headless mode** — Offscreen rendering for CI and robotics pipelines

# Roadmap

High-level feature goals for Gamecomp. No hard dates — contributions welcome.

## Done

- [x] DRM/KMS backend with atomic modesetting
- [x] Headless backend for CI and robotics
- [x] Wayland backend (run inside another Wayland compositor)
- [x] Direct scanout (zero-copy presentation)
- [x] Zero-copy DMA-BUF forwarding in wayland backend
- [x] Host format/modifier passthrough for optimal buffer allocation
- [x] Aspect-preserving letterboxing via subsurface architecture
- [x] Vulkan blit fallback for unsupported modifiers
- [x] DMA-BUF import and format negotiation
- [x] Configuration via CLI args + TOML file
- [x] SPIR-V shader compilation at build time
- [x] Vulkan compute shader composition (blit.comp + VulkanBlitter pipeline)
- [x] Session switching — VT switch handling via libseat (Ctrl+Alt+Fn, pause/resume, fd revocation)
- [x] Keyboard monitor — raw evdev with udev hotplug, modifier tracking, connect/disconnect support
- [x] Pointer monitor — raw evdev with udev hotplug, motion/button/scroll, connect/disconnect support
- [x] GBM-backed output buffer allocation (native GEM handles, no PRIME corruption)
- [x] Async DMA-BUF implicit sync (non-blocking poll before blit)
- [x] Input routing — keyboard, pointer, scroll forwarding to focused client (DRM evdev + nested host passthrough)
- [x] Cursor passthrough — client cursor images forwarded to host compositor via SHM
- [x] XKB keymap and modifier forwarding from host to clients (nested mode)
- [x] Multi-XWayland server lifecycle — spawn, monitor, respawn with per-server XWM threads
- [x] Focus gating — 4-phase cross-server focus arbitration with per-surface commit gating
- [x] Baselayer focus control — `GAMECOMP_BASELAYER_APPID` atom for cross-server focus pinning
- [x] Adaptive frame pacing — rolling peak draw-time, compositing floor, VRR mode, FPS limiting
- [x] XWayland window management — XWM with fullscreen atoms, reparenting, per-server window tracking
- [x] Runtime resolution control — `GAMECOMP_XWAYLAND_MODE_CONTROL` atom for per-server resolution switching

## In Progress

- [ ] VRR (variable refresh rate) — frame pacer support exists, needs per-connector toggle
- [ ] Multi-plane assignment — overlay/cursor plane offload to reduce GPU work

## Planned

- [ ] **HDR** — HDR10 metadata passthrough, PQ tone mapping (feature-gated: `hdr`)
- [ ] **FSR upscaling** — AMD FidelityFX Super Resolution via compute shader (`fsr`)
- [ ] **NIS upscaling** — NVIDIA Image Scaling via compute shader (`nis`)
- [ ] **CAS sharpening** — Contrast Adaptive Sharpening post-process
- [ ] **PipeWire screen capture** — zero-copy DMA-BUF stream for recording/streaming (`pipewire`)
- [ ] **Tracy profiling** — per-frame instrumentation (`profile-with-tracy`)
- [ ] **Gamepad input** — evdev gamepad passthrough to client
- [ ] **Color management** — ICC/LUT loading, color-blind filters
- [ ] **Touch input** — touchscreen support for handheld devices
- [ ] **Multi-display** — span or mirror across multiple connected outputs (multi-XWayland spawning done, display routing in progress)

## Building

```sh
cargo build --release
```

Requires Rust stable (edition 2024) and the following system dependencies:
- `libdrm`, `libgbm` — DRM/KMS support
- `libseat`, `libudev` — Session and device management  
- `libwayland` — Wayland protocol
- `vulkan-loader` — Vulkan runtime

## Usage

```sh
# Run a game fullscreen on DRM
gamecomp -- my-game

# Run nested inside your desktop
gamecomp --nested -- my-game

# Headless mode for CI
gamecomp --backend headless -- my-app

# 2 XWayland servers (focus gating + baselayer control)
gamecomp --xwayland-count 2 -- steam

# Baselayer pinning (pin focus to a specific server's app)
# Set GAMECOMP_BASELAYER_APPID on a server's root window to pin focus

# Multiple XWayland servers (server 0 = platform, 1+ = game)
gamecomp --xwayland-count 2 -- my-game
```

### Options

| Flag | Description |
|------|-------------|
| `-n`, `--nested` | Run inside another Wayland compositor (alias: `--wayland`) |
| `--backend headless` | Offscreen rendering for CI |
| `-W`, `--output-width` | Output width in pixels |
| `-H`, `--output-height` | Output height in pixels |
| `-w`, `--nested-width` | Game render width (client-side resolution) |
| `-h`, `--nested-height` | Game render height (client-side resolution) |
| `-r`, `--refresh-rate` | Refresh rate in Hz |
| `-o`, `--output` | Preferred output connector (e.g., `eDP-1`) |
| `--xwayland-count N` | Number of XWayland servers to spawn (default: 1). Sets `STEAM_GAME_DISPLAY_N` env vars for child processes. |
| `--vrr` / `--no-vrr` | Enable/disable variable refresh rate |
| `--hdr` / `--no-hdr` | Enable/disable HDR |
| `--upscale MODE` | Upscaling algorithm: `fsr`, `nis`, `cas`, or `none` |
| `--fps-limit N` | FPS cap (0 = match display refresh rate) |
| `--stats-pipe PATH` | Write per-frame stats to a named pipe |
| `--log LEVEL` | Log level (`trace`, `debug`, `info`, `warn`, `error`) |

## Configuration

Create `~/.config/gamecomp/config.toml`:

```toml
vrr = true
hdr = false
upscale = "none"  # or "fsr", "nis", "cas"
```

## Documentation

- [Architecture](docs/architecture.md) — Module map, thread model, design decisions
- [Wayland Backend](docs/nested-backend.md) — Zero-copy DMA-BUF forwarding, subsurface letterboxing, host format passthrough

## Disclaimer

Portions of this codebase were generated with the assistance of large language models (LLMs). All AI-generated code has been reviewed and tested by the project maintainers.

## License

GPL-3.0. See [LICENSE](LICENSE) for details.
