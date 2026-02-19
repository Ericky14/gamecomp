# Gamecomp

High-performance single-app fullscreen Wayland compositor for gaming and robotics.

## Features

- **Direct scanout** — Zero-copy, zero-GPU presentation when possible (DRM backend)
- **Zero-copy DMA-BUF forwarding** — Forward client buffers to host compositor without GPU work (Wayland backend)
- **Vulkan compute compositor** — Fast shader-based composition when overlays or scaling are needed
- **Aspect-preserving letterboxing** — Subsurface architecture with automatic pillarbox/letterbox
- **XWayland** — Full X11 application support
- **VRR support** — Variable refresh rate for compatible displays
- **Wayland backend** — Run inside another Wayland compositor for development/testing
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

## In Progress

- [ ] Vulkan compute shader composition — infrastructure ready, pipeline/dispatch not wired up
- [ ] XWayland window management — core WM works, close/resize/title reading stubbed
- [ ] Adaptive frame pacing — algorithm complete, needs timerfd scheduling integration
- [ ] libinput integration — event types defined, device enumeration/polling not implemented
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
- [ ] **Session switching** — VT switch handling via libseat

## Building

```sh
cargo build --release
```

Requires Rust 1.93.1+ and the following system dependencies:
- `libdrm`, `libgbm` — DRM/KMS support
- `libinput`, `libseat`, `libudev` — Input and session management  
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
```

See `gamecomp --help` for all options.

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

MIT. See [LICENSE](LICENSE) for details.
