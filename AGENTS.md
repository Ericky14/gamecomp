# Gamecomp — Agent Instructions

## Project Overview

Gamecomp is a high-performance, single-app Wayland compositor purpose-built for gaming.
It composites client surfaces via Vulkan, presents through DRM/KMS (direct) or a host Wayland compositor (nested), and manages one or more XWayland servers for X11 game compatibility.
Steam mode (`-e`/`--steam`) enables multi-server focus gating with zero GPU waste — only the focused server's clients receive frame callbacks.

## Language & Toolchain

- **Rust stable**, edition 2024.
- Always run `cargo clippy` before committing. **Zero warnings policy.**
- Format all code with `cargo fmt` (config in `rustfmt.toml`).
- Run `cargo test` — all tests must pass before committing.

## Source Layout

```
src/
├── main.rs              # Entry point, event loop orchestration, helper dispatchers
├── config.rs            # CLI args and runtime configuration
├── focus.rs             # FocusArbiter — 4-phase cross-server focus arbitration
├── render_thread.rs     # Render thread — Vulkan compositing, DRM event loop, present
├── xwayland_mgr.rs      # XWayland server lifecycle (spawn, monitor, respawn)
├── frame_pacer.rs       # Frame pacing / timing
├── stats.rs             # Runtime statistics
├── retry.rs             # Retry utilities
├── test_harness.rs      # Test infrastructure
├── backend/
│   ├── mod.rs           # Backend trait + backend selection
│   ├── drm.rs           # DRM/KMS direct scanout backend
│   ├── headless.rs      # Headless backend (CI testing)
│   ├── session.rs       # Libseat session management
│   ├── gpu/             # Vulkan blitter (device, images, modifiers, blit, tests)
│   └── wayland/         # Nested Wayland backend (host_state, event_loop)
├── compositor/
│   ├── mod.rs           # Vulkan compositor setup
│   ├── scene.rs         # Scene graph / layer composition
│   ├── texture.rs       # Texture management
│   └── shaders/         # GLSL shaders (compiled at build time)
├── render/
│   ├── mod.rs           # Render pipeline orchestration
│   ├── color.rs         # Color space handling
│   └── post_process.rs  # Post-processing (FSR, NIS, etc.)
├── input/
│   ├── mod.rs           # Input subsystem entry
│   ├── keyboard.rs      # Evdev keyboard → Wayland forwarding
│   ├── keyboard_tests.rs
│   └── pointer.rs       # Evdev pointer → Wayland forwarding
└── wayland/
    ├── mod.rs           # Wayland server setup, commit gating
    ├── atoms.rs         # X11 atom definitions (STEAM_GAME, GAMECOMP_*, etc.)
    ├── window_tracker.rs# Per-server window tracking and local focus
    ├── xwayland.rs      # XWM thread, X11 event handling, focus feedback
    └── protocols/       # Wayland protocol implementations
        ├── compositor.rs, seat.rs, output.rs, shm.rs, dmabuf.rs
        ├── xdg_shell.rs, subcompositor.rs, data_device.rs
        └── wl_drm.rs
```

## Threading Model

- **Main thread**: Owns all Wayland server state, runs the calloop event loop, dispatches input, drives focus arbitration.
- **Render thread**: Owns Vulkan device and DRM resources exclusively. Receives committed buffers via channel, presents to display.
- **XWM threads** (one per XWayland server): Each owns its X11 connection. Communicates to main thread via `calloop::channel` and `Arc<AtomicU32>`.

**No shared mutable state between threads.** Use channels (`calloop::channel`, SPSC ring buffers) for inter-thread communication.

## Performance Rules

- **Zero allocation in hot paths.** Use fixed-size arrays, arenas, or pre-allocated buffers. Never `Vec::push` or `String::from` in per-frame code.
- **No `clone()` on hot paths** without a comment justifying why.
- **`#[inline(always)]`** on small functions called in the render loop (< 10 lines). `#[cold]` on error-handling paths.
- **Prefer `Copy` types** for data passed between threads (e.g., `FrameInfo`, `Layer`).
- **Use `core::hint::assert_unchecked`** to elide bounds checks in proven-safe hot loops. Document the safety invariant.
- **Atomic ordering:** Use `Relaxed`, `Acquire`, or `Release` — never default `SeqCst` unless required. Comment the ordering choice.
- **Prefer `parking_lot`** over `std::sync` when locks are unavoidable (no poisoning, faster). But prefer channels and lock-free designs.

## Code Quality Standards

### Readability First

- **Avoid deep nesting.** Max 3 levels of indentation in any function body. Use early returns, `let-else`, `?`, and helper functions to flatten logic.
- **Functions should be ≤ 150 lines.** Extract coherent sub-steps into well-named helpers. Exception: single GPU command buffer recordings where splitting breaks readability.
- **Descriptive names over comments.** If a variable or function needs a comment to explain *what* it is, rename it instead. Reserve comments for *why*.
- **One responsibility per function.** If a function does setup *and* processing *and* cleanup, split it.
- **Guard clauses at the top.** Handle error/skip cases early with `return`, not deep `if-else` branches.

### Code Organization

- Every module starts with a `//!` doc comment explaining purpose and design rationale.
- Public types and functions get `///` doc comments.
- **Extract repeated flag/constant combinations into module-level `const`.** Vulkan usage flags or DRM modifier sentinels used in multiple functions belong at the top of the module.
- **No duplicated logic across functions.** If two functions share > 50% structure, extract the common pattern into a parameterized helper.
- Constants and enums over magic numbers. Use `const` generics where applicable.

### Error Handling

- Error types use `thiserror`. Propagation uses `anyhow::Result` at application boundaries, typed errors within libraries.
- Use `tracing::{info, warn, error, debug, trace}` for logging. Never `println!` or `eprintln!`.

### Naming & Style

- File names: `snake_case`. Types: `PascalCase`. Functions and variables: `snake_case`.
- Prefer `&[T]` slices over `Vec<T>` in function signatures.

## Architecture Rules

- **`unsafe` only at FFI boundaries.** Every `unsafe` block must have a `// SAFETY:` comment explaining the invariant.
- **Feature-gate optional capabilities:** HDR, FSR, NIS, PipeWire, Tracy. Unused features compile to zero code.
- The main thread owns all Wayland state exclusively. The render thread owns Vulkan and DRM exclusively. The XWM thread owns the X11 connection exclusively.

## Testing

- Unit tests live in separate `*_tests.rs` files, referenced via `#[cfg(test)] #[path = "..._tests.rs"] mod tests;` at the bottom of each module.
- Integration tests in `tests/` for end-to-end compositor verification.
- Use `HeadlessBackend` for CI testing without GPU/display hardware.

## Documentation

- Feature docs go in `docs/<feature>.md` — minimal, focused.
- Explain *what* the feature does, *how* it works internally, and *why* the design was chosen.
- Keep docs under 200 lines. Link to source code where relevant.

## Dependencies

- Every C library dependency must be justified. Prefer pure-Rust alternatives when performance-equivalent.
- Pin dependency versions in `Cargo.toml`. Review changelogs before bumping.
- Use `[build-dependencies]` only for code generation (shader compilation, protocol generation).

## Git Practices

- Commit messages: `<module>: <imperative description>` (e.g., `backend/drm: implement atomic modesetting`).
- One logical change per commit. Separate refactors from features.
- Tag releases with `v<semver>`.

## Debugging

- Use `Taskfile.yml` for common debug commands: `task x11:atoms`, `task x11:focus`, `task x11:windows`, `task x11:servers`.
- Steam mode control: `task x11:set-steam-game`, `task x11:set-baselayer-appid`.
- Resolution control: `task x11:set-resolution`.
