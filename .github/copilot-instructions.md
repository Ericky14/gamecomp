# Gamecomp — Project Best Practices

## Language & Toolchain

- **Rust 1.93.1 stable**, edition 2024.
- Always run `cargo clippy` before committing. Zero warnings policy.
- Format all code with `cargo fmt` (config in `rustfmt.toml`).

## Performance Rules

- **Zero allocation in hot paths.** Use fixed-size arrays, arenas, or pre-allocated buffers. Never `Vec::push` or `String::from` in per-frame code.
- **No `clone()` on hot paths** without a comment justifying why.
- **`#[inline(always)]`** on small functions called in the render loop (< 10 lines). `#[cold]` on error-handling paths.
- **Prefer `Copy` types** for data passed between threads (e.g., `FrameInfo`, `Layer`).
- **Use `core::hint::assert_unchecked`** to elide bounds checks in proven-safe hot loops. Document the safety invariant.
- **Atomic ordering:** Use `Relaxed`, `Acquire`, or `Release` — never default `SeqCst` unless required. Comment the ordering choice.
- **Prefer `parking_lot`** over `std::sync` when locks are unavoidable (no poisoning, faster). But prefer channels and lock-free designs.

## Architecture Rules

- **No shared mutable state between threads.** Use channels (`calloop::channel`, SPSC ring buffers) for inter-thread communication.
- The main thread owns all Wayland state exclusively. The render thread owns Vulkan and DRM exclusively. The XWM thread owns the X11 connection exclusively.
- **`unsafe` only at FFI boundaries.** Every `unsafe` block must have a `// SAFETY:` comment explaining the invariant.
- **Feature-gate optional capabilities:** HDR, FSR, NIS, PipeWire, Tracy. Unused features compile to zero code.

## Code Style

- Every module starts with a `//!` doc comment explaining purpose and design rationale.
- Public types and functions get `///` doc comments.
- Error types use `thiserror`. Propagation uses `anyhow::Result` at application boundaries, typed errors within libraries.
- Use `tracing::{info, warn, error, debug, trace}` for logging. Never `println!` or `eprintln!`.
- Constants and enums over magic numbers. Use `const` generics where applicable.
- **Extract repeated flag/constant combinations into module-level `const`.** For example, Vulkan usage flags or DRM modifier sentinels used in multiple functions belong at the top of the module, not redefined locally in each function.
- Prefer `&[T]` slices over `Vec<T>` in function signatures.
- File names are `snake_case`. Types are `PascalCase`. Functions and variables are `snake_case`.
- **Functions should be ≤ 150 lines.** If longer, extract coherent sub-steps into helpers. Exception: single GPU command buffer recordings where splitting breaks readability.
- **No duplicated logic across functions.** If two functions share > 50% structure, extract the common pattern into a parameterized helper.

## Testing

- Unit tests in `#[cfg(test)] mod tests` at the bottom of each module.
- Integration tests in `tests/` for end-to-end compositor verification.
- Use `HeadlessBackend` for CI testing without GPU/display hardware.

## Documentation

- Feature docs go in `docs/<feature>.md` — minimal, focused, no mentions of other projects.
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
