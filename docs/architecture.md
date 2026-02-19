# Architecture

## Overview

Gamecomp is a single-app fullscreen Wayland compositor optimized for gaming and robotics.
It renders one primary client at a time using either direct scanout (zero-copy, zero-GPU)
or Vulkan compute shader composition when overlays or scaling are needed.

## Module Map

```
src/
  main.rs              CLI parsing, thread spawning, startup orchestration
  config.rs            Configuration (CLI args + TOML file)
  stats.rs             Performance counters, stats pipe output

  backend/
    mod.rs             Backend trait — display output abstraction
    drm.rs             DRM/KMS backend — atomic modesetting, page flips, plane assignment
    headless.rs        Headless backend — offscreen rendering for CI/robotics
    gpu/
      mod.rs           Shared GPU utilities
      vulkan_blitter.rs Vulkan DMA-BUF blit pipeline — import, scale, export
    wayland/
      mod.rs           Wayland backend — runs inside another compositor, zero-copy DMA-BUF forwarding
      event_loop.rs    Host compositor event loop — frame presentation, subsurface letterboxing
      host_state.rs    Host Wayland protocol dispatch — registry, DMA-BUF formats, input, viewporter

  render/
    mod.rs             Renderer trait — GPU abstraction (Vulkan, V3D, software)
    post_process.rs    PostProcessor trait + chain — FSR, NIS, CAS, sharpening
    color.rs           ColorPipeline trait — HDR, tonemapping, LUT application

  compositor/
    mod.rs             VulkanCompositor — device setup, pipeline management, composition dispatch
    texture.rs         VulkanTexture — DMA-BUF import, image lifecycle, format negotiation
    scene.rs           FrameInfo + Layer — fixed-size scene description, zero-alloc
    shaders/           GLSL compute shader sources (compiled to SPIR-V at build time)

  wayland/
    mod.rs             Wayland server — display, globals, client dispatch
    atoms.rs           X11 atom definitions for XWayland window management
    window_tracker.rs  Track client window geometry and focus state
    xwayland.rs        XWayland launch, X11 window manager, surface import
    protocols/
      mod.rs           Protocol module root
      compositor.rs    wl_compositor, wl_surface — surface management
      data_device.rs   wl_data_device_manager — clipboard / drag-and-drop
      dmabuf.rs        zwp_linux_dmabuf_v1 — DMA-BUF import, host format passthrough
      output.rs        wl_output — display mode advertisement
      seat.rs          wl_seat — keyboard, pointer input
      shm.rs           wl_shm — shared-memory buffer support
      subcompositor.rs wl_subcompositor — subsurface management
      wl_drm.rs        wl_drm — XWayland glamor GPU buffer negotiation
      xdg_shell.rs     xdg_wm_base, xdg_surface, xdg_toplevel — window management

  input/
    mod.rs             libinput integration, event batching, keyboard/pointer dispatch

  frame_pacer.rs       Adaptive VBlank scheduling, timerfd wakeups, VRR support
```

## Thread Model

Three or four threads, communicating exclusively via channels. No shared mutable state.

### DRM Mode (bare-metal)

```
┌─────────────────────┐       FrameInfo (SPSC)       ┌──────────────────────┐
│  gamecomp-main      │ ──────────────────────────▶ │  gamecomp-render     │
│                     │                              │                      │
│  • Wayland dispatch │   ◀──── flip complete ───── │  • Vulkan compute    │
│  • libinput events  │                              │  • DRM atomic commit │
│  • Config/timers    │                              │  • Page flip wait    │
│  • Frame pacer      │                              │                      │
└─────────┬───────────┘                              └──────────────────────┘
          │
          │  X11 events (channel)
          │
┌─────────┴───────────┐
│  gamecomp-xwm       │
│                     │
│  • x11rb connection │
│  • Window manage    │
│  • Atom handling    │
└─────────────────────┘
```

### Wayland Mode (nested inside another compositor)

```
┌─────────────────────┐     CommittedBuffer (mpsc)   ┌──────────────────────┐
│  gamecomp-main      │ ──────────────────────────▶ │  gamecomp-wayland    │
│                     │                              │                      │
│  • Wayland server   │   ◀──── WaylandEvent ────── │  • Host wl_surface   │
│  • Frame pacer      │                              │  • Zero-copy DMA-BUF │
│  • Config/timers    │                              │  • Vulkan blit fbk   │
│                     │                              │  • Subsurface lgbx   │
└─────────┬───────────┘                              └──────────────────────┘
          │
          │  X11 events (channel)
          │
┌─────────┴───────────┐
│  gamecomp-xwm       │
│                     │
│  • x11rb connection │
│  • Window manage    │
│  • Atom handling    │
└─────────────────────┘
```

### Main Thread (`gamecomp-main`)
Runs the `calloop` event loop. Owns all Wayland protocol state, input devices,
configuration, and the frame pacer. Builds a `FrameInfo` scene description each
frame and sends it to the render thread. Receives X11 window events from the XWM thread.

### Render Thread (`gamecomp-render`)
DRM mode only. Owns the `VulkanCompositor` and `DrmBackend` exclusively. Receives
`FrameInfo` from the main thread. Either performs direct scanout (DMA-BUF → DRM
plane, zero GPU work) or dispatches a Vulkan compute shader for composition. Waits
for page flip completion and signals back to the main thread.

### Wayland Thread (`gamecomp-wayland`)
Wayland backend mode only. Owns the host compositor connection (`wl_display`,
`xdg_toplevel`). Receives committed client buffers from the
main thread and presents them to the host via one of three paths:
- **Zero-copy DMA-BUF**: Forward client DMA-BUF planes directly to the host
  (no GPU work) when the host supports the client's format and modifier.
- **Vulkan blit**: Import the client DMA-BUF, blit to a new buffer, and forward
  the result. Used when the host does not support the format/modifier, or when
  forced via `GAMECOMP_FORCE_BLIT=1`. The `VulkanBlitter` is lazily initialized
  on the first frame that requires a blit.
- **SHM fallback**: Shared-memory upload for non-DMA-BUF buffers.

Uses a subsurface architecture for aspect-preserving letterboxing: the parent
surface displays a black background at the full window size, while the game
content is presented on a `wl_subsurface` with a `wp_viewport` destination
set to the aspect-fit rectangle.

### XWM Thread (`gamecomp-xwm`)
Owns the `x11rb` connection to XWayland. Handles X11 window management
(SubstructureRedirect, fullscreen atoms, reparenting). Sends map/unmap/configure
events to the main thread via channel.

## Data Flow (Per Frame)

### DRM Backend

1. Client attaches a buffer (DMA-BUF or SHM) via Wayland protocol
2. Main thread receives the buffer commit, resolves focus, builds `FrameInfo`
3. Main thread sends `FrameInfo` to render thread via lock-free SPSC channel
4. Render thread checks if direct scanout is possible:
   - **Yes**: assigns client DMA-BUF to primary DRM plane → atomic commit → done
   - **No**: imports textures, dispatches compute shader, writes to scanout buffer → atomic commit
5. DRM page flip completes → render thread signals main thread
6. Main thread sends frame callback to client, schedules next wakeup via frame pacer

### Wayland Backend

1. Client attaches a buffer (DMA-BUF or SHM) via Wayland protocol
2. Main thread receives the buffer commit and sends the `CommittedBuffer` to the wayland thread
3. Wayland thread selects presentation path based on buffer type and modifier:
   - **Zero-copy**: host supports client's format and modifier → forward DMA-BUF planes
     verbatim via `zwp_linux_buffer_params_v1.create_immed()` → no GPU work
   - **Vulkan blit**: host does not support client's format/modifier, or forced via env var →
     `VulkanBlitter` (lazily initialized) imports, scales, exports to host-compatible buffer
   - **SHM fallback**: non-DMA-BUF buffer → `wl_shm` upload
4. Wayland thread sets `wp_viewport` destination to aspect-fit rectangle on the game subsurface
5. Host compositor presents the frame, sends frame callback → wayland thread sends
   `WaylandEvent::FrameCallback` to main thread
6. Main thread sends frame callback to client, schedules next wakeup via frame pacer

## Composition Modes

| Mode              | When Used                                                    | GPU Work             | Latency |
|-------------------|--------------------------------------------------------------|----------------------|---------|
| Direct scanout    | DRM: single fullscreen app, no overlays, compatible format   | None                 | Lowest  |
| Zero-copy forward | Wayland: host supports client's format + modifier          | None                 | Lowest  |
| Vulkan blit       | Wayland: unsupported format/modifier or forced             | 1 blit dispatch      | Low     |
| Vulkan composite  | DRM: overlays active, scaling needed, format conversion      | 1 compute dispatch   | Low     |
| Vulkan + upscale  | FSR/NIS enabled                                              | 2 compute dispatches | Medium  |

## Key Design Decisions

- **Fixed-size `FrameInfo`**: 4 layers max, `Copy` trait, stack-allocated. Zero heap allocation per frame.
- **Compute-only Vulkan**: No graphics pipeline. Compute shaders sample input layers and write directly to output. Uses compute-only queue when available.
- **Push constants only**: Scene description passed via push constants — no descriptor set updates per frame.
- **Plane assignment without libliftoff**: Direct `DRM_MODE_ATOMIC_TEST_ONLY` probing. For ≤4 layers the search space is trivial.
- **`timerfd` frame pacing**: Rolling average draw time + red zone buffer for adaptive wakeup scheduling.

## Render Pipeline

The render pipeline is decoupled from specific GPU APIs via traits in `src/render/`:

```
Client buffer → Renderer::import_dmabuf()
                    │
                    ▼
              Renderer::blit_contain()  ←── aspect-preserving scale + clear
                    │
                    ▼
              PostProcessChain::apply()  ←── optional: FSR, NIS, CAS (feature-gated)
                    │
                    ▼
              ColorPipeline::apply()     ←── optional: HDR tonemapping, LUT (feature-gated)
                    │
                    ▼
              ExportedFrame → Backend::present()
```

**`Renderer` trait** — Core GPU abstraction. Implementations:
- `VulkanRenderer` — Production path using `ash` (desktop GPUs: NVIDIA, AMD, Intel)
- `V3dRenderer` — Broadcom V3D/V4 for Raspberry Pi 5 (planned)
- `SoftwareRenderer` — CPU fallback for headless CI (planned)

**`PostProcessor` trait** — Per-frame image processing. Chained via `PostProcessChain`.
Empty chain = zero overhead. Each processor is behind a feature gate (`fsr`, `nis`).

**`ColorPipeline` trait** — Color management. `IdentityColorPipeline` is the default
no-op for SDR. HDR implementation handles PQ/HLG transfer functions, gamut mapping,
and 3D LUT application (behind `hdr` feature gate).
