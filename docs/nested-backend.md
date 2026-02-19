# Wayland Backend

Run gamecomp as a window inside another Wayland or X11 compositor.

## Purpose

The wayland backend enables three workflows:

1. **Development** — Test the compositor without a dedicated display.
2. **CI** — Run integration tests inside Weston/Sway headless sessions.
3. **Embedding** — Render games inside another compositor's window.

## Architecture

```
┌──────────────────────────────────────────────────────┐
│  Host Compositor (Sway, KDE, GNOME, etc.)            │
│   ┌──────────────────────────────────────────────┐   │
│   │  gamecomp xdg_toplevel (full window)         │   │
│   │   ┌─────────────────────────────────────┐    │   │
│   │   │  Parent wl_surface (black bg)       │    │   │
│   │   │   ┌─────────────────────────────┐   │    │   │
│   │   │   │  Game wl_subsurface         │   │    │   │
│   │   │   │  (aspect-fit viewport)      │   │    │   │
│   │   │   └─────────────────────────────┘   │    │   │
│   │   └─────────────────────────────────────┘    │   │
│   └──────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────┘
```

The parent surface is filled with black pixels and covers the entire window.
The game content is presented on a `wl_subsurface` with a `wp_viewport`
destination sized to the largest aspect-preserving rectangle that fits
within the window, producing automatic letterboxing or pillarboxing.

### Presentation Paths

The wayland backend selects the optimal presentation path per frame:

| Path            | Condition                                          | GPU Work | Description |
|-----------------|--------------------------------------------------  |----------|-------------|
| Zero-copy       | Host supports the client's format + modifier       | None     | Forward DMA-BUF planes verbatim via `zwp_linux_buffer_params_v1.create_immed()` |
| Vulkan blit     | Host does not support format/modifier, or forced   | 1 blit   | Import → scale → export via `VulkanBlitter` |
| SHM fallback    | Non-DMA-BUF buffers                                | None     | `wl_shm` upload via shared memory pool |

Zero-copy forwarding is the fast path — it avoids all GPU work by passing
the client's DMA-BUF file descriptors directly to the host compositor.
This works whenever the host advertises support for the client's format
and modifier.

The Vulkan blit fallback is used only when the host compositor does not
support the client's format/modifier pair, or when explicitly forced via
`GAMECOMP_FORCE_BLIT=1`. The `VulkanBlitter` imports the client buffer,
blits to a host-compatible format, and forwards the result.

The `VulkanBlitter` is lazily initialized — it is only created when the
first blit-requiring frame arrives. If every frame takes the zero-copy
path, no Vulkan device or output images are ever allocated, saving ~100 ms
of startup time and significant GPU memory.

### Host Format Passthrough

On startup, the wayland thread performs two roundtrips to collect all
`zwp_linux_dmabuf_v1` format/modifier pairs the host compositor advertises.
These formats are then shared with the Wayland server via an `Arc<Mutex<HashMap>>`,
allowing the `dmabuf` protocol handler to advertise the host's real formats to
clients. This enables clients to allocate buffers with modifiers the host
directly supports, maximizing zero-copy hits.

The main thread blocks (`wait_for_host_formats()`) until formats are ready
before spawning XWayland, ensuring clients never see stale fallback formats.

### Thread Model

- **gamecomp-wayland** thread: Owns the host compositor connection and
  `VulkanBlitter`. Dispatches host events (resize, input, frame callbacks),
  presents committed client buffers, and sends `WaylandEvent`s to the main
  thread via `std::sync::mpsc`.
- **Main thread**: Forwards input events from the wayland backend to gamecomp's
  own Wayland server (for the game client). Sends committed buffers to the
  wayland thread.

### Differences from DRM Backend

| Feature          | DRM           | Wayland              |
|------------------|---------------|----------------------|
| Zero-copy        | Direct scanout| DMA-BUF forwarding   |
| VRR              | If supported  | No                   |
| HDR              | If supported  | No (planned)         |
| Tearing          | If supported  | No                   |
| Explicit sync    | If supported  | No                   |
| Page flip        | DRM atomic    | Host frame callback  |
| Input            | libinput      | Host forwarding      |
| Letterboxing     | N/A           | Subsurface + viewport|

## Usage

```sh
# Auto-detect: uses wayland backend if WAYLAND_DISPLAY or DISPLAY is set
gamecomp -- my-game

# Explicit wayland mode
gamecomp --nested -- my-game
gamecomp --backend nested -- my-game

# Custom window size
gamecomp --nested --width 1920 --height 1080 -- my-game
```

## Configuration

In `config.toml`:

```toml
backend = "nested"
width = 1280
height = 720
```

## Environment Variables

- `GAMECOMP_FORCE_BLIT=1` — Force the Vulkan blit path even when zero-copy
  would be possible. Useful for debugging blitter issues.

## Limitations

- No VRR or tearing — presentation is throttled by the host compositor.
- Input latency is slightly higher due to the host → nested forwarding path.
- HDR passthrough is not yet implemented.

## Implementation

Source: [`src/backend/wayland/`](../src/backend/wayland/)

The `WaylandBackend` struct implements the `Backend` trait. Key components:

- `WaylandConfig` — Window dimensions, title, fullscreen preference.
- `WaylandEvent` — Events from the host (resize, input, close, frame callback).
- `HostState` — Host compositor client-side Wayland state and protocol dispatch.
- `run_host_connection()` — Host event dispatch loop, frame presentation, and
  subsurface management.
- `VulkanBlitter` — Vulkan DMA-BUF import/blit pipeline for the fallback path.
  Lazily initialized on the first frame that requires a blit.
