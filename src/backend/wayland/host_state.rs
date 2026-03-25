//! Host compositor client state and Wayland protocol dispatch implementations.
//!
//! `HostState` holds the Wayland client-side state for the connection to the
//! host compositor. The `Dispatch` implementations handle protocol events
//! from the host: registry globals, surface configure, input, DMA-BUF
//! format advertisements, window decorations, etc.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};

use tracing::{debug, info, trace, warn};
use wayland_client::protocol::{
    wl_compositor, wl_keyboard, wl_output, wl_pointer, wl_region, wl_registry, wl_seat, wl_shm,
    wl_shm_pool, wl_subcompositor, wl_subsurface, wl_surface,
};
use wayland_client::{Connection, Dispatch, Proxy, QueueHandle};
use wayland_protocols::wp::fractional_scale::v1::client::{
    wp_fractional_scale_manager_v1, wp_fractional_scale_v1,
};
use wayland_protocols::wp::linux_dmabuf::zv1::client::{
    zwp_linux_buffer_params_v1::{self, ZwpLinuxBufferParamsV1},
    zwp_linux_dmabuf_v1::{self, ZwpLinuxDmabufV1},
};
use wayland_protocols::wp::viewporter::client::{wp_viewport, wp_viewporter};
use wayland_protocols::xdg::decoration::zv1::client::{
    zxdg_decoration_manager_v1, zxdg_toplevel_decoration_v1,
};
use wayland_protocols::xdg::shell::client::{xdg_surface, xdg_toplevel, xdg_wm_base};

use super::WaylandEvent;

// ─── Host compositor client state ───────────────────────────────────

/// State for the host compositor connection (Wayland client side).
pub(super) struct HostState {
    pub configured: bool,
    pub closed: bool,
    /// Current host window dimensions (logical pixels, as per xdg_toplevel configure).
    pub width: u32,
    pub height: u32,
    /// Desired physical pixel size from config (-W×-H). Used to compute the
    /// logical window size when responding to xdg_toplevel.configure, so the
    /// host compositor allocates the right number of physical pixels.
    pub desired_width: u32,
    pub desired_height: u32,
    pub compositor: Option<wl_compositor::WlCompositor>,
    pub subcompositor: Option<wl_subcompositor::WlSubcompositor>,
    pub wm_base: Option<xdg_wm_base::XdgWmBase>,
    pub shm: Option<wl_shm::WlShm>,
    pub linux_dmabuf: Option<ZwpLinuxDmabufV1>,
    pub viewporter: Option<wp_viewporter::WpViewporter>,
    pub fractional_scale_mgr: Option<wp_fractional_scale_manager_v1::WpFractionalScaleManagerV1>,
    pub decoration_mgr: Option<zxdg_decoration_manager_v1::ZxdgDecorationManagerV1>,
    /// Host seat for keyboard/pointer input.
    pub seat: Option<wl_seat::WlSeat>,
    /// Fractional scale denominator from wp_fractional_scale_v1.
    /// 120 = scale 1.0, 240 = scale 2.0, 150 = scale 1.25, etc.
    pub fractional_scale: u32,
    pub surface: Option<wl_surface::WlSurface>,
    pub xdg_surface: Option<xdg_surface::XdgSurface>,
    pub xdg_toplevel: Option<xdg_toplevel::XdgToplevel>,
    pub tx: std::sync::mpsc::Sender<WaylandEvent>,
    /// Shared atomic for detected host display refresh rate (millihertz).
    pub detected_refresh_mhz: Arc<AtomicU32>,
    /// Refresh rate (mHz) of each known host output, keyed by registry name.
    pub output_refresh_rates: HashMap<u32, i32>,
    /// Registry names of the outputs our surface currently overlaps.
    /// Updated via `wl_surface.enter` / `wl_surface.leave`.
    /// The last entry is the most recently entered output.
    pub surface_outputs: Vec<u32>,
    /// DMA-BUF format → modifier list advertised by the host compositor.
    /// Populated from `zwp_linux_dmabuf_v1.modifier` events during the
    /// initial roundtrip. Used to decide whether zero-copy forwarding is
    /// possible for a given client buffer.
    pub host_dmabuf_formats: HashMap<u32, Vec<u64>>,
    /// Whether the host compositor advertised any non-INVALID modifier.
    /// When true, we include explicit modifiers in `create_immed` calls.
    pub can_use_modifiers: bool,
    /// Serial from the most recent `wl_pointer.enter` event on the host.
    /// Needed for `wl_pointer.set_cursor` on the host pointer.
    pub pointer_enter_serial: u32,
    /// Host `wl_pointer` proxy — needed to call `set_cursor`.
    pub pointer: Option<wl_pointer::WlPointer>,
    /// Last-presented game buffer dimensions. Updated on each present
    /// call. Used together with `width`/`height` to compute the viewport
    /// offset and scale for pointer coordinate mapping.
    pub game_buf_w: u32,
    pub game_buf_h: u32,
}

impl HostState {
    /// Map host surface-local pointer coordinates to client buffer coordinates.
    ///
    /// Accounts for the viewport-based aspect-fit letterboxing: the game
    /// subsurface is centered within the host window, viewport-scaled from
    /// `game_buf_w × game_buf_h` to a `fit_w × fit_h` destination. Pointer
    /// coordinates from the host include the letterbox offset, so we must
    /// subtract it and rescale to the buffer's native resolution.
    ///
    /// Returns `(buf_x, buf_y)` in the client buffer's coordinate space,
    /// clamped to `[0, game_buf_w) × [0, game_buf_h)`.
    #[inline(always)]
    pub fn host_to_buffer_coords(&self, host_x: f64, host_y: f64) -> (f64, f64) {
        // Before the first frame, no buffer dimensions are known — pass through.
        if self.game_buf_w == 0 || self.game_buf_h == 0 {
            return (host_x, host_y);
        }

        let win_w = self.width.max(1) as f64;
        let win_h = self.height.max(1) as f64;
        let buf_w = self.game_buf_w as f64;
        let buf_h = self.game_buf_h as f64;

        // Replicate the contain_fit() logic from event_loop.rs.
        let (fit_w, fit_h) = {
            let lhs = (self.game_buf_w as u64) * (self.height as u64);
            let rhs = (self.game_buf_h as u64) * (self.width as u64);
            if lhs > rhs {
                let fw = self.width;
                let fh = ((self.width as u64) * (self.game_buf_h as u64) / (self.game_buf_w as u64))
                    as u32;
                (fw.max(1) as f64, fh.max(1) as f64)
            } else {
                let fh = self.height;
                let fw = ((self.height as u64) * (self.game_buf_w as u64)
                    / (self.game_buf_h as u64)) as u32;
                (fw.max(1) as f64, fh.max(1) as f64)
            }
        };

        let off_x = (win_w - fit_w) / 2.0;
        let off_y = (win_h - fit_h) / 2.0;

        // Subtract letterbox offset, then map viewport-dest → buffer coords.
        let bx = ((host_x - off_x) * buf_w / fit_w).clamp(0.0, buf_w - 1.0);
        let by = ((host_y - off_y) * buf_h / fit_h).clamp(0.0, buf_h - 1.0);

        (bx, by)
    }

    /// Check whether the host compositor supports a given format+modifier pair.
    ///
    /// Returns `true` when the host advertised the exact (format, modifier)
    /// combination, OR when no explicit modifiers were advertised at all
    /// (the host only supports implicit sync / `DRM_FORMAT_MOD_INVALID`).
    #[inline(always)]
    pub fn host_supports_format(&self, format: u32, modifier: u64) -> bool {
        if let Some(mods) = self.host_dmabuf_formats.get(&format) {
            // Host advertised this format — check modifier.
            mods.contains(&modifier)
                || modifier == DRM_FORMAT_MOD_INVALID
                || !self.can_use_modifiers
        } else {
            // Host didn't advertise this format at all.
            false
        }
    }

    /// Update the active refresh rate to match the most recently entered
    /// output and store it in the shared atomic.
    ///
    /// Uses the **last entered** output (back of `surface_outputs`) since
    /// that represents the display the window is moving onto. If the
    /// surface has left all outputs, the atomic is left unchanged.
    pub fn update_active_refresh_rate(&self) {
        // Most recently entered output is at the back of the Vec.
        let mhz = self
            .surface_outputs
            .last()
            .and_then(|name| self.output_refresh_rates.get(name))
            .copied()
            .unwrap_or(0);
        if mhz > 0 {
            let hz = (mhz as u32 + 500) / 1000;
            info!(
                refresh_mhz = mhz,
                refresh_hz = hz,
                active_outputs = self.surface_outputs.len(),
                "active display refresh rate updated"
            );
            // Ordering: Release so the main thread's Relaxed load sees it.
            self.detected_refresh_mhz
                .store(mhz as u32, Ordering::Release);
        }
    }
}

// Registry handler — bind globals.
impl Dispatch<wl_registry::WlRegistry, ()> for HostState {
    fn event(
        state: &mut Self,
        registry: &wl_registry::WlRegistry,
        event: wl_registry::Event,
        _data: &(),
        _conn: &Connection,
        qh: &QueueHandle<Self>,
    ) {
        if let wl_registry::Event::Global {
            name,
            interface,
            version,
        } = event
        {
            match interface.as_str() {
                "wl_compositor" => {
                    let comp = registry.bind::<wl_compositor::WlCompositor, _, _>(
                        name,
                        version.min(6),
                        qh,
                        (),
                    );
                    state.compositor = Some(comp);
                }
                "wl_subcompositor" => {
                    let subcomp = registry.bind::<wl_subcompositor::WlSubcompositor, _, _>(
                        name,
                        version.min(1),
                        qh,
                        (),
                    );
                    info!("host compositor supports wl_subcompositor");
                    state.subcompositor = Some(subcomp);
                }
                "xdg_wm_base" => {
                    let wm =
                        registry.bind::<xdg_wm_base::XdgWmBase, _, _>(name, version.min(6), qh, ());
                    state.wm_base = Some(wm);
                }
                "wl_shm" => {
                    let shm = registry.bind::<wl_shm::WlShm, _, _>(name, version.min(2), qh, ());
                    state.shm = Some(shm);
                }
                "zwp_linux_dmabuf_v1" => {
                    let dmabuf =
                        registry.bind::<ZwpLinuxDmabufV1, _, _>(name, version.min(3), qh, ());
                    info!("host compositor supports zwp_linux_dmabuf_v1");
                    state.linux_dmabuf = Some(dmabuf);
                }
                "wp_viewporter" => {
                    let vp = registry.bind::<wp_viewporter::WpViewporter, _, _>(
                        name,
                        version.min(1),
                        qh,
                        (),
                    );
                    info!("host compositor supports wp_viewporter");
                    state.viewporter = Some(vp);
                }
                "wp_fractional_scale_manager_v1" => {
                    let mgr = registry
                        .bind::<wp_fractional_scale_manager_v1::WpFractionalScaleManagerV1, _, _>(
                            name,
                            version.min(1),
                            qh,
                            (),
                        );
                    info!("host compositor supports wp_fractional_scale_manager_v1");
                    state.fractional_scale_mgr = Some(mgr);
                }
                "zxdg_decoration_manager_v1" => {
                    let mgr = registry
                        .bind::<zxdg_decoration_manager_v1::ZxdgDecorationManagerV1, _, _>(
                            name,
                            version.min(1),
                            qh,
                            (),
                        );
                    info!("host compositor supports zxdg_decoration_manager_v1");
                    state.decoration_mgr = Some(mgr);
                }
                "wl_output" => {
                    // Bind wl_output to detect host display refresh rate.
                    // Pass the registry `name` as user-data so we can identify
                    // this output in surface enter/leave events.
                    let _output =
                        registry.bind::<wl_output::WlOutput, _, _>(name, version.min(4), qh, name);
                }
                "wl_seat" => {
                    let seat = registry.bind::<wl_seat::WlSeat, _, _>(name, version.min(9), qh, ());
                    state.seat = Some(seat);
                    info!("bound host wl_seat for input");
                }
                _ => {}
            }
        }
    }
}

// Compositor — no events to handle.
impl Dispatch<wl_compositor::WlCompositor, ()> for HostState {
    fn event(
        _state: &mut Self,
        _proxy: &wl_compositor::WlCompositor,
        _event: wl_compositor::Event,
        _data: &(),
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
    ) {
    }
}

// Subcompositor — no events to handle.
impl Dispatch<wl_subcompositor::WlSubcompositor, ()> for HostState {
    fn event(
        _state: &mut Self,
        _proxy: &wl_subcompositor::WlSubcompositor,
        _event: wl_subcompositor::Event,
        _data: &(),
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
    ) {
    }
}

// Subsurface — no events.
impl Dispatch<wl_subsurface::WlSubsurface, ()> for HostState {
    fn event(
        _state: &mut Self,
        _proxy: &wl_subsurface::WlSubsurface,
        _event: wl_subsurface::Event,
        _data: &(),
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
    ) {
    }
}

// wl_output — detect host display refresh rate.
// User-data is the registry `name` (u32) that uniquely identifies this output.
impl Dispatch<wl_output::WlOutput, u32> for HostState {
    fn event(
        state: &mut Self,
        _proxy: &wl_output::WlOutput,
        event: wl_output::Event,
        data: &u32,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
    ) {
        if let wl_output::Event::Mode {
            flags,
            width,
            height,
            refresh,
        } = event
        {
            // `refresh` is in millihertz (mHz). Only use the preferred/current mode.
            let is_preferred = flags
                .into_result()
                .map(|f| f.intersects(wl_output::Mode::Preferred | wl_output::Mode::Current))
                .unwrap_or(false);
            if is_preferred && refresh > 0 {
                let output_name = *data;
                let hz = (refresh as u32 + 500) / 1000;
                info!(
                    output_name,
                    refresh_mhz = refresh,
                    refresh_hz = hz,
                    mode_width = width,
                    mode_height = height,
                    "host output mode detected"
                );
                state.output_refresh_rates.insert(output_name, refresh);
                // If the surface is already on this output, update immediately.
                if state.surface_outputs.contains(&output_name) {
                    state.update_active_refresh_rate();
                }
            }
        }
    }
}

// Surface events — track which outputs our window is displayed on.
impl Dispatch<wl_surface::WlSurface, ()> for HostState {
    fn event(
        state: &mut Self,
        _proxy: &wl_surface::WlSurface,
        event: wl_surface::Event,
        _data: &(),
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
    ) {
        match event {
            wl_surface::Event::Enter { output } => {
                // Identify the output by its user-data (registry name).
                if let Some(&name) = output.data::<u32>()
                    && !state.surface_outputs.contains(&name)
                {
                    state.surface_outputs.push(name);
                    info!(output_name = name, "surface entered output");
                    state.update_active_refresh_rate();
                }
            }
            wl_surface::Event::Leave { output } => {
                if let Some(&name) = output.data::<u32>() {
                    state.surface_outputs.retain(|&n| n != name);
                    info!(output_name = name, "surface left output");
                    state.update_active_refresh_rate();
                }
            }
            _ => {}
        }
    }
}

// SHM events.
impl Dispatch<wl_shm::WlShm, ()> for HostState {
    fn event(
        _state: &mut Self,
        _proxy: &wl_shm::WlShm,
        _event: wl_shm::Event,
        _data: &(),
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
    ) {
    }
}

impl Dispatch<wl_shm_pool::WlShmPool, ()> for HostState {
    fn event(
        _state: &mut Self,
        _proxy: &wl_shm_pool::WlShmPool,
        _event: wl_shm_pool::Event,
        _data: &(),
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
    ) {
    }
}

impl Dispatch<wl_region::WlRegion, ()> for HostState {
    fn event(
        _state: &mut Self,
        _proxy: &wl_region::WlRegion,
        _event: wl_region::Event,
        _data: &(),
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
    ) {
    }
}

impl Dispatch<wayland_client::protocol::wl_buffer::WlBuffer, ()> for HostState {
    fn event(
        _state: &mut Self,
        _proxy: &wayland_client::protocol::wl_buffer::WlBuffer,
        event: wayland_client::protocol::wl_buffer::Event,
        _data: &(),
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
    ) {
        if let wayland_client::protocol::wl_buffer::Event::Release = event {
            trace!("wayland: host released wl_buffer");
        }
    }
}

/// `DRM_FORMAT_MOD_INVALID` — the driver chooses the tiling format.
const DRM_FORMAT_MOD_INVALID: u64 = 0x00ff_ffff_ffff_ffff;

// ZwpLinuxDmabufV1 — format/modifier advertisements from host.
impl Dispatch<ZwpLinuxDmabufV1, ()> for HostState {
    fn event(
        state: &mut Self,
        _proxy: &ZwpLinuxDmabufV1,
        event: zwp_linux_dmabuf_v1::Event,
        _data: &(),
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
    ) {
        match event {
            zwp_linux_dmabuf_v1::Event::Format { format } => {
                trace!(format = format!("0x{:08x}", format), "host dmabuf format");
            }
            zwp_linux_dmabuf_v1::Event::Modifier {
                format,
                modifier_hi,
                modifier_lo,
            } => {
                let modifier = ((modifier_hi as u64) << 32) | modifier_lo as u64;
                trace!(
                    format = format!("0x{:08x}", format),
                    modifier = format!("0x{:016x}", modifier),
                    "host dmabuf format+modifier"
                );
                state
                    .host_dmabuf_formats
                    .entry(format)
                    .or_default()
                    .push(modifier);
                if modifier != DRM_FORMAT_MOD_INVALID {
                    state.can_use_modifiers = true;
                }
            }
            _ => {}
        }
    }
}

// ZwpLinuxBufferParamsV1 — buffer creation results.
impl Dispatch<ZwpLinuxBufferParamsV1, ()> for HostState {
    fn event(
        _state: &mut Self,
        _proxy: &ZwpLinuxBufferParamsV1,
        event: zwp_linux_buffer_params_v1::Event,
        _data: &(),
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
    ) {
        match event {
            zwp_linux_buffer_params_v1::Event::Created { buffer } => {
                info!("host dmabuf buffer created via async path");
                let _ = buffer;
            }
            zwp_linux_buffer_params_v1::Event::Failed => {
                warn!("host dmabuf buffer creation FAILED");
            }
            _ => {}
        }
    }
}

// XDG WM Base — respond to ping.
impl Dispatch<xdg_wm_base::XdgWmBase, ()> for HostState {
    fn event(
        _state: &mut Self,
        wm_base: &xdg_wm_base::XdgWmBase,
        event: xdg_wm_base::Event,
        _data: &(),
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
    ) {
        if let xdg_wm_base::Event::Ping { serial } = event {
            wm_base.pong(serial);
        }
    }
}

// XDG Surface — handle configure.
impl Dispatch<xdg_surface::XdgSurface, ()> for HostState {
    fn event(
        state: &mut Self,
        xdg_surface: &xdg_surface::XdgSurface,
        event: xdg_surface::Event,
        _data: &(),
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
    ) {
        if let xdg_surface::Event::Configure { serial } = event {
            xdg_surface.ack_configure(serial);
            state.configured = true;
            debug!(serial, "host xdg_surface configured");
        }
    }
}

// XDG Toplevel — handle configure + close.
impl Dispatch<xdg_toplevel::XdgToplevel, ()> for HostState {
    fn event(
        state: &mut Self,
        _toplevel: &xdg_toplevel::XdgToplevel,
        event: xdg_toplevel::Event,
        _data: &(),
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
    ) {
        match event {
            xdg_toplevel::Event::Configure {
                width,
                height,
                states,
                ..
            } => {
                info!(
                    width,
                    height,
                    states_len = states.len(),
                    "xdg_toplevel configure from host"
                );

                // Compute our desired logical window size from the physical
                // pixel dimensions and the current fractional scale.
                let frac = state.fractional_scale.max(120);
                let desired_logical_w =
                    (state.desired_width as u64 * 120).div_ceil(frac as u64) as u32;
                let desired_logical_h =
                    (state.desired_height as u64 * 120).div_ceil(frac as u64) as u32;

                // When the host sends 0×0 it means "use your preferred size".
                // Otherwise accept the host's actual dimensions — it may be
                // constrained by decorations, screen size, or tiling.
                let new_w = if width > 0 {
                    width as u32
                } else {
                    desired_logical_w
                };
                let new_h = if height > 0 {
                    height as u32
                } else {
                    desired_logical_h
                };
                if new_w != state.width || new_h != state.height {
                    info!(
                        old_w = state.width,
                        old_h = state.height,
                        new_w,
                        new_h,
                        desired_logical_w,
                        desired_logical_h,
                        fractional_scale = frac,
                        "host resize: accepting host configure"
                    );
                    state.width = new_w;
                    state.height = new_h;
                    // Compute physical pixel dimensions
                    let phys_w = ((new_w as u64 * frac as u64) / 120) as u32;
                    let phys_h = ((new_h as u64 * frac as u64) / 120) as u32;
                    let _ = state.tx.send(WaylandEvent::Resized {
                        width: new_w,
                        height: new_h,
                        physical_width: phys_w,
                        physical_height: phys_h,
                    });
                }
            }
            xdg_toplevel::Event::Close => {
                info!("host window close requested");
                state.closed = true;
                let _ = state.tx.send(WaylandEvent::CloseRequested);
            }
            _ => {}
        }
    }
}

// wp_viewporter — no events, just a factory for wp_viewport objects.
impl Dispatch<wp_viewporter::WpViewporter, ()> for HostState {
    fn event(
        _state: &mut Self,
        _proxy: &wp_viewporter::WpViewporter,
        _event: wp_viewporter::Event,
        _data: &(),
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
    ) {
    }
}

// wp_viewport — no events, only requests (set_source, set_destination).
impl Dispatch<wp_viewport::WpViewport, ()> for HostState {
    fn event(
        _state: &mut Self,
        _proxy: &wp_viewport::WpViewport,
        _event: wp_viewport::Event,
        _data: &(),
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
    ) {
    }
}

// zxdg_decoration_manager_v1 — no events, just a factory.
impl Dispatch<zxdg_decoration_manager_v1::ZxdgDecorationManagerV1, ()> for HostState {
    fn event(
        _state: &mut Self,
        _proxy: &zxdg_decoration_manager_v1::ZxdgDecorationManagerV1,
        _event: zxdg_decoration_manager_v1::Event,
        _data: &(),
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
    ) {
    }
}

// zxdg_toplevel_decoration_v1 — receives configure events with the negotiated mode.
impl Dispatch<zxdg_toplevel_decoration_v1::ZxdgToplevelDecorationV1, ()> for HostState {
    fn event(
        _state: &mut Self,
        _proxy: &zxdg_toplevel_decoration_v1::ZxdgToplevelDecorationV1,
        event: zxdg_toplevel_decoration_v1::Event,
        _data: &(),
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
    ) {
        if let zxdg_toplevel_decoration_v1::Event::Configure { mode } = event {
            info!(
                mode = ?mode,
                "host decoration mode configured"
            );
        }
    }
}

// wp_fractional_scale_manager_v1 — no events.
impl Dispatch<wp_fractional_scale_manager_v1::WpFractionalScaleManagerV1, ()> for HostState {
    fn event(
        _state: &mut Self,
        _proxy: &wp_fractional_scale_manager_v1::WpFractionalScaleManagerV1,
        _event: wp_fractional_scale_manager_v1::Event,
        _data: &(),
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
    ) {
    }
}

// wp_fractional_scale_v1 — receives preferred_scale events.
impl Dispatch<wp_fractional_scale_v1::WpFractionalScaleV1, ()> for HostState {
    fn event(
        state: &mut Self,
        _proxy: &wp_fractional_scale_v1::WpFractionalScaleV1,
        event: wp_fractional_scale_v1::Event,
        _data: &(),
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
    ) {
        if let wp_fractional_scale_v1::Event::PreferredScale { scale } = event {
            info!(
                scale,
                effective = format!("{:.2}", scale as f64 / 120.0),
                "host fractional scale changed"
            );
            state.fractional_scale = scale;
        }
    }
}

// ─── Host input: wl_seat / wl_keyboard / wl_pointer ─────────────────

// wl_seat — acquire keyboard and pointer when capabilities are advertised.
impl Dispatch<wl_seat::WlSeat, ()> for HostState {
    fn event(
        state: &mut Self,
        seat: &wl_seat::WlSeat,
        event: wl_seat::Event,
        _data: &(),
        _conn: &Connection,
        qh: &QueueHandle<Self>,
    ) {
        if let wl_seat::Event::Capabilities { capabilities } = event {
            let caps = wl_seat::Capability::from_bits_truncate(capabilities.into());
            if caps.contains(wl_seat::Capability::Keyboard) {
                seat.get_keyboard(qh, ());
                info!("acquired host wl_keyboard");
            }
            if caps.contains(wl_seat::Capability::Pointer) {
                let ptr = seat.get_pointer(qh, ());
                state.pointer = Some(ptr);
                info!("acquired host wl_pointer");
            }
        }
    }
}

// wl_keyboard — forward key, keymap, and modifier events to the main thread.
impl Dispatch<wl_keyboard::WlKeyboard, ()> for HostState {
    fn event(
        state: &mut Self,
        _proxy: &wl_keyboard::WlKeyboard,
        event: wl_keyboard::Event,
        _data: &(),
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
    ) {
        match event {
            wl_keyboard::Event::Keymap { format, fd, size } => {
                let format_val = match format {
                    wayland_client::WEnum::Value(wl_keyboard::KeymapFormat::XkbV1) => 1,
                    wayland_client::WEnum::Value(wl_keyboard::KeymapFormat::NoKeymap) => 0,
                    _ => return,
                };
                // The fd is owned by us for the duration of this callback.
                // Convert to OwnedFd to send through the channel.
                let owned = fd;
                info!(format = format_val, size, "received host keymap");
                let _ = state.tx.send(WaylandEvent::Keymap {
                    format: format_val,
                    fd: owned,
                    size,
                });
            }
            wl_keyboard::Event::Modifiers {
                mods_depressed,
                mods_latched,
                mods_locked,
                group,
                ..
            } => {
                let _ = state.tx.send(WaylandEvent::Modifiers {
                    mods_depressed,
                    mods_latched,
                    mods_locked,
                    group,
                });
            }
            wl_keyboard::Event::Key {
                key,
                state: key_state,
                ..
            } => {
                let pressed =
                    key_state == wayland_client::WEnum::Value(wl_keyboard::KeyState::Pressed);
                let _ = state.tx.send(WaylandEvent::Key { key, pressed });
            }
            wl_keyboard::Event::Enter { .. } => {
                let _ = state.tx.send(WaylandEvent::FocusIn);
            }
            wl_keyboard::Event::Leave { .. } => {
                let _ = state.tx.send(WaylandEvent::FocusOut);
            }
            _ => {}
        }
    }
}

// wl_pointer — forward motion, button, and axis events to the main thread.
impl Dispatch<wl_pointer::WlPointer, ()> for HostState {
    fn event(
        state: &mut Self,
        _proxy: &wl_pointer::WlPointer,
        event: wl_pointer::Event,
        _data: &(),
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
    ) {
        match event {
            wl_pointer::Event::Motion {
                surface_x,
                surface_y,
                ..
            } => {
                let (bx, by) = state.host_to_buffer_coords(surface_x, surface_y);
                let _ = state.tx.send(WaylandEvent::PointerMotion { x: bx, y: by });
            }
            wl_pointer::Event::Button {
                button,
                state: btn_state,
                ..
            } => {
                let pressed =
                    btn_state == wayland_client::WEnum::Value(wl_pointer::ButtonState::Pressed);
                let _ = state
                    .tx
                    .send(WaylandEvent::PointerButton { button, pressed });
            }
            wl_pointer::Event::Axis { axis, value, .. } => {
                let (dx, dy) = match axis {
                    wayland_client::WEnum::Value(wl_pointer::Axis::VerticalScroll) => (0.0, value),
                    wayland_client::WEnum::Value(wl_pointer::Axis::HorizontalScroll) => {
                        (value, 0.0)
                    }
                    _ => (0.0, 0.0),
                };
                if dx != 0.0 || dy != 0.0 {
                    let _ = state.tx.send(WaylandEvent::Scroll { dx, dy });
                }
            }
            wl_pointer::Event::Enter {
                serial,
                surface_x,
                surface_y,
                ..
            } => {
                state.pointer_enter_serial = serial;
                let (bx, by) = state.host_to_buffer_coords(surface_x, surface_y);
                let _ = state.tx.send(WaylandEvent::PointerMotion { x: bx, y: by });
            }
            _ => {}
        }
    }
}
