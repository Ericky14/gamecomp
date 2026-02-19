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
    wl_compositor, wl_output, wl_region, wl_registry, wl_shm, wl_shm_pool, wl_subcompositor,
    wl_subsurface, wl_surface,
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
    pub width: u32,
    pub height: u32,
    pub compositor: Option<wl_compositor::WlCompositor>,
    pub subcompositor: Option<wl_subcompositor::WlSubcompositor>,
    pub wm_base: Option<xdg_wm_base::XdgWmBase>,
    pub shm: Option<wl_shm::WlShm>,
    pub linux_dmabuf: Option<ZwpLinuxDmabufV1>,
    pub viewporter: Option<wp_viewporter::WpViewporter>,
    pub fractional_scale_mgr: Option<wp_fractional_scale_manager_v1::WpFractionalScaleManagerV1>,
    pub decoration_mgr: Option<zxdg_decoration_manager_v1::ZxdgDecorationManagerV1>,
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
}

impl HostState {
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
                if width > 0 && height > 0 {
                    let new_w = width as u32;
                    let new_h = height as u32;
                    if new_w != state.width || new_h != state.height {
                        info!(
                            old_w = state.width,
                            old_h = state.height,
                            new_w,
                            new_h,
                            "host resize: updating dimensions"
                        );
                        state.width = new_w;
                        state.height = new_h;
                        let _ = state.tx.send(WaylandEvent::Resized {
                            width: new_w,
                            height: new_h,
                        });
                    }
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
