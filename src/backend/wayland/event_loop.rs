//! Host compositor event loop — runs on the `gamecomp-wayland` thread.
//!
//! Owns the Wayland client connection to the host compositor: creates a
//! window, dispatches events, and forwards committed frames via zero-copy
//! DMA-BUF forwarding or Vulkan blit fallback.

use std::collections::HashMap;
use std::os::unix::io::AsFd;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};

use anyhow::Context;
use tracing::{debug, info, trace, warn};
use wayland_client::protocol::wl_buffer::WlBuffer;
use wayland_client::protocol::wl_subsurface::WlSubsurface;
use wayland_client::protocol::{wl_shm, wl_shm_pool, wl_surface};
use wayland_client::{Connection, EventQueue, QueueHandle};
use wayland_protocols::wp::linux_dmabuf::zv1::client::zwp_linux_buffer_params_v1;
use wayland_protocols::wp::linux_dmabuf::zv1::client::zwp_linux_dmabuf_v1::ZwpLinuxDmabufV1;
use wayland_protocols::wp::viewporter::client::wp_viewport::WpViewport;
use wayland_protocols::xdg::decoration::zv1::client::zxdg_toplevel_decoration_v1;

use super::WaylandEvent;
use super::host_state::HostState;
use crate::backend::BackendError;
use crate::backend::gpu::vulkan_blitter::VulkanBlitter;
use crate::wayland::protocols::{CommittedBuffer, CommittedDmaBufPlane};

/// Parameters for the host compositor event loop thread.
pub(super) struct HostLoopParams {
    pub running: Arc<AtomicBool>,
    pub tx: std::sync::mpsc::Sender<WaylandEvent>,
    pub width: u32,
    pub height: u32,
    pub title: String,
    pub host_display: Option<String>,
    pub committed_rx: Option<std::sync::mpsc::Receiver<CommittedBuffer>>,
    pub detected_refresh_mhz: Arc<AtomicU32>,
    pub host_dmabuf_formats: Arc<parking_lot::Mutex<std::collections::HashMap<u32, Vec<u64>>>>,
}

// ── Scaling helpers ─────────────────────────────────────────────────────────

/// Largest rectangle preserving `src` aspect ratio that fits in `dst`.
/// Uses integer cross-multiplication to avoid floating-point.
#[inline(always)]
fn aspect_fit(src_w: u32, src_h: u32, dst_w: u32, dst_h: u32) -> (u32, u32) {
    if src_w == 0 || src_h == 0 || dst_w == 0 || dst_h == 0 {
        return (dst_w, dst_h);
    }
    if src_w == dst_w && src_h == dst_h {
        return (dst_w, dst_h);
    }
    let lhs = (src_w as u64) * (dst_h as u64);
    let rhs = (src_h as u64) * (dst_w as u64);
    if lhs > rhs {
        let fit_h = ((dst_w as u64) * (src_h as u64) / (src_w as u64)) as u32;
        (dst_w, fit_h.max(1))
    } else {
        let fit_w = ((dst_h as u64) * (src_w as u64) / (src_h as u64)) as u32;
        (fit_w.max(1), dst_h)
    }
}

// ── Surface bundle ──────────────────────────────────────────────────────────

/// Parent + game subsurface and their viewports.
///
/// Layout: parent surface (black SHM background) fills the window,
/// game subsurface is viewport-scaled to aspect-fit within it.
struct Surfaces {
    parent: wl_surface::WlSurface,
    parent_viewport: Option<WpViewport>,
    game: wl_surface::WlSurface,
    game_sub: WlSubsurface,
    game_viewport: Option<WpViewport>,
}

impl Surfaces {
    /// Set game subsurface viewport + position for aspect-fit letterboxing.
    #[inline(always)]
    fn position_game(&self, buf_w: u32, buf_h: u32, win_w: u32, win_h: u32) {
        let (fit_w, fit_h) = aspect_fit(buf_w, buf_h, win_w, win_h);
        let off_x = (win_w.saturating_sub(fit_w)) / 2;
        let off_y = (win_h.saturating_sub(fit_h)) / 2;

        self.game.set_buffer_scale(1);
        if let Some(ref gvp) = self.game_viewport {
            gvp.set_source(0.0, 0.0, buf_w as f64, buf_h as f64);
            gvp.set_destination(fit_w as i32, fit_h as i32);
        }
        self.game_sub.set_position(off_x as i32, off_y as i32);
    }
}

// ── Presentation state ──────────────────────────────────────────────────────

/// Mutable state threaded through the event loop.
#[allow(clippy::struct_field_names)]
struct PresentState {
    logical_w: u32,
    logical_h: u32,
    prev_w: u32,
    prev_h: u32,
    force_blit: bool,
    vulkan_blitter: Option<VulkanBlitter>,
    blitter_width: u32,
    blitter_height: u32,
    output_buffer_cache: HashMap<usize, WlBuffer>,
    in_flight_buffers: std::collections::VecDeque<WlBuffer>,
    current_shm_frame: Option<ShmFrame>,
    first_dmabuf_logged: bool,
    present_count: u64,
}

#[allow(clippy::too_many_arguments)]
impl PresentState {
    /// React to host compositor resize (xdg_toplevel.configure).
    fn handle_resize(
        &mut self,
        host_state: &HostState,
        surfaces: &Surfaces,
        shm: &wl_shm::WlShm,
        qh: &QueueHandle<HostState>,
    ) {
        if host_state.width == self.prev_w && host_state.height == self.prev_h {
            return;
        }
        let (w, h) = (host_state.width, host_state.height);
        info!(
            old_w = self.logical_w,
            old_h = self.logical_h,
            w,
            h,
            "wayland: host resize detected"
        );
        self.prev_w = w;
        self.prev_h = h;
        self.logical_w = w;
        self.logical_h = h;

        // Update xdg geometry.
        if let Some(ref xdg_surf) = host_state.xdg_surface {
            xdg_surf.set_window_geometry(0, 0, w as i32, h as i32);
        }

        // Re-present black SHM background at new size.
        if let Ok(frame) = present_shm_frame(shm, &surfaces.parent, qh, w, h, None) {
            self.current_shm_frame = Some(frame);
        }

        // Update parent viewport.
        if let Some(ref vp) = surfaces.parent_viewport {
            vp.set_source(0.0, 0.0, w as f64, h as f64);
            vp.set_destination(w as i32, h as i32);
        }

        // Invalidate blitter — will recreate lazily on next blit frame.
        self.output_buffer_cache.clear();
        if self.vulkan_blitter.take().is_some() {
            info!(w, h, "wayland: dropped blitter for resize");
        }
        self.blitter_width = w;
        self.blitter_height = h;

        surfaces.parent.commit();
    }

    /// Present a committed buffer on the host. Returns `true` if presented.
    fn present_frame(
        &mut self,
        buffer: CommittedBuffer,
        host_state: &HostState,
        surfaces: &Surfaces,
        shm: &wl_shm::WlShm,
        qh: &QueueHandle<HostState>,
        event_queue: &mut EventQueue<HostState>,
        shm_test_mode: bool,
    ) -> bool {
        if shm_test_mode {
            return self.present_shm_test(buffer, surfaces, shm, qh);
        }
        match buffer {
            CommittedBuffer::DmaBuf {
                planes,
                width,
                height,
                format,
                modifier,
            } => self.present_dmabuf(
                &planes,
                width,
                height,
                format,
                modifier,
                host_state,
                surfaces,
                qh,
                event_queue,
            ),
            CommittedBuffer::Shm {
                pixels,
                width,
                height,
                stride: _,
            } => self.present_shm_content(&pixels, width, height, surfaces, shm, qh),
        }
    }

    /// Zero-copy or blit DMA-BUF presentation.
    fn present_dmabuf(
        &mut self,
        planes: &[CommittedDmaBufPlane],
        width: u32,
        height: u32,
        format: u32,
        modifier: u64,
        host_state: &HostState,
        surfaces: &Surfaces,
        qh: &QueueHandle<HostState>,
        event_queue: &mut EventQueue<HostState>,
    ) -> bool {
        let Some(linux_dmabuf) = &host_state.linux_dmabuf else {
            if self.present_count == 0 {
                warn!("host lacks zwp_linux_dmabuf_v1, dropping DMA-BUF frames");
            }
            return false;
        };

        let needs_blit = self.force_blit || !host_state.host_supports_format(format, modifier);

        if !self.first_dmabuf_logged {
            self.first_dmabuf_logged = true;
            info!(
                needs_blit,
                self.force_blit,
                format = format!("0x{:08x}", format),
                modifier = format!("0x{:016x}", modifier),
                width,
                height,
                "wayland: first DMA-BUF presentation path"
            );
        }

        if needs_blit {
            self.ensure_blitter();
            self.blit_present(
                linux_dmabuf,
                surfaces,
                qh,
                event_queue,
                planes,
                width,
                height,
                format,
                modifier,
            )
        } else {
            self.present_zerocopy(
                linux_dmabuf,
                surfaces,
                qh,
                planes,
                width,
                height,
                format,
                modifier,
            )
        }
    }

    /// Forward client DMA-BUF planes verbatim to the host compositor.
    fn present_zerocopy(
        &mut self,
        linux_dmabuf: &ZwpLinuxDmabufV1,
        surfaces: &Surfaces,
        qh: &QueueHandle<HostState>,
        planes: &[CommittedDmaBufPlane],
        width: u32,
        height: u32,
        format: u32,
        modifier: u64,
    ) -> bool {
        let params = linux_dmabuf.create_params(qh, ());
        let mod_hi = (modifier >> 32) as u32;
        let mod_lo = modifier as u32;
        for (idx, plane) in planes.iter().enumerate() {
            params.add(
                plane.fd.as_fd(),
                idx as u32,
                plane.offset,
                plane.stride,
                mod_hi,
                mod_lo,
            );
        }
        let host_buffer = params.create_immed(
            width as i32,
            height as i32,
            format,
            zwp_linux_buffer_params_v1::Flags::empty(),
            qh,
            (),
        );

        surfaces.position_game(width, height, self.logical_w, self.logical_h);
        surfaces.game.attach(Some(&host_buffer), 0, 0);
        surfaces.game.damage(0, 0, i32::MAX, i32::MAX);
        surfaces.game.commit();

        // Keep buffer alive until host releases it (typically 1–2 frames).
        self.in_flight_buffers.push_back(host_buffer);
        while self.in_flight_buffers.len() > 4 {
            self.in_flight_buffers.pop_front();
        }
        self.current_shm_frame = None;
        self.present_count += 1;
        trace!(
            self.present_count,
            planes = planes.len(),
            format = format!("0x{:08x}", format),
            modifier = format!("0x{:016x}", modifier),
            "wayland: committed zero-copy frame"
        );
        true
    }

    /// Vulkan-blit fallback: import client DMA-BUF, blit to output image,
    /// export and present on host. Returns `true` if successful.
    fn blit_present(
        &mut self,
        linux_dmabuf: &ZwpLinuxDmabufV1,
        surfaces: &Surfaces,
        qh: &QueueHandle<HostState>,
        event_queue: &mut EventQueue<HostState>,
        planes: &[CommittedDmaBufPlane],
        width: u32,
        height: u32,
        format: u32,
        modifier: u64,
    ) -> bool {
        let Some(blitter) = self.vulkan_blitter.as_mut() else {
            return false;
        };
        let first_plane = &planes[0];
        let exported = match blitter.blit(
            first_plane.fd.as_fd(),
            width,
            height,
            format,
            modifier,
            first_plane.offset,
            first_plane.stride,
        ) {
            Ok(e) => e,
            Err(e) => {
                warn!(error = %e, "wayland: Vulkan blit failed, dropping frame");
                return false;
            }
        };

        let out_idx = if blitter.output_index() == 0 {
            blitter.output_count() - 1
        } else {
            blitter.output_index() - 1
        };

        let host_buffer = self
            .output_buffer_cache
            .entry(out_idx)
            .or_insert_with(|| {
                let params = linux_dmabuf.create_params(qh, ());
                let hi = (exported.modifier >> 32) as u32;
                let lo = exported.modifier as u32;
                for (idx, plane) in exported.planes.iter().enumerate() {
                    params.add(
                        plane.fd.as_fd(),
                        idx as u32,
                        plane.offset,
                        plane.stride,
                        hi,
                        lo,
                    );
                }
                info!(
                    out_idx,
                    modifier = format!("0x{:016x}", exported.modifier),
                    format = format!("0x{:08x}", exported.format),
                    width = exported.width,
                    height = exported.height,
                    "wayland: created host wl_buffer for blit output"
                );
                params.create_immed(
                    exported.width as i32,
                    exported.height as i32,
                    exported.format,
                    zwp_linux_buffer_params_v1::Flags::empty(),
                    qh,
                    (),
                )
            })
            .clone();

        surfaces.position_game(
            exported.width,
            exported.height,
            self.logical_w,
            self.logical_h,
        );
        surfaces.game.attach(Some(&host_buffer), 0, 0);
        surfaces.game.damage(0, 0, i32::MAX, i32::MAX);
        surfaces.game.commit();

        self.present_count += 1;
        self.current_shm_frame = None;
        trace!(
            self.present_count,
            out_idx, "wayland: committed blitted frame"
        );
        let _ = event_queue.flush();
        true
    }

    /// SHM CPU-copy fallback for client SHM buffers.
    fn present_shm_content(
        &mut self,
        pixels: &[u8],
        width: u32,
        height: u32,
        surfaces: &Surfaces,
        shm: &wl_shm::WlShm,
        qh: &QueueHandle<HostState>,
    ) -> bool {
        debug!(
            width,
            height,
            bytes = pixels.len(),
            "wayland: presenting SHM frame"
        );
        let Ok(frame) = present_shm_frame(shm, &surfaces.game, qh, width, height, Some(pixels))
        else {
            return false;
        };
        surfaces.position_game(width, height, self.logical_w, self.logical_h);
        surfaces.game.commit();
        self.current_shm_frame = Some(frame);
        self.in_flight_buffers.clear();
        self.present_count += 1;
        true
    }

    /// Diagnostic: present a solid red SHM frame instead of forwarding content.
    fn present_shm_test(
        &mut self,
        buffer: CommittedBuffer,
        surfaces: &Surfaces,
        shm: &wl_shm::WlShm,
        qh: &QueueHandle<HostState>,
    ) -> bool {
        let (w, h) = match &buffer {
            CommittedBuffer::DmaBuf { width, height, .. } => (*width, *height),
            CommittedBuffer::Shm { width, height, .. } => (*width, *height),
        };
        let pixel_count = (w * h) as usize;
        let mut red_pixels = vec![0u8; pixel_count * 4];
        for pixel in red_pixels.chunks_exact_mut(4) {
            pixel[0] = 0x00; // B
            pixel[1] = 0x00; // G
            pixel[2] = 0xFF; // R
            pixel[3] = 0xFF; // X
        }
        let Ok(frame) = present_shm_frame(shm, &surfaces.game, qh, w, h, Some(&red_pixels)) else {
            return false;
        };
        surfaces.position_game(w, h, self.logical_w, self.logical_h);
        surfaces.game.commit();
        self.current_shm_frame = Some(frame);
        self.in_flight_buffers.clear();
        self.present_count += 1;
        debug!(self.present_count, "wayland: SHM test frame committed");
        true
    }

    /// Lazy-init the Vulkan blitter on first blit-needing frame.
    fn ensure_blitter(&mut self) {
        if self.vulkan_blitter.is_some() {
            return;
        }
        self.vulkan_blitter = match VulkanBlitter::new(self.blitter_width, self.blitter_height) {
            Ok(b) => {
                info!(
                    w = self.blitter_width,
                    h = self.blitter_height,
                    "wayland: Vulkan blitter initialized on demand"
                );
                Some(b)
            }
            Err(e) => {
                warn!(error = %e, "wayland: Vulkan blitter init failed");
                None
            }
        };
    }
}

// ── SHM buffer lifetime ─────────────────────────────────────────────────────

/// SHM buffer resources kept alive until the host replaces the buffer.
/// Dropping sends `wl_buffer.destroy` + `wl_shm_pool.destroy`.
struct ShmFrame {
    _buffer: WlBuffer,
    _pool: wl_shm_pool::WlShmPool,
}

// ── Entry point ─────────────────────────────────────────────────────────────

/// Host compositor event loop (runs on `gamecomp-wayland` thread).
pub(super) fn wayland_event_loop(params: HostLoopParams) {
    info!(?params.host_display, "wayland event thread started");

    if let Err(e) = run_host_connection(
        &params.running,
        params.tx,
        params.width,
        params.height,
        &params.title,
        params.host_display,
        params.committed_rx,
        params.detected_refresh_mhz,
        params.host_dmabuf_formats,
    ) {
        tracing::error!(?e, "wayland event thread failed");
    }

    // Release: so the render thread's Acquire load sees shutdown.
    params.running.store(false, Ordering::Release);
    info!("wayland event thread exited");
}

// ── Host connection setup ───────────────────────────────────────────────────

/// Connect to the host Wayland compositor. Returns a live connection.
fn connect_to_host(host_display: &Option<String>) -> anyhow::Result<Connection> {
    if let Some(display) = host_display {
        let runtime_dir = std::env::var("XDG_RUNTIME_DIR").context("XDG_RUNTIME_DIR not set")?;
        let socket_path = if display.starts_with('/') {
            std::path::PathBuf::from(display)
        } else {
            std::path::PathBuf::from(&runtime_dir).join(display)
        };
        info!(path = %socket_path.display(), "connecting to host compositor");
        let stream = std::os::unix::net::UnixStream::connect(&socket_path)
            .context("failed to connect to host compositor socket")?;
        Connection::from_socket(stream).context("failed to create Wayland connection from socket")
    } else {
        Connection::connect_to_env()
            .context("failed to connect to host Wayland compositor — no host display available")
    }
}

/// Bind globals, discover DMA-BUF formats, and publish them to the shared map.
fn init_host_state(
    event_queue: &mut EventQueue<HostState>,
    host_state: &mut HostState,
    qh: &QueueHandle<HostState>,
    display: &wayland_client::protocol::wl_display::WlDisplay,
    shared_host_formats: &Arc<parking_lot::Mutex<HashMap<u32, Vec<u64>>>>,
) -> anyhow::Result<()> {
    display.get_registry(qh, ());

    // First roundtrip: bind globals. Second: receive dmabuf format events.
    event_queue
        .roundtrip(host_state)
        .context("failed to roundtrip host compositor")?;
    event_queue
        .roundtrip(host_state)
        .context("failed second roundtrip for dmabuf formats")?;

    // Publish host formats for client advertisement.
    let mut shared = shared_host_formats.lock();
    *shared = host_state.host_dmabuf_formats.clone();
    info!(
        formats = shared.len(),
        modifiers_total = shared.values().map(|v| v.len()).sum::<usize>(),
        can_use_modifiers = host_state.can_use_modifiers,
        "published host dmabuf formats"
    );
    Ok(())
}

/// Create the host window: parent surface, xdg shell, decorations,
/// game subsurface, and viewports. Returns the surface bundle.
fn create_host_window(
    host_state: &mut HostState,
    event_queue: &mut EventQueue<HostState>,
    qh: &QueueHandle<HostState>,
    title: &str,
) -> anyhow::Result<(Surfaces, wl_shm::WlShm)> {
    let compositor = host_state
        .compositor
        .clone()
        .context("host compositor missing wl_compositor")?;
    let wm_base = host_state
        .wm_base
        .as_ref()
        .context("host compositor missing xdg_wm_base")?;
    let shm = host_state
        .shm
        .as_ref()
        .context("host compositor missing wl_shm")?
        .clone();

    // Parent surface.
    let surface = compositor.create_surface(qh, ());
    host_state.surface = Some(surface.clone());
    surface.set_buffer_scale(1);

    // Mark fully opaque.
    let opaque_region = compositor.create_region(qh, ());
    opaque_region.add(0, 0, i32::MAX, i32::MAX);
    surface.set_opaque_region(Some(&opaque_region));

    // Viewport for 1:1 buffer mapping regardless of display scale.
    let viewport = host_state.viewporter.as_ref().map(|vp| {
        let viewport = vp.get_viewport(&surface, qh, ());
        info!("wayland: created wp_viewport for surface");
        viewport
    });

    // Fractional scale events.
    if let Some(mgr) = host_state.fractional_scale_mgr.as_ref() {
        mgr.get_fractional_scale(&surface, qh, ());
    }

    // xdg shell.
    let xdg_surface = wm_base.get_xdg_surface(&surface, qh, ());
    let xdg_toplevel = xdg_surface.get_toplevel(qh, ());
    xdg_toplevel.set_title(title.to_string());
    xdg_toplevel.set_app_id("gamecomp".to_string());

    // Request server-side decorations.
    if let Some(ref decoration_mgr) = host_state.decoration_mgr {
        let deco = decoration_mgr.get_toplevel_decoration(&xdg_toplevel, qh, ());
        deco.set_mode(zxdg_toplevel_decoration_v1::Mode::ServerSide);
        info!("wayland: requested server-side decorations");
    } else {
        warn!("wayland: host does not support zxdg_decoration_manager_v1");
    }

    host_state.xdg_surface = Some(xdg_surface);
    host_state.xdg_toplevel = Some(xdg_toplevel);

    // Commit to trigger initial configure.
    surface.commit();
    event_queue
        .roundtrip(host_state)
        .context("failed to get initial configure")?;

    // Game content subsurface (desync mode).
    let game_surface = compositor.create_surface(qh, ());
    game_surface.set_buffer_scale(1);

    let game_sub = host_state
        .subcompositor
        .as_ref()
        .context("host compositor missing wl_subcompositor")?
        .get_subsurface(&game_surface, &surface, qh, ());
    game_sub.set_desync();

    let game_viewport = host_state.viewporter.as_ref().map(|vp| {
        let gvp = vp.get_viewport(&game_surface, qh, ());
        info!("wayland: created wp_viewport for game subsurface");
        gvp
    });

    // Opaque region on game subsurface.
    let game_opaque = compositor.create_region(qh, ());
    game_opaque.add(0, 0, i32::MAX, i32::MAX);
    game_surface.set_opaque_region(Some(&game_opaque));

    let surfaces = Surfaces {
        parent: surface,
        parent_viewport: viewport,
        game: game_surface,
        game_sub,
        game_viewport,
    };

    Ok((surfaces, shm))
}

// ── Main event loop ─────────────────────────────────────────────────────────

/// Internal: connect to host, create window, run event dispatch.
#[allow(clippy::too_many_arguments)]
fn run_host_connection(
    running: &AtomicBool,
    tx: std::sync::mpsc::Sender<WaylandEvent>,
    width: u32,
    height: u32,
    title: &str,
    host_display: Option<String>,
    committed_rx: Option<std::sync::mpsc::Receiver<CommittedBuffer>>,
    detected_refresh_mhz: Arc<AtomicU32>,
    shared_host_formats: Arc<parking_lot::Mutex<std::collections::HashMap<u32, Vec<u64>>>>,
) -> anyhow::Result<()> {
    let conn = connect_to_host(&host_display)?;
    let display = conn.display();
    let mut event_queue: EventQueue<HostState> = conn.new_event_queue();
    let qh = event_queue.handle();

    let mut host_state = HostState {
        configured: false,
        closed: false,
        width,
        height,
        compositor: None,
        subcompositor: None,
        wm_base: None,
        shm: None,
        linux_dmabuf: None,
        viewporter: None,
        fractional_scale_mgr: None,
        decoration_mgr: None,
        fractional_scale: 120,
        surface: None,
        xdg_surface: None,
        xdg_toplevel: None,
        tx: tx.clone(),
        detected_refresh_mhz,
        output_refresh_rates: HashMap::new(),
        surface_outputs: Vec::new(),
        host_dmabuf_formats: HashMap::new(),
        can_use_modifiers: false,
    };

    init_host_state(
        &mut event_queue,
        &mut host_state,
        &qh,
        &display,
        &shared_host_formats,
    )?;

    let (surfaces, shm) = create_host_window(&mut host_state, &mut event_queue, &qh, title)?;

    // Compute logical dimensions from fractional scale.
    let frac_scale = host_state.fractional_scale;
    let (logical_w, logical_h) = if frac_scale > 120 {
        let lw = (host_state.width * 120).div_ceil(frac_scale);
        let lh = (host_state.height * 120).div_ceil(frac_scale);
        (lw, lh)
    } else {
        (host_state.width, host_state.height)
    };

    info!(
        width = host_state.width,
        height = host_state.height,
        fractional_scale = frac_scale,
        effective_scale = format!("{:.2}", frac_scale as f64 / 120.0),
        logical_w,
        logical_h,
        "host window configured"
    );

    // Set xdg geometry to logical size (prevents fractional upscaling).
    if let Some(ref xdg_surf) = host_state.xdg_surface {
        xdg_surf.set_window_geometry(0, 0, logical_w as i32, logical_h as i32);
    }

    // Present initial black SHM frame at logical size.
    let initial_shm_frame = Some(present_shm_frame(
        &shm,
        &surfaces.parent,
        &qh,
        logical_w,
        logical_h,
        None,
    )?);
    if let Some(vp) = &surfaces.parent_viewport {
        vp.set_source(0.0, 0.0, logical_w as f64, logical_h as f64);
        vp.set_destination(logical_w as i32, logical_h as i32);
    }
    surfaces.parent.commit();

    info!(logical_w, logical_h, "wayland: game subsurface ready");

    // Normalize host_state to logical coordinates. All subsequent
    // xdg_toplevel.configure events use logical coordinates.
    host_state.width = logical_w;
    host_state.height = logical_h;

    let mut state = PresentState {
        logical_w,
        logical_h,
        prev_w: logical_w,
        prev_h: logical_h,
        force_blit: std::env::var("GAMECOMP_FORCE_BLIT").is_ok(),
        vulkan_blitter: None,
        blitter_width: logical_w,
        blitter_height: logical_h,
        output_buffer_cache: HashMap::new(),
        in_flight_buffers: std::collections::VecDeque::with_capacity(4),
        current_shm_frame: initial_shm_frame,
        first_dmabuf_logged: false,
        present_count: 0,
    };

    let shm_test_mode = std::env::var("GAMECOMP_SHM_TEST").is_ok();
    if shm_test_mode {
        info!("wayland: SHM test mode enabled");
    }

    let mut loop_count: u64 = 0;
    let mut fps_frame_count: u64 = 0;
    let mut fps_start = std::time::Instant::now();
    let mut last_status = std::time::Instant::now();

    debug!(
        has_committed_rx = committed_rx.is_some(),
        has_linux_dmabuf = host_state.linux_dmabuf.is_some(),
        configured = host_state.configured,
        shm_test_mode,
        "wayland: entering event loop"
    );

    // ── Dispatch loop ──
    while running.load(Ordering::Acquire) && !host_state.closed {
        loop_count += 1;

        // Non-blocking dispatch.
        event_queue
            .dispatch_pending(&mut host_state)
            .context("dispatch_pending failed")?;
        if let Some(guard) = event_queue.prepare_read() {
            let _ = guard.read();
            event_queue
                .dispatch_pending(&mut host_state)
                .context("dispatch_pending after read failed")?;
        }
        let _ = event_queue.flush();

        // Handle resize.
        state.handle_resize(&host_state, &surfaces, &shm, &qh);
        if state.prev_w != host_state.width || state.prev_h != host_state.height {
            // handle_resize already ran above, but we need to flush.
            let _ = event_queue.flush();
        }

        // Present committed buffers.
        let mut presented = false;
        if let Some(ref rx) = committed_rx {
            let mut latest: Option<CommittedBuffer> = None;
            let mut drain_count = 0u32;
            while let Ok(buf) = rx.try_recv() {
                latest = Some(buf);
                drain_count += 1;
            }
            if drain_count > 0 {
                trace!(drain_count, "wayland: drained buffers");
            }
            if let (Some(buffer), true) = (latest, host_state.surface.is_some()) {
                presented = state.present_frame(
                    buffer,
                    &host_state,
                    &surfaces,
                    &shm,
                    &qh,
                    &mut event_queue,
                    shm_test_mode,
                );
                if presented {
                    let _ = event_queue.flush();
                }
            }
        } else if loop_count == 1 {
            warn!("wayland: no committed_rx channel — will never receive frames");
        }

        // Periodic FPS logging.
        if last_status.elapsed() >= std::time::Duration::from_secs(2) {
            let elapsed = fps_start.elapsed().as_secs_f64();
            let fps = if elapsed > 0.0 {
                fps_frame_count as f64 / elapsed
            } else {
                0.0
            };
            info!(
                fps = format!("{fps:.1}"),
                present_count = state.present_count,
                loop_count,
                "wayland: status"
            );
            fps_frame_count = 0;
            fps_start = std::time::Instant::now();
            last_status = std::time::Instant::now();
        }

        if presented {
            fps_frame_count += 1;
        } else {
            // Brief yield (~4 kHz polling) to avoid busy-spinning.
            // TODO: Replace with poll() on eventfd for zero-wait wake-up.
            std::thread::sleep(std::time::Duration::from_micros(250));
        }
    }

    Ok(())
}

// ── SHM frame helper ────────────────────────────────────────────────────────

/// Present a frame via wl_shm. If `pixels` is `Some`, copies client data;
/// otherwise fills with black. Caller must keep the [`ShmFrame`] alive until
/// the next `surface.commit()` replaces it.
fn present_shm_frame(
    shm: &wl_shm::WlShm,
    surface: &wl_surface::WlSurface,
    qh: &QueueHandle<HostState>,
    width: u32,
    height: u32,
    pixels: Option<&[u8]>,
) -> anyhow::Result<ShmFrame> {
    let stride = width * 4;
    let size = (stride * height) as usize;

    let name = c"gamecomp-shm";
    let fd = rustix::fs::memfd_create(name, rustix::fs::MemfdFlags::CLOEXEC)
        .context("failed to create memfd")?;
    rustix::fs::ftruncate(&fd, size as u64).context("failed to truncate memfd")?;

    // SAFETY: Valid fd, valid size, no concurrent access.
    let ptr = unsafe {
        libc::mmap(
            std::ptr::null_mut(),
            size,
            libc::PROT_READ | libc::PROT_WRITE,
            libc::MAP_SHARED,
            rustix::fd::AsRawFd::as_raw_fd(&fd),
            0,
        )
    };
    if ptr == libc::MAP_FAILED {
        return Err(BackendError::MmapFailed.into());
    }

    // SAFETY: ptr is valid for `size` bytes, munmap'd immediately after write.
    unsafe {
        if let Some(px) = pixels {
            let copy_len = px.len().min(size);
            std::ptr::copy_nonoverlapping(px.as_ptr(), ptr as *mut u8, copy_len);
        } else {
            let buf = std::slice::from_raw_parts_mut(ptr as *mut u32, (width * height) as usize);
            buf.fill(0xFF00_0000);
        }
        libc::munmap(ptr, size);
    }

    let pool = shm.create_pool(rustix::fd::AsFd::as_fd(&fd), size as i32, qh, ());
    let buffer = pool.create_buffer(
        0,
        width as i32,
        height as i32,
        stride as i32,
        wl_shm::Format::Xrgb8888,
        qh,
        (),
    );

    surface.attach(Some(&buffer), 0, 0);
    surface.damage(0, 0, i32::MAX, i32::MAX);

    Ok(ShmFrame {
        _buffer: buffer,
        _pool: pool,
    })
}
