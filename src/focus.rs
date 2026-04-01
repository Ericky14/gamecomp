//! Cross-server focus arbitration.
//!
//! When multiple XWayland servers run simultaneously (platform + game),
//! the compositor must decide which server's surface is the "winner" —
//! the one whose commits are presented and whose client receives frame
//! callbacks.
//!
//! The [`FocusArbiter`] implements a 4-phase strategy modelled after
//! gamescope's `determine_and_apply_focus`:
//!
//! 0. **Baselayer priority** — if `GAMESCOPECTRL_BASELAYER_APPID` is set,
//!    prefer the server whose focused AppID matches.
//! 1. **Stealer detection** — if a server just received a new
//!    `STEAM_GAME` atom (app_id changed to non-zero), it steals focus.
//! 2. **Current winner retention** — keep the current winner while it
//!    still has a non-zero app_id.
//! 3. **Fallback** — pick any server with a non-zero app_id.

use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};

use tracing::{debug, trace};

use crate::wayland::xwayland::XwmEvent;

/// Per-XWayland server focus state visible to the arbiter.
///
/// The XWM thread writes these atomics; the main loop reads them.
pub struct ServerFocusState {
    /// Server index (0 = platform, 1+ = game).
    pub index: u32,
    /// Focused app ID for this server (0 = no focused game window).
    pub focused_app_id: Arc<AtomicU32>,
    /// Focused surface protocol ID for this server (0 = none).
    pub focused_wl_surface_id: Arc<AtomicU32>,
}

/// Result of a single arbitration tick.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FocusResult {
    /// Winning app ID (0 = no focus).
    pub app_id: u32,
    /// Winning surface protocol ID (0 = no focus).
    pub surface_id: u32,
    /// Winning server index (`u32::MAX` = no focus).
    pub server_index: u32,
    /// Whether the focus target changed since last tick.
    pub changed: bool,
}

/// Encapsulates all cross-server focus arbitration state.
pub struct FocusArbiter {
    /// Previous per-server app_id snapshots for change detection.
    prev_server_app_ids: Vec<u32>,
    /// Last winning app ID.
    prev_winner_app: u32,
    /// Last winning surface protocol ID.
    prev_surface_id: u32,
    /// Last winning server index.
    prev_server_index: u32,
    /// Baselayer AppIDs requested by an external controller (Steam).
    baselayer_app_ids: Vec<u32>,
}

impl FocusArbiter {
    /// Create a new arbiter for `server_count` XWayland servers.
    pub fn new(server_count: usize) -> Self {
        Self {
            prev_server_app_ids: vec![0; server_count],
            prev_winner_app: 0,
            prev_surface_id: 0,
            prev_server_index: u32::MAX,
            baselayer_app_ids: Vec::new(),
        }
    }

    /// Drain the XWM event channel and update internal state.
    ///
    /// Must be called before [`update`] each tick.
    pub fn drain_events(&mut self, event_rx: &calloop::channel::Channel<XwmEvent>) {
        while let Ok(evt) = event_rx.try_recv() {
            if let XwmEvent::BaselayerAppIdsChanged(ids) = evt {
                debug!(?ids, "focus arbiter: baselayer app IDs updated");
                self.baselayer_app_ids = ids;
                // Reset stealer detection so all servers re-compete.
                // Without this, clearing baselayer would leave the
                // Phase-0 winner stuck in Phase 2 ("keep current winner")
                // because no server's app_id actually changed.
                self.prev_server_app_ids.fill(0);
            }
        }
    }

    /// Run one tick of the 4-phase focus arbitration.
    ///
    /// Reads per-server atomics, applies the priority strategy, and
    /// returns the winning focus target. The caller is responsible for
    /// storing the result into the global shared atomics and firing
    /// frame callbacks on focus changes.
    pub fn update(&mut self, servers: &[ServerFocusState]) -> FocusResult {
        let mut winner_app: u32 = 0;
        let mut winner_surface: u32 = 0;
        let mut winner_server: u32 = u32::MAX;

        // Phase 0: baselayer priority (GAMESCOPECTRL_BASELAYER_APPID).
        let mut baselayer_matched = false;
        if !self.baselayer_app_ids.is_empty() {
            for srv in servers {
                let app = srv.focused_app_id.load(Ordering::Relaxed);
                if app != 0 && self.baselayer_app_ids.contains(&app) {
                    winner_app = app;
                    winner_surface = srv.focused_wl_surface_id.load(Ordering::Relaxed);
                    winner_server = srv.index;
                    baselayer_matched = true;
                    break;
                }
            }
        }

        // Update prev_server_app_ids for stealer detection regardless
        // of whether baselayer matched — we always want accurate history.
        let mut stealer: Option<usize> = None;
        for (i, srv) in servers.iter().enumerate() {
            let app = srv.focused_app_id.load(Ordering::Relaxed);
            let prev = self.prev_server_app_ids[i];
            if app != prev && app != 0 {
                stealer = Some(i);
            }
            self.prev_server_app_ids[i] = app;
        }

        if !baselayer_matched {
            // Phase 1: stealer detection.
            if let Some(i) = stealer {
                let srv = &servers[i];
                winner_app = srv.focused_app_id.load(Ordering::Relaxed);
                winner_surface = srv.focused_wl_surface_id.load(Ordering::Relaxed);
                winner_server = srv.index;
            } else {
                // Phase 2: keep current winner if still valid.
                //
                // A server remains "valid" if either:
                // - It still has a non-zero app_id, OR
                // - Its surface is still alive (surface_id != 0) AND the
                //   baselayer hasn't explicitly switched to another client.
                //
                // The second condition prevents focus flicker when
                // STEAM_GAME transiently clears to 0 during dependency
                // installation (proton, dxdiag, wine patches). The game
                // window is still mapped but the app_id disappears briefly.
                // Without this, the arbiter would bounce focus to Grid and
                // back, causing a visible flicker.
                let cur_srv = self.prev_server_index;
                if cur_srv != u32::MAX
                    && let Some(srv) = servers.iter().find(|s| s.index == cur_srv)
                {
                    let app = srv.focused_app_id.load(Ordering::Relaxed);
                    let surf = srv.focused_wl_surface_id.load(Ordering::Relaxed);
                    if app != 0 {
                        winner_app = app;
                        winner_surface = surf;
                        winner_server = cur_srv;
                    } else if surf != 0 && !self.baselayer_app_ids.is_empty() {
                        // Surface still alive but app_id cleared. Retain
                        // focus to avoid flicker — use the previous app_id
                        // for the focus feedback atoms.
                        winner_app = self.prev_winner_app;
                        winner_surface = surf;
                        winner_server = cur_srv;
                        debug!(
                            winner_app,
                            surf,
                            cur_srv,
                            "focus arbiter: retaining winner despite app_id=0 (surface still alive)"
                        );
                    }
                }
                // Phase 3: fallback — any server with non-zero app_id.
                if winner_app == 0 {
                    for srv in servers {
                        let app = srv.focused_app_id.load(Ordering::Relaxed);
                        if app != 0 {
                            winner_app = app;
                            winner_surface = srv.focused_wl_surface_id.load(Ordering::Relaxed);
                            winner_server = srv.index;
                            break;
                        }
                    }
                }
            }
        }

        let changed =
            winner_surface != self.prev_surface_id || winner_server != self.prev_server_index;

        if changed {
            let phase = if baselayer_matched {
                "baselayer"
            } else if stealer.is_some() {
                "stealer"
            } else if winner_app != 0 && self.prev_server_index != u32::MAX {
                "retention"
            } else if winner_app != 0 {
                "fallback"
            } else {
                "none"
            };
            debug!(
                phase,
                winner_app,
                winner_surface,
                winner_server,
                prev_surface = self.prev_surface_id,
                prev_server = self.prev_server_index,
                ?self.baselayer_app_ids,
                "focus arbiter: winner changed"
            );
            // Log all server states for diagnosis.
            for (i, srv) in servers.iter().enumerate() {
                let app = srv.focused_app_id.load(Ordering::Relaxed);
                let surf = srv.focused_wl_surface_id.load(Ordering::Relaxed);
                trace!(
                    slot = i,
                    server_index = srv.index,
                    app,
                    surf,
                    "focus arbiter: server state"
                );
            }
        }

        self.prev_surface_id = winner_surface;
        self.prev_server_index = winner_server;
        self.prev_winner_app = winner_app;

        FocusResult {
            app_id: winner_app,
            surface_id: winner_surface,
            server_index: winner_server,
            changed,
        }
    }
}

#[cfg(test)]
#[path = "focus_tests.rs"]
mod tests;
