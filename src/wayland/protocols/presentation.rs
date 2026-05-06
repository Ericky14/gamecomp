//! `wp_presentation_time` protocol implementation (version 2).
//!
//! Provides accurate frame presentation feedback to clients. Most
//! critically, **XWayland uses this to drive its X11 Present extension**:
//! without it, X11 clients (GTK, Flutter, Chromium) fall back to a
//! software 60Hz-ish vsync timer and render at unstable, low frame rates.
//!
//! # Pipeline
//!
//! 1. Client binds `wp_presentation` → server sends `clock_id(CLOCK_MONOTONIC)`.
//! 2. Client calls `wp_presentation.feedback(surface, callback)` → server
//!    queues the callback on `SurfaceData.pending_feedbacks`.
//! 3. On `wl_surface.commit` with a buffer: queued feedbacks move onto the
//!    `WaylandState.staged_feedbacks`, associated with the staged buffer.
//!    If a newer commit replaces an un-forwarded staged frame, the
//!    displaced feedbacks are `discarded()`.
//! 4. On forward (staged → render thread): staged feedbacks become
//!    in-flight feedbacks, awaiting the page flip.
//! 5. On page flip completion: each in-flight feedback emits `sync_output`
//!    for every bound `wl_output` of its client, then `presented(time,
//!    refresh, sequence, vsync|hw_clock|hw_completion)`.
//! 6. On session inactive or client surface destroy: in-flight / pending
//!    feedbacks are `discarded()` — the frame will never be displayed.
//!
//! # Spec compliance notes
//!
//! * `sync_output` is sent before `presented` for each `wl_output` the
//!   client has bound. Without this, XWayland cannot associate the
//!   presentation event with an X11 screen and falls back to software
//!   vsync timing.
//! * `refresh` is the predicted nanoseconds to the next vblank.
//! * `seq_hi`/`seq_lo` is a monotonic MSC counter — X11 Present requires
//!   this to be strictly increasing.
//! * `presented` and `discarded` are destructor events on
//!   `wp_presentation_feedback` — the object is auto-destroyed after
//!   either fires.

use tracing::{info, trace};
use wayland_protocols::wp::presentation_time::server::{
    wp_presentation::{self, WpPresentation},
    wp_presentation_feedback::{self, WpPresentationFeedback},
};
use wayland_server::{Client, DataInit, Dispatch, DisplayHandle, GlobalDispatch, New, Resource};

use crate::wayland::WaylandState;
use crate::wayland::protocols::SurfaceData;

/// `CLOCK_MONOTONIC` value defined by `<time.h>`. Sent to clients on bind
/// so they know which clock our presentation timestamps use.
const CLOCK_MONOTONIC: u32 = 1;

impl GlobalDispatch<WpPresentation, ()> for WaylandState {
    fn bind(
        _state: &mut Self,
        _dh: &DisplayHandle,
        client: &Client,
        resource: New<WpPresentation>,
        _data: &(),
        data_init: &mut DataInit<'_, Self>,
    ) {
        let presentation = data_init.init(resource, ());
        presentation.clock_id(CLOCK_MONOTONIC);
        info!(
            client_id = ?client.id(),
            "wp_presentation: client bound (clock_id=CLOCK_MONOTONIC)"
        );
    }
}

impl Dispatch<WpPresentation, ()> for WaylandState {
    fn request(
        state: &mut Self,
        _client: &Client,
        _resource: &WpPresentation,
        request: wp_presentation::Request,
        _data: &(),
        _dh: &DisplayHandle,
        data_init: &mut DataInit<'_, Self>,
    ) {
        match request {
            wp_presentation::Request::Feedback { surface, callback } => {
                let feedback = data_init.init(callback, ());
                if let Some(data) = surface.data::<SurfaceData>() {
                    state.presentation_requests += 1;
                    let mut pending = data.pending_feedbacks.lock();
                    pending.push(feedback);
                    trace!(
                        surface_id = surface.id().protocol_id(),
                        count = pending.len(),
                        total_requests = state.presentation_requests,
                        "wp_presentation: feedback queued on surface"
                    );
                } else {
                    // Not one of our surfaces — destroy with `discarded`.
                    feedback.discarded();
                }
            }
            wp_presentation::Request::Destroy => {}
            _ => {}
        }
    }
}

impl Dispatch<WpPresentationFeedback, ()> for WaylandState {
    fn request(
        _state: &mut Self,
        _client: &Client,
        _resource: &WpPresentationFeedback,
        _request: wp_presentation_feedback::Request,
        _data: &(),
        _dh: &DisplayHandle,
        _data_init: &mut DataInit<'_, Self>,
    ) {
        // wp_presentation_feedback has no requests.
    }
}

/// Presented flags: hardware-timed vsync'd presentation from DRM atomic
/// page flip events.
fn presented_flags() -> wp_presentation_feedback::Kind {
    wp_presentation_feedback::Kind::Vsync
        | wp_presentation_feedback::Kind::HwClock
        | wp_presentation_feedback::Kind::HwCompletion
}

/// For every `wl_output` the feedback's client has bound, send
/// `sync_output`. The protocol requires this to precede `presented`.
///
/// Returns the number of `sync_output` events emitted.
fn emit_sync_outputs(
    feedback: &WpPresentationFeedback,
    bound_outputs: &[wayland_server::protocol::wl_output::WlOutput],
) -> usize {
    let Some(client) = feedback.client() else {
        return 0;
    };
    let client_id = client.id();
    let mut count = 0;
    for output in bound_outputs {
        if output.client().map(|c| c.id()) == Some(client_id.clone()) {
            feedback.sync_output(output);
            count += 1;
        }
    }
    count
}

/// Send `presented` events to all currently in-flight feedbacks, then
/// clear the list. Called from the main loop on page flip completion.
///
/// # Arguments
/// * `presentation_ns` - `CLOCK_MONOTONIC` timestamp of the page flip.
/// * `refresh_ns` - predicted nanoseconds until the next vblank.
/// * `sequence` - monotonic MSC counter.
pub fn fire_presented(
    state: &mut WaylandState,
    presentation_ns: u64,
    refresh_ns: u32,
    sequence: u64,
) {
    if state.inflight_feedbacks.is_empty() {
        return;
    }
    let count = state.inflight_feedbacks.len();
    let tv_sec = presentation_ns / 1_000_000_000;
    let tv_nsec = (presentation_ns % 1_000_000_000) as u32;
    let tv_sec_hi = (tv_sec >> 32) as u32;
    let tv_sec_lo = (tv_sec & 0xFFFF_FFFF) as u32;
    let seq_hi = (sequence >> 32) as u32;
    let seq_lo = (sequence & 0xFFFF_FFFF) as u32;
    let flags = presented_flags();

    let feedbacks: Vec<WpPresentationFeedback> = state.inflight_feedbacks.drain(..).collect();
    let mut total_syncs = 0;
    for fb in feedbacks {
        total_syncs += emit_sync_outputs(&fb, &state.bound_outputs);
        fb.presented(tv_sec_hi, tv_sec_lo, tv_nsec, refresh_ns, seq_hi, seq_lo, flags);
    }
    trace!(
        count,
        total_syncs,
        refresh_ns,
        sequence,
        "wp_presentation: fired presented events"
    );
}

/// Send `discarded` to all currently staged feedbacks and clear them.
/// Called when a staged frame is replaced by a newer commit before being
/// forwarded.
pub fn discard_staged(state: &mut WaylandState) {
    if state.staged_feedbacks.is_empty() {
        return;
    }
    let count = state.staged_feedbacks.len();
    for fb in state.staged_feedbacks.drain(..) {
        fb.discarded();
    }
    trace!(count, "wp_presentation: discarded stale staged feedbacks");
}

/// Send `discarded` to all currently in-flight feedbacks and clear them.
/// Called when the in-flight buffer was never actually presented (e.g.
/// session went inactive, focus changed before flip).
pub fn discard_inflight(state: &mut WaylandState) {
    if state.inflight_feedbacks.is_empty() {
        return;
    }
    let count = state.inflight_feedbacks.len();
    for fb in state.inflight_feedbacks.drain(..) {
        fb.discarded();
    }
    trace!(count, "wp_presentation: discarded in-flight feedbacks");
}

/// Called when a `wl_surface` is destroyed. Discards any pending (not yet
/// committed) feedbacks so the client's `wp_presentation_feedback`
/// objects don't leak.
pub fn discard_surface_pending(surface_data: &SurfaceData) {
    let mut pending = surface_data.pending_feedbacks.lock();
    if pending.is_empty() {
        return;
    }
    let count = pending.len();
    for fb in pending.drain(..) {
        fb.discarded();
    }
    trace!(
        count,
        "wp_presentation: discarded pending feedbacks on surface destroy"
    );
}
