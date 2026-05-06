# Failed Attempts — Overlay Rendering

Tracking failed approaches to fixing the Steam overlay (Grid/Flutter) compositing
over game surfaces in gamecomp's DRM direct scanout path.

## Environment

- NVIDIA RTX 4080 SUPER, HDMI-A-1, 2560×1440 @ 144Hz VRR
- DRM direct scanout (no nested Wayland)
- XWayland 24.1.9 (Fedora 42) — does NOT bind `wp_presentation`
- Multi-XWayland: Server 0 = Grid/Flutter (display :2), Server 1 = game (display :3)
- Grid uses a single GTK window for both main UI and overlay
- Overlay format: ARGB8888 (`0x34325241`, `VK_FORMAT_B8G8R8A8_UNORM`)
- Overlay activation: playserve sets `STEAM_OVERLAY=1` X11 property → XWM detects → commit gate opens

## What Works

- Game compositing (single-layer): solid 144 fps
- Overlay buffer staging and dup: `staged_overlay_buffer` populated correctly
- Vulkan overlay blending shader: `overlay_blend.comp` composites when buffer exists
- Overlay commit gate: accepts commits from overlay surface alongside game
- Per-surface callback isolation: game and overlay callbacks no longer steal from each other
- Buffer release: `held_buffers.push()` prevents XWayland buffer pool exhaustion
- Syncobj cross-server collision fix: composite key `(surface_id, server_index)`

---

## Attempt 1: Alpha Reconstruction Shader

**Date:** Session before current  
**Problem:** Overlay buffer had RGB content with alpha=0 everywhere (premultiplied alpha with zero alpha = invisible).  
**Approach:** Modified `overlay_blend.comp` to reconstruct alpha as `max(R, G, B)` when `overlay.a == 0`.  
**Result:** Produced incorrect/garbled blending. Colors were wrong because the RGB values were already premultiplied against an alpha that GTK/Flutter calculated internally.  
**Why it failed:** The content is premultiplied by GTK/Skia before hitting the buffer. Reconstructing alpha from premultiplied RGB doesn't recover the original alpha. The real issue was that the buffer content was stale (committed before `hasOverlay:true`), not that alpha was genuinely zero.  
**Reverted:** Yes, back to standard premultiplied alpha blend.

## Attempt 2: NudgeOverlayWindows on Every VBlank

**Date:** Current session, early  
**Problem:** After overlay activation, Flutter commits 2 stale frames then stops. No more commits because `staged_overlay_buffer` being `Some` caused the keepalive to stop sending Expose events.  
**Approach:** Added `overlay_wake_remaining` timer (300 ticks ≈ 2s). On every vblank tick during wake period, sent `NudgeOverlayWindows` (resize window by -1px then restore) + `fire_frame_callbacks()` to all surfaces.  
**Result:** Three problems:
1. **Garbled rendering:** 140 resize/restore cycles per second. Flutter can't keep up — reallocates surfaces on each ConfigureNotify. Observed 46 frames at 2559×1439 mixed with 406 frames at 2560×1440.
2. **Game frame pacing destroyed:** `fire_frame_callbacks()` was global — fired ALL servers' callbacks, including the game's. Game's `callbacks_fired` dropped to 0.
3. **Alpha shader still wrong** (from Attempt 1, still active at this point).  
**Why it failed:** NudgeOverlayWindows is destructive at high frequency. And global callback firing crosses server boundaries.  
**Reverted:** Partially — removed NudgeOverlayWindows + fire_frame_callbacks from per-vblank, kept only RefreshWindows (Expose).

## Attempt 3: fire_frame_callbacks Every VBlank During Overlay Wake

**Date:** Current session, mid  
**Problem:** After removing NudgeOverlayWindows, the overlay still needed waking.  
**Approach:** Kept `fire_frame_callbacks()` on every vblank during `overlay_wake_remaining` period.  
**Result:** Game went permanently black. `callbacks_fired=0` for the game from `:37` onwards — even AFTER overlay dismissed at `:44`. Game NEVER recovered its frame callback rhythm.  
**Why it failed:** `fire_frame_callbacks()` was GLOBAL — drained all servers' pending callbacks into done(). When overlay commits called `fire_frame_callbacks()`, they consumed the game's callbacks. The game's XWayland never received `wl_callback.done` and stopped rendering.  
**Root cause identified:** Global `pending_frame_callbacks: Vec<WlCallback>` mixed all servers' callbacks.  
**Reverted:** Yes.

## Attempt 4: Per-Surface Callback Storage (Deferred-Only on VBlank)

**Date:** Current session  
**Problem:** Global callback lists caused cross-server callback theft.  
**Approach:** Replaced global `pending_frame_callbacks` / `deferred_frame_callbacks` with per-surface `HashMap<(u32, u32), Vec<WlCallback>>` keyed by `(surface_protocol_id, server_index)`. On vblank, only fired **deferred** callbacks (`fire_all_deferred_callbacks()`). Overlay callbacks fired inline on commit via `fire_surface_pending()`.  
**Result:** Game rendered fine (288 commits/2s, no callback theft). But overlay was invisible — only 2 stale commits, then stopped. Overlay's pending callbacks were never deferred (only App commits defer), so the vblank tick never fired them. Overlay callback cycle died after 2 inline fires.  
**Why it failed:** Overlay surfaces use `fire_surface_pending()` (immediate fire on commit), not `defer_surface_callbacks()`. After the 2 stale commits, no more overlay commits → no more inline callback fires. The vblank only checked deferred callbacks, missing the overlay's pending ones.  
**Fixed forward:** Changed vblank to fire BOTH pending and deferred via `fire_all_surface_callbacks()`.

## Attempt 5: Per-Surface Callbacks — Fire Both Pending + Deferred on VBlank

**Date:** Current session  
**Problem:** Attempt 4 only fired deferred on vblank, missing overlay's pending callbacks.  
**Approach:** Changed `fire_all_deferred_callbacks()` → `fire_all_surface_callbacks()` which drains BOTH `surface_callbacks` (pending) AND `surface_deferred_callbacks` (deferred) on every vblank tick.  
**Result:** Game keeps rendering (288 commits/2s). Overlay active (`ov_id=11`, `has_overlay_buffer=true`, `has_overlays=true` in render thread). **But overlay is invisible to user.** Only 2 overlay commits ever happen despite callbacks being fired. The stale overlay buffer IS being composited — the render thread shows `has_overlays=true` and `blit_composite: importing overlay` — but the content is the pre-overlay Grid main screen (not the overlay UI).  
**Why it failed:** The 2 overlay commits happen BEFORE `hasOverlay:true` reaches Flutter (178ms delay via playserve D-Bus round-trip). So the overlay buffer contains Grid's normal app content, not the overlay panel. After those 2 commits, XWayland's Present flip chain for server 0's surface goes dormant — XWayland has no pending PresentPixmap from Flutter (frame scheduler idle, nothing dirty to render). Even though per-surface `wl_callback.done` fires on vblank, XWayland ignores it because there's no PresentPixmap to flip. Expose events don't restart Flutter's Present/DRI3 scheduling.  
**Status:** Fixed forward — added periodic low-rate NudgeOverlayWindows (~200ms interval) during overlay_wake.

---

## Root Cause: XWayland Present Chain Deadlock

**Corrected understanding:** Grid has ONE X11 window ever (same window for main UI and overlay). Server 0 (Grid) has wl_surface#11 with syncobj#26. Server 1 (game) also has wl_surface#11 with syncobj#20 (protocol IDs are per-client; WAYLAND_DEBUG interleaves both servers).

XWayland's Present extension only commits to Wayland when the X11 client calls `PresentPixmap`. Flutter's frame scheduler goes idle when the widget tree is clean. The Present flip chain goes dormant. Even though `wl_callback.done` fires on vblank via per-surface callbacks, XWayland has no pending PresentPixmap to flip, so it ignores the callback.

**Chicken-and-egg**: Flutter needs `wl_callback.done` → to schedule render → calls `PresentPixmap` → XWayland registers `wl_surface.frame()` → callback fires on vblank. But XWayland only registers `wl_surface.frame()` when it has a PresentPixmap to flip, and Flutter only calls PresentPixmap after its frame scheduler wakes up.

**The ONLY way to restart this cycle** is `NudgeOverlayWindows` (resize by -1px then restore), which triggers `ConfigureNotify` → GTK reallocates surfaces → Flutter schedules a frame → PresentPixmap → Wayland commit.

## Key Timing Problem

1. Overlay activation: `STEAM_OVERLAY=1` set at T+0ms
2. NudgeOverlayWindows (resize): T+1ms → triggers 2 stale commits (pre-overlay content)
3. `hasOverlay:true` reaches Flutter: T+178ms (playserve D-Bus round-trip)
4. Flutter rebuilds widget tree: T+178ms+
5. But by T+178ms, Present chain is already dormant — no more commits
6. Periodic nudge at ~200ms intervals restarts the cycle

## Insights from Gamescope Research

1. **Gamescope does NOT force overlay re-render.** It composites the last committed buffer indefinitely. The stale period is invisible because Steam's overlay responds near-instantly (same process, no D-Bus delay).
2. **Gamescope uses wlroots** which gives each X11 window its own `wlr_surface`.
3. **The `nudged` mechanism** (`XMoveWindow` jiggle) applies ONLY to `focusWindow` (the game), not overlays.
4. **Expose events are ignored** in gamescope (`case Expose: break;`).
5. **Frame callbacks require both flags**: `unlockedForFrameCallback` AND `receivedDoneCommit`. When overlay stops committing, callbacks stop — by design.

## Attempt 6: Periodic Low-Rate NudgeOverlayWindows

**Date:** Current session  
**Problem:** XWayland Present chain goes dormant after 2 stale commits; single nudge at activation is consumed before `hasOverlay:true` arrives.  
**Approach:** Send NudgeOverlayWindows every ~200ms (every 30 vblank ticks at 144Hz) during the overlay_wake period (~2s). Also send RefreshWindows every vblank tick during overlay_wake. Previous attempt 2 nudged every vblank (~7ms) causing garbled rendering; ~200ms gives Flutter time to complete each resize+render cycle.  
**Result:** Screen blinked during nudge period, then went completely BLACK. Both game and overlay stopped rendering. commits=0, callbacks_fired=0.  
**Why it failed:** The nudge successfully triggered overlay commits, but each overlay commit included a DRM syncobj acquire fence (SyncobjHandle(3)) that took ~5 seconds to signal on the GPU. The render thread blocks on this acquire fence before it can present ANY frame — including game frames. While blocked:
1. No DRM page flips happen
2. No `wl_callback.done` fires (vblank_ticks=288 but vblank_cb=0)
3. Game's Present chain starves (no callback → no PresentPixmap → no commit)
4. Screen goes black (last presented frame was overlay-composited, then nothing)

Pattern repeats every ~5 seconds: overlay commits at point=9 → 5s fence wait → present one frame → idle → point=11 → 5s wait → etc.

**Root cause of blackout:** The render thread's pipeline is serialized — it processes ONE pending frame at a time and blocks on its acquire fence. If the overlay's acquire fence is slow/stuck, the game can't present. The overlay should be treated as a non-blocking "best effort" layer.  
**Status:** Fixed — see attempt 7.

## Attempt 7: Decouple Overlay From Game Pipeline

**Date:** Current session  
**Problem:** Attempt 6's periodic nudge triggered overlay commits, but the `CommitTarget::Overlay` handler duped the overlay buffer into `staged_buffer` (the game's primary pipeline). This put the overlay's acquire fence into `frame.app`, causing the render thread to block on a 5+ second overlay fence. Additionally, `wait_overlay_acquire()` polled for 10ms per frame, slowing the game pipeline.  
**Approach:** Two fixes matching gamescope's design:

1. **Guard overlay dup-to-staged:** Only stage overlay as `staged_buffer` when `focused_wl_surface_id == 0` (no game focused). When a game IS running, overlay goes through `staged_overlay_buffer` → `dup_or_clear_overlay()` in `forward_staged_frame()` → `CommittedFrame.overlay`. The game's pipeline is never contaminated with overlay acquire fences.

2. **Non-blocking overlay fences:** Replaced `wait_overlay_acquire()` (10ms blocking poll) with `check_overlay_acquire()` (log-only, never blocks). Matching gamescope: overlay acquire fences are NEVER blocking GPU dependencies. The GPU may read partially-rendered overlay data, but the overlay is alpha-blended on top and brief glitches are invisible.

**Result:** SUCCESS — overlay renders over game! But visible blink: periodic NudgeOverlayWindows (every ~200ms) resized overlay to 2559×1439 then restored to 2560×1440, causing 15 frames at wrong size mixed with 3714 frames at correct size. Each resize cycle produced one visible dimension jitter frame.
**Status:** Works but needs blink fix.

## Attempt 8: Remove Periodic Nudge (Expose-Only Wake)

**Date:** Current session  
**Problem:** Attempt 7's periodic NudgeOverlayWindows caused visible blink (2559×1439 ↔ 2560×1440 dimension jitter).
**Approach:** Removed periodic NudgeOverlayWindows entirely. Kept single initial nudge at activation + RefreshWindows (Expose) on every vblank during overlay_wake period. Theory: Flutter was committing continuously between nudges (372 commits in attempt 7), so nudges were redundant.
**Result:** Only 3 overlay commits total (vs 372 in attempt 7). Overlay never became visible. Render thread composited stale buffer 1880 times with unsignaled acquire fence (point=1). Overlay invisible to user despite `has_overlays=true` in DRM present.
**Why it failed:** Expose events do NOT restart Flutter's XWayland Present chain. XWayland's Present extension only responds to PresentPixmap (triggered by ConfigureNotify → surface reallocation → Flutter frame). Without periodic nudges, the Present chain dies after the initial 3 stale commits. The 372 commits in attempt 7 were caused BY the nudges, not independent of them.
**Key insight:** Flutter REQUIRES periodic ConfigureNotify-with-size-change to keep committing. Expose is insufficient.

## Attempt 9: Synthetic ConfigureNotify at Same Size (TickleOverlayWindows)

**Date:** Current session  
**Problem:** Attempt 8 showed nudges are essential, but attempt 7's nudges cause dimension jitter blink.
**Approach:** Added `TickleOverlayWindows` command that sends synthetic `ConfigureNotify` events at the **current** window size (no resize). Used periodic tickle every ~200ms during overlay_wake instead of NudgeOverlayWindows. Theory: ConfigureNotify alone (not the actual resize) triggers Flutter to schedule frames.
**Result:** Only 3 overlay commits total — same as attempt 8. 10 tickles sent over 2s, zero triggered any overlay commits. 498 "overlay acquire fence not ready" messages — stale buffer, invisible overlay.
**Why it failed:** Flutter/GTK's X11 embedder ignores ConfigureNotify when the size hasn't changed. The surface reallocation and frame scheduling is triggered by the SIZE CHANGE, not the ConfigureNotify event itself. A synthetic ConfigureNotify at the same dimensions is a no-op.
**Key insight:** The -1px resize in NudgeOverlayWindows is the active ingredient — not the ConfigureNotify event. Flutter must see an actual size change to reallocate surfaces and restart its frame scheduler.

## Attempt 10: Size Filter in dup_or_clear_overlay

**Date:** Current session
**Problem:** Attempt 7's periodic NudgeOverlayWindows worked but caused visible blink from 2559×1439 frames mixed with 2560×1440 frames.
**Approach:** Restored periodic NudgeOverlayWindows (from attempt 7) but added a size filter in `dup_or_clear_overlay()` — only pass overlay buffers whose dimensions match `output_width × output_height` (2560×1440). This should filter out the 2559×1439 frames from the resize phase while keeping the 2560×1440 frames.
**Result:** Overlay invisible to user despite 1728 composite frames (all 2560×1440), 23 overlay commits, DRM present with `has_overlays=true`. The compositor correctly composited the overlay, but the content was transparent.
**Why it failed:** The 2560×1440 buffer content is Grid's transparent idle surface. The actually-visible overlay content was in the 2559×1439 frames (from the resize phase). When Flutter receives ConfigureNotify with a new size, it reallocates surfaces and re-renders — this re-render produces the visible overlay content. When it gets the restore back to 2560×1440, it just restores the previous (transparent) surface at the original size. The frames we were FILTERING OUT were the ones with actual visible content.
**Key insight:** The NudgeOverlayWindows resize is not just waking Flutter — it's the mechanism that forces Flutter to REDRAW. The 2559×1439 frames are the only ones with actual overlay content. The 2560×1440 frames after restore have transparent content from before the overlay was activated.

## Attempt 11: Alternate Undersized + No Size Filter

**Date:** Current session
**Problem:** Attempt 10 showed that restore frames (2560×1440) are stale/transparent and only undersized frames have visible content.
**Approach:** Modified NudgeOverlayWindows to alternate between `(output-1, output-1)` and `(output-2, output-2)` instead of shrink+restore. Removed size filter. Both sizes are undersize so Flutter always re-renders fresh content. Never restores to original transparent size.
**Result:** Partial success — overlay worked on FIRST open. But when user closed and reopened it, the overlay was invisible for ~5 seconds before appearing.
**Analysis:**
- First open: 175 overlay commits (39.6s-43.1s), 498 composited frames → visible ✓
- Second open: ZERO commits for 5 seconds (43.8s-48.6s), then 1 commit, then works again
- `has_overlay_buffer=false` throughout the 5-second gap — buffer was cleared on deactivation
- NudgeOverlayWindows sent 11 nudges during second open — Flutter did NOT respond
- All nudges showed `w=2560 target_w=2559` (alternation never triggered: something restores the window to 2560 between nudges)
- The commit gate rejected overlay surface commits while overlay was `val=0` (logged at `trace` level, not visible in debug logs)
**Root cause:** When overlay deactivates → reactivates:
1. `dup_or_clear_overlay()` clears the cached buffer when `overlay_wl_surface_id == 0`
2. Commit gate rejects overlay surface commits during the brief deactivation gap
3. XWayland's Present chain dies — PresentCompleteNotify never fires for rejected commits
4. Nudge resizes don't restart a dead Present chain (Flutter/GTK doesn't re-enter its render loop from ConfigureNotify alone)
5. After ~5 seconds, some internal Flutter/GTK timer or event eventually restarts rendering
**Key insight:** Gamescope NEVER rejects overlay commits and NEVER clears overlay buffers. All windows can commit anytime. The commit gate is gamecomp's design pattern that gamescope doesn't have — it breaks the XWayland Present chain.
**Fix (Attempt 12):** Two-pronged approach matching gamescope:
1. Add "dormant overlay" tracking — accept commits from last-known overlay surface even when deactivated (release buffer + fire callbacks, but don't stage). Keeps Present chain alive.
2. Don't clear cached overlay buffer on deactivation. Return None (don't composite) but keep the buffer for instant reuse when overlay reactivates.

## Attempt 12: PassThrough Commits + Buffer Caching

**Date:** Next session
**Problem:** Attempt 11's commit gate rejected overlay surface commits when overlay was deactivated, killing XWayland's Present chain. Gamescope never rejects commits.
**Approach:** Added `CommitTarget::PassThrough` for all server 0 surfaces. When a surface doesn't match focused/overlay/ext_overlay but comes from server 0, accept the commit: consume syncobj, release buffer, fire callbacks. This keeps XWayland's Present chain alive at all times without staging the buffer. Also preserved overlay buffer cache across deactivation (return None but don't clear).
**Result:** PassThrough worked perfectly for keeping the Present chain alive. But overlay only got 1 commit in 10 seconds after activation. Flutter stops committing when widget tree is idle — PassThrough fires `wl_callback.done` but XWayland only acts on it if there's a pending PresentPixmap. Flutter has nothing to present.
**Why it failed:** PassThrough solves the commit rejection problem but doesn't solve the fundamental issue: Flutter's frame scheduler goes idle when nothing animates. Gamescope doesn't face this because Steam's overlay never stops rendering. Frame callbacks alone can't restart Flutter's render loop.
**Status:** PassThrough kept as foundation — correct behavior matching gamescope's accept-all approach.

## Attempt 13: Continuous Nudging (Alternating Undersize)

**Date:** Same session as 12
**Problem:** PassThrough alone got only 1 overlay commit. Flutter's frame scheduler is idle.
**Approach:** Restored continuous NudgeOverlayWindows for the ENTIRE overlay lifetime (not just a wake period). Alternated between `(W-1)×(H-1)` and `(W-2)×(H-2)` every 10 vblank ticks (~70ms at 144Hz). No size filter — both undersized frames composited.
**Result:**
- First trigger: overlay appeared but **continuously resized visibly**. User could see gaps on right and top side growing/shrinking rapidly as the overlay alternated between 2559×1439 and 2558×1438. Never rendered at full 2560×1440.
- Second trigger: overlay didn't show up at all (user perception). Logs showed 464 `has_overlays=true` frames — overlay WAS composited but content was likely stale/transparent from the cached buffer between cycles.
**Why it failed:** Two separate problems:
1. **Visible resize artifacts:** The 1-2px undersize IS visible. Flutter renders its content at the undersized dimensions — the drawn content doesn't fill the screen edge-to-edge. CLAMP_TO_EDGE sampling repeats edge pixels but the Flutter-drawn content itself has a gap at the border. This is not a shader problem — it's Flutter drawing a 2558px-wide surface for a 2560px-wide screen.
2. **Cycle 2 failure:** Overlay buffer cached from cycle 1's close animation (transparent/fading content). New Flutter commits in cycle 2 used the undersized dimensions from nudging, but the cached buffer displayed while waiting had stale transparent content.
**Key insight:** Gamescope NEVER nudges overlays. The nudge approach is fundamentally wrong — it forces the overlay to render at non-native resolution, causing visible artifacts that gamescope never has.

## Attempt 14: Expose-Only Keepalive (No Nudging)

**Date:** Same session as 13
**Problem:** All nudging approaches cause visible resize artifacts. Gamescope doesn't nudge.
**Approach:** Removed ALL nudging code entirely (NudgeOverlayWindow command, variables, vblank handler, activation/deactivation resize). Replaced with `overlay_keepalive` bool — while overlay is active, sends `RefreshWindows` (X11 Expose events) to server 0 on every vblank tick. Overlay renders at full native resolution (2560×1440) at all times.
**Result:** Overlay didn't show at all. Same as Attempt 8 — Expose events do not restart Flutter's Present chain.
**Why it failed:** This is a re-confirmation of Attempt 8's finding. X11 Expose events cannot restart Flutter's frame scheduling. Flutter/GTK's X11 embedder does not use Expose events to trigger PresentPixmap. The only mechanism that forces Flutter to re-render is an actual window size change (ConfigureNotify with new dimensions), which triggers surface reallocation in GTK's GDK X11 backend.
**Key insight:** We are stuck between two contradictory requirements:
1. Nudging (ConfigureNotify with size change) is the ONLY way to restart Flutter's dormant frame scheduler
2. Nudging causes visible resize artifacts because Flutter renders at the nudged dimensions

## Attempt 15: Defer Frame Callbacks to VBlank (Match Gamescope)

**Date:** Same session as 14
**Problem:** Gamecomp fired `wl_callback.done` IMMEDIATELY during `wl_surface.commit()` via `fire_surface_pending()`. Gamescope gates callbacks on BOTH `receivedDoneCommit` AND `unlockedForFrameCallback` (next vblank). The immediate fire collapses the commit→callback round-trip to zero, potentially causing XWayland's Present chain to idle.
**Approach:** Changed ALL non-App paths from `fire_surface_pending()` (instant) to `defer_surface_callbacks()` (deferred to next vblank via `fire_all_surface_callbacks()`):
- **PassThrough**: consume syncobj, release buffer immediately, defer callback to vblank
- **Overlay/ExternalOverlay**: signal old release point, store buffer, defer callback to vblank
- **Rejected**: release buffer, defer callback to vblank

This correctly matches gamescope's vblank-gated `flush_frame_done` behavior.
**Result:** Overlay still invisible. Log analysis (`new.log`) showed:
- Only **2 overlay commits** in 15+ seconds (at 52.004 and 62.320, 10.3s apart)
- 2222 frames rendered with `has_overlays=true` but stale/transparent content
- Server 0 STOPPED COMMITTING entirely at wayland timestamp ~1957946 (when game took focus at ~22:40:46)
- After game took focus: grep confirms only 2 server 0 commits ever (syncobj points 5 and 7 on timeline#27)
**Why it failed:** The deferred callback fix is correct in principle but doesn't address the root cause. Flutter/server 0 stops committing because:
1. GDK's frame clock goes idle when widget tree is clean
2. Without a commit, there's no `wl_surface.frame()` request
3. Without a frame request, there's no callback to defer
4. Without a callback, there's no `PresentCompleteNotify`
5. Without `PresentCompleteNotify`, Flutter doesn't schedule a new frame
The cycle is dead regardless of callback timing because the entry condition (a commit from Flutter) never occurs.

**Additional bugs discovered during investigation:**
1. **PassThrough double-release**: Buffer is `release()`d immediately AND pushed to `held_buffers` (which releases again in `release_stale_buffers()`). Protocol violation.
2. **Idle keepalive never fires for server 0**: `last_commit_ns` tracks the GAME's commits (via `staged_at_ns`). While game runs at 60+ fps, `last_commit_ns` is always fresh → idle keepalive condition never triggers → server 0 gets NO external stimuli after expose grace period expires.
3. **Shared `held_buffers` pool**: Overlay buffer release is coupled to game commits (via `release_stale_buffers()` in `forward_staged_frame()`). In gamescope, each window has its own `commit_queue` with independent 1-for-1 buffer exchange.

---

## Attempt 16: Overlay Keepalive + PassThrough Double-Release Fix

**Date:** Same session as 15
**Problem:** Attempt 15 proved deferred callbacks alone don't help. Three bugs identified: (1) PassThrough double-release protocol violation, (2) idle keepalive blind to server 0 because `last_commit_ns` tracks game commits, (3) shared `held_buffers` pool.
**Approach:** Two fixes:
1. **Fixed PassThrough double-release**: Removed immediate `buffer.release()` in PassThrough path. Buffer now only goes into `held_buffers` (released later by `release_stale_buffers()`). Matches gamescope's model of releasing only when a new commit replaces the old buffer.
2. **Added overlay keepalive**: While any overlay is active (`overlay_wl_surface_id != 0` or `external_overlay_wl_surface_id != 0`), sends Expose events (`RefreshWindows`) to server 0 on every vblank, independently of `last_commit_ns`. This ensures server 0 gets continuous stimuli regardless of game commit rate.

**Result:** Overlay still invisible. Log analysis (`new.log`) showed:
- Only **3 overlay commits** in ~21 seconds (at 17:15:16.5, 17:15:26.7, 17:15:36.9)
- Gaps: 10.2s, 10.2s — identical to attempt 15 (10.3s gaps)
- 2960 frames with `has_overlays=true` but stale content
- Expose events sent every vblank confirmed by code path, but **Flutter doesn't react to them**

**Detailed protocol trace analysis:**
```
17:15:06.195  Last pre-overlay commit (point 3983), PassThrough path
              callback#22.done fires at 98201 (6ms after commit) ✓
              buffer#25.release fires at 98215 (20ms after commit) ✓
              → Both signals delivered promptly. Flutter DOESN'T commit again.

17:15:12.840  Overlay activates. fire_server_callbacks(0) fires callback#39
              from a pending PassThrough commit. Expose events start.
              
17:15:16.506  First overlay commit (point 3985) — 3.66s after activation
              callback fires within 1ms (108507) ✓
              → Flutter received PresentCompleteNotify. Still no new commit for 10.2s.

17:15:26.730  Second overlay commit (point 3987) — 10.22s gap
17:15:36.963  Third overlay commit (point 3989) — 10.23s gap
```

**Why it failed:** The ~10.2 second commit interval is a **Flutter-internal timer**, not a compositor issue. Verified by:
1. Frame callbacks fire within 1-6ms of commit — no delay ✓
2. Buffer releases fire within 20ms — no delay ✓
3. Expose events sent every vblank — confirmed ✓
4. XWayland's present fallback timer is TIMER_LEN_FLIP=1000ms, not 10s
5. After frame callback fires, XWayland's present state machine frees its timer if no new PresentPixmap is issued
6. Flutter's GDK frame clock goes idle when widget tree is clean; Expose events are insufficient to wake it

**XWayland present timer analysis (from source, hw/xwayland/xwayland-present.c):**
- `TIMER_LEN_FLIP = 1000ms`, `TIMER_LEN_COPY = 17ms`
- Timer only active when `has_pending_events()` = true (flip_pending->sync_flip OR wait_list OR blocked_queue)
- After `msc_bump` processes all events and no new PresentPixmap: timer freed → chain dead
- The 10.2s commit interval comes from Flutter itself (likely a Dart periodic timer or GC-triggered render), not from XWayland or the compositor

**Key insight:** This proves the problem is NOT in gamecomp's callback/release timing. Both signals arrive promptly. The issue is that **Expose events cannot wake Flutter's frame scheduler**. GDK on XWayland either doesn't process Expose events as damage, or processes them but Flutter's Skia backend doesn't trigger eglSwapBuffers for Expose-only repaints.

---

## Attempt 17: Signal syncobj release point for PassThrough and rejected commits

**Approach:** The protocol trace from attempt 16 showed server 0 commits exactly 2 times after focus change, then stops for 5+ seconds. Analysis of the syncobj timeline IDs revealed the client switched timelines between the two commits (timeline#24 → #27), indicating the first timeline's release point was never signaled.

**Root cause identified:** The PassThrough path consumed the syncobj pending state via `take_pending_sync()` but discarded the result (`let _ = ...`). The release point was never signaled. With explicit sync (DRM syncobj), the client's GPU driver waits for the release fence before allowing buffer reuse. By never signaling it, XWayland's buffer pool was exhausted after 2 commits — the client had no buffers left to render into, killing the Present chain.

**Evidence chain:**
1. Surface#19 commits at 60fps before focus change (92275, 92291, ..., 92490)
2. Focus change → `fire_all_callbacks()` → callback.done → Flutter commits at 92671 (with syncobj timeline#24, release point 6308)
3. Callback fires at 92673 → Flutter commits at 92691 (SWITCHES to timeline#27, release point 3982 — timeline#24's release was never signaled!)
4. Callback fires at 92694 → Flutter NEVER commits again (both timelines' release points are stuck)
5. 5.5 second gap, then only ~10.2s-interval commits

**Fix:**
1. **PassThrough path**: After `take_pending_sync()`, signal the release point immediately via `device.timeline_signal(rp.handle(), rp.point)`. The buffer is never composited, so the compositor is "done" with it right away.
2. **Rejected commit path**: Same fix — signal release point before returning. Rejected commits from non-server-0 surfaces with explicit sync would have the same bug.

**Files changed:** `src/wayland/protocols/compositor.rs`

---

## Summary: Root Causes (Updated After Attempt 17)

### Primary Root Cause: Syncobj Release Point Never Signaled (FIXED)

The PassThrough commit path consumed the DRM syncobj pending state but discarded the release point without signaling it. With explicit sync, the client GPU driver requires the compositor to signal the release fence before allowing buffer reuse. By never signaling, XWayland's buffer pool was exhausted after 2 commits, permanently killing the Present → callback → commit cycle.

This explains why gamescope doesn't have the issue: gamescope composites ALL surfaces every frame, and its render pipeline signals release points for every buffer it blits. In gamecomp, PassThrough buffers were never composited and their release points were silently dropped.

### Secondary: Shared Buffer Pool (Still Unfixed)

All surfaces share one `held_buffers` Vec. `release_stale_buffers()` fires on game commits, coupling overlay buffer release to game timing. Not the root cause of the dead cycle (that was the syncobj), but should be fixed for correctness.

### Approaches That Haven't Been Tried (If Syncobj Fix Insufficient)

**Most Promising (External Signal Approaches):**

1. **X11 `_NET_WM_STATE` toggle** — Toggle a harmless atom (e.g., `_NET_WM_STATE_FOCUSED`) on the overlay window. GDK watches `_NET_WM_STATE` via `PropertyNotify` and triggers style/layout recalculation, which may wake the frame clock.
2. **Synthetic mouse motion** — Send a `MotionNotify` event to the overlay window via XCB. GDK's event handler should process it and schedule a paint (event processing thaws the frame clock even for no-op hover).
3. **ConfigureNotify with same size** — Send `ConfigureNotify` with identical dimensions but a different position, or stack order change. GDK might process the configure without actually resizing.
4. **FocusIn event to overlay window** — When overlay activates, send explicit `FocusIn` to the overlay X11 window. GDK's focus handling schedules a redraw to update focus indicators.
5. **X11 ClientMessage (`_NET_ACTIVE_WINDOW`)** — Send an EWMH activation message to the overlay window. GDK processes `_NET_ACTIVE_WINDOW` ClientMessage as a focus/activation event.
6. **wl_surface configure with same size** — Send an `xdg_surface.configure` event at the Wayland level (not X11). XWayland would translate this into ConfigureNotify for the X11 window.

**Investigation Approaches:**

7. **Capture gamescope's protocol trace** — Run Grid under gamescope with `WAYLAND_DEBUG=1` and compare what protocol events server 0 receives during overlay activation vs what gamecomp sends. This would definitively show what signal gamescope provides that wakes Flutter.
8. **Capture X11 events in gamescope** — Run `xev` on the overlay window in gamescope to see what X11 events it receives during overlay activation.

**Client-Side Approaches:**

9. **Flutter-side fix** — Add `SchedulerBinding.instance.scheduleFrame()` on a periodic timer when overlay is active. Most reliable, but requires Grid code changes.
10. **GDK environment variable** — Check if `GDK_FRAME_CLOCK_RATE` or similar env vars can force GDK to always tick.

**Lower Priority:**

11. **Per-surface buffer tracking** — Per-(surface_id, server_index) buffer queues with 1-for-1 exchange. Correct but not root cause.
12. **Synthetic X11 Expose with explicit damage region** — Targeted Expose to just the overlay window. Likely fails for same reason as attempts 8, 14, 16.
13. **GDK-specific atoms** — Set `_GTK_FRAME_EXTENTS` or similar GTK-specific properties.
14. **MapNotify/UnmapNotify cycle** — Brief unmap/remap to reset GDK state. Risky — may cause visual flash.
