# Frame Pipeline Comparison: Gamescope vs Gamecomp

## Executive Summary

The critical difference: **With VRR active, gamescope treats EVERY loop iteration as a VBlank** (`if (bVRR) vblank = true`). This means callbacks fire on the SAME iteration that observes the commit — zero delay. Gamecomp gates callbacks on a timerfd/page-flip event that may arrive on a DIFFERENT iteration, adding up to 6.9ms latency (one timer period at 144Hz).

---

## Gamescope Pipeline (DRM + VRR)

### Architecture: Single-Threaded Compositor

```
┌─────────────── steamcompmgr main loop ───────────────┐
│                                                       │
│  1. Dispatch XWayland events (X11 property changes)   │
│  2. PollEvents() — poll timerfd + nudge pipe          │
│  3. ProcessVBlank() — check if timerfd fired          │
│  4. if (VRR) → vblank = true (ALWAYS!)               │
│  5. handle_done_commits() — process commit queue      │
│     → sets hasRepaint=true, receivedDoneCommit=true   │
│  6. latch_frame_done() — sets unlockedForFrameCallback│
│  7. flush_frame_done() — if BOTH flags → fire callback│
│  8. if (VRR) bShouldPaint = hasRepaint                │
│     BUT gate: PresentsInFlight != 0 → bShouldPaint=0 │
│  9. paint_all() → Vulkan composite → DRM atomic flip  │
│ 10. page_flip_handler → MarkVBlank() + re-arm timer   │
│                                                       │
└───────────────────────────────────────────────────────┘
```

### Key Behaviors with VRR:

1. **`vblank = true` on EVERY iteration** — No waiting for timer or page flip to "unlock"
2. **Callback fires on the SAME iteration as commit** — Since `vblank=true` always:
   - `latch_frame_done()` → sets `unlockedForFrameCallback = true`
   - `handle_done_commits()` → sets `receivedDoneCommit = true`  
   - `flush_frame_done()` → both flags true → **fires callback immediately**
   - Both flags cleared — ensures only ONE callback per commit
3. **Paint gated ONLY by `PresentsInFlight`** — If a page flip is still pending, `bShouldPaint = false`. Otherwise paint immediately.
4. **Buffer release** — wlroots releases the buffer when a new commit replaces the old one (implicit in `wlr_buffer_lock/unlock` lifecycle)
5. **VBlank timer role with VRR** — Only used for:
   - Overlay repaint timing (non-base plane)
   - FPS limiting (if target FPS set)
   - Pre-emptive re-arm to avoid stalls

### VRR Flow (happy path, no FPS limit):

```
Iteration N:
  PollEvents() → nudge (from wayland_commit in wlserver)
  vblank = true (VRR)
  handle_done_commits() → commit dequeued → hasRepaint=true, receivedDoneCommit=true
  latch_frame_done() → unlockedForFrameCallback=true (vblank=true, no FPS limit)
  flush_frame_done() → BOTH flags set → FIRE CALLBACK → both flags cleared
  bShouldPaint = hasRepaint = true
  PresentsInFlight == 0 → paint_all()
  DRM atomic commit (NONBLOCK + PAGE_FLIP_EVENT)
  PresentsInFlight = 1

Iteration N+1..M (client rendering):
  PollEvents() → no events (client busy rendering)
  vblank = true (VRR)
  handle_done_commits() → nothing
  latch_frame_done() → sets unlockedForFrameCallback=true (but receivedDoneCommit=false → no fire)
  flush_frame_done() → receivedDoneCommit=false → SKIP
  hasRepaint = false → bShouldPaint = false
  (loop spins idle, occasional timer poll)

Iteration M (page flip completes):
  page_flip_handler() → PresentsInFlight = 0, MarkVBlank(), ArmNextVBlank()
  (No callback fire here — waits for next commit)

Iteration M+K (new commit arrives):
  Same as Iteration N — callback fires IMMEDIATELY
```

### Timing Discipline:

- Callback fires: **0μs after commit is observed** (same loop iteration)
- Paint happens: **0μs after commit** (unless PresentsInFlight > 0)
- Page flip latency: kernel atomic commit → display scanout (~100μs with VRR)
- **Total per-frame pipeline overhead: <1ms**
- **Effective FPS = 1/client_render_time** (limited only by client speed + one page flip pending)

---

## Gamecomp Pipeline (DRM + VRR)

### Architecture: Main Thread + Render Thread

```
┌──────────── Main thread (calloop) ────────────┐    ┌──── Render thread ────┐
│                                                │    │                       │
│  1. poll_or_sleep() — poll wayland_fd + timer  │    │ recv committed_frame  │
│  2. Wayland dispatch → commits arrive          │    │ Vulkan blit (<1ms)    │
│     → staged_buffer set, defer_frame_callbacks │    │ DRM atomic commit     │
│  3. Drain vblank_rx (page flip completions)    │    │ (NONBLOCK+FLIP_EVENT) │
│     → set unlocked_for_callback = true         │    │ page_flip_handler     │
│  4. VBlank timer tick                          │    │ → send vblank_tx      │
│     → set unlocked_for_callback = true         │    │                       │
│  5. Two-phase flush:                           │    └───────────────────────┘
│     if unlocked && deferred_callbacks → fire   │
│  6. forward_staged_frame()                     │
│     → send to render thread                    │
│     → release_all_buffers()                    │
│     → (NO callback fire here)                  │
│                                                │
└────────────────────────────────────────────────┘
```

### Key Behaviors (CURRENT — after two-phase change):

1. **`unlocked_for_callback`** set by page flip completion OR VBlank timer
2. **Callback fires when BOTH conditions met**: unlocked=true AND deferred callbacks exist
3. **Buffer release happens on forward** (immediately, before page flip completes)
4. **Paint NOT gated by PresentsInFlight** — always forwards immediately
5. **VBlank timer fires at 144Hz** — provides the unlock signal

### Current Flow (VRR):

```
Iteration N (timer tick while client rendering):
  poll_or_sleep() wakes on timerfd (6.9ms tick)
  Wayland dispatch → no new commits
  vblank_rx → empty (page flip already completed earlier)
  VBlank timer → ticks > 0 → unlocked_for_callback = true
  Two-phase flush: unlocked=true BUT deferred_callbacks EMPTY → no fire
  forward_staged_frame() → no staged buffer → skip

Iteration N+K (commit arrives):
  poll_or_sleep() wakes on wayland_fd (commit event)
  Wayland dispatch → commit → staged_buffer set, defer_frame_callbacks()
  vblank_rx → empty
  VBlank timer → may or may NOT have ticked!
    If NOT ticked: unlocked_for_callback still true from previous tick → FIRE
    If already fired previously and cleared: WAIT until next tick/pageflip!
  Two-phase flush:
    CASE A: unlocked still true → fire immediately (0ms delay)
    CASE B: unlocked was cleared → MUST WAIT for next timer tick (0-6.9ms!)
  forward_staged_frame() → send to render thread
```

### THE PROBLEM:

**Case B happens frequently.** Here's why:

1. Timer ticks at 144Hz (every 6.9ms). When it ticks AND deferred callbacks exist → fire + clear unlocked.
2. Client renders for ~20ms (Flutter debug). During this time, ~2-3 timer ticks happen. Each tick sets unlocked=true, but since no deferred callbacks exist, it's a no-op.
3. Eventually the commit arrives. The LAST timer tick that set unlocked=true may have been up to 6.9ms ago. If that tick already fired a callback (from a previous commit's deferred callbacks), unlocked was cleared. The NEXT timer tick hasn't arrived yet → **client waits 0-6.9ms for the unlock**.

This creates **jitter**: sometimes the commit arrives right after a tick (0ms wait), sometimes between ticks (up to 6.9ms wait). At 144Hz this averages ~3.5ms added latency and produces the irregular frame intervals seen in logs.

---

## Critical Differences

| Aspect | Gamescope (VRR) | Gamecomp (VRR) |
|--------|----------------|----------------|
| Callback trigger | Same iteration as commit (vblank=true always) | Requires prior unlock from timer/pageflip |
| Latency to fire | 0μs | 0-6.9ms (timer period jitter) |
| Paint trigger | hasRepaint + !PresentsInFlight | Always (no backpressure gate) |
| Buffer release | On next commit (wlroots implicit) | On forward (before page flip) |
| Threading | Single-threaded | Main + Render thread |
| Present backpressure | PresentsInFlight gate on paint | None (always presents) |
| VBlank timer role (VRR) | Timer NOT needed for callbacks | Timer IS the callback trigger |

---

## The Fix

With VRR active, we should match gamescope's behavior:
- **Fire callback IMMEDIATELY when a commit is observed** (in `forward_staged_frame`)  
- **Gate PAINTING on PresentsInFlight** (don't forward if a flip is pending)

This is the exact opposite of our current approach:
- Current: We gate CALLBACKS on VBlank timing, but always paint immediately
- Gamescope: Fires callbacks immediately (VRR=always vblank), but gates PAINTING on PresentsInFlight

The gamescope approach ensures:
1. Client gets callback the instant its commit is processed → starts next frame immediately
2. Display never queues two flips simultaneously → clean VRR timing
3. No timer-induced jitter on the callback path

---

## Recommended Changes

1. **With VRR (or always for now):** Fire deferred callbacks **in `forward_staged_frame()`** immediately after sending to render thread. Remove timer-based callback entirely for the VRR path.

2. **Add PresentsInFlight gate on forward:** Don't forward the staged frame if `presents_in_flight > 0`. Hold it until the page flip completes. This prevents queuing two flips and matches gamescope's `bShouldPaint = false` when `PresentsInFlight != 0`.

3. **Buffer release timing:** Stay as-is (release on forward). Since we blit to GBM, client buffer is free after blit completes (<1ms). Gamescope holds the wlr_buffer until the next commit replaces it, but our blit architecture doesn't require that.

The combination of (1) and (2) means:
- Flip completes → `presents_in_flight = 0`  
- Next commit arrives → forward immediately + fire callback
- Client starts next frame → render → commit
- If flip still pending when commit arrives → hold frame → no callback fire yet
- Flip completes → forward held frame + fire callback → client resumes

This provides **gamescope-identical pacing**: one frame in flight, callback immediately on commit acceptance, backpressure from display, not from arbitrary timer.
