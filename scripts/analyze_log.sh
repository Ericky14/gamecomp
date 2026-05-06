#!/usr/bin/env bash
# gamecomp log analyzer — extracts key state transitions and freeze diagnostics
# Usage: ./scripts/analyze_log.sh <logfile>
set -uo pipefail

LOG="${1:?Usage: $0 <logfile>}"

# Strip ANSI escape codes for clean parsing
CLEAN=$(mktemp)
trap 'rm -f "$CLEAN"' EXIT
sed 's/\x1b\[[0-9;]*m//g' "$LOG" > "$CLEAN"

TOTAL_LINES=$(wc -l < "$CLEAN")
echo "=== GAMECOMP LOG ANALYSIS ==="
echo "File: $LOG ($TOTAL_LINES lines)"
echo

# ── Time range ───────────────────────────────────────────────
FIRST_TS=$(grep -oP '^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+Z' "$CLEAN" | head -1)
LAST_TS=$(grep -oP '^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+Z' "$CLEAN" | tail -1)
echo "Time range: $FIRST_TS → $LAST_TS"
echo

# ── Configuration ────────────────────────────────────────────
echo "=== CONFIG ==="
grep -m1 'configuration config=' "$CLEAN" | sed 's/.*config=//' || true
echo

# ── VT switches ──────────────────────────────────────────────
echo "=== VT SWITCHES ==="
grep -n 'VT switch\|session paused\|session resumed\|firing frame callbacks.*VT' "$CLEAN" || echo "(none)"
echo

# ── Focus changes ────────────────────────────────────────────
echo "=== FOCUS CHANGES ==="
grep -n 'logical focus changed\|focus arbiter.*winner changed' "$CLEAN" || echo "(none)"
echo

# ── Overlay events ───────────────────────────────────────────
echo "=== OVERLAY EVENTS ==="
grep -n 'STEAM_OVERLAY\|overlay activated\|overlay state' "$CLEAN" | head -20 || echo "(none)"
echo

# ── Warnings & errors ───────────────────────────────────────
echo "=== WARNINGS ==="
grep -n ' WARN \| ERROR ' "$CLEAN" || echo "(none)"
echo

# ── Acquire fence timeouts ───────────────────────────────────
echo "=== ACQUIRE FENCE TIMEOUTS ==="
grep -cn 'acquire fence timed out' "$CLEAN" | xargs -I{} echo "Count: {}"
grep -n 'acquire fence timed out' "$CLEAN" || true
echo

# ── Frame stats timeline (commits/callbacks/idle) ────────────
echo "=== FRAME STATS TIMELINE ==="
echo "timestamp                      | commits | cb_fired | fwd | inflight | held | idle_ms | expose_grace | pending_cb"
echo "-------------------------------|---------|----------|-----|----------|------|---------|--------------|----------"
grep 'frame_stats: periodic summary' "$CLEAN" | while IFS= read -r line; do
    ts=$(echo "$line" | grep -oP '^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}')
    commits=$(echo "$line" | grep -oP 'commits=\K\d+')
    cb=$(echo "$line" | grep -oP 'callbacks_fired=\K\d+')
    fwd=$(echo "$line" | grep -oP 'frames_forwarded=\K\d+')
    inflight=$(echo "$line" | grep -oP 'inflight=\K\d+')
    held=$(echo "$line" | grep -oP ' held=\K\d+')
    idle=$(echo "$line" | grep -oP 'idle_ms=\K\d+')
    grace=$(echo "$line" | grep -oP 'expose_grace=\K\d+')
    pending=$(echo "$line" | grep -oP 'pending_cb=\K\d+')
    printf "%-31s | %7s | %8s | %3s | %8s | %4s | %7s | %12s | %s\n" \
        "$ts" "$commits" "$cb" "$fwd" "$inflight" "$held" "$idle" "$grace" "$pending"
done
echo

# ── Frozen periods (commits=0 for consecutive intervals) ────
echo "=== FROZEN PERIODS (commits=0, consecutive) ==="
prev_zero=false
freeze_start=""
grep 'frame_stats: periodic summary' "$CLEAN" | while IFS= read -r line; do
    ts=$(echo "$line" | grep -oP '^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}')
    commits=$(echo "$line" | grep -oP 'commits=\K\d+')
    idle=$(echo "$line" | grep -oP 'idle_ms=\K\d+')
    held=$(echo "$line" | grep -oP ' held=\K\d+')
    pending=$(echo "$line" | grep -oP 'pending_cb=\K\d+')
    if [ "$commits" = "0" ]; then
        if [ "$prev_zero" = false ]; then
            freeze_start="$ts"
            prev_zero=true
        fi
        echo "  FROZEN at $ts  idle=${idle}ms held=$held pending_cb=$pending"
    else
        if [ "$prev_zero" = true ]; then
            echo "  --- THAWED at $ts (was frozen since $freeze_start) commits=$commits"
            prev_zero=false
        fi
    fi
done
echo

# ── Buffer lifecycle: held buffers accumulating ──────────────
echo "=== HELD BUFFER TREND ==="
grep 'frame_stats: periodic summary' "$CLEAN" | grep -oP ' held=\K\d+' | sort -n | uniq -c | sort -rn | head -5 | \
    while read count val; do echo "  held=$val appeared $count times"; done
echo

# ── Callback flow: are deferred callbacks being fired? ───────
echo "=== CALLBACK FLOW ==="
echo -n "Total vblank callback fires: "
grep 'frame_stats: periodic summary' "$CLEAN" | grep -oP 'vblank_cb=\K\d+' | awk '{s+=$1}END{print s}'
echo -n "Total commits: "
grep 'frame_stats: periodic summary' "$CLEAN" | grep -oP 'commits=\K\d+' | awk '{s+=$1}END{print s}'
echo -n "Total frames forwarded: "
grep 'frame_stats: periodic summary' "$CLEAN" | grep -oP 'frames_forwarded=\K\d+' | awk '{s+=$1}END{print s}'
echo

# ── Syncobj fence registrations ──────────────────────────────
echo "=== SYNCOBJ FENCE REGISTRATIONS ==="
grep -cn 'registered acquire eventfd' "$CLEAN" | xargs -I{} echo "Count: {}"
grep -n 'registered acquire eventfd' "$CLEAN" | head -10
echo

# ── Key event sequence around freezes ────────────────────────
echo "=== EVENTS AROUND FIRST FREEZE ==="
# Find the first frame_stats line with commits=0 that follows a commits>0
prev_commits="-1"
grep -n 'frame_stats: periodic summary' "$CLEAN" | while IFS= read -r line; do
    lineno=$(echo "$line" | cut -d: -f1)
    commits=$(echo "$line" | grep -oP 'commits=\K\d+')
    if [ "$prev_commits" != "0" ] && [ "$prev_commits" != "-1" ] && [ "$commits" = "0" ]; then
        # Show 30 lines before and 10 after the freeze start
        start=$((lineno - 30))
        [ "$start" -lt 1 ] && start=1
        end=$((lineno + 10))
        echo "First freeze at line $lineno. Context (lines $start-$end):"
        sed -n "${start},${end}p" "$CLEAN"
        break
    fi
    prev_commits="$commits"
done
echo

echo "=== ANALYSIS COMPLETE ==="
