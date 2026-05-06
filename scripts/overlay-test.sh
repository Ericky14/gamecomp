#!/usr/bin/env bash
set -euo pipefail

# Overlay integration test: launches gamecomp with an app, then launches
# a second app as an overlay on top.
#
# Usage: ./scripts/overlay-test.sh <app> <overlay_app> [width] [height] [opacity]
# Example: ./scripts/overlay-test.sh vkcube glxgears

APP="${1:?Usage: $0 <app> <overlay_app> [width] [height] [opacity]}"
OVERLAY_APP="${2:?Usage: $0 <app> <overlay_app> [width] [height] [opacity]}"
WIDTH="${3:-1280}"
HEIGHT="${4:-720}"
OPACITY="${5:-0xB0000000}"
GCLOG=/tmp/gamecomp-overlay-test.log

echo "=== Overlay Integration Test ==="
echo "App:     $APP"
echo "Overlay: $OVERLAY_APP"

# Build gamecomp first.
cargo build || exit 1

# Snapshot existing X11 sockets BEFORE launching gamecomp.
BEFORE_SOCKETS=$(ls /tmp/.X11-unix/ 2>/dev/null | sort)

# Launch gamecomp with the app, capturing logs to a file.
: > "$GCLOG"
./target/debug/gamecomp --xwayland-count 1 -W "$WIDTH" -H "$HEIGHT" -- "$APP" > "$GCLOG" 2>&1 &
GAMECOMP_PID=$!

cleanup() {
    kill "$GAMECOMP_PID" 2>/dev/null || true
    kill "$OV_PID" 2>/dev/null || true
    exit 0
}
trap cleanup INT TERM

echo "gamecomp started (PID $GAMECOMP_PID), log at $GCLOG"
echo "waiting for XWayland display..."

# Poll for a NEW X11 socket that wasn't present before we launched.
# XWayland may take 6+ seconds due to host configure timeout.
GC_DISPLAY=""
for _ in $(seq 1 60); do
    AFTER_SOCKETS=$(ls /tmp/.X11-unix/ 2>/dev/null | sort)
    NEW=$(comm -13 <(echo "$BEFORE_SOCKETS") <(echo "$AFTER_SOCKETS") | head -1)
    if [[ -n "$NEW" ]]; then
        num="${NEW#X}"
        GC_DISPLAY=":${num}"
        # Wait for XWM to set up atoms.
        sleep 1
        break
    fi
    sleep 0.5
done

if [[ -z "$GC_DISPLAY" ]]; then
    echo "ERROR: Could not find gamecomp display after 30s"
    echo "--- gamecomp log tail ---"
    tail -20 "$GCLOG" 2>/dev/null
    kill "$GAMECOMP_PID" 2>/dev/null
    exit 1
fi
echo "gamecomp display: $GC_DISPLAY"

# Launch the overlay app on gamecomp's display.
sleep 1
echo "launching overlay: DISPLAY=$GC_DISPLAY $OVERLAY_APP"
DISPLAY="$GC_DISPLAY" $OVERLAY_APP &
OV_PID=$!
echo "overlay launched (PID $OV_PID)"

# Wait for the overlay window.
OV_WIN=""
for _ in $(seq 1 50); do
    OV_WIN=$(DISPLAY="$GC_DISPLAY" xdotool search --pid "$OV_PID" 2>/dev/null | head -1) || true
    if [[ -z "$OV_WIN" ]]; then
        OV_WIN=$(DISPLAY="$GC_DISPLAY" xdotool search --name "$(basename "$OVERLAY_APP")" 2>/dev/null | head -1) || true
    fi
    if [[ -n "$OV_WIN" ]]; then break; fi
    sleep 0.1
done

if [[ -n "$OV_WIN" ]]; then
    OV_HEX=$(printf "0x%x" "$OV_WIN")
    echo "Overlay window: $OV_HEX (PID $OV_PID)"

    DISPLAY="$GC_DISPLAY" xprop -id "$OV_WIN" -f STEAM_OVERLAY 32c \
        -set STEAM_OVERLAY 1
    DISPLAY="$GC_DISPLAY" xprop -id "$OV_WIN" -f STEAM_INPUT_FOCUS 32c \
        -set STEAM_INPUT_FOCUS 1
    DISPLAY="$GC_DISPLAY" xprop -id "$OV_WIN" -f _NET_WM_OPACITY 32c \
        -set _NET_WM_OPACITY "$OPACITY"
    echo "Overlay marked: STEAM_OVERLAY=1, STEAM_INPUT_FOCUS=1, opacity=$OPACITY"
else
    echo "WARNING: Could not find overlay window (continuing without overlay)"
    echo "--- gamecomp log tail ---"
    tail -20 "$GCLOG"
fi

echo ""
echo "Test running. Press Ctrl+C to stop."
echo "gamecomp log: $GCLOG"
wait "$GAMECOMP_PID" 2>/dev/null || true
kill "$OV_PID" 2>/dev/null || true
