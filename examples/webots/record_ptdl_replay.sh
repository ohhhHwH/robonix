#!/usr/bin/env bash
# Record an exact-query PTDL replay beside the live Webots window.
#
# Usage:
#   bash examples/webots/record_ptdl_replay.sh [output.webm]
#
# Prerequisites: Webots and Robonix are already ready at the route origin,
# DISPLAY points at the visible desktop, and the local fake-LLM request log
# exists. The default query replays the verified manual route to the door.
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DISPLAY="${DISPLAY:-:0}"
VIDEO_DIR="${HOME}/Videos"
STAMP="$(date +%Y%m%d-%H%M%S)"
OUTPUT="${1:-${VIDEO_DIR}/robonix-ptdl-replay-${STAMP}.webm}"
TERMINAL_LOG="${OUTPUT%.webm}.terminal.log"
RECORDER_LOG="${OUTPUT%.webm}.gstreamer.log"
FAKE_REQUEST_LOG="${FAKE_LLM_REQUEST_LOG:-/tmp/robonix-fake-llm/fake-requests.log}"
DEFAULT_QUERY='手动探索到办公室西侧实体门位置：先向后转，再以每次前进 1 米的离散步骤移动，只探索到门旁安全位置后停止。'
QUERY="${ROBONIX_REPLAY_QUERY:-$DEFAULT_QUERY}"
RECORDER_PID=""
TERMINAL_PID=""
START_MARKER="/tmp/robonix-ptdl-replay-start-$$"

# Stop GStreamer through SIGINT so `-e` writes the final WebM cues/index.
stop_recorder() {
    if [[ -z "$RECORDER_PID" ]] || ! kill -0 "$RECORDER_PID" 2>/dev/null; then
        return
    fi
    kill -INT "$RECORDER_PID" 2>/dev/null || true
    for _ in $(seq 1 50); do
        if ! kill -0 "$RECORDER_PID" 2>/dev/null; then
            wait "$RECORDER_PID" 2>/dev/null || true
            return
        fi
        sleep 0.2
    done
    kill -TERM "$RECORDER_PID" 2>/dev/null || true
    wait "$RECORDER_PID" 2>/dev/null || true
}

# Close only the terminal instance launched by this recording run.
stop_terminal() {
    if [[ -n "$TERMINAL_PID" ]] && kill -0 "$TERMINAL_PID" 2>/dev/null; then
        kill -TERM "$TERMINAL_PID" 2>/dev/null || true
        wait "$TERMINAL_PID" 2>/dev/null || true
    fi
}

cleanup() {
    stop_terminal
    stop_recorder
    rm -f -- "$START_MARKER"
}
trap cleanup EXIT INT TERM

for command in gst-launch-1.0 gnome-terminal rbnx; do
    command -v "$command" >/dev/null || {
        echo "missing required command: $command" >&2
        exit 1
    }
done

for plugin in ximagesrc compositor vp8enc webmmux; do
    gst-inspect-1.0 "$plugin" >/dev/null 2>&1 || {
        echo "GStreamer $plugin plugin is unavailable" >&2
        exit 1
    }
done
DISPLAY="$DISPLAY" xdpyinfo >/dev/null 2>&1 || {
    echo "X display $DISPLAY is unavailable" >&2
    exit 1
}

mkdir -p "$(dirname "$OUTPUT")"
touch "$FAKE_REQUEST_LOG"
if [[ "$(wc -l < "$FAKE_REQUEST_LOG")" -ne 0 ]]; then
    echo "fake LLM request log is not empty before recording: $FAKE_REQUEST_LOG" >&2
    exit 1
fi

cd "$ROOT_DIR"
rbnx caps >/dev/null

WEBOTS_WINDOW="$(DISPLAY="$DISPLAY" xwininfo -root -tree 2>/dev/null \
    | awk '/Webots R[0-9]/ && /webots-bin/ {print $1; exit}')"
if [[ -z "$WEBOTS_WINDOW" ]]; then
    echo "could not locate the Webots client window" >&2
    exit 1
fi

export ROBONIX_REPLAY_QUERY="$QUERY"
export ROBONIX_REPLAY_LOG="$TERMINAL_LOG"
export ROBONIX_REPLAY_START_MARKER="$START_MARKER"
rm -f -- "$START_MARKER"

# Open the terminal first and hold the header long enough for the recorder to
# attach directly to both client windows. Direct window capture avoids a black
# root image on NVIDIA X11 desktops that use compositor redirection.
set +e
DISPLAY="$DISPLAY" gnome-terminal --wait \
    --geometry=65x24+40+80 \
    --zoom=1.0 \
    --title='Robonix PTDL History Replay' \
    -- bash -lc '
        set -o pipefail
        cd /home/user/robonix
        clear
        printf "\n============================================================\n"
        printf " Robonix PTDL exact-query replay (fake API key validation)\n"
        printf "============================================================\n\n"
        printf "Query:\n%s\n\n" "$ROBONIX_REPLAY_QUERY"
        printf "Expected: hit historical question, skip LLM, execute stored RTDL.\n\n"
        printf "Command ready; recording starts before dispatch.\n"
        while [[ ! -e "$ROBONIX_REPLAY_START_MARKER" ]]; do sleep 0.1; done
        printf "\n$ rbnx ask \"%s\"\n\n" "$ROBONIX_REPLAY_QUERY"
        timeout 1800 rbnx ask "$ROBONIX_REPLAY_QUERY" 2>&1 | tee "$ROBONIX_REPLAY_LOG"
        status=${PIPESTATUS[0]}
        printf "\nrbnx ask exit status: %d\n" "$status"
        if [[ "$status" -eq 0 ]]; then
            printf "Replay command completed.\n"
        else
            printf "Replay command failed.\n"
        fi
        printf "Keeping this result visible for 10 seconds...\n"
        sleep 10
        exit "$status"
    ' &
TERMINAL_PID=$!
set -e

TERMINAL_WINDOW=""
for _ in $(seq 1 80); do
    TERMINAL_WINDOW="$(DISPLAY="$DISPLAY" xwininfo -root -tree 2>/dev/null \
        | awk '/Robonix PTDL History Replay/ && /gnome-terminal-server/ {print $1; exit}')"
    [[ -n "$TERMINAL_WINDOW" ]] && break
    sleep 0.1
done
if [[ -z "$TERMINAL_WINDOW" ]]; then
    echo "could not locate the replay terminal window" >&2
    exit 1
fi

# Capture Webots as the 1080p background and overlay the terminal. BGRx keeps
# both X11 sources opaque; alpha formats expose X's unused zero alpha channel.
DISPLAY="$DISPLAY" gst-launch-1.0 -e \
    compositor name=mix background=black start-time-selection=first \
        sink_0::zorder=0 sink_0::ypos=34 \
        sink_1::zorder=1 sink_1::xpos=30 sink_1::ypos=45 sink_1::alpha=0.97 \
    '!' 'video/x-raw,format=BGRx,width=1920,height=1080,framerate=12/1' \
    '!' videoconvert \
    '!' queue \
    '!' vp8enc deadline=1 cpu-used=8 threads=8 target-bitrate=5000000 keyframe-max-dist=120 \
    '!' webmmux \
    '!' filesink location="$OUTPUT" \
    ximagesrc display-name="$DISPLAY" xid=$((WEBOTS_WINDOW)) \
        do-timestamp=true use-damage=false show-pointer=false \
    '!' videoconvert \
    '!' videoscale \
    '!' 'video/x-raw,format=BGRx,width=1920,height=1011,framerate=12/1' \
    '!' queue \
    '!' mix.sink_0 \
    ximagesrc display-name="$DISPLAY" xid=$((TERMINAL_WINDOW)) \
        do-timestamp=true use-damage=false show-pointer=false \
    '!' videoconvert \
    '!' videoscale \
    '!' 'video/x-raw,format=BGRx,width=920,height=760,framerate=12/1' \
    '!' queue \
    '!' mix.sink_1 \
    >"$RECORDER_LOG" 2>&1 &
RECORDER_PID=$!
sleep 3
kill -0 "$RECORDER_PID" 2>/dev/null || {
    cat "$RECORDER_LOG" >&2
    exit 1
}
touch "$START_MARKER"

set +e
wait "$TERMINAL_PID"
ASK_STATUS=$?
set -e
TERMINAL_PID=""

sleep 3
stop_recorder
RECORDER_PID=""
trap - EXIT INT TERM

REQUESTS="$(wc -l < "$FAKE_REQUEST_LOG")"
HISTORY_HIT=0
grep -q '已命中完整历史方案' "$TERMINAL_LOG" && HISTORY_HIT=1

[[ "$ASK_STATUS" -eq 0 ]] || exit "$ASK_STATUS"
[[ "$HISTORY_HIT" -eq 1 ]] || {
    echo "terminal output did not contain the PTDL history-hit message" >&2
    exit 1
}
[[ "$REQUESTS" -eq 0 ]] || {
    echo "fake LLM endpoint was called" >&2
    cat "$FAKE_REQUEST_LOG" >&2
    exit 1
}
[[ -s "$OUTPUT" ]] || {
    echo "recording output is empty" >&2
    exit 1
}

printf 'video=%s\n' "$OUTPUT"
printf 'terminal_log=%s\n' "$TERMINAL_LOG"
printf 'recorder_log=%s\n' "$RECORDER_LOG"
printf 'ask_status=%s\n' "$ASK_STATUS"
printf 'history_hit=%s\n' "$HISTORY_HIT"
printf 'fake_llm_requests=%s\n' "$REQUESTS"
