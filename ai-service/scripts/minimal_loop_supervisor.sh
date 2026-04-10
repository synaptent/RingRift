#!/usr/bin/env bash
# Restart a supported minimal AlphaZero loop if the Python process exits.
#
# This is intentionally small and process-local: it supervises one command on
# one node without touching unrelated Python, P2P, or selfplay jobs.

set -u

RESTART_DELAY_SECONDS=60
MAX_RESTARTS=10
FAST_CRASH_SECONDS=60
CONFIG=""

usage() {
  echo "Usage: $0 [--config CONFIG] [--restart-delay-seconds N] [--max-restarts N] -- <command> [args...]" >&2
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      CONFIG="${2:-}"
      if [[ -z "$CONFIG" ]]; then
        echo "ERROR: --config requires a value" >&2
        exit 2
      fi
      shift 2
      ;;
    --restart-delay-seconds)
      RESTART_DELAY_SECONDS="${2:-}"
      if [[ -z "$RESTART_DELAY_SECONDS" ]]; then
        echo "ERROR: --restart-delay-seconds requires a value" >&2
        exit 2
      fi
      shift 2
      ;;
    --max-restarts)
      MAX_RESTARTS="${2:-}"
      if [[ -z "$MAX_RESTARTS" ]]; then
        echo "ERROR: --max-restarts requires a value" >&2
        exit 2
      fi
      shift 2
      ;;
    --)
      shift
      break
      ;;
    *)
      usage
      exit 2
      ;;
  esac
done

if [[ $# -eq 0 ]]; then
  echo "ERROR: missing supervised command" >&2
  usage
  exit 2
fi

infer_config() {
  local prev=""
  local arg=""
  for arg in "$@"; do
    if [[ "$prev" == "--model" && "$arg" =~ canonical_([^/]+)\.pth$ ]]; then
      echo "${BASH_REMATCH[1]}"
      return
    fi
    prev="$arg"
  done
  echo "unknown"
}

safe_config_name() {
  printf '%s' "$1" | tr -c 'A-Za-z0-9_.-' '_'
}

if [[ -z "$CONFIG" ]]; then
  CONFIG="$(infer_config "$@")"
fi

SAFE_CONFIG="$(safe_config_name "$CONFIG")"
HEARTBEAT_FILE="/tmp/supervisor_${SAFE_CONFIG}.heartbeat"
CONSECUTIVE_FAST_CRASHES=0
CHILD_PID=""
HEARTBEAT_PID=""
CHILD_TAIL_FILE=""

write_heartbeat() {
  local state="$1"
  local child_pid="${2:-}"
  local now
  now="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  cat >"$HEARTBEAT_FILE" <<EOF
{"timestamp":"$now","config":"$CONFIG","state":"$state","child_pid":"$child_pid","consecutive_fast_crashes":$CONSECUTIVE_FAST_CRASHES,"max_restarts":$MAX_RESTARTS}
EOF
}

cleanup() {
  if [[ -n "${HEARTBEAT_PID:-}" ]]; then
    kill "$HEARTBEAT_PID" 2>/dev/null || true
  fi
  if [[ -n "${CHILD_PID:-}" ]]; then
    kill "$CHILD_PID" 2>/dev/null || true
  fi
  if [[ -n "${CHILD_TAIL_FILE:-}" && -f "$CHILD_TAIL_FILE" ]]; then
    rm -f "$CHILD_TAIL_FILE"
  fi
  write_heartbeat "stopped" ""
}
trap cleanup EXIT INT TERM

echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) supervisor: config=$CONFIG heartbeat=$HEARTBEAT_FILE command=$*"

while true; do
  if [[ "$MAX_RESTARTS" -gt 0 && "$CONSECUTIVE_FAST_CRASHES" -ge "$MAX_RESTARTS" ]]; then
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) supervisor: stopping after ${CONSECUTIVE_FAST_CRASHES} consecutive fast crashes (<${FAST_CRASH_SECONDS}s)" >&2
    write_heartbeat "fast_crash_limit" ""
    exit 1
  fi

  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) supervisor: starting"
  CHILD_TAIL_FILE="$(mktemp "/tmp/minimal_loop_${SAFE_CONFIG}.tail.XXXXXX")"
  start_ts="$(date +%s)"
  "$@" > >(tee -a "$CHILD_TAIL_FILE") 2> >(tee -a "$CHILD_TAIL_FILE" >&2) &
  CHILD_PID="$!"

  (
    while kill -0 "$CHILD_PID" 2>/dev/null; do
      write_heartbeat "running" "$CHILD_PID"
      sleep 30
    done
  ) &
  HEARTBEAT_PID="$!"

  wait "$CHILD_PID"
  rc=$?
  end_ts="$(date +%s)"
  runtime=$((end_ts - start_ts))

  kill "$HEARTBEAT_PID" 2>/dev/null || true
  HEARTBEAT_PID=""
  CHILD_PID=""

  if [[ "$rc" -ne 0 && "$runtime" -lt "$FAST_CRASH_SECONDS" ]]; then
    CONSECUTIVE_FAST_CRASHES=$((CONSECUTIVE_FAST_CRASHES + 1))
  else
    CONSECUTIVE_FAST_CRASHES=0
  fi

  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) supervisor: exited rc=$rc runtime=${runtime}s fast_crashes=${CONSECUTIVE_FAST_CRASHES}; restarting in ${RESTART_DELAY_SECONDS}s"
  if [[ "$rc" -ne 0 ]]; then
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) supervisor: child crash tail follows" >&2
    tail -n 80 "$CHILD_TAIL_FILE" >&2 || true
  fi
  rm -f "$CHILD_TAIL_FILE"
  CHILD_TAIL_FILE=""
  write_heartbeat "restarting" ""
  sleep "$RESTART_DELAY_SECONDS"
done
