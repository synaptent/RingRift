#!/usr/bin/env bash
# Restart a supported minimal AlphaZero loop if the Python process exits.
#
# This is intentionally small and process-local: it supervises one command on
# one node without touching unrelated Python, P2P, or selfplay jobs.

set -u

RESTART_DELAY_SECONDS=60

while [[ $# -gt 0 ]]; do
  case "$1" in
    --restart-delay-seconds)
      RESTART_DELAY_SECONDS="${2:-}"
      if [[ -z "$RESTART_DELAY_SECONDS" ]]; then
        echo "ERROR: --restart-delay-seconds requires a value" >&2
        exit 2
      fi
      shift 2
      ;;
    --)
      shift
      break
      ;;
    *)
      echo "Usage: $0 [--restart-delay-seconds N] -- <command> [args...]" >&2
      exit 2
      ;;
  esac
done

if [[ $# -eq 0 ]]; then
  echo "ERROR: missing supervised command" >&2
  exit 2
fi

echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) supervisor: command=$*"

while true; do
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) supervisor: starting"
  "$@"
  rc=$?
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) supervisor: exited rc=$rc; restarting in ${RESTART_DELAY_SECONDS}s"
  sleep "$RESTART_DELAY_SECONDS"
done
