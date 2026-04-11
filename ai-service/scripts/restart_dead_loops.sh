#!/usr/bin/env bash
# Restart supported minimal-loop training jobs when the Python loop is absent.
#
# This intentionally bypasses the supervisor wrapper and uses the proven
# direct `ssh -f -n ... nohup ... &` launch path.

set -u

KEY="${HOME}/.ssh/id_cluster"
SSH_USER="ubuntu"
SSH_CONNECT_TIMEOUT=15
VERIFY_DELAY_SECONDS=10
DRY_RUN=false
ONLY_CONFIG=""

SSH_OPTS=()

NODES=(
  "100.121.230.110|hex8_2p|data/minimal_loop_gh200-8|--board-type hex8 --num-players 2 --iterations 50 --games-per-iter 100 --selfplay-budget 200 --eval-budget 128"
  "100.127.168.116|square8_2p|data/minimal_loop_square8_2p|--board-type square8 --num-players 2 --iterations 50 --games-per-iter 100 --selfplay-budget 128 --eval-budget 128 --lr 5e-5 --lr-schedule fixed --train-window 3"
  "100.86.51.4|square8_3p|data/minimal_loop_square8_3p|--board-type square8 --num-players 3 --iterations 50 --games-per-iter 50 --eval-games 30 --selfplay-budget 128 --eval-budget 128 --train-window 3"
  "100.100.19.96|square8_4p|data/minimal_loop_square8_4p|--board-type square8 --num-players 4 --iterations 50 --games-per-iter 40 --eval-games 30 --selfplay-budget 128 --eval-budget 128 --train-window 3"
)

usage() {
  echo "Usage: $0 [--dry-run] [--only <config_key>] [--ssh-key <path>] [--ssh-user <user>] [--verify-delay-seconds <n>]" >&2
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN=true
      shift
      ;;
    --only)
      ONLY_CONFIG="${2:-}"
      if [[ -z "$ONLY_CONFIG" ]]; then
        echo "ERROR: --only requires a config key" >&2
        exit 2
      fi
      shift 2
      ;;
    --ssh-key)
      KEY="${2:-}"
      if [[ -z "$KEY" ]]; then
        echo "ERROR: --ssh-key requires a path" >&2
        exit 2
      fi
      shift 2
      ;;
    --ssh-user)
      SSH_USER="${2:-}"
      if [[ -z "$SSH_USER" ]]; then
        echo "ERROR: --ssh-user requires a value" >&2
        exit 2
      fi
      shift 2
      ;;
    --verify-delay-seconds)
      VERIFY_DELAY_SECONDS="${2:-}"
      if [[ -z "$VERIFY_DELAY_SECONDS" ]]; then
        echo "ERROR: --verify-delay-seconds requires a value" >&2
        exit 2
      fi
      shift 2
      ;;
    *)
      usage
      exit 2
      ;;
  esac
done

SSH_OPTS=(-i "$KEY" -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout="$SSH_CONNECT_TIMEOUT")
OVERALL_RC=0
MATCHED=false

remote_repo_prefix() {
  printf '%s' "cd ~/ringrift/ai-service 2>/dev/null || cd /home/ubuntu/ringrift/ai-service 2>/dev/null || exit 2"
}

for entry in "${NODES[@]}"; do
  IFS='|' read -r ip config workdir loop_args <<<"$entry"
  if [[ -n "$ONLY_CONFIG" && "$config" != "$ONLY_CONFIG" ]]; then
    continue
  fi
  MATCHED=true

  loop_pattern="scripts/[m]inimal_alphazero_loop.py.*--work-dir $workdir"
  supervisor_pattern="scripts/[m]inimal_loop_supervisor.sh.*$workdir"
  watchdog_pattern="scripts/[p]ipeline_watchdog.py.*$workdir"
  remote_log="/tmp/minimal_alphazero_${config}.log"

  echo "=== $config ($ip) ==="
  echo "  work_dir: $workdir"
  echo "  args:     $loop_args"

  if $DRY_RUN; then
    echo "  [DRY] Would probe for '$loop_pattern'"
    echo "  [DRY] If dead, would verify models/canonical_${config}.pth and relaunch via direct nohup"
    echo
    continue
  fi

  alive_output="$(
    ssh "${SSH_OPTS[@]}" "${SSH_USER}@${ip}" "$(remote_repo_prefix); if pgrep -f '$loop_pattern' >/dev/null; then echo alive; else echo dead; fi" 2>/dev/null
  )"
  if [[ "$alive_output" == "alive" ]]; then
    echo "  Loop already alive; no restart needed."
    echo
    continue
  fi

  echo "  Loop not running; verifying model file..."
  if ! ssh "${SSH_OPTS[@]}" "${SSH_USER}@${ip}" "$(remote_repo_prefix); test -f models/canonical_${config}.pth"; then
    echo "  ERROR: missing model file models/canonical_${config}.pth" >&2
    OVERALL_RC=1
    echo
    continue
  fi

  echo "  Cleaning up stale supervisor/watchdog state..."
  ssh "${SSH_OPTS[@]}" "${SSH_USER}@${ip}" "
    $(remote_repo_prefix)
    pkill -f '$supervisor_pattern' 2>/dev/null || true
    pkill -f '$loop_pattern' 2>/dev/null || true
    pkill -f '$watchdog_pattern' 2>/dev/null || true
  " 2>/dev/null || true

  echo "  Launching direct minimal loop..."
  if ! ssh -f -n "${SSH_OPTS[@]}" "${SSH_USER}@${ip}" "
    $(remote_repo_prefix)
    nohup env PYTHONPATH=. venv/bin/python scripts/minimal_alphazero_loop.py \
      --model models/canonical_${config}.pth \
      --work-dir $workdir \
      $loop_args \
      > $remote_log 2>&1 < /dev/null &
  " 2>/dev/null; then
    echo "  ERROR: launch command failed" >&2
    OVERALL_RC=1
    echo
    continue
  fi

  echo "  Verifying launch after ${VERIFY_DELAY_SECONDS}s..."
  sleep "$VERIFY_DELAY_SECONDS"
  if ! ssh "${SSH_OPTS[@]}" "${SSH_USER}@${ip}" "
    $(remote_repo_prefix)
    echo '  Processes:'
    pgrep -af '$loop_pattern' | head -n 4 || true
    echo '  Log tail:'
    tail -n 12 $remote_log 2>/dev/null || true
    if ! pgrep -f '$loop_pattern' >/dev/null; then
      echo 'ERROR: loop not alive after restart' >&2
      exit 1
    fi
  "; then
    OVERALL_RC=1
  fi
  echo
done

if [[ -n "$ONLY_CONFIG" && "$MATCHED" == false ]]; then
  echo "ERROR: config '$ONLY_CONFIG' not found in supported restart list" >&2
  exit 2
fi

if [[ "$OVERALL_RC" -eq 0 ]]; then
  echo "Dead-loop restart check completed."
fi

exit "$OVERALL_RC"
