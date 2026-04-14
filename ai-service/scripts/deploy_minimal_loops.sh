#!/usr/bin/env bash
# Deploy and restart the supported minimal AlphaZero loop configs on cluster nodes.
#
# Usage:
#   cd ai-service && bash scripts/deploy_minimal_loops.sh
#   bash scripts/deploy_minimal_loops.sh --dry-run
#   bash scripts/deploy_minimal_loops.sh --only square8_3p

set -euo pipefail

KEY="${HOME}/.ssh/id_cluster"
SSH_OPTS=(-i "$KEY" -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15)
SCRIPT="scripts/minimal_alphazero_loop.py"
WATCHDOG="scripts/pipeline_watchdog.py"
SUPERVISOR="scripts/minimal_loop_supervisor.sh"
LOCAL_PYTHON="venv/bin/python"
if [[ ! -x "$LOCAL_PYTHON" ]]; then
  LOCAL_PYTHON="python3"
fi
PREFLIGHT_TEST="tests/unit/scripts/test_minimal_alphazero_loop.py"

# Node assignments:
#   ip|config|workdir|args...
#
# These are the current supported experiment profiles:
# - hex8_2p: split-budget plateau canary
# - square8_2p: fixed-LR canary with short train window
# - square8_3p: seat-fair multiplayer revalidation canary
# - square8_4p: smaller 4p revalidation canary
NODES=(
  "100.121.230.110|hex8_2p|data/minimal_loop_gh200-8|--board-type hex8 --num-players 2 --iterations 50 --games-per-iter 100 --selfplay-budget 200 --eval-budget 128"
  "100.127.168.116|square8_2p|data/minimal_loop_square8_2p|--board-type square8 --num-players 2 --iterations 50 --games-per-iter 100 --selfplay-budget 128 --eval-budget 128 --lr 5e-5 --lr-schedule fixed --train-window 3"
  "100.86.51.4|square8_3p|data/minimal_loop_square8_3p|--board-type square8 --num-players 3 --iterations 50 --games-per-iter 50 --eval-games 30 --selfplay-budget 128 --eval-budget 128 --train-window 3"
  "100.100.19.96|square8_4p|data/minimal_loop_square8_4p|--board-type square8 --num-players 4 --iterations 50 --games-per-iter 40 --eval-games 30 --selfplay-budget 128 --eval-budget 128 --train-window 3"
)

DRY_RUN=false
ONLY_CONFIG=""
SKIP_PREFLIGHT=false
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
        exit 1
      fi
      shift 2
      ;;
    --skip-preflight)
      SKIP_PREFLIGHT=true
      shift
      ;;
    *)
      echo "Usage: $0 [--dry-run] [--only <config_key>] [--skip-preflight]" >&2
      exit 1
      ;;
  esac
done

if $SKIP_PREFLIGHT; then
  echo "Preflight: skipped (--skip-preflight)"
elif $DRY_RUN; then
  echo "Preflight: [DRY] Would run PYTHONPATH=. $LOCAL_PYTHON -m pytest -q $PREFLIGHT_TEST"
else
  echo "Preflight: running PYTHONPATH=. $LOCAL_PYTHON -m pytest -q $PREFLIGHT_TEST"
  PYTHONPATH=. "$LOCAL_PYTHON" -m pytest -q "$PREFLIGHT_TEST"
fi
echo

MATCHED=false

for entry in "${NODES[@]}"; do
  IFS='|' read -r ip config workdir loop_args <<<"$entry"
  if [[ -n "$ONLY_CONFIG" && "$config" != "$ONLY_CONFIG" ]]; then
    continue
  fi

  MATCHED=true
  echo "=== $config ($ip) ==="
  echo "  work_dir: $workdir"
  echo "  args:     $loop_args"

  if $DRY_RUN; then
    echo "  [DRY] Would deploy $SCRIPT and restart"
    echo
    continue
  fi

  echo "  Deploying scripts..."
  scp "${SSH_OPTS[@]}" "$SCRIPT" "ubuntu@${ip}:~/ringrift/ai-service/$SCRIPT" >/dev/null
  scp "${SSH_OPTS[@]}" "$WATCHDOG" "ubuntu@${ip}:~/ringrift/ai-service/$WATCHDOG" >/dev/null
  scp "${SSH_OPTS[@]}" "$SUPERVISOR" "ubuntu@${ip}:~/ringrift/ai-service/$SUPERVISOR" >/dev/null

  echo "  Stopping old supported loop for $workdir..."
  ssh "${SSH_OPTS[@]}" "ubuntu@${ip}" "
    pkill -f 'scripts/[m]inimal_loop_supervisor.sh.*$workdir' 2>/dev/null || true
    pkill -f 'scripts/[m]inimal_alphazero_loop.py.*--work-dir $workdir' 2>/dev/null || true
    pkill -f 'scripts/[p]ipeline_watchdog.py.*$workdir' 2>/dev/null || true
  " 2>/dev/null

  echo "  Starting supervised minimal loop ($config)..."
  remote_log="/tmp/minimal_alphazero_${config}.log"
  ssh -f -n "${SSH_OPTS[@]}" "ubuntu@${ip}" "
    cd ~/ringrift/ai-service && \
    chmod +x $SUPERVISOR && \
    nohup env PYTHONPATH=. bash $SUPERVISOR --config $config --restart-delay-seconds 60 --max-restarts 10 -- \
      venv/bin/python scripts/minimal_alphazero_loop.py \
        --model models/canonical_${config}.pth \
        --work-dir $workdir \
        $loop_args \
      > $remote_log 2>&1 < /dev/null &
  " 2>/dev/null

  echo "  Verifying launch after 10s..."
  sleep 10
  ssh "${SSH_OPTS[@]}" "ubuntu@${ip}" "
    echo '  Processes:'
    pgrep -af 'scripts/[m]inimal_loop_supervisor.sh.*$workdir|scripts/[m]inimal_alphazero_loop.py.*--work-dir $workdir' | head -n 6 || true
    echo '  Log tail:'
    tail -n 12 $remote_log 2>/dev/null || true
    if ! pgrep -f 'scripts/[m]inimal_loop_supervisor.sh.*$workdir' >/dev/null; then
      echo 'ERROR: supervisor not alive after launch' >&2
      exit 1
    fi
  "
  echo
done

if [[ -n "$ONLY_CONFIG" && "$MATCHED" == false ]]; then
  echo "ERROR: config '$ONLY_CONFIG' not found in deploy list" >&2
  exit 1
fi

echo "Selected nodes deployed and restarted."
