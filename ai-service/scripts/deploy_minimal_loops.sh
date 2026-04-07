#!/usr/bin/env bash
# Deploy and restart minimal AlphaZero loops on all Lambda nodes.
# Always kills the old process after deploying new code.
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

# Node assignments:
#   ip config board players workdir games_per_iter eval_games budget
#
# hex8_2p is close to the promotion threshold and has been rejecting
# borderline candidates on a 50-game eval gate. Give that config more
# evaluation games to reduce promotion noise without slowing the other
# minimal-loop nodes.
#
# square8_3p is the 3p canary: run a smaller 50-game selfplay / 30-game eval
# profile so we can prove iterative improvement faster before changing the
# rest of the multiplayer fleet.
NODES=(
  "100.121.230.110 hex8_2p hex8 2 data/minimal_loop_gh200-8 100 100 128"
  "100.127.168.116 square8_2p square8 2 data/minimal_loop_square8_2p 100 50 128"
  "100.86.51.4 square8_3p square8 3 data/minimal_loop_square8_3p 50 30 128"
  "100.91.39.59 hex8_3p hex8 3 data/minimal_loop_hex8_3p 100 50 128"
)

DRY_RUN=false
ONLY_CONFIG=""
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
    *)
      echo "Usage: $0 [--dry-run] [--only <config_key>]" >&2
      exit 1
      ;;
  esac
done

MATCHED=false

for entry in "${NODES[@]}"; do
  read -r ip config board players workdir games_per_iter eval_games budget <<< "$entry"
  if [[ -n "$ONLY_CONFIG" && "$config" != "$ONLY_CONFIG" ]]; then
    continue
  fi
  MATCHED=true
  echo "=== $config ($ip) ==="

  if $DRY_RUN; then
    echo "  [DRY] Would deploy $SCRIPT and restart"
    continue
  fi

  # 1. Deploy scripts
  echo "  Deploying scripts..."
  scp "${SSH_OPTS[@]}" "$SCRIPT" "ubuntu@${ip}:~/ringrift/ai-service/$SCRIPT" >/dev/null
  scp "${SSH_OPTS[@]}" "$WATCHDOG" "ubuntu@${ip}:~/ringrift/ai-service/$WATCHDOG" >/dev/null

  # 2. Kill old process (always — prevents stale process bug)
  echo "  Killing old process..."
  ssh "${SSH_OPTS[@]}" "ubuntu@${ip}" "killall -u ubuntu python 2>/dev/null || true" 2>/dev/null

  # 3. Kill any competing P2P selfplay
  ssh "${SSH_OPTS[@]}" "ubuntu@${ip}" "sudo killall -9 gumbel_selfplay p2p_orchestrator 2>/dev/null || true" 2>/dev/null

  # 4. Start new process
  echo "  Starting minimal loop ($config)..."
  ssh "${SSH_OPTS[@]}" "ubuntu@${ip}" "
    sleep 2
    cd ~/ringrift/ai-service && \
    PYTHONPATH=. nohup venv/bin/python scripts/minimal_alphazero_loop.py \
      --model models/canonical_${config}.pth \
      --board-type $board --num-players $players \
      --iterations 50 --games-per-iter $games_per_iter --eval-games $eval_games --budget $budget \
      --work-dir $workdir \
      </dev/null > /tmp/minimal_alphazero.log 2>&1 &
    echo PID=\$!
  " 2>/dev/null

  # 5. Verify
  sleep 3
  ssh "${SSH_OPTS[@]}" "ubuntu@${ip}" "pgrep -c -f minimal_alphazero 2>/dev/null | xargs echo '  Running:' 'processes'" 2>/dev/null

  echo "  Done."
  echo ""
done

if [[ -n "$ONLY_CONFIG" && "$MATCHED" == false ]]; then
  echo "ERROR: config '$ONLY_CONFIG' not found in deploy list" >&2
  exit 1
fi

echo "All nodes deployed and restarted."
