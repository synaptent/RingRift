#!/usr/bin/env bash
# Role-aware deployment for trainer, selfplay-worker, and evaluator services.
#
# Reads config/node_roles.yaml plus distributed_hosts.yaml and deploys the
# appropriate systemd unit and config file per node role while preserving P2P.
set -euo pipefail

SSH_KEY="${HOME}/.ssh/id_cluster"
SSH_OPTS=(-o IdentitiesOnly=yes -i "$SSH_KEY" -o ConnectTimeout=10 -o StrictHostKeyChecking=no)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
AI_DIR="$(dirname "$SCRIPT_DIR")"

ONLY=""
DRY_RUN=false
RESTART=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --only) ONLY="$2"; shift 2 ;;
    --dry-run) DRY_RUN=true; shift ;;
    --restart) RESTART=true; shift ;;
    *) echo "Unknown flag: $1" >&2; exit 1 ;;
  esac
done

PLAN_JSON="$(AI_DIR="${AI_DIR}" ONLY="${ONLY}" python3 <<'PY'
import json
import os
from pathlib import Path

import yaml

ai_dir = Path(os.environ["AI_DIR"])
only = os.environ.get("ONLY", "").strip()
hosts = yaml.safe_load((ai_dir / "config" / "distributed_hosts.yaml").read_text()) or {}
roles = yaml.safe_load((ai_dir / "config" / "node_roles.yaml").read_text()) or {}

role_nodes = roles.get("nodes", {})
host_nodes = hosts.get("hosts", {})

def normalize(value: str) -> str:
    return (value or "").lower().replace("-", "").replace("_", "")

def find_host_config(name: str):
    norm = normalize(name)
    for host_name, cfg in host_nodes.items():
        if normalize(host_name) == norm or norm in normalize(host_name) or normalize(host_name) in norm:
            return host_name, cfg
    raise KeyError(f"Host not found for node role entry: {name}")

trainer_specs = {
    "hex8_2p": {
        "board_type": "hex8",
        "num_players": 2,
        "games_per_iter": 100,
        "selfplay_budget": 200,
        "eval_budget": 128,
        "lr": "5e-5",
        "lr_schedule": "fixed",
        "train_lr_scheduler": "none",
        "train_window": 5,
        "work_dir": "data/minimal_loop_gh200-8",
        "iterations": 50,
    },
    "square8_2p": {
        "board_type": "square8",
        "num_players": 2,
        "games_per_iter": 100,
        "selfplay_budget": 128,
        "eval_budget": 128,
        "lr": "5e-5",
        "lr_schedule": "fixed",
        "train_lr_scheduler": "none",
        "train_window": 3,
        "work_dir": "data/minimal_loop_square8_2p",
        "iterations": 50,
    },
    "square8_3p": {
        "board_type": "square8",
        "num_players": 3,
        "games_per_iter": 200,
        "selfplay_budget": 128,
        "eval_budget": 128,
        "lr": "5e-5",
        "lr_schedule": "fixed",
        "train_lr_scheduler": "none",
        "train_window": 5,
        "work_dir": "data/minimal_loop_square8_3p",
        "iterations": 50,
    },
    "square19_2p": {
        "board_type": "square19",
        "num_players": 2,
        "games_per_iter": 50,
        "selfplay_budget": 128,
        "eval_budget": 128,
        "lr": "5e-5",
        "lr_schedule": "fixed",
        "train_lr_scheduler": "none",
        "train_window": 5,
        "work_dir": "data/minimal_loop_square19_2p",
        "iterations": 50,
    },
}

plan = []
for node_name, role_cfg in role_nodes.items():
    role = str(role_cfg.get("role", "")).strip()
    if only and only not in {node_name, role_cfg.get("target_config", ""), role}:
        continue

    host_name, host_cfg = find_host_config(node_name)
    ip = host_cfg.get("tailscale_ip") or host_cfg.get("ssh_host") or host_cfg.get("host")
    if not ip:
        continue

    target_config = str(role_cfg.get("target_config", "")).strip()
    trainer_spec = trainer_specs.get(target_config, {})
    entry = {
        "node_name": node_name,
        "host_name": host_name,
        "ip": ip,
        "role": role,
        "target_config": target_config,
        "assigned_configs": role_cfg.get("assigned_configs", []),
        "feeds_trainer": role_cfg.get("feeds_trainer", ""),
        "trainer_spec": trainer_spec,
    }
    if role == "selfplay-worker":
        feed_name = str(role_cfg.get("feeds_trainer", "")).strip()
        feed_cfg = role_nodes.get(feed_name, {})
        feed_target = str(feed_cfg.get("target_config", "")).strip()
        feed_spec = trainer_specs.get(feed_target, {})
        if feed_name and feed_spec:
            _feed_host_name, feed_host_cfg = find_host_config(feed_name)
            entry["trainer_ip"] = (
                feed_host_cfg.get("tailscale_ip") or feed_host_cfg.get("ssh_host") or feed_host_cfg.get("host")
            )
            entry["trainer_work_dir"] = feed_spec.get("work_dir", "")
    plan.append(entry)

print(json.dumps(plan))
PY
)"

if [[ -z "${PLAN_JSON}" || "${PLAN_JSON}" == "[]" ]]; then
  echo "No matching nodes found" >&2
  exit 1
fi

write_remote_config() {
  local ip="$1"
  local remote_path="$2"
  local content="$3"
  local tmp_file
  tmp_file="$(mktemp)"
  printf '%s' "${content}" > "${tmp_file}"
  scp "${SSH_OPTS[@]}" "${tmp_file}" "ubuntu@${ip}:/tmp/$(basename "${remote_path}")"
  ssh -n "${SSH_OPTS[@]}" "ubuntu@${ip}" "sudo mkdir -p /etc/ringrift && sudo mv /tmp/$(basename "${remote_path}") ${remote_path}"
  rm -f "${tmp_file}"
}

install_service() {
  local ip="$1"
  local local_path="$2"
  local remote_name="$3"
  scp "${SSH_OPTS[@]}" "${local_path}" "ubuntu@${ip}:/tmp/${remote_name}"
  ssh -n "${SSH_OPTS[@]}" "ubuntu@${ip}" "sudo mv /tmp/${remote_name} /etc/systemd/system/${remote_name}"
}

while IFS= read -r row; do
  [[ -z "${row}" ]] && continue
  node_name="$(python3 -c 'import json,sys; print(json.loads(sys.argv[1])["node_name"])' "${row}")"
  role="$(python3 -c 'import json,sys; print(json.loads(sys.argv[1])["role"])' "${row}")"
  ip="$(python3 -c 'import json,sys; print(json.loads(sys.argv[1])["ip"])' "${row}")"
  target_config="$(python3 -c 'import json,sys; print(json.loads(sys.argv[1]).get("target_config",""))' "${row}")"
  echo "=== ${node_name} (${role}${target_config:+ / ${target_config}} @ ${ip}) ==="

  if ${DRY_RUN}; then
    python3 - <<'PY' "${row}"
import json, sys
row = json.loads(sys.argv[1])
print("  [dry-run] Would update code on remote node")
print("  [dry-run] Would install the current ringrift-p2p.service and keep it enabled")
role = row["role"]
spec = row.get("trainer_spec", {})
if role == "trainer":
    print(f"  [dry-run] Would write /etc/ringrift/training.conf with work_dir={spec.get('work_dir','')}")
    print("  [dry-run] Would install ringrift-training.service and restart it")
elif role == "selfplay-worker":
    print(f"  [dry-run] Would write /etc/ringrift/selfplay.conf targeting trainer={row.get('feeds_trainer','')}")
    print("  [dry-run] Would install ringrift-selfplay-worker.service and restart it")
elif role == "evaluator":
    print(f"  [dry-run] Would write /etc/ringrift/evaluator.conf for configs={','.join(row.get('assigned_configs', []))}")
    print("  [dry-run] Would install ringrift-evaluator.service and restart it")
PY
    echo ""
    continue
  fi

  echo "  Updating code..."
  ssh -n "${SSH_OPTS[@]}" "ubuntu@${ip}" 'cd ~/ringrift && git fetch origin && git checkout -f origin/main --detach >/dev/null 2>&1 || true'
  ssh -n "${SSH_OPTS[@]}" "ubuntu@${ip}" 'mkdir -p ~/ringrift/ai-service/logs ~/ringrift/ai-service/logs/selfplay'
  install_service "${ip}" "${AI_DIR}/config/systemd/ringrift-p2p.service" "ringrift-p2p.service"

  case "${role}" in
    trainer)
      TRAINING_CONF="$(python3 - <<'PY' "${row}"
import json, sys
row = json.loads(sys.argv[1])
spec = row["trainer_spec"]
config_key = row["target_config"]
print(f"""# RingRift trainer config for {row['node_name']}
TRAINING_MODEL=models/canonical_{config_key}.pth
TRAINING_WORK_DIR={spec['work_dir']}
TRAINING_SUPPLEMENTAL_DATA_DIR={spec['work_dir']}/supplemental
TRAINING_BOARD_TYPE={spec['board_type']}
TRAINING_NUM_PLAYERS={spec['num_players']}
TRAINING_ITERATIONS={spec['iterations']}
TRAINING_GAMES_PER_ITER={spec['games_per_iter']}
TRAINING_SELFPLAY_BUDGET={spec['selfplay_budget']}
TRAINING_EVAL_BUDGET={spec['eval_budget']}
TRAINING_LR={spec['lr']}
TRAINING_LR_SCHEDULE={spec['lr_schedule']}
TRAINING_TRAIN_LR_SCHEDULER={spec['train_lr_scheduler']}
TRAINING_TRAIN_WINDOW={spec['train_window']}
""", end="")
PY
)"
      write_remote_config "${ip}" "/etc/ringrift/training.conf" "${TRAINING_CONF}"
      install_service "${ip}" "${AI_DIR}/config/systemd/ringrift-training.service" "ringrift-training.service"
      ssh -n "${SSH_OPTS[@]}" "ubuntu@${ip}" '
        sudo systemctl stop ringrift-selfplay-worker ringrift-evaluator 2>/dev/null || true
        sudo systemctl daemon-reload
        sudo systemctl enable ringrift-training
        sudo systemctl restart ringrift-training
        sudo systemctl enable ringrift-p2p 2>/dev/null || true
        sudo systemctl restart ringrift-p2p 2>/dev/null || true
      '
      ;;
    selfplay-worker)
      SELFPLAY_CONF="$(python3 - <<'PY' "${row}"
import json, sys
row = json.loads(sys.argv[1])
spec = row["trainer_spec"]
config_key = row["target_config"]
trainer_ip = row.get("trainer_ip", "")
trainer_work_dir = row.get("trainer_work_dir", "")
print(f"""# RingRift selfplay worker config for {row['node_name']}
SELFPLAY_CONFIG_KEY={config_key}
SELFPLAY_MODEL=models/canonical_{config_key}.pth
SELFPLAY_BOARD_TYPE={spec['board_type']}
SELFPLAY_NUM_PLAYERS={spec['num_players']}
SELFPLAY_BATCH_GAMES={max(16, spec['games_per_iter'] // 2)}
SELFPLAY_SIMULATION_BUDGET={spec['selfplay_budget']}
SELFPLAY_RAW_OUTPUT_DIR=data/selfplay/policy_gumbel/{config_key}/raw
SELFPLAY_SUPPLEMENTAL_OUTPUT_DIR=data/selfplay/policy_gumbel/{config_key}/supplemental
SELFPLAY_STATE_DIR=data/selfplay/policy_gumbel/{config_key}/state
SELFPLAY_SLEEP_SECONDS=60
SELFPLAY_REMOTE_HOST={trainer_ip}
SELFPLAY_REMOTE_DIR=/home/ubuntu/ringrift/ai-service/{trainer_work_dir}/supplemental
SELFPLAY_REMOTE_USER=ubuntu
SELFPLAY_REMOTE_SSH_KEY=/home/ubuntu/.ssh/id_cluster
SELFPLAY_REMOTE_PORT=22
SELFPLAY_OPPONENT_TYPE=selfplay
""", end="")
PY
)"
      write_remote_config "${ip}" "/etc/ringrift/selfplay.conf" "${SELFPLAY_CONF}"
      install_service "${ip}" "${AI_DIR}/config/systemd/ringrift-selfplay-worker.service" "ringrift-selfplay-worker.service"
      ssh -n "${SSH_OPTS[@]}" "ubuntu@${ip}" '
        sudo systemctl stop ringrift-training ringrift-evaluator 2>/dev/null || true
        sudo systemctl daemon-reload
        sudo systemctl enable ringrift-selfplay-worker
        sudo systemctl restart ringrift-selfplay-worker
        sudo systemctl enable ringrift-p2p 2>/dev/null || true
        sudo systemctl restart ringrift-p2p 2>/dev/null || true
      '
      ;;
    evaluator)
      EVALUATOR_CONF="$(python3 - <<'PY' "${row}"
import json, sys
row = json.loads(sys.argv[1])
assigned = ",".join(row.get("assigned_configs", []))
print(f"""# RingRift evaluator config for {row['node_name']}
EVALUATOR_INTERVAL_SECONDS=3600
EVALUATOR_WORKERS=64
EVALUATOR_BOARD_FILTER=
EVALUATOR_ASSIGNED_CONFIGS={assigned}
""", end="")
PY
)"
      write_remote_config "${ip}" "/etc/ringrift/evaluator.conf" "${EVALUATOR_CONF}"
      install_service "${ip}" "${AI_DIR}/config/systemd/ringrift-evaluator.service" "ringrift-evaluator.service"
      ssh -n "${SSH_OPTS[@]}" "ubuntu@${ip}" '
        sudo systemctl stop ringrift-training ringrift-selfplay-worker 2>/dev/null || true
        sudo systemctl daemon-reload
        sudo systemctl enable ringrift-evaluator
        sudo systemctl restart ringrift-evaluator
        sudo systemctl enable ringrift-p2p 2>/dev/null || true
        sudo systemctl restart ringrift-p2p 2>/dev/null || true
      '
      ;;
    *)
      echo "  Unsupported role: ${role}" >&2
      exit 1
      ;;
  esac

  if ${RESTART}; then
    ssh -n "${SSH_OPTS[@]}" "ubuntu@${ip}" 'sudo systemctl restart ringrift-p2p 2>/dev/null || true'
  fi

  SERVICE_NAME="ringrift-${role}"
  if [[ "${role}" == "selfplay-worker" ]]; then
    SERVICE_NAME="ringrift-selfplay-worker"
  elif [[ "${role}" == "trainer" ]]; then
    SERVICE_NAME="ringrift-training"
  elif [[ "${role}" == "evaluator" ]]; then
    SERVICE_NAME="ringrift-evaluator"
  fi
  STATUS="$(ssh -n "${SSH_OPTS[@]}" "ubuntu@${ip}" "systemctl is-active ${SERVICE_NAME} 2>/dev/null || echo unknown")"
  P2P_STATUS="$(ssh -n "${SSH_OPTS[@]}" "ubuntu@${ip}" 'systemctl is-active ringrift-p2p 2>/dev/null || echo unknown')"
  echo "  Status: ${SERVICE_NAME}=${STATUS}, ringrift-p2p=${P2P_STATUS}"
  echo ""
done < <(python3 - <<'PY' "${PLAN_JSON}"
import json, sys
for row in json.loads(sys.argv[1]):
    print(json.dumps(row))
PY
)

echo "Done. Use --dry-run to inspect the role plan without SSH."
