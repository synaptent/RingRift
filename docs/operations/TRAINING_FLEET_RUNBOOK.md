# Training Fleet Runbook

This runbook covers the checked-in operational path for RingRift training nodes.
It is intentionally read-only guidance: do not restart or redeploy live nodes
unless you are doing an explicit operational task.

Use this with:

- `docs/data/training_fleet_manifest.json` for the current role/config map.
- `ai-service/config/node_roles.yaml` for checked-in workload roles.
- `ai-service/config/distributed_hosts.yaml` for the private runtime host
  inventory used by role-aware deployment.

## Preconditions

- Run commands from the repository root unless a command says `cd ai-service`.
- SSH access uses `~/.ssh/id_cluster`.
- The target nodes are expected to have `~/ringrift` checked out and a Python
  virtual environment under `~/ringrift/ai-service/venv`.
- The role-aware systemd deployment requires the private, untracked
  `ai-service/config/distributed_hosts.yaml`. The checked-in
  `distributed_hosts.yaml.example` and `distributed_hosts.template.yaml` files
  are templates, not live inventory.

## Local Preflight

Before deploying code or restarting a trainer, run the local preflight slice:

```bash
cd ai-service
PYTHONPATH=. python3 scripts/validate_training_fleet_docs.py
PYTHONPATH=. python3 -m pytest -q tests/unit/scripts/test_minimal_alphazero_loop.py
bash scripts/deploy_minimal_loops.sh --dry-run
```

If you are using the role-aware systemd path and have the private host inventory:

```bash
cd ai-service
bash scripts/deploy_training_service.sh --dry-run
```

Do not use `--skip-preflight` except for an explicit break-glass recovery where
the risk is understood and documented.

## Minimal-Loop Canary Deployment

`deploy_minimal_loops.sh` is the current supported canary path for selected
minimal AlphaZero loops. It copies the minimal loop script, watchdog, and
supervisor to selected nodes, stops matching old processes for the same work
directory, then starts `minimal_loop_supervisor.sh` under `nohup`.

Examples:

```bash
cd ai-service
bash scripts/deploy_minimal_loops.sh --dry-run
bash scripts/deploy_minimal_loops.sh --only square8_2p
bash scripts/deploy_minimal_loops.sh --only hex8_2p
```

Status files on the remote node:

- `~/ringrift/ai-service/<work_dir>/progress.json`
- `~/ringrift/ai-service/<work_dir>/metrics.jsonl`
- `/tmp/minimal_alphazero_<config>.log`

Important reboot caveat: this path is not boot-persistent. It uses a remote
`nohup` supervisor, not an enabled systemd unit. After a node reboot, rerun the
deploy script for that config or migrate the node to the role-aware systemd
path.

## Role-Aware Systemd Deployment

`deploy_training_service.sh` reads `config/node_roles.yaml` plus the private
runtime inventory in `config/distributed_hosts.yaml`. It installs the right
systemd unit per node role:

- `ringrift-training.service` for trainer nodes.
- `ringrift-selfplay-worker.service` for self-play workers.
- `ringrift-evaluator.service` for evaluator nodes.
- `ringrift-p2p.service` for P2P sync/health.

Examples:

```bash
cd ai-service
bash scripts/deploy_training_service.sh --dry-run
bash scripts/deploy_training_service.sh --only gh200-9 --restart
bash scripts/deploy_training_service.sh --only square8_2p --restart
```

Systemd units include `Restart=always` and `[Install] WantedBy=multi-user.target`.
They are boot-persistent when installed and enabled by the deployment flow.

## Health Checks

Use read-only checks first:

```bash
cd ai-service
PYTHONPATH=. python3 scripts/autonomy_fleet_check.py
PYTHONPATH=. python3 scripts/fleet_health_check.py
PYTHONPATH=. python3 scripts/training_status.py
```

For a specific minimal-loop node, check process and progress state over SSH:

```bash
ssh -i ~/.ssh/id_cluster ubuntu@<host> \
  "pgrep -af 'minimal_loop_supervisor|minimal_alphazero_loop' || true"

ssh -i ~/.ssh/id_cluster ubuntu@<host> \
  "tail -n 40 ~/ringrift/ai-service/<work_dir>/metrics.jsonl 2>/dev/null || true"

ssh -i ~/.ssh/id_cluster ubuntu@<host> \
  "cat ~/ringrift/ai-service/<work_dir>/progress.json 2>/dev/null || true"
```

For systemd nodes:

```bash
ssh -i ~/.ssh/id_cluster ubuntu@<host> \
  "systemctl status ringrift-training ringrift-p2p --no-pager || true"

ssh -i ~/.ssh/id_cluster ubuntu@<host> \
  "journalctl -u ringrift-training -n 100 --no-pager || true"
```

## Reboot Behavior

- Minimal-loop canary deployment is not boot-persistent. Restart it manually
  with `deploy_minimal_loops.sh` after reboot.
- Role-aware systemd deployment is boot-persistent when units are installed and
  enabled. The training, self-play, evaluator, and P2P units use
  `Restart=always`.
- `ringrift-p2p.service` refreshes code on startup before launching the P2P
  service. Check its journal after a reboot because startup may fail if the node
  cannot reach GitHub or the working tree is unhealthy.

## Safe Stop And Rollback

For minimal-loop canaries, stop only the matching config/work directory:

```bash
ssh -i ~/.ssh/id_cluster ubuntu@<host> "
  pkill -f 'scripts/[m]inimal_loop_supervisor.sh.*<work_dir>' 2>/dev/null || true
  pkill -f 'scripts/[m]inimal_alphazero_loop.py.*--work-dir <work_dir>' 2>/dev/null || true
  pkill -f 'scripts/[p]ipeline_watchdog.py.*<work_dir>' 2>/dev/null || true
"
```

For systemd nodes:

```bash
ssh -i ~/.ssh/id_cluster ubuntu@<host> \
  "sudo systemctl stop ringrift-training ringrift-selfplay-worker ringrift-evaluator 2>/dev/null || true"
```

Rollback is a code deployment operation. Prefer reverting the offending commit
or checking out the previous known-good commit on the node, then restarting only
the affected service. Do not delete model checkpoints, replay data, or work
directories during rollback unless explicitly approved.

## Known Gaps

- The full role-aware deployment cannot be reproduced from a fresh clone alone
  because `config/distributed_hosts.yaml` is private runtime inventory.
- The minimal-loop canary path is reliable for live experiments but does not
  survive node reboot.
- `docs/data/training_fleet_manifest.json` is an orientation snapshot, not a
  live source of truth for node liveness or current iteration progress.
- Current result docs should be updated only after promotion evidence is checked
  in or otherwise captured durably.
