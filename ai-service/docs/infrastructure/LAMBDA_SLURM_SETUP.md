# Lambda Slurm Setup Playbook

This playbook turns the Lambda GPU fleet into a stable Slurm cluster with a
shared filesystem and a RingRift-compatible layout.

## 0. Prerequisites

- A designated head node (recommended: `lambda-h100`).
- Shared filesystem mounted on all nodes (example: `/lambda/nfs/RingRift`).
- Consistent CUDA + driver stack across nodes.
- SSH access to all nodes.

## 1. Prepare Shared Filesystem

On every node:

```bash
sudo mkdir -p /lambda/nfs/RingRift
sudo chown -R ubuntu:ubuntu /lambda/nfs/RingRift
```

Ensure the mount is stable and identical on all nodes. If only the head node
has the mount today, add it to the other nodes before proceeding.

## 2. Install Slurm + Munge

On **all** nodes:

```bash
sudo apt-get update
sudo apt-get install -y munge slurm-wlm
```

Generate a Munge key on the head node:

```bash
sudo /usr/sbin/create-munge-key
sudo systemctl enable --now munge
sudo ls -la /etc/munge/munge.key
```

Copy the Munge key to each compute node:

```bash
scp /etc/munge/munge.key ubuntu@<node>:/tmp/munge.key
ssh ubuntu@<node> 'sudo mv /tmp/munge.key /etc/munge/munge.key && sudo chown munge:munge /etc/munge/munge.key && sudo chmod 400 /etc/munge/munge.key'
ssh ubuntu@<node> 'sudo systemctl enable --now munge'
```

## 3. Configure Slurm

Copy the provided templates:

```bash
sudo mkdir -p /etc/slurm
sudo cp /lambda/nfs/RingRift/ai-service/config/slurm/slurm.conf.lambda.example /etc/slurm/slurm.conf
sudo cp /lambda/nfs/RingRift/ai-service/config/slurm/gres.conf.lambda.example /etc/slurm/gres.conf
```

Update `/etc/slurm/slurm.conf`:

- Set `SlurmctldHost` to the head node hostname.
- Verify `NodeName`, `NodeAddr`, `CPUs`, and `RealMemory` values.
- Adjust partition membership if needed.

## 4. Start Slurm Services

Head node:

```bash
sudo systemctl enable --now slurmctld
```

Compute nodes:

```bash
sudo systemctl enable --now slurmd
```

Verify:

```bash
sinfo
scontrol show nodes
```

## 5. Place RingRift on Shared FS

On the head node:

```bash
cd /lambda/nfs/RingRift
git clone https://github.com/an0mium/RingRift.git
cd RingRift/ai-service
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Ensure the repo is visible at `/lambda/nfs/RingRift/ai-service` on **all** nodes.

## 6. Preflight + Smoke Test

On the head node:

```bash
cd /lambda/nfs/RingRift/ai-service
PYTHONPATH=. venv/bin/python scripts/slurm_preflight_check.py --config config/unified_loop.slurm.example.yaml
PYTHONPATH=. venv/bin/python scripts/slurm_smoke_test.py --config config/unified_loop.slurm.example.yaml --work-type training
```

Fix any missing Slurm binaries, partitions, or shared filesystem errors before proceeding.

## 7. Start Unified Loop with Slurm Backend

Update your config (or use the example):

```yaml
execution_backend: 'slurm'
slurm:
  enabled: true
  shared_root: '/lambda/nfs/RingRift'
  repo_subdir: 'ai-service'
  venv_activate: '/lambda/nfs/RingRift/ai-service/venv/bin/activate'
```

Then run:

```bash
cd /lambda/nfs/RingRift/ai-service
PYTHONPATH=. venv/bin/python scripts/unified_ai_loop.py --foreground --verbose
```

## Troubleshooting

- `sinfo` empty: check `slurmctld` is running and `slurm.conf` is identical across nodes.
- `node DOWN`: verify `munge` key and `slurmd` service on the node.
- `sbatch` missing: install `slurm-wlm` on the head node or add it to PATH.
- `gres` mismatch: update `/etc/slurm/gres.conf` on the affected node and restart `slurmd`.
