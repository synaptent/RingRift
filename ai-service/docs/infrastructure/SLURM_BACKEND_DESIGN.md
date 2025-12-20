# Slurm Backend Design (Draft)

## Summary

This document proposes an optional Slurm execution backend for the RingRift AI
training loop. The goal is to support users with stable Slurm-managed HPC
clusters without changing the default P2P/SSH workflows.

The design targets a minimal, high-ROI integration:

- Add a Slurm backend to the existing `OrchestratorBackend` abstraction.
- Use shared filesystem paths for datasets, models, and logs.
- Keep P2P/SSH as the default for elastic cloud clusters.

## Goals

- Provide first-class support for Slurm HPC clusters.
- Enable training, selfplay, and tournaments to run via `sbatch`.
- Reuse existing scripts (`run_self_play_soak.py`, `run_nn_training_baseline.py`,
  `run_unified_tournament.py`) without rewriting core logic.
- Prefer shared filesystem paths over rsync where possible.

## Non-Goals

- Replace P2P orchestration for elastic cloud fleets.
- Implement cluster provisioning or Slurm installation automation.
- Support non-shared-filesystem deployments in the initial version.

## When Slurm Adds Value (and When It Does Not)

Slurm is highest ROI when the cluster is stable and shared-state is reliable:

- Stable node pool with predictable availability (no frequent churn).
- Shared filesystem mounted at the same path on every node.
- Clear partitions for training vs selfplay vs eval.
- Need for fair scheduling, quota enforcement, and job accounting.

P2P/SSH remains the better default when:

- Nodes are elastic or ephemeral (spot/preemptible fleets).
- You cannot rely on a shared filesystem.
- Latency and multi-provider networking are primary constraints.

## Proposed Architecture

### Integration Point

Extend `ai-service/app/execution/backends.py`:

- Add `BackendType.SLURM`.
- Add `SlurmBackend(OrchestratorBackend)` implementing:
  - `run_selfplay(...)`
  - `run_training(...)`
  - `run_tournament(...)`
  - `get_available_workers(...)` (optional: uses `sinfo`/`squeue`)
  - `sync_models(...)` and `sync_data(...)` as no-ops when shared FS is present.

This keeps the unified loop API stable and allows the rest of the codebase to
select Slurm execution via config.

### Job Submission Flow

1. Build a job spec for the task (resources, time, partition).
2. Write a small wrapper script to a shared location (example:
   `data/slurm/jobs/<job_id>.sh`).
3. Submit with `sbatch --parsable` and capture the job ID.
4. Track completion via:
   - `squeue` for running/pending state.
   - `sacct` for exit code and final state.
5. Gather logs from `data/slurm/logs/<job_id>.out|err`.

### Resource Mapping by Work Type

Default resource expectations, with per-task overrides:

| Work Type  | GPUs | CPUs | RAM  | Walltime | Notes                      |
| ---------- | ---- | ---- | ---- | -------- | -------------------------- |
| training   | 1-4  | 8-32 | 64G+ | 4-12h    | Prefers GPU partitions     |
| selfplay   | 0-1  | 4-16 | 16G  | 1-4h     | Use job arrays for scale   |
| tournament | 0-1  | 4-8  | 16G  | 1-2h     | CPU OK unless GPU required |

### Slurm Command Usage

- Submit: `sbatch --parsable <script>`
- Status: `squeue -j <job_id> -h -o "%T"`
- Accounting: `sacct -j <job_id> --format=State,ExitCode -n`
- Cancel: `scancel <job_id>`

### Shared Filesystem Assumptions

Assume a shared filesystem mounted on all nodes:

- `data/games`, `data/training`, `models`, `logs`, `runs`
- No rsync required for shared paths.
- Local scratch can be used for temporary files, but outputs should sync to the
  shared root.

### Configuration (Proposed)

Add a `slurm` config block to `config/unified_loop.yaml` and a `SlurmConfig`
dataclass in `app/config/unified_config.py`:

```yaml
execution_backend: 'slurm' # or 'auto' with slurm.enabled: true
slurm:
  enabled: false
  partition_training: 'gpu-train'
  partition_selfplay: 'gpu-selfplay'
  partition_tournament: 'cpu-eval'
  account: 'ringrift'
  qos: 'normal'
  default_time_training: '08:00:00'
  default_time_selfplay: '02:00:00'
  default_time_tournament: '02:00:00'
  gpus_training: 1
  cpus_training: 16
  mem_training: '64G'
  gpus_selfplay: 0
  cpus_selfplay: 8
  mem_selfplay: '16G'
  job_dir: 'data/slurm/jobs'
  log_dir: 'data/slurm/logs'
  shared_root: '/shared/ringrift'
  container_runtime: null # optional: apptainer|singularity|docker
  container_image_x86_64: null
  container_image_arm64: null
```

Example config:

- `config/unified_loop.slurm.example.yaml`
- `config/slurm/slurm.conf.lambda.example`
- `config/slurm/gres.conf.lambda.example`

Preflight and smoke test helpers:

- `scripts/slurm_preflight_check.py`
- `scripts/slurm_smoke_test.py`

Lambda cluster playbook:

- `docs/infrastructure/LAMBDA_SLURM_SETUP.md`

### Job Wrapper Template (Draft)

Each Slurm job runs a minimal wrapper that:

- Activates the venv or module.
- Changes to the repo root on the shared filesystem.
- Runs the canonical script with explicit args.

Example:

```bash
#!/usr/bin/env bash
set -euo pipefail
source /shared/ringrift/ai-service/venv/bin/activate
cd /shared/ringrift/ai-service
python scripts/run_nn_training_baseline.py \
  --run-dir runs/sq8_2p_12345 \
  --data-path data/training/canonical_square8.square8.2p.hl3.fv2.npz \
  --board square8 \
  --num-players 2 \
  --epochs 100
```

### Failure Handling

- Treat `FAILED`, `CANCELLED`, or non-zero `ExitCode` as job failure.
- Retry logic should remain in the unified loop, not inside Slurm.
- Timeouts should map to Slurm `--time` and/or local monitoring with `scancel`.

## Lambda Nodes: Transition to a Stable HPC Cluster

Lambda nodes with a shared filesystem are not an HPC cluster by default. They
become HPC-like once you add a scheduler and enforce stable allocations.

Classification:

- **Not a stable HPC cluster** if nodes are ad-hoc, lack Slurm, or the shared
  filesystem is only best-effort.
- **Yes, HPC-like** once there is a Slurm controller, consistent hostnames,
  shared filesystem, and enforced partitions/allocations.

### Required Components

1. **Shared filesystem**: NFS/FSx/Lustre mounted on all nodes.
2. **Slurm controller**: One stable head node (slurmctld).
3. **Slurm compute nodes**: slurmd on every Lambda node.
4. **GPU resource config**: `gres.conf` per node.
5. **Accounting (optional but recommended)**: slurmdbd for usage tracking.

### Steps to Transition

1. Standardize OS, CUDA, and driver versions across nodes.
2. Mount the shared filesystem to the same path on every node.
3. Deploy Slurm controller on the primary Lambda node.
4. Add all Lambda nodes to `slurm.conf` with stable hostnames.
5. Create partitions for:
   - training (GPU-heavy)
   - selfplay (GPU/CPU mixed)
   - evaluation (CPU)
6. Configure `gres.conf` for GPU visibility and enforce cgroups.
7. Update RingRift config to use Slurm backend and shared root paths.
8. Validate with:
   - `sinfo`, `squeue`, `sacct`
   - A small training job submission

### Recommended Partition Strategy

- `gpu-train`: x86_64 H100-class nodes (longer walltime).
- `gpu-selfplay`: mid-tier GPUs (A10/H100), short walltime, job arrays.
- `cpu-eval`: CPU-rich x86 nodes, tournaments and data merges.
- `gpu-gh200` (optional): ARM64 GH200 nodes kept separate until the runtime
  environment is validated for aarch64.

### Mixed-Architecture Caveat (GH200)

GH200 nodes are aarch64. The shared NFS venv built on x86_64 nodes cannot run on
ARM. To include GH200 in standard partitions:

Decision: prefer a containerized aarch64 runtime for GH200, with a source-build
fallback when containers are unavailable.

Recommended path (container-first):

1. Build or pull a CUDA aarch64 image that includes PyTorch 2.6.0.
2. Configure `slurm.container_runtime` and `slurm.container_image_arm64`.
3. Update the Slurm backend to launch jobs via `srun --container-image` (or
   Apptainer/Singularity) when `uname -m == aarch64`.
4. Move GH200 nodes back into `gpu-train` and `gpu-selfplay` after parity and
   training smoke tests pass.

Fallback (source-build venv):

1. Build PyTorch 2.6.0 + torchvision 0.21.0 from source on a GH200 node into
   aarch64 venvs on each GH200 node (for example,
   `/home/ubuntu/venv-arm64-local`) or a stable shared filesystem if available.
   - If CMake 4.x errors on protobuf, set `CMAKE_POLICY_VERSION_MINIMUM=3.5`
     during the build or use the system CMake (>= 3.18).
2. Set `slurm.venv_activate_arm64` to that path (local or shared) so Slurm jobs
   auto-select the correct venv based on `uname -m`.
3. Move GH200 nodes back into `gpu-train` and `gpu-selfplay` once validated.

Until the aarch64 venv is ready, keep GH200 nodes in a separate `gpu-gh200`
partition and avoid scheduling training/selfplay there.

## Adoption Plan (Phased)

1. **Phase 1**: Training only via Slurm backend.
2. **Phase 2**: Add tournaments via Slurm.
3. **Phase 3**: Selfplay via Slurm job arrays.
4. **Phase 4**: Optional model sync and monitoring integration.

## Open Questions

- Partition names, account, and QoS defaults for public docs.
- Shared filesystem path conventions for different providers.
- Whether to support multi-node `torchrun` in the first version.
- Preferred log retention and job history policy.
