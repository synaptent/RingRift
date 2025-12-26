# Orchestrator Selection Guide

## Quick Reference

| Use Case              | Script                                 | When to Use                                          |
| --------------------- | -------------------------------------- | ---------------------------------------------------- |
| **Full AI Loop**      | `unified_ai_loop.py`                   | Production deployment, continuous improvement        |
| **P2P Cluster**       | `p2p_orchestrator.py`                  | Multi-node P2P coordination, distributed selfplay    |
| **Slurm HPC**         | `unified_ai_loop.py`                   | Stable Slurm cluster with shared filesystem          |
| **Sync Operations**   | `app/distributed/sync_orchestrator.py` | Unified entry point for data/model/elo/registry sync |
| **Multi-Board Train** | `unified_ai_loop.py`                   | Multi-board training via unified loop config         |
| **Elo Tournament**    | `auto_elo_tournament.py`               | Automated Elo evaluation with Slack alerts           |
| **Model Promotion**   | `model_promotion_manager.py`           | Manual promotion, Elo testing, rollback              |

---

## Script Details

### unified_ai_loop.py (Canonical - Use This)

**Purpose**: Complete AI self-improvement feedback loop

**Features**:

- Selfplay generation with curriculum learning
- Automatic data sync to training pool
- Training trigger on data thresholds
- Model evaluation via tournaments
- Promotion with regression testing
- Health monitoring with Prometheus metrics
- Emergency halt mechanism

**CLI**:

```bash
# Start in background
python scripts/unified_ai_loop.py --start

# Run in foreground with verbose output
python scripts/unified_ai_loop.py --foreground --verbose

# Check status
python scripts/unified_ai_loop.py --status

# Stop gracefully
python scripts/unified_ai_loop.py --stop

# Emergency halt (stops at next health check)
python scripts/unified_ai_loop.py --halt

# Resume after halt
python scripts/unified_ai_loop.py --resume
```

**Config**: `config/unified_loop.yaml`

---

### p2p_orchestrator.py

**Purpose**: Distributed P2P cluster coordination and selfplay orchestration

**Features**:

- Self-healing compute cluster with leader election
- Peer discovery and resource monitoring
- Auto-starts selfplay/training jobs across nodes
- Vast.ai and Lambda Labs instance integration
- Supports all board types: square8, hex8, square19, hexagonal
- Keepalive and unretire management for cloud instances

**CLI**:

```bash
# Start as node in P2P cluster (replace with your node ID and coordinator IP)
python scripts/p2p_orchestrator.py --node-id gpu-node-1 --peers COORDINATOR_IP:8770

# View cluster status
curl http://localhost:8770/status
```

**When to use**:

- Production distributed training across 3+ nodes
- Vast.ai or Lambda Labs GPU instances
- Self-healing cluster with automatic recovery

---

### Slurm Backend (Optional)

**Purpose**: Run the unified AI loop on a stable Slurm-managed HPC cluster.

**When to use**:

- You have a shared filesystem mounted on all nodes.
- You want queue-based scheduling, accounting, and fair-share.

**Notes**:

- Use `unified_ai_loop.py` with `execution_backend: "slurm"` (or `auto` + `slurm.enabled: true`).
- See `docs/infrastructure/SLURM_BACKEND_DESIGN.md` for details.

---

### Sync Orchestrator (Module)

**Purpose**: Unified entry point for data, model, Elo, and registry sync operations.

**When to use**:

- You need one place to coordinate sync scheduling across components.
- You want quality-driven sync triggers wired to the event bus.

**Usage**:

```python
from app.distributed.sync_orchestrator import get_sync_orchestrator

orchestrator = get_sync_orchestrator()
await orchestrator.initialize()
result = await orchestrator.sync_all()
```

---

### ~~pipeline_orchestrator.py~~ (DEPRECATED)

> ⚠️ **Deprecated**: This script has been removed. Use `unified_ai_loop.py` instead.
> **Archive location**: Removed (see git history)

**Original Purpose**: CI/CD pipeline for automated testing and validation

The functionality has been integrated into `unified_ai_loop.py` with:

- Regression detection via the unified loop (see `run_strength_regression_gate.py` for a standalone gate)
- Automated model quality validation
- Shadow tournament validation every 5 minutes

**Migration**: Replace `pipeline_orchestrator.py` calls with `unified_ai_loop.py`:

```bash
# Old (deprecated)
# python scripts/pipeline_orchestrator.py --pr 123 --validate

# New (use unified loop with validation)
python scripts/unified_ai_loop.py --foreground --verbose
```

---

### model_promotion_manager.py

**Purpose**: Direct model promotion with Elo validation

**Features**:

- Elo threshold validation
- Statistical significance testing
- Cluster sync via SSH
- Rollback on regression
- Daemon mode for continuous promotion

**CLI**:

```bash
# Single promotion check
python scripts/model_promotion_manager.py --candidate new_model.pt

# Continuous daemon mode
python scripts/model_promotion_manager.py --daemon

# Rollback
python scripts/model_promotion_manager.py --rollback
```

**When to use instead of unified_ai_loop.py**:

- Manual promotion control
- Testing specific models
- Debugging promotion issues

---

## Deprecated Scripts (Do Not Use)

| Script                             | Replacement                                 |
| ---------------------------------- | ------------------------------------------- |
| `continuous_improvement_daemon.py` | `unified_ai_loop.py`                        |
| `improvement_cycle_manager.py`     | `unified_ai_loop.py`                        |
| `auto_promote_best_models.py`      | Archived (use `model_promotion_manager.py`) |
| `auto_promote_weights.py`          | Archived (use `model_promotion_manager.py`) |

---

## Decision Tree

```
Do you need continuous AI improvement?
├─ Yes → unified_ai_loop.py
│        ├─ Have a stable Slurm cluster? → Use Slurm backend
│        └─ Need distributed across 3+ nodes? → Also use p2p_orchestrator.py
│
├─ No, just need model promotion
│  └─ model_promotion_manager.py
│
├─ No, need CI/CD validation
│  └─ unified_ai_loop.py (with regression gate)
│
├─ No, need multi-board/multi-player training
│  └─ unified_ai_loop.py (multi-board config)
│
└─ No, need distributed P2P selfplay
   └─ p2p_orchestrator.py
```

---

## Configuration Files

| Script                       | Config File                                     |
| ---------------------------- | ----------------------------------------------- |
| `unified_ai_loop.py`         | `config/unified_loop.yaml`                      |
| `p2p_orchestrator.py`        | `config/unified_loop.yaml` (p2p section)        |
| `model_promotion_manager.py` | Uses CLI args or `config/promotion_daemon.yaml` |
| `auto_elo_tournament.py`     | `config/unified_loop.yaml`                      |
