# Orchestrator Selection Guide

## Quick Reference

| Use Case               | Script                       | When to Use                                    |
| ---------------------- | ---------------------------- | ---------------------------------------------- |
| **Full AI Loop**       | `unified_ai_loop.py`         | Production deployment, continuous improvement  |
| **Cluster Management** | `cluster_orchestrator.py`    | Multi-node coordination, distributed selfplay  |
| **P2P Matchmaking**    | `p2p_orchestrator.py`        | Development, human-vs-AI games, casual testing |
| **CI/CD Pipeline**     | `pipeline_orchestrator.py`   | Automated testing, PR validation               |
| **Model Promotion**    | `model_promotion_manager.py` | Manual promotion, Elo testing, rollback        |

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

### cluster_orchestrator.py

**Purpose**: Coordinate distributed selfplay across multiple nodes

**Features**:

- SSH-based node coordination
- Distributed locking for consistency
- Work distribution across cluster
- Model sync between nodes
- Version skew healing

**CLI**:

```bash
python scripts/cluster_orchestrator.py --config config/cluster.yaml
```

**When to use instead of unified_ai_loop.py**:

- Managing 3+ remote nodes
- Need fine-grained control over distributed work
- unified_ai_loop.py handles single-node or simple multi-node via SSH sync

---

### p2p_orchestrator.py

**Purpose**: P2P matchmaking server for human-AI and AI-AI games

**Features**:

- WebSocket-based real-time matchmaking
- Multiple AI difficulty levels
- Human vs AI games
- Rating tracking

**CLI**:

```bash
python scripts/p2p_orchestrator.py --port 8080
```

**When to use**:

- Development testing
- Human vs AI matches
- Not for continuous training loops

---

### pipeline_orchestrator.py

**Purpose**: CI/CD pipeline for automated testing and validation

**Features**:

- PR validation gates
- Regression testing
- Model quality checks
- Automated tournament validation

**CLI**:

```bash
python scripts/pipeline_orchestrator.py --pr 123 --validate
```

**When to use**:

- GitHub Actions CI/CD
- PR merge gates
- Automated quality validation

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
│        └─ Need distributed across 3+ nodes? → Also use cluster_orchestrator.py
│
├─ No, just need model promotion
│  └─ model_promotion_manager.py
│
├─ No, need CI/CD validation
│  └─ pipeline_orchestrator.py
│
└─ No, need human vs AI games
   └─ p2p_orchestrator.py
```

---

## Configuration Files

| Script                       | Config File                              |
| ---------------------------- | ---------------------------------------- |
| `unified_ai_loop.py`         | `config/unified_loop.yaml`               |
| `cluster_orchestrator.py`    | `config/cluster.yaml`                    |
| `p2p_orchestrator.py`        | `config/p2p.yaml`                        |
| `pipeline_orchestrator.py`   | `config/pipeline.yaml`                   |
| `model_promotion_manager.py` | Uses CLI args or `config/promotion.yaml` |
