# Pipeline Stage Failures Runbook

This runbook covers diagnosis and resolution of failures in the RingRift training pipeline stages: export, train, evaluate, and promote.

**Created**: December 28, 2025
**Version**: 1.0
**Severity**: High

---

## Overview

The training pipeline has 4 main stages:

```
Selfplay → Export → Train → Evaluate → Promote
```

Each stage can fail independently, blocking downstream stages. This runbook covers diagnosis and recovery for each.

---

## Pipeline Stage Detection

### Check Pipeline Status

```python
from app.coordination.data_pipeline_orchestrator import get_pipeline_orchestrator

orchestrator = get_pipeline_orchestrator()

# Get status for all configs
for config, status in orchestrator.get_all_stage_status().items():
    print(f"\n=== {config} ===")
    for stage, info in status.items():
        state = "✓" if info.get("completed") else "✗"
        print(f"  {state} {stage}: {info.get('status', 'unknown')}")
        if info.get("error"):
            print(f"      Error: {info['error']}")
```

### Check Stage Events

```bash
# Look for stage completion/failure events
grep -E "(EXPORT|TRAIN|EVALUATE|PROMOTE)_(COMPLETED|FAILED)" logs/coordination.log | tail -30
```

---

## Stage 1: Export Failures

### Detection

```bash
# Check export status
python -c "
from app.coordination.data_pipeline_orchestrator import get_pipeline_orchestrator

orch = get_pipeline_orchestrator()
status = orch.get_stage_status('hex8_2p', 'export')
print(f'Status: {status}')
"
```

### Common Causes

| Symptom                    | Cause                  | Fix                          |
| -------------------------- | ---------------------- | ---------------------------- |
| "No games found"           | Empty/missing database | Run selfplay first           |
| "Parity validation failed" | Move replay mismatch   | See PARITY_MISMATCH_DEBUG.md |
| "Permission denied"        | NPZ write permission   | `chmod 755 data/training`    |
| "Out of memory"            | Large dataset          | Use `--max-samples` flag     |

### Recovery

```bash
# Manual export with debugging
python scripts/export_replay_dataset.py \
  --db data/games/canonical_hex8.db \
  --board-type hex8 --num-players 2 \
  --output data/training/hex8_2p.npz \
  --verbose \
  --max-samples 100000

# Skip parity validation if needed (temporary)
export RINGRIFT_SKIP_PARITY=1
python scripts/export_replay_dataset.py ...
```

### Trigger Re-export

```python
from app.coordination.data_pipeline_orchestrator import get_pipeline_orchestrator

orch = get_pipeline_orchestrator()
await orch.trigger_stage("hex8_2p", "export", force=True)
```

---

## Stage 2: Training Failures

### Detection

```bash
# Check for training errors
grep -i "training\|train" logs/training.log | grep -i "error\|failed" | tail -20

# Check GPU status
nvidia-smi
```

### Common Causes

| Symptom                  | Cause                   | Fix                      |
| ------------------------ | ----------------------- | ------------------------ |
| "CUDA out of memory"     | Batch size too large    | Reduce `--batch-size`    |
| "No training data"       | NPZ file missing        | Run export first         |
| "Loss is NaN"            | Learning rate too high  | Reduce `--learning-rate` |
| "Early stopping"         | Validation loss plateau | May need more data       |
| "Checkpoint load failed" | Corrupted checkpoint    | Remove and retrain       |

### Recovery

```bash
# Resume from last checkpoint
python -m app.training.train \
  --board-type hex8 --num-players 2 \
  --data-path data/training/hex8_2p.npz \
  --resume-from models/checkpoints/hex8_2p_epoch_10.pth

# Start fresh with smaller batch size
python -m app.training.train \
  --board-type hex8 --num-players 2 \
  --data-path data/training/hex8_2p.npz \
  --batch-size 256 \
  --epochs 50
```

### Clear Training Lock

If training appears stuck:

```python
from app.coordination.locking_integration import release_training_lock

# Force release training lock for config
release_training_lock("hex8_2p", force=True)
```

---

## Stage 3: Evaluation Failures

### Detection

```python
from app.coordination.evaluation_daemon import get_evaluation_daemon

daemon = get_evaluation_daemon()
status = daemon.get_evaluation_status("hex8_2p")
print(f"Last evaluation: {status}")
```

### Common Causes

| Symptom              | Cause                        | Fix                 |
| -------------------- | ---------------------------- | ------------------- |
| "Model not found"    | Model path wrong             | Check symlinks      |
| "Opponent not found" | Missing heuristic AI         | Check imports       |
| "Game timeout"       | Slow inference               | Reduce simulations  |
| "All games crashed"  | Model produces invalid moves | Check model version |

### Recovery

```bash
# Manual gauntlet evaluation
python scripts/quick_gauntlet.py \
  --model models/canonical_hex8_2p.pth \
  --board-type hex8 --num-players 2 \
  --games 50 \
  --verbose

# With specific opponents
python scripts/quick_gauntlet.py \
  --model models/canonical_hex8_2p.pth \
  --board-type hex8 --num-players 2 \
  --opponents random,heuristic
```

### Trigger Re-evaluation

```python
from app.coordination.evaluation_daemon import get_evaluation_daemon

daemon = get_evaluation_daemon()
await daemon.queue_evaluation(
    model_path="models/canonical_hex8_2p.pth",
    config_key="hex8_2p",
    priority="high"
)
```

---

## Stage 4: Promotion Failures

### Detection

```bash
# Check promotion events
grep "PROMOTION" logs/coordination.log | tail -10
```

### Common Causes

| Symptom               | Cause             | Fix                                       |
| --------------------- | ----------------- | ----------------------------------------- |
| "Below threshold"     | Win rate too low  | Train more or adjust threshold            |
| "Symlink failed"      | Permission issues | `chmod 755 models/`                       |
| "Distribution failed" | Network issues    | See MODEL_DISTRIBUTION_TROUBLESHOOTING.md |

### Recovery

```bash
# Manual promotion (bypasses thresholds)
python scripts/auto_promote.py \
  --model models/my_model.pth \
  --board-type hex8 --num-players 2 \
  --force-promote

# Create symlinks manually
ln -sf canonical_hex8_2p.pth models/ringrift_best_hex8_2p.pth
```

### Force Promotion Event

```python
from app.coordination.event_emitters import emit_model_promoted

emit_model_promoted(
    config_key="hex8_2p",
    model_path="models/canonical_hex8_2p.pth",
    source="manual_promotion"
)
```

---

## Circuit Breaker Status

The pipeline uses circuit breakers to prevent repeated failures:

```python
from app.coordination.pipeline_actions import get_circuit_breaker_status

status = get_circuit_breaker_status()
for action, info in status.items():
    state = "OPEN" if info["open"] else "CLOSED"
    print(f"{action}: {state} (failures: {info['failures']})")
```

### Reset Circuit Breaker

```python
from app.coordination.pipeline_actions import reset_circuit_breaker

# Reset specific action
reset_circuit_breaker("export_action")

# Reset all
reset_circuit_breaker(None)
```

---

## Full Pipeline Reset

For severe issues, reset the entire pipeline state:

```python
from app.coordination.data_pipeline_orchestrator import get_pipeline_orchestrator

orch = get_pipeline_orchestrator()

# Reset specific config
await orch.reset_pipeline_state("hex8_2p")

# Reset all configs
await orch.reset_all_pipeline_states()

# Trigger fresh run
await orch.start_pipeline_for_config("hex8_2p")
```

---

## Monitoring

### Key Metrics

| Metric                | Alert Threshold | Source                   |
| --------------------- | --------------- | ------------------------ |
| stage_failures        | > 3 per hour    | DataPipelineOrchestrator |
| circuit_breaker_opens | > 1 per day     | PipelineActions          |
| pipeline_stall_time   | > 2 hours       | DataPipelineOrchestrator |

### Health Check

```python
from app.coordination.data_pipeline_orchestrator import get_pipeline_orchestrator

orch = get_pipeline_orchestrator()
health = orch.health_check()

print(f"Healthy: {health.is_healthy}")
print(f"Active pipelines: {health.details.get('active_count', 0)}")
print(f"Stalled pipelines: {health.details.get('stalled_count', 0)}")
```

---

## Related Documentation

- [TRAINING_LOOP_STALLED.md](TRAINING_LOOP_STALLED.md) - Training loop issues
- [PARITY_MISMATCH_DEBUG.md](PARITY_MISMATCH_DEBUG.md) - Parity issues
- [MODEL_PROMOTION_WORKFLOW.md](MODEL_PROMOTION_WORKFLOW.md) - Promotion flow
