# Training Pipeline Status Report

> **Date:** 2025-12-13
> **Purpose:** Document comprehensive pipeline orchestrator improvements and production readiness

---

## Executive Summary

Major improvements to the pipeline orchestrator making it production-ready for continuous AI improvement:

- **Robustness:** SSH retry with exponential backoff, smart polling replaces fixed waits
- **Observability:** Elo tracking, model registry, resource monitoring, enhanced error logging
- **Automation:** Tier gating integration, game deduplication, checkpointing with resume
- **Scalability:** 22 diverse selfplay jobs, CMA-ES for all 9 board×player configs

**All instances synced:** mac-studio, mbp-16gb, mbp-64gb, aws-staging, lambda-a10, lambda-h100, vast-3090, vast-5090-dual, vast-5090-quad

---

## New Features Implemented

### 1. SSH Retry with Exponential Backoff

Transient network failures no longer abort entire iterations:

```python
SSH_MAX_RETRIES = 3
SSH_BASE_DELAY = 2.0   # seconds
SSH_MAX_DELAY = 30.0   # seconds
SSH_BACKOFF_FACTOR = 2.0
```

Retry sequence: 2s → 4s → 8s (with jitter)

### 2. Smart Polling

Replaced fixed 45-minute waits with intelligent completion detection:

| Phase    | Detection Method                 | Default Timeout |
| -------- | -------------------------------- | --------------- |
| Selfplay | Poll game counts until 50+ games | 30 min          |
| CMA-ES   | Poll for process completion      | 120 min         |
| Training | Poll for process completion      | 60 min          |

### 3. Elo Rating System

Standard Elo formula (K=32) for model comparison:

- Ratings initialized at 1500
- Full history persisted for trend analysis
- Leaderboard printed after each iteration

### 4. Model Registry

Track all trained models with lineage:

```json
{
  "model_id": "square8_2p_iter5",
  "path": "models/square8_2p_iter5.pth",
  "config": "square8_2p",
  "parent_id": "square8_2p_iter4",
  "created_at": "2025-12-13T10:30:00",
  "status": "active"
}
```

### 5. Tier Gating Integration

Automatic D2→D4→D6→D8 promotion:

- 55% win rate threshold
- Minimum 10 matches required
- Per-config tracking in state file

### 6. Game Deduplication

Prevents duplicate games in training data:

- SHA256 hash of moves + board + players + outcome
- 16-char hash prefix for efficiency
- Seen hashes persisted across iterations

### 7. Resource Monitoring

Track CPU/MEM/DISK/GPU across all workers:

```
=== Resource Usage ===
  mac-studio: CPU=45%, MEM=67%, DISK=34%, GPU=0%
  lambda-h100: CPU=12%, MEM=23%, DISK=15%, GPU=89%
```

### 8. Enhanced Error Logging

Full stdout/stderr captured to daily log files:

```
logs/pipeline/errors_20251213.log
```

### 9. Checkpointing and Resume

Interrupted iterations can be resumed:

```bash
python scripts/pipeline_orchestrator.py --iterations 10 --resume
```

Completed phases are skipped automatically.

---

## Board-Specific Heuristic Profiles

Implemented full 9-config matrix for CMA-ES optimization:

| Board     | Players | Profile Key            |
| --------- | ------- | ---------------------- |
| square8   | 2       | `heuristic_v1_sq8_2p`  |
| square8   | 3       | `heuristic_v1_sq8_3p`  |
| square8   | 4       | `heuristic_v1_sq8_4p`  |
| square19  | 2       | `heuristic_v1_sq19_2p` |
| square19  | 3       | `heuristic_v1_sq19_3p` |
| square19  | 4       | `heuristic_v1_sq19_4p` |
| hexagonal | 2       | `heuristic_v1_hex_2p`  |
| hexagonal | 3       | `heuristic_v1_hex_3p`  |
| hexagonal | 4       | `heuristic_v1_hex_4p`  |

---

## Selfplay Job Matrix

22 diverse configurations for comprehensive training data:

| Config               | Board   | Players | Engine Mode    | Games |
| -------------------- | ------- | ------- | -------------- | ----- |
| square8_2p_mixed     | square8 | 2       | mixed          | 40    |
| square8_2p_heuristic | square8 | 2       | heuristic-only | 20    |
| square8_2p_minimax   | square8 | 2       | minimax-only   | 15    |
| square8_2p_mcts      | square8 | 2       | mcts-only      | 15    |
| square8_2p_descent   | square8 | 2       | descent-only   | 20    |
| square8_2p_nn        | square8 | 2       | nn-only        | 10    |
| square8_3p_mixed     | square8 | 3       | mixed          | 20    |
| ...                  | ...     | ...     | ...            | ...   |

All engine modes benefit from trained heuristic profiles via `--heuristic-weights-file`.

---

## Tournament Matchups

8 tournament matchups per iteration with game recording:

| Matchup                      | Board    | Games |
| ---------------------------- | -------- | ----- |
| Heuristic D5 vs MCTS D6      | Square8  | 10    |
| Heuristic D5 vs Minimax D5   | Square8  | 10    |
| MCTS D6 vs Minimax D5        | Square8  | 10    |
| Heuristic D3 vs Heuristic D5 | Square8  | 8     |
| MCTS D5 vs MCTS D7           | Square8  | 8     |
| Heuristic D5 vs MCTS D5      | Square19 | 6     |
| Heuristic D5 vs MCTS D5      | Hex      | 6     |
| MCTS D7 vs MCTS D8           | Square8  | 6     |

Tournament games saved to `logs/tournaments/` and synced for training.

---

## Cluster Status

### Synced Instances (2025-12-13)

| Instance       | Role                | Status    |
| -------------- | ------------------- | --------- |
| mac-studio     | nn_training_mps     | ✅ Synced |
| mbp-16gb       | selfplay_light      | ✅ Synced |
| mbp-64gb       | selfplay            | ✅ Synced |
| aws-staging    | selfplay_cmaes      | ✅ Synced |
| lambda-a10     | nn_training         | ✅ Synced |
| lambda-h100    | nn_training_primary | ✅ Synced |
| vast-3090      | nn_training         | ✅ Synced |
| vast-5090-dual | nn_training         | ✅ Synced |
| vast-5090-quad | nn_training_primary | ✅ Synced |

---

## CLI Quick Reference

```bash
# Run full iteration
python scripts/pipeline_orchestrator.py --iterations 1

# Resume interrupted run
python scripts/pipeline_orchestrator.py --iterations 10 --resume

# Run specific phase
python scripts/pipeline_orchestrator.py --phase selfplay
python scripts/pipeline_orchestrator.py --phase tier-gating
python scripts/pipeline_orchestrator.py --phase resources

# Dry run
python scripts/pipeline_orchestrator.py --dry-run
```

---

## State File Location

```
logs/pipeline/state.json
```

Contains: iteration, phase_completed, games_generated, elo_ratings, elo_history,
model_registry, tier_promotions, seen_game_hashes, errors.

---

## Documentation Updated

| Document                | Location                                                 |
| ----------------------- | -------------------------------------------------------- |
| Pipeline Orchestrator   | `ai-service/docs/PIPELINE_ORCHESTRATOR.md`               |
| AI Improvement Progress | `docs/ai/AI_IMPROVEMENT_PROGRESS.md`                     |
| This Status Report      | `ai-service/docs/TRAINING_PIPELINE_STATUS_2025_12_13.md` |

---

## Next Steps

1. **Run continuous improvement loop:** `--iterations 10` with resume
2. **Monitor Elo leaderboard** for model performance trends
3. **Review tier gating** for automatic promotions
4. **Scale selfplay** on GPU instances during off-peak hours
5. **Generate canonical hex data** for hexagonal board support
