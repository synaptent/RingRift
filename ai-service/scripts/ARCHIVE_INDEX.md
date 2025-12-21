# Archive Index - December 2025

This document catalogs all archived scripts in `scripts/archive/` with their purpose,
archival date, and status.

## Status Legend

- **HISTORICAL**: Kept for reference, may still provide useful patterns
- **DEPRECATED**: Superseded by newer implementation, safe to delete after review
- **DEBUG**: One-off debugging scripts, safe to delete

---

## archive/debug/ (24 files)

Debug scripts created during development to investigate specific issues.
All are safe to delete as issues have been resolved.

| Script                          | Purpose                        | Status |
| ------------------------------- | ------------------------------ | ------ |
| `debug_after_capture.py`        | Investigate capture phase bugs | DEBUG  |
| `debug_capture_chain.py`        | Chain capture debugging        | DEBUG  |
| `debug_capture_detection.py`    | Capture detection logic        | DEBUG  |
| `debug_capture_phase.py`        | Capture phase transitions      | DEBUG  |
| `debug_capture_source.py`       | Capture source tracking        | DEBUG  |
| `debug_chain_after_48.py`       | Move 48 chain issue            | DEBUG  |
| `debug_chain_capture.py`        | Chain capture mechanics        | DEBUG  |
| `debug_chain_capture_detail.py` | Detailed chain analysis        | DEBUG  |
| `debug_chain_detection.py`      | Chain detection algorithm      | DEBUG  |
| `debug_chain_divergence.py`     | GPU/CPU chain divergence       | DEBUG  |
| `debug_chain_state.py`          | Chain state tracking           | DEBUG  |
| `debug_export_moves.py`         | Move export debugging          | DEBUG  |
| `debug_game.py`                 | General game debugging         | DEBUG  |
| `debug_gpu_cpu_parity.py`       | GPU/CPU parity issues          | DEBUG  |
| `debug_gpu_markers.py`          | GPU marker debugging           | DEBUG  |
| `debug_hex8_hang.py`            | Hex8 infinite loop issue       | DEBUG  |
| `debug_hex8_replay.py`          | Hex8 replay validation         | DEBUG  |
| `debug_move_48.py`              | Move 48 specific issue         | DEBUG  |
| `debug_move_48_detailed.py`     | Move 48 detailed trace         | DEBUG  |
| `debug_recovery_45.py`          | Recovery at move 45            | DEBUG  |
| `debug_recovery_moves.py`       | Recovery move generation       | DEBUG  |
| `debug_specific_move.py`        | Specific move debugging        | DEBUG  |
| `debug_ts_python_state_diff.py` | TypeScript/Python state diff   | DEBUG  |
| `minimal_hex8_debug.py`         | Minimal hex8 reproduction      | DEBUG  |

---

## archive/deprecated/ (14 files)

Scripts explicitly superseded by newer implementations.
Review before deletion to ensure functionality migrated.

| Script                                    | Purpose                   | Replacement                                    | Status     |
| ----------------------------------------- | ------------------------- | ---------------------------------------------- | ---------- |
| `alerting.py`                             | Monitor alerting          | `node_health_orchestrator.py`                  | DEPRECATED |
| `auto_elo_tournament.py`                  | Auto Elo tournament       | `run_model_elo_tournament.py`                  | DEPRECATED |
| `auto_model_promotion.py`                 | Auto-promote models       | `unified_orchestrator.py` promotion gates      | DEPRECATED |
| `auto_promote.py`                         | Simple auto-promotion     | `elo_promotion_gate.py`                        | DEPRECATED |
| `baseline_gauntlet.py`                    | Baseline gauntlet         | `run_gauntlet.py`                              | DEPRECATED |
| `elo_promotion_gate.py`                   | Elo promotion gate        | `unified_orchestrator.py`                      | DEPRECATED |
| `model_sync_aria2.py`                     | Model sync via aria2      | `sync_models.py`                               | DEPRECATED |
| `p2p_model_sync.py`                       | P2P model sync            | `aria2_data_sync.py`                           | DEPRECATED |
| `regression_gate.py`                      | Regression detection      | `unified_orchestrator.py` regression detection | DEPRECATED |
| `run_ai_tournament.py`                    | Basic tournament runner   | `run_model_elo_tournament.py`                  | DEPRECATED |
| `run_axis_aligned_tournament.py`          | Axis-aligned evaluation   | `composite_elo_dashboard.py`                   | DEPRECATED |
| `run_crossboard_difficulty_tournament.py` | Cross-board difficulty    | Integrated into curriculum                     | DEPRECATED |
| `shadow_tournament_service.py`            | Shadow tournament service | `run_model_elo_tournament.py`                  | DEPRECATED |
| `training_dashboard.py`                   | Training dashboard        | `composite_elo_dashboard.py`                   | DEPRECATED |

---

## archive/ebmo/ (16 files)

Experimental EBMO (Energy-Based Model Operator) implementations.
Some approaches may be harvested for future use.

| Script                          | Purpose                    | Status     |
| ------------------------------- | -------------------------- | ---------- |
| `benchmark_ebmo_ladder.py`      | EBMO ladder benchmark      | HISTORICAL |
| `diagnose_ebmo.py`              | EBMO diagnostics           | HISTORICAL |
| `eval_ebmo_56ch.py`             | 56-channel EBMO eval       | HISTORICAL |
| `eval_ebmo_quick.py`            | Quick EBMO evaluation      | HISTORICAL |
| `generate_ebmo_expert_data.py`  | Expert data generation     | HISTORICAL |
| `generate_ebmo_selfplay.py`     | EBMO selfplay data         | HISTORICAL |
| `generate_ebmo_vs_heuristic.py` | EBMO vs heuristic games    | HISTORICAL |
| `test_ebmo_online.py`           | Online EBMO testing        | HISTORICAL |
| `train_ebmo.py`                 | Basic EBMO training        | HISTORICAL |
| `train_ebmo_contrastive.py`     | Contrastive EBMO training  | HISTORICAL |
| `train_ebmo_curriculum.py`      | Curriculum EBMO training   | HISTORICAL |
| `train_ebmo_expert.py`          | Expert EBMO training       | HISTORICAL |
| `train_ebmo_improved.py`        | Improved EBMO training     | HISTORICAL |
| `train_ebmo_outcome.py`         | Outcome-based EBMO         | HISTORICAL |
| `train_ebmo_quality.py`         | Quality-weighted EBMO      | HISTORICAL |
| `tune_ebmo_hyperparams.py`      | EBMO hyperparameter tuning | HISTORICAL |

---

## archive/monitors/ (2 files)

Legacy monitoring scripts replaced by unified monitoring.

| Script                      | Purpose                   | Replacement                   | Status     |
| --------------------------- | ------------------------- | ----------------------------- | ---------- |
| `cluster_health_monitor.py` | Cluster health checks     | `node_health_orchestrator.py` | DEPRECATED |
| `simple_cluster_monitor.py` | Simple cluster monitoring | `node_health_orchestrator.py` | DEPRECATED |

---

## archive/training/ (2 files)

Legacy training loops replaced by unified orchestrator.

| Script                          | Purpose               | Replacement               | Status     |
| ------------------------------- | --------------------- | ------------------------- | ---------- |
| `continuous_training_loop.py`   | Continuous training   | `unified_orchestrator.py` | DEPRECATED |
| `multi_config_training_loop.py` | Multi-config training | `unified_orchestrator.py` | DEPRECATED |

---

## archive/orphaned_modules/

Subdirectory for modules that were removed from main codebase.
Currently empty - modules integrated or deleted.

---

## Cleanup Recommendations

### Safe to Delete (30 files)

- All `archive/debug/` scripts (24 files) - debugging complete
- All `archive/deprecated/` scripts (6 files) - functionality migrated

### Keep for Reference (20 files)

- All `archive/ebmo/` scripts (16 files) - experimental approaches may be useful
- All `archive/monitors/` scripts (2 files) - reference for monitoring patterns
- All `archive/training/` scripts (2 files) - reference for training patterns

---

## Archive Date

- Debug scripts: Archived October-November 2025
- EBMO scripts: Archived November 2025
- Deprecated scripts: Archived December 2025
- Monitor scripts: Archived December 2025
- Training scripts: Archived December 2025

---

## Notes

When deleting archived scripts, ensure:

1. No active imports from main codebase
2. No valuable patterns not captured elsewhere
3. Git history preserved for reference
