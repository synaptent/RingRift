# App Import Audit

Updated: April 11, 2026

This audit records the Phase 17 import-graph review for `ai-service/app/`.

## Method

Commands used:

```bash
cd ai-service && PYTHONPATH=. python scripts/audit_import_graph.py --module-prefix app --report zero-inbound
cd ai-service && PYTHONPATH=. python scripts/audit_import_graph.py --module-prefix app.coordination --report cycles --max-depth 8
```

The new `scripts/audit_import_graph.py` script resolves:

- absolute imports
- relative imports
- `from package import submodule` imports
- inbound references across `app/`, `scripts/`, and `tests/`

This avoids the false positives produced by grep-only or absolute-import-only scans.

## Dead Files Removed

The only clearly dead files found during the Phase 17 pass were embedded test modules under `app/` that were neither imported nor discovered by the active test suite:

- `ai-service/app/coordination/hashgraph/tests/__init__.py`
- `ai-service/app/coordination/hashgraph/tests/test_consensus.py`
- `ai-service/app/coordination/hashgraph/tests/test_dag.py`
- `ai-service/app/coordination/hashgraph/tests/test_evaluation_consensus.py`
- `ai-service/app/coordination/hashgraph/tests/test_event.py`
- `ai-service/app/coordination/hashgraph/tests/test_gossip_ancestry.py`
- `ai-service/app/coordination/hashgraph/tests/test_promotion_consensus.py`

## Zero-Inbound Modules Retained

The accurate import-graph scan still reports 57 zero-inbound modules under `app/`. These were retained because they are public entry points, compatibility surfaces, runtime-selectable implementations, or leaf helpers consumed by external callers or operational flows outside the static in-repo import graph.

### `app/ai`

- `ai-service/app/ai/entropy_mcts.py`
- `ai-service/app/ai/gumbel_engine.py`
- `ai-service/app/ai/marl_framework.py`
- `ai-service/app/ai/move_ordering.py`
- `ai-service/app/ai/neural_net/v6_large.py`
- `ai-service/app/ai/parallel_eval.py`
- `ai-service/app/ai/tensor_tree.py`
- `ai-service/app/ai/tree_reuse.py`

### `app/config`

- `ai-service/app/config/constants.py`
- `ai-service/app/config/daemon_thresholds.py`
- `ai-service/app/config/registry.py`
- `ai-service/app/config/schema.py`
- `ai-service/app/config/training_targets.py`

### `app/coordination`

- `ai-service/app/coordination/atomic_leadership.py`
- `ai-service/app/coordination/canonical_model_watchdog.py`
- `ai-service/app/coordination/error_handling.py`
- `ai-service/app/coordination/model_registry_daemon.py`
- `ai-service/app/coordination/pipeline_health_watchdog.py`
- `ai-service/app/coordination/resilience_orchestrator.py`
- `ai-service/app/coordination/selfplay_upload_daemon.py`
- `ai-service/app/coordination/training_data_resolver.py`
- `ai-service/app/coordination/unified_data_catalog.py`
- `ai-service/app/coordination/unified_health.py`

### `app/core`

- `ai-service/app/core/initializable.py`
- `ai-service/app/core/locking.py`
- `ai-service/app/core/task_spawner.py`

### `app/distributed`

- `ai-service/app/distributed/external_drive_sync.py`
- `ai-service/app/distributed/ssh_connection_manager.py`
- `ai-service/app/distributed/subscription_registry.py`

### `app/monitoring`

- `ai-service/app/monitoring/keepalive_dashboard.py`

### `app/p2p`

- `ai-service/app/p2p/config_profiles.py`

### `app/rules`

- `ai-service/app/rules/lazy_state.py`
- `ai-service/app/rules/mutators/recovery.py`

### `app/training`

- `ai-service/app/training/board_hyperparams.py`
- `ai-service/app/training/ebmo_trainer.py`
- `ai-service/app/training/evaluate_gmo_baselines.py`
- `ai-service/app/training/exploration_diversity_cli.py`
- `ai-service/app/training/generate_territory_dataset.py`
- `ai-service/app/training/nnue_quality_metrics.py`
- `ai-service/app/training/npz_atomic_writer.py`
- `ai-service/app/training/npz_model_validation.py`
- `ai-service/app/training/pbt.py`
- `ai-service/app/training/tier_calibrator.py`
- `ai-service/app/training/train_checkpointing.py`
- `ai-service/app/training/train_data_validation.py`
- `ai-service/app/training/train_gmo_diverse.py`
- `ai-service/app/training/train_gmo_online.py`
- `ai-service/app/training/train_gmo_v2.py`
- `ai-service/app/training/train_gnn_policy.py`
- `ai-service/app/training/train_hybrid.py`
- `ai-service/app/training/train_ig_gmo.py`
- `ai-service/app/training/train_loop.py`
- `ai-service/app/training/train_model_init.py`
- `ai-service/app/training/uncertainty_calibration.py`

### `app/utils`

- `ai-service/app/utils/board_type_utils.py`
- `ai-service/app/utils/exceptions.py`
- `ai-service/app/utils/load_throttle.py`

## Interpretation

- Zero-inbound does not mean dead by itself in this repository. Many modules exist to be called by top-level scripts, imported dynamically, or preserved as compatibility surfaces.
- The static scan is still useful for triage. It identified the embedded hashgraph test package as truly dead and gives a bounded review set for future cleanup sessions.
- Future dead-code removal should start from this list, then prove runtime non-use before deleting additional modules.
