# App Import Audit

Updated: April 16, 2026

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

The April 16 surface-area pass removed the 20 largest direct-unimported `app/`
modules after checking for direct imports, test-only imports, and daemon dynamic
instantiation paths:

- `ai-service/app/ai/entropy_mcts.py`
- `ai-service/app/ai/gumbel_engine.py`
- `ai-service/app/ai/marl_framework.py`
- `ai-service/app/ai/tensor_tree.py`
- `ai-service/app/config/daemon_thresholds.py`
- `ai-service/app/config/registry.py`
- `ai-service/app/config/schema.py`
- `ai-service/app/coordination/atomic_leadership.py`
- `ai-service/app/coordination/model_registry_daemon.py`
- `ai-service/app/coordination/resilience_orchestrator.py`
- `ai-service/app/coordination/training_data_resolver.py`
- `ai-service/app/coordination/unified_data_catalog.py`
- `ai-service/app/coordination/unified_health.py`
- `ai-service/app/core/initializable.py`
- `ai-service/app/core/locking.py`
- `ai-service/app/core/task_spawner.py`
- `ai-service/app/distributed/ssh_connection_manager.py`
- `ai-service/app/distributed/subscription_registry.py`
- `ai-service/app/rules/lazy_state.py`
- `ai-service/app/training/tier_calibrator.py`

`UNIFIED_DATA_CATALOG` remains a deprecated daemon type, but its runner is a
no-op and does not import the deleted implementation file.

## Direct-Unimported Modules Retained

The current contract dashboard reports 20 direct-unimported modules under
`app/`. These were retained until a follow-up proves whether each one is a
dynamic entrypoint, script-only helper, or true orphan.

### `app/ai`

- `ai-service/app/ai/move_ordering.py`
- `ai-service/app/ai/parallel_eval.py`
- `ai-service/app/ai/tree_reuse.py`
- `ai-service/app/ai/neural_net/v6_large.py`

### `app/config`

- `ai-service/app/config/constants.py`
- `ai-service/app/config/training_targets.py`

### `app/coordination`

- `ai-service/app/coordination/canonical_model_watchdog.py`
- `ai-service/app/coordination/error_handling.py`
- `ai-service/app/coordination/pipeline_health_watchdog.py`

### `app/core`

- `ai-service/app/core/registry_base.py`

### `app/p2p`

- `ai-service/app/p2p/config_profiles.py`

### `app/training`

- `ai-service/app/training/board_hyperparams.py`
- `ai-service/app/training/npz_atomic_writer.py`
- `ai-service/app/training/npz_model_validation.py`
- `ai-service/app/training/train_checkpointing.py`
- `ai-service/app/training/train_data_validation.py`
- `ai-service/app/training/train_model_init.py`

### `app/utils`

- `ai-service/app/utils/board_type_utils.py`
- `ai-service/app/utils/exceptions.py`
- `ai-service/app/utils/load_throttle.py`

## Interpretation

- Zero-inbound does not mean dead by itself in this repository. Many modules exist to be called by top-level scripts, imported dynamically, or preserved as compatibility surfaces.
- The static scan is still useful for triage. It identified the embedded hashgraph test package as truly dead and gives a bounded review set for future cleanup sessions.
- Future dead-code removal should start from this list, then prove runtime non-use before deleting additional modules.
