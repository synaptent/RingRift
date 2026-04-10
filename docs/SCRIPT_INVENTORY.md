# Script Inventory

Initial inventory generated on April 10, 2026 from top-level `ai-service/scripts/*.py` files before the Part 3 archive move below. Categorization is heuristic and based on each file name plus the first 30 lines, with CLI detection based on `argparse`, `click`, or `typer` usage.

## Summary

- Total top-level Python scripts: 601
- Scripts without argparse/click/typer: 83
- analysis: 18
- deployment: 255
- deprecated: 12
- monitoring: 92
- training: 210
- utility: 14

## Archived In Part 3

The following 23 tracked top-level scripts were moved to `ai-service/scripts/archive/part3_deprecated/` after the initial inventory pass. The archive set intentionally excludes active compatibility entry points that still have code references, including `quick_gauntlet.py`, `run_tournament.py`, `unified_data_sync.py`, and the dashboard wrappers.

| Previous File                                          | Archived File                                                                   | Reason                                                                                       |
| ------------------------------------------------------ | ------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------- |
| `ai-service/scripts/db_to_training_npz.py`             | `ai-service/scripts/archive/part3_deprecated/db_to_training_npz.py`             | Explicitly deprecated in favor of `export_replay_dataset.py`; no active code references.     |
| `ai-service/scripts/run_vast_gauntlet.py`              | `ai-service/scripts/archive/part3_deprecated/run_vast_gauntlet.py`              | Explicitly deprecated in favor of `run_gauntlet.py --parallel`; docs-only references remain. |
| `ai-service/scripts/quick_model_bench.py`              | `ai-service/scripts/archive/part3_deprecated/quick_model_bench.py`              | Unreferenced one-off quick benchmark.                                                        |
| `ai-service/scripts/quick_eval_gmo_v2.py`              | `ai-service/scripts/archive/part3_deprecated/quick_eval_gmo_v2.py`              | Unreferenced one-off quick evaluation.                                                       |
| `ai-service/scripts/diagnose_non_termination.py`       | `ai-service/scripts/archive/part3_deprecated/diagnose_non_termination.py`       | Unreferenced one-off diagnostic.                                                             |
| `ai-service/scripts/diagnose_weight_application.py`    | `ai-service/scripts/archive/part3_deprecated/diagnose_weight_application.py`    | Unreferenced one-off diagnostic.                                                             |
| `ai-service/scripts/test_import_expansion.py`          | `ai-service/scripts/archive/part3_deprecated/test_import_expansion.py`          | Unreferenced ad-hoc script test outside pytest tree.                                         |
| `ai-service/scripts/test_hex8_territory_bug.py`        | `ai-service/scripts/archive/part3_deprecated/test_hex8_territory_bug.py`        | Unreferenced ad-hoc script test outside pytest tree.                                         |
| `ai-service/scripts/test_hex8_geometry.py`             | `ai-service/scripts/archive/part3_deprecated/test_hex8_geometry.py`             | Unreferenced ad-hoc script test outside pytest tree.                                         |
| `ai-service/scripts/test_health_report.py`             | `ai-service/scripts/archive/part3_deprecated/test_health_report.py`             | Unreferenced ad-hoc script test outside pytest tree.                                         |
| `ai-service/scripts/test_descent_debug.py`             | `ai-service/scripts/archive/part3_deprecated/test_descent_debug.py`             | Unreferenced ad-hoc script test outside pytest tree.                                         |
| `ai-service/scripts/validate_training_integrations.py` | `ai-service/scripts/archive/part3_deprecated/validate_training_integrations.py` | Unreferenced one-off validation script.                                                      |
| `ai-service/scripts/validate_trained_weights.py`       | `ai-service/scripts/archive/part3_deprecated/validate_trained_weights.py`       | Unreferenced one-off validation script.                                                      |
| `ai-service/scripts/validate_swap_decisions.py`        | `ai-service/scripts/archive/part3_deprecated/validate_swap_decisions.py`        | Unreferenced one-off validation script.                                                      |
| `ai-service/scripts/validate_selfplay_games.py`        | `ai-service/scripts/archive/part3_deprecated/validate_selfplay_games.py`        | Unreferenced one-off validation script.                                                      |
| `ai-service/scripts/validate_selfplay_data.py`         | `ai-service/scripts/archive/part3_deprecated/validate_selfplay_data.py`         | Unreferenced one-off validation script.                                                      |
| `ai-service/scripts/validate_phase_recording.py`       | `ai-service/scripts/archive/part3_deprecated/validate_phase_recording.py`       | Unreferenced one-off validation script.                                                      |
| `ai-service/scripts/validate_npz_encoding.py`          | `ai-service/scripts/archive/part3_deprecated/validate_npz_encoding.py`          | Unreferenced one-off validation script.                                                      |
| `ai-service/scripts/validate_minimax_policy.py`        | `ai-service/scripts/archive/part3_deprecated/validate_minimax_policy.py`        | Unreferenced one-off validation script.                                                      |
| `ai-service/scripts/validate_mcts_policy.py`           | `ai-service/scripts/archive/part3_deprecated/validate_mcts_policy.py`           | Unreferenced one-off validation script.                                                      |
| `ai-service/scripts/validate_hex8_batch.py`            | `ai-service/scripts/archive/part3_deprecated/validate_hex8_batch.py`            | Unreferenced one-off validation script.                                                      |
| `ai-service/scripts/validate_gpu_mcts_data.py`         | `ai-service/scripts/archive/part3_deprecated/validate_gpu_mcts_data.py`         | Unreferenced one-off validation script.                                                      |
| `ai-service/scripts/transfer_learning_experiment.py`   | `ai-service/scripts/archive/part3_deprecated/transfer_learning_experiment.py`   | Unreferenced one-off experiment script.                                                      |

## Candidates For Archive

These scripts had explicit deprecation, supersession, or legacy-compatibility language in the initial inventory pass. Some were moved in the Part 3 archive step above; the remaining entries require import/reference checks before moving.

| File                                                      | Reason                                                                               |
| --------------------------------------------------------- | ------------------------------------------------------------------------------------ |
| `ai-service/scripts/audit_deprecated_imports.py`          | explicit deprecation marker; points to replacement script                            |
| `ai-service/scripts/check_deprecated_imports.py`          | explicit deprecation marker; points to replacement script                            |
| `ai-service/scripts/composite_elo_dashboard.py`           | explicit deprecation marker; points to replacement script                            |
| `ai-service/scripts/data_prep.py`                         | points to replacement script                                                         |
| `ai-service/scripts/db_to_training_npz.py`                | explicit deprecation marker; legacy compatibility shim; points to replacement script |
| `ai-service/scripts/elo_dashboard.py`                     | explicit deprecation marker; points to replacement script                            |
| `ai-service/scripts/export_replay_dataset.py`             | points to replacement script                                                         |
| `ai-service/scripts/launch_daemons.py`                    | explicit deprecation marker                                                          |
| `ai-service/scripts/launch_distributed_elo_tournament.py` | points to replacement script                                                         |
| `ai-service/scripts/master_loop.py`                       | points to replacement script                                                         |
| `ai-service/scripts/pipeline_dashboard.py`                | explicit deprecation marker; points to replacement script                            |
| `ai-service/scripts/quick_gauntlet.py`                    | explicit deprecation marker                                                          |
| `ai-service/scripts/run_model_elo_tournament.py`          | points to replacement script                                                         |
| `ai-service/scripts/run_parity_promotion_gate.py`         | explicit deprecation marker; points to replacement script                            |
| `ai-service/scripts/run_self_play_soak.py`                | points to replacement script                                                         |
| `ai-service/scripts/run_vast_gauntlet.py`                 | explicit deprecation marker; points to replacement script                            |
| `ai-service/scripts/selfplay.py`                          | explicit deprecation marker; points to replacement script                            |

## Help Coverage Review

The most-used script references were gathered from `package.json`, `CLAUDE.md`, `CLAUDE.local.md`, and deployment scripts. Every referenced top-level Python entrypoint already uses `argparse`, `click`, or `typer`; the only referenced Python file without a CLI parser was `ai-service/scripts/p2p/constants.py`, which is a constants module copied by a deploy helper rather than an executable script. No additional help-text edits were needed for Task 27.

## Inventory

| File                                                          | Category   | CLI | Last Modified |  LOC |
| ------------------------------------------------------------- | ---------- | --- | ------------- | ---: |
| `ai-service/scripts/audit_circular_deps.py`                   | analysis   | y   | 2025-12-27    |  321 |
| `ai-service/scripts/audit_exception_handlers.py`              | analysis   | y   | 2025-12-29    |  316 |
| `ai-service/scripts/audit_module_complexity.py`               | analysis   | y   | 2025-12-29    |  382 |
| `ai-service/scripts/audit_test_coverage.py`                   | analysis   | y   | 2025-12-29    |  293 |
| `ai-service/scripts/benchmark_gpu_batch.py`                   | analysis   | y   | 2025-12-26    |  142 |
| `ai-service/scripts/benchmark_move_application.py`            | analysis   | y   | 2025-12-20    |  359 |
| `ai-service/scripts/check_vectorized_movement.py`             | analysis   | y   | 2025-12-28    |  275 |
| `ai-service/scripts/debug_ts_python_state_diff.py`            | analysis   | n   | 2025-12-20    |   29 |
| `ai-service/scripts/diagnose_anm_divergence.py`               | analysis   | y   | 2025-12-21    |  330 |
| `ai-service/scripts/diff_state_bundle.py`                     | analysis   | y   | 2025-12-26    |  196 |
| `ai-service/scripts/export_heuristic_weights.py`              | analysis   | y   | 2025-12-20    |  148 |
| `ai-service/scripts/generate_architecture_report.py`          | analysis   | y   | 2025-12-30    |  396 |
| `ai-service/scripts/generate_event_schemas.py`                | analysis   | y   | 2026-01-03    |  442 |
| `ai-service/scripts/run_gpu_lps_ablation.py`                  | analysis   | y   | 2025-12-20    |  390 |
| `ai-service/scripts/run_lps_ablation.py`                      | analysis   | y   | 2025-12-20    |  554 |
| `ai-service/scripts/test_gpu_minimax.py`                      | analysis   | y   | 2026-01-12    |  559 |
| `ai-service/scripts/validate_swap_decisions.py`               | analysis   | y   | 2025-12-20    |  550 |
| `ai-service/scripts/verify_event_wiring.py`                   | analysis   | n   | 2025-12-28    |  232 |
| `ai-service/scripts/aggregate_cluster_data.py`                | deployment | y   | 2025-12-26    |  319 |
| `ai-service/scripts/aggregate_cluster_dbs.py`                 | deployment | y   | 2026-01-13    |  596 |
| `ai-service/scripts/aggregate_elo_results.py`                 | deployment | y   | 2025-12-20    |  308 |
| `ai-service/scripts/analyze_cluster_games.py`                 | deployment | y   | 2025-12-25    | 1620 |
| `ai-service/scripts/analyze_game_statistics.py`               | deployment | y   | 2025-12-26    | 2550 |
| `ai-service/scripts/aria2_data_sync.py`                       | deployment | y   | 2025-12-27    | 1139 |
| `ai-service/scripts/audit_and_cleanup_databases.py`           | deployment | y   | 2025-12-24    |  366 |
| `ai-service/scripts/audit_cluster_data.py`                    | deployment | y   | 2025-12-26    |  829 |
| `ai-service/scripts/auto_composite_eval.py`                   | deployment | y   | 2025-12-20    |  320 |
| `ai-service/scripts/auto_deploy_models.py`                    | deployment | y   | 2025-12-26    |  541 |
| `ai-service/scripts/auto_promote.py`                          | deployment | y   | 2026-02-28    | 1377 |
| `ai-service/scripts/auto_start_idle_selfplay.py`              | deployment | y   | 2025-12-28    |  292 |
| `ai-service/scripts/auto_sync_and_promote.py`                 | deployment | n   | 2025-12-30    |  213 |
| `ai-service/scripts/auto_training_pipeline.py`                | deployment | y   | 2025-12-17    | 1244 |
| `ai-service/scripts/auto_training_trigger.py`                 | deployment | y   | 2026-01-12    |  229 |
| `ai-service/scripts/autonomous_monitor.py`                    | deployment | y   | 2025-12-29    |  724 |
| `ai-service/scripts/baseline_gauntlet.py`                     | deployment | y   | 2025-12-26    |  306 |
| `ai-service/scripts/batch_deploy_p2p.py`                      | deployment | y   | 2025-12-25    |  135 |
| `ai-service/scripts/benchmark_ai_algorithms.py`               | deployment | y   | 2025-12-21    |  439 |
| `ai-service/scripts/benchmark_cluster.py`                     | deployment | y   | 2025-12-28    |  388 |
| `ai-service/scripts/benchmark_gpu_mcts.py`                    | deployment | y   | 2025-12-25    |  131 |
| `ai-service/scripts/benchmark_make_unmake.py`                 | deployment | n   | 2025-12-26    |  571 |
| `ai-service/scripts/benchmark_policy.py`                      | deployment | y   | 2025-12-26    |  589 |
| `ai-service/scripts/bootstrap_multiplayer_elo.py`             | deployment | y   | 2025-12-26    |  195 |
| `ai-service/scripts/bootstrap_training.py`                    | deployment | y   | 2026-01-23    |  416 |
| `ai-service/scripts/bootstrap_v5_heavy.py`                    | deployment | y   | 2026-03-03    |  542 |
| `ai-service/scripts/build_canonical_training_pool_db.py`      | deployment | y   | 2025-12-26    |  918 |
| `ai-service/scripts/check_cluster_ladder_artifacts.py`        | deployment | y   | 2025-12-20    |  252 |
| `ai-service/scripts/check_daemon_reference_integrity.py`      | deployment | n   | 2026-03-24    |   99 |
| `ai-service/scripts/check_import_integrity.py`                | deployment | n   | 2026-04-03    |   67 |
| `ai-service/scripts/check_ladder_artifacts.py`                | deployment | y   | 2025-12-24    |  262 |
| `ai-service/scripts/check_p2p_cluster_status.py`              | deployment | n   | 2025-12-26    |  136 |
| `ai-service/scripts/check_p2p_comprehensive.py`               | deployment | n   | 2025-12-27    |  401 |
| `ai-service/scripts/check_p2p_status.py`                      | deployment | y   | 2025-12-28    |  495 |
| `ai-service/scripts/check_p2p_status_all_nodes.py`            | deployment | n   | 2025-12-26    |  304 |
| `ai-service/scripts/check_sync_health.py`                     | deployment | y   | 2025-12-26    |  332 |
| `ai-service/scripts/claude_monitor_loop.py`                   | deployment | n   | 2026-01-12    |  207 |
| `ai-service/scripts/cleanup_phantom_elo_entries.py`           | deployment | y   | 2026-01-12    |  793 |
| `ai-service/scripts/cli.py`                                   | deployment | y   | 2025-12-20    |  620 |
| `ai-service/scripts/cluster_activator.py`                     | deployment | y   | 2025-12-28    | 1297 |
| `ai-service/scripts/cluster_cleanup_corrupt_data.py`          | deployment | y   | 2025-12-29    |  494 |
| `ai-service/scripts/cluster_cleanup_orphan_games.py`          | deployment | y   | 2026-01-10    |  633 |
| `ai-service/scripts/cluster_data_status.py`                   | deployment | y   | 2026-01-27    |  460 |
| `ai-service/scripts/cluster_db_cleanup.py`                    | deployment | y   | 2025-12-27    |  788 |
| `ai-service/scripts/cluster_file_sync.py`                     | deployment | y   | 2026-01-12    | 1042 |
| `ai-service/scripts/cluster_health_check.py`                  | deployment | y   | 2025-12-25    |  328 |
| `ai-service/scripts/cluster_health_cli.py`                    | deployment | y   | 2025-12-25    |  431 |
| `ai-service/scripts/cluster_health_monitor.py`                | deployment | y   | 2026-01-25    |  173 |
| `ai-service/scripts/cluster_health_summary.py`                | deployment | y   | 2025-12-25    |  471 |
| `ai-service/scripts/cluster_master_deploy.py`                 | deployment | y   | 2025-12-25    |  365 |
| `ai-service/scripts/cluster_speedup.py`                       | deployment | y   | 2025-12-26    |  500 |
| `ai-service/scripts/cluster_submit.py`                        | deployment | y   | 2025-12-25    |  490 |
| `ai-service/scripts/cluster_supervisor.py`                    | deployment | y   | 2025-12-25    |  517 |
| `ai-service/scripts/cluster_update_coordinator.py`            | deployment | y   | 2026-04-06    | 1565 |
| `ai-service/scripts/cluster_watchdog.py`                      | deployment | y   | 2025-12-26    |  282 |
| `ai-service/scripts/cluster_worker.py`                        | deployment | y   | 2025-12-25    |  406 |
| `ai-service/scripts/cmaes_cloud_worker.py`                    | deployment | y   | 2025-12-20    |  426 |
| `ai-service/scripts/collect_last24h_selfplay_reports.py`      | deployment | y   | 2025-12-27    |  544 |
| `ai-service/scripts/configure_s3_lifecycle.py`                | deployment | y   | 2026-01-23    |  210 |
| `ai-service/scripts/consolidate_cluster_games.py`             | deployment | y   | 2025-12-25    |  380 |
| `ai-service/scripts/consolidate_hexagonal.py`                 | deployment | n   | 2025-12-29    |  271 |
| `ai-service/scripts/consolidate_lambda_backup.py`             | deployment | y   | 2025-12-29    |  389 |
| `ai-service/scripts/consolidate_owc_data.py`                  | deployment | y   | 2025-12-29    |  524 |
| `ai-service/scripts/consolidated_cluster_monitor.py`          | deployment | y   | 2026-02-24    |  516 |
| `ai-service/scripts/continuous_smoke_test.py`                 | deployment | y   | 2026-04-05    |  255 |
| `ai-service/scripts/convert_hetzner_jsonl.py`                 | deployment | n   | 2025-12-25    |  142 |
| `ai-service/scripts/convert_local_jsonl.py`                   | deployment | y   | 2026-01-12    |  167 |
| `ai-service/scripts/coordinator_s3_backup.py`                 | deployment | y   | 2025-12-28    |  376 |
| `ai-service/scripts/copy_missing_moves.py`                    | deployment | n   | 2025-12-29    |  105 |
| `ai-service/scripts/cross_board_transfer.py`                  | deployment | y   | 2026-01-23    |  208 |
| `ai-service/scripts/daemon_health_monitor.py`                 | deployment | y   | 2026-01-12    |  326 |
| `ai-service/scripts/dashboard.py`                             | deployment | y   | 2025-12-26    |  433 |
| `ai-service/scripts/dashboard_server.py`                      | deployment | y   | 2026-02-23    | 1296 |
| `ai-service/scripts/data_aggregator.py`                       | deployment | y   | 2025-12-26    |  489 |
| `ai-service/scripts/data_pipeline_coordinator.py`             | deployment | y   | 2026-01-12    |  828 |
| `ai-service/scripts/data_status.py`                           | deployment | y   | 2026-01-13    |  370 |
| `ai-service/scripts/demo_auto_export.py`                      | deployment | n   | 2026-01-05    |  205 |
| `ai-service/scripts/deploy_cluster_protocols.py`              | deployment | y   | 2025-12-28    |  429 |
| `ai-service/scripts/deploy_distributed_cmaes.py`              | deployment | y   | 2025-12-25    |  249 |
| `ai-service/scripts/deploy_gmo_training.py`                   | deployment | y   | 2025-12-25    |  352 |
| `ai-service/scripts/deploy_keepalive.py`                      | deployment | y   | 2025-12-25    |  414 |
| `ai-service/scripts/deploy_lps_ablation.py`                   | deployment | y   | 2025-12-26    |  474 |
| `ai-service/scripts/deploy_nodejs_to_cluster.py`              | deployment | y   | 2025-12-28    |  452 |
| `ai-service/scripts/deploy_p2p_autorestart.py`                | deployment | y   | 2025-12-26    |  314 |
| `ai-service/scripts/deploy_p2p_cluster.py`                    | deployment | y   | 2026-02-16    |  687 |
| `ai-service/scripts/deploy_p2p_service.py`                    | deployment | y   | 2025-12-25    |  423 |
| `ai-service/scripts/deploy_p2p_supervision.py`                | deployment | y   | 2025-12-29    |  259 |
| `ai-service/scripts/deploy_p2p_supervisor.py`                 | deployment | y   | 2026-01-18    |  412 |
| `ai-service/scripts/deploy_p2p_systemd.py`                    | deployment | y   | 2026-01-12    |  356 |
| `ai-service/scripts/deploy_persona_tournament.py`             | deployment | y   | 2025-12-25    |  381 |
| `ai-service/scripts/deploy_smoke_runner.py`                   | deployment | n   | 2026-04-05    |   85 |
| `ai-service/scripts/deploy_smoke_test.py`                     | deployment | y   | 2026-04-05    |  205 |
| `ai-service/scripts/deploy_tailscale_vast.py`                 | deployment | y   | 2026-01-19    |  325 |
| `ai-service/scripts/disaster_recovery_cli.py`                 | deployment | y   | 2026-01-03    |  310 |
| `ai-service/scripts/disk_monitor.py`                          | deployment | y   | 2026-02-24    |  713 |
| `ai-service/scripts/distillation_daemon.py`                   | deployment | y   | 2025-12-26    |  464 |
| `ai-service/scripts/distribute_models_aria2.py`               | deployment | y   | 2025-12-26    |  277 |
| `ai-service/scripts/distributed_model_evaluator.py`           | deployment | y   | 2025-12-17    |  534 |
| `ai-service/scripts/distributed_nas.py`                       | deployment | y   | 2025-12-26    | 1723 |
| `ai-service/scripts/distributed_tournament.py`                | deployment | y   | 2026-01-12    |  562 |
| `ai-service/scripts/dynamic_data_distribution.py`             | deployment | y   | 2025-12-27    |  723 |
| `ai-service/scripts/dynamic_space_manager.py`                 | deployment | y   | 2025-12-27    |  444 |
| `ai-service/scripts/elo_db_sync.py`                           | deployment | y   | 2026-01-23    | 1095 |
| `ai-service/scripts/elo_progress.py`                          | deployment | y   | 2026-01-12    |  340 |
| `ai-service/scripts/elo_reconciliation_cli.py`                | deployment | y   | 2025-12-20    |  865 |
| `ai-service/scripts/enable_cluster_protocols.py`              | deployment | y   | 2026-02-26    |  387 |
| `ai-service/scripts/enable_s3_sync_cluster.py`                | deployment | y   | 2026-01-12    |  367 |
| `ai-service/scripts/ensure_cluster_env.py`                    | deployment | y   | 2025-12-26    |  294 |
| `ai-service/scripts/export_training_from_cluster.py`          | deployment | y   | 2025-12-26    |  288 |
| `ai-service/scripts/external_drive_sync_daemon.py`            | deployment | y   | 2025-12-17    | 1489 |
| `ai-service/scripts/filter_quality_games.py`                  | deployment | y   | 2026-01-23    |  285 |
| `ai-service/scripts/filter_timeout_games.py`                  | deployment | y   | 2025-12-20    |  546 |
| `ai-service/scripts/fix_autonomous_training.py`               | deployment | y   | 2026-01-23    |  311 |
| `ai-service/scripts/fleet_health_check.py`                    | deployment | y   | 2026-04-10    |  586 |
| `ai-service/scripts/force_restart_p2p_cluster.py`             | deployment | y   | 2026-02-16    |  393 |
| `ai-service/scripts/gauntlet_runner.py`                       | deployment | n   | 2026-03-31    |  345 |
| `ai-service/scripts/generate_p2p_config.py`                   | deployment | y   | 2025-12-26    |  180 |
| `ai-service/scripts/gpu_cluster_manager.py`                   | deployment | y   | 2025-12-25    |  701 |
| `ai-service/scripts/harvest_cluster_training_data.py`         | deployment | y   | 2025-12-25    |  553 |
| `ai-service/scripts/harvest_local_training_data.py`           | deployment | y   | 2025-12-20    |  442 |
| `ai-service/scripts/health_alerting.py`                       | deployment | y   | 2026-02-24    |  590 |
| `ai-service/scripts/http_pull.py`                             | deployment | y   | 2025-12-29    |  401 |
| `ai-service/scripts/idle_gpu_alert.py`                        | deployment | y   | 2025-12-20    |  355 |
| `ai-service/scripts/idle_node_alerter.py`                     | deployment | y   | 2025-12-25    |  454 |
| `ai-service/scripts/import_orphaned_databases.py`             | deployment | y   | 2025-12-25    |  508 |
| `ai-service/scripts/job_state_sync_daemon.py`                 | deployment | y   | 2025-12-20    |  113 |
| `ai-service/scripts/lambda_cli.py`                            | deployment | n   | 2026-03-10    |  176 |
| `ai-service/scripts/launch_coordinator_disk_manager.py`       | deployment | n   | 2025-12-27    |  129 |
| `ai-service/scripts/launch_daemons.py`                        | deployment | y   | 2026-03-23    |  673 |
| `ai-service/scripts/launch_distributed_elo_tournament.py`     | deployment | y   | 2025-12-25    | 1395 |
| `ai-service/scripts/launch_distributed_nas.py`                | deployment | y   | 2025-12-26    |  312 |
| `ai-service/scripts/launch_local_p2p.py`                      | deployment | y   | 2026-02-16    |  134 |
| `ai-service/scripts/launch_node_availability.py`              | deployment | y   | 2025-12-28    |  164 |
| `ai-service/scripts/master_cluster_automation.py`             | deployment | y   | 2025-12-28    |  477 |
| `ai-service/scripts/master_loop.py`                           | deployment | y   | 2026-04-10    | 3328 |
| `ai-service/scripts/master_loop_runner.py`                    | deployment | y   | 2026-01-12    |  279 |
| `ai-service/scripts/master_loop_watchdog.py`                  | deployment | y   | 2026-04-05    |  638 |
| `ai-service/scripts/minimal_alphazero_loop.py`                | deployment | y   | 2026-04-08    |  879 |
| `ai-service/scripts/model_distillation.py`                    | deployment | y   | 2025-12-24    |  675 |
| `ai-service/scripts/model_promotion_manager.py`               | deployment | y   | 2026-04-05    | 1913 |
| `ai-service/scripts/monitor_10h_selfplay.py`                  | deployment | y   | 2025-12-27    |  311 |
| `ai-service/scripts/monitor_48h.py`                           | deployment | y   | 2026-01-23    |  302 |
| `ai-service/scripts/monitor_cluster_10h.py`                   | deployment | y   | 2026-01-12    |  308 |
| `ai-service/scripts/monitor_cluster_jobs.py`                  | deployment | n   | 2025-12-25    |  135 |
| `ai-service/scripts/monitor_gpu_mcts_jobs.py`                 | deployment | y   | 2025-12-25    |  216 |
| `ai-service/scripts/monitor_hex8_2p.py`                       | deployment | n   | 2025-12-26    |  127 |
| `ai-service/scripts/monitor_p2p_cluster.py`                   | deployment | y   | 2026-01-24    |  216 |
| `ai-service/scripts/monitor_p2p_stability.py`                 | deployment | y   | 2026-01-23    |  106 |
| `ai-service/scripts/monitor_selfplay_jobs.py`                 | deployment | y   | 2025-12-28    |  282 |
| `ai-service/scripts/node_resilience.py`                       | deployment | y   | 2026-02-24    | 1488 |
| `ai-service/scripts/orchestrated_data_sync.py`                | deployment | y   | 2025-12-27    |  261 |
| `ai-service/scripts/owc_lifecycle_manager.py`                 | deployment | y   | 2026-03-18    |  512 |
| `ai-service/scripts/owc_s3_mirror.py`                         | deployment | y   | 2026-03-18    |  709 |
| `ai-service/scripts/p2p_cluster_status.py`                    | deployment | y   | 2025-12-28    |  502 |
| `ai-service/scripts/p2p_file_monitor.py`                      | deployment | y   | 2026-01-24    |  125 |
| `ai-service/scripts/p2p_health_monitor.py`                    | deployment | y   | 2026-01-24    |  134 |
| `ai-service/scripts/p2p_model_distribution.py`                | deployment | y   | 2026-01-12    |  813 |
| `ai-service/scripts/p2p_monitor.py`                           | deployment | n   | 2026-01-22    |   73 |
| `ai-service/scripts/p2p_orchestrator.py`                      | deployment | y   | 2026-04-10    | 2591 |
| `ai-service/scripts/p2p_stability_monitor.py`                 | deployment | y   | 2026-01-24    |  152 |
| `ai-service/scripts/p2p_supervisor.py`                        | deployment | y   | 2026-03-05    |  344 |
| `ai-service/scripts/p2p_watchdog.py`                          | deployment | y   | 2025-12-28    |  341 |
| `ai-service/scripts/parallel_training_orchestrator.py`        | deployment | y   | 2025-12-26    |  512 |
| `ai-service/scripts/preflight_48h.py`                         | deployment | y   | 2025-12-29    |  799 |
| `ai-service/scripts/production_elo_sync.py`                   | deployment | n   | 2026-01-16    |  178 |
| `ai-service/scripts/provision_node_id.py`                     | deployment | y   | 2026-01-13    |  359 |
| `ai-service/scripts/recover_backup_models.py`                 | deployment | y   | 2025-12-29    |  414 |
| `ai-service/scripts/recover_cluster.py`                       | deployment | y   | 2026-01-02    |  265 |
| `ai-service/scripts/recover_p2p_cluster.py`                   | deployment | y   | 2026-01-27    |  369 |
| `ai-service/scripts/recover_tailscale_nodes.py`               | deployment | y   | 2025-12-29    |  382 |
| `ai-service/scripts/refresh_stale_elo.py`                     | deployment | y   | 2026-01-11    |  249 |
| `ai-service/scripts/register_node.py`                         | deployment | y   | 2025-12-26    |  238 |
| `ai-service/scripts/remote_watchdog.py`                       | deployment | n   | 2025-12-26    |  185 |
| `ai-service/scripts/resource_aware_router.py`                 | deployment | y   | 2025-12-24    |  752 |
| `ai-service/scripts/restart_all_p2p.py`                       | deployment | y   | 2026-01-20    |  677 |
| `ai-service/scripts/restart_p2p_cluster.py`                   | deployment | y   | 2026-01-27    |  402 |
| `ai-service/scripts/robust_consolidate.py`                    | deployment | n   | 2026-01-10    |  327 |
| `ai-service/scripts/run_10hour_cluster_push.py`               | deployment | y   | 2025-12-26    |  306 |
| `ai-service/scripts/run_cluster_elo_refresh.py`               | deployment | y   | 2026-01-11    |  536 |
| `ai-service/scripts/run_composite_gauntlet.py`                | deployment | y   | 2025-12-20    |  326 |
| `ai-service/scripts/run_continuous_training.py`               | deployment | n   | 2025-12-24    |   49 |
| `ai-service/scripts/run_data_pipeline.py`                     | deployment | y   | 2025-12-26    |  813 |
| `ai-service/scripts/run_distributed_gpu_cmaes.py`             | deployment | y   | 2025-12-23    |  693 |
| `ai-service/scripts/run_distributed_selfplay.py`              | deployment | y   | 2026-01-01    | 1071 |
| `ai-service/scripts/run_distributed_selfplay_soak.py`         | deployment | y   | 2025-12-28    | 1659 |
| `ai-service/scripts/run_distributed_tournament.py`            | deployment | y   | 2026-02-26    | 1882 |
| `ai-service/scripts/run_diverse_tournaments.py`               | deployment | y   | 2025-12-21    |  881 |
| `ai-service/scripts/run_evotorch_cmaes.py`                    | deployment | y   | 2025-12-23    |  525 |
| `ai-service/scripts/run_gauntlet.py`                          | deployment | y   | 2026-01-12    |  347 |
| `ai-service/scripts/run_gpu_tests_cloud.py`                   | deployment | y   | 2025-12-26    |  358 |
| `ai-service/scripts/run_improvement_eval.py`                  | deployment | y   | 2026-01-12    |  237 |
| `ai-service/scripts/run_iterative_cmaes.py`                   | deployment | y   | 2025-12-26    | 1022 |
| `ai-service/scripts/run_massive_tournament.py`                | deployment | y   | 2026-01-04    |  502 |
| `ai-service/scripts/run_multigame_gumbel_selfplay.py`         | deployment | y   | 2026-02-16    |  461 |
| `ai-service/scripts/run_p2p_elo_tournament.py`                | deployment | y   | 2025-12-29    |  809 |
| `ai-service/scripts/run_partitioned_gauntlet.py`              | deployment | y   | 2025-12-20    |  108 |
| `ai-service/scripts/run_python_parity_gate.py`                | deployment | y   | 2025-12-28    |  331 |
| `ai-service/scripts/run_robust_cmaes.py`                      | deployment | y   | 2025-12-23    |  895 |
| `ai-service/scripts/run_ssh_distributed_tournament.py`        | deployment | y   | 2025-12-21    |  894 |
| `ai-service/scripts/run_targeted_gauntlet.py`                 | deployment | y   | 2025-12-20    |  141 |
| `ai-service/scripts/run_tests_cluster.py`                     | deployment | y   | 2025-12-30    |  573 |
| `ai-service/scripts/run_training_loop.py`                     | deployment | y   | 2026-03-31    |  503 |
| `ai-service/scripts/run_unified_cmaes.py`                     | deployment | y   | 2025-12-23    | 1641 |
| `ai-service/scripts/s3_backup.py`                             | deployment | y   | 2026-02-11    |  311 |
| `ai-service/scripts/s3_gauntlet_promoter.py`                  | deployment | y   | 2026-03-23    |  251 |
| `ai-service/scripts/scan_canonical_games.py`                  | deployment | y   | 2025-12-28    |  796 |
| `ai-service/scripts/scheduled_npz_export.py`                  | deployment | y   | 2026-02-26    |  445 |
| `ai-service/scripts/selfplay_campaign_hex8_2p.py`             | deployment | y   | 2025-12-27    |  328 |
| `ai-service/scripts/serf_event_handler.py`                    | deployment | n   | 2025-12-27    |  447 |
| `ai-service/scripts/setup_mac_coordinators.py`                | deployment | y   | 2025-12-30    |  705 |
| `ai-service/scripts/sharded_gauntlet.py`                      | deployment | y   | 2026-01-21    |  293 |
| `ai-service/scripts/smart_work_router.py`                     | deployment | y   | 2026-01-12    |  733 |
| `ai-service/scripts/soft_target_transfer_pipeline.py`         | deployment | y   | 2026-01-12    |  317 |
| `ai-service/scripts/spawn_10h_selfplay.py`                    | deployment | y   | 2025-12-28    |  219 |
| `ai-service/scripts/start_coordinator.py`                     | deployment | y   | 2026-03-23    |  187 |
| `ai-service/scripts/start_p2p_cluster.py`                     | deployment | y   | 2026-01-12    |  229 |
| `ai-service/scripts/sync_all_nodes.py`                        | deployment | y   | 2025-12-25    |  289 |
| `ai-service/scripts/sync_models.py`                           | deployment | y   | 2026-03-30    | 1424 |
| `ai-service/scripts/sync_models_to_production.py`             | deployment | n   | 2026-02-22    |  200 |
| `ai-service/scripts/sync_monitor.py`                          | deployment | y   | 2025-12-25    |  174 |
| `ai-service/scripts/sync_selfplay_data.py`                    | deployment | y   | 2025-12-28    |  233 |
| `ai-service/scripts/sync_staging_ai_artifacts.py`             | deployment | y   | 2025-12-26    |  352 |
| `ai-service/scripts/sync_staging_ai_pipeline.py`              | deployment | y   | 2025-12-20    |  128 |
| `ai-service/scripts/sync_training_data.py`                    | deployment | y   | 2025-12-28    |  209 |
| `ai-service/scripts/test_gpu_all.py`                          | deployment | y   | 2025-12-20    |  200 |
| `ai-service/scripts/train_all_nnue_models.py`                 | deployment | y   | 2025-12-23    |  414 |
| `ai-service/scripts/training_monitor.py`                      | deployment | y   | 2025-12-25    |  189 |
| `ai-service/scripts/training_orchestrator.py`                 | deployment | y   | 2025-12-28    |  528 |
| `ai-service/scripts/training_status.py`                       | deployment | y   | 2026-04-10    |  371 |
| `ai-service/scripts/transfer_2p_to_4p.py`                     | deployment | y   | 2026-04-05    |  239 |
| `ai-service/scripts/transfer_learning.py`                     | deployment | y   | 2025-12-24    |  589 |
| `ai-service/scripts/transfer_learning_experiment.py`          | deployment | y   | 2025-12-26    |  549 |
| `ai-service/scripts/trigger_priority_selfplay.py`             | deployment | y   | 2026-01-12    |  268 |
| `ai-service/scripts/two_stage_gauntlet.py`                    | deployment | y   | 2025-12-26    |  931 |
| `ai-service/scripts/unified_promotion_daemon.py`              | deployment | y   | 2025-12-26    |  946 |
| `ai-service/scripts/universal_keepalive.py`                   | deployment | y   | 2026-02-24    | 1019 |
| `ai-service/scripts/update_all_nodes.py`                      | deployment | y   | 2026-04-05    | 1356 |
| `ai-service/scripts/update_cluster_code.py`                   | deployment | y   | 2025-12-26    |  326 |
| `ai-service/scripts/update_distributed_hosts.py`              | deployment | y   | 2025-12-26    |  123 |
| `ai-service/scripts/update_ssh_config_from_tailscale.py`      | deployment | y   | 2026-01-05    |  320 |
| `ai-service/scripts/validate_cluster_elo.py`                  | deployment | y   | 2026-01-23    |  307 |
| `ai-service/scripts/validate_models.py`                       | deployment | y   | 2026-01-12    |  774 |
| `ai-service/scripts/vast_autoscaler.py`                       | deployment | y   | 2025-12-26    |  777 |
| `ai-service/scripts/vast_cpu_pipeline_daemon.py`              | deployment | y   | 2025-12-29    |  438 |
| `ai-service/scripts/vast_keepalive.py`                        | deployment | y   | 2026-01-12    |  468 |
| `ai-service/scripts/vast_lifecycle.py`                        | deployment | y   | 2025-12-25    |  696 |
| `ai-service/scripts/vast_p2p_manager.py`                      | deployment | y   | 2025-12-26    |  267 |
| `ai-service/scripts/vast_p2p_setup.py`                        | deployment | y   | 2025-12-26    |  628 |
| `ai-service/scripts/vast_p2p_sync.py`                         | deployment | y   | 2025-12-25    | 1044 |
| `ai-service/scripts/vast_selfplay_wrapper.py`                 | deployment | y   | 2026-01-19    |  314 |
| `ai-service/scripts/vastai_termination_guard.py`              | deployment | y   | 2025-12-28    |  566 |
| `ai-service/scripts/verify_architecture_changes.py`           | deployment | n   | 2026-03-01    |  189 |
| `ai-service/scripts/verify_elo_pipeline.py`                   | deployment | n   | 2025-12-29    |  208 |
| `ai-service/scripts/verify_nfs_sync.py`                       | deployment | y   | 2025-12-23    |  498 |
| `ai-service/scripts/weekly_gauntlet.py`                       | deployment | y   | 2026-01-10    |  247 |
| `ai-service/scripts/audit_deprecated_imports.py`              | deprecated | y   | 2025-12-30    |  474 |
| `ai-service/scripts/check_deprecated_imports.py`              | deprecated | y   | 2025-12-30    |  461 |
| `ai-service/scripts/composite_elo_dashboard.py`               | deprecated | y   | 2025-12-20    |  256 |
| `ai-service/scripts/db_to_training_npz.py`                    | deprecated | y   | 2025-12-23    |  502 |
| `ai-service/scripts/elo_dashboard.py`                         | deprecated | y   | 2025-12-21    |  212 |
| `ai-service/scripts/model_archival_daemon.py`                 | deprecated | y   | 2025-12-26    |  414 |
| `ai-service/scripts/pipeline_dashboard.py`                    | deprecated | n   | 2025-12-26    |  124 |
| `ai-service/scripts/quick_gauntlet.py`                        | deprecated | y   | 2026-01-21    |  179 |
| `ai-service/scripts/run_parity_promotion_gate.py`             | deprecated | y   | 2025-12-20    |  366 |
| `ai-service/scripts/run_tournament.py`                        | deprecated | y   | 2025-12-20    |  528 |
| `ai-service/scripts/run_vast_gauntlet.py`                     | deprecated | y   | 2025-12-20    |  106 |
| `ai-service/scripts/unified_data_sync.py`                     | deprecated | y   | 2025-12-27    |  315 |
| `ai-service/scripts/aggregate_games.py`                       | monitoring | n   | 2025-12-20    |   85 |
| `ai-service/scripts/apply_tier_promotion_plan.py`             | monitoring | y   | 2025-12-20    |  449 |
| `ai-service/scripts/audit_event_wiring.py`                    | monitoring | y   | 2025-12-28    |  285 |
| `ai-service/scripts/auto_convert_gumbel.py`                   | monitoring | y   | 2025-12-21    |  213 |
| `ai-service/scripts/auto_retrain.py`                          | monitoring | y   | 2026-01-11    |  545 |
| `ai-service/scripts/benchmark_ai_memory.py`                   | monitoring | y   | 2026-04-02    |  619 |
| `ai-service/scripts/benchmark_engine.py`                      | monitoring | n   | 2025-12-20    |   97 |
| `ai-service/scripts/benchmark_nnue_models.py`                 | monitoring | n   | 2025-12-26    |   91 |
| `ai-service/scripts/bootstrap_ladder_artifacts.py`            | monitoring | y   | 2025-12-26    |  377 |
| `ai-service/scripts/calibrate_elo_system.py`                  | monitoring | y   | 2025-12-29    |  762 |
| `ai-service/scripts/chunked_jsonl_converter.py`               | monitoring | y   | 2026-01-14    |  588 |
| `ai-service/scripts/cleanup_useless_replay_dbs.py`            | monitoring | y   | 2025-12-20    |  291 |
| `ai-service/scripts/compare_at_move.py`                       | monitoring | y   | 2025-12-22    |  135 |
| `ai-service/scripts/convert_selfplay_to_state_pool.py`        | monitoring | y   | 2025-12-20    |  230 |
| `ai-service/scripts/cpu_capture_check.py`                     | monitoring | n   | 2025-12-22    |  134 |
| `ai-service/scripts/data_quality_monitor.py`                  | monitoring | y   | 2025-12-20    |  739 |
| `ai-service/scripts/db_health_check.py`                       | monitoring | y   | 2025-12-20    |  348 |
| `ai-service/scripts/debug_rng_tiebreak.py`                    | monitoring | n   | 2025-12-29    |   95 |
| `ai-service/scripts/diagnose_non_termination.py`              | monitoring | n   | 2025-12-22    |  183 |
| `ai-service/scripts/direct_parity_check.py`                   | monitoring | y   | 2025-12-26    |  189 |
| `ai-service/scripts/dlq_dashboard.py`                         | monitoring | y   | 2026-01-12    |  409 |
| `ai-service/scripts/elo_alerts.py`                            | monitoring | y   | 2025-12-26    |  259 |
| `ai-service/scripts/elo_leaderboard.py`                       | monitoring | y   | 2025-12-21    |  191 |
| `ai-service/scripts/elo_monitor.py`                           | monitoring | y   | 2025-12-26    |  614 |
| `ai-service/scripts/elo_progress_monitor.py`                  | monitoring | y   | 2025-12-26    |  148 |
| `ai-service/scripts/elo_progress_report.py`                   | monitoring | y   | 2026-03-05    |  376 |
| `ai-service/scripts/elo_status_dashboard.py`                  | monitoring | y   | 2025-12-26    |  126 |
| `ai-service/scripts/elo_velocity_report.py`                   | monitoring | y   | 2026-01-14    |  245 |
| `ai-service/scripts/eval_cnn_policy.py`                       | monitoring | y   | 2026-01-12    |  389 |
| `ai-service/scripts/evaluate_ai_models.py`                    | monitoring | y   | 2025-12-26    | 1212 |
| `ai-service/scripts/find_capture_divergence.py`               | monitoring | y   | 2025-12-20    |  224 |
| `ai-service/scripts/find_golden_candidates.py`                | monitoring | y   | 2025-12-26    |  305 |
| `ai-service/scripts/fix_database_integrity.py`                | monitoring | y   | 2025-12-20    |  467 |
| `ai-service/scripts/generate_improvement_report.py`           | monitoring | y   | 2026-01-15    |  422 |
| `ai-service/scripts/generate_statistical_report.py`           | monitoring | n   | 2026-01-23    |  743 |
| `ai-service/scripts/inspect_nn_checkpoint.py`                 | monitoring | y   | 2025-12-24    |  230 |
| `ai-service/scripts/inspect_replay_db.py`                     | monitoring | y   | 2025-12-29    |  273 |
| `ai-service/scripts/list_golden_games.py`                     | monitoring | n   | 2025-12-20    |   87 |
| `ai-service/scripts/merge_and_validate_games.py`              | monitoring | y   | 2025-12-28    |  896 |
| `ai-service/scripts/migrate_checkpoints.py`                   | monitoring | y   | 2025-12-24    |  607 |
| `ai-service/scripts/migrate_game_db.py`                       | monitoring | y   | 2025-12-20    |  250 |
| `ai-service/scripts/migrate_inline_moves_to_table.py`         | monitoring | y   | 2026-01-13    |  364 |
| `ai-service/scripts/minimal_hex8_debug.py`                    | monitoring | n   | 2025-12-26    |  105 |
| `ai-service/scripts/model_culling.py`                         | monitoring | y   | 2026-01-12    |  449 |
| `ai-service/scripts/model_maintenance.py`                     | monitoring | y   | 2025-12-20    |  257 |
| `ai-service/scripts/model_registry_cli.py`                    | monitoring | y   | 2025-12-26    |  678 |
| `ai-service/scripts/monitor_curriculum_allocation.py`         | monitoring | y   | 2026-01-06    |  322 |
| `ai-service/scripts/monitor_elo.py`                           | monitoring | y   | 2026-01-10    |  189 |
| `ai-service/scripts/monitor_improvement.py`                   | monitoring | y   | 2026-01-12    |  286 |
| `ai-service/scripts/normalize_termination_reason.py`          | monitoring | y   | 2025-12-20    |  199 |
| `ai-service/scripts/p2p_monitor_check.py`                     | monitoring | n   | 2026-01-24    |  109 |
| `ai-service/scripts/pipeline_health.py`                       | monitoring | y   | 2026-03-03    |  353 |
| `ai-service/scripts/pipeline_watchdog.py`                     | monitoring | y   | 2026-04-03    |  336 |
| `ai-service/scripts/quarantine_non_canonical_games.py`        | monitoring | y   | 2025-12-29    |  318 |
| `ai-service/scripts/quick_benchmark.py`                       | monitoring | n   | 2025-12-26    |  184 |
| `ai-service/scripts/quick_eval_gmo_v2.py`                     | monitoring | y   | 2026-01-12    |  146 |
| `ai-service/scripts/quick_hex8_test.py`                       | monitoring | n   | 2025-12-26    |   78 |
| `ai-service/scripts/quick_model_bench.py`                     | monitoring | y   | 2025-12-26    |  156 |
| `ai-service/scripts/run_automated_parity_gate.py`             | monitoring | y   | 2025-12-21    |  494 |
| `ai-service/scripts/run_bulk_canonical_generation.py`         | monitoring | y   | 2025-12-26    |  473 |
| `ai-service/scripts/run_canonical_selfplay_parity_gate.py`    | monitoring | y   | 2025-12-26    |  826 |
| `ai-service/scripts/run_full_tier_gating.py`                  | monitoring | y   | 2025-12-26    |  425 |
| `ai-service/scripts/run_generation_tournaments.py`            | monitoring | y   | 2026-01-15    |  351 |
| `ai-service/scripts/run_initial_tournaments.py`               | monitoring | y   | 2025-12-29    |  189 |
| `ai-service/scripts/run_multiconfig_nnue_training.py`         | monitoring | y   | 2025-12-26    |  557 |
| `ai-service/scripts/run_parity_healthcheck.py`                | monitoring | y   | 2025-12-21    |  437 |
| `ai-service/scripts/run_profile_tournament.py`                | monitoring | y   | 2026-01-12    |  407 |
| `ai-service/scripts/run_self_play_soak.py`                    | monitoring | y   | 2026-01-01    | 3870 |
| `ai-service/scripts/run_simple_soft_targets.py`               | monitoring | n   | 2025-12-24    |  159 |
| `ai-service/scripts/run_weight_sensitivity_test.py`           | monitoring | y   | 2026-01-01    |  495 |
| `ai-service/scripts/show_all_moves_29256.py`                  | monitoring | n   | 2025-12-26    |   20 |
| `ai-service/scripts/show_moves.py`                            | monitoring | y   | 2025-12-22    |   52 |
| `ai-service/scripts/simple_canonical_gen.py`                  | monitoring | y   | 2025-12-21    |  163 |
| `ai-service/scripts/slurm_smoke_test.py`                      | monitoring | y   | 2025-12-20    |  226 |
| `ai-service/scripts/tag_games_parity_status.py`               | monitoring | y   | 2025-12-21    |  456 |
| `ai-service/scripts/test_ai_balance_parallel.py`              | monitoring | n   | 2025-12-26    |  165 |
| `ai-service/scripts/test_color_disconnected.py`               | monitoring | n   | 2025-12-26    |  154 |
| `ai-service/scripts/test_consolidate.py`                      | monitoring | n   | 2026-01-01    |   58 |
| `ai-service/scripts/test_health_report.py`                    | monitoring | y   | 2026-04-03    |   98 |
| `ai-service/scripts/test_import_expansion.py`                 | monitoring | n   | 2025-12-20    |   68 |
| `ai-service/scripts/track_elo_improvement.py`                 | monitoring | y   | 2025-12-20    |  281 |
| `ai-service/scripts/track_elo_progress.py`                    | monitoring | y   | 2026-01-12    |  186 |
| `ai-service/scripts/training_completion_watcher.py`           | monitoring | y   | 2025-12-26    |  262 |
| `ai-service/scripts/training_dashboard.py`                    | monitoring | y   | 2026-04-10    |  152 |
| `ai-service/scripts/truncate_games_at_victory.py`             | monitoring | y   | 2026-01-23    |  207 |
| `ai-service/scripts/upgrade_jsonl_schema.py`                  | monitoring | y   | 2025-12-20    |  296 |
| `ai-service/scripts/validate_canonical_training_sources.py`   | monitoring | y   | 2025-12-20    |  128 |
| `ai-service/scripts/validate_hex8_batch.py`                   | monitoring | n   | 2025-12-21    |  145 |
| `ai-service/scripts/validate_selfplay_data.py`                | monitoring | y   | 2025-12-22    |  299 |
| `ai-service/scripts/validate_training_data.py`                | monitoring | y   | 2025-12-22    |  550 |
| `ai-service/scripts/validate_training_db.py`                  | monitoring | y   | 2026-04-10    |  269 |
| `ai-service/scripts/verify_canonical_db.py`                   | monitoring | y   | 2025-12-21    |  237 |
| `ai-service/scripts/ab_test_gpu_training_quality.py`          | training   | y   | 2025-12-26    |  514 |
| `ai-service/scripts/ab_test_policy_models.py`                 | training   | y   | 2025-12-26    |  689 |
| `ai-service/scripts/ai_inference_smoke.py`                    | training   | y   | 2026-04-10    |  184 |
| `ai-service/scripts/analyze_descent_vs_mcts_results.py`       | training   | y   | 2025-12-26    |  165 |
| `ai-service/scripts/analyze_difficulty_calibration.py`        | training   | y   | 2025-12-20    |  806 |
| `ai-service/scripts/analyze_game_mechanics.py`                | training   | y   | 2026-01-12    |  522 |
| `ai-service/scripts/analyze_parity_failures.py`               | training   | y   | 2025-12-20    |  235 |
| `ai-service/scripts/analyze_recovery_across_games.py`         | training   | y   | 2026-01-12    |  513 |
| `ai-service/scripts/analyze_recovery_eligibility.py`          | training   | n   | 2025-12-20    |  249 |
| `ai-service/scripts/analyze_recovery_opportunities.py`        | training   | y   | 2026-01-12    |  440 |
| `ai-service/scripts/analyze_smoke_test.py`                    | training   | n   | 2025-12-24    |   69 |
| `ai-service/scripts/analyze_surprise_from_npz.py`             | training   | y   | 2025-12-26    |  299 |
| `ai-service/scripts/analyze_surprise_metric.py`               | training   | y   | 2025-12-26    |  347 |
| `ai-service/scripts/analyze_training_run.py`                  | training   | y   | 2025-12-24    |  432 |
| `ai-service/scripts/analyze_weight_sensitivity.py`            | training   | y   | 2025-12-20    |  288 |
| `ai-service/scripts/apply_cull_manifest.py`                   | training   | n   | 2025-12-20    |   35 |
| `ai-service/scripts/archive_incompatible_models.py`           | training   | y   | 2025-12-23    |  320 |
| `ai-service/scripts/archive_stale_elo_entries.py`             | training   | y   | 2026-01-12    |  413 |
| `ai-service/scripts/audit_parity_compliance.py`               | training   | y   | 2025-12-20    |  199 |
| `ai-service/scripts/audit_stale_elo.py`                       | training   | y   | 2026-01-12    |  441 |
| `ai-service/scripts/auto_export_training_data.py`             | training   | y   | 2026-02-26    |  458 |
| `ai-service/scripts/backfill_elo_ratings.py`                  | training   | y   | 2026-01-14    |  286 |
| `ai-service/scripts/backfill_generation_tracking.py`          | training   | y   | 2026-03-03    |  238 |
| `ai-service/scripts/backfill_heuristics.py`                   | training   | y   | 2026-01-12    |  452 |
| `ai-service/scripts/backfill_snapshots.py`                    | training   | y   | 2025-12-20    |  320 |
| `ai-service/scripts/benchmark_gnn.py`                         | training   | y   | 2025-12-26    |  356 |
| `ai-service/scripts/benchmark_gpu_ai_parity.py`               | training   | y   | 2025-12-20    |  722 |
| `ai-service/scripts/benchmark_gpu_cpu.py`                     | training   | y   | 2025-12-21    |  626 |
| `ai-service/scripts/benchmark_gumbel_gpu.py`                  | training   | y   | 2025-12-20    |  184 |
| `ai-service/scripts/benchmark_search_algorithms.py`           | training   | y   | 2026-01-21    |  620 |
| `ai-service/scripts/benchmark_search_board_large_board.py`    | training   | n   | 2025-12-20    |  257 |
| `ai-service/scripts/build_canonical_dataset.py`               | training   | y   | 2025-12-20    |  237 |
| `ai-service/scripts/check_board_after_13.py`                  | training   | n   | 2025-12-20    |  130 |
| `ai-service/scripts/check_canonical_phase_history.py`         | training   | y   | 2025-12-20    |   78 |
| `ai-service/scripts/check_database_integrity.py`              | training   | y   | 2026-01-12    |  374 |
| `ai-service/scripts/check_production_candidates.py`           | training   | y   | 2025-12-21    |  155 |
| `ai-service/scripts/check_ts_python_replay_parity.py`         | training   | y   | 2026-01-13    | 2024 |
| `ai-service/scripts/classify_parity_structural_mismatches.py` | training   | y   | 2025-12-25    |  298 |
| `ai-service/scripts/cleanup_corrupt_games.py`                 | training   | y   | 2025-12-29    |  409 |
| `ai-service/scripts/cleanup_games_without_moves.py`           | training   | y   | 2025-12-28    |  405 |
| `ai-service/scripts/cleanup_models.py`                        | training   | y   | 2026-02-17    |  348 |
| `ai-service/scripts/compare_architectures.py`                 | training   | y   | 2025-12-30    |  728 |
| `ai-service/scripts/compare_gpu_cpu_states.py`                | training   | n   | 2025-12-20    |  143 |
| `ai-service/scripts/compare_models_elo.py`                    | training   | y   | 2025-12-26    |  659 |
| `ai-service/scripts/compare_nnue_models.py`                   | training   | y   | 2026-01-12    |  338 |
| `ai-service/scripts/compare_soft_hard_policy.py`              | training   | y   | 2025-12-24    |  192 |
| `ai-service/scripts/consolidate_elo_databases.py`             | training   | y   | 2025-12-26    |  334 |
| `ai-service/scripts/consolidate_elo_entries.py`               | training   | y   | 2026-01-12    |  412 |
| `ai-service/scripts/consolidate_jsonl_databases.py`           | training   | y   | 2026-01-25    |  780 |
| `ai-service/scripts/consolidate_selfplay.py`                  | training   | n   | 2026-01-07    |  154 |
| `ai-service/scripts/convert_jsonl_to_npz.py`                  | training   | y   | 2025-12-26    |  143 |
| `ai-service/scripts/convert_npz_to_hdf5.py`                   | training   | y   | 2025-12-20    |  491 |
| `ai-service/scripts/count_actual_recovery_opportunities.py`   | training   | n   | 2025-12-26    |  219 |
| `ai-service/scripts/create_synthetic_soft_targets.py`         | training   | y   | 2025-12-24    |  163 |
| `ai-service/scripts/crossboard_tier_orchestrator.py`          | training   | y   | 2025-12-21    |  552 |
| `ai-service/scripts/data_prep.py`                             | training   | y   | 2026-02-12    | 3061 |
| `ai-service/scripts/debug_game_evolution.py`                  | training   | n   | 2025-12-26    |   83 |
| `ai-service/scripts/debug_initial_state.py`                   | training   | n   | 2025-12-26    |   63 |
| `ai-service/scripts/debug_move_selection.py`                  | training   | n   | 2025-12-26    |   90 |
| `ai-service/scripts/debug_score_ranges.py`                    | training   | n   | 2025-12-26    |   56 |
| `ai-service/scripts/debug_swap_decisions.py`                  | training   | n   | 2025-12-26    |  106 |
| `ai-service/scripts/detect_biased_games.py`                   | training   | y   | 2025-12-26    |  418 |
| `ai-service/scripts/detect_gpu_cpu_divergence.py`             | training   | y   | 2025-12-21    |  156 |
| `ai-service/scripts/diagnose_heuristic_landscape.py`          | training   | n   | 2025-12-20    |  100 |
| `ai-service/scripts/diagnose_policy_equivalence.py`           | training   | y   | 2025-12-20    |  502 |
| `ai-service/scripts/diagnose_weight_application.py`           | training   | n   | 2026-04-02    |  288 |
| `ai-service/scripts/distill_cnn_to_nnue.py`                   | training   | y   | 2026-01-12    |  514 |
| `ai-service/scripts/distill_to_nnue.py`                       | training   | y   | 2025-12-26    |  733 |
| `ai-service/scripts/elo_metrics_exporter.py`                  | training   | y   | 2025-12-26    |  463 |
| `ai-service/scripts/ensemble_gauntlet.py`                     | training   | y   | 2026-01-18    |  216 |
| `ai-service/scripts/ensemble_models.py`                       | training   | y   | 2025-12-24    |  276 |
| `ai-service/scripts/estimate_elo.py`                          | training   | y   | 2025-12-21    |  366 |
| `ai-service/scripts/eval_canonical_models.py`                 | training   | y   | 2026-01-04    |  145 |
| `ai-service/scripts/evaluate_gnn_model.py`                    | training   | y   | 2026-01-23    |  212 |
| `ai-service/scripts/evaluate_nn_models.py`                    | training   | y   | 2026-01-21    |  519 |
| `ai-service/scripts/evaluate_nnue.py`                         | training   | y   | 2025-12-23    |  408 |
| `ai-service/scripts/evaluate_v4_models.py`                    | training   | y   | 2025-12-22    |  197 |
| `ai-service/scripts/export_for_all_architectures.py`          | training   | y   | 2026-04-06    |  316 |
| `ai-service/scripts/export_jsonl_db_to_npz.py`                | training   | y   | 2026-01-12    |  340 |
| `ai-service/scripts/export_replay_dataset.py`                 | training   | y   | 2026-04-10    | 3061 |
| `ai-service/scripts/export_replay_dataset_parallel.py`        | training   | y   | 2026-03-09    |  485 |
| `ai-service/scripts/extract_golden_games.py`                  | training   | y   | 2025-12-26    |  180 |
| `ai-service/scripts/extract_hex_late_game.py`                 | training   | y   | 2026-01-12    |  343 |
| `ai-service/scripts/filter_training_data.py`                  | training   | y   | 2025-12-26    |  519 |
| `ai-service/scripts/fix_elo_database.py`                      | training   | y   | 2026-01-12    |  504 |
| `ai-service/scripts/fix_hex_checkpoint_metadata.py`           | training   | y   | 2026-01-12    |  186 |
| `ai-service/scripts/fix_model_checksums.py`                   | training   | y   | 2026-01-21    |  310 |
| `ai-service/scripts/fix_model_naming.py`                      | training   | y   | 2026-01-12    |  410 |
| `ai-service/scripts/game_analysis.py`                         | training   | y   | 2025-12-20    |  930 |
| `ai-service/scripts/gauntlet_to_elo.py`                       | training   | y   | 2026-01-12    |  329 |
| `ai-service/scripts/generate_antiheuristic_data.py`           | training   | y   | 2025-12-24    |  620 |
| `ai-service/scripts/generate_axis_aligned_profiles.py`        | training   | y   | 2025-12-20    |  219 |
| `ai-service/scripts/generate_blended_selfplay.py`             | training   | y   | 2025-12-26    |  717 |
| `ai-service/scripts/generate_canonical_selfplay.py`           | training   | y   | 2025-12-26    | 1351 |
| `ai-service/scripts/generate_forced_elimination_fixtures.py`  | training   | y   | 2026-04-02    |  500 |
| `ai-service/scripts/generate_gpu_training_data.py`            | training   | y   | 2026-01-12    |  246 |
| `ai-service/scripts/generate_gumbel_selfplay.py`              | training   | y   | 2026-03-29    | 1039 |
| `ai-service/scripts/generate_mcts_data_56ch.py`               | training   | y   | 2025-12-26    |  191 |
| `ai-service/scripts/generate_opening_book.py`                 | training   | y   | 2026-01-12    |  661 |
| `ai-service/scripts/generate_parity_vectors.py`               | training   | y   | 2025-12-26    |  568 |
| `ai-service/scripts/generate_search_labeled_data.py`          | training   | y   | 2025-12-26    |  391 |
| `ai-service/scripts/generate_territory_fixtures.py`           | training   | y   | 2025-12-20    |  244 |
| `ai-service/scripts/gmo_eval_strong.py`                       | training   | y   | 2026-01-12    |  332 |
| `ai-service/scripts/gmo_integration.py`                       | training   | y   | 2026-01-12    |  671 |
| `ai-service/scripts/gmo_post_training_eval.py`                | training   | y   | 2025-12-23    |  164 |
| `ai-service/scripts/gmo_uncertainty_calibration.py`           | training   | y   | 2025-12-26    |  401 |
| `ai-service/scripts/holdout_validation.py`                    | training   | y   | 2025-12-26    |  973 |
| `ai-service/scripts/hot_data_path.py`                         | training   | y   | 2025-12-26    |  455 |
| `ai-service/scripts/hyperparameter_ab_testing.py`             | training   | y   | 2025-12-27    |  652 |
| `ai-service/scripts/hyperparameter_tuning.py`                 | training   | y   | 2025-12-20    |  467 |
| `ai-service/scripts/import_gpu_selfplay_to_db.py`             | training   | y   | 2025-12-28    |  809 |
| `ai-service/scripts/initialize_elo_database.py`               | training   | y   | 2025-12-29    |  132 |
| `ai-service/scripts/iterative_selfplay_train.py`              | training   | y   | 2025-12-21    |  456 |
| `ai-service/scripts/jsonl_to_npz.py`                          | training   | y   | 2025-12-26    | 1887 |
| `ai-service/scripts/memory_guard.py`                          | training   | n   | 2026-04-03    |   45 |
| `ai-service/scripts/merge_balanced_values.py`                 | training   | y   | 2025-12-20    |  154 |
| `ai-service/scripts/merge_game_dbs.py`                        | training   | y   | 2025-12-26    |  409 |
| `ai-service/scripts/merge_game_statistics_reports.py`         | training   | y   | 2025-12-26    |  525 |
| `ai-service/scripts/merge_trained_weights.py`                 | training   | y   | 2025-12-26    |  389 |
| `ai-service/scripts/merge_training_datasets.py`               | training   | y   | 2025-12-23    |  146 |
| `ai-service/scripts/migrate_elo_to_unified.py`                | training   | y   | 2025-12-21    |  463 |
| `ai-service/scripts/migrate_model_names.py`                   | training   | y   | 2025-12-24    |  303 |
| `ai-service/scripts/mine_critical_positions.py`               | training   | y   | 2025-12-20    |  528 |
| `ai-service/scripts/model_cleanup.py`                         | training   | y   | 2025-12-20    |  233 |
| `ai-service/scripts/model_compression.py`                     | training   | y   | 2025-12-26    |  611 |
| `ai-service/scripts/model_lineage.py`                         | training   | y   | 2025-12-20    |  482 |
| `ai-service/scripts/model_regression_tests.py`                | training   | y   | 2026-01-21    |  462 |
| `ai-service/scripts/multi_game_gumbel_selfplay.py`            | training   | y   | 2025-12-24    |  210 |
| `ai-service/scripts/multi_objective_optimizer.py`             | training   | y   | 2025-12-20    |  567 |
| `ai-service/scripts/per_move_jsonl_to_npz.py`                 | training   | y   | 2025-12-26    |  235 |
| `ai-service/scripts/periodic_harvest.py`                      | training   | y   | 2025-12-20    |  491 |
| `ai-service/scripts/policy_gauntlet.py`                       | training   | y   | 2026-01-21    |  223 |
| `ai-service/scripts/post_training_pipeline.py`                | training   | y   | 2025-12-20    |  440 |
| `ai-service/scripts/prioritized_replay.py`                    | training   | y   | 2025-12-20    |  539 |
| `ai-service/scripts/probe_plateau_diagnostics.py`             | training   | y   | 2025-12-20    |  355 |
| `ai-service/scripts/prune_models.py`                          | training   | y   | 2025-12-26    |  416 |
| `ai-service/scripts/prune_weak_models.py`                     | training   | y   | 2025-12-22    |  204 |
| `ai-service/scripts/quantize_nnue.py`                         | training   | y   | 2026-01-12    |  275 |
| `ai-service/scripts/quarantine_bad_games.py`                  | training   | y   | 2025-12-20    |  354 |
| `ai-service/scripts/quarantine_old_data.py`                   | training   | y   | 2025-12-24    |  244 |
| `ai-service/scripts/reanalyze_mcts_policy.py`                 | training   | y   | 2026-01-23    |  479 |
| `ai-service/scripts/reanalyze_replay_dataset.py`              | training   | y   | 2025-12-26    |  467 |
| `ai-service/scripts/reconcile_elo_ids.py`                     | training   | y   | 2026-02-17    |  407 |
| `ai-service/scripts/refresh_npz.py`                           | training   | n   | 2026-01-29    |   39 |
| `ai-service/scripts/register_composite_models.py`             | training   | y   | 2026-01-23    |  312 |
| `ai-service/scripts/regression_test_pipeline.py`              | training   | y   | 2026-04-05    | 1272 |
| `ai-service/scripts/replay_infer_termination.py`              | training   | y   | 2026-01-12    |  451 |
| `ai-service/scripts/run_canonical_diverse_selfplay.py`        | training   | y   | 2025-12-26    |  319 |
| `ai-service/scripts/run_canonical_guards.py`                  | training   | y   | 2025-12-26    |  279 |
| `ai-service/scripts/run_cmaes_optimization.py`                | training   | y   | 2026-04-02    | 2916 |
| `ai-service/scripts/run_crossboard_tier_gating.py`            | training   | y   | 2025-12-20    |  265 |
| `ai-service/scripts/run_descent_vs_mcts_experiment.py`        | training   | y   | 2026-01-12    |  349 |
| `ai-service/scripts/run_eval_tournaments.py`                  | training   | y   | 2025-12-20    |  526 |
| `ai-service/scripts/run_genetic_heuristic_search.py`          | training   | y   | 2025-12-20    |  559 |
| `ai-service/scripts/run_gpu_canonical_parity_gate.py`         | training   | y   | 2025-12-20    |  464 |
| `ai-service/scripts/run_gpu_cmaes.py`                         | training   | y   | 2025-12-20    |  581 |
| `ai-service/scripts/run_gpu_persona_tournament.py`            | training   | y   | 2025-12-23    |  767 |
| `ai-service/scripts/run_gpu_selfplay.py`                      | training   | n   | 2026-02-03    | 2376 |
| `ai-service/scripts/run_gpu_tree_parity_gate.py`              | training   | y   | 2026-01-23    |  427 |
| `ai-service/scripts/run_heuristic_experiment.py`              | training   | y   | 2026-04-02    |  823 |
| `ai-service/scripts/run_improvement_loop.py`                  | training   | y   | 2025-12-26    | 2773 |
| `ai-service/scripts/run_invariant_soak.py`                    | training   | y   | 2025-12-20    |  171 |
| `ai-service/scripts/run_model_elo_tournament.py`              | training   | y   | 2026-01-21    | 2960 |
| `ai-service/scripts/run_nn_training_baseline.py`              | training   | y   | 2025-12-27    |  433 |
| `ai-service/scripts/run_parallel_self_play.py`                | training   | y   | 2026-01-12    |  643 |
| `ai-service/scripts/run_parity_and_history_gate.py`           | training   | y   | 2025-12-26    |  246 |
| `ai-service/scripts/run_parity_validation.py`                 | training   | y   | 2026-01-12    |  279 |
| `ai-service/scripts/run_persona_tournament.py`                | training   | y   | 2025-12-26    |  519 |
| `ai-service/scripts/run_policy_selfplay.py`                   | training   | y   | 2025-12-29    |  155 |
| `ai-service/scripts/run_soft_fast.py`                         | training   | n   | 2025-12-24    |   68 |
| `ai-service/scripts/run_soft_targets_gpu.py`                  | training   | n   | 2025-12-24    |   97 |
| `ai-service/scripts/run_strength_regression_gate.py`          | training   | y   | 2026-01-01    |  631 |
| `ai-service/scripts/run_tier_evaluation.py`                   | training   | y   | 2025-12-20    |  168 |
| `ai-service/scripts/run_tier_gate.py`                         | training   | y   | 2025-12-26    |  553 |
| `ai-service/scripts/run_tier_perf_benchmark.py`               | training   | y   | 2025-12-20    |  199 |
| `ai-service/scripts/run_tier_training_pipeline.py`            | training   | y   | 2025-12-26    |  540 |
| `ai-service/scripts/sanity_check_multiboard_eval.py`          | training   | y   | 2025-12-20    |  311 |
| `ai-service/scripts/scan_canonical_phase_dbs.py`              | training   | y   | 2025-12-20    |  203 |
| `ai-service/scripts/seed_generation_tracking.py`              | training   | y   | 2026-01-23    |  198 |
| `ai-service/scripts/select_best_checkpoint_by_elo.py`         | training   | y   | 2025-12-26    |  419 |
| `ai-service/scripts/selfplay.py`                              | training   | n   | 2025-12-26    |  332 |
| `ai-service/scripts/td_error_prioritization.py`               | training   | y   | 2026-01-12    |  317 |
| `ai-service/scripts/test_ai_balance.py`                       | training   | y   | 2025-12-26    |  231 |
| `ai-service/scripts/test_descent_debug.py`                    | training   | n   | 2025-12-26    |   75 |
| `ai-service/scripts/test_gpu_cpu_parity.py`                   | training   | y   | 2025-12-23    |  506 |
| `ai-service/scripts/test_gpu_data_generation.py`              | training   | y   | 2026-01-12    |  661 |
| `ai-service/scripts/test_heuristic_balance.py`                | training   | y   | 2025-12-26    |   78 |
| `ai-service/scripts/test_hex8_geometry.py`                    | training   | n   | 2025-12-20    |  106 |
| `ai-service/scripts/test_hex8_territory_bug.py`               | training   | n   | 2025-12-20    |   68 |
| `ai-service/scripts/train_enhanced.py`                        | training   | y   | 2026-01-12    |  371 |
| `ai-service/scripts/train_nnue.py`                            | training   | y   | 2025-12-29    | 5525 |
| `ai-service/scripts/train_nnue_policy.py`                     | training   | y   | 2025-12-26    | 1306 |
| `ai-service/scripts/train_square8_2p_optimized.py`            | training   | y   | 2026-01-12    |  312 |
| `ai-service/scripts/training_preflight_check.py`              | training   | y   | 2026-01-12    | 1131 |
| `ai-service/scripts/tune_hyperparameters.py`                  | training   | y   | 2025-12-26    |  663 |
| `ai-service/scripts/upgrade_phase_annotations.py`             | training   | y   | 2025-12-28    |  410 |
| `ai-service/scripts/validate_and_promote_weights.py`          | training   | y   | 2026-03-05    |  505 |
| `ai-service/scripts/validate_databases.py`                    | training   | y   | 2026-01-12    |  304 |
| `ai-service/scripts/validate_db_games.py`                     | training   | y   | 2025-12-25    |  285 |
| `ai-service/scripts/validate_gpu_mcts_data.py`                | training   | y   | 2025-12-24    |  212 |
| `ai-service/scripts/validate_mcts_policy.py`                  | training   | n   | 2025-12-20    |  194 |
| `ai-service/scripts/validate_minimax_policy.py`               | training   | n   | 2025-12-19    |  200 |
| `ai-service/scripts/validate_npz_encoding.py`                 | training   | y   | 2026-01-12    |  284 |
| `ai-service/scripts/validate_phase_recording.py`              | training   | y   | 2026-01-12    |  311 |
| `ai-service/scripts/validate_selfplay_games.py`               | training   | y   | 2026-01-23    |  268 |
| `ai-service/scripts/validate_trained_weights.py`              | training   | y   | 2025-12-23    |  336 |
| `ai-service/scripts/validate_training_integrations.py`        | training   | n   | 2025-12-20    |  345 |
| `ai-service/scripts/verify_feedback_loop.py`                  | training   | n   | 2025-12-26    |  105 |
| `ai-service/scripts/verify_freshness_check.py`                | training   | n   | 2025-12-26    |  215 |
| `ai-service/scripts/verify_position_mismatch.py`              | training   | n   | 2025-12-20    |  100 |
| `ai-service/scripts/archive_processed_jsonl.py`               | utility    | y   | 2025-12-20    |  237 |
| `ai-service/scripts/check_gpu_before_51.py`                   | utility    | n   | 2025-12-20    |   74 |
| `ai-service/scripts/check_layer_violations.py`                | utility    | n   | 2025-12-27    |   53 |
| `ai-service/scripts/cleanup_victory_types.py`                 | utility    | y   | 2025-12-26    |  429 |
| `ai-service/scripts/convert_single.py`                        | utility    | n   | 2025-12-26    |  105 |
| `ai-service/scripts/generate_event_reference.py`              | utility    | y   | 2025-12-29    |  420 |
| `ai-service/scripts/gmo_ablation_study.py`                    | utility    | y   | 2026-01-12    |  412 |
| `ai-service/scripts/gmo_hyperparam_sweep.py`                  | utility    | y   | 2026-01-12    |  556 |
| `ai-service/scripts/gpu_step_trace.py`                        | utility    | y   | 2025-12-22    |   81 |
| `ai-service/scripts/migrate_jsonl_moves.py`                   | utility    | y   | 2026-01-23    |  322 |
| `ai-service/scripts/neural_architecture_search.py`            | utility    | y   | 2025-12-26    |  885 |
| `ai-service/scripts/populate_initial_states.py`               | utility    | y   | 2026-01-20    |  297 |
| `ai-service/scripts/regenerate_vectors.py`                    | utility    | n   | 2025-12-20    |  346 |
| `ai-service/scripts/slurm_preflight_check.py`                 | utility    | y   | 2025-12-20    |  130 |
