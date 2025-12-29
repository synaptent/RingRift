# Internal Environment Flags (Auto-Extracted)

This appendix lists environment variables referenced in code but not yet described in
`docs/operations/ENVIRONMENT_VARIABLES.md`, `ai-service/docs/ENV_REFERENCE.md`,
`ai-service/docs/ENV_REFERENCE_COMPREHENSIVE.md`, or
`ai-service/docs/AI_SERVICE_ENVIRONMENT_REFERENCE.md`. Entries are auto-extracted and
may be experimental or internal.

Generated: 2025-12-29

| Name                                                   | First Usage (code)                                                          | Notes          |
| ------------------------------------------------------ | --------------------------------------------------------------------------- | -------------- |
| `RINGRIFT_ACCEL_MIN_GAMES`                             | `ai-service/app/training/feedback_accelerator.py:83`                        |                |
| `RINGRIFT_ADAPTIVE_TIMEOUTS`                           | `ai-service/app/config/coordination_defaults.py:1057`                       |                |
| `RINGRIFT_ADMIN_TOKEN`                                 | `ai-service/scripts/p2p/handlers/admin.py:102`                              |                |
| `RINGRIFT_ADVERTISE_`                                  | `ai-service/deploy/setup_node_resilience.sh:22`                             | prefix/pattern |
| `RINGRIFT_ADVERTISE_HOST`                              | `ai-service/deploy/deploy_cluster_resilience.py:21`                         |                |
| `RINGRIFT_ADVERTISE_PORT`                              | `ai-service/deploy/setup_node_resilience.sh:213`                            |                |
| `RINGRIFT_AGGREGATION_INTERVAL`                        | `ai-service/app/config/coordination_defaults.py:1307`                       |                |
| `RINGRIFT_AI_SERVICE`                                  | `ai-service/scripts/ops/maintain_selfplay_load.sh:21`                       |                |
| `RINGRIFT_AI_SERVICE_DIR__`                            | `ai-service/scripts/install_launchd_services.sh:95`                         | prefix/pattern |
| `RINGRIFT_AI_SERVICE_ROOT`                             | `ai-service/scripts/p2p/handlers/file_download.py:61`                       |                |
| `RINGRIFT_ALERT_COOLDOWN`                              | `ai-service/app/config/coordination_defaults.py:734`                        |                |
| `RINGRIFT_ALERT_COOLDOWN_MIN`                          | `ai-service/scripts/monitoring/selfplay_throughput_monitor.sh:14`           |                |
| `RINGRIFT_ALERT_MAX_PER_HOUR`                          | `ai-service/config/cluster_env.sh:21`                                       |                |
| `RINGRIFT_ALERT_MIN_INTERVAL`                          | `ai-service/config/cluster_env.sh:20`                                       |                |
| `RINGRIFT_ALERT_STATE_DIR`                             | `ai-service/scripts/monitoring/cluster_health_check.sh:29`                  |                |
| `RINGRIFT_ALLOW_STALE_TRAINING`                        | `ai-service/scripts/master_loop.py:1652`                                    |                |
| `RINGRIFT_ARBITER_URL`                                 | `ai-service/app/p2p/constants.py:269`                                       |                |
| `RINGRIFT_ARCHIVE_DIR`                                 | `ai-service/app/coordination/maintenance_daemon.py:72`                      |                |
| `RINGRIFT_ARCHIVE_S3_BUCKET`                           | `ai-service/app/coordination/maintenance_daemon.py:80`                      |                |
| `RINGRIFT_ARCHIVE_TO_S3`                               | `ai-service/app/coordination/maintenance_daemon.py:77`                      |                |
| `RINGRIFT_ASYNC_SUBPROCESS_TIMEOUT`                    | `ai-service/app/config/coordination_defaults.py:1530`                       |                |
| `RINGRIFT_AUTO_ASSIGN_ENABLED`                         | `ai-service/app/utils/env_config.py:303`                                    |                |
| `RINGRIFT_AUTO_BATCH_SCALE`                            | `ai-service/app/training/config.py:465`                                     |                |
| `RINGRIFT_AUTO_STREAMING_THRESHOLD_GB`                 | `ai-service/app/training/data_loader_factory.py:36`                         |                |
| `RINGRIFT_AUTO_UPDATE`                                 | `ai-service/scripts/p2p/handlers/admin.py:19`                               |                |
| `RINGRIFT_BANDWIDTH_DB`                                | `ai-service/tests/test_coordination_integration.py:188`                     | test-only      |
| `RINGRIFT_BANDWIDTH_MEASUREMENT_WINDOW`                | `ai-service/app/config/coordination_defaults.py:508`                        |                |
| `RINGRIFT_BATCH_OPERATION_TIMEOUT`                     | `ai-service/app/config/coordination_defaults.py:1548`                       |                |
| `RINGRIFT_BLOCK_PROMOTION_INCOMPLETE`                  | `ai-service/app/config/coordination_defaults.py:190`                        |                |
| `RINGRIFT_BP_CACHE_TTL`                                | `ai-service/app/config/coordination_defaults.py:1256`                       |                |
| `RINGRIFT_BP_COOLDOWN`                                 | `ai-service/app/config/coordination_defaults.py:1259`                       |                |
| `RINGRIFT_BP_QUEUE_CRITICAL`                           | `ai-service/app/config/coordination_defaults.py:1243`                       |                |
| `RINGRIFT_BP_QUEUE_HIGH`                               | `ai-service/app/config/coordination_defaults.py:1242`                       |                |
| `RINGRIFT_BP_QUEUE_LOW`                                | `ai-service/app/config/coordination_defaults.py:1240`                       |                |
| `RINGRIFT_BP_QUEUE_MEDIUM`                             | `ai-service/app/config/coordination_defaults.py:1241`                       |                |
| `RINGRIFT_BP_WEIGHT_DISK`                              | `ai-service/app/config/coordination_defaults.py:1235`                       |                |
| `RINGRIFT_BP_WEIGHT_MEMORY`                            | `ai-service/app/config/coordination_defaults.py:1237`                       |                |
| `RINGRIFT_BP_WEIGHT_QUEUE`                             | `ai-service/app/config/coordination_defaults.py:1233`                       |                |
| `RINGRIFT_BP_WEIGHT_SYNC`                              | `ai-service/app/config/coordination_defaults.py:1236`                       |                |
| `RINGRIFT_BP_WEIGHT_TRAINING`                          | `ai-service/app/config/coordination_defaults.py:1234`                       |                |
| `RINGRIFT_CACHE_CLEANUP_INTERVAL`                      | `ai-service/app/config/coordination_defaults.py:780`                        |                |
| `RINGRIFT_CACHE_DEFAULT_TTL`                           | `ai-service/app/config/coordination_defaults.py:774`                        |                |
| `RINGRIFT_CACHE_MAX_ENTRIES_PER_NODE`                  | `ai-service/app/config/coordination_defaults.py:777`                        |                |
| `RINGRIFT_CACHE_STALE_THRESHOLD`                       | `ai-service/app/config/coordination_defaults.py:783`                        |                |
| `RINGRIFT_CAPACITY_CYCLE_INTERVAL`                     | `ai-service/app/coordination/availability/capacity_planner.py:168`          |                |
| `RINGRIFT_CATALOG_DB`                                  | `ai-service/app/coordination/data_catalog.py:149`                           |                |
| `RINGRIFT_CATALOG_REFRESH_INTERVAL`                    | `ai-service/app/coordination/unified_data_plane_daemon.py:135`              |                |
| `RINGRIFT_CHECKPOINT_TIMEOUT`                          | `ai-service/app/config/coordination_defaults.py:1539`                       |                |
| `RINGRIFT_CIRCUIT_BACKOFF_SECONDS`                     | `ai-service/app/coordination/pipeline_actions.py:100`                       |                |
| `RINGRIFT_CIRCUIT_FAILURE_THRESHOLD`                   | `ai-service/app/coordination/pipeline_actions.py:99`                        |                |
| `RINGRIFT_CLEANUP_COOLDOWN`                            | `ai-service/app/config/coordination_defaults.py:1304`                       |                |
| `RINGRIFT_CLEANUP_THRESHOLD_PERCENT`                   | `ai-service/app/config/coordination_defaults.py:1834`                       |                |
| `RINGRIFT_CLUSTER_CHECK_INTERVAL`                      | `ai-service/app/config/coordination_defaults.py:728`                        |                |
| `RINGRIFT_CLUSTER_DOMAIN`                              | `ai-service/scripts/setup_aws_cluster_proxy.sh:229`                         |                |
| `RINGRIFT_CLUSTER_HEALTH_CACHE_TTL`                    | `ai-service/app/config/coordination_defaults.py:463`                        |                |
| `RINGRIFT_CMAES_`                                      | `ai-service/app/config/training_config.py:41`                               | prefix/pattern |
| `RINGRIFT_CMAES_GENERATIONS`                           | `ai-service/app/config/training_config.py:42`                               |                |
| `RINGRIFT_CMAES_POPULATION_SIZE`                       | `ai-service/app/config/training_config.py:43`                               |                |
| `RINGRIFT_CNN_V2_VERSION`                              | `ai-service/app/training/model_versioning.py:361`                           |                |
| `RINGRIFT_CNN_V3_VERSION`                              | `ai-service/app/training/model_versioning.py:362`                           |                |
| `RINGRIFT_CNN_V4_VERSION`                              | `ai-service/app/training/model_versioning.py:363`                           |                |
| `RINGRIFT_CONNECT_BASE_DELAY`                          | `ai-service/app/config/coordination_defaults.py:679`                        |                |
| `RINGRIFT_CONNECT_MAX_DELAY`                           | `ai-service/app/config/coordination_defaults.py:680`                        |                |
| `RINGRIFT_CONNECT_MAX_RETRIES`                         | `ai-service/app/config/coordination_defaults.py:678`                        |                |
| `RINGRIFT_CONSOLIDATION_INTERVAL`                      | `ai-service/app/coordination/data_consolidation_daemon.py:78`               |                |
| `RINGRIFT_CONSOLIDATION_MIN_GAMES`                     | `ai-service/app/coordination/data_consolidation_daemon.py:82`               |                |
| `RINGRIFT_CONSUMER_MAX_SELFPLAY`                       | `ai-service/app/config/coordination_defaults.py:526`                        |                |
| `RINGRIFT_CONTINUOUS_COOLDOWN`                         | `ai-service/scripts/run_continuous_training.py:31`                          |                |
| `RINGRIFT_CONTINUOUS_ENGINE`                           | `ai-service/scripts/run_continuous_training.py:30`                          |                |
| `RINGRIFT_CONTINUOUS_GAMES`                            | `ai-service/scripts/run_continuous_training.py:29`                          |                |
| `RINGRIFT_COORDINATOR_DEGRADED_COOLDOWN`               | `ai-service/app/config/coordination_defaults.py:1713`                       |                |
| `RINGRIFT_COORDINATOR_DIR`                             | `ai-service/app/coordination/task_coordinator.py:883`                       |                |
| `RINGRIFT_COORDINATOR_HEARTBEAT_STALE_THRESHOLD`       | `ai-service/app/config/coordination_defaults.py:1709`                       |                |
| `RINGRIFT_COORDINATOR_INIT_FAILURE_MAX_RETRIES`        | `ai-service/app/config/coordination_defaults.py:1717`                       |                |
| `RINGRIFT_CPU_TARGET_MAX`                              | `ai-service/app/config/coordination_defaults.py:482`                        |                |
| `RINGRIFT_CPU_TARGET_MIN`                              | `ai-service/app/config/coordination_defaults.py:481`                        |                |
| `RINGRIFT_CRITICAL_GAME_THRESHOLD`                     | `ai-service/app/config/coordination_defaults.py:1766`                       |                |
| `RINGRIFT_CROSS_PROCESS_RETENTION_HOURS`               | `ai-service/app/config/coordination_defaults.py:1811`                       |                |
| `RINGRIFT_CURRICULUM_CHECK_INTERVAL`                   | `ai-service/app/config/coordination_defaults.py:2092`                       |                |
| `RINGRIFT_DAEMON_CHECK_INTERVAL`                       | `ai-service/app/config/coordination_defaults.py:639`                        |                |
| `RINGRIFT_DAEMON_CRITICAL_CHECK_INTERVAL`              | `ai-service/app/config/coordination_defaults.py:1387`                       |                |
| `RINGRIFT_DAEMON_DEPENDENCY_POLL`                      | `ai-service/app/config/coordination_defaults.py:1421`                       |                |
| `RINGRIFT_DAEMON_DEPENDENCY_TIMEOUT`                   | `ai-service/app/config/coordination_defaults.py:1425`                       |                |
| `RINGRIFT_DAEMON_ERROR_BACKOFF_BASE`                   | `ai-service/app/config/coordination_defaults.py:642`                        |                |
| `RINGRIFT_DAEMON_ERROR_BACKOFF_MAX`                    | `ai-service/app/config/coordination_defaults.py:645`                        |                |
| `RINGRIFT_DAEMON_ERROR_RATE_THRESHOLD`                 | `ai-service/app/config/coordination_defaults.py:659`                        |                |
| `RINGRIFT_DAEMON_HEALTH_CHECK_TIMEOUT`                 | `ai-service/app/config/coordination_defaults.py:1409`                       |                |
| `RINGRIFT_DAEMON_HEALTH_TIMEOUT`                       | `ai-service/app/config/coordination_defaults.py:656`                        |                |
| `RINGRIFT_DAEMON_MAX_CONSECUTIVE_ERRORS`               | `ai-service/app/config/coordination_defaults.py:648`                        |                |
| `RINGRIFT_DAEMON_MAX_FAILURES`                         | `ai-service/app/config/coordination_defaults.py:1391`                       |                |
| `RINGRIFT_DAEMON_MAX_PARALLEL_HEALTH`                  | `ai-service/app/config/coordination_defaults.py:1417`                       |                |
| `RINGRIFT_DAEMON_MIN_CYCLES_ERROR_CHECK`               | `ai-service/app/config/coordination_defaults.py:662`                        |                |
| `RINGRIFT_DAEMON_PARALLEL_HEALTH`                      | `ai-service/app/config/coordination_defaults.py:1413`                       |                |
| `RINGRIFT_DAEMON_RESTART_BACKOFF`                      | `ai-service/app/config/coordination_defaults.py:1394`                       |                |
| `RINGRIFT_DAEMON_RESTART_BACKOFF_MAX`                  | `ai-service/app/config/coordination_defaults.py:1397`                       |                |
| `RINGRIFT_DAEMON_SHUTDOWN_GRACE`                       | `ai-service/app/config/coordination_defaults.py:653`                        |                |
| `RINGRIFT_DAEMON_SHUTDOWN_TIMEOUT`                     | `ai-service/app/config/coordination_defaults.py:1404`                       |                |
| `RINGRIFT_DAEMON_STARTUP_TIMEOUT`                      | `ai-service/app/config/coordination_defaults.py:1400`                       |                |
| `RINGRIFT_DAILY_BUDGET_USD`                            | `ai-service/app/coordination/availability/capacity_planner.py:180`          |                |
| `RINGRIFT_DATACENTER_MAX_SELFPLAY`                     | `ai-service/app/config/coordination_defaults.py:528`                        |                |
| `RINGRIFT_DATA_SERVER_PORT`                            | `ai-service/app/config/coordination_defaults.py:940`                        |                |
| `RINGRIFT_DATA_SYNC_SIZE_RATIO`                        | `ai-service/app/coordination/training_data_sync_daemon.py:63`               |                |
| `RINGRIFT_DATA_SYNC_TIMEOUT`                           | `ai-service/app/coordination/training_data_sync_daemon.py:67`               |                |
| `RINGRIFT_DB_PASSWORD`                                 | `tests/unit/runProdPreviewGoNoGo.test.ts:155`                               | test-only      |
| `RINGRIFT_DEBUG`                                       | `ai-service/app/utils/env_config.py:89`                                     |                |
| `RINGRIFT_DEBUG_CAPTURE`                               | `src/shared/engine/core.ts:313`                                             |                |
| `RINGRIFT_DEBUG_ENGINE`                                | `ai-service/app/_game_engine_legacy.py:118`                                 |                |
| `RINGRIFT_DEBUG_MOVEMENT`                              | `src/shared/engine/core.ts:488`                                             |                |
| `RINGRIFT_DEBUG_VALIDATION`                            | `src/shared/engine/validators/utils.ts:26`                                  |                |
| `RINGRIFT_DEFAULT_DOWNLOAD_MBPS`                       | `ai-service/app/config/coordination_defaults.py:512`                        |                |
| `RINGRIFT_DEFAULT_JOB_TIMEOUT`                         | `ai-service/app/config/coordination_defaults.py:1679`                       |                |
| `RINGRIFT_DEFAULT_MAX_RETRIES`                         | `ai-service/app/config/coordination_defaults.py:1563`                       |                |
| `RINGRIFT_DEFAULT_UPLOAD_MBPS`                         | `ai-service/app/config/coordination_defaults.py:511`                        |                |
| `RINGRIFT_DEPLOY_NODES`                                | `ai-service/scripts/deploy_resource_exhaustion_fixes.sh:25`                 |                |
| `RINGRIFT_DESCENT_HEURISTIC_FALLBACK`                  | `ai-service/app/ai/descent_ai.py:186`                                       |                |
| `RINGRIFT_DISABLE_FSM_VALIDATION`                      | `ai-service/scripts/run_lps_ablation.py:51`                                 |                |
| `RINGRIFT_DISABLE_MPS`                                 | `ai-service/app/ai/_neural_net_legacy.py:3151`                              |                |
| `RINGRIFT_DISCOVERY_INTERVAL`                          | `ai-service/app/utils/env_config.py:286`                                    |                |
| `RINGRIFT_DISK_CLEANUP_THRESHOLD`                      | `ai-service/app/config/coordination_defaults.py:1312`                       |                |
| `RINGRIFT_DISK_CRITICAL_THRESHOLD`                     | `ai-service/app/config/coordination_defaults.py:714`                        |                |
| `RINGRIFT_DISK_SPACE`                                  | `ai-service/app/coordination/disk_space_manager_daemon.py:236`              |                |
| `RINGRIFT_DISK_SPACE_CHECK_INTERVAL`                   | `ai-service/scripts/launch_coordinator_disk_manager.py:28`                  |                |
| `RINGRIFT_DISK_SPACE_CRITICAL_THRESHOLD`               | `ai-service/tests/unit/coordination/test_disk_space_manager_daemon.py:999`  | test-only      |
| `RINGRIFT_DISK_SPACE_EMIT_EVENTS`                      | `ai-service/tests/unit/coordination/test_disk_space_manager_daemon.py:1004` | test-only      |
| `RINGRIFT_DISK_SPACE_ENABLE_CLEANUP`                   | `ai-service/tests/unit/coordination/test_disk_space_manager_daemon.py:124`  | test-only      |
| `RINGRIFT_DISK_SPACE_LOG_RETENTION_DAYS`               | `ai-service/tests/unit/coordination/test_disk_space_manager_daemon.py:123`  | test-only      |
| `RINGRIFT_DISK_SPACE_MIN_FREE_GB`                      | `ai-service/tests/unit/coordination/test_disk_space_manager_daemon.py:1001` | test-only      |
| `RINGRIFT_DISK_SPACE_PROACTIVE_THRESHOLD`              | `ai-service/scripts/launch_coordinator_disk_manager.py:29`                  |                |
| `RINGRIFT_DISK_SPACE_TARGET_USAGE`                     | `ai-service/scripts/launch_coordinator_disk_manager.py:30`                  |                |
| `RINGRIFT_DISK_SPACE_WARNING_THRESHOLD`                | `ai-service/tests/unit/coordination/test_disk_space_manager_daemon.py:122`  | test-only      |
| `RINGRIFT_DISK_WARNING_THRESHOLD`                      | `ai-service/app/config/coordination_defaults.py:713`                        |                |
| `RINGRIFT_DISTRIBUTED_BACKEND`                         | `ai-service/app/training/distributed_unified.py:73`                         |                |
| `RINGRIFT_DISTRIBUTION_RETRY_INTERVAL`                 | `ai-service/app/config/coordination_defaults.py:180`                        |                |
| `RINGRIFT_DISTRIBUTION_TIMEOUT`                        | `ai-service/app/config/coordination_defaults.py:175`                        |                |
| `RINGRIFT_DNS_MAX_RETRIES`                             | `ai-service/app/config/coordination_defaults.py:698`                        |                |
| `RINGRIFT_DNS_TIMEOUT`                                 | `ai-service/app/config/coordination_defaults.py:699`                        |                |
| `RINGRIFT_DRIVE_SYNC_MAX_CONCURRENT`                   | `ai-service/app/coordination/external_drive_sync.py:76`                     |                |
| `RINGRIFT_DURATION_CMAES`                              | `ai-service/app/config/coordination_defaults.py:862`                        |                |
| `RINGRIFT_DURATION_EVALUATION`                         | `ai-service/app/config/coordination_defaults.py:864`                        |                |
| `RINGRIFT_DURATION_EXPORT`                             | `ai-service/app/config/coordination_defaults.py:866`                        |                |
| `RINGRIFT_DURATION_GPU_SELFPLAY`                       | `ai-service/app/config/coordination_defaults.py:860`                        |                |
| `RINGRIFT_DURATION_IMPROVEMENT`                        | `ai-service/app/config/coordination_defaults.py:868`                        |                |
| `RINGRIFT_DURATION_PIPELINE`                           | `ai-service/app/config/coordination_defaults.py:867`                        |                |
| `RINGRIFT_DURATION_SELFPLAY`                           | `ai-service/app/config/coordination_defaults.py:859`                        |                |
| `RINGRIFT_DURATION_SYNC`                               | `ai-service/app/config/coordination_defaults.py:865`                        |                |
| `RINGRIFT_DURATION_TOURNAMENT`                         | `ai-service/app/config/coordination_defaults.py:863`                        |                |
| `RINGRIFT_DURATION_TRAINING`                           | `ai-service/app/config/coordination_defaults.py:861`                        |                |
| `RINGRIFT_E2E_SET_SANDBOX_STALL__`                     | `src/client/contexts/SandboxContext.tsx:327`                                | prefix/pattern |
| `RINGRIFT_EARLY_TERM_MIN_MOVES`                        | `ai-service/app/ai/heuristic_ai.py:176`                                     |                |
| `RINGRIFT_EARLY_TERM_THRESHOLD`                        | `ai-service/app/ai/heuristic_ai.py:91`                                      |                |
| `RINGRIFT_ELIMINATION_AUDIT`                           | `src/shared/engine/aggregates/EliminationAggregate.ts:404`                  |                |
| `RINGRIFT_ELO_COORDINATOR`                             | `ai-service/scripts/p2p_orchestrator.py:1542`                               |                |
| `RINGRIFT_EMERGENCY_SYNC_COOLDOWN`                     | `ai-service/app/config/coordination_defaults.py:146`                        |                |
| `RINGRIFT_ENABLED`                                     | `ai-service/tests/unit/coordination/test_base_daemon.py:74`                 | test-only      |
| `RINGRIFT_ENABLE_`                                     | `scripts/validate-deployment-config.ts:513`                                 | prefix/pattern |
| `RINGRIFT_ENABLE_BACKEND_AI_SIM`                       | `tests/unit/GameEngine.aiSimulation.test.ts:47`                             | test-only      |
| `RINGRIFT_ENABLE_BACKEND_BOARD_INVARIANTS`             | `src/server/game/BoardManager.ts:43`                                        |                |
| `RINGRIFT_ENABLE_IMPROVEMENT_DAEMON`                   | `ai-service/deploy/deploy_cluster_resilience.py:440`                        |                |
| `RINGRIFT_ENABLE_SANDBOX_AI_SIM`                       | `tests/unit/ClientSandboxEngine.aiSimulation.test.ts:49`                    | test-only      |
| `RINGRIFT_ENABLE_SANDBOX_INTERNAL_PARITY`              | `tests/parity/Backend_vs_Sandbox.seed5.internalStateParity.test.ts:226`     | test-only      |
| `RINGRIFT_ENABLE_SEED14_PARITY`                        | `tests/unit/Seed14Move35LineParity.test.ts:40`                              | test-only      |
| `RINGRIFT_ENABLE_SEED17_PARITY`                        | `tests/unit/Seed17Move16And33Parity.GameEngine_vs_Sandbox.test.ts:43`       | test-only      |
| `RINGRIFT_EPHEMERAL`                                   | `ai-service/scripts/cluster_join.sh:23`                                     |                |
| `RINGRIFT_EPHEMERAL_CHECKPOINT_INTERVAL`               | `ai-service/app/config/coordination_defaults.py:1757`                       |                |
| `RINGRIFT_EPHEMERAL_DEADLINE`                          | `ai-service/app/coordination/sync_planner_v2.py:229`                        |                |
| `RINGRIFT_ERROR`                                       | `ai-service/app/errors.py:77`                                               |                |
| `RINGRIFT_EVACUATION_THRESHOLD`                        | `ai-service/app/config/coordination_defaults.py:1763`                       |                |
| `RINGRIFT_EXPLORATION_BOOST_DURATION`                  | `ai-service/app/coordination/selfplay_scheduler.py:2804`                    |                |
| `RINGRIFT_EXPLORATION_BOOST_FACTOR`                    | `ai-service/app/config/coordination_defaults.py:2096`                       |                |
| `RINGRIFT_EXPORT_PARITY_SNAPSHOTS`                     | `tests/unit/ExportLineAndTerritoryMultiRegionSnapshot.test.ts:20`           | test-only      |
| `RINGRIFT_FALLBACK_SELFPLAY_SCRIPT`                    | `ai-service/scripts/node_resilience.py:1431`                                |                |
| `RINGRIFT_FLAG`                                        | `tests/unit/envFlags.test.ts:32`                                            | test-only      |
| `RINGRIFT_FORCE_BOOKKEEPING_MOVES`                     | `ai-service/app/rules/default_engine.py:134`                                |                |
| `RINGRIFT_FORCE_COORDINATOR`                           | `ai-service/scripts/cluster_join.sh:22`                                     |                |
| `RINGRIFT_FSM_TRACE_DEBUG`                             | `src/shared/utils/envFlags.ts:206`                                          |                |
| `RINGRIFT_FUTURE_RESULT_TIMEOUT`                       | `ai-service/app/config/coordination_defaults.py:1536`                       |                |
| `RINGRIFT_GAMES_PER_JOB`                               | `ai-service/scripts/ops/maintain_selfplay_load.sh:17`                       |                |
| `RINGRIFT_GIT_UPDATE_INTERVAL`                         | `ai-service/scripts/p2p/loops/maintenance_loops.py:33`                      |                |
| `RINGRIFT_GPU_`                                        | `ai-service/app/training/config.py:55`                                      | prefix/pattern |
| `RINGRIFT_GPU_GH200_BATCH_MULTIPLIER`                  | `ai-service/tests/unit/training/test_config.py:79`                          | test-only      |
| `RINGRIFT_GPU_H100_BATCH_MULTIPLIER`                   | `ai-service/tests/unit/training/test_config.py:80`                          | test-only      |
| `RINGRIFT_GPU_MAX_BATCH_SIZE`                          | `ai-service/tests/unit/training/test_config.py:60`                          | test-only      |
| `RINGRIFT_GPU_MEMORY_CRITICAL`                         | `ai-service/app/config/coordination_defaults.py:722`                        |                |
| `RINGRIFT_GPU_MEMORY_WARNING`                          | `ai-service/app/config/coordination_defaults.py:721`                        |                |
| `RINGRIFT_GPU_RESERVED_MEMORY_GB`                      | `ai-service/tests/unit/training/test_config.py:66`                          | test-only      |
| `RINGRIFT_GPU_SCALE_DOWN_THRESHOLD`                    | `ai-service/app/config/coordination_defaults.py:842`                        |                |
| `RINGRIFT_GPU_SCALE_UP_THRESHOLD`                      | `ai-service/app/config/coordination_defaults.py:841`                        |                |
| `RINGRIFT_GPU_TARGET_MAX`                              | `ai-service/app/config/coordination_defaults.py:478`                        |                |
| `RINGRIFT_GPU_TARGET_MIN`                              | `ai-service/app/config/coordination_defaults.py:477`                        |                |
| `RINGRIFT_HEALTHY_CACHE_TTL`                           | `ai-service/app/config/coordination_defaults.py:451`                        |                |
| `RINGRIFT_HEALTH_CHECK_INTERVAL`                       | `ai-service/config/cluster_env.sh:17`                                       |                |
| `RINGRIFT_HEALTH_CHECK_MAX_CONCURRENT`                 | `ai-service/app/config/coordination_defaults.py:2153`                       |                |
| `RINGRIFT_HEALTH_CHECK_ORCHESTRATOR_INTERVAL`          | `ai-service/app/config/coordination_defaults.py:2146`                       |                |
| `RINGRIFT_HEALTH_CHECK_TIMEOUT`                        | `ai-service/app/config/coordination_defaults.py:1518`                       |                |
| `RINGRIFT_HIGH_CPU_MAX_SELFPLAY`                       | `ai-service/app/config/coordination_defaults.py:529`                        |                |
| `RINGRIFT_HOT_MIN_GAMES`                               | `ai-service/app/training/feedback_accelerator.py:84`                        |                |
| `RINGRIFT_HOURLY_BUDGET_USD`                           | `ai-service/app/coordination/availability/capacity_planner.py:177`          |                |
| `RINGRIFT_HTTP_PULL_SOURCE`                            | `ai-service/scripts/http_pull.py:65`                                        |                |
| `RINGRIFT_HYBRID_DISABLE_GPU`                          | `ai-service/app/ai/hybrid_tree_policy_ai.py:42`                             |                |
| `RINGRIFT_HYBRID_HTTP_WEIGHT`                          | `ai-service/app/config/coordination_defaults.py:1077`                       |                |
| `RINGRIFT_HYBRID_SWIM_WEIGHT`                          | `ai-service/app/config/coordination_defaults.py:1076`                       |                |
| `RINGRIFT_IMPROVEMENT_LEADER_ONLY`                     | `ai-service/deploy/setup_node_resilience_macos.sh:266`                      |                |
| `RINGRIFT_INTEGRITY`                                   | `ai-service/app/coordination/integrity_check_daemon.py:58`                  |                |
| `RINGRIFT_INTEGRITY_CHECK_TIMEOUT`                     | `ai-service/app/db/integrity.py:36`                                         |                |
| `RINGRIFT_INTEGRITY_DATA_DIR`                          | `ai-service/tests/unit/coordination/test_integrity_check_daemon.py:169`     | test-only      |
| `RINGRIFT_INTEGRITY_MAX_ORPHANS`                       | `ai-service/tests/unit/coordination/test_integrity_check_daemon.py:171`     | test-only      |
| `RINGRIFT_INTEGRITY_QUARANTINE_DAYS`                   | `ai-service/tests/unit/coordination/test_integrity_check_daemon.py:170`     | test-only      |
| `RINGRIFT_INTERVAL`                                    | `ai-service/tests/unit/coordination/test_base_daemon.py:84`                 | test-only      |
| `RINGRIFT_INVENTORY_CACHE_TTL`                         | `ai-service/app/config/coordination_defaults.py:1883`                       |                |
| `RINGRIFT_INVENTORY_FETCH_TIMEOUT`                     | `ai-service/app/config/coordination_defaults.py:1892`                       |                |
| `RINGRIFT_INVENTORY_MAX_NODES`                         | `ai-service/app/config/coordination_defaults.py:1889`                       |                |
| `RINGRIFT_INVENTORY_REFRESH_INTERVAL`                  | `ai-service/app/config/coordination_defaults.py:1880`                       |                |
| `RINGRIFT_INVENTORY_STALE_THRESHOLD`                   | `ai-service/app/config/coordination_defaults.py:1886`                       |                |
| `RINGRIFT_JOB_EXPORT_TIMEOUT`                          | `ai-service/app/config/coordination_defaults.py:1983`                       |                |
| `RINGRIFT_JOB_GAUNTLET_TIMEOUT`                        | `ai-service/app/config/coordination_defaults.py:1980`                       |                |
| `RINGRIFT_JOB_HEALTH_CHECK_TIMEOUT`                    | `ai-service/app/config/coordination_defaults.py:1974`                       |                |
| `RINGRIFT_JOB_HEARTBEAT_TIMEOUT`                       | `ai-service/scripts/p2p/managers/job_manager.py:178`                        |                |
| `RINGRIFT_JOB_MODEL_SYNC_TIMEOUT`                      | `ai-service/app/config/coordination_defaults.py:1986`                       |                |
| `RINGRIFT_JOB_REAPER_CHECK_INTERVAL`                   | `ai-service/app/config/coordination_defaults.py:1676`                       |                |
| `RINGRIFT_JOB_REAPER_SSH_TIMEOUT`                      | `ai-service/app/config/coordination_defaults.py:1688`                       |                |
| `RINGRIFT_JOB_SELFPLAY_TIMEOUT`                        | `ai-service/app/config/coordination_defaults.py:1965`                       |                |
| `RINGRIFT_JOB_STATUS_TIMEOUT`                          | `ai-service/app/config/coordination_defaults.py:1971`                       |                |
| `RINGRIFT_JOB_TIMEOUT_CMAES`                           | `ai-service/app/config/coordination_defaults.py:1184`                       |                |
| `RINGRIFT_JOB_TIMEOUT_CPU_SELFPLAY`                    | `ai-service/app/config/coordination_defaults.py:1166`                       |                |
| `RINGRIFT_JOB_TIMEOUT_DATA_EXPORT`                     | `ai-service/app/config/coordination_defaults.py:1175`                       |                |
| `RINGRIFT_JOB_TIMEOUT_EVALUATION`                      | `ai-service/app/config/coordination_defaults.py:1178`                       |                |
| `RINGRIFT_JOB_TIMEOUT_GPU_SELFPLAY`                    | `ai-service/app/config/coordination_defaults.py:1163`                       |                |
| `RINGRIFT_JOB_TIMEOUT_MODEL_SYNC`                      | `ai-service/app/config/coordination_defaults.py:1181`                       |                |
| `RINGRIFT_JOB_TIMEOUT_PIPELINE_STAGE`                  | `ai-service/app/config/coordination_defaults.py:1187`                       |                |
| `RINGRIFT_JOB_TIMEOUT_TOURNAMENT`                      | `ai-service/app/config/coordination_defaults.py:1172`                       |                |
| `RINGRIFT_JOB_TIMEOUT_TRAINING`                        | `ai-service/app/config/coordination_defaults.py:1169`                       |                |
| `RINGRIFT_JOB_TOURNAMENT_TIMEOUT`                      | `ai-service/app/config/coordination_defaults.py:1977`                       |                |
| `RINGRIFT_JOB_TRAINING_TIMEOUT`                        | `ai-service/app/config/coordination_defaults.py:1968`                       |                |
| `RINGRIFT_LARGE_BOARD_MCTS_BUDGET`                     | `ai-service/app/config/coordination_defaults.py:1797`                       |                |
| `RINGRIFT_LARGE_DB_THRESHOLD`                          | `ai-service/app/config/coordination_defaults.py:1913`                       |                |
| `RINGRIFT_LATENCY_PERCENTILE`                          | `ai-service/app/config/coordination_defaults.py:1069`                       |                |
| `RINGRIFT_LATENCY_WINDOW`                              | `ai-service/app/config/coordination_defaults.py:1066`                       |                |
| `RINGRIFT_LATENCY_WINDOW_SIZE`                         | `ai-service/app/config/coordination_defaults.py:1743`                       |                |
| `RINGRIFT_LAUNCHD_PATH`                                | `ai-service/deploy/setup_node_resilience_macos.sh:60`                       |                |
| `RINGRIFT_LEADER_CHECK_TIMEOUT`                        | `ai-service/app/config/coordination_defaults.py:1691`                       |                |
| `RINGRIFT_LEADER_RETRY_DELAY`                          | `ai-service/app/config/coordination_defaults.py:1694`                       |                |
| `RINGRIFT_LEGACY_SYNC`                                 | `ai-service/scripts/local_data_sync.sh:20`                                  |                |
| `RINGRIFT_LOAD_BACKOFF_SECONDS`                        | `ai-service/app/utils/load_throttle.py:10`                                  |                |
| `RINGRIFT_LOG_DIR`                                     | `ai-service/app/config/env_validator.py:197`                                |                |
| `RINGRIFT_LOW_QUALITY_THRESHOLD`                       | `ai-service/app/config/coordination_defaults.py:2101`                       |                |
| `RINGRIFT_MANIFEST_CACHE`                              | `ai-service/app/coordination/training_data_manifest.py:161`                 |                |
| `RINGRIFT_MASTERY_THRESHOLD`                           | `ai-service/app/config/coordination_defaults.py:2089`                       |                |
| `RINGRIFT_MAX_CONCURRENT_TRANSFERS`                    | `ai-service/app/config/coordination_defaults.py:505`                        |                |
| `RINGRIFT_MAX_CURRICULUM_WEIGHT`                       | `ai-service/app/config/coordination_defaults.py:2111`                       |                |
| `RINGRIFT_MAX_DEFAULT_WORKERS`                         | `ai-service/app/training/parallel_selfplay.py:659`                          |                |
| `RINGRIFT_MAX_DISK_USAGE_PERCENT`                      | `ai-service/scripts/p2p/handlers/sync.py:569`                               |                |
| `RINGRIFT_MAX_HEALTH_CHECKS`                           | `ai-service/app/config/coordination_defaults.py:457`                        |                |
| `RINGRIFT_MAX_HOURLY_COST`                             | `ai-service/app/config/coordination_defaults.py:845`                        |                |
| `RINGRIFT_MAX_INSTANCES`                               | `ai-service/app/config/coordination_defaults.py:837`                        |                |
| `RINGRIFT_MAX_JOB_REASSIGNMENT_ATTEMPTS`               | `ai-service/scripts/p2p/managers/job_manager.py:184`                        |                |
| `RINGRIFT_MAX_LOAD_ABSOLUTE`                           | `ai-service/app/utils/load_throttle.py:9`                                   |                |
| `RINGRIFT_MAX_LOAD_FACTOR`                             | `ai-service/app/utils/load_throttle.py:8`                                   |                |
| `RINGRIFT_MAX_PROCESSES`                               | `ai-service/app/config/coordination_defaults.py:1831`                       |                |
| `RINGRIFT_MAX_REASSIGN_ATTEMPTS`                       | `ai-service/app/config/coordination_defaults.py:1682`                       |                |
| `RINGRIFT_MAX_RETRIES`                                 | `ai-service/app/config/coordination_defaults.py:100`                        |                |
| `RINGRIFT_MEMORY_CRITICAL_THRESHOLD`                   | `ai-service/app/config/coordination_defaults.py:718`                        |                |
| `RINGRIFT_MEMORY_WARNING_THRESHOLD`                    | `ai-service/app/config/coordination_defaults.py:717`                        |                |
| `RINGRIFT_METRICS_ANOMALY_THRESHOLD`                   | `ai-service/app/config/coordination_defaults.py:760`                        |                |
| `RINGRIFT_METRICS_PLATEAU_THRESHOLD`                   | `ai-service/app/config/coordination_defaults.py:751`                        |                |
| `RINGRIFT_METRICS_PLATEAU_WINDOW`                      | `ai-service/app/config/coordination_defaults.py:754`                        |                |
| `RINGRIFT_METRICS_REGRESSION_THRESHOLD`                | `ai-service/app/config/coordination_defaults.py:757`                        |                |
| `RINGRIFT_METRICS_WINDOW_SIZE`                         | `ai-service/app/config/coordination_defaults.py:748`                        |                |
| `RINGRIFT_MIN_AVAILABILITY_SCORE`                      | `ai-service/app/config/coordination_defaults.py:185`                        |                |
| `RINGRIFT_MIN_CURRICULUM_WEIGHT`                       | `ai-service/app/config/coordination_defaults.py:2114`                       |                |
| `RINGRIFT_MIN_DISK_FREE_GB`                            | `ai-service/scripts/p2p/managers/job_manager.py:404`                        |                |
| `RINGRIFT_MIN_GAMES_FOR_UPDATE`                        | `ai-service/app/config/coordination_defaults.py:2105`                       |                |
| `RINGRIFT_MIN_GAMES_PER_ALLOCATION`                    | `ai-service/app/config/coordination_defaults.py:1784`                       |                |
| `RINGRIFT_MIN_GPU_NODES`                               | `ai-service/app/coordination/availability/capacity_planner.py:183`          |                |
| `RINGRIFT_MIN_GPU_UTILIZATION`                         | `ai-service/app/config/coordination_defaults.py:1837`                       |                |
| `RINGRIFT_MIN_HEALTHY_FRACTION`                        | `ai-service/app/config/coordination_defaults.py:731`                        |                |
| `RINGRIFT_MIN_HEALTHY_HOSTS`                           | `ai-service/app/config/coordination_defaults.py:460`                        |                |
| `RINGRIFT_MIN_IDLE_BEFORE_TERMINATION`                 | `ai-service/scripts/p2p/adapters/scale_adapters.py:138`                     |                |
| `RINGRIFT_MIN_INSTANCES`                               | `ai-service/app/config/coordination_defaults.py:838`                        |                |
| `RINGRIFT_MIN_MEMORY_GB_FOR_TASKS`                     | `ai-service/app/config/coordination_defaults.py:1788`                       |                |
| `RINGRIFT_MIN_MOVES`                                   | `ai-service/scripts/jsonl_to_npz.py:1373`                                   |                |
| `RINGRIFT_MIN_NODES_FOR_EVALUATION`                    | `ai-service/app/config/coordination_defaults.py:171`                        |                |
| `RINGRIFT_MIN_NODES_FOR_PROMOTION`                     | `ai-service/app/config/coordination_defaults.py:168`                        |                |
| `RINGRIFT_MIN_REPLICATION`                             | `ai-service/app/coordination/sync_planner_v2.py:220`                        |                |
| `RINGRIFT_MIN_THROUGHPUT`                              | `ai-service/scripts/monitoring/selfplay_throughput_monitor.sh:13`           |                |
| `RINGRIFT_MIN_VOTER_QUORUM`                            | `ai-service/app/config/coordination_defaults.py:1087`                       |                |
| `RINGRIFT_MONITOR_INTERVAL`                            | `ai-service/scripts/daemon_health_monitor.py:25`                            |                |
| `RINGRIFT_MONITOR_LOG_FILE`                            | `ai-service/scripts/daemon_health_monitor.py:26`                            |                |
| `RINGRIFT_NAS_BOARD`                                   | `ai-service/scripts/launch_distributed_nas.py:141`                          |                |
| `RINGRIFT_NAS_PLAYERS`                                 | `ai-service/scripts/launch_distributed_nas.py:142`                          |                |
| `RINGRIFT_NAS_REAL_TRAINING`                           | `ai-service/scripts/launch_distributed_nas.py:140`                          |                |
| `RINGRIFT_NNUE_ZERO_SUM_EVAL`                          | `ai-service/app/ai/minimax_ai.py:1097`                                      |                |
| `RINGRIFT_NN_`                                         | `ai-service/app/config/training_config.py:146`                              | prefix/pattern |
| `RINGRIFT_NN_ARCHITECTURE`                             | `ai-service/app/ai/_neural_net_legacy.py:3155`                              |                |
| `RINGRIFT_NN_EVAL_BATCH_TIMEOUT_MS`                    | `ai-service/app/ai/async_nn_eval.py:248`                                    |                |
| `RINGRIFT_NN_EVAL_MAX_BATCH`                           | `ai-service/app/ai/async_nn_eval.py:247`                                    |                |
| `RINGRIFT_NN_EVAL_QUEUE`                               | `ai-service/app/ai/async_nn_eval.py:47`                                     |                |
| `RINGRIFT_NN_MODEL_ID`                                 | `ai-service/app/config/training_config.py:188`                              |                |
| `RINGRIFT_NN_RESOLVE_MAX_PROBE`                        | `ai-service/app/ai/_neural_net_legacy.py:3488`                              |                |
| `RINGRIFT_NODE_AVAILABILITY`                           | `ai-service/app/coordination/node_availability/daemon.py:83`                |                |
| `RINGRIFT_NODE_AVAILABILITY_AUTO_VOTERS`               | `ai-service/tests/unit/coordination/node_availability/test_daemon.py:169`   | test-only      |
| `RINGRIFT_NODE_AVAILABILITY_DRY_RUN`                   | `ai-service/app/coordination/node_availability/__init__.py:23`              |                |
| `RINGRIFT_NODE_AVAILABILITY_ENABLED`                   | `ai-service/app/coordination/node_availability/__init__.py:22`              |                |
| `RINGRIFT_NODE_AVAILABILITY_GRACE_PERIOD`              | `ai-service/tests/unit/coordination/node_availability/test_daemon.py:151`   | test-only      |
| `RINGRIFT_NODE_AVAILABILITY_INTERVAL`                  | `ai-service/app/coordination/node_availability/__init__.py:24`              |                |
| `RINGRIFT_NODE_AVAILABILITY_LAMBDA`                    | `ai-service/tests/unit/coordination/node_availability/test_daemon.py:159`   | test-only      |
| `RINGRIFT_NODE_AVAILABILITY_RUNPOD`                    | `ai-service/tests/unit/coordination/node_availability/test_daemon.py:160`   | test-only      |
| `RINGRIFT_NODE_AVAILABILITY_VAST`                      | `ai-service/tests/unit/coordination/node_availability/test_daemon.py:158`   | test-only      |
| `RINGRIFT_NODE_BLACKLIST_DURATION`                     | `ai-service/app/config/coordination_defaults.py:1685`                       |                |
| `RINGRIFT_NODE_ID__`                                   | `ai-service/scripts/install_launchd_services.sh:99`                         | prefix/pattern |
| `RINGRIFT_NODE_OFFLINE_THRESHOLD`                      | `ai-service/app/config/coordination_defaults.py:725`                        |                |
| `RINGRIFT_NODE_OVERLOAD_THRESHOLD`                     | `ai-service/app/config/coordination_defaults.py:1740`                       |                |
| `RINGRIFT_NODE_RECOVERY`                               | `ai-service/app/coordination/node_recovery_daemon.py:127`                   |                |
| `RINGRIFT_NODE_RESILIENCE_LOCK_FILE`                   | `ai-service/scripts/node_resilience.py:94`                                  |                |
| `RINGRIFT_NODE_RESILIENCE_LOG_FILE`                    | `ai-service/scripts/node_resilience.py:73`                                  |                |
| `RINGRIFT_NONEXISTENT`                                 | `ai-service/tests/unit/config/test_env.py:239`                              | test-only      |
| `RINGRIFT_NOTIFICATION_CONFIG`                         | `ai-service/app/training/notification_config.py:300`                        |                |
| `RINGRIFT_NUM_PLAYERS`                                 | `ai-service/app/training/config.py:985`                                     |                |
| `RINGRIFT_OPTIMIZATION_COOLDOWN`                       | `ai-service/app/config/coordination_defaults.py:599`                        |                |
| `RINGRIFT_OPTIMIZATION_INTERVAL`                       | `ai-service/app/config/coordination_defaults.py:491`                        |                |
| `RINGRIFT_OPTIMIZATION_MAX_HISTORY`                    | `ai-service/app/config/coordination_defaults.py:602`                        |                |
| `RINGRIFT_OPTIMIZATION_MIN_EPOCHS`                     | `ai-service/app/config/coordination_defaults.py:596`                        |                |
| `RINGRIFT_OPTIMIZATION_PLATEAU_THRESHOLD`              | `ai-service/app/config/coordination_defaults.py:593`                        |                |
| `RINGRIFT_OPTIMIZATION_PLATEAU_WINDOW`                 | `ai-service/app/config/coordination_defaults.py:590`                        |                |
| `RINGRIFT_ORCHESTRATOR_HOST`                           | `ai-service/scripts/train_nnue.py:235`                                      |                |
| `RINGRIFT_ORCHESTRATOR_PORT`                           | `ai-service/scripts/train_nnue.py:236`                                      |                |
| `RINGRIFT_ORCHESTRATOR_URL`                            | `ai-service/scripts/hyperparameter_ab_testing.py:54`                        |                |
| `RINGRIFT_ORPHAN_ALERT_THRESHOLD`                      | `ai-service/app/config/coordination_defaults.py:1339`                       |                |
| `RINGRIFT_ORPHAN_MIN_AGE_HOURS`                        | `ai-service/app/config/coordination_defaults.py:1336`                       |                |
| `RINGRIFT_ORPHAN_MIN_GAMES`                            | `ai-service/app/config/coordination_defaults.py:1333`                       |                |
| `RINGRIFT_ORPHAN_SCAN_INTERVAL`                        | `ai-service/app/config/coordination_defaults.py:1330`                       |                |
| `RINGRIFT_OWC_ENABLED`                                 | `ai-service/app/coordination/unified_data_plane_daemon.py:145`              |                |
| `RINGRIFT_OWC_HOST`                                    | `ai-service/app/coordination/training_data_manifest.py:141`                 |                |
| `RINGRIFT_OWC_PATH`                                    | `ai-service/app/coordination/training_data_manifest.py:144`                 |                |
| `RINGRIFT_OWC_SSH_KEY`                                 | `ai-service/app/coordination/training_data_manifest.py:146`                 |                |
| `RINGRIFT_OWC_USER`                                    | `ai-service/app/coordination/training_data_manifest.py:142`                 |                |
| `RINGRIFT_P2P_`                                        | `ai-service/app/p2p/config.py:73`                                           | prefix/pattern |
| `RINGRIFT_P2P_ADVERTISE_HOST`                          | `ai-service/app/p2p/constants.py:372`                                       |                |
| `RINGRIFT_P2P_ADVERTISE_PORT`                          | `ai-service/app/p2p/constants.py:373`                                       |                |
| `RINGRIFT_P2P_AGGRESSIVE_ELECTION_TIMEOUT`             | `ai-service/app/p2p/constants.py:341`                                       |                |
| `RINGRIFT_P2P_AGGRESSIVE_FAILOVER`                     | `ai-service/app/p2p/constants.py:335`                                       |                |
| `RINGRIFT_P2P_AGGRESSIVE_LEASE_DURATION`               | `ai-service/app/p2p/constants.py:340`                                       |                |
| `RINGRIFT_P2P_AGGRESSIVE_PEER_TIMEOUT`                 | `ai-service/app/p2p/constants.py:338`                                       |                |
| `RINGRIFT_P2P_AGGRESSIVE_SUSPECT_TIMEOUT`              | `ai-service/app/p2p/constants.py:339`                                       |                |
| `RINGRIFT_P2P_AUTO_ASSIGN`                             | `ai-service/app/p2p/constants.py:487`                                       |                |
| `RINGRIFT_P2P_AUTO_TRAINING_THRESHOLD_MB`              | `ai-service/app/p2p/constants.py:423`                                       |                |
| `RINGRIFT_P2P_AUTO_WORK_BATCH_SIZE`                    | `ai-service/app/p2p/constants.py:488`                                       |                |
| `RINGRIFT_P2P_BASE_DELAY`                              | `ai-service/app/config/coordination_defaults.py:694`                        |                |
| `RINGRIFT_P2P_BIND_ADDR`                               | `ai-service/app/p2p/constants.py:539`                                       |                |
| `RINGRIFT_P2P_BOOTSTRAP_MAX_SEEDS_PER_RUN`             | `ai-service/scripts/p2p_orchestrator.py:17322`                              |                |
| `RINGRIFT_P2P_BOOTSTRAP_SEEDS`                         | `ai-service/app/p2p/constants.py:230`                                       |                |
| `RINGRIFT_P2P_CHECK_INTERVAL`                          | `ai-service/app/config/coordination_defaults.py:2131`                       |                |
| `RINGRIFT_P2P_DATA_MANAGEMENT_INTERVAL`                | `ai-service/app/p2p/constants.py:419`                                       |                |
| `RINGRIFT_P2P_DATA_SYNC_BASE`                          | `ai-service/app/p2p/constants.py:445`                                       |                |
| `RINGRIFT_P2P_DATA_SYNC_MAX`                           | `ai-service/app/p2p/constants.py:447`                                       |                |
| `RINGRIFT_P2P_DATA_SYNC_MIN`                           | `ai-service/app/p2p/constants.py:446`                                       |                |
| `RINGRIFT_P2P_DB_EXPORT_THRESHOLD_MB`                  | `ai-service/app/p2p/constants.py:420`                                       |                |
| `RINGRIFT_P2P_DEFAULT_SEEDS`                           | `ai-service/app/coordination/p2p_integration.py:159`                        |                |
| `RINGRIFT_P2P_DISK_CLEANUP_THRESHOLD`                  | `ai-service/app/p2p/config.py:84`                                           |                |
| `RINGRIFT_P2P_DISK_CRITICAL_THRESHOLD`                 | `ai-service/app/p2p/config.py:78`                                           |                |
| `RINGRIFT_P2P_DISK_WARNING_THRESHOLD`                  | `ai-service/app/p2p/config.py:81`                                           |                |
| `RINGRIFT_P2P_DYNAMIC_VOTER`                           | `ai-service/app/p2p/constants.py:382`                                       |                |
| `RINGRIFT_P2P_DYNAMIC_VOTER_MAX_QUORUM`                | `ai-service/app/p2p/constants.py:385`                                       |                |
| `RINGRIFT_P2P_DYNAMIC_VOTER_MIN`                       | `ai-service/app/p2p/constants.py:383`                                       |                |
| `RINGRIFT_P2P_DYNAMIC_VOTER_TARGET`                    | `ai-service/app/p2p/constants.py:384`                                       |                |
| `RINGRIFT_P2P_ELECTION_TIMEOUT`                        | `ai-service/app/config/coordination_defaults.py:952`                        |                |
| `RINGRIFT_P2P_ENDPOINT`                                | `ai-service/scripts/serf_event_handler.py:50`                               |                |
| `RINGRIFT_P2P_GH200_MAX_SELFPLAY`                      | `ai-service/app/p2p/config.py:110`                                          |                |
| `RINGRIFT_P2P_GH200_MIN_SELFPLAY`                      | `ai-service/app/p2p/config.py:107`                                          |                |
| `RINGRIFT_P2P_GIT_BRANCH`                              | `ai-service/app/p2p/constants.py:402`                                       |                |
| `RINGRIFT_P2P_GIT_REMOTE`                              | `ai-service/app/p2p/constants.py:403`                                       |                |
| `RINGRIFT_P2P_GIT_UPDATE_CHECK_INTERVAL`               | `ai-service/app/p2p/config.py:157`                                          |                |
| `RINGRIFT_P2P_GOSSIP_FANOUT`                           | `ai-service/app/p2p/constants.py:168`                                       |                |
| `RINGRIFT_P2P_GOSSIP_INTERVAL`                         | `ai-service/app/config/coordination_defaults.py:943`                        |                |
| `RINGRIFT_P2P_GOSSIP_JITTER`                           | `ai-service/app/p2p/constants.py:173`                                       |                |
| `RINGRIFT_P2P_GOSSIP_MAX_PEER_ENDPOINTS`               | `ai-service/app/p2p/constants.py:176`                                       |                |
| `RINGRIFT_P2P_GRACEFUL_SHUTDOWN_BEFORE_UPDATE`         | `ai-service/app/p2p/constants.py:405`                                       |                |
| `RINGRIFT_P2P_GRACE_PERIOD`                            | `ai-service/scripts/node_resilience.py:1356`                                |                |
| `RINGRIFT_P2P_HEALTH_PORT`                             | `ai-service/app/config/coordination_defaults.py:937`                        |                |
| `RINGRIFT_P2P_HEARTBEAT_INTERVAL`                      | `ai-service/app/config/coordination_defaults.py:946`                        |                |
| `RINGRIFT_P2P_HTTP_CONNECT_TIMEOUT`                    | `ai-service/app/p2p/constants.py:158`                                       |                |
| `RINGRIFT_P2P_HTTP_TOTAL_TIMEOUT`                      | `ai-service/app/p2p/constants.py:159`                                       |                |
| `RINGRIFT_P2P_IDLE_CHECK_INTERVAL`                     | `ai-service/app/p2p/constants.py:411`                                       |                |
| `RINGRIFT_P2P_IDLE_GPU_THRESHOLD`                      | `ai-service/app/p2p/constants.py:412`                                       |                |
| `RINGRIFT_P2P_IDLE_GRACE_PERIOD`                       | `ai-service/app/p2p/constants.py:413`                                       |                |
| `RINGRIFT_P2P_INITIAL_CLUSTER_EPOCH`                   | `ai-service/app/p2p/constants.py:247`                                       |                |
| `RINGRIFT_P2P_ISOLATED_BOOTSTRAP_INTERVAL`             | `ai-service/app/p2p/constants.py:237`                                       |                |
| `RINGRIFT_P2P_JOB_CHECK_INTERVAL`                      | `ai-service/app/p2p/config.py:25`                                           |                |
| `RINGRIFT_P2P_LEADER_DEGRADED_STEPDOWN_DELAY`          | `ai-service/app/p2p/constants.py:396`                                       |                |
| `RINGRIFT_P2P_LEADER_HEALTH_CHECK_INTERVAL`            | `ai-service/app/p2p/constants.py:394`                                       |                |
| `RINGRIFT_P2P_LEADER_MIN_RESPONSE_RATE`                | `ai-service/app/p2p/constants.py:395`                                       |                |
| `RINGRIFT_P2P_LOAD_AVG_MAX_MULT`                       | `ai-service/app/p2p/config.py:149`                                          |                |
| `RINGRIFT_P2P_LOAD_MAX_FOR_NEW_JOBS`                   | `ai-service/app/p2p/config.py:96`                                           |                |
| `RINGRIFT_P2P_MANIFEST_JSONL_LINECOUNT_CHUNK_BYTES`    | `ai-service/app/p2p/constants.py:427`                                       |                |
| `RINGRIFT_P2P_MANIFEST_JSONL_LINECOUNT_MAX_BYTES`      | `ai-service/app/p2p/constants.py:428`                                       |                |
| `RINGRIFT_P2P_MANIFEST_JSONL_SAMPLE_BYTES`             | `ai-service/app/p2p/constants.py:426`                                       |                |
| `RINGRIFT_P2P_MAX_CONCURRENT_CMAES_EVALS`              | `ai-service/deploy/deploy_cluster_resilience.py:451`                        |                |
| `RINGRIFT_P2P_MAX_CONCURRENT_EXPORTS`                  | `ai-service/app/p2p/constants.py:422`                                       |                |
| `RINGRIFT_P2P_MAX_DELAY`                               | `ai-service/app/config/coordination_defaults.py:695`                        |                |
| `RINGRIFT_P2P_MAX_GAUNTLET_RUNTIME`                    | `ai-service/app/p2p/constants.py:481`                                       |                |
| `RINGRIFT_P2P_MAX_PEERS`                               | `ai-service/app/config/coordination_defaults.py:964`                        |                |
| `RINGRIFT_P2P_MAX_RETRIES`                             | `ai-service/app/config/coordination_defaults.py:693`                        |                |
| `RINGRIFT_P2P_MAX_SELFPLAY_RUNTIME`                    | `ai-service/app/p2p/constants.py:478`                                       |                |
| `RINGRIFT_P2P_MAX_TOURNAMENT_RUNTIME`                  | `ai-service/app/p2p/constants.py:480`                                       |                |
| `RINGRIFT_P2P_MAX_TRAINING_RUNTIME`                    | `ai-service/app/p2p/constants.py:479`                                       |                |
| `RINGRIFT_P2P_MEMORY_CRITICAL_THRESHOLD`               | `ai-service/app/p2p/config.py:87`                                           |                |
| `RINGRIFT_P2P_MEMORY_WARNING_THRESHOLD`                | `ai-service/app/p2p/config.py:90`                                           |                |
| `RINGRIFT_P2P_MIN_BOOTSTRAP_ATTEMPTS`                  | `ai-service/app/p2p/constants.py:234`                                       |                |
| `RINGRIFT_P2P_MIN_CONNECTED_PEERS`                     | `ai-service/app/p2p/constants.py:240`                                       |                |
| `RINGRIFT_P2P_MIN_GAMES_FOR_SYNC`                      | `ai-service/app/p2p/constants.py:437`                                       |                |
| `RINGRIFT_P2P_MIN_MEMORY_GB`                           | `ai-service/app/p2p/config.py:93`                                           |                |
| `RINGRIFT_P2P_MIN_MEMORY_GB_TRAINING`                  | `ai-service/app/p2p/constants.py:96`                                        |                |
| `RINGRIFT_P2P_MODEL_SYNC_BASE`                         | `ai-service/app/p2p/constants.py:450`                                       |                |
| `RINGRIFT_P2P_MODEL_SYNC_INTERVAL`                     | `ai-service/app/p2p/constants.py:438`                                       |                |
| `RINGRIFT_P2P_MODEL_SYNC_MAX`                          | `ai-service/app/p2p/constants.py:452`                                       |                |
| `RINGRIFT_P2P_MODEL_SYNC_MIN`                          | `ai-service/app/p2p/constants.py:451`                                       |                |
| `RINGRIFT_P2P_NAT_STALE_SECONDS`                       | `ai-service/app/p2p/constants.py:198`                                       |                |
| `RINGRIFT_P2P_NETWORK`                                 | `ai-service/app/p2p/constants.py:547`                                       |                |
| `RINGRIFT_P2P_PEERS`                                   | `ai-service/scripts/p2p_keepalive.sh:14`                                    |                |
| `RINGRIFT_P2P_PEER_CACHE_MAX_ENTRIES`                  | `ai-service/app/p2p/constants.py:190`                                       |                |
| `RINGRIFT_P2P_PEER_CACHE_TTL_SECONDS`                  | `ai-service/app/p2p/constants.py:189`                                       |                |
| `RINGRIFT_P2P_PEER_PURGE_AFTER_SECONDS`                | `ai-service/app/p2p/constants.py:186`                                       |                |
| `RINGRIFT_P2P_PEER_RECOVERY_ENABLED`                   | `ai-service/scripts/p2p/loops/peer_recovery_loop.py:82`                     |                |
| `RINGRIFT_P2P_PEER_RECOVERY_INTERVAL`                  | `ai-service/app/p2p/constants.py:183`                                       |                |
| `RINGRIFT_P2P_PEER_REPUTATION_ALPHA`                   | `ai-service/app/p2p/constants.py:191`                                       |                |
| `RINGRIFT_P2P_PEER_RETIRE_AFTER_SECONDS`               | `ai-service/app/p2p/config.py:123`                                          |                |
| `RINGRIFT_P2P_PEER_TIMEOUT`                            | `ai-service/app/config/coordination_defaults.py:949`                        |                |
| `RINGRIFT_P2P_PYTHON__`                                | `ai-service/scripts/install_launchd_services.sh:98`                         | prefix/pattern |
| `RINGRIFT_P2P_QUORUM`                                  | `ai-service/app/config/coordination_defaults.py:961`                        |                |
| `RINGRIFT_P2P_RELAY_HEARTBEAT_INTERVAL`                | `ai-service/app/p2p/constants.py:199`                                       |                |
| `RINGRIFT_P2P_RETRY_RETIRED_NODE_INTERVAL`             | `ai-service/app/p2p/config.py:128`                                          |                |
| `RINGRIFT_P2P_SPAWN_RATE_LIMIT`                        | `ai-service/app/p2p/config.py:152`                                          |                |
| `RINGRIFT_P2P_STALE_PROCESS_CHECK_INTERVAL`            | `ai-service/app/p2p/constants.py:467`                                       |                |
| `RINGRIFT_P2P_STALE_PROCESS_PATTERNS`                  | `ai-service/app/p2p/constants.py:469`                                       |                |
| `RINGRIFT_P2P_STARTUP_JSONL_GRACE_PERIOD`              | `ai-service/app/p2p/constants.py:429`                                       |                |
| `RINGRIFT_P2P_STATE_DIR`                               | `ai-service/app/p2p/constants.py:569`                                       |                |
| `RINGRIFT_P2P_STORAGE_ROOT`                            | `ai-service/app/p2p/constants.py:561`                                       |                |
| `RINGRIFT_P2P_SUSPECT_TIMEOUT`                         | `ai-service/app/p2p/constants.py:58`                                        |                |
| `RINGRIFT_P2P_SYNC_BACKOFF_FACTOR`                     | `ai-service/app/p2p/constants.py:461`                                       |                |
| `RINGRIFT_P2P_SYNC_SPEEDUP_FACTOR`                     | `ai-service/app/p2p/constants.py:460`                                       |                |
| `RINGRIFT_P2P_TARGET_GPU_UTIL_MAX`                     | `ai-service/app/p2p/config.py:104`                                          |                |
| `RINGRIFT_P2P_TARGET_GPU_UTIL_MIN`                     | `ai-service/app/p2p/config.py:101`                                          |                |
| `RINGRIFT_P2P_TRAINING_DATA_SYNC_THRESHOLD_MB`         | `ai-service/app/p2p/constants.py:421`                                       |                |
| `RINGRIFT_P2P_TRAINING_DB_SYNC_BASE`                   | `ai-service/app/p2p/constants.py:455`                                       |                |
| `RINGRIFT_P2P_TRAINING_DB_SYNC_MAX`                    | `ai-service/app/p2p/constants.py:457`                                       |                |
| `RINGRIFT_P2P_TRAINING_DB_SYNC_MIN`                    | `ai-service/app/p2p/constants.py:456`                                       |                |
| `RINGRIFT_P2P_TRAINING_NODE_COUNT`                     | `ai-service/app/p2p/constants.py:435`                                       |                |
| `RINGRIFT_P2P_TRAINING_SYNC_INTERVAL`                  | `ai-service/app/p2p/constants.py:436`                                       |                |
| `RINGRIFT_P2P_UNIFIED_DISCOVERY_INTERVAL`              | `ai-service/app/p2p/constants.py:494`                                       |                |
| `RINGRIFT_P2P_VERBOSE`                                 | `ai-service/scripts/p2p_orchestrator.py:1388`                               |                |
| `RINGRIFT_P2P_VOTERS`                                  | `ai-service/deploy/deploy_cluster_resilience.py:175`                        |                |
| `RINGRIFT_P2P_VOTER_DEMOTION_FAILURES`                 | `ai-service/app/p2p/constants.py:386`                                       |                |
| `RINGRIFT_P2P_VOTER_HEALTH_THRESHOLD`                  | `ai-service/app/p2p/constants.py:387`                                       |                |
| `RINGRIFT_P2P_VOTER_MIN_QUORUM`                        | `ai-service/app/p2p/constants.py:226`                                       |                |
| `RINGRIFT_P2P_VOTER_PROMOTION_UPTIME`                  | `ai-service/app/p2p/constants.py:388`                                       |                |
| `RINGRIFT_PARITY`                                      | `ai-service/app/coordination/parity_validation_daemon.py:61`                |                |
| `RINGRIFT_PARITY_DATA_DIR`                             | `ai-service/tests/unit/coordination/test_parity_validation_daemon.py:52`    | test-only      |
| `RINGRIFT_PARITY_MAX_GAMES_PER_DB`                     | `ai-service/tests/unit/coordination/test_parity_validation_daemon.py:53`    | test-only      |
| `RINGRIFT_PATH`                                        | `ai-service/deploy/scripts/ringrift-p2p-start.sh:14`                        |                |
| `RINGRIFT_PEAK_HOURS_END`                              | `ai-service/app/config/coordination_defaults.py:872`                        |                |
| `RINGRIFT_PEAK_HOURS_START`                            | `ai-service/app/config/coordination_defaults.py:871`                        |                |
| `RINGRIFT_PEER_BOOTSTRAP_INTERVAL`                     | `ai-service/app/config/coordination_defaults.py:2018`                       |                |
| `RINGRIFT_PEER_ELECTION_TIMEOUT`                       | `ai-service/app/config/coordination_defaults.py:2015`                       |                |
| `RINGRIFT_PEER_GOSSIP_INTERVAL`                        | `ai-service/app/config/coordination_defaults.py:2009`                       |                |
| `RINGRIFT_PEER_HEARTBEAT_INTERVAL`                     | `ai-service/app/config/coordination_defaults.py:2003`                       |                |
| `RINGRIFT_PEER_MANIFEST_TIMEOUT`                       | `ai-service/app/config/coordination_defaults.py:2012`                       |                |
| `RINGRIFT_PEER_RETRY_DEAD_INTERVAL`                    | `ai-service/app/config/coordination_defaults.py:2024`                       |                |
| `RINGRIFT_PEER_SUSPECT_TIMEOUT`                        | `ai-service/app/config/coordination_defaults.py:2021`                       |                |
| `RINGRIFT_PEER_TIMEOUT`                                | `ai-service/app/config/coordination_defaults.py:2006`                       |                |
| `RINGRIFT_PER_FILE_TIMEOUT`                            | `ai-service/app/config/coordination_defaults.py:1551`                       |                |
| `RINGRIFT_PROCESS_FORCE_KILL_DELAY`                    | `ai-service/scripts/lib/process.py:54`                                      |                |
| `RINGRIFT_PROCESS_GRACE_PERIOD`                        | `ai-service/scripts/lib/process.py:53`                                      |                |
| `RINGRIFT_PROSUMER_MAX_SELFPLAY`                       | `ai-service/app/config/coordination_defaults.py:527`                        |                |
| `RINGRIFT_PROVIDER_CHECK_INTERVAL`                     | `ai-service/app/config/coordination_defaults.py:1854`                       |                |
| `RINGRIFT_PROVIDER_ERROR_COOLDOWN`                     | `ai-service/app/config/coordination_defaults.py:1866`                       |                |
| `RINGRIFT_PROVIDER_MAX_RETRIES`                        | `ai-service/app/config/coordination_defaults.py:1863`                       |                |
| `RINGRIFT_PROVIDER_RETRY_DELAY`                        | `ai-service/app/config/coordination_defaults.py:1860`                       |                |
| `RINGRIFT_PROVIDER_TIMEOUT`                            | `ai-service/app/config/coordination_defaults.py:1857`                       |                |
| `RINGRIFT_PYTHON_RULES_HTTP_INTEGRATION`               | `tests/integration/PythonRulesClient.live.integration.test.ts:12`           | test-only      |
| `RINGRIFT_PYTHON__`                                    | `ai-service/scripts/install_launchd_services.sh:97`                         | prefix/pattern |
| `RINGRIFT_QUALITY_GATE_THRESHOLD`                      | `ai-service/app/coordination/pipeline_actions.py:104`                       |                |
| `RINGRIFT_QUEUE_BACKPRESSURE_THRESHOLD`                | `ai-service/app/config/coordination_defaults.py:1733`                       |                |
| `RINGRIFT_QUEUE_EVAL_HARD`                             | `ai-service/app/config/coordination_defaults.py:808`                        |                |
| `RINGRIFT_QUEUE_EVAL_SOFT`                             | `ai-service/app/config/coordination_defaults.py:807`                        |                |
| `RINGRIFT_QUEUE_EVAL_TARGET`                           | `ai-service/app/config/coordination_defaults.py:809`                        |                |
| `RINGRIFT_QUEUE_GAMES_HARD`                            | `ai-service/app/config/coordination_defaults.py:803`                        |                |
| `RINGRIFT_QUEUE_GAMES_SOFT`                            | `ai-service/app/config/coordination_defaults.py:802`                        |                |
| `RINGRIFT_QUEUE_GAMES_TARGET`                          | `ai-service/app/config/coordination_defaults.py:804`                        |                |
| `RINGRIFT_QUEUE_MONITOR_DB`                            | `ai-service/tests/test_coordination_integration.py:138`                     | test-only      |
| `RINGRIFT_QUEUE_SYNC_HARD`                             | `ai-service/app/config/coordination_defaults.py:813`                        |                |
| `RINGRIFT_QUEUE_SYNC_SOFT`                             | `ai-service/app/config/coordination_defaults.py:812`                        |                |
| `RINGRIFT_QUEUE_SYNC_TARGET`                           | `ai-service/app/config/coordination_defaults.py:814`                        |                |
| `RINGRIFT_QUEUE_TRAINING_HARD`                         | `ai-service/app/config/coordination_defaults.py:798`                        |                |
| `RINGRIFT_QUEUE_TRAINING_SOFT`                         | `ai-service/app/config/coordination_defaults.py:797`                        |                |
| `RINGRIFT_QUEUE_TRAINING_TARGET`                       | `ai-service/app/config/coordination_defaults.py:799`                        |                |
| `RINGRIFT_RAFT_BATCH_SIZE`                             | `ai-service/app/config/coordination_defaults.py:1044`                       |                |
| `RINGRIFT_RAFT_COMPACTION_THRESHOLD`                   | `ai-service/app/config/coordination_defaults.py:1047`                       |                |
| `RINGRIFT_RAFT_ELECTION_TIMEOUT`                       | `ai-service/app/config/coordination_defaults.py:1038`                       |                |
| `RINGRIFT_RAFT_FALLBACK`                               | `ai-service/app/config/coordination_defaults.py:1080`                       |                |
| `RINGRIFT_RAFT_HEARTBEAT_INTERVAL`                     | `ai-service/app/config/coordination_defaults.py:1041`                       |                |
| `RINGRIFT_RAFT_SNAPSHOT_CHUNK_SIZE`                    | `ai-service/app/config/coordination_defaults.py:1050`                       |                |
| `RINGRIFT_RECOVERY_STACK_STRIKE_V1`                    | `ai-service/app/rules/recovery.py:64`                                       |                |
| `RINGRIFT_RELAY_HOST`                                  | `ai-service/scripts/node_resilience.py:282`                                 |                |
| `RINGRIFT_REMOTE_PARITY_VALIDATION`                    | `ai-service/scripts/run_distributed_selfplay_soak.py:771`                   |                |
| `RINGRIFT_REPETITION_THRESHOLD`                        | `ai-service/app/training/env.py:108`                                        |                |
| `RINGRIFT_REPLICATION_CHECK_INTERVAL`                  | `ai-service/app/coordination/unified_data_plane_daemon.py:138`              |                |
| `RINGRIFT_REPO`                                        | `ai-service/scripts/setup_vast_networking.sh:32`                            |                |
| `RINGRIFT_REPO_ROOT__`                                 | `ai-service/scripts/install_launchd_services.sh:96`                         | prefix/pattern |
| `RINGRIFT_RESOURCE_CHECK_INTERVAL`                     | `ai-service/app/config/coordination_defaults.py:1301`                       |                |
| `RINGRIFT_RESOURCE_UPDATE_INTERVAL`                    | `ai-service/app/config/coordination_defaults.py:573`                        |                |
| `RINGRIFT_RESOURCE_WAIT_TIMEOUT`                       | `ai-service/app/config/coordination_defaults.py:1545`                       |                |
| `RINGRIFT_RETRY_BACKOFF_MULTIPLIER`                    | `ai-service/app/config/coordination_defaults.py:1572`                       |                |
| `RINGRIFT_RETRY_BASE_DELAY`                            | `ai-service/app/config/coordination_defaults.py:1566`                       |                |
| `RINGRIFT_RETRY_JITTER_FACTOR`                         | `ai-service/app/config/coordination_defaults.py:1575`                       |                |
| `RINGRIFT_RETRY_MAX_DELAY`                             | `ai-service/app/config/coordination_defaults.py:1569`                       |                |
| `RINGRIFT_ROUTE53_ZONE_ID`                             | `ai-service/scripts/setup_aws_cluster_proxy.sh:228`                         |                |
| `RINGRIFT_S3_BACKUP_ENABLED`                           | `ai-service/app/coordination/unified_data_plane_daemon.py:140`              |                |
| `RINGRIFT_S3_BANDWIDTH_LIMIT`                          | `ai-service/app/coordination/s3_node_sync_daemon.py:124`                    |                |
| `RINGRIFT_S3_NODE_SYNC`                                | `ai-service/app/coordination/s3_node_sync_daemon.py:18`                     |                |
| `RINGRIFT_S3_PREFIX`                                   | `ai-service/scripts/p2p/handlers/sync.py:238`                               |                |
| `RINGRIFT_S3_PULL_MODELS`                              | `ai-service/app/coordination/s3_node_sync_daemon.py:108`                    |                |
| `RINGRIFT_S3_PULL_NPZ`                                 | `ai-service/app/coordination/s3_node_sync_daemon.py:25`                     |                |
| `RINGRIFT_S3_PUSH_GAMES`                               | `ai-service/app/coordination/s3_node_sync_daemon.py:23`                     |                |
| `RINGRIFT_S3_PUSH_MODELS`                              | `ai-service/app/coordination/s3_node_sync_daemon.py:24`                     |                |
| `RINGRIFT_S3_PUSH_NPZ`                                 | `ai-service/app/coordination/s3_node_sync_daemon.py:100`                    |                |
| `RINGRIFT_S3_SYNC_INTERVAL`                            | `ai-service/app/coordination/s3_node_sync_daemon.py:22`                     |                |
| `RINGRIFT_S3_TRAINING_PREFIX`                          | `ai-service/app/coordination/training_data_manifest.py:151`                 |                |
| `RINGRIFT_SANDBOX_`                                    | `scripts/validate-deployment-config.ts:513`                                 | prefix/pattern |
| `RINGRIFT_SANDBOX_AI_META__`                           | `src/client/sandbox/sandboxAiDiagnostics.ts:45`                             | prefix/pattern |
| `RINGRIFT_SANDBOX_ANIMATION_DEBUG`                     | `src/shared/utils/envFlags.ts:106`                                          |                |
| `RINGRIFT_SANDBOX_LPS_DEBUG`                           | `src/shared/utils/envFlags.ts:92`                                           |                |
| `RINGRIFT_SANDBOX_TRACE__`                             | `src/client/contexts/SandboxContext.tsx:272`                                | prefix/pattern |
| `RINGRIFT_SB_BENCH_ITERS`                              | `ai-service/scripts/benchmark_search_board_large_board.py:18`               |                |
| `RINGRIFT_SCALE_DOWN_COOLDOWN_MINUTES`                 | `ai-service/app/config/coordination_defaults.py:834`                        |                |
| `RINGRIFT_SCALE_DOWN_IDLE_MINUTES`                     | `ai-service/app/config/coordination_defaults.py:832`                        |                |
| `RINGRIFT_SCALE_DOWN_QUEUE_DEPTH`                      | `ai-service/app/config/coordination_defaults.py:829`                        |                |
| `RINGRIFT_SCALE_UP_COOLDOWN_MINUTES`                   | `ai-service/app/config/coordination_defaults.py:833`                        |                |
| `RINGRIFT_SCALE_UP_QUEUE_DEPTH`                        | `ai-service/app/config/coordination_defaults.py:828`                        |                |
| `RINGRIFT_SCHEDULER_CPU_ONLY_JOB_MIN_CPUS`             | `ai-service/app/p2p/constants.py:521`                                       |                |
| `RINGRIFT_SCHEDULER_DB`                                | `ai-service/tests/test_coordination_integration.py:109`                     | test-only      |
| `RINGRIFT_SCHEDULER_EXPLORATION_BOOST_DURATION`        | `ai-service/app/p2p/constants.py:501`                                       |                |
| `RINGRIFT_SCHEDULER_HIGH_PRIORITY_THRESHOLD`           | `ai-service/app/p2p/constants.py:514`                                       |                |
| `RINGRIFT_SCHEDULER_PLATEAU_CLEAR_WIN_RATE`            | `ai-service/app/p2p/constants.py:510`                                       |                |
| `RINGRIFT_SCHEDULER_PLATEAU_PENALTY_DURATION`          | `ai-service/app/p2p/constants.py:504`                                       |                |
| `RINGRIFT_SCHEDULER_PRIORITY_CHANGE_THRESHOLD`         | `ai-service/app/p2p/constants.py:513`                                       |                |
| `RINGRIFT_SCHEDULER_PROMOTION_PENALTY_CRITICAL`        | `ai-service/app/p2p/constants.py:524`                                       |                |
| `RINGRIFT_SCHEDULER_PROMOTION_PENALTY_FACTOR_CRITICAL` | `ai-service/app/p2p/constants.py:529`                                       |                |
| `RINGRIFT_SCHEDULER_PROMOTION_PENALTY_FACTOR_MULTIPLE` | `ai-service/app/p2p/constants.py:530`                                       |                |
| `RINGRIFT_SCHEDULER_PROMOTION_PENALTY_FACTOR_SINGLE`   | `ai-service/app/p2p/constants.py:531`                                       |                |
| `RINGRIFT_SCHEDULER_PROMOTION_PENALTY_MULTIPLE`        | `ai-service/app/p2p/constants.py:525`                                       |                |
| `RINGRIFT_SCHEDULER_PROMOTION_PENALTY_SINGLE`          | `ai-service/app/p2p/constants.py:526`                                       |                |
| `RINGRIFT_SCHEDULER_RELATIVE_CHANGE_THRESHOLD`         | `ai-service/app/p2p/constants.py:515`                                       |                |
| `RINGRIFT_SCHEDULER_TARGET_CHANGE_THRESHOLD`           | `ai-service/app/p2p/constants.py:518`                                       |                |
| `RINGRIFT_SCHEDULER_TRAINING_BOOST_DURATION`           | `ai-service/app/p2p/constants.py:507`                                       |                |
| `RINGRIFT_SCP_TIMEOUT`                                 | `ai-service/app/config/coordination_defaults.py:1936`                       |                |
| `RINGRIFT_SEED`                                        | `ai-service/app/training/seed_utils.py:130`                                 |                |
| `RINGRIFT_SELFPLAY_`                                   | `ai-service/app/config/training_config.py:315`                              | prefix/pattern |
| `RINGRIFT_SELFPLAY_BOARD`                              | `ai-service/scripts/ops/maintain_selfplay_load.sh:13`                       |                |
| `RINGRIFT_SELFPLAY_BUDGET`                             | `ai-service/scripts/ops/maintain_selfplay_load.sh:16`                       |                |
| `RINGRIFT_SELFPLAY_ENGINE`                             | `ai-service/scripts/ops/maintain_selfplay_load.sh:15`                       |                |
| `RINGRIFT_SELFPLAY_GAMES_PER_CONFIG`                   | `ai-service/app/config/coordination_defaults.py:1780`                       |                |
| `RINGRIFT_SELFPLAY_PLAYERS`                            | `ai-service/scripts/ops/maintain_selfplay_load.sh:14`                       |                |
| `RINGRIFT_SELFPLAY_STABILITY_DIFFICULTY_BAND`          | `ai-service/tests/test_self_play_stability.py:56`                           | test-only      |
| `RINGRIFT_SELFPLAY_STABILITY_GAMES`                    | `ai-service/tests/test_self_play_stability.py:13`                           | test-only      |
| `RINGRIFT_SERVER_ENABLE_MUTATOR_FIRST`                 | `ai-service/app/rules/default_engine.py:104`                                |                |
| `RINGRIFT_SKIP_DAEMONS`                                | `ai-service/scripts/cluster_join.sh:24`                                     |                |
| `RINGRIFT_SKIP_MOVE_VALIDATION`                        | `ai-service/app/rules/move_validation.py:91`                                |                |
| `RINGRIFT_SKIP_PHASE_INVARIANT`                        | `ai-service/app/_game_engine_legacy.py:127`                                 |                |
| `RINGRIFT_SPLIT_BRAIN_STEPDOWN`                        | `ai-service/app/config/coordination_defaults.py:1095`                       |                |
| `RINGRIFT_SPLIT_BRAIN_TIMEOUT`                         | `ai-service/app/config/coordination_defaults.py:1091`                       |                |
| `RINGRIFT_SQLITE_`                                     | `ai-service/scripts/p2p/managers/state_manager.py:207`                      | prefix/pattern |
| `RINGRIFT_SQLITE_BUSY_TIMEOUT_MS`                      | `ai-service/app/config/coordination_defaults.py:1470`                       |                |
| `RINGRIFT_SQLITE_HEAVY_TIMEOUT`                        | `ai-service/app/config/coordination_defaults.py:1461`                       |                |
| `RINGRIFT_SQLITE_MERGE_TIMEOUT`                        | `ai-service/app/config/coordination_defaults.py:1464`                       |                |
| `RINGRIFT_SQLITE_QUICK_TIMEOUT`                        | `ai-service/app/config/coordination_defaults.py:1449`                       |                |
| `RINGRIFT_SQLITE_READ_TIMEOUT`                         | `ai-service/app/config/coordination_defaults.py:1452`                       |                |
| `RINGRIFT_SQLITE_STANDARD_TIMEOUT`                     | `ai-service/app/config/coordination_defaults.py:1455`                       |                |
| `RINGRIFT_SQLITE_WAL_CHECKPOINT`                       | `ai-service/app/config/coordination_defaults.py:1467`                       |                |
| `RINGRIFT_SQLITE_WRITE_TIMEOUT`                        | `ai-service/app/config/coordination_defaults.py:1458`                       |                |
| `RINGRIFT_SSH_BASE_DELAY`                              | `ai-service/app/config/coordination_defaults.py:689`                        |                |
| `RINGRIFT_SSH_CHECK_INTERVAL`                          | `ai-service/app/config/coordination_defaults.py:2134`                       |                |
| `RINGRIFT_SSH_COMMAND_TIMEOUT`                         | `ai-service/app/config/coordination_defaults.py:1930`                       |                |
| `RINGRIFT_SSH_HEALTH_CHECK_TIMEOUT`                    | `ai-service/app/config/coordination_defaults.py:1945`                       |                |
| `RINGRIFT_SSH_LONG_COMMAND_TIMEOUT`                    | `ai-service/app/config/coordination_defaults.py:1933`                       |                |
| `RINGRIFT_SSH_MAX_DELAY`                               | `ai-service/app/config/coordination_defaults.py:690`                        |                |
| `RINGRIFT_STAGING_`                                    | `ai-service/scripts/run_improvement_loop.py:2163`                           | prefix/pattern |
| `RINGRIFT_STUCK_JOB_THRESHOLD`                         | `ai-service/app/config/coordination_defaults.py:1737`                       |                |
| `RINGRIFT_SUBDIR`                                      | `ai-service/app/utils/ramdrive.py:49`                                       |                |
| `RINGRIFT_SUBSCRIBER_TIMEOUT`                          | `ai-service/app/config/coordination_defaults.py:1814`                       |                |
| `RINGRIFT_SWIM_GOSSIP_FANOUT`                          | `ai-service/app/config/coordination_defaults.py:1028`                       |                |
| `RINGRIFT_SWIM_PING_TIMEOUT`                           | `ai-service/app/config/coordination_defaults.py:1022`                       |                |
| `RINGRIFT_SWIM_RETRANSMIT_LIMIT`                       | `ai-service/app/config/coordination_defaults.py:1031`                       |                |
| `RINGRIFT_SYNC_BASE_DELAY`                             | `ai-service/scripts/sync_models.py:164`                                     |                |
| `RINGRIFT_SYNC_CRITICAL_STALE`                         | `ai-service/app/config/coordination_defaults.py:886`                        |                |
| `RINGRIFT_SYNC_DEFAULT_BW_KBPS`                        | `ai-service/app/coordination/sync_bandwidth.py:167`                         |                |
| `RINGRIFT_SYNC_DEFAULT_CHUNK_SIZE`                     | `ai-service/app/config/coordination_defaults.py:1906`                       |                |
| `RINGRIFT_SYNC_DIR`                                    | `ai-service/scripts/data_aggregator.py:165`                                 |                |
| `RINGRIFT_SYNC_FRESHNESS_INTERVAL`                     | `ai-service/app/config/coordination_defaults.py:889`                        |                |
| `RINGRIFT_SYNC_LARGE_CHUNK_SIZE`                       | `ai-service/app/config/coordination_defaults.py:1909`                       |                |
| `RINGRIFT_SYNC_MAX_BW_KBPS`                            | `ai-service/app/coordination/sync_bandwidth.py:168`                         |                |
| `RINGRIFT_SYNC_MAX_PER_HOST`                           | `ai-service/app/coordination/sync_bandwidth.py:178`                         |                |
| `RINGRIFT_SYNC_MAX_RETRIES`                            | `ai-service/scripts/sync_models.py:163`                                     |                |
| `RINGRIFT_SYNC_MAX_TOTAL`                              | `ai-service/app/coordination/sync_bandwidth.py:179`                         |                |
| `RINGRIFT_SYNC_MUTEX_DB`                               | `ai-service/tests/test_coordination_integration.py:164`                     | test-only      |
| `RINGRIFT_SYNC_PER_HOST_KBPS`                          | `ai-service/app/coordination/sync_bandwidth.py:173`                         |                |
| `RINGRIFT_SYNC_PUSH`                                   | `ai-service/app/coordination/sync_push_daemon.py:83`                        |                |
| `RINGRIFT_SYNC_PUSH_CLEANUP_THRESHOLD`                 | `ai-service/tests/unit/coordination/test_sync_push_daemon.py:135`           | test-only      |
| `RINGRIFT_SYNC_PUSH_INTERVAL`                          | `ai-service/tests/unit/coordination/test_sync_push_daemon.py:137`           | test-only      |
| `RINGRIFT_SYNC_PUSH_MIN_COPIES`                        | `ai-service/tests/unit/coordination/test_sync_push_daemon.py:136`           | test-only      |
| `RINGRIFT_SYNC_PUSH_THRESHOLD`                         | `ai-service/tests/unit/coordination/test_sync_push_daemon.py:133`           | test-only      |
| `RINGRIFT_SYNC_PUSH_URGENT_THRESHOLD`                  | `ai-service/tests/unit/coordination/test_sync_push_daemon.py:134`           | test-only      |
| `RINGRIFT_SYNC_STALL_TIMEOUT`                          | `ai-service/app/config/coordination_defaults.py:143`                        |                |
| `RINGRIFT_SYNC_TOTAL_KBPS`                             | `ai-service/app/coordination/sync_bandwidth.py:174`                         |                |
| `RINGRIFT_TARGET_REPLICATION`                          | `ai-service/app/coordination/sync_planner_v2.py:223`                        |                |
| `RINGRIFT_TARGET_SELFPLAY_JOBS`                        | `ai-service/scripts/ops/maintain_selfplay_load.sh:12`                       |                |
| `RINGRIFT_TASK_CLEANUP_INTERVAL`                       | `ai-service/app/config/coordination_defaults.py:625`                        |                |
| `RINGRIFT_TASK_HEARTBEAT_TIMEOUT`                      | `ai-service/app/config/coordination_defaults.py:616`                        |                |
| `RINGRIFT_TASK_MAX_HISTORY`                            | `ai-service/app/config/coordination_defaults.py:622`                        |                |
| `RINGRIFT_TASK_ORPHAN_GRACE`                           | `ai-service/app/config/coordination_defaults.py:619`                        |                |
| `RINGRIFT_TERMINATION_GRACE_PERIOD`                    | `ai-service/app/config/coordination_defaults.py:1840`                       |                |
| `RINGRIFT_TEST_BOOL`                                   | `ai-service/tests/unit/config/test_env.py:218`                              | test-only      |
| `RINGRIFT_TEST_FLAG`                                   | `tests/unit/envFlags.test.ts:24`                                            | test-only      |
| `RINGRIFT_TEST_FLOAT`                                  | `ai-service/tests/unit/config/test_env.py:210`                              | test-only      |
| `RINGRIFT_TEST_FLOAT_KEY`                              | `ai-service/tests/unit/config/test_coordination_defaults.py:73`             | test-only      |
| `RINGRIFT_TEST_INT`                                    | `ai-service/tests/unit/config/test_env.py:199`                              | test-only      |
| `RINGRIFT_TEST_INT_KEY`                                | `ai-service/tests/unit/config/test_coordination_defaults.py:58`             | test-only      |
| `RINGRIFT_TEST_NONEXISTENT_FLOAT_KEY`                  | `ai-service/tests/unit/config/test_coordination_defaults.py:66`             | test-only      |
| `RINGRIFT_TEST_NONEXISTENT_KEY_12345`                  | `ai-service/tests/unit/config/test_coordination_defaults.py:51`             | test-only      |
| `RINGRIFT_TEST_SET`                                    | `ai-service/tests/unit/config/test_env.py:232`                              | test-only      |
| `RINGRIFT_TEST_VAR`                                    | `ai-service/tests/unit/config/test_env.py:182`                              | test-only      |
| `RINGRIFT_THREAD_JOIN_TIMEOUT`                         | `ai-service/app/config/coordination_defaults.py:1533`                       |                |
| `RINGRIFT_TIMEOUT_`                                    | `ai-service/app/config/timeout_config.py:16`                                | prefix/pattern |
| `RINGRIFT_TIMEOUT_MAX_MULTIPLIER`                      | `ai-service/app/config/coordination_defaults.py:1063`                       |                |
| `RINGRIFT_TIMEOUT_MIN_MULTIPLIER`                      | `ai-service/app/config/coordination_defaults.py:1060`                       |                |
| `RINGRIFT_TIMEOUT_SYNC_INTERVAL`                       | `ai-service/app/config/timeout_config.py:18`                                |                |
| `RINGRIFT_TOURNAMENT_ENTRYPOINT`                       | `ai-service/scripts/run_tournament.py:492`                                  |                |
| `RINGRIFT_TRACE_LPS`                                   | `scripts/selfplay-db-ts-replay.ts:1356`                                     |                |
| `RINGRIFT_TRAINING_ACTIVITY`                           | `ai-service/app/coordination/training_activity_daemon.py:63`                |                |
| `RINGRIFT_TRAINING_ACTIVITY_ENABLED`                   | `ai-service/tests/unit/coordination/test_training_activity_daemon.py:82`    | test-only      |
| `RINGRIFT_TRAINING_ACTIVITY_TRIGGER_SYNC`              | `ai-service/tests/unit/coordination/test_training_activity_daemon.py:83`    | test-only      |
| `RINGRIFT_TRAINING_DEADLINE`                           | `ai-service/app/coordination/sync_planner_v2.py:226`                        |                |
| `RINGRIFT_TRAINING_DIR`                                | `ai-service/app/coordination/training_data_manifest.py:156`                 |                |
| `RINGRIFT_TRAINING_EPOCHS`                             | `ai-service/app/training/curriculum.py:209`                                 |                |
| `RINGRIFT_TRAINING_JOB_TIMEOUT`                        | `ai-service/app/config/coordination_defaults.py:1542`                       |                |
| `RINGRIFT_TRANSFER_BASE_DELAY`                         | `ai-service/app/config/coordination_defaults.py:684`                        |                |
| `RINGRIFT_TRANSFER_MAX_DELAY`                          | `ai-service/app/config/coordination_defaults.py:685`                        |                |
| `RINGRIFT_TRANSFER_MAX_RETRIES`                        | `ai-service/app/config/coordination_defaults.py:683`                        |                |
| `RINGRIFT_TS_REPLAY_MINIMAL`                           | `ai-service/scripts/check_ts_python_replay_parity.py:720`                   |                |
| `RINGRIFT_UNHEALTHY_CACHE_TTL`                         | `ai-service/app/config/coordination_defaults.py:454`                        |                |
| `RINGRIFT_URL_FETCH_QUICK_TIMEOUT`                     | `ai-service/app/config/coordination_defaults.py:1521`                       |                |
| `RINGRIFT_URL_FETCH_TIMEOUT`                           | `ai-service/app/config/coordination_defaults.py:1524`                       |                |
| `RINGRIFT_USE_GMO_POLICY`                              | `ai-service/app/ai/gmo_policy_provider.py:5`                                |                |
| `RINGRIFT_USE_HYBRID_D7`                               | `ai-service/app/ai/factory.py:413`                                          |                |
| `RINGRIFT_UTILIZATION_CHECK_INTERVAL`                  | `ai-service/app/config/coordination_defaults.py:2141`                       |                |
| `RINGRIFT_UTILIZATION_UPDATE_INTERVAL`                 | `ai-service/app/config/coordination_defaults.py:488`                        |                |
| `RINGRIFT_WATCHDOG`                                    | `ai-service/app/coordination/cluster_watchdog_daemon.py:106`                |                |
| `RINGRIFT_WATCHDOG_`                                   | `ai-service/app/coordination/base_daemon.py:78`                             | prefix/pattern |
| `RINGRIFT_WATCHDOG_ACTIVATION_COOLDOWN`                | `ai-service/app/config/coordination_defaults.py:1361`                       |                |
| `RINGRIFT_WATCHDOG_ENABLED`                            | `ai-service/tests/unit/coordination/test_cluster_watchdog_daemon.py:90`     | test-only      |
| `RINGRIFT_WATCHDOG_INTERVAL`                           | `ai-service/tests/unit/coordination/test_cluster_watchdog_daemon.py:100`    | test-only      |
| `RINGRIFT_WATCHDOG_MAX_ACTIVATIONS`                    | `ai-service/app/config/coordination_defaults.py:1367`                       |                |
| `RINGRIFT_WATCHDOG_MAX_FAILURES`                       | `ai-service/app/config/coordination_defaults.py:1364`                       |                |
| `RINGRIFT_WATCHDOG_MIN_GPU`                            | `ai-service/tests/unit/coordination/test_cluster_watchdog_daemon.py:106`    | test-only      |
| `RINGRIFT_WATCHDOG_MIN_GPU_UTIL`                       | `ai-service/app/config/coordination_defaults.py:1358`                       |                |
| `RINGRIFT_WATCHDOG_SSH_TIMEOUT`                        | `ai-service/app/config/coordination_defaults.py:1355`                       |                |
| `RINGRIFT_WEIGHT_STALE_DECAY`                          | `ai-service/app/config/coordination_defaults.py:2108`                       |                |
| `RINGRIFT_WORKER_ID`                                   | `ai-service/scripts/start_aws_worker.sh:24`                                 |                |
| `RINGRIFT_WORKER_PORT`                                 | `ai-service/scripts/start_aws_worker.sh:23`                                 |                |
