# Internal Environment Flags (Auto-Extracted)

This appendix lists environment variables referenced in code but not yet described in
`docs/operations/ENVIRONMENT_VARIABLES.md` or `ai-service/docs/AI_SERVICE_ENVIRONMENT_REFERENCE.md`.
Entries are auto-extracted and may be experimental or internal.

Generated: 2025-12-27

| Name                                                | First Usage (code)                                                         | Notes                        |
| --------------------------------------------------- | -------------------------------------------------------------------------- | ---------------------------- |
| `RINGRIFT_ADMIN_TOKEN`                              | `ai-service/scripts/p2p/handlers/admin.py:97`                              |                              |
| `RINGRIFT_ADVERTISE_PORT`                           | `ai-service/scripts/p2p_orchestrator.py:4163`                              |                              |
| `RINGRIFT_AGGREGATION_INTERVAL`                     | `ai-service/app/config/coordination_defaults.py:951`                       |                              |
| `RINGRIFT_ALERT_COOLDOWN`                           | `ai-service/app/config/coordination_defaults.py:551`                       |                              |
| `RINGRIFT_ALERT_LEVEL`                              | `ai-service/scripts/p2p_orchestrator.py:1691`                              |                              |
| `RINGRIFT_ARBITER_URL`                              | `ai-service/app/p2p/constants.py:259`                                      |                              |
| `RINGRIFT_ARCHIVE_DIR`                              | `ai-service/app/coordination/maintenance_daemon.py:72`                     |                              |
| `RINGRIFT_ARCHIVE_S3_BUCKET`                        | `ai-service/app/coordination/maintenance_daemon.py:80`                     |                              |
| `RINGRIFT_ARCHIVE_TO_S3`                            | `ai-service/app/coordination/maintenance_daemon.py:77`                     |                              |
| `RINGRIFT_AUTO_ROLLBACK`                            | `ai-service/scripts/p2p_orchestrator.py:16209`                             |                              |
| `RINGRIFT_AUTO_SYNC`                                | `ai-service/scripts/verify_nfs_sync.py:408`                                |                              |
| `RINGRIFT_AUTO_UPDATE`                              | `ai-service/scripts/p2p/handlers/admin.py:19`                              |                              |
| `RINGRIFT_AUTOSCALE_MAX_WORKERS`                    | `ai-service/scripts/p2p_orchestrator.py:16237`                             |                              |
| `RINGRIFT_AUTOSCALE_MIN_WORKERS`                    | `ai-service/scripts/p2p_orchestrator.py:16238`                             |                              |
| `RINGRIFT_AUTOSCALE_SCALE_DOWN_GPH`                 | `ai-service/scripts/p2p_orchestrator.py:16240`                             |                              |
| `RINGRIFT_AUTOSCALE_SCALE_UP_GPH`                   | `ai-service/scripts/p2p_orchestrator.py:16239`                             |                              |
| `RINGRIFT_AUTOSCALE_TARGET_FRESHNESS_HOURS`         | `ai-service/scripts/p2p_orchestrator.py:16241`                             |                              |
| `RINGRIFT_BANDWIDTH_DB`                             | `ai-service/tests/test_coordination_integration.py:188`                    | test-only (first occurrence) |
| `RINGRIFT_BP_CACHE_TTL`                             | `ai-service/app/config/coordination_defaults.py:900`                       |                              |
| `RINGRIFT_BP_COOLDOWN`                              | `ai-service/app/config/coordination_defaults.py:903`                       |                              |
| `RINGRIFT_BP_QUEUE_CRITICAL`                        | `ai-service/app/config/coordination_defaults.py:887`                       |                              |
| `RINGRIFT_BP_QUEUE_HIGH`                            | `ai-service/app/config/coordination_defaults.py:886`                       |                              |
| `RINGRIFT_BP_QUEUE_LOW`                             | `ai-service/app/config/coordination_defaults.py:884`                       |                              |
| `RINGRIFT_BP_QUEUE_MEDIUM`                          | `ai-service/app/config/coordination_defaults.py:885`                       |                              |
| `RINGRIFT_BP_WEIGHT_DISK`                           | `ai-service/app/config/coordination_defaults.py:879`                       |                              |
| `RINGRIFT_BP_WEIGHT_MEMORY`                         | `ai-service/app/config/coordination_defaults.py:881`                       |                              |
| `RINGRIFT_BP_WEIGHT_QUEUE`                          | `ai-service/app/config/coordination_defaults.py:877`                       |                              |
| `RINGRIFT_BP_WEIGHT_SYNC`                           | `ai-service/app/config/coordination_defaults.py:880`                       |                              |
| `RINGRIFT_BP_WEIGHT_TRAINING`                       | `ai-service/app/config/coordination_defaults.py:878`                       |                              |
| `RINGRIFT_CANONICAL_DIR`                            | `ai-service/app/coordination/data_consolidation_daemon.py:80`              |                              |
| `RINGRIFT_CLEANUP_COOLDOWN`                         | `ai-service/app/config/coordination_defaults.py:948`                       |                              |
| `RINGRIFT_CLUSTER_API`                              | `ai-service/app/config/env.py:842`                                         |                              |
| `RINGRIFT_CLUSTER_CHECK_INTERVAL`                   | `ai-service/app/config/coordination_defaults.py:545`                       |                              |
| `RINGRIFT_CLUSTER_HOSTS`                            | `ai-service/scripts/deploy_lps_ablation.py:22`                             |                              |
| `RINGRIFT_CLUSTER_NAME`                             | `ai-service/app/p2p/constants.py:439`                                      |                              |
| `RINGRIFT_CMAES_`                                   | `ai-service/app/config/training_config.py:41`                              | prefix/pattern               |
| `RINGRIFT_CNN_V2_VERSION`                           | `ai-service/app/training/model_versioning.py:301`                          |                              |
| `RINGRIFT_CNN_V3_VERSION`                           | `ai-service/app/training/model_versioning.py:302`                          |                              |
| `RINGRIFT_CNN_V4_VERSION`                           | `ai-service/app/training/model_versioning.py:303`                          |                              |
| `RINGRIFT_CONNECT_BASE_DELAY`                       | `ai-service/app/config/coordination_defaults.py:496`                       |                              |
| `RINGRIFT_CONNECT_MAX_DELAY`                        | `ai-service/app/config/coordination_defaults.py:497`                       |                              |
| `RINGRIFT_CONNECT_MAX_RETRIES`                      | `ai-service/app/config/coordination_defaults.py:495`                       |                              |
| `RINGRIFT_CONTINUOUS_COOLDOWN`                      | `ai-service/scripts/run_continuous_training.py:31`                         |                              |
| `RINGRIFT_CONTINUOUS_ENGINE`                        | `ai-service/scripts/run_continuous_training.py:30`                         |                              |
| `RINGRIFT_CONTINUOUS_GAMES`                         | `ai-service/scripts/run_continuous_training.py:29`                         |                              |
| `RINGRIFT_COORDINATOR_DEGRADED_COOLDOWN`            | `ai-service/app/config/coordination_defaults.py:1296`                      |                              |
| `RINGRIFT_COORDINATOR_HEARTBEAT_STALE_THRESHOLD`    | `ai-service/app/config/coordination_defaults.py:1292`                      |                              |
| `RINGRIFT_COORDINATOR_INIT_FAILURE_MAX_RETRIES`     | `ai-service/app/config/coordination_defaults.py:1300`                      |                              |
| `RINGRIFT_CPU_CRITICAL`                             | `ai-service/scripts/lib/alerts.py:220`                                     |                              |
| `RINGRIFT_CPU_WARNING`                              | `ai-service/scripts/lib/alerts.py:219`                                     |                              |
| `RINGRIFT_CRITICAL_GAME_THRESHOLD`                  | `ai-service/app/config/coordination_defaults.py:1349`                      |                              |
| `RINGRIFT_CROSS_PROCESS_RETENTION_HOURS`            | `ai-service/app/config/coordination_defaults.py:1394`                      |                              |
| `RINGRIFT_DAEMON_CHECK_INTERVAL`                    | `ai-service/app/config/coordination_defaults.py:458`                       |                              |
| `RINGRIFT_DAEMON_CRITICAL_CHECK_INTERVAL`           | `ai-service/app/config/coordination_defaults.py:1031`                      |                              |
| `RINGRIFT_DAEMON_ERROR_BACKOFF_BASE`                | `ai-service/app/config/coordination_defaults.py:461`                       |                              |
| `RINGRIFT_DAEMON_ERROR_BACKOFF_MAX`                 | `ai-service/app/config/coordination_defaults.py:464`                       |                              |
| `RINGRIFT_DAEMON_ERROR_RATE_THRESHOLD`              | `ai-service/app/config/coordination_defaults.py:476`                       |                              |
| `RINGRIFT_DAEMON_HEALTH_INTERVAL`                   | `ai-service/app/config/coordination_defaults.py:1027`                      |                              |
| `RINGRIFT_DAEMON_HEALTH_TIMEOUT`                    | `ai-service/app/config/coordination_defaults.py:473`                       |                              |
| `RINGRIFT_DAEMON_MAX_CONSECUTIVE_ERRORS`            | `ai-service/app/config/coordination_defaults.py:467`                       |                              |
| `RINGRIFT_DAEMON_MAX_FAILURES`                      | `ai-service/app/config/coordination_defaults.py:1035`                      |                              |
| `RINGRIFT_DAEMON_MIN_CYCLES_ERROR_CHECK`            | `ai-service/app/config/coordination_defaults.py:479`                       |                              |
| `RINGRIFT_DAEMON_RESTART_BACKOFF`                   | `ai-service/app/config/coordination_defaults.py:1038`                      |                              |
| `RINGRIFT_DAEMON_RESTART_BACKOFF_MAX`               | `ai-service/app/config/coordination_defaults.py:1041`                      |                              |
| `RINGRIFT_DAEMON_SHUTDOWN_GRACE`                    | `ai-service/app/config/coordination_defaults.py:470`                       |                              |
| `RINGRIFT_DAEMON_SHUTDOWN_TIMEOUT`                  | `ai-service/app/config/coordination_defaults.py:1047`                      |                              |
| `RINGRIFT_DAEMON_STARTUP_TIMEOUT`                   | `ai-service/app/config/coordination_defaults.py:1044`                      |                              |
| `RINGRIFT_DATA_SERVER_PORT`                         | `ai-service/app/config/coordination_defaults.py:757`                       |                              |
| `RINGRIFT_DB_PASSWORD`                              | `tests/unit/runProdPreviewGoNoGo.test.ts:155`                              |                              |
| `RINGRIFT_DEBUG_CAPTURE`                            | `src/shared/engine/core.ts:313`                                            |                              |
| `RINGRIFT_DEBUG_MOVEMENT`                           | `src/shared/engine/core.ts:488`                                            |                              |
| `RINGRIFT_DEBUG_VALIDATION`                         | `src/shared/engine/validators/utils.ts:26`                                 |                              |
| `RINGRIFT_DEFAULT_JOB_TIMEOUT`                      | `ai-service/app/config/coordination_defaults.py:1262`                      |                              |
| `RINGRIFT_DIR`                                      | `ai-service/scripts/disk_monitor.py:680`                                   |                              |
| `RINGRIFT_DISABLE_FSM_VALIDATION`                   | `ai-service/scripts/run_lps_ablation.py:51`                                |                              |
| `RINGRIFT_DISK_CLEANUP_THRESHOLD`                   | `ai-service/app/config/coordination_defaults.py:956`                       |                              |
| `RINGRIFT_DISK_CRITICAL`                            | `ai-service/scripts/lib/alerts.py:216`                                     |                              |
| `RINGRIFT_DISK_CRITICAL_THRESHOLD`                  | `ai-service/app/config/coordination_defaults.py:531`                       |                              |
| `RINGRIFT_DISK_SPACE`                               | `ai-service/app/coordination/disk_space_manager_daemon.py:155`             |                              |
| `RINGRIFT_DISK_SPACE_ENABLE_CLEANUP`                | `ai-service/tests/unit/coordination/test_disk_space_manager_daemon.py:124` | test-only (first occurrence) |
| `RINGRIFT_DISK_SPACE_LOG_RETENTION_DAYS`            | `ai-service/tests/unit/coordination/test_disk_space_manager_daemon.py:123` | test-only (first occurrence) |
| `RINGRIFT_DISK_SPACE_PROACTIVE_THRESHOLD`           | `ai-service/scripts/launch_coordinator_disk_manager.py:29`                 |                              |
| `RINGRIFT_DISK_SPACE_TARGET_USAGE`                  | `ai-service/scripts/launch_coordinator_disk_manager.py:30`                 |                              |
| `RINGRIFT_DISK_SPACE_WARNING_THRESHOLD`             | `ai-service/tests/unit/coordination/test_disk_space_manager_daemon.py:122` | test-only (first occurrence) |
| `RINGRIFT_DISK_WARNING`                             | `ai-service/scripts/lib/alerts.py:215`                                     |                              |
| `RINGRIFT_DISK_WARNING_THRESHOLD`                   | `ai-service/app/config/coordination_defaults.py:530`                       |                              |
| `RINGRIFT_DNS_MAX_RETRIES`                          | `ai-service/app/config/coordination_defaults.py:515`                       |                              |
| `RINGRIFT_DNS_TIMEOUT`                              | `ai-service/app/config/coordination_defaults.py:516`                       |                              |
| `RINGRIFT_E2E_SET_SANDBOX_STALL__`                  | `src/client/contexts/SandboxContext.tsx:327`                               | prefix/pattern               |
| `RINGRIFT_ELO_COORDINATOR`                          | `ai-service/scripts/p2p_orchestrator.py:2167`                              |                              |
| `RINGRIFT_ENABLE_IMPROVEMENT_DAEMON`                | `ai-service/deploy/deploy_cluster_resilience.py:440`                       |                              |
| `RINGRIFT_ENABLE_SANDBOX_INTERNAL_PARITY`           | `tests/parity/Backend_vs_Sandbox.seed5.internalStateParity.test.ts:226`    |                              |
| `RINGRIFT_ENABLED`                                  | `ai-service/tests/unit/coordination/test_base_daemon.py:74`                | test-only (first occurrence) |
| `RINGRIFT_EPHEMERAL_CHECKPOINT_INTERVAL`            | `ai-service/app/config/coordination_defaults.py:1340`                      |                              |
| `RINGRIFT_EVACUATION_THRESHOLD`                     | `ai-service/app/config/coordination_defaults.py:1346`                      |                              |
| `RINGRIFT_EVAL_HEURISTIC_EVAL_MODE`                 | `ai-service/scripts/evaluate_ai_models.py:255`                             |                              |
| `RINGRIFT_EVAL_MOVE_SAMPLE_LIMIT`                   | `ai-service/scripts/evaluate_ai_models.py:256`                             |                              |
| `RINGRIFT_EXPLORATION_BOOST_DURATION`               | `ai-service/app/coordination/selfplay_scheduler.py:1773`                   |                              |
| `RINGRIFT_FALLBACK_SELFPLAY_SCRIPT`                 | `ai-service/scripts/node_resilience.py:1422`                               |                              |
| `RINGRIFT_FLAG`                                     | `tests/unit/envFlags.test.ts:32`                                           |                              |
| `RINGRIFT_FSM_TRACE_DEBUG`                          | `src/shared/utils/envFlags.ts:206`                                         |                              |
| `RINGRIFT_GPU_`                                     | `ai-service/app/training/config.py:51`                                     | prefix/pattern               |
| `RINGRIFT_GPU_MEMORY_CRITICAL`                      | `ai-service/app/config/coordination_defaults.py:539`                       |                              |
| `RINGRIFT_GPU_MEMORY_WARNING`                       | `ai-service/app/config/coordination_defaults.py:538`                       |                              |
| `RINGRIFT_INTERVAL`                                 | `ai-service/tests/unit/coordination/test_base_daemon.py:84`                | test-only (first occurrence) |
| `RINGRIFT_JOB_ORIGIN`                               | `ai-service/scripts/node_resilience.py:806`                                |                              |
| `RINGRIFT_JOB_REAPER_CHECK_INTERVAL`                | `ai-service/app/config/coordination_defaults.py:1259`                      |                              |
| `RINGRIFT_JOB_REAPER_FALLBACK_ENABLED`              | `ai-service/scripts/p2p_orchestrator.py:205`                               |                              |
| `RINGRIFT_JOB_REAPER_SSH_TIMEOUT`                   | `ai-service/app/config/coordination_defaults.py:1271`                      |                              |
| `RINGRIFT_JOB_TIMEOUT_CMAES`                        | `ai-service/app/config/coordination_defaults.py:828`                       |                              |
| `RINGRIFT_JOB_TIMEOUT_CPU_SELFPLAY`                 | `ai-service/app/config/coordination_defaults.py:810`                       |                              |
| `RINGRIFT_JOB_TIMEOUT_DATA_EXPORT`                  | `ai-service/app/config/coordination_defaults.py:819`                       |                              |
| `RINGRIFT_JOB_TIMEOUT_EVALUATION`                   | `ai-service/app/config/coordination_defaults.py:822`                       |                              |
| `RINGRIFT_JOB_TIMEOUT_GPU_SELFPLAY`                 | `ai-service/app/config/coordination_defaults.py:807`                       |                              |
| `RINGRIFT_JOB_TIMEOUT_MODEL_SYNC`                   | `ai-service/app/config/coordination_defaults.py:825`                       |                              |
| `RINGRIFT_JOB_TIMEOUT_PIPELINE_STAGE`               | `ai-service/app/config/coordination_defaults.py:831`                       |                              |
| `RINGRIFT_JOB_TIMEOUT_TOURNAMENT`                   | `ai-service/app/config/coordination_defaults.py:816`                       |                              |
| `RINGRIFT_LAMBDA_IPS`                               | `ai-service/scripts/universal_keepalive.py:62`                             |                              |
| `RINGRIFT_LARGE_BOARD_MCTS_BUDGET`                  | `ai-service/app/config/coordination_defaults.py:1380`                      |                              |
| `RINGRIFT_LATENCY_WINDOW_SIZE`                      | `ai-service/app/config/coordination_defaults.py:1326`                      |                              |
| `RINGRIFT_LEADER_CHECK_TIMEOUT`                     | `ai-service/app/config/coordination_defaults.py:1274`                      |                              |
| `RINGRIFT_LEADER_RETRY_DELAY`                       | `ai-service/app/config/coordination_defaults.py:1277`                      |                              |
| `RINGRIFT_LOG_DIR`                                  | `ai-service/app/config/env_validator.py:197`                               |                              |
| `RINGRIFT_MAX_DEFAULT_WORKERS`                      | `ai-service/app/training/parallel_selfplay.py:659`                         |                              |
| `RINGRIFT_MAX_REASSIGN_ATTEMPTS`                    | `ai-service/app/config/coordination_defaults.py:1265`                      |                              |
| `RINGRIFT_MEMORY_CRITICAL`                          | `ai-service/scripts/lib/alerts.py:218`                                     |                              |
| `RINGRIFT_MEMORY_CRITICAL_THRESHOLD`                | `ai-service/app/config/coordination_defaults.py:535`                       |                              |
| `RINGRIFT_MEMORY_WARNING`                           | `ai-service/scripts/lib/alerts.py:217`                                     |                              |
| `RINGRIFT_MEMORY_WARNING_THRESHOLD`                 | `ai-service/app/config/coordination_defaults.py:534`                       |                              |
| `RINGRIFT_MIN_GAMES_PER_ALLOCATION`                 | `ai-service/app/config/coordination_defaults.py:1367`                      |                              |
| `RINGRIFT_MIN_HEALTHY_FRACTION`                     | `ai-service/app/config/coordination_defaults.py:548`                       |                              |
| `RINGRIFT_MIN_MEMORY_GB_FOR_TASKS`                  | `ai-service/app/config/coordination_defaults.py:1371`                      |                              |
| `RINGRIFT_MIN_MOVES`                                | `ai-service/scripts/jsonl_to_npz.py:1373`                                  |                              |
| `RINGRIFT_NAS_BOARD`                                | `ai-service/scripts/launch_distributed_nas.py:141`                         |                              |
| `RINGRIFT_NAS_PLAYERS`                              | `ai-service/scripts/launch_distributed_nas.py:142`                         |                              |
| `RINGRIFT_NAS_REAL_TRAINING`                        | `ai-service/scripts/launch_distributed_nas.py:140`                         |                              |
| `RINGRIFT_NN_`                                      | `ai-service/app/config/training_config.py:146`                             | prefix/pattern               |
| `RINGRIFT_NN_WARN_TIMEOUT`                          | `ai-service/app/ai/gpu_batch.py:651`                                       |                              |
| `RINGRIFT_NODE_BLACKLIST_DURATION`                  | `ai-service/app/config/coordination_defaults.py:1268`                      |                              |
| `RINGRIFT_NODE_OFFLINE_THRESHOLD`                   | `ai-service/app/config/coordination_defaults.py:542`                       |                              |
| `RINGRIFT_NODE_OVERLOAD_THRESHOLD`                  | `ai-service/app/config/coordination_defaults.py:1323`                      |                              |
| `RINGRIFT_NODE_RECOVERY`                            | `ai-service/app/coordination/node_recovery_daemon.py:99`                   |                              |
| `RINGRIFT_NODE_RESILIENCE_LOCK_FILE`                | `ai-service/scripts/node_resilience.py:85`                                 |                              |
| `RINGRIFT_NODE_RESILIENCE_LOG_FILE`                 | `ai-service/scripts/node_resilience.py:64`                                 |                              |
| `RINGRIFT_NONEXISTENT`                              | `ai-service/tests/unit/config/test_env.py:239`                             | test-only (first occurrence) |
| `RINGRIFT_NUM_PLAYERS`                              | `ai-service/app/training/config.py:952`                                    |                              |
| `RINGRIFT_ORCHESTRATOR_HOST`                        | `ai-service/scripts/train_nnue.py:235`                                     |                              |
| `RINGRIFT_ORCHESTRATOR_PORT`                        | `ai-service/scripts/train_nnue.py:236`                                     |                              |
| `RINGRIFT_ORCHESTRATOR_URL`                         | `ai-service/scripts/hyperparameter_ab_testing.py:54`                       |                              |
| `RINGRIFT_ORPHAN_ALERT_THRESHOLD`                   | `ai-service/app/config/coordination_defaults.py:983`                       |                              |
| `RINGRIFT_ORPHAN_MIN_AGE_HOURS`                     | `ai-service/app/config/coordination_defaults.py:980`                       |                              |
| `RINGRIFT_ORPHAN_MIN_GAMES`                         | `ai-service/app/config/coordination_defaults.py:977`                       |                              |
| `RINGRIFT_ORPHAN_SCAN_INTERVAL`                     | `ai-service/app/config/coordination_defaults.py:974`                       |                              |
| `RINGRIFT_P2P_`                                     | `ai-service/app/p2p/config.py:72`                                          | prefix/pattern               |
| `RINGRIFT_P2P_ADVERTISE_HOST`                       | `ai-service/app/p2p/constants.py:292`                                      |                              |
| `RINGRIFT_P2P_ADVERTISE_PORT`                       | `ai-service/app/p2p/constants.py:293`                                      |                              |
| `RINGRIFT_P2P_AUTO_ASSIGN`                          | `ai-service/app/p2p/constants.py:407`                                      |                              |
| `RINGRIFT_P2P_AUTO_TRAINING_THRESHOLD_MB`           | `ai-service/app/p2p/constants.py:343`                                      |                              |
| `RINGRIFT_P2P_AUTO_WORK_BATCH_SIZE`                 | `ai-service/app/p2p/constants.py:408`                                      |                              |
| `RINGRIFT_P2P_BASE_DELAY`                           | `ai-service/app/config/coordination_defaults.py:511`                       |                              |
| `RINGRIFT_P2P_BIND_ADDR`                            | `ai-service/app/p2p/constants.py:422`                                      |                              |
| `RINGRIFT_P2P_BOOTSTRAP_MAX_SEEDS_PER_RUN`          | `ai-service/scripts/p2p_orchestrator.py:19778`                             |                              |
| `RINGRIFT_P2P_BOOTSTRAP_SEEDS`                      | `ai-service/app/p2p/constants.py:220`                                      |                              |
| `RINGRIFT_P2P_DATA_MANAGEMENT_INTERVAL`             | `ai-service/app/p2p/constants.py:339`                                      |                              |
| `RINGRIFT_P2P_DATA_SYNC_BASE`                       | `ai-service/app/p2p/constants.py:365`                                      |                              |
| `RINGRIFT_P2P_DATA_SYNC_MAX`                        | `ai-service/app/p2p/constants.py:367`                                      |                              |
| `RINGRIFT_P2P_DATA_SYNC_MIN`                        | `ai-service/app/p2p/constants.py:366`                                      |                              |
| `RINGRIFT_P2P_DB_EXPORT_THRESHOLD_MB`               | `ai-service/app/p2p/constants.py:340`                                      |                              |
| `RINGRIFT_P2P_DEFAULT_SEEDS`                        | `ai-service/app/coordination/p2p_integration.py:159`                       |                              |
| `RINGRIFT_P2P_DYNAMIC_VOTER`                        | `ai-service/app/p2p/constants.py:302`                                      |                              |
| `RINGRIFT_P2P_DYNAMIC_VOTER_MAX_QUORUM`             | `ai-service/app/p2p/constants.py:305`                                      |                              |
| `RINGRIFT_P2P_DYNAMIC_VOTER_MIN`                    | `ai-service/app/p2p/constants.py:303`                                      |                              |
| `RINGRIFT_P2P_DYNAMIC_VOTER_TARGET`                 | `ai-service/app/p2p/constants.py:304`                                      |                              |
| `RINGRIFT_P2P_ELECTION_TIMEOUT`                     | `ai-service/app/config/coordination_defaults.py:769`                       |                              |
| `RINGRIFT_P2P_ENDPOINT`                             | `ai-service/scripts/serf_event_handler.py:50`                              |                              |
| `RINGRIFT_P2P_GIT_BRANCH`                           | `ai-service/app/p2p/constants.py:322`                                      |                              |
| `RINGRIFT_P2P_GIT_REMOTE`                           | `ai-service/app/p2p/constants.py:323`                                      |                              |
| `RINGRIFT_P2P_GOSSIP_FANOUT`                        | `ai-service/app/p2p/constants.py:166`                                      |                              |
| `RINGRIFT_P2P_GOSSIP_INTERVAL`                      | `ai-service/app/p2p/constants.py:168`                                      |                              |
| `RINGRIFT_P2P_GOSSIP_MAX_PEER_ENDPOINTS`            | `ai-service/app/p2p/constants.py:171`                                      |                              |
| `RINGRIFT_P2P_GRACE_PERIOD`                         | `ai-service/scripts/node_resilience.py:1347`                               |                              |
| `RINGRIFT_P2P_GRACEFUL_SHUTDOWN_BEFORE_UPDATE`      | `ai-service/app/p2p/constants.py:325`                                      |                              |
| `RINGRIFT_P2P_HEALTH_PORT`                          | `ai-service/app/config/coordination_defaults.py:754`                       |                              |
| `RINGRIFT_P2P_HEARTBEAT_INTERVAL`                   | `ai-service/app/config/coordination_defaults.py:763`                       |                              |
| `RINGRIFT_P2P_HTTP_CONNECT_TIMEOUT`                 | `ai-service/app/p2p/constants.py:156`                                      |                              |
| `RINGRIFT_P2P_HTTP_TOTAL_TIMEOUT`                   | `ai-service/app/p2p/constants.py:157`                                      |                              |
| `RINGRIFT_P2P_IDLE_CHECK_INTERVAL`                  | `ai-service/app/p2p/constants.py:331`                                      |                              |
| `RINGRIFT_P2P_IDLE_GPU_THRESHOLD`                   | `ai-service/app/p2p/constants.py:332`                                      |                              |
| `RINGRIFT_P2P_IDLE_GRACE_PERIOD`                    | `ai-service/app/p2p/constants.py:333`                                      |                              |
| `RINGRIFT_P2P_INITIAL_CLUSTER_EPOCH`                | `ai-service/app/p2p/constants.py:237`                                      |                              |
| `RINGRIFT_P2P_ISOLATED_BOOTSTRAP_INTERVAL`          | `ai-service/app/p2p/constants.py:227`                                      |                              |
| `RINGRIFT_P2P_LEADER_DEGRADED_STEPDOWN_DELAY`       | `ai-service/app/p2p/constants.py:316`                                      |                              |
| `RINGRIFT_P2P_LEADER_HEALTH_CHECK_INTERVAL`         | `ai-service/app/p2p/constants.py:314`                                      |                              |
| `RINGRIFT_P2P_LEADER_MIN_RESPONSE_RATE`             | `ai-service/app/p2p/constants.py:315`                                      |                              |
| `RINGRIFT_P2P_MANIFEST_JSONL_LINECOUNT_CHUNK_BYTES` | `ai-service/app/p2p/constants.py:347`                                      |                              |
| `RINGRIFT_P2P_MANIFEST_JSONL_LINECOUNT_MAX_BYTES`   | `ai-service/app/p2p/constants.py:348`                                      |                              |
| `RINGRIFT_P2P_MANIFEST_JSONL_SAMPLE_BYTES`          | `ai-service/app/p2p/constants.py:346`                                      |                              |
| `RINGRIFT_P2P_MAX_CONCURRENT_CMAES_EVALS`           | `ai-service/deploy/deploy_cluster_resilience.py:451`                       |                              |
| `RINGRIFT_P2P_MAX_CONCURRENT_EXPORTS`               | `ai-service/app/p2p/constants.py:342`                                      |                              |
| `RINGRIFT_P2P_MAX_DELAY`                            | `ai-service/app/config/coordination_defaults.py:512`                       |                              |
| `RINGRIFT_P2P_MAX_GAUNTLET_RUNTIME`                 | `ai-service/app/p2p/constants.py:401`                                      |                              |
| `RINGRIFT_P2P_MAX_PEERS`                            | `ai-service/app/config/coordination_defaults.py:781`                       |                              |
| `RINGRIFT_P2P_MAX_RETRIES`                          | `ai-service/app/config/coordination_defaults.py:510`                       |                              |
| `RINGRIFT_P2P_MAX_SELFPLAY_RUNTIME`                 | `ai-service/app/p2p/constants.py:398`                                      |                              |
| `RINGRIFT_P2P_MAX_TOURNAMENT_RUNTIME`               | `ai-service/app/p2p/constants.py:400`                                      |                              |
| `RINGRIFT_P2P_MAX_TRAINING_RUNTIME`                 | `ai-service/app/p2p/constants.py:399`                                      |                              |
| `RINGRIFT_P2P_MIN_BOOTSTRAP_ATTEMPTS`               | `ai-service/app/p2p/constants.py:224`                                      |                              |
| `RINGRIFT_P2P_MIN_CONNECTED_PEERS`                  | `ai-service/app/p2p/constants.py:230`                                      |                              |
| `RINGRIFT_P2P_MIN_GAMES_FOR_SYNC`                   | `ai-service/app/p2p/constants.py:357`                                      |                              |
| `RINGRIFT_P2P_MIN_MEMORY_GB_TRAINING`               | `ai-service/app/p2p/constants.py:87`                                       |                              |
| `RINGRIFT_P2P_MODEL_SYNC_BASE`                      | `ai-service/app/p2p/constants.py:370`                                      |                              |
| `RINGRIFT_P2P_MODEL_SYNC_INTERVAL`                  | `ai-service/app/p2p/constants.py:358`                                      |                              |
| `RINGRIFT_P2P_MODEL_SYNC_MAX`                       | `ai-service/app/p2p/constants.py:372`                                      |                              |
| `RINGRIFT_P2P_MODEL_SYNC_MIN`                       | `ai-service/app/p2p/constants.py:371`                                      |                              |
| `RINGRIFT_P2P_NETWORK`                              | `ai-service/app/p2p/constants.py:430`                                      |                              |
| `RINGRIFT_P2P_PEER_CACHE_MAX_ENTRIES`               | `ai-service/app/p2p/constants.py:181`                                      |                              |
| `RINGRIFT_P2P_PEER_CACHE_TTL_SECONDS`               | `ai-service/app/p2p/constants.py:180`                                      |                              |
| `RINGRIFT_P2P_PEER_PURGE_AFTER_SECONDS`             | `ai-service/app/p2p/constants.py:177`                                      |                              |
| `RINGRIFT_P2P_PEER_REPUTATION_ALPHA`                | `ai-service/app/p2p/constants.py:182`                                      |                              |
| `RINGRIFT_P2P_PEER_TIMEOUT`                         | `ai-service/app/config/coordination_defaults.py:766`                       |                              |
| `RINGRIFT_P2P_QUORUM`                               | `ai-service/app/config/coordination_defaults.py:778`                       |                              |
| `RINGRIFT_P2P_STALE_PROCESS_CHECK_INTERVAL`         | `ai-service/app/p2p/constants.py:387`                                      |                              |
| `RINGRIFT_P2P_STALE_PROCESS_PATTERNS`               | `ai-service/app/p2p/constants.py:389`                                      |                              |
| `RINGRIFT_P2P_STARTUP_JSONL_GRACE_PERIOD`           | `ai-service/app/p2p/constants.py:349`                                      |                              |
| `RINGRIFT_P2P_STATE_DIR`                            | `ai-service/app/p2p/constants.py:452`                                      |                              |
| `RINGRIFT_P2P_STORAGE_ROOT`                         | `ai-service/app/p2p/constants.py:444`                                      |                              |
| `RINGRIFT_P2P_SUSPECT_TIMEOUT`                      | `ai-service/app/p2p/constants.py:54`                                       |                              |
| `RINGRIFT_P2P_SYNC_BACKOFF_FACTOR`                  | `ai-service/app/p2p/constants.py:381`                                      |                              |
| `RINGRIFT_P2P_SYNC_SPEEDUP_FACTOR`                  | `ai-service/app/p2p/constants.py:380`                                      |                              |
| `RINGRIFT_P2P_TRAINING_DATA_SYNC_THRESHOLD_MB`      | `ai-service/app/p2p/constants.py:341`                                      |                              |
| `RINGRIFT_P2P_TRAINING_DB_SYNC_BASE`                | `ai-service/app/p2p/constants.py:375`                                      |                              |
| `RINGRIFT_P2P_TRAINING_DB_SYNC_MAX`                 | `ai-service/app/p2p/constants.py:377`                                      |                              |
| `RINGRIFT_P2P_TRAINING_DB_SYNC_MIN`                 | `ai-service/app/p2p/constants.py:376`                                      |                              |
| `RINGRIFT_P2P_TRAINING_NODE_COUNT`                  | `ai-service/app/p2p/constants.py:355`                                      |                              |
| `RINGRIFT_P2P_TRAINING_SYNC_INTERVAL`               | `ai-service/app/p2p/constants.py:356`                                      |                              |
| `RINGRIFT_P2P_UNIFIED_DISCOVERY_INTERVAL`           | `ai-service/app/p2p/constants.py:414`                                      |                              |
| `RINGRIFT_P2P_VOTER_DEMOTION_FAILURES`              | `ai-service/app/p2p/constants.py:306`                                      |                              |
| `RINGRIFT_P2P_VOTER_HEALTH_THRESHOLD`               | `ai-service/app/p2p/constants.py:307`                                      |                              |
| `RINGRIFT_P2P_VOTER_MIN_QUORUM`                     | `ai-service/app/p2p/constants.py:216`                                      |                              |
| `RINGRIFT_P2P_VOTER_PROMOTION_UPTIME`               | `ai-service/app/p2p/constants.py:308`                                      |                              |
| `RINGRIFT_PARITY_PROGRESS_EVERY`                    | `ai-service/scripts/check_ts_python_replay_parity.py:507`                  |                              |
| `RINGRIFT_PROFILE_TRAINING`                         | `ai-service/scripts/p2p_orchestrator.py:12030`                             |                              |
| `RINGRIFT_QUEUE_BACKPRESSURE_THRESHOLD`             | `ai-service/app/config/coordination_defaults.py:1316`                      |                              |
| `RINGRIFT_QUEUE_MONITOR_DB`                         | `ai-service/tests/test_coordination_integration.py:138`                    | test-only (first occurrence) |
| `RINGRIFT_RAFT_AUTO_UNLOCK_TIME`                    | `ai-service/app/p2p/constants.py:268`                                      |                              |
| `RINGRIFT_RAFT_BIND_PORT`                           | `ai-service/app/p2p/constants.py:266`                                      |                              |
| `RINGRIFT_RAFT_COMPACTION_MIN_ENTRIES`              | `ai-service/app/p2p/constants.py:267`                                      |                              |
| `RINGRIFT_REGRESSION_HARD_BLOCK`                    | `ai-service/scripts/model_promotion_manager.py:1418`                       |                              |
| `RINGRIFT_RELAY_HOST`                               | `ai-service/scripts/node_resilience.py:273`                                |                              |
| `RINGRIFT_REMOTE_DIR`                               | `ai-service/archive/deprecated_coordination/sync_coordination_core.py:517` |                              |
| `RINGRIFT_REMOTE_PARITY_VALIDATION`                 | `ai-service/scripts/run_distributed_selfplay_soak.py:765`                  |                              |
| `RINGRIFT_RESOURCE_CHECK_INTERVAL`                  | `ai-service/app/config/coordination_defaults.py:945`                       |                              |
| `RINGRIFT_ROLLBACK_ELO_DROP`                        | `ai-service/scripts/model_promotion_manager.py:761`                        |                              |
| `RINGRIFT_ROLLBACK_MIN_GAMES`                       | `ai-service/scripts/model_promotion_manager.py:771`                        |                              |
| `RINGRIFT_ROOT`                                     | `ai-service/deploy/deploy_cluster_resilience.py:566`                       |                              |
| `RINGRIFT_SANDBOX_`                                 | `scripts/validate-deployment-config.ts:513`                                | prefix/pattern               |
| `RINGRIFT_SANDBOX_AI_META__`                        | `src/client/sandbox/sandboxAiDiagnostics.ts:45`                            | prefix/pattern               |
| `RINGRIFT_SANDBOX_ANIMATION_DEBUG`                  | `src/shared/utils/envFlags.ts:106`                                         |                              |
| `RINGRIFT_SANDBOX_LPS_DEBUG`                        | `src/shared/utils/envFlags.ts:92`                                          |                              |
| `RINGRIFT_SB_BENCH_ITERS`                           | `ai-service/scripts/benchmark_search_board_large_board.py:18`              |                              |
| `RINGRIFT_SCHEDULER_DB`                             | `ai-service/tests/test_coordination_integration.py:109`                    | test-only (first occurrence) |
| `RINGRIFT_SELFPLAY_`                                | `ai-service/app/config/training_config.py:314`                             | prefix/pattern               |
| `RINGRIFT_SELFPLAY_GAMES_PER_CONFIG`                | `ai-service/app/config/coordination_defaults.py:1363`                      |                              |
| `RINGRIFT_SKIP_POST_TRAINING_GAUNTLET`              | `ai-service/scripts/p2p/managers/training_coordinator.py:803`              |                              |
| `RINGRIFT_SKIP_REGRESSION_TESTS`                    | `ai-service/scripts/model_promotion_manager.py:1421`                       |                              |
| `RINGRIFT_SKIP_SCRIPT_INIT_IMPORTS`                 | `ai-service/scripts/__init__.py:19`                                        |                              |
| `RINGRIFT_SKIP_SELFPLAY_CONFIG`                     | `ai-service/scripts/run_distributed_selfplay_soak.py:117`                  |                              |
| `RINGRIFT_SKIP_SYNC_LOCK_IMPORT`                    | `ai-service/scripts/run_distributed_selfplay_soak.py:74`                   |                              |
| `RINGRIFT_SKIP_TRAINING_COORD`                      | `ai-service/scripts/train_nnue.py:5253`                                    |                              |
| `RINGRIFT_SOAK_FAILURE_DIR`                         | `ai-service/scripts/run_self_play_soak.py:2469`                            |                              |
| `RINGRIFT_SQLITE_`                                  | `ai-service/scripts/p2p/managers/state_manager.py:102`                     | prefix/pattern               |
| `RINGRIFT_SQLITE_BUSY_TIMEOUT_MS`                   | `ai-service/app/config/coordination_defaults.py:1092`                      |                              |
| `RINGRIFT_SQLITE_HEAVY_TIMEOUT`                     | `ai-service/app/config/coordination_defaults.py:1083`                      |                              |
| `RINGRIFT_SQLITE_MERGE_TIMEOUT`                     | `ai-service/app/config/coordination_defaults.py:1086`                      |                              |
| `RINGRIFT_SQLITE_QUICK_TIMEOUT`                     | `ai-service/app/config/coordination_defaults.py:1071`                      |                              |
| `RINGRIFT_SQLITE_READ_TIMEOUT`                      | `ai-service/app/config/coordination_defaults.py:1074`                      |                              |
| `RINGRIFT_SQLITE_STANDARD_TIMEOUT`                  | `ai-service/app/config/coordination_defaults.py:1077`                      |                              |
| `RINGRIFT_SQLITE_WAL_CHECKPOINT`                    | `ai-service/app/config/coordination_defaults.py:1089`                      |                              |
| `RINGRIFT_SQLITE_WRITE_TIMEOUT`                     | `ai-service/app/config/coordination_defaults.py:1080`                      |                              |
| `RINGRIFT_SSH_BASE_DELAY`                           | `ai-service/app/config/coordination_defaults.py:506`                       |                              |
| `RINGRIFT_SSH_DEFAULT_TIMEOUT`                      | `ai-service/app/distributed/ssh_connection_manager.py:109`                 |                              |
| `RINGRIFT_SSH_MAX_DELAY`                            | `ai-service/app/config/coordination_defaults.py:507`                       |                              |
| `RINGRIFT_STAGING_`                                 | `ai-service/scripts/run_improvement_loop.py:2163`                          | prefix/pattern               |
| `RINGRIFT_STAGING_COMPOSE_FILE`                     | `ai-service/scripts/sync_staging_ai_pipeline.py:22`                        |                              |
| `RINGRIFT_STAGING_LADDER_HEALTH_TIMEOUT_SECONDS`    | `ai-service/scripts/sync_staging_ai_artifacts.py:255`                      |                              |
| `RINGRIFT_STAGING_LADDER_HEALTH_URL`                | `ai-service/scripts/sync_staging_ai_artifacts.py:249`                      |                              |
| `RINGRIFT_STAGING_RESTART_SERVICES`                 | `ai-service/scripts/sync_staging_ai_artifacts.py:239`                      |                              |
| `RINGRIFT_STAGING_ROOT`                             | `ai-service/scripts/sync_staging_ai_pipeline.py:16`                        |                              |
| `RINGRIFT_STAGING_SSH_HOST`                         | `ai-service/scripts/sync_staging_ai_pipeline.py:15`                        |                              |
| `RINGRIFT_STAGING_SSH_KEY`                          | `ai-service/scripts/sync_staging_ai_pipeline.py:21`                        |                              |
| `RINGRIFT_STAGING_SSH_PORT`                         | `ai-service/scripts/sync_staging_ai_pipeline.py:20`                        |                              |
| `RINGRIFT_STAGING_SSH_USER`                         | `ai-service/scripts/sync_staging_ai_pipeline.py:19`                        |                              |
| `RINGRIFT_STUCK_JOB_THRESHOLD`                      | `ai-service/app/config/coordination_defaults.py:1320`                      |                              |
| `RINGRIFT_SUBDIR`                                   | `ai-service/app/utils/ramdrive.py:49`                                      |                              |
| `RINGRIFT_SUBSCRIBER_TIMEOUT`                       | `ai-service/app/config/coordination_defaults.py:1397`                      |                              |
| `RINGRIFT_SWIM_BIND_PORT`                           | `ai-service/app/p2p/constants.py:275`                                      |                              |
| `RINGRIFT_SWIM_FAILURE_TIMEOUT`                     | `ai-service/app/p2p/constants.py:276`                                      |                              |
| `RINGRIFT_SWIM_INDIRECT_PING_COUNT`                 | `ai-service/app/p2p/constants.py:279`                                      |                              |
| `RINGRIFT_SWIM_PING_INTERVAL`                       | `ai-service/app/p2p/constants.py:278`                                      |                              |
| `RINGRIFT_SWIM_SUSPICION_TIMEOUT`                   | `ai-service/app/p2p/constants.py:277`                                      |                              |
| `RINGRIFT_SYNC_DEFAULT_CHUNK_SIZE`                  | `ai-service/app/config/coordination_defaults.py:1489`                      |                              |
| `RINGRIFT_SYNC_DIR`                                 | `ai-service/scripts/data_aggregator.py:165`                                |                              |
| `RINGRIFT_SYNC_LARGE_CHUNK_SIZE`                    | `ai-service/app/config/coordination_defaults.py:1492`                      |                              |
| `RINGRIFT_SYNC_MUTEX_DB`                            | `ai-service/tests/test_coordination_integration.py:164`                    | test-only (first occurrence) |
| `RINGRIFT_SYNC_TARGET`                              | `ai-service/scripts/vastai_termination_guard.py:424`                       |                              |
| `RINGRIFT_TEST_BOOL`                                | `ai-service/tests/unit/config/test_env.py:218`                             | test-only (first occurrence) |
| `RINGRIFT_TEST_FLAG`                                | `tests/unit/envFlags.test.ts:24`                                           |                              |
| `RINGRIFT_TEST_FLOAT`                               | `ai-service/tests/unit/config/test_env.py:210`                             | test-only (first occurrence) |
| `RINGRIFT_TEST_INT`                                 | `ai-service/tests/unit/config/test_env.py:199`                             | test-only (first occurrence) |
| `RINGRIFT_TEST_SET`                                 | `ai-service/tests/unit/config/test_env.py:232`                             | test-only (first occurrence) |
| `RINGRIFT_TEST_VAR`                                 | `ai-service/tests/unit/config/test_env.py:182`                             | test-only (first occurrence) |
| `RINGRIFT_TIMEOUT_`                                 | `ai-service/app/config/timeout_config.py:16`                               | prefix/pattern               |
| `RINGRIFT_TIMEOUT_SYNC_INTERVAL`                    | `ai-service/app/config/timeout_config.py:18`                               |                              |
| `RINGRIFT_TOURNAMENT_ENTRYPOINT`                    | `ai-service/scripts/run_tournament.py:492`                                 |                              |
| `RINGRIFT_TRACE_LPS`                                | `scripts/selfplay-db-ts-replay.ts:1356`                                    |                              |
| `RINGRIFT_TRAINING_ACTIVITY`                        | `ai-service/app/coordination/training_activity_daemon.py:62`               |                              |
| `RINGRIFT_TRAINING_ACTIVITY_ENABLED`                | `ai-service/tests/unit/coordination/test_training_activity_daemon.py:82`   | test-only (first occurrence) |
| `RINGRIFT_TRAINING_ACTIVITY_TRIGGER_SYNC`           | `ai-service/tests/unit/coordination/test_training_activity_daemon.py:83`   | test-only (first occurrence) |
| `RINGRIFT_TRANSFER_BASE_DELAY`                      | `ai-service/app/config/coordination_defaults.py:501`                       |                              |
| `RINGRIFT_TRANSFER_MAX_DELAY`                       | `ai-service/app/config/coordination_defaults.py:502`                       |                              |
| `RINGRIFT_TRANSFER_MAX_RETRIES`                     | `ai-service/app/config/coordination_defaults.py:500`                       |                              |
| `RINGRIFT_TS_REPLAY_MINIMAL`                        | `scripts/selfplay-db-ts-replay.ts:833`                                     |                              |
| `RINGRIFT_USE_AUTOSSH`                              | `ai-service/scripts/node_resilience.py:302`                                |                              |
| `RINGRIFT_WATCHDOG`                                 | `ai-service/app/coordination/cluster_watchdog_daemon.py:105`               |                              |
| `RINGRIFT_WATCHDOG_`                                | `ai-service/app/coordination/base_daemon.py:75`                            | prefix/pattern               |
| `RINGRIFT_WATCHDOG_ACTIVATION_COOLDOWN`             | `ai-service/app/config/coordination_defaults.py:1005`                      |                              |
| `RINGRIFT_WATCHDOG_ENABLED`                         | `ai-service/tests/unit/coordination/test_cluster_watchdog_daemon.py:90`    | test-only (first occurrence) |
| `RINGRIFT_WATCHDOG_INTERVAL`                        | `ai-service/tests/unit/coordination/test_cluster_watchdog_daemon.py:100`   | test-only (first occurrence) |
| `RINGRIFT_WATCHDOG_MAX_ACTIVATIONS`                 | `ai-service/app/config/coordination_defaults.py:1011`                      |                              |
| `RINGRIFT_WATCHDOG_MAX_FAILURES`                    | `ai-service/app/config/coordination_defaults.py:1008`                      |                              |
| `RINGRIFT_WATCHDOG_MIN_GPU`                         | `ai-service/tests/unit/coordination/test_cluster_watchdog_daemon.py:106`   | test-only (first occurrence) |
| `RINGRIFT_WATCHDOG_MIN_GPU_UTIL`                    | `ai-service/app/config/coordination_defaults.py:1002`                      |                              |
| `RINGRIFT_WATCHDOG_SSH_TIMEOUT`                     | `ai-service/app/config/coordination_defaults.py:999`                       |                              |
