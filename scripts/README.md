# Root Scripts

This directory mixes supported operational scripts with one-off debugging and analysis tools. If you are new to the repo, start with the scripts below and treat most `debug*`, `analyze*`, and one-shot drill scripts as investigation tooling rather than stable entrypoints.

For Python training/runtime operations inside `ai-service`, also read [ai-service/scripts/README.md](/Users/armand/Development/RingRift/ai-service/scripts/README.md).

## Essential Scripts

### Development and local setup

- [`bootstrap_ai.sh`](/Users/armand/Development/RingRift/scripts/bootstrap_ai.sh)
  Bootstraps the local AI-service environment and common prerequisites.

- [`dev_doctor.ts`](/Users/armand/Development/RingRift/scripts/dev_doctor.ts)
  Checks local repository and environment health before deeper debugging.

- [`dev-db.sh`](/Users/armand/Development/RingRift/scripts/dev-db.sh)
  Starts or manages the local development database workflow.

### Test and parity gates

- [`run-tests-with-timeout.sh`](/Users/armand/Development/RingRift/scripts/run-tests-with-timeout.sh)
  Canonical top-level TypeScript/Jest timeout wrapper.

- [`run-python-tests-with-timeout.sh`](/Users/armand/Development/RingRift/scripts/run-python-tests-with-timeout.sh)
  Timeout wrapper for the Python test suite.

- [`run-python-contract-tests.sh`](/Users/armand/Development/RingRift/scripts/run-python-contract-tests.sh)
  Focused Python contract/parity test runner.

- [`check_supported_path.sh`](/Users/armand/Development/RingRift/scripts/check_supported_path.sh)
  Quick supported-path health check across the current stack.

- [`run-ts-python-parity-metric.ts`](/Users/armand/Development/RingRift/scripts/run-ts-python-parity-metric.ts)
  TS↔Python parity metric runner.

- [`check-parity-metrics.ts`](/Users/armand/Development/RingRift/scripts/check-parity-metrics.ts)
  Parity regression check/reporting helper.

### Contract vectors and fixtures

- [`generate-orchestrator-contract-vectors.ts`](/Users/armand/Development/RingRift/scripts/generate-orchestrator-contract-vectors.ts)
  Generates orchestrator contract vectors.

- [`generate-extended-contract-vectors.ts`](/Users/armand/Development/RingRift/scripts/generate-extended-contract-vectors.ts)
  Generates extended rule/engine contract vectors.

- [`generate-meta-move-vectors.ts`](/Users/armand/Development/RingRift/scripts/generate-meta-move-vectors.ts)
  Generates meta-move vector fixtures.

- [`generate-golden-fixtures.ts`](/Users/armand/Development/RingRift/scripts/generate-golden-fixtures.ts)
  Produces canonical golden fixture artifacts.

- [`curate-golden-fixtures.ts`](/Users/armand/Development/RingRift/scripts/curate-golden-fixtures.ts)
  Curates and normalizes golden fixtures for stable reuse.

### Operational checks and deployment

- [`deploy-staging.sh`](/Users/armand/Development/RingRift/scripts/deploy-staging.sh)
  Staging deployment entrypoint.

- [`teardown-staging.sh`](/Users/armand/Development/RingRift/scripts/teardown-staging.sh)
  Tears down the staging environment cleanly.

- [`cluster-update.sh`](/Users/armand/Development/RingRift/scripts/cluster-update.sh)
  Cluster update helper for shared runtime changes.

- [`validate-deployment-config.ts`](/Users/armand/Development/RingRift/scripts/validate-deployment-config.ts)
  Static validation for deployment config integrity.

- [`validate-monitoring-configs.sh`](/Users/armand/Development/RingRift/scripts/validate-monitoring-configs.sh)
  Verifies monitoring config health before rollout.

- [`cluster_alerting.sh`](/Users/armand/Development/RingRift/scripts/cluster_alerting.sh)
  Alerting/bootstrap helper for cluster monitoring.

### Product and runtime checks

- [`product_smoke_test.sh`](/Users/armand/Development/RingRift/scripts/product_smoke_test.sh)
  High-level product smoke test entrypoint.

- [`replay-db-healthcheck.ts`](/Users/armand/Development/RingRift/scripts/replay-db-healthcheck.ts)
  Replay database health probe.

- [`rules-health-report.sh`](/Users/armand/Development/RingRift/scripts/rules-health-report.sh)
  Rules/engine health report helper.

- [`prometheus_metrics.sh`](/Users/armand/Development/RingRift/scripts/prometheus_metrics.sh)
  Metrics collection/export helper.

- [`prometheus_p2p_exporter.py`](/Users/armand/Development/RingRift/scripts/prometheus_p2p_exporter.py)
  P2P-specific metrics exporter.

## Everything Else

Most remaining root scripts are one of:

- debugging probes
- operational drills
- historical analysis scripts
- fixture generation variants
- incident-specific cleanup helpers

Do not assume those scripts are on the supported path unless they are listed above or referenced by a current runbook or architecture document.
