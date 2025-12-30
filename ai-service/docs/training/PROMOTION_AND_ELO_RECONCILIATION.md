# Model Promotion and Elo Reconciliation

This document describes the unified promotion controller and Elo reconciliation system for RingRift AI.

## Overview

The promotion and Elo reconciliation system provides:

1. **Unified Promotion Controller** (`app/training/promotion_controller.py`) - Centralized promotion decision-making with standardized criteria
2. **Elo Reconciliation** (`app/training/elo_reconciliation.py`) - Distributed Elo consistency management across P2P nodes
3. **Prometheus Metrics** - Observability for promotion decisions and Elo drift
4. **CLI Tools** - Manual reconciliation and debugging

## Promotion Controller

### Purpose

The `PromotionController` consolidates promotion logic from multiple modules into a single entry point with:

- Standardized promotion criteria
- Prometheus metrics emission
- Support for multiple promotion types

### Promotion Types

| Type         | Description                         |
| ------------ | ----------------------------------- |
| `STAGING`    | Development to staging promotion    |
| `PRODUCTION` | Staging to production promotion     |
| `TIER`       | Tier ladder promotion (D1→D2, etc.) |
| `CHAMPION`   | Tournament champion promotion       |
| `ROLLBACK`   | Rollback to previous version        |

### Usage

```python
from app.training.promotion_controller import (
    PromotionController,
    PromotionCriteria,
    PromotionType,
)

# Create controller with custom criteria
controller = PromotionController(
    criteria=PromotionCriteria(
        min_elo_improvement=25.0,
        min_games_played=50,
        min_win_rate=0.52,
    )
)

# Evaluate a promotion candidate
decision = controller.evaluate_promotion(
    model_id="model_v42",
    board_type="square8",
    num_players=2,
    promotion_type=PromotionType.PRODUCTION,
    baseline_model_id="model_v41",
)

if decision.should_promote:
    # Execute the promotion
    success = controller.execute_promotion(decision)

    # Or dry run first
    controller.execute_promotion(decision, dry_run=True)
```

### Criteria

| Criterion                   | Default | Description                       |
| --------------------------- | ------- | --------------------------------- |
| `min_elo_improvement`       | 25.0    | Minimum Elo gain over baseline    |
| `min_games_played`          | 50      | Minimum games for evaluation      |
| `min_win_rate`              | 0.52    | Minimum win rate threshold        |
| `max_value_mse_degradation` | 0.05    | Max allowed value MSE increase    |
| `confidence_threshold`      | 0.95    | Statistical confidence required   |
| `tier_games_required`       | 100     | Games required for tier promotion |

### Integration with Unified AI Loop

The `ModelPromoter` in `scripts/unified_loop/promotion.py` uses `PromotionController` for:

- Centralized criteria evaluation
- Metrics emission on all promotion decisions
- Consistent promotion behavior across the system

## Elo Reconciliation

### Purpose

In a distributed P2P setup, multiple nodes may have local Elo databases that drift apart. The `EloReconciler` provides:

- Drift detection between databases
- Match history synchronization
- Conflict detection and resolution
- Reconciliation reporting

### Drift Detection

```python
from app.training.elo_reconciliation import EloReconciler, check_elo_drift

# Quick check
drift = check_elo_drift(board_type="square8", num_players=2)
if drift.is_significant:
    print(f"WARNING: Max drift = {drift.max_rating_diff}")

# Full reconciler
reconciler = EloReconciler()
drift = reconciler.check_drift(
    remote_db_path="/path/to/remote.db",
    board_type="square8",
    num_players=2,
)

print(f"Participants in source: {drift.participants_in_source}")
print(f"Participants in target: {drift.participants_in_target}")
print(f"Max rating diff: {drift.max_rating_diff}")
print(f"Is significant: {drift.is_significant}")
```

### Synchronization

```python
# Sync from a remote host (replace with your node's IP or hostname)
result = reconciler.sync_from_remote(
    remote_host="gpu-node-1",  # Or use Tailscale IP
    remote_db_path="~/ringrift/ai-service/data/unified_elo.db",
    ssh_user="ubuntu",
)

print(f"Added: {result.matches_added}")
print(f"Skipped: {result.matches_skipped}")
print(f"Conflicts: {result.matches_conflict}")
print(f"Resolved: {result.matches_resolved}")
```

### Conflict Resolution

When importing matches, conflicts may occur if the same `match_id` exists with different data (e.g., different winner). The reconciler supports configurable conflict resolution strategies:

| Strategy           | Description                                                  |
| ------------------ | ------------------------------------------------------------ |
| `SKIP`             | Keep existing record, count as unresolved conflict (default) |
| `LAST_WRITE_WINS`  | Accept match with more recent timestamp                      |
| `FIRST_WRITE_WINS` | Keep existing record, mark as resolved                       |
| `RAISE`            | Raise an exception on first conflict                         |

```python
from app.training.elo_reconciliation import EloReconciler, ConflictResolution

# Create reconciler with last-write-wins strategy
reconciler = EloReconciler(
    conflict_resolution=ConflictResolution.LAST_WRITE_WINS,
)

# Conflicts are now automatically resolved by timestamp
result = reconciler.sync_from_remote("gpu-node-1")
print(f"Resolved: {result.matches_resolved}")
```

The unified AI loop uses `LAST_WRITE_WINS` by default for automatic conflict resolution.

### Full Reconciliation

```python
# Reconcile all configured hosts
report = reconciler.reconcile_all()
print(report.summary())

# Custom host list (replace with your actual node hostnames or IPs)
report = reconciler.reconcile_all(hosts=["gpu-node-1", "gpu-node-2"])
```

### Historical Drift Tracking

The `EloReconciler` tracks drift history for trend analysis:

```python
from app.training.elo_reconciliation import EloReconciler, DriftHistory

# Create reconciler with history tracking (enabled by default)
reconciler = EloReconciler(track_history=True)

# Each check_drift call records a snapshot
reconciler.check_drift(board_type="square8", num_players=2)

# Later, check drift history
history = reconciler.get_drift_history("square8_2")
if history:
    print(f"Trend: {history.trend}")  # 'improving', 'stable', 'worsening', 'unknown'
    print(f"Persistent drift: {history.persistent_drift}")
    print(f"Avg drift (last hour): {history.avg_drift_last_hour}")

# Get all tracked configurations
all_histories = reconciler.get_all_drift_histories()
for config_key, hist in all_histories.items():
    if hist.persistent_drift:
        print(f"WARNING: {config_key} has persistent drift!")
```

| Property              | Description                                                |
| --------------------- | ---------------------------------------------------------- |
| `trend`               | Drift trend: improving, stable, worsening, or unknown      |
| `persistent_drift`    | True if last 3 checks all had significant drift            |
| `avg_drift_last_hour` | Average max drift over last 2 snapshots (30-min intervals) |
| `snapshot_count`      | Number of recorded drift snapshots                         |

## CLI Tools

### Elo Reconciliation CLI

```bash
# Check current status
python scripts/elo_reconciliation_cli.py status

# Check for drift
python scripts/elo_reconciliation_cli.py check-drift
python scripts/elo_reconciliation_cli.py check-drift --board-type square8 --num-players 2
python scripts/elo_reconciliation_cli.py check-drift --json

# Sync from specific host (replace with your node hostname or IP)
python scripts/elo_reconciliation_cli.py sync --host gpu-node-1

# Full reconciliation
python scripts/elo_reconciliation_cli.py reconcile-all
python scripts/elo_reconciliation_cli.py reconcile-all --hosts "host1,host2"
```

## Prometheus Metrics

### Promotion Metrics

| Metric                                | Type      | Labels                      | Description                  |
| ------------------------------------- | --------- | --------------------------- | ---------------------------- |
| `ringrift_promotion_decisions_total`  | Counter   | `promotion_type`, `outcome` | Total promotion decisions    |
| `ringrift_promotion_executions_total` | Counter   | `promotion_type`, `result`  | Total promotion executions   |
| `ringrift_promotion_elo_improvement`  | Histogram | `promotion_type`            | Elo improvement distribution |

### Elo Reconciliation Metrics

| Metric                                  | Type    | Labels                      | Description             |
| --------------------------------------- | ------- | --------------------------- | ----------------------- |
| `ringrift_elo_sync_operations_total`    | Counter | `remote_host`, `result`     | Total sync operations   |
| `ringrift_elo_sync_matches_added_total` | Counter | `remote_host`               | Matches added via sync  |
| `ringrift_elo_sync_conflicts_total`     | Counter | `remote_host`               | Sync conflicts detected |
| `ringrift_elo_drift_max`                | Gauge   | `board_type`, `num_players` | Maximum Elo drift       |
| `ringrift_elo_drift_avg`                | Gauge   | `board_type`, `num_players` | Average Elo drift       |
| `ringrift_elo_drift_significant`        | Gauge   | `board_type`, `num_players` | Significant drift flag  |

## Alerting

Alerts are configured in `config/monitoring/alerting-rules.yaml`:

| Alert                        | Severity | Description                           |
| ---------------------------- | -------- | ------------------------------------- |
| `PromotionExecutionFailures` | High     | Multiple promotion execution failures |
| `HighPromotionRejectionRate` | Warning  | >80% promotion rejections             |
| `SignificantEloDrift`        | High     | Significant Elo drift detected        |
| `EloSyncFailureSpike`        | High     | Multiple sync failures                |
| `EloSyncConflicts`           | Warning  | High conflict rate                    |
| `HighEloDriftPersistent`     | Warning  | Drift >100 for >1 hour                |

## Grafana Dashboard

The Grafana dashboard (`deploy/grafana/unified-ai-loop-dashboard.json`) includes panels for:

- **Model Promotions Section**:
  - Promotion Decisions (approved/rejected)
  - Promotion Executions (success/failure/dry_run)

- **Elo Reconciliation Section**:
  - Elo Drift (Max/Avg) with threshold lines
  - Significant Drift indicator
  - Sync Operations by host
  - Sync Conflicts count

## Testing

### Unit Tests

```bash
# Promotion controller tests
pytest tests/test_promotion_controller.py -v

# Elo reconciliation tests
pytest tests/test_elo_reconciliation.py -v
```

### Integration Tests

```bash
pytest tests/integration/test_promotion_elo_integration.py -v
```

### Performance Benchmarks

```bash
# Run all benchmarks
python tests/benchmarks/test_promotion_elo_benchmarks.py

# With pytest
pytest tests/benchmarks/test_promotion_elo_benchmarks.py -v
```

Expected performance:

- `evaluate_promotion`: <1ms (without DB)
- `check_drift(1000 participants)`: <1ms
- `_import_matches`: >100,000 matches/sec

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Unified AI Loop                          │
│                                                             │
│  ┌──────────────────┐     ┌──────────────────────────────┐ │
│  │  ModelPromoter   │────▶│    PromotionController       │ │
│  │(unified_loop/)   │     │   - evaluate_promotion()     │ │
│  └──────────────────┘     │   - execute_promotion()      │ │
│                           │   - _emit_decision_metrics() │ │
│                           │   - _emit_execution_metrics()│ │
│                           └──────────────────────────────┘ │
│                                        │                    │
│                                        ▼                    │
│                           ┌──────────────────────────────┐ │
│                           │       Prometheus Metrics      │ │
│                           └──────────────────────────────┘ │
│                                                             │
│  ┌──────────────────┐     ┌──────────────────────────────┐ │
│  │  Data Collector  │     │      EloReconciler           │ │
│  │                  │────▶│   - check_drift()            │ │
│  └──────────────────┘     │   - sync_from_remote()       │ │
│                           │   - reconcile_all()          │ │
│                           │   - _emit_drift_metrics()    │ │
│                           │   - _emit_sync_metrics()     │ │
│                           └──────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## Configuration

### Remote Hosts

Configure P2P voters in `config/distributed_hosts.yaml`:

```yaml
p2p_voters:
  - nebius-backbone-1
  - nebius-h100-3
  - hetzner-cpu1
  - hetzner-cpu2
  - vultr-a100-20gb

hosts:
  gpu-node-1:
    ssh_host: 10.0.0.1 # Replace with actual IP or hostname
    role: selfplay
    p2p_enabled: true
  gpu-node-2:
    ssh_host: 10.0.0.2 # Replace with actual IP or hostname
    role: selfplay
```

### Promotion Criteria

Configure via `PromotionCriteria` or in unified loop config:

```yaml
promotion:
  auto_promote: true
  elo_threshold: 25.0
  min_games: 50
  sync_to_cluster: true
```

## Automatic Rollback

### Purpose

The `RollbackMonitor` provides automated rollback capabilities for promoted models that show regression:

- Continuous monitoring of promoted model performance
- Automatic detection of Elo regression or win rate drops
- Configurable thresholds and consecutive check requirements
- Automatic rollback to previous model when regression is confirmed

### Usage

```python
from app.training.promotion_controller import (
    RollbackMonitor,
    RollbackCriteria,
    get_rollback_monitor,
)

# Create monitor with custom criteria
monitor = RollbackMonitor(
    criteria=RollbackCriteria(
        elo_regression_threshold=-30.0,
        min_games_for_regression=20,
        consecutive_checks_required=3,
        min_win_rate=0.40,
    )
)

# Check if rollback is needed
should_rollback, event = monitor.check_for_regression(
    model_id="model_v42",
    board_type="square8",
    num_players=2,
    previous_model_id="model_v41",
    baseline_elo=1500.0,  # Elo at promotion time
)

if should_rollback:
    # Dry run first
    monitor.execute_rollback(event, dry_run=True)

    # Execute rollback
    success = monitor.execute_rollback(event)

# Check regression status
status = monitor.get_regression_status("model_v42")
if status["at_risk"]:
    print(f"Model at risk: {status['consecutive_regressions']} consecutive regressions")
```

### Rollback Criteria

| Criterion                     | Default | Description                                 |
| ----------------------------- | ------- | ------------------------------------------- |
| `elo_regression_threshold`    | -30.0   | Elo drop that counts as regression          |
| `min_games_for_regression`    | 20      | Minimum games before checking for rollback  |
| `consecutive_checks_required` | 3       | Consecutive regression checks before action |
| `min_win_rate`                | 0.40    | Win rate below which triggers rollback      |
| `time_window_seconds`         | 3600    | Time window for regression history (1 hour) |

### Rollback Triggers

Rollback is triggered in these scenarios:

1. **Severe Regression**: Elo drops below `elo_regression_threshold * 2` (e.g., -60 with default settings)
2. **Consecutive Regression**: Multiple checks (default 3) show regression below threshold
3. **Low Win Rate**: Win rate falls below `min_win_rate` threshold (40% default)

### Integration with Unified AI Loop

The unified AI loop can optionally include automatic rollback monitoring:

```python
# In master_loop.py
from app.training.promotion_controller import RollbackMonitor

class UnifiedAILoop:
    def __init__(self):
        self.rollback_monitor = RollbackMonitor()

    async def check_model_health(self, model_id, previous_model_id, baseline_elo):
        should_rollback, event = self.rollback_monitor.check_for_regression(
            model_id=model_id,
            previous_model_id=previous_model_id,
            baseline_elo=baseline_elo,
        )
        if should_rollback:
            logger.warning(f"Auto-rollback triggered: {event.reason}")
            self.rollback_monitor.execute_rollback(event)
```

### Prometheus Metrics

| Metric                             | Type    | Labels                             | Description             |
| ---------------------------------- | ------- | ---------------------------------- | ----------------------- |
| `ringrift_rollback_checks_total`   | Counter | `model_id`, `triggered`            | Total regression checks |
| `ringrift_auto_rollbacks_total`    | Counter | `from_model`, `to_model`, `result` | Auto-rollback count     |
| `ringrift_rollback_elo_regression` | Gauge   | `model_id`                         | Current Elo regression  |

## Troubleshooting

### High Promotion Rejection Rate

1. Check promotion criteria thresholds
2. Review Elo trends for the configuration
3. Verify training data quality
4. Consider lowering `min_elo_improvement`

### Significant Elo Drift

1. Run manual reconciliation:
   ```bash
   python scripts/elo_reconciliation_cli.py reconcile-all
   ```
2. Check network connectivity to P2P nodes
3. Review sync errors in logs
4. Check for database corruption

### Sync Failures

1. Verify SSH connectivity:
   ```bash
   ssh ubuntu@<host> "ls -la ~/ringrift/ai-service/data/"
   ```
2. Check database file exists on remote
3. Review SSH timeout settings
4. Check disk space on local and remote

### Match Conflicts

Match conflicts indicate the same match_id exists with different data:

1. Review conflict details in sync results
2. Check for race conditions in match recording
3. May need manual database inspection
4. Consider implementing conflict resolution policy

## Automatic Rollback Monitoring

### Overview

The `RollbackMonitor` automatically detects model regression and triggers rollbacks when necessary.

### Features

- **Cooldown Period**: Prevents rapid successive rollbacks (default: 1 hour)
- **Daily Limit**: Maximum rollbacks per day before requiring manual intervention (default: 3)
- **Multi-Model Baseline Comparison**: Compare against multiple historical models
- **Notification Hooks**: Customizable notifications via webhooks, PagerDuty, or OpsGenie

### Usage

```python
from app.training.promotion_controller import (
    RollbackMonitor,
    RollbackCriteria,
)

# Create monitor with custom criteria
monitor = RollbackMonitor(
    criteria=RollbackCriteria(
        elo_regression_threshold=-30.0,
        min_games_for_regression=20,
        consecutive_checks_required=3,
        cooldown_seconds=3600,  # 1 hour
        max_rollbacks_per_day=3,
    )
)

# Check for regression
should_rollback, event = monitor.check_for_regression(
    model_id="model_v42",
    board_type="square8",
    num_players=2,
    previous_model_id="model_v41",
)

if should_rollback:
    success = monitor.execute_rollback(event)
```

### Cooldown Management

```python
# Check if cooldown is active
is_active, remaining = monitor.is_cooldown_active("square8", 2)
if is_active:
    print(f"Cooldown active: {remaining}s remaining")

# Check daily rollback limit
max_reached, count = monitor.is_max_daily_rollbacks_reached()
if max_reached:
    print(f"Max daily rollbacks reached: {count}")

# Bypass cooldown for manual rollbacks (use with caution)
monitor.set_cooldown_bypass(True)
```

### Rollback Criteria

| Criterion                     | Default | Description                                  |
| ----------------------------- | ------- | -------------------------------------------- |
| `elo_regression_threshold`    | -30.0   | Elo drop triggering regression               |
| `min_games_for_regression`    | 20      | Minimum games before checking                |
| `consecutive_checks_required` | 3       | Consecutive failing checks to trigger        |
| `min_win_rate`                | 0.40    | Win rate below which triggers rollback       |
| `cooldown_seconds`            | 3600    | Seconds between rollbacks for same config    |
| `max_rollbacks_per_day`       | 3       | Max rollbacks/day before manual intervention |

## Notification Hooks

### Overview

Notification hooks enable alerts for rollback events via multiple channels.

### Configuration

Configure hooks in `config/notification_hooks.yaml`:

```yaml
enabled: true

logging:
  enabled: true
  logger_name: 'ringrift.rollback'

slack:
  enabled: true
  webhook_url: 'https://hooks.slack.com/services/YOUR/WEBHOOK/URL'
  events:
    at_risk: true
    rollback_triggered: true
    rollback_completed: true

pagerduty:
  enabled: true
  routing_key: 'your-pagerduty-routing-key'
  severity_mapping:
    at_risk: 'warning'
    rollback_triggered: 'critical'
    rollback_completed: 'info'

opsgenie:
  enabled: true
  api_key: 'your-opsgenie-api-key'
  region: 'us' # or 'eu'
```

### Loading Configuration

```python
from app.training.notification_config import (
    load_notification_hooks,
    load_rollback_config,
)

# Load hooks from config file
hooks = load_notification_hooks()

# Or load full config including criteria overrides
config = load_rollback_config()
monitor = RollbackMonitor(
    criteria=config.criteria,
    notification_hooks=config.hooks,
)
```

### Custom Hooks

```python
from app.training.promotion_controller import NotificationHook

class CustomHook(NotificationHook):
    def on_regression_detected(self, model_id, status):
        print(f"Regression detected: {model_id}")

    def on_at_risk(self, model_id, status):
        print(f"Model at risk: {model_id}")

    def on_rollback_triggered(self, event):
        print(f"Rollback: {event.current_model_id} -> {event.rollback_model_id}")

    def on_rollback_completed(self, event, success):
        print(f"Rollback {'succeeded' if success else 'FAILED'}")

monitor.add_notification_hook(CustomHook())
```

## A/B Testing

### Overview

The `ABTestManager` enables concurrent model comparison before promotion.

### Usage

```python
from app.training.promotion_controller import (
    ABTestManager,
    ABTestConfig,
)

manager = ABTestManager()

# Start a test
config = ABTestConfig(
    test_id="test_v42_vs_v41",
    control_model_id="model_v41",
    treatment_model_id="model_v42",
    traffic_split=0.5,  # 50/50 split
    min_games_per_variant=100,
    significance_threshold=0.95,
    auto_promote=False,
)
manager.start_test(config)

# Get model for a game (routes based on traffic split)
model_id = manager.get_model_for_game("test_v42_vs_v41")

# Analyze results
result = manager.analyze_test("test_v42_vs_v41")
print(f"Elo difference: {result.elo_difference}")
print(f"Confidence: {result.confidence:.1%}")
print(f"Winner: {result.winner}")
print(f"Recommendation: {result.recommendation}")

# Stop test and get final results
final_result = manager.stop_test("test_v42_vs_v41")
```

### Test Configuration

| Parameter                | Default | Description                             |
| ------------------------ | ------- | --------------------------------------- |
| `test_id`                | —       | Unique identifier for the test          |
| `control_model_id`       | —       | Baseline model                          |
| `treatment_model_id`     | —       | New model being tested                  |
| `traffic_split`          | 0.5     | Fraction of traffic to treatment (0-1)  |
| `min_games_per_variant`  | 100     | Min games before drawing conclusions    |
| `significance_threshold` | 0.95    | Required statistical significance       |
| `auto_promote`           | False   | Auto-promote winner when test concludes |

## CLI Commands

### Rollback Commands

```bash
# Check if a model is showing regression
python scripts/elo_reconciliation_cli.py check-regression \
    --model-id model_v42 \
    --baseline-model model_v41 \
    --board-type square8

# Compare against multiple baselines
python scripts/elo_reconciliation_cli.py compare-baselines \
    --model-id model_v42 \
    --num-baselines 3

# Manual rollback (requires confirmation)
python scripts/elo_reconciliation_cli.py rollback \
    --from-model model_v42 \
    --to-model model_v41 \
    --reason "Performance regression"

# Dry run rollback
python scripts/elo_reconciliation_cli.py rollback \
    --from-model model_v42 \
    --to-model model_v41 \
    --dry-run

# View rollback history and at-risk models
python scripts/elo_reconciliation_cli.py rollback-status
```

### Drift History Commands

```bash
# View drift history with trend analysis
python scripts/elo_reconciliation_cli.py drift-history

# Backfill historical drift data from existing matches
python scripts/elo_reconciliation_cli.py backfill-history
```

## Grafana Dashboard Panels

The Rollback Monitoring section includes:

| Panel                | Type       | Description                      |
| -------------------- | ---------- | -------------------------------- |
| Models At Risk       | Stat       | Count of at-risk models          |
| Auto Rollbacks (24h) | Stat       | Rollbacks in last 24 hours       |
| Model Elo Regression | Timeseries | Elo regression over time         |
| Rollback Events      | Timeseries | Rollback events with results     |
| Regression Checks    | Timeseries | Check frequency and trigger rate |

## Alerting Rules

Additional alerts in `config/monitoring/alerting-rules.yaml`:

| Alert                    | Severity | Description                        |
| ------------------------ | -------- | ---------------------------------- |
| `AutoRollbackTriggered`  | High     | Automatic rollback triggered       |
| `AutoRollbackFailed`     | Critical | Automatic rollback FAILED          |
| `ModelAtRiskOfRollback`  | Warning  | Model at risk for 15+ minutes      |
| `SevereEloRegression`    | High     | Elo regression >50 for 10+ minutes |
| `MultipleRollbacksInDay` | Warning  | 3+ rollbacks in 24 hours           |

---

## Elo Calculation Methods

### Glicko-Style Rating Deviation

The Elo system uses Glicko-style rating deviation (RD) to track uncertainty in ratings (`app/tournament/elo.py`):

```python
from app.tournament.elo import EloRating

# Ratings include deviation (uncertainty)
rating = EloRating(rating=1500.0, rating_deviation=350.0)

# Get 95% confidence interval
lower, upper = rating.confidence_interval(0.95)
print(f"Rating: {rating.rating} ± {rating.ci_95}")  # e.g., "1500 ± 94"

# Rating deviation decays over games
# New model: RD = 350 (high uncertainty)
# After 100 games: RD → 50 (minimum, baseline uncertainty)
```

| Parameter  | Value     | Description               |
| ---------- | --------- | ------------------------- |
| Initial RD | 350.0     | New model uncertainty     |
| Minimum RD | 50.0      | Baseline after many games |
| Decay rate | 100 games | Games to reach minimum RD |

### Adaptive K-Factors

K-factor scales with player experience and rating level:

| Condition      | K-Factor | Description                      |
| -------------- | -------- | -------------------------------- |
| First 30 games | 40.0     | Fast adjustment for new models   |
| Established    | 32.0     | Normal adjustment rate           |
| Above 2400 Elo | 16.0     | Slower for stable top performers |

### Wilson Score Confidence Intervals

Promotion gates use Wilson score intervals for statistical significance (`app/training/significance.py`):

```python
from app.training.significance import wilson_score_interval, wilson_lower_bound

# Calculate confidence interval for win rate
wins, total = 35, 50
lower, upper = wilson_score_interval(wins, total, confidence=0.95)
print(f"Win rate: {wins/total:.1%}, 95% CI: [{lower:.1%}, {upper:.1%}]")

# For promotion: use lower bound as conservative estimate
threshold = 0.55
if wilson_lower_bound(wins, total) > threshold:
    print("Statistically significant - promote!")
```

**Promotion only proceeds when the lower bound of the confidence interval exceeds the threshold**, preventing promotions based on lucky streaks.

### Pinned Baseline Anchoring

Random baselines are pinned at 400 Elo to prevent rating inflation (primary:
`app/training/elo_service.py`; legacy utility in `app/tournament/unified_elo_db.py`):

```python
# baseline_random models are immutable anchors
# This calibrates the entire rating scale

# Reset if drift occurs
db.reset_pinned_baselines()  # Resets to 400 Elo
```

### Multiplayer Elo Decomposition

For 3-4 player games, results are decomposed into virtual pairwise matchups:

```python
# 4-player game with final rankings: [P1, P3, P2, P4]
# Decomposed into: P1>P3, P1>P2, P1>P4, P3>P2, P3>P4, P2>P4
# K-factor scaled by 1/(n_players-1) for stability
```

## O(n) Gauntlet Evaluation

For tournaments with 100+ models, gauntlet evaluation provides O(n) scaling instead of O(n²) round-robin (`app/tournament/distributed_gauntlet.py`):

```python
import asyncio
from pathlib import Path
from app.tournament.distributed_gauntlet import DistributedNNGauntlet

# Each model plays fixed baselines instead of all other models
gauntlet = DistributedNNGauntlet(
    elo_db_path=Path("data/unified_elo.db"),
    model_dir=Path("models"),
)

# Baselines are selected automatically per config:
# - Best model by Elo
# - Median model by Elo
# - Lower quartile model by Elo
# - random_ai anchor

# Results tracked in gauntlet_runs and gauntlet_results tables
asyncio.run(gauntlet.run_gauntlet("hex8_2p"))
```

**When to use gauntlet vs round-robin:**

| Models | Method      | Games | Time     |
| ------ | ----------- | ----- | -------- |
| <50    | Round-robin | O(n²) | Fast     |
| 50-100 | Hybrid      | Mixed | Medium   |
| >100   | Gauntlet    | O(n)  | Scalable |

## Game Deduplication

The database prevents double-counting games from retries/crashes (`app/tournament/unified_elo_db.py`):

```sql
-- Unique index on game_id prevents duplicates
CREATE UNIQUE INDEX idx_match_game_id ON match_history(game_id);

-- Transaction uses BEGIN IMMEDIATE to prevent TOCTOU races
```

## Board Type Support

All Elo features support the following board types:

| Board Type | Description                     | Features |
| ---------- | ------------------------------- | -------- |
| square8    | 8×8 square grid                 | 768      |
| hex8       | Radius-4 hexagonal (61 cells)   | 972      |
| square19   | 19×19 square grid               | 4332     |
| hexagonal  | Radius-12 hexagonal (469 cells) | 7500     |

Elo is tracked separately per `(board_type, num_players)` configuration.
