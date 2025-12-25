# AI Ladder Health & Drift Monitoring Specification (H-AI-15)

> **Status (2025-12-06): New – H-AI-15 specification.**  
> **Role:** Define metrics, thresholds, and procedures for ongoing health monitoring of the AI difficulty ladder, detecting drift and triggering remediation when calibration targets are violated.

---

## Overview

### Purpose

This specification establishes the framework for **ongoing health monitoring** of the AI difficulty ladder for Square-8 2-player (primary scope), with extension notes for other board types. It defines:

1. What metrics constitute ladder "health"
2. How to detect drift from calibration targets
3. When to trigger alerts requiring investigation or action
4. How metrics are collected, stored, and visualized
5. What response procedures apply when thresholds are breached

### Scope

- **Primary focus:** Square-8 2-player ladder tiers D2, D4, D6, D8
- **Extension notes:** Hexagonal, Square-19, and 3/4-player configurations
- **Integration:** Builds upon infrastructure from:
  - [`AI_CALIBRATION_RUNBOOK.md`](AI_CALIBRATION_RUNBOOK.md:1) – calibration procedure
  - [`AI_DIFFICULTY_CALIBRATION_ANALYSIS.md`](AI_DIFFICULTY_CALIBRATION_ANALYSIS.md:1) – analysis design
  - [`AI_TIER_TRAINING_AND_PROMOTION_PIPELINE.md`](AI_TIER_TRAINING_AND_PROMOTION_PIPELINE.md:1) – promotion pipeline
  - [`AI_TIER_PERF_BUDGETS.md`](AI_TIER_PERF_BUDGETS.md:1) – performance constraints

### Relationship to Calibration Runbook

The [`AI_CALIBRATION_RUNBOOK.md`](AI_CALIBRATION_RUNBOOK.md:1) defines **point-in-time calibration cycles** where an operator gathers data, runs analysis, and produces recommendations. This specification defines **continuous monitoring** that:

- Tracks metrics between calibration cycles
- Detects drift that warrants an unscheduled calibration run
- Provides alerting for immediate issues requiring investigation
- Establishes baselines for comparison across time windows

---

## Health Metrics

### Per-Tier Metrics

For each tier (D2, D4, D6, D8) on Square-8 2-player:

#### Win Rate vs Target

| Tier | Intended Segment | Target Human Win Rate | Acceptable Variance |
| ---- | ---------------- | --------------------- | ------------------- |
| D2   | New players      | 30-50%                | ±15%                |
| D4   | Intermediate     | 30-70%                | ±15%                |
| D6   | Strong players   | 40-60%                | ±10%                |
| D8   | Expert players   | 25-45%                | ±10%                |

**Metric:** `ladder_human_win_rate{tier, board_type, num_players, segment}`

- Collected from calibration telemetry (`isCalibrationOptIn=true` games)
- Segmented by player profile (new, intermediate, strong)
- Window: 7-day rolling average for alerts, 28-day for trends

#### Elo Rating

**Metric:** `ladder_tier_elo{tier, board_type, num_players}`

| Tier | Baseline Elo (relative) | Drift Tolerance |
| ---- | ----------------------- | --------------- |
| D2   | 1000 (reference)        | ±50             |
| D4   | 1200                    | ±75             |
| D6   | 1400                    | ±75             |
| D8   | 1600                    | ±100            |

- Calculated from automated AI-vs-AI evaluation tournaments
- Updated with each tier evaluation run
- Baseline established after promotion, tracked thereafter

#### Game Length Distribution

**Metric:** `ladder_game_length{tier, board_type, num_players, quantile}`

| Tier | Expected Mean (moves) | Expected Std Dev | Outlier Threshold |
| ---- | --------------------- | ---------------- | ----------------- |
| D2   | 40-60                 | ≤20              | >100 moves        |
| D4   | 50-80                 | ≤25              | >120 moves        |
| D6   | 60-100                | ≤30              | >150 moves        |
| D8   | 80-120                | ≤35              | >180 moves        |

- Abnormally short games may indicate AI weakness
- Abnormally long games may indicate stalemate-prone behavior

#### Decision Time

**Metric:** `ladder_decision_time_ms{tier, board_type, num_players, quantile}`

From [`AI_TIER_PERF_BUDGETS.md`](AI_TIER_PERF_BUDGETS.md:14):

| Tier | think_time_ms | max_avg_move_ms | max_p95_move_ms |
| ---- | ------------- | --------------- | --------------- |
| D2   | 200           | 220             | 250             |
| D4   | 2100          | 2310            | 2625            |
| D6   | 4800          | 5280            | 6000            |
| D8   | 9600          | 10560           | 12000           |

- Based on `think_time_ms * 1.10` (avg) and `think_time_ms * 1.25` (p95)
- Regression beyond these values triggers performance alerts

### Cross-Tier Metrics

#### Ladder Monotonicity

**Invariant:** Higher tiers must beat lower tiers in head-to-head evaluation.

| Matchup  | Minimum Expected Win Rate |
| -------- | ------------------------- |
| D4 vs D2 | ≥55%                      |
| D6 vs D4 | ≥55%                      |
| D8 vs D6 | ≥55%                      |

**Metric:** `ladder_monotonicity_win_rate{higher_tier, lower_tier, board_type, num_players}`

- Evaluated via periodic cross-tier tournaments
- Any violation is a **Critical** alert

#### Transition Sharpness

The difficulty step between adjacent tiers should be perceptible.

| Transition | Target Elo Gap | Minimum Gap |
| ---------- | -------------- | ----------- |
| D2→D4      | 200            | 100         |
| D4→D6      | 200            | 100         |
| D6→D8      | 200            | 100         |

**Metric:** `ladder_elo_gap{higher_tier, lower_tier, board_type, num_players}`

- Gaps below minimum indicate tiers are too similar (poor player experience)
- Gaps above 400 may indicate missing intermediate tier

#### Coverage Gaps

**Metric:** `ladder_tier_active{tier, board_type, num_players}`

- Binary indicator: 1 if tier has a promoted, active model; 0 otherwise
- Any tier without an active model is an **Informational** alert
- Tracked via [`tier_candidate_registry.square8_2p.json`](../../ai-service/config/tier_candidate_registry.square8_2p.json:1)

### Human Calibration Metrics

#### Perceived Difficulty Score

**Metric:** `ladder_perceived_difficulty{tier, board_type, num_players}`

- From `perceivedDifficulty` field (1-5 scale) in calibration telemetry
- Expected distribution:

| Tier | Target Mean | Low Tail (<2) | High Tail (>4) |
| ---- | ----------- | ------------- | -------------- |
| D2   | 2.0-2.5     | ≤30%          | ≤10%           |
| D4   | 2.8-3.2     | ≤15%          | ≤25%           |
| D6   | 3.2-3.8     | ≤10%          | ≤35%           |
| D8   | 3.8-4.5     | ≤5%           | ≤50%           |

#### Abandonment Rate by Tier

**Metric:** `ladder_abandonment_rate{tier, board_type, num_players}`

| Tier | Maximum Acceptable Rate |
| ---- | ----------------------- |
| D2   | ≤10%                    |
| D4   | ≤15%                    |
| D6   | ≤20%                    |
| D8   | ≤25%                    |

- High abandonment may indicate frustration (too hard) or boredom (too easy)
- Tracked from `result="abandoned"` in calibration telemetry

#### Tier Selection Distribution

**Metric:** `ladder_tier_selection_count{tier, board_type, num_players}`

- Tracks which tiers players choose
- Expected healthy distribution (rough):
  - D2: 20-30%
  - D4: 30-40%
  - D6: 20-30%
  - D8: 10-20%
- Severe skew may indicate perception issues or calibration problems

---

## Drift Detection

### Statistical Methods

#### Rolling Window Comparisons

Compare current window metrics against baseline:

| Comparison | Current Window | Baseline Window |
| ---------- | -------------- | --------------- |
| Short-term | 7 days         | 30 days         |
| Long-term  | 30 days        | 90 days         |

**Formula:**

```
drift_score = (current_mean - baseline_mean) / baseline_std_dev
```

#### Z-Score Thresholds

| Drift Score | Classification |
| ----------- | -------------- | ----- | ----------------- |
|             | z              | < 1.0 | Normal variance   |
| 1.0 ≤       | z              | < 2.0 | Elevated, monitor |
| 2.0 ≤       | z              | < 3.0 | Warning           |
|             | z              | ≥ 3.0 | Critical          |

#### Trend Detection

Distinguish between oscillation and monotonic drift:

- **Oscillation:** Metric varies around baseline with no clear direction
- **Monotonic Drift:** Metric trends consistently up or down over 3+ consecutive windows

**Method:** Linear regression slope over rolling windows

- Slope magnitude > 0.5% per day for 2+ weeks = potential drift
- Slope direction indicates strengthening or weakening

### Drift Categories

| Category                  | Description                              | Example                                              | Severity                |
| ------------------------- | ---------------------------------------- | ---------------------------------------------------- | ----------------------- |
| Win Rate Drift            | Tier win rate moves outside target band  | D4 drops from 45% to 35% human win rate over 2 weeks | Warning → Critical      |
| Elo Collapse              | Tier Elo drops significantly vs baseline | D6 loses 100+ Elo points in evaluation               | Critical                |
| Monotonicity Violation    | Lower tier starts beating higher tier    | D3 beats D4 in H2H (>50% win rate)                   | Critical                |
| Latency Regression        | Decision time exceeds budget             | D2 averaging 3s instead of 200ms                     | Critical                |
| Perceived Difficulty Skew | Player ratings diverge from target       | D4 getting mostly 1-2 ratings (too easy)             | Warning                 |
| Game Length Anomaly       | Games consistently too short or too long | D8 games averaging 40 moves (expected 80-120)        | Warning                 |
| Selection Avoidance       | Players avoiding a tier                  | D6 selection drops from 25% to 5%                    | Informational → Warning |

---

## Alert Thresholds

### Critical (Immediate Action Required)

Response time: **< 15 minutes** acknowledgement, **< 1 hour** resolution or mitigation

| Alert                           | Condition                                              | Impact                                   |
| ------------------------------- | ------------------------------------------------------ | ---------------------------------------- |
| **LadderMonotonicityViolation** | Higher tier loses to lower tier (≥50% over 100+ games) | Ladder ordering broken, player confusion |
| **TierWinRateSevereDrift**      | Human win rate >20% outside target band for 7+ days    | Tier is unplayable for intended segment  |
| **TierLatencyCritical**         | avg_ms > 3× budget OR p95 > 3× budget                  | Unacceptable UX, game feels broken       |
| **TierEloCriticalDrop**         | Elo drops >150 from baseline                           | Model severely weakened                  |
| **NoActiveModelForTier**        | Tier has no promoted model in production               | Tier completely unavailable              |

### Warning (Investigation Required)

Response time: **< 4 hours** acknowledgement, **< 24 hours** investigation complete

| Alert                       | Condition                                             | Impact                        |
| --------------------------- | ----------------------------------------------------- | ----------------------------- |
| **TierWinRateDrift**        | Human win rate 10-20% outside target band for 7+ days | Tier calibration drifting     |
| **TierEloDrift**            | Elo drifts >50 points from baseline over 7 days       | Model strength changing       |
| **TierLatencyElevated**     | avg_ms > 1.5× budget OR p95 > 2× budget               | UX degradation for some users |
| **GameLengthAnomaly**       | Mean game length >2 std dev from expected             | Gameplay patterns abnormal    |
| **PerceivedDifficultySkew** | Mean rating drifts >0.5 from target for 14+ days      | Player perception misaligned  |
| **HighAbandonmentRate**     | Abandonment >1.5× threshold for 7+ days               | Players frustrated or bored   |
| **TransitionSharpnessLow**  | Elo gap between adjacent tiers <100                   | Tiers feel indistinguishable  |

### Informational (Track and Log)

Response time: **Next business day** review, **weekly** summary

| Alert                       | Condition                                  | Impact                             |
| --------------------------- | ------------------------------------------ | ---------------------------------- |
| **MinorMetricFluctuation**  | Any metric fluctuates <10% from baseline   | Normal variance, no action         |
| **NewCandidateNotPromoted** | Candidate passes gate but not yet promoted | Action pending, not urgent         |
| **CalibrationDataStale**    | No calibration games for tier in 30+ days  | Data may not reflect current state |
| **TierSelectionSkew**       | Tier selection varies >50% from expected   | Possible player perception issue   |
| **EvaluationDue**           | Last tier evaluation >14 days ago          | Routine refresh recommended        |

---

## Monitoring Infrastructure

### Data Collection

#### Sources

| Source                | Metrics Collected                                              | Collection Method                                                                                                   |
| --------------------- | -------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| Calibration Telemetry | Human win rate, perceived difficulty, game length, abandonment | Client events via [`difficultyCalibrationTelemetry.ts`](../../src/client/utils/difficultyCalibrationTelemetry.ts:1) |
| Tier Evaluation Runs  | Win rates vs opponents, Elo calculations                       | [`run_tier_evaluation.py`](../../ai-service/scripts/run_tier_evaluation.py:1) output                                |
| Perf Benchmarks       | Decision latency (avg, p95)                                    | [`run_tier_perf_benchmark.py`](../../ai-service/scripts/run_tier_perf_benchmark.py:1) output                        |
| Candidate Registry    | Model status, promotion history                                | [`tier_candidate_registry.square8_2p.json`](../../ai-service/config/tier_candidate_registry.square8_2p.json:1)      |
| Calibration Analysis  | Aggregated calibration metrics per window                      | [`analyze_difficulty_calibration.py`](../../ai-service/scripts/analyze_difficulty_calibration.py:1) output          |

#### Collection Frequency

| Metric Type                  | Collection Frequency | Retention                            |
| ---------------------------- | -------------------- | ------------------------------------ |
| Calibration telemetry events | Real-time            | 180 days (aggregates), 30 days (raw) |
| Per-game metrics             | Per game completion  | 90 days                              |
| Tier evaluation results      | On evaluation run    | Indefinite                           |
| Perf benchmark results       | On benchmark run     | Indefinite                           |
| Aggregated health metrics    | Daily rollup         | Indefinite                           |

#### Data Retention Policy

- **Raw calibration events:** 30 days (storage-intensive)
- **Aggregated daily metrics:** 180 days
- **Evaluation/benchmark artifacts:** Archive indefinitely under `ai-service/logs/`
- **Calibration run summaries:** Archive indefinitely under `docs/ai/calibration_runs/`

### Storage

#### Primary Storage Locations

| Data Type              | Location                         | Format                    |
| ---------------------- | -------------------------------- | ------------------------- |
| Calibration aggregates | Prometheus/metrics warehouse     | Time-series metrics       |
| Evaluation results     | `ai-service/logs/tier_eval/`     | JSON per run              |
| Perf reports           | `ai-service/logs/tier_perf/`     | JSON per run              |
| Calibration summaries  | `docs/ai/calibration_runs/`      | JSON + Markdown per cycle |
| Health snapshots       | `ai-service/logs/ladder_health/` | Daily JSON snapshots      |

#### Schema for Ladder Health Data

```json
{
  "snapshot_date": "2025-12-06",
  "board_type": "square8",
  "num_players": 2,
  "tiers": {
    "D2": {
      "model_id": "heuristic_v1_2p",
      "elo": 1000,
      "elo_baseline": 1000,
      "elo_drift": 0,
      "human_win_rate_7d": 0.42,
      "human_win_rate_target": 0.4,
      "perceived_difficulty_mean": 2.3,
      "game_length_mean": 52,
      "decision_time_avg_ms": 180,
      "decision_time_p95_ms": 220,
      "abandonment_rate": 0.05,
      "selection_percentage": 0.25,
      "last_evaluation": "2025-12-01T14:00:00Z",
      "last_perf_benchmark": "2025-12-01T14:30:00Z",
      "status": "healthy"
    }
    // ... D4, D6, D8
  },
  "cross_tier": {
    "monotonicity_violations": [],
    "elo_gaps": {
      "D4_D2": 195,
      "D6_D4": 210,
      "D8_D6": 190
    }
  },
  "alerts_active": []
}
```

#### Relationship to Existing Databases

- **GameReplayDB:** Source for game outcomes, not consumed directly by health monitoring
- **Tier Candidate Registry:** Consulted for current model assignments
- **Prometheus:** Primary metrics storage for alerting integration

### Visualization

#### Dashboard Requirements

**Ladder Health Dashboard** (Grafana or equivalent):

1. **Overview Panel**
   - Current status of all tiers (healthy/warning/critical)
   - Active alerts count
   - Days since last calibration cycle

2. **Per-Tier Panels** (one per tier)
   - Human win rate trend (7d, 30d)
   - Elo trend vs baseline
   - Decision latency (avg, p95) vs budget
   - Game length distribution
   - Perceived difficulty distribution

3. **Cross-Tier Panel**
   - Elo ladder visualization
   - Monotonicity check results
   - Tier selection distribution

4. **Alert History Panel**
   - Recent alerts with resolution status
   - Alert frequency trends

#### Key Charts and Graphs

| Chart                | Purpose                                   | Data Source                       |
| -------------------- | ----------------------------------------- | --------------------------------- |
| Win Rate Time Series | Track human win rate per tier over time   | Calibration telemetry aggregates  |
| Elo Ladder Bar Chart | Visualize tier strength ordering          | Evaluation results                |
| Latency Heatmap      | Show decision time by tier and percentile | Perf benchmark results            |
| Drift Score Gauge    | Show current drift status per metric      | Computed from rolling comparisons |
| Alert Timeline       | Show alert history and resolution         | Alert system logs                 |

#### Example Queries for Common Monitoring Tasks

**Win rate drift for D4 (PromQL):**

```promql
abs(
  avg_over_time(ladder_human_win_rate{tier="D4", board_type="square8", num_players="2"}[7d])
  - avg_over_time(ladder_human_win_rate{tier="D4", board_type="square8", num_players="2"}[30d])
) / stddev_over_time(ladder_human_win_rate{tier="D4", board_type="square8", num_players="2"}[30d])
```

**Monotonicity check (pseudo-SQL):**

```sql
SELECT higher_tier, lower_tier, win_rate
FROM ladder_monotonicity_results
WHERE board_type = 'square8' AND num_players = 2
  AND win_rate < 0.55
ORDER BY evaluation_date DESC
LIMIT 10;
```

**Tier selection distribution (PromQL):**

```promql
sum(ladder_tier_selection_count{board_type="square8", num_players="2"}) by (tier)
/ ignoring(tier) group_left sum(ladder_tier_selection_count{board_type="square8", num_players="2"})
```

### Automation

#### Scheduled Health Checks

| Check                        | Schedule         | Action                                     |
| ---------------------------- | ---------------- | ------------------------------------------ |
| Daily health snapshot        | 00:00 UTC        | Generate and store daily health JSON       |
| Weekly cross-tier evaluation | Sunday 02:00 UTC | Run abbreviated cross-tier tournament      |
| Weekly perf benchmark        | Sunday 04:00 UTC | Run perf benchmarks for all tiers          |
| Monthly full calibration     | 1st of month     | Trigger full calibration cycle per runbook |

#### Alert Delivery Mechanism

Following patterns from [`ALERTING_THRESHOLDS.md`](../operations/ALERTING_THRESHOLDS.md:1):

| Severity      | Channels                                      | Routing         |
| ------------- | --------------------------------------------- | --------------- |
| Critical      | Slack #ringrift-ai-critical, PagerDuty, Email | AI/ML on-call   |
| Warning       | Slack #ringrift-ai-alerts                     | AI team channel |
| Informational | Slack #ringrift-ai-info                       | Weekly digest   |

#### Integration with Existing Alerting

- **Prometheus Alertmanager:** Primary alert pipeline
- **Alert rule file:** `monitoring/prometheus/alerts.yml` (add `ai-ladder-health` group)
- **Grafana:** Dashboard alerts as secondary channel
- **MetricsService:** Expose ladder health metrics via `/metrics` endpoint

---

## Response Procedures

### Critical Alert Response

#### Immediate Steps (0-15 minutes)

1. **Acknowledge alert** in Alertmanager/Slack
2. **Assess scope:** Which tier(s) affected? How many games/players impacted?
3. **Check for obvious causes:**
   - Recent deployments or model promotions
   - AI service health (see [`AIServiceDown`](../operations/ALERTING_THRESHOLDS.md:110))
   - Dependency failures (DB, Redis)
4. **Initial mitigation if needed:**
   - For latency: Consider temporary difficulty reduction
   - For monotonicity: Flag affected tiers for review

#### Investigation (15-60 minutes)

1. **Gather evidence:**
   - Pull latest health snapshot
   - Review recent evaluation/benchmark logs
   - Check calibration telemetry for anomalies
2. **Identify root cause:**
   - Model regression after promotion?
   - Infrastructure issue?
   - Data pipeline issue (stale or incorrect metrics)?
3. **Determine rollback eligibility:**
   - If recent promotion caused issue, prepare rollback plan
   - If not promotion-related, escalate to engineering

#### Rollback Criteria

Rollback to previous model if:

- Issue started within 7 days of promotion
- Previous model was stable for 14+ days
- Rollback can be executed via registry update without code changes

Rollback procedure:

1. Update `tier_candidate_registry.square8_2p.json` to revert `current` to previous model
2. Restart AI service to pick up new configuration
3. Run abbreviated evaluation to confirm rollback success
4. Update incident channel with status

#### Escalation Path

| Time   | Action                                               |
| ------ | ---------------------------------------------------- |
| 15 min | Notify AI team lead if unresolved                    |
| 30 min | Engage AI/ML on-call if still investigating          |
| 1 hour | Escalate to engineering leadership if user-impacting |

#### Post-Incident Review

Required within 48 hours for all Critical alerts:

- Root cause analysis document
- Contributing factors identified
- Preventive measures proposed
- Alert threshold review (too sensitive? too late?)

### Warning Response

#### Investigation Timeline

| Milestone              | Timeframe  | Action                             |
| ---------------------- | ---------- | ---------------------------------- |
| Acknowledge            | < 4 hours  | Confirm alert seen, assign owner   |
| Initial assessment     | < 8 hours  | Review metrics, form hypothesis    |
| Investigation complete | < 24 hours | Root cause identified or escalated |
| Resolution plan        | < 48 hours | Plan documented and scheduled      |

#### Diagnostic Procedures

1. **Pull extended metric history** (30-60 days)
2. **Run ad-hoc evaluation** if Elo drift suspected:
   ```bash
   python -m ai-service.scripts.run_tier_gate \
     --tier D4 --seed 42 --output-json /tmp/d4_check.json
   ```
3. **Run ad-hoc perf benchmark** if latency suspected:
   ```bash
   python -m ai-service.scripts.run_tier_perf_benchmark \
     --tier D4 --num-games 4 --output-json /tmp/d4_perf.json
   ```
4. **Compare with last known good state:**
   - Find last healthy calibration run
   - Diff current vs historical metrics

#### Remediation Options

| Issue                           | Options                                                  |
| ------------------------------- | -------------------------------------------------------- |
| Win rate drift (too easy)       | Queue stronger candidate, increase search depth/time     |
| Win rate drift (too hard)       | Increase randomness, reduce search depth                 |
| Elo drift                       | Re-evaluate model, consider rollback if recent promotion |
| Latency regression              | Profile AI, optimize or reduce search budget             |
| Perceived difficulty misaligned | Review tier descriptors, consider reclassification       |

### Routine Maintenance

#### Periodic Recalibration Schedule

| Activity               | Frequency                        | Owner           |
| ---------------------- | -------------------------------- | --------------- |
| Full calibration cycle | Monthly or after major promotion | AI/Ladder Owner |
| Cross-tier evaluation  | Weekly automated                 | Automation      |
| Perf benchmark sweep   | Weekly automated                 | Automation      |
| Registry health check  | Daily automated                  | Automation      |
| Alert threshold review | Quarterly                        | AI Team         |

#### Model Refresh Cadence

- **D2 (Heuristic):** Refresh when new optimized weights available (~quarterly)
- **D4 (Minimax):** Refresh on heuristic or search config improvements (~quarterly)
- **D6/D8 (NN-backed):** Refresh on new training runs (~monthly when active)

#### Baseline Update Procedures

After each successful calibration cycle:

1. **Capture current state** as new baseline:
   ```bash
   python -m ai-service.scripts.capture_ladder_baseline \
     --output ai-service/data/baselines/square8_2p_YYYYMMDD.json
   ```
2. **Update alert thresholds** if baseline shifted significantly
3. **Archive previous baseline** for historical comparison
4. **Document baseline change** in calibration run notes

---

## Implementation Plan

### Phase 1: Manual Monitoring (Immediate)

**Objective:** Establish monitoring coverage with minimal automation using existing tools.

#### Metrics to Track Manually

| Metric                  | Collection Method                               | Frequency            | Owner         |
| ----------------------- | ----------------------------------------------- | -------------------- | ------------- |
| Human win rate by tier  | Query calibration telemetry                     | Weekly               | AI/Data Owner |
| Tier evaluation results | Run `run_tier_gate.py`                          | Monthly or on-demand | AI Owner      |
| Perf benchmark results  | Run `run_tier_perf_benchmark.py`                | Monthly              | AI Owner      |
| Cross-tier monotonicity | Manual tournament via `run_tournament.py basic` | Quarterly            | AI Owner      |

#### Cadence for Manual Review

- **Weekly:** Check calibration telemetry dashboards for obvious anomalies
- **Bi-weekly:** Review tier evaluation results, Elo trends
- **Monthly:** Run full calibration cycle per runbook
- **After any promotion:** Run abbreviated evaluation and perf check

#### Documentation of Findings

Maintain running log at `docs/ai/calibration_runs/health_log.md`:

```markdown
# Ladder Health Log

## 2025-12-06

- D2: Healthy, win rate 42%, Elo stable
- D4: Healthy, win rate 55%, Elo stable
- D6: Monitoring - win rate at 62% (above 60% target), perceived difficulty trending low
- D8: Healthy, win rate 35%, Elo stable

Action: Schedule D6 evaluation for next week.
```

### Phase 2: Automated Collection (1-2 Months)

**Objective:** Automate metric collection and basic alerting.

#### Scripts/Tools to Automate

1. **Daily health snapshot generator:**
   - New script: `ai-service/scripts/generate_ladder_health_snapshot.py`
   - Collects all metrics from sources
   - Writes daily JSON to `ai-service/logs/ladder_health/`

2. **Prometheus metrics exporter:**
   - Extend [`MetricsService`](../../src/server/services/MetricsService.ts:1) with ladder health gauges:
     - `ringrift_ladder_human_win_rate{tier, board_type, num_players, segment}`
     - `ringrift_ladder_tier_elo{tier, board_type, num_players}`
     - `ringrift_ladder_decision_time_ms{tier, board_type, num_players, quantile}`
   - Feed from AI service or aggregation layer

3. **Registry health checker:**
   - Cron job to validate registry integrity
   - Alert if tier missing current model

#### Storage Setup

1. Create `ai-service/logs/ladder_health/` directory structure
2. Define JSON schema (see Storage section above)
3. Set up retention/archival scripts

#### Basic Alerting

Add to `monitoring/prometheus/alerts.yml`:

```yaml
groups:
  - name: ai-ladder-health
    rules:
      - alert: LadderTierEloDropped
        expr: ringrift_ladder_tier_elo < (ringrift_ladder_tier_elo_baseline - 100)
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: 'Tier {{ $labels.tier }} Elo dropped significantly'

      - alert: LadderLatencyBudgetExceeded
        expr: ringrift_ladder_decision_time_ms{quantile="0.95"} > (ringrift_ladder_latency_budget_p95 * 1.5)
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: 'Tier {{ $labels.tier }} latency exceeds budget'
```

### Phase 3: Dashboard & Full Alerts (2-4 Months)

**Objective:** Complete visualization and alert infrastructure.

#### Visualization Dashboard

1. Create Grafana dashboard `ai-ladder-health.json`:
   - Overview panel with tier status indicators
   - Per-tier metric panels
   - Cross-tier comparison panels
   - Alert history panel

2. Add to dashboard registry and deployment

#### Automated Alert Rules

Full implementation of alerts from "Alert Thresholds" section:

```yaml
# Critical alerts
- alert: LadderMonotonicityViolation
  expr: ringrift_ladder_monotonicity_win_rate{comparison="higher_vs_lower"} < 0.50
  for: 0m # Immediate on detection
  labels:
    severity: critical

- alert: TierWinRateSevereDrift
  # Win rate >20% outside band for 7 days
  expr: |
    abs(
      avg_over_time(ringrift_ladder_human_win_rate[7d]) 
      - ringrift_ladder_win_rate_target
    ) > 0.20
  for: 7d
  labels:
    severity: critical

# Warning alerts
- alert: TierWinRateDrift
  # Win rate 10-20% outside band for 7 days
  expr: |
    abs(
      avg_over_time(ringrift_ladder_human_win_rate[7d])
      - ringrift_ladder_win_rate_target
    ) > 0.10
    and
    abs(
      avg_over_time(ringrift_ladder_human_win_rate[7d])
      - ringrift_ladder_win_rate_target  
    ) <= 0.20
  for: 7d
  labels:
    severity: warning
```

#### Integration Testing

1. Create test fixtures simulating drift scenarios
2. Verify alerts fire correctly in staging
3. Test alert routing to appropriate channels
4. Validate dashboard updates in real-time

---

## Appendix

### Metric Calculation Examples

#### Win Rate Calculation

```python
# From calibration telemetry aggregates
def calculate_human_win_rate(tier, window_days=7):
    games = query_calibration_games(
        tier=tier,
        board_type="square8",
        num_players=2,
        is_calibration_opt_in=True,
        last_n_days=window_days
    )
    wins = sum(1 for g in games if g.result == "win")
    return wins / len(games) if games else None
```

#### Elo Calculation (from evaluation)

```python
# From tier evaluation results
def calculate_tier_elo(eval_result: TierEvaluationResult, k=32):
    """
    Calculate Elo based on win rates vs known-strength opponents.
    Uses simple Elo update model with k-factor.
    """
    base_elo = 1000  # D2 baseline
    elo = base_elo + (eval_result.candidate_difficulty - 2) * 200

    # Adjust based on actual performance vs expectations
    for matchup in eval_result.matchups:
        expected = 1 / (1 + 10 ** ((matchup.opponent_elo - elo) / 400))
        actual = matchup.win_rate
        elo += k * (actual - expected)

    return elo
```

#### Drift Score Calculation

```python
def calculate_drift_score(current_window, baseline_window):
    """
    Calculate z-score drift between windows.
    """
    current_mean = np.mean(current_window)
    baseline_mean = np.mean(baseline_window)
    baseline_std = np.std(baseline_window)

    if baseline_std == 0:
        return 0 if current_mean == baseline_mean else float('inf')

    return (current_mean - baseline_mean) / baseline_std
```

### Historical Baseline Data

Reference values from most recent calibration run (placeholder for actual run):

| Tier | Elo  | Human Win Rate | Perceived Difficulty | Avg Decision Time | P95 Decision Time |
| ---- | ---- | -------------- | -------------------- | ----------------- | ----------------- |
| D2   | 1000 | 42%            | 2.3                  | 180ms             | 220ms             |
| D4   | 1195 | 52%            | 2.9                  | 1800ms            | 2200ms            |
| D6   | 1410 | 48%            | 3.4                  | 4200ms            | 5100ms            |
| D8   | 1590 | 38%            | 4.1                  | 8500ms            | 10500ms           |

_Note: Update these values after each successful calibration cycle._

### Related Documents

| Document                                                                                     | Relationship                                                                      |
| -------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------- |
| [`AI_CALIBRATION_RUNBOOK.md`](AI_CALIBRATION_RUNBOOK.md:1)                                   | Defines per-cycle calibration procedure; this spec adds continuous monitoring     |
| [`AI_DIFFICULTY_CALIBRATION_ANALYSIS.md`](AI_DIFFICULTY_CALIBRATION_ANALYSIS.md:1)           | Defines calibration metrics and targets; this spec operationalizes them           |
| [`AI_TIER_TRAINING_AND_PROMOTION_PIPELINE.md`](AI_TIER_TRAINING_AND_PROMOTION_PIPELINE.md:1) | Defines training/promotion loop; this spec monitors promotion outcomes            |
| [`AI_TIER_PERF_BUDGETS.md`](AI_TIER_PERF_BUDGETS.md:1)                                       | Defines latency budgets; this spec monitors adherence                             |
| [`ALERTING_THRESHOLDS.md`](../operations/ALERTING_THRESHOLDS.md:1)                           | Defines general alerting patterns; this spec follows those patterns for AI ladder |
| [`ladder_config.py`](../../ai-service/app/config/ladder_config.py:1)                         | Defines ladder tier configurations; this spec monitors their health               |
| [`tier_eval_runner.py`](../../ai-service/app/training/tier_eval_runner.py:1)                 | Implements tier evaluation; this spec consumes its outputs                        |

### Extension Notes for Other Board Types

This specification focuses on Square-8 2-player but can be extended to:

#### Square-19 2-Player

- Adjust expected game lengths (longer games)
- Adjust latency budgets (larger search space)
- Same tier structure (D2/D4/D6/D8) where available

#### Hexagonal 2-Player

- Adjust expected game lengths
- Same monitoring structure
- May have different tier coverage initially

#### 3/4-Player Variants

- Adjust human win rate targets (multi-player dynamics differ)
- Adjust Elo calculations for 3/4-way games
- May require new perceived difficulty calibration

For each new configuration:

1. Define tier-specific baselines
2. Configure appropriate alert thresholds
3. Add to health snapshot schema
4. Create dashboard panels

---

_This specification completes H-AI-15 of the AI difficulty calibration remediation track._
