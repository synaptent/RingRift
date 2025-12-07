# AI Ladder Change Guardrails (H-AI-16)

> **Status (2025-12-06): New – H-AI-16 specification.**  
> **Role:** Define guardrails to prevent ladder changes from breaking difficulty progression.

---

## Overview

### Purpose

The AI difficulty ladder is a **mission-critical component** for player experience. Changes to tier configurations, models, or promotion criteria can have cascading effects on the entire difficulty progression. This specification establishes **guardrails** to prevent well-intentioned changes from breaking ladder integrity.

### Scope

All changes to:

- Tier models (heuristic profiles, neural network checkpoints, search personas)
- Ladder configurations in [`ladder_config.py`](../../ai-service/app/config/ladder_config.py:1)
- Tier evaluation and promotion criteria
- Performance budgets and constraints
- Cross-tier matchup expectations

### Relationship to Other Documents

| Document                                                                                     | Relationship                                                            |
| -------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------- |
| [`AI_CALIBRATION_RUNBOOK.md`](AI_CALIBRATION_RUNBOOK.md:1)                                   | Source of calibration cycle procedures referenced in Category C changes |
| [`AI_LADDER_HEALTH_MONITORING_SPEC.md`](AI_LADDER_HEALTH_MONITORING_SPEC.md:1)               | Defines post-change monitoring metrics and alert thresholds             |
| [`AI_TIER_TRAINING_AND_PROMOTION_PIPELINE.md`](AI_TIER_TRAINING_AND_PROMOTION_PIPELINE.md:1) | Defines the training and promotion loop that produces candidates        |
| [`AI_TIER_PERF_BUDGETS.md`](AI_TIER_PERF_BUDGETS.md:1)                                       | Defines latency budgets that must be respected                          |

---

## Change Categories

### Category A: Low Risk (Self-Service)

Changes that can be made with minimal validation. These do not affect gameplay or perceived difficulty.

**Examples:**

- Updating tier metadata (names, descriptions, notes fields)
- Adjusting configuration that does not affect AI behavior:
  - Logging levels
  - Debug output formats
  - Telemetry verbosity
- Adding new candidate models to staging (not yet promoted to production)
- Updating documentation and comments in configuration files
- Adding new experimental tier configs for boards/player counts not yet in production

**Validation Required:**

- Standard code review
- Unit tests pass
- No changes to production AI behavior

### Category B: Medium Risk (Requires Validation)

Changes that require tier gate validation to ensure the ladder remains properly calibrated.

**Examples:**

- Promoting a new model to a tier (via [`run_tier_gate.py`](../../ai-service/scripts/run_tier_gate.py:1))
- Adjusting tier performance budgets (latency, memory)
- Modifying tier evaluation parameters:
  - `min_win_rate_vs_baseline`
  - `max_regression_vs_previous_tier`
  - `num_games` for evaluation
- Changing `think_time_ms` or `randomness` values for existing tiers
- Updating heuristic weight profiles used by production tiers

**Validation Required:**

- Tier gate evaluation for affected tier(s)
- Performance benchmark within budget
- Monotonicity preserved across adjacent tiers
- Previous tier snapshot saved for rollback

### Category C: High Risk (Requires Full Cycle)

Changes that require a full calibration cycle as defined in [`AI_CALIBRATION_RUNBOOK.md`](AI_CALIBRATION_RUNBOOK.md:1).

**Examples:**

- Adding or removing tiers from the ladder
- Changing win-rate targets for multiple tiers
- Modifying cross-tier matchup expectations (e.g., D4 vs D2 minimum win rate)
- Changing the model selection algorithm in [`get_ladder_tier_config()`](../../ai-service/app/config/ladder_config.py:279)
- Restructuring ladder tier definitions (e.g., changing from 4 tiers to 5)
- Changing the AI type for a tier (e.g., MINIMAX to MCTS)
- Modifying evaluation opponent configurations in [`TIER_EVAL_CONFIGS`](../../ai-service/app/training/tier_eval_config.py:53)

**Validation Required:**

- Full calibration cycle per runbook
- All tier gates pass
- Cross-tier monotonicity validated via [`run_full_tier_gating.py`](../../ai-service/scripts/run_full_tier_gating.py:1)
- Human calibration consideration (when available)
- Peer review required
- Rollback plan documented

### Category D: Critical (Exceptional Approval)

Changes that fundamentally alter ladder architecture and require exceptional approval.

**Examples:**

- Changing fundamental ladder architecture (e.g., moving from fixed tiers to dynamic Elo matching)
- Modifying core difficulty progression logic
- Emergency production hotfixes that bypass normal validation
- Changes affecting multiple board types simultaneously
- Architectural changes to how models are loaded or selected at runtime

**Validation Required:**

- All Category C requirements
- Business justification documented
- Impact assessment completed
- Explicit approval from project lead documented in PR/issue
- Staged rollout plan defined
- Extended monitoring period scheduled (14 days minimum)

---

## Pre-Change Checklists

### For Category A Changes

No formal checklist required. Standard development workflow applies:

- [ ] Code review completed
- [ ] Unit tests pass
- [ ] Change does not affect AI gameplay behavior

### For Category B Changes

Before proceeding with a Category B change:

- [ ] **Tier gate evaluation completed** for all affected tier(s)
  ```bash
  python -m ai-service.scripts.run_tier_gate \
    --tier D4 \
    --candidate-model-id <candidate_id> \
    --output-json logs/tier_gate/D4_gate.json \
    --promotion-plan-out logs/tier_gate/D4_promotion.json
  ```
- [ ] **Win rate within acceptable band** vs target (±10% as per [`AI_LADDER_HEALTH_MONITORING_SPEC.md`](AI_LADDER_HEALTH_MONITORING_SPEC.md:49))
- [ ] **Latency within budget** (for D4/D6/D8 tiers):
  ```bash
  python -m ai-service.scripts.run_tier_perf_benchmark \
    --tier D4 \
    --output-json logs/tier_perf/D4_perf.json
  ```
- [ ] **Monotonicity preserved**: Higher tier beats lower tier at ≥55% in head-to-head
- [ ] **Previous tier snapshot saved** to enable rollback:
  - Copy of current [`ladder_config.py`](../../ai-service/app/config/ladder_config.py:1)
  - Current model artifacts/checkpoints backed up
- [ ] **Change documented** in change log (see §Change Governance)

### For Category C Changes

Before proceeding with a Category C change:

- [ ] **Full calibration cycle completed** per [`AI_CALIBRATION_RUNBOOK.md`](AI_CALIBRATION_RUNBOOK.md:153)
  - Calibration aggregates JSON collected
  - Registry snapshot taken
  - Analysis CLI run with outputs archived
- [ ] **All tier gates pass**:
  ```bash
  python -m ai-service.scripts.run_full_tier_gating \
    --tier D6 \
    --candidate-id <candidate_id> \
    --run-dir logs/tier_gate/D6_full
  ```
- [ ] **Cross-tier monotonicity validated** for all tier pairs:
      | Matchup | Required Win Rate |
      |---------|------------------|
      | D4 vs D2 | ≥55% |
      | D6 vs D4 | ≥55% |
      | D8 vs D6 | ≥55% |
- [ ] **Human calibration consideration** (if calibration cohort data available):
  - Review perceived difficulty ratings
  - Check abandonment rates by tier
- [ ] **Change reviewed by at least one other team member**
- [ ] **Rollback plan documented** (specific model IDs to revert to)
- [ ] **Pre-change health snapshot captured** for comparison

### For Category D Changes

Before proceeding with a Category D change:

- [ ] **All Category C requirements met**
- [ ] **Business justification documented**:
  - Why is this change necessary?
  - What player experience problem does it solve?
  - What alternatives were considered?
- [ ] **Impact assessment completed**:
  - Which tiers/boards/player counts affected?
  - Estimated number of players impacted?
  - Risk of player churn if change fails?
- [ ] **Exceptional approval obtained**:
  - Approver name: ****\_\_\_\_****
  - Approval date: ****\_\_\_\_****
  - PR/Issue link: ****\_\_\_\_****
- [ ] **Staged rollout plan defined**:
  - Phase 1 target percentage: **\_**%
  - Escalation criteria for each phase
  - Full rollout timeline
- [ ] **Extended monitoring period scheduled**:
  - Minimum 14 days post-deployment
  - Specific metrics to watch documented
  - On-call escalation path defined

---

## Validation Gates

### Automated Gates (Must Pass)

These gates are enforced by tooling and must pass before promotion.

| Gate             | Command                                 | Pass Criteria                                                    | Applies To         |
| ---------------- | --------------------------------------- | ---------------------------------------------------------------- | ------------------ |
| **Tier Gate**    | `run_tier_gate.py --tier <N>`           | Win rate vs baseline ≥ threshold; no regression vs previous tier | All tiers          |
| **Perf Budget**  | `run_tier_perf_benchmark.py --tier <N>` | avg_ms ≤ max_avg; p95_ms ≤ max_p95                               | D4, D6, D8         |
| **Full Gating**  | `run_full_tier_gating.py --tier <N>`    | Both tier gate and perf pass; promotion plan = "promote"         | D4, D6, D8         |
| **Model Load**   | Health check endpoint                   | Model loads without error                                        | All tiers          |
| **Monotonicity** | Cross-tier tournament                   | Higher tier wins ≥55% of games vs lower tier                     | All adjacent pairs |

**Tier-specific thresholds** (from [`TIER_EVAL_CONFIGS`](../../ai-service/app/training/tier_eval_config.py:53)):

| Tier | min_win_rate_vs_baseline | max_regression_vs_previous_tier | Perf Budget                        |
| ---- | ------------------------ | ------------------------------- | ---------------------------------- |
| D2   | 60% vs D1 random         | N/A                             | None                               |
| D4   | 70% vs baselines         | 5%                              | max_avg: 2310ms, max_p95: 2625ms   |
| D6   | 75% vs baselines         | 5%                              | max_avg: 5280ms, max_p95: 6000ms   |
| D8   | 80% vs baselines         | 5%                              | max_avg: 10560ms, max_p95: 12000ms |

### Manual Gates (Judgment Required)

These gates require human assessment and cannot be fully automated.

| Gate                            | Procedure                                       | Criteria                                                      | When Required |
| ------------------------------- | ----------------------------------------------- | ------------------------------------------------------------- | ------------- |
| **Perceived Difficulty Review** | Play 3+ games at each affected tier             | Subjective difficulty feels consistent with tier expectations | Category B, C |
| **Cross-Tier Transition**       | Play sequence from D0 to affected tier          | Difficulty progression feels smooth and perceptible           | Category C    |
| **Edge Case Sampling**          | Review 10+ games from validation corpus         | No obvious misplays, stalls, or degenerate behaviors          | Category B, C |
| **UX Review**                   | Check difficulty descriptions match AI behavior | Labels (Easy/Medium/Hard) align with actual difficulty        | Category C    |

**Documentation for manual gates:**

When completing manual gates, record in your change notes:

```markdown
### Manual Gate: Perceived Difficulty Review

- Tester: [name]
- Date: [YYYY-MM-DD]
- Games played: [count per tier]
- Assessment: [description]
- Issues found: [none / description]
```

---

## Rollback Procedures

### Immediate Rollback (< 1 hour)

For changes that cause **obvious breakage** (e.g., crashes, game stalls, extreme win rates).

**Procedure:**

1. **Identify the problematic change**
   - Check recent deployments in deployment log
   - Review recent merges to `ladder_config.py` or model files

2. **Revert configuration**

   ```bash
   # Revert ladder_config.py to previous version
   git checkout <previous_commit> -- ai-service/app/config/ladder_config.py

   # If model files changed, restore from backup
   cp /backup/models/<previous_model>.pth ai-service/models/
   ```

3. **Redeploy AI service**

   ```bash
   # Follow standard deployment procedure for ai-service
   ```

4. **Verify health check passes**
   - Confirm `/health` endpoint returns healthy
   - Confirm tier model loading succeeds
   - Run quick smoke test game if possible

5. **Notify stakeholders**
   - Post to #ringrift-ai-alerts
   - Update incident ticket with rollback status

6. **Document in incident log**
   - Time of issue detection
   - Time of rollback
   - Impact (games affected, players impacted)
   - Root cause (if known)

### Delayed Rollback (< 24 hours)

For changes that cause **drift detected by monitoring** but not immediate breakage.

**Procedure:**

1. **Follow immediate rollback steps 1-6**

2. **Collect diagnostic data before rollback**
   - Export health snapshot: `generate_ladder_health_snapshot.py`
   - Save recent calibration telemetry
   - Archive evaluation logs from the problematic period

3. **Archive problematic model/config for analysis**

   ```bash
   mkdir -p ai-service/archived/incident_$(date +%Y%m%d_%H%M)
   cp ai-service/app/config/ladder_config.py ai-service/archived/incident_*/
   cp ai-service/models/<problematic_model>.pth ai-service/archived/incident_*/
   ```

4. **Schedule retrospective**
   - Within 48 hours for Category B issues
   - Within 24 hours for Category C/D issues

### Staged Rollback

For **critical changes rolled out gradually** that show problems mid-rollout.

**Decision tree:**

1. **Halt staged rollout** at current percentage
   - Freeze feature flag
   - Prevent automatic progression to next phase

2. **Evaluate whether to proceed with full rollback**
   - If impact is severe (>10% affected games broken): proceed to full rollback
   - If impact is contained: consider holding at current phase pending investigation

3. **If proceeding with full rollback**
   - Follow immediate rollback procedure for affected population
   - Disable feature flag entirely
   - Document decision and rationale

4. **Document decision and rationale**
   - Why was rollback chosen vs continuing investigation?
   - What data informed the decision?
   - What follow-up is planned?

---

## Change Governance

### Documentation Requirements

Every ladder change must include the following documentation:

1. **Change Description**
   - What is being changed (files, models, configurations)
   - Why the change is needed (business/player experience rationale)

2. **Category Assignment**
   - Category A/B/C/D
   - Justification for chosen category

3. **Validation Evidence**
   - Gate results (tier eval JSON, perf benchmark JSON)
   - Test outputs (tournament results, calibration summaries)
   - Manual gate documentation (if applicable)

4. **Rollback Plan**
   - Specific previous model/config to revert to
   - Estimated rollback time
   - Who is responsible for executing rollback

5. **Monitoring Plan**
   - What metrics to watch post-deployment
   - Duration of elevated monitoring
   - Alert thresholds that would trigger investigation

### Change Log Location

All significant ladder changes must be logged at:

```
docs/ai/calibration_runs/CHANGE_LOG.md
```

**Entry format:**

```markdown
## [YYYY-MM-DD] [Category] – Brief Summary

**Changed**: What files/models/configs were modified
**Reason**: Why the change was made
**Validation**: Which gates passed / evidence collected
**Outcome**: Success / Issues encountered
**Follow-up**: Any needed actions or scheduled reviews
```

**Example entry:**

```markdown
## [2025-12-06] [B] – Promoted D4 model sq8_2p_d4_v3

**Changed**: Updated ladder_config.py D4 entry to use model_id="sq8_2p_d4_v3"
**Reason**: Previous D4 was too easy for intermediate players (67% human win rate)
**Validation**:

- Tier gate passed (win_rate_vs_baseline=0.73, win_rate_vs_previous_tier=0.58)
- Perf benchmark passed (avg=1842ms, p95=2203ms)
- Monotonicity preserved (D4 wins 61% vs D2)
  **Outcome**: Success – deployed to 100%
  **Follow-up**: Monitor human win rates for 7 days; next calibration cycle in 3 weeks
```

### Approval Matrix

| Category | Approver               | Method                                           |
| -------- | ---------------------- | ------------------------------------------------ |
| A        | Self                   | Standard PR merge workflow                       |
| B        | Self + automated gates | PR with gate CI check passing                    |
| C        | Peer review required   | PR with at least one review approval + all gates |
| D        | Project lead           | Explicit approval comment/message in PR/issue    |

**CI Integration:**

For Category B+ changes, the PR should include:

1. Gate output logs or links to artifacts
2. Promotion plan JSON (for model promotions)
3. Checklist completion via PR template

---

## Integration with Monitoring

### Post-Change Monitoring Period

Based on [`AI_LADDER_HEALTH_MONITORING_SPEC.md`](AI_LADDER_HEALTH_MONITORING_SPEC.md:1):

| Category | Monitoring Period | Escalation Criteria                                 |
| -------- | ----------------- | --------------------------------------------------- |
| A        | None required     | -                                                   |
| B        | 24 hours          | Any warning-level alert fires                       |
| C        | 7 days            | Any warning-level alert, or 2+ informational alerts |
| D        | 14 days           | Any alert (including informational)                 |

### Metrics to Watch

During the monitoring period, actively track:

| Metric                 | Source                | Alert Threshold                    |
| ---------------------- | --------------------- | ---------------------------------- |
| Human win rate by tier | Calibration telemetry | ±10% from target for 24h           |
| Elo drift              | Evaluation runs       | >50 point drift from baseline      |
| Decision latency       | Perf benchmarks       | >1.5× budget avg or >2× budget p95 |
| Game length anomaly    | Game records          | >2 std dev from expected           |
| Abandonment rate       | Calibration telemetry | >1.5× tier threshold for 24h       |

### Health Check Integration

Reference [`AI_LADDER_HEALTH_MONITORING_SPEC.md`](AI_LADDER_HEALTH_MONITORING_SPEC.md:285) for:

- Specific metric collection procedures
- Alert routing and escalation paths
- Response procedures when alerts fire post-change

**Post-change health snapshot:**

After any Category B+ change, capture a baseline health snapshot within 1 hour:

```bash
python -m ai-service.scripts.generate_ladder_health_snapshot \
  --output ai-service/logs/ladder_health/post_change_$(date +%Y%m%d_%H%M).json
```

Compare against pre-change snapshot during monitoring period.

---

## Appendix

### Example Change Workflow

**Scenario**: Promoting a new D4 model `sq8_2p_d4_v3`

1. **Train candidate model** (via tier training pipeline)
   - Produces `training_report.json` in run directory

2. **Determine change category**
   - Model promotion = **Category B**
   - Complete Category B checklist

3. **Run tier gate evaluation**

   ```bash
   python -m ai-service.scripts.run_tier_gate \
     --tier D4 \
     --candidate-model-id sq8_2p_d4_v3 \
     --output-json logs/tier_gate/D4_v3_eval.json \
     --promotion-plan-out logs/tier_gate/D4_v3_promotion.json
   ```

4. **Verify gate passes**
   - Check `overall_pass: true` in output JSON
   - Confirm win rates meet thresholds

5. **Run perf benchmark**

   ```bash
   python -m ai-service.scripts.run_tier_perf_benchmark \
     --tier D4 \
     --output-json logs/tier_perf/D4_v3_perf.json
   ```

6. **Verify perf within budget**
   - Check `within_avg: true` and `within_p95: true`

7. **Create PR with**
   - Updated [`ladder_config.py`](../../ai-service/app/config/ladder_config.py:1)
   - Gate output logs attached or linked
   - Rollback plan: "Revert to model_id='v1-minimax-4'"
   - Checklist completed

8. **Merge after CI passes**

9. **Monitor for 24 hours**
   - Watch for any warning-level alerts
   - Check human win rates in calibration telemetry

10. **Document outcome in change log**

### Templates

#### Change Log Entry Template

```markdown
## [YYYY-MM-DD] [Category] – Brief Summary

**Changed**: What was modified
**Reason**: Why the change was made
**Validation**: Which gates passed / evidence
**Outcome**: Success / Issues encountered
**Follow-up**: Any needed actions
```

#### PR Description Template

```markdown
## AI Ladder Change: [Brief Description]

### Change Category

<!-- Select one: A / B / C / D -->

**Category**:

### Justification

<!-- Why is this the appropriate category? -->

### Validation Evidence

<!-- Attach or link gate outputs -->

- Tier gate result:
- Perf benchmark result:
- Manual gates completed:

### Checklist

<!-- Complete the applicable checklist from AI_LADDER_CHANGE_GUARDRAILS.md -->

- [ ] ...

### Rollback Plan

**Previous model/config to revert to**:
**Estimated rollback time**:
**Responsible party**:

### Monitoring Plan

**Duration**:
**Key metrics**:
**Alert escalation**:
```

#### Rollback Notification Template

```markdown
## 🔄 Ladder Change Rollback

**Original change**: [Link to PR]
**Rollback time**: [YYYY-MM-DD HH:MM UTC]
**Impacted tier(s)**: [D2/D4/D6/D8]

### Reason for rollback

<!-- Brief description of what went wrong -->

### Reverted to

**Model/Config**:
**Commit**:

### Impact

**Duration of issue**:
**Games/players affected**:

### Next steps

<!-- Investigation, retrospective, etc. -->
```

### Related Documents

- [`AI_CALIBRATION_RUNBOOK.md`](AI_CALIBRATION_RUNBOOK.md:1) – Step-by-step calibration cycle procedure
- [`AI_LADDER_HEALTH_MONITORING_SPEC.md`](AI_LADDER_HEALTH_MONITORING_SPEC.md:1) – Ongoing health monitoring and drift detection
- [`AI_TIER_TRAINING_AND_PROMOTION_PIPELINE.md`](AI_TIER_TRAINING_AND_PROMOTION_PIPELINE.md:1) – Training and promotion loop for tier candidates
- [`AI_TIER_PERF_BUDGETS.md`](AI_TIER_PERF_BUDGETS.md:1) – Latency budgets for D4/D6/D8 tiers
- [`AI_HUMAN_CALIBRATION_GUIDE.md`](AI_HUMAN_CALIBRATION_GUIDE.md:1) – Human calibration experiment templates

---

_This specification completes H-AI-16 of the AI difficulty calibration remediation track._
