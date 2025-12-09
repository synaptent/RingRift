# FSM Validation Rollout Runbook

> **Doc Status (2025-12-07): Active Runbook**
> **Role:** Step-by-step guide for enabling FSM validation in production

---

## Table of Contents

1. [Overview](#1-overview)
2. [Pre-Deployment Validation](#2-pre-deployment-validation)
3. [Rollout Phases](#3-rollout-phases)
4. [Monitoring](#4-monitoring)
5. [Rollback Procedure](#5-rollback-procedure)
6. [Troubleshooting](#6-troubleshooting)

---

## 1. Overview

The FSM (Finite State Machine) validation layer provides phase/type correctness validation for all game moves. It ensures moves are only allowed in their proper game phases.

### Validation Modes

| Mode     | Behavior                                    | Risk                      |
| -------- | ------------------------------------------- | ------------------------- |
| `off`    | No FSM validation (legacy)                  | None                      |
| `shadow` | FSM runs in parallel, logs divergences      | None (observability only) |
| `active` | FSM is authoritative, rejects invalid moves | Medium (may reject moves) |

### Key Files

- Implementation: `src/shared/engine/fsm/`
- Adapter: `src/shared/engine/fsm/FSMAdapter.ts`
- Orchestrator integration: `src/shared/engine/orchestration/turnOrchestrator.ts`
- Validation script: `scripts/validate-fsm-active-mode.ts`
- Environment flags: `src/shared/utils/envFlags.ts`

---

## 2. Pre-Deployment Validation

### Step 1: Run Validation Script

Before any production rollout, validate FSM against recorded games:

```bash
# Validate against all available game databases
TS_NODE_PROJECT=tsconfig.server.json npx ts-node -T scripts/validate-fsm-active-mode.ts --mode active

# Validate specific database with verbose output
TS_NODE_PROJECT=tsconfig.server.json npx ts-node -T scripts/validate-fsm-active-mode.ts \
  --db ai-service/data/games/selfplay.db \
  --limit 50 \
  --mode active \
  --verbose
```

**Expected Output:**

```
✅ VALIDATION PASSED - Safe to enable active mode
```

### Step 2: Run FSM Unit Tests

```bash
NODE_ENV=test npx jest tests/unit/fsm/FSMAdapter.test.ts
```

**Expected:** All 25+ tests pass

### Step 3: Run Full Test Suite

```bash
npm test
```

**Expected:** All tests pass with FSM enabled

---

## 3. Rollout Phases

### Phase 1: Shadow Mode (Monitor Only)

**Duration:** 1-2 days minimum

```bash
# Environment configuration
RINGRIFT_FSM_VALIDATION_MODE=shadow
RINGRIFT_FSM_STRUCTURED_LOGGING=1
```

**What to Monitor:**

- Structured logs for divergences (`"divergence": true`)
- Error rates in application logs
- Game completion rates

**Success Criteria:**

- [ ] Zero divergences in shadow logs
- [ ] No increase in error rates
- [ ] Games completing normally

### Phase 2: Active Mode (Enforcement)

**Prerequisites:**

- Phase 1 completed with 0 divergences
- Validation script passes: `--mode active`

```bash
# Environment configuration
RINGRIFT_FSM_VALIDATION_MODE=active
RINGRIFT_FSM_STRUCTURED_LOGGING=1
```

**What to Monitor:**

- FSM rejection errors in logs
- Player-reported issues
- Game abandonment rates

### Phase 3: Cleanup (Optional)

After stable operation in active mode:

- Consider removing legacy validation code
- Remove `RINGRIFT_FSM_SHADOW_VALIDATION` flag support
- Update documentation

---

## 4. Monitoring

### Structured Log Queries

**Find all FSM validation events:**

```bash
grep '"event":"fsm_validation"' /var/log/ringrift/*.log | jq .
```

**Find divergences only:**

```bash
grep '"divergence":true' /var/log/ringrift/*.log | jq .
```

**Count validations by mode:**

```bash
grep '"event":"fsm_validation"' /var/log/ringrift/*.log | jq -r '.mode' | sort | uniq -c
```

### Key Metrics to Watch

| Metric               | Normal | Warning | Critical |
| -------------------- | ------ | ------- | -------- |
| FSM divergence rate  | 0%     | >0.1%   | >1%      |
| Move rejection rate  | 0%     | >0.01%  | >0.1%    |
| Game completion rate | >95%   | <95%    | <90%     |

### Alert Configuration

Add to monitoring alerts:

```yaml
# FSM Divergence Alert
- alert: FSMDivergenceDetected
  expr: rate(fsm_validation_divergences_total[5m]) > 0
  for: 1m
  labels:
    severity: warning
  annotations:
    summary: 'FSM validation divergences detected'
    description: 'Shadow mode detected FSM/legacy disagreement'

# FSM Rejection Alert (Active Mode)
- alert: FSMMoveRejectionHigh
  expr: rate(fsm_validation_rejections_total[5m]) > 0.001
  for: 5m
  labels:
    severity: critical
  annotations:
    summary: 'FSM rejecting player moves'
    description: 'Active mode is rejecting moves at elevated rate'
```

---

## 5. Rollback Procedure

### Immediate Rollback

If issues are detected in active mode:

```bash
# Option 1: Disable FSM completely
RINGRIFT_FSM_VALIDATION_MODE=off

# Option 2: Revert to shadow mode (safer for debugging)
RINGRIFT_FSM_VALIDATION_MODE=shadow
```

**Steps:**

1. Update environment variable
2. Restart affected services
3. Verify games are completing normally
4. Investigate root cause

### Rollback Checklist

- [ ] Set `RINGRIFT_FSM_VALIDATION_MODE=off` or `shadow`
- [ ] Restart services
- [ ] Verify no more FSM rejections
- [ ] Check game completion rates returning to normal
- [ ] Capture logs for post-mortem

---

## 6. Troubleshooting

### Common Issues

#### Issue: Shadow Mode Shows Divergences

**Symptom:** Logs show `"divergence": true`

**Investigation:**

1. Check the specific move type causing divergence
2. Compare FSM phase with expected phase
3. Review the guard conditions in FSM definition

**Common Causes:**

- FSM guard too strict/lenient
- Phase transition mismatch
- Turn tracking discrepancy

#### Issue: Active Mode Rejecting Valid Moves

**Symptom:** Players cannot make moves that should be valid

**Immediate Action:**

1. Rollback to shadow mode
2. Capture the game state and move that was rejected
3. Analyze FSM logs for rejection reason

**Investigation:**

```bash
# Find rejection details
grep '"fsmValid":false' /var/log/ringrift/*.log | jq '{gameId, moveType, currentPhase, errorCode, reason}'
```

#### Issue: Performance Degradation

**Symptom:** Increased latency after enabling FSM

**Investigation:**

1. Check `durationMs` in structured logs
2. Compare latency distributions with FSM off/on
3. Profile FSM validation code

**Mitigation:**

- FSM validation is typically <1ms
- If >5ms consistently, investigate guard function efficiency

---

## Quick Reference

### Environment Variables

| Variable                          | Values                    | Description                         |
| --------------------------------- | ------------------------- | ----------------------------------- |
| `RINGRIFT_FSM_VALIDATION_MODE`    | `off`, `shadow`, `active` | FSM validation mode                 |
| `RINGRIFT_FSM_STRUCTURED_LOGGING` | `1`, `true`               | Enable JSON event logging           |
| `RINGRIFT_FSM_SHADOW_VALIDATION`  | `1`                       | Legacy: equivalent to `shadow` mode |

### Validation Script Usage

```bash
# Full validation (default databases, both modes)
TS_NODE_PROJECT=tsconfig.server.json npx ts-node -T scripts/validate-fsm-active-mode.ts

# Active mode only, specific database
TS_NODE_PROJECT=tsconfig.server.json npx ts-node -T scripts/validate-fsm-active-mode.ts \
  --db path/to/games.db \
  --mode active \
  --limit 100 \
  --verbose

# Fail fast on first error
TS_NODE_PROJECT=tsconfig.server.json npx ts-node -T scripts/validate-fsm-active-mode.ts \
  --fail-fast
```

### Related Documentation

- [Environment Variables](../operations/ENVIRONMENT_VARIABLES.md#fsm-validation-mode)
- [Production Deployment Runbook](./PRODUCTION_DEPLOYMENT_RUNBOOK.md)
- [Orchestrator Rollout Feature Flags](../archive/ORCHESTRATOR_ROLLOUT_FEATURE_FLAGS.md) (historical reference)

---

**Document Maintainer:** Claude Code
**Last Updated:** December 7, 2025
**Validation Status:** Tested with 686 moves across 9 games, 0 divergences
