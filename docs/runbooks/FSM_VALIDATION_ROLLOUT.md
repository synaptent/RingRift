# FSM Validation Runbook

> **Doc Status (2025-12-09): Active Runbook - FSM Canonical**
> **Role:** Operational guide for FSM validation (canonical game state orchestrator)

---

> **RR-CANON Compliance Note:** FSM is now the canonical game state orchestrator.
> Shadow mode has been fully removed. FSM validation is enabled by default (`active` mode).
> See `CANONICAL_ENGINE_API.md` for the authoritative API specification.

## Table of Contents

1. [Overview](#1-overview)
2. [Pre-Deployment Validation](#2-pre-deployment-validation)
3. [Monitoring](#3-monitoring)
4. [Rollback Procedure](#4-rollback-procedure)
5. [Troubleshooting](#5-troubleshooting)

---

## 1. Overview

The FSM (Finite State Machine) validation layer provides phase/type correctness validation for all game moves. It ensures moves are only allowed in their proper game phases. **FSM is now the canonical game state orchestrator** (RR-CANON compliance).

### Validation Modes

| Mode     | Behavior                                    | Status                  |
| -------- | ------------------------------------------- | ----------------------- |
| `off`    | No FSM validation (legacy)                  | Not recommended         |
| `active` | FSM is authoritative, rejects invalid moves | **Default (canonical)** |

> **Note:** The `shadow` mode was removed when FSM became canonical. FSM is now the
> single source of truth for game state validation.

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

## 3. Monitoring

### Structured Log Queries

**Find all FSM validation events:**

```bash
grep '"event":"fsm_validation"' /var/log/ringrift/*.log | jq .
```

**Find FSM rejections:**

```bash
grep '"fsmValid":false' /var/log/ringrift/*.log | jq .
```

**Count validations:**

```bash
grep '"event":"fsm_validation"' /var/log/ringrift/*.log | wc -l
```

### Key Metrics to Watch

| Metric                 | Normal | Warning | Critical |
| ---------------------- | ------ | ------- | -------- |
| Move rejection rate    | 0%     | >0.01%  | >0.1%    |
| Game completion rate   | >95%   | <95%    | <90%     |
| FSM validation latency | <1ms   | >5ms    | >10ms    |

### Alert Configuration

Add to monitoring alerts:

```yaml
# FSM Rejection Alert
- alert: FSMMoveRejectionHigh
  expr: rate(fsm_validation_rejections_total[5m]) > 0.001
  for: 5m
  labels:
    severity: critical
  annotations:
    summary: 'FSM rejecting player moves'
    description: 'FSM validation is rejecting moves at elevated rate'
```

---

## 4. Rollback Procedure

### Emergency Rollback

If FSM is causing issues, disable it:

```bash
# Disable FSM validation completely (emergency only)
RINGRIFT_FSM_VALIDATION_MODE=off
```

> **Warning:** Disabling FSM removes canonical game state validation.
> Only use in emergencies and re-enable after investigation.

**Steps:**

1. Update environment variable
2. Restart affected services
3. Verify games are completing normally
4. Investigate root cause
5. **Re-enable FSM as soon as possible**

### Rollback Checklist

- [ ] Set `RINGRIFT_FSM_VALIDATION_MODE=off`
- [ ] Restart services
- [ ] Verify no more FSM rejections
- [ ] Check game completion rates returning to normal
- [ ] Capture logs for post-mortem
- [ ] Create incident ticket for investigation
- [ ] Re-enable `RINGRIFT_FSM_VALIDATION_MODE=active` after fix

---

## 5. Troubleshooting

### Common Issues

#### Issue: Active Mode Rejecting Valid Moves

**Symptom:** Players cannot make moves that should be valid

**Immediate Action:**

1. Capture the game state and move that was rejected
2. Analyze FSM logs for rejection reason
3. If widespread, consider emergency rollback to `off` mode

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

| Variable                          | Values          | Description                           |
| --------------------------------- | --------------- | ------------------------------------- |
| `RINGRIFT_FSM_VALIDATION_MODE`    | `off`, `active` | FSM validation mode (default: active) |
| `RINGRIFT_FSM_STRUCTURED_LOGGING` | `1`, `true`     | Enable JSON event logging             |

> **Note:** `RINGRIFT_FSM_SHADOW_VALIDATION` has been removed. FSM is now canonical.

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
- [Production Deployment Runbook](PRODUCTION_DEPLOYMENT_RUNBOOK.md)
- [Orchestrator Rollout Feature Flags](../archive/ORCHESTRATOR_ROLLOUT_FEATURE_FLAGS.md) (historical reference)

---

**Document Maintainer:** Claude Code
**Last Updated:** December 9, 2025
**FSM Status:** Canonical game state orchestrator (RR-CANON compliant)
