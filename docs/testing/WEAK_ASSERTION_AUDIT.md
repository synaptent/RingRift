# Weak Assertion Audit

**Created:** 2025-12-11
**Status:** Phase 1 Complete
**Priority:** Medium (per PROJECT_WEAKNESS_ASSESSMENT.md)
**Last Updated:** 2025-12-11

## Overview

This document tracks the audit and remediation of weak assertions in the test suite.
Weak assertions pass without validating actual behavior, reducing test reliability.

## Current Counts

| Assertion Type       | Count     | Risk Level |
| -------------------- | --------- | ---------- |
| `toBeDefined()`      | 827       | Medium     |
| `not.toBeNull()`     | 161       | Medium     |
| `toBeGreaterThan(0)` | 358       | Low        |
| **Total**            | **1,346** | -          |

## Assessment Criteria

Not all uses of these assertions are problematic. Valid uses include:

### Valid Uses (Keep As-Is)

- Guard assertions before property access: `expect(result).toBeDefined(); expect(result.value).toBe(5);`
- Existence checks where existence IS the test: `expect(optionalFeature).toBeDefined();`
- Non-negative counters: `expect(count).toBeGreaterThan(0);` when any positive value is acceptable

### Problematic Uses (Should Fix)

- `expect(result).toBeDefined();` as the only assertion (no follow-up)
- `expect(value).toBeGreaterThan(0);` when a specific value is expected
- `expect(array.length).toBeGreaterThan(0);` when array contents should be validated

## Priority Files for Review

Based on impact, prioritize these test files:

1. **Core Rules Tests** (highest impact)
   - `tests/unit/territoryDecisionHelpers.*.test.ts`
   - `tests/unit/turnOrchestrator.*.test.ts`
   - `tests/unit/ClientSandboxEngine.*.test.ts`

2. **Parity Tests** (critical for correctness)
   - `tests/parity/*.test.ts`

3. **Contract Tests** (API stability)
   - `tests/contracts/*.test.ts`

## Progress Tracking

| File                                                     | Total Weak | Fixed | Remaining | Status                             |
| -------------------------------------------------------- | ---------- | ----- | --------- | ---------------------------------- |
| turnOrchestrator.advanced.branchCoverage.test.ts         | 22         | 10    | 12        | ✅ Audited                         |
| turnOrchestrator.anmRecovery.branchCoverage.test.ts      | 22         | 8     | 14        | ✅ Audited                         |
| turnOrchestrator.phaseTransitions.branchCoverage.test.ts | 21         | 0     | 21        | ✅ Reviewed - mostly guard clauses |

**2025-12-11 Audit Notes:**

- Many `toBeDefined()` uses in turnOrchestrator tests are valid guard clauses followed by specific assertions
- Strengthened 18 weak assertions with phase/status/player validation
- Focus shifted to assertions that are ONLY validation in tests

## Guidelines for Fixing

When replacing weak assertions:

```typescript
// ❌ Weak
expect(result).toBeDefined();
expect(result.length).toBeGreaterThan(0);

// ✅ Strong
expect(result).toEqual({ type: 'move_stack', player: 1, from: { x: 0, y: 0 }, to: { x: 1, y: 0 } });
expect(result).toHaveLength(3);
expect(result[0].type).toBe('place_ring');
```

## Notes

- The goal is 80%+ strong assertions (per PROJECT_WEAKNESS_ASSESSMENT.md)
- Some weak assertions are acceptable as guard clauses
- Focus on assertions that are the ONLY validation in a test
