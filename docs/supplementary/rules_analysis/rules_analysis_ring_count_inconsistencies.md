# Ring Count Inconsistency Analysis

This document details the inconsistencies found in ring counts per player across the codebase, documentation, and test fixtures.

**Status:** Canonical counts are **18 / 60 / 72** (square8 / square19 / hex). Key rulebook sections (Quick Start, §3.2.1, victory thresholds) now reflect these values; re-audit remaining references and fixtures to confirm alignment.

## Canonical Specification (SSoT)

According to `RULES_CANONICAL_SPEC.md` (RR-CANON-R020):

- **square8**: 18 rings
- **square19**: 60 rings
- **hexagonal**: 72 rings

## Findings

### 1. Documentation Inconsistencies

Primary rulebook references have been updated to match **18 / 60 / 72**:

- `RULES_CANONICAL_SPEC.md` (RR-CANON-R020 / R061 examples)
- `ringrift_compact_rules.md` (board config table + threshold examples)
- `ringrift_complete_rules.md` (version tables, setup sections, FAQ defaults)
- `ringrift_simple_human_rules.md` (board overview + threshold list)

### 2. Codebase Consistency (TypeScript)

- **`src/shared/types/game.ts`**: `BOARD_CONFIGS` correctly defines:
  - `square8`: 18
  - `square19`: 60
  - `hexagonal`: 72
- **`src/client/adapters/gameViewModels.ts`**: HUD ring stats read `ringsPerPlayer` from shared `BOARD_CONFIGS` (no local duplication).

### 3. Codebase Consistency (Python)

- **`ai-service/app/rules/core.py`** mirrors TS `BOARD_CONFIGS`:
  - square8: 18
  - square19: 60
  - hexagonal: 72
- **`ai-service/app/game_engine.py`** reads ring caps from `app.rules.core.BOARD_CONFIGS` for TS-aligned semantics.

### 4. Test Fixtures & Analysis Docs

- **Test fixtures:** `tests/utils/fixtures.ts` and the sample config tests in `tests/unit/board.test.ts` have been updated to match canonical counts.
- **Historical analysis docs:** Some older analysis/proposal documents may still reference pre-canonical ring counts; treat those sections as historical unless explicitly updated.

## Remediation Plan

1. ✅ Update canonical rules docs to **18 / 60 / 72**.
2. ✅ Update TS + Python `BOARD_CONFIGS` to **18 / 60 / 72**.
3. ✅ Update tests/fixtures that encoded old ring counts.
