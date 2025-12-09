# Ring Count Inconsistency Analysis

This document details the inconsistencies found in ring counts per player across the codebase, documentation, and test fixtures.

**Status:** Canonical counts are **18 / 48 / 72** (square8 / square19 / hex). Key rulebook sections (Quick Start, §3.2.1, victory thresholds) now reflect these values; re-audit remaining references and fixtures to confirm alignment.

## Canonical Specification (SSoT)

According to `RULES_CANONICAL_SPEC.md` (RR-CANON-R020):

- **square8**: 18 rings
- **square19**: 48 rings
- **hexagonal**: 72 rings

## Findings

### 1. Documentation Inconsistencies

- **`ringrift_complete_rules.md`**:
  - **Section 1.1**: Correctly lists 18 (8x8), 48 (19x19), 72 (Hex).
  - **Section 1.2.1 (Table)**: Correctly lists 18 (8x8), 48 (19x19), 72 (Hex).
  - **Section 1.3 (Quick Start Guide)**: Incorrectly states "36 for 19x19/Hexagonal".
  - **Section 3.2.1**: Incorrectly states "36 in the 19x19 version, 48 in the Hexagonal version".
  - **Section 6.2**: Incorrectly states "36 for 19x19/Hexagonal".
  - **Section 15.1 (FAQ)**: Correctly states "48 rings per player" for 19x19 default.
  - **Section 16.1 (Table)**: Correctly lists 18 (8x8), 48 (19x19), 72 (Hex).
  - **Section 16.3**: Incorrectly states "Each player has 18 rings... instead of 36".
  - **Section 16.9.2**: Correctly states "Each player has 48 rings".
  - **Section 16.10**: Correctly lists 18 (8x8), 48 (19x19), 72 (Hex).

### 2. Codebase Consistency (TypeScript)

- **`src/shared/types/game.ts`**: `BOARD_CONFIGS` correctly defines:
  - `square8`: 18
  - `square19`: 48
  - `hexagonal`: 72
- **`src/client/adapters/gameViewModels.ts`**: `BOARD_CONFIGS_LOCAL` correctly defines:
  - `square8`: 18
  - `square19`: 48
  - `hexagonal`: 72

### 3. Codebase Consistency (Python)

- **`ai-service/app/game_engine.py`**: `_estimate_rings_per_player` method has **INCORRECT** values:
  - `square8`: 18 (Correct)
  - `square19`: 36 (**INCORRECT** - should be 48)
  - `hexagonal`: 48 (**INCORRECT** - should be 72)

### 4. Test Fixtures & Analysis Docs

- **`rules_analysis_ring_count_increase.md`**: This document discusses the increase, confirming the intent to move to 48/72. It lists the "current" (old) values as 36/48 and proposed as 48/72, and the canonical spec now uses 48/72.
- **`src/server/game/testFixtures/decisionPhaseFixtures.ts`**: Mentions "on square8, each player has 19 rings" in a comment, which contradicts the 18 ring standard.

## Remediation Plan

1.  **Update `ringrift_complete_rules.md`**: Fix all instances of outdated ring counts (36 for 19x19, 48 for Hex) to match the canonical spec (48 for 19x19, 72 for Hex).
2.  **Update `ai-service/app/game_engine.py`**: Correct the `_estimate_rings_per_player` method to return 48 for `square19` and 72 for `hexagonal`.
3.  **Verify `decisionPhaseFixtures.ts`**: Correct the comment about 19 rings on square8 if it's intended to be standard play, or clarify if it's a specific test scenario override.
