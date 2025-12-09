# Ring Count Remediation Plan

This plan outlines the steps to fix the ring count inconsistencies identified in `rules_analysis_ring_count_inconsistencies.md`.

**Status:** Canonical docs now use **18 / 48 / 72** ring counts; verify whether any code paths or fixtures still assume older values before proceeding with remaining steps.

## Objective

Ensure all parts of the codebase and documentation align with the canonical specification (RR-CANON-R020):

- **square8**: 18 rings
- **square19**: 48 rings
- **hexagonal**: 72 rings

## Steps

### 1. Update Documentation (`ringrift_complete_rules.md`)

- **Section 1.3 (Quick Start Guide)**: Change "36 for 19x19/Hexagonal" to "48 for 19x19, 72 for Hexagonal".
- **Section 3.2.1**: Change "36 in the 19x19 version, 48 in the Hexagonal version" to "48 in the 19x19 version, 72 in the Hexagonal version".
- **Section 6.2**: Change "36 for 19x19/Hexagonal" to "48 for 19x19, 72 for Hexagonal".
- **Section 16.3**: Change "instead of 36" to "instead of 48".

### 2. Update Python AI Service (`ai-service/app/game_engine.py`)

- **Method `_estimate_rings_per_player`**:
  - Update `SQUARE19` return value from `36` to `48`.
  - Update `HEXAGONAL` return value from `48` to `72`.

### 3. Verify Test Fixtures (`src/server/game/testFixtures/decisionPhaseFixtures.ts`)

- **Comment Correction**: Update the comment "on square8, each player has 19 rings" to "on square8, each player has 18 rings" (or clarify if 19 is a specific test override, though 18 is standard). Given the context of "near-victory state where Player 1 is one capture away from winning by ring elimination", and victory threshold is >18 (so 19), having 18 eliminated means 1 more to win. The comment likely meant "victory threshold is 19". I will clarify this comment to align with the 18-ring standard.

## Verification

After applying these changes, I will:

1.  Re-read the modified files to ensure correctness.
2.  Run relevant tests (if possible/applicable in this mode) or verify the logic manually.
