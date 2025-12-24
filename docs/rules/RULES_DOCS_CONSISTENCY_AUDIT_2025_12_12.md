# Rules Docs Mutual Consistency Audit (2025-12-12)

> **Status:** Completed
>
> **Purpose:** Quick audit that the four primary rules documents stay mutually consistent (and consistent with the canonical spec), with a focused check of adjacent rules/docs that commonly drift.
>
> **Re-verified:** 2025-12-13 (post recovery + selfplay doc updates). No new inconsistencies found.
> **Updated:** 2025-12-17 (territory processing eligibility contradiction fixed).

---

## Scope

Primary rules docs:

- `RULES_CANONICAL_SPEC.md` (normative canonical SSoT)
- `COMPLETE_RULES.md` (player-facing rulebook)
- `COMPACT_RULES.md` (implementation-oriented spec)
- `HUMAN_RULES.md` (teaching / simplified)

Associated docs sampled (high-risk drift areas):

- `docs/rules/ACTIVE_NO_MOVES_BEHAVIOUR.md`
- `docs/rules/EXPERIMENTAL_RECOVERY_STACK_STRIKE_V1.md`
- `docs/rules/RULES_IMPLEMENTATION_MAPPING.md`
- `docs/rules/RULES_DOCS_CONSISTENCY_PASS_2025_12_11.md`

---

## Findings

### A) Board configs (rings per player) ✅ consistent

All four primary rules docs agree on:

- `square8`: **18** rings/player
- `square19`: **72** rings/player
- `hexagonal`: **96** rings/player

### B) Ring Elimination victory threshold ✅ consistent

All four primary rules docs agree on RR‑CANON‑R061:

- `victoryThreshold = round((2/3) × ringsPerPlayer + (1/3) × opponentsCombinedStartingRings)`
- Simplified: `round(ringsPerPlayer × (2/3 + 1/3 × (numPlayers - 1)))`

And the derived thresholds match:

- `square19` (72): 2p=72, 3p=96, 4p=120
- `hexagonal` (96): 2p=96, 3p=128, 4p=160

### C) Recovery + territory disconnection ✅ consistent

All four primary rules docs agree on RR‑CANON‑R110–R115:

- If any **line-forming** recovery slide exists, recovery must use **line mode** (fallback-class recovery is only permitted when no line-forming recovery slide exists).
- Fallback-class recovery includes (b1) adjacent empty-cell repositioning (including slides that cause territory disconnection) and (b2) stack‑strike when enabled.
- If the recovery slide (line or fallback) creates disconnected territory regions, process them normally, using the **recovery self-elimination rule** (buried ring extraction) per region.

### D) Code/config mirrors ✅ consistent (TS + Python)

Spot-check confirms the executable configs match the docs:

- TS: `src/shared/types/game.ts` (`BOARD_CONFIGS`) uses `square19.ringsPerPlayer=72` and `hexagonal.ringsPerPlayer=96`.
- Python: GPU + rules helpers use the same defaults (e.g. `ai-service/app/ai/gpu_kernels.py` and `ai-service/app/ai/gpu_parallel_games.py`).

### E) Agent-facing quick reference ✅ consistent

- `AGENTS.md` quick-reference tables match the canonical ring supplies and thresholds.

### F) Territory processing stack eligibility ✅ NOW CONSISTENT (Fixed 2025-12-17)

**Issue found:** Multiple documents and code comments incorrectly stated that height-1 standalone rings are NOT eligible for territory processing. This contradicted RR-CANON-R022 and RR-CANON-R145, which state all controlled stacks (including height-1) are eligible.

**Affected locations fixed:**

- `HUMAN_RULES.md` (line 89) - now correctly states all controlled stacks eligible
- `TerritoryAggregate.ts` header comments - updated to match canonical spec
- `EliminationAggregate.ts` - actual implementation already correct (allowing height-1)
- `territoryProcessing.ts` comments - already correct
- `game_engine/__init__.py` and `gpu_parallel_games.py` - already correct
- 25+ additional code comments, test fixtures, and documentation files updated

**Fix applied:** Updated all stale references to align with RR-CANON-R022/R145. All controlled stacks (including height-1 standalone rings) are now consistently documented as eligible for territory processing across all rules docs, code comments, and user-facing content.

**Verification:** `grep -r "NOT eligible" --include="*.md" --include="*.ts" --include="*.py"` returns only historical documentation describing the fix itself.

---

## Notes (non-issues / historical docs)

- Some supplementary / deprecation docs intentionally reference legacy ring counts (e.g. 36/48) for historical analysis or migration context (e.g. `docs/supplementary/rules_analysis/rules_analysis_ring_count_increase.md`, `ai-service/docs/HEX_ARTIFACTS_DEPRECATED.md`). These are explicitly marked as legacy and are not contradictions in the canonical rules set.
