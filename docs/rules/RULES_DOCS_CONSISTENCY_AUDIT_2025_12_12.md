# Rules Docs Mutual Consistency Audit (2025-12-12)

> **Status:** Completed
>
> **Purpose:** Quick audit that the four primary rules documents stay mutually consistent (and consistent with the canonical spec), with a focused check of adjacent rules/docs that commonly drift.

---

## Scope

Primary rules docs:

- `RULES_CANONICAL_SPEC.md` (normative canonical SSoT)
- `ringrift_complete_rules.md` (player-facing rulebook)
- `ringrift_compact_rules.md` (implementation-oriented spec)
- `ringrift_simple_human_rules.md` (teaching / simplified)

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

---

## Notes (non-issues / historical docs)

- Some supplementary / deprecation docs intentionally reference legacy ring counts (e.g. 36/48) for historical analysis or migration context (e.g. `docs/supplementary/rules_analysis/rules_analysis_ring_count_increase.md`, `ai-service/docs/HEX_ARTIFACTS_DEPRECATED.md`). These are explicitly marked as legacy and are not contradictions in the canonical rules set.
