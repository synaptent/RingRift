# Experimental Recovery Stack‑Strike v1

Status: **experimental**. This rule is **not canonical** and is gated behind an
environment flag. Canonical RR‑CANON recovery semantics remain unchanged when
the flag is off.

## Rule Summary

When a player is eligible for recovery in the movement phase (RR‑CANON‑R110),
and **no line‑forming recovery slide exists** for them anywhere on the board
(RR‑CANON‑R112(a) is impossible), then if
`RINGRIFT_RECOVERY_STACK_STRIKE_V1=1` they may choose a **stack‑strike**
recovery slide:

- Slide a marker onto an **adjacent stack**.
- The sliding marker is **removed from play** (not placed at destination).
- The **top ring of the attacked stack** is eliminated and **credited to the
  recovering player**.
- The move still pays the normal recovery fallback cost:
  **one buried‑ring extraction** (RR‑CANON‑R113).

On the wire this is a normal `recovery_slide` move with
`recoveryMode: "stack_strike"`.

## Rationale

Recent large‑board self‑play shows:

- `square19` 2P stalemate rate ≈ 97%.
- `hexagonal` 3P stalemate rate ≈ 86%.
- Recovery eligibility is common on large boards, but line‑forming recovery
  opportunities are comparatively scarce.

Allowing a low‑power stack strike aims to:

1. Convert otherwise “dead” recovery turns into measurable progress.
2. Reduce structural stalemates without making recovery a dominant win path.
3. Preserve LPS as a meaningful endgame option (recovery remains excluded from
   LPS “real action” detection in canonical rules).

## Implementation Notes

- **TS SSoT**: `src/shared/engine/aggregates/RecoveryAggregate.ts` adds
  `RecoveryMode = "stack_strike"` and enumerates/apply/validate under the flag.
- **Python mirror**: `ai-service/app/rules/recovery.py` matches TS semantics and
  emits `recoveryMode="stack_strike"` moves when enabled.
- **Flag gate**:
  - TS reads via `flagEnabled("RINGRIFT_RECOVERY_STACK_STRIKE_V1")`.
  - Python reads via `os.getenv("RINGRIFT_RECOVERY_STACK_STRIKE_V1")`.

Flag off preserves canonical behaviour exactly.

## Evaluation Plan

Run fresh distributed canonical self‑play + TS↔Python parity gates with the
flag on, then compare to baseline:

- `square19` 2P
- `hexagonal` 3P

Primary metrics:

- Stalemate rate (goal: materially lower than baseline).
- LPS win rate (goal: does not collapse into trivial dominance).
- Territory vs elimination victory mix.
- Recovery usage frequency and per‑move net progress.

Driver (local):

`PYTHONPATH=ai-service RINGRIFT_RECOVERY_STACK_STRIKE_V1=1 python ai-service/scripts/generate_canonical_selfplay.py ...`

Use the distributed host config and orchestration described in
`ai-service/docs/AI_TRAINING_PIPELINE_PLAN.md`. Store parity‑gate summaries
alongside generated DBs and only promote to canonical data if all gates pass
and the experiment is explicitly accepted into RR‑CANON.
