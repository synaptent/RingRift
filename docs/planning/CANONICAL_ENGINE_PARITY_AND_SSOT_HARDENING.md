# Canonical Engine Parity & SSoT Hardening

> **Doc Status:** Active worklog (implementation tracking)
> **Created:** 2025-12-11
> **Scope:** TS canonical engine + types (`src/shared/**`), Python mirror (`ai-service/app/**`), and SSoT checks/docs (`scripts/ssot/**`, `docs/**`)
> **Rules SSoT:** `RULES_CANONICAL_SPEC.md`

This document tracks concrete, correctness‑impacting parity and tooling work that:

- Keeps TypeScript (executable engine SSoT) and Python mirrors in lockstep.
- Prevents drift by removing/repairing duplicate “contracts” and stale docs.
- Keeps checks (`npm run ssot-check`, parity gates, history validators) actionable.

It exists to prevent duplicate work across overlapping planning/review docs.

---

## Non‑Negotiable Invariants

- **Canonical rules are normative:** if code disagrees with `RULES_CANONICAL_SPEC.md`, the code is wrong unless the spec is intentionally revised.
- **No silent phase transitions:** phase progression must be attributable to explicit moves (or explicit bookkeeping moves) per RR‑CANON‑R074–R076.
- **No silent forced elimination:** FE must be its own phase (`forced_elimination`) with explicit `forced_elimination` moves.
- **Skip vs no‑op stays distinct:** `skip_*` = voluntary forgo; `no_*_action` = forced no‑op (RR‑CANON‑R075).
- **Parity is a first‑class artifact:** TS ↔ Python must agree on `GamePhase`, `MoveType`, and their contracts.

---

## Work Items (Tracking Table)

Status legend: `TODO` → `IN_PROGRESS` → `DONE` (or `DEFERRED` with rationale).

| ID      | Area                | Summary                                                         | Status |
| ------- | ------------------- | --------------------------------------------------------------- | ------ |
| ENG-01  | TS engine/types     | Add canonical `skip_recovery` support end‑to‑end                | DONE   |
| ENG-02  | Python engine/types | Implement canonical `skip_capture` support end‑to‑end           | TODO   |
| ENG-03  | TS engine API       | Remove/align duplicate `phaseValidation.ts` contract            | DONE   |
| DOC-01  | Docs                | Refresh `CANONICAL_ENGINE_API.md` MoveType/phase surfaces       | DONE   |
| DOC-02  | Docs                | Update `RULES_IMPLEMENTATION_MAPPING.md` rule references        | DONE   |
| TOOL-01 | Tooling             | Fix `npm run ssot-check` doc path drift                         | DONE   |
| TOOL-02 | Tooling             | Make `python-parity-ssot-check` actually verify enums/contracts | DONE   |
| SEC-01  | Security            | Bump `jsonwebtoken` to clear `jws` advisory                     | DONE   |

---

## Baseline (2025-12-11)

**Observed failures**

- `npm run ssot-check` currently fails due to:
  - Missing rule ID references in `docs/rules/RULES_IMPLEMENTATION_MAPPING.md` (e.g., R073–R076, R093, R110–R115, R130, R175–R179, R208–R209).
  - Lifecycle/API doc missing MoveType mentions (e.g., `no_*_action`, `recovery_slide`, `resign`).
  - Several docs moved under subfolders but checks/links still reference legacy paths (e.g., `docs/API_REFERENCE.md`, `docs/ENVIRONMENT_VARIABLES.md`, `docs/SECRETS_MANAGEMENT.md`, `docs/SUPPLY_CHAIN_AND_CI_SECURITY.md`).
- `npm audit --omit=dev` reports **1 high vulnerability**: `jws < 3.2.3` (transitive; patch available).

These are addressed by items `DOC-01`, `DOC-02`, `TOOL-01`, `TOOL-02`, and `SEC-01`.

---

## ENG-01 — TypeScript: `skip_recovery` (canonical movement-phase skip)

**Why it matters**

RR‑CANON‑R115 requires that recovery‑eligible players can explicitly decline recovery with a distinct move type (`skip_recovery`) during `movement`. Without it:

- The TS engine cannot represent a canonical, player‑visible decision that exists in the spec.
- Cross‑language parity (and any DBs containing this move) becomes brittle or impossible.
- History validation becomes inconsistent across layers.

**Spec references**

- `RULES_CANONICAL_SPEC.md` RR‑CANON‑R201/R115 (recovery‑eligible flow + `skip_recovery` in `movement`).

**Expected implementation scope**

- `src/shared/types/game.ts`: add `skip_recovery` to `MoveType` and fix any recovery phase commentary (recovery is in `movement`, not a separate phase).
- `src/shared/engine/orchestration/turnOrchestrator.ts`: enumerate `skip_recovery` when recovery moves exist, and apply it as a no‑op that advances to post‑movement phases per spec.
- `src/shared/engine/fsm/**`: ensure the canonical FSM adapter/state machine accepts the move/event and the allowed‑move mapping reflects the canonical contract.
- Jest tests: add coverage for “recovery eligible chooses to skip” and ensure phase + history semantics are correct.

**Acceptance criteria**

- TS can emit `skip_recovery` as a legal move in `movement` when recovery‑eligible and at least one recovery slide exists.
- Applying `skip_recovery` is canonical, deterministic, and results in the correct phase progression.
- Unit tests cover enumeration + application + FSM validation.

**Status update (2025-12-12)**

- TS support for `skip_recovery` was already present in `turnOrchestrator.ts` and FSM surfaces.
- Python parity bug discovered in canonical self-play: repeated `skip_recovery` moves were possible because phase transitions did not advance after a skip.
- Fixed in `ai-service/app/rules/phase_machine.py` by advancing `skip_recovery → line_processing`, matching TS and RR‑CANON‑R115.
- Canonical parity gate now passes on fresh Square‑8 self-play DBs.

---

## ENG-02 — Python: `skip_capture` (canonical capture-phase skip)

**Why it matters**

The canonical spec (RR‑CANON‑R073/R078 note) defines `skip_capture` as an explicit player choice when optional capture exists. If Python’s engine rejects or can’t apply this move while TS allows it:

- TS↔Python parity on recorded games can fail.
- Canonical history validation becomes inconsistent between TS replays and Python replays.

**Expected implementation scope**

- `ai-service/app/models/core.py`: add `MoveType.SKIP_CAPTURE`.
- `ai-service/app/game_engine/__init__.py` + `ai-service/app/rules/phase_machine.py`: accept and apply `skip_capture` as a no‑board‑change move that transitions `capture → line_processing`.
- Pytest: add a focused test that ensures `skip_capture` is legal/handled when the phase machine is in `capture`.

**Acceptance criteria**

- Python can replay a TS recording that includes `skip_capture`.
- Python phase machine transitions `capture → line_processing` on `skip_capture`.

---

## ENG-03 — TypeScript: eliminate drift from `src/shared/engine/phaseValidation.ts`

**Why it matters**

`phaseValidation.ts` currently encodes a separate “phase ↔ move type” matrix that can (and does) drift from:

- `src/shared/types/game.ts` (canonical TS MoveType)
- The FSM adapter’s canonical contract
- `RULES_CANONICAL_SPEC.md`

This is a long‑term correctness risk: it can mislead maintainers and can be imported by code outside this repo via public exports.

**Expected implementation scope**

- Refactor `phaseValidation.ts` to derive its allowed moves from the canonical FSM adapter/mappings (or remove it and update exports/tests accordingly).
- Update `tests/unit/engine/phaseValidation.test.ts` to assert against the canonical contract, not a duplicated one.

**Acceptance criteria**

- There is exactly one authoritative phase↔move contract surface on the TS side (FSM adapter / orchestration contract), and any “helper” wrappers cannot drift.

---

## DOC-01 — Refresh `docs/architecture/CANONICAL_ENGINE_API.md`

**Why it matters**

`CANONICAL_ENGINE_API.md` is treated as a lifecycle/API SSoT in multiple places; stale MoveType lists or phase/move contracts cause:

- Wrong integrations (especially for replay/parity scripts).
- Confusion when debugging parity.

**Acceptance criteria**

- Move types and phase/move contract documented in `docs/architecture/CANONICAL_ENGINE_API.md` match `src/shared/types/game.ts` and canonical engine behavior.

---

## TOOL-01/02 — SSoT checks: repair path drift + enforce parity contracts

**Why it matters**

SSoT checks are only useful if they:

- Actually run (`npm run ssot-check` must not fail due to stale paths), and
- Actually validate the important drift surfaces (TS enums vs Python enums vs canonical contract).

**Acceptance criteria**

- `npm run ssot-check` passes on a clean working tree.
- The “python parity SSoT” check fails if TS/Python enums or contracts drift.

---

## SEC-01 — Security: bump `jsonwebtoken`

**Why it matters**

Dependency advisories are easy to accumulate and hard to triage later. If an in‑range fix exists (e.g., patch release), bumping early is low‑risk and keeps CI clean.

**Acceptance criteria**

- `npm audit --omit=dev` no longer reports the `jws` high vulnerability.

---

## Progress Log

- **2025-12-11:** Created tracking doc; captured baseline `ssot-check` + `npm audit` failures.
- **2025-12-11:** `SEC-01` DONE — bumped `jsonwebtoken` `9.0.2 → 9.0.3` (transitive `jws` updated; `npm audit --omit=dev` now clean).
- **2025-12-21:** `DOC-01` DONE — refreshed `CANONICAL_ENGINE_API.md` to separate canonical vs legacy move types and align phase/move surfaces.
- **2025-12-21:** `DOC-02` DONE — updated `RULES_IMPLEMENTATION_MAPPING.md` links and top-level engine surface references.
- **2025-12-21:** `TOOL-01` DONE — updated ssot-check doc paths (API/Env/Secrets/CI security) to match current docs layout.
- **2025-12-21:** `TOOL-02` DONE — expanded python parity ssot check to validate TS/Python enum alignment with legacy alias allowances.
