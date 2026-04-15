# AI Work Plan (Unstable Working Tree Assumed)

This plan assumes the working tree may change from other agents. Each lane is
scoped to concrete, testable improvements and executed in order of dependency.

## Lane 1: Parity & Canonical Replay Integrity

Goal: Eliminate TS↔Python semantic drift and harden RR-CANON-R075 invariants.

Scope:

- `ai-service/app/rules/history_contract.py`
- Canonical history validation (`ai-service/app/rules/history_validation.py`)
- Parity harnesses (`ai-service/scripts/check_ts_python_replay_parity.py`)

Immediate steps:

- Add unit tests for `history_contract` (contract completeness, inference,
  forced-elimination/no-action round-tripping, and rejection of legacy moves).
- Document the canonical phase+move contract in a single test to prevent
  accidental regression.

Deliverables:

- New pytest coverage for contract invariants.
- Clear failure modes when canonical move types drift.

## Lane 2: Orchestrator Consolidation & Legacy Removal

Goal: Route all rule execution through the canonical engine and remove
duplicate/legacy paths.

Scope:

- `src/shared/engine/orchestration/**`
- `src/server/game/**` adapters
- `src/client/sandbox/**` (if it diverges from the canonical engine)

Immediate steps:

- Inventory all turn/phase transitions outside `turnOrchestrator.ts`.
- Identify legacy adapters that bypass canonical phase recording.
- Add a thin integration test that ensures server and sandbox use the same
  orchestrator transitions for a representative turn sequence.

Deliverables:

- A consolidation map of non-canonical paths.
- One or more integration tests guarding against drift.

## Lane 3: AI Service Determinism & Boundary Stability

Goal: Reproducible AI behavior and stable API contracts across versions.

Scope:

- `src/server/services/AIServiceClient.ts`
- `ai-service/app/main.py` and `ai-service/app/ai/**`

Immediate steps:

- Ensure seed propagation is present and consistent in service requests.
- Add service boundary tests for AI type mapping (incl. IG_GMO).
- Add explicit error handling for AI fallback scenarios.

Deliverables:

- Deterministic AI behavior across replays.
- Tests that lock down mapping and fallback behavior.

## Lane 4: GPU Pipeline Correctness & MPS/CUDA Edge Cases

Goal: GPU acceleration without semantic drift from CPU rules.

Scope:

- `ai-service/app/ai/gpu_*.py`
- `ai-service/tests/gpu/**`

Immediate steps:

- Add GPU/CPU parity fixtures for movement, capture, and territory phases.
- Stabilize device selection and batch sizing behavior.
- Ensure deterministic outputs for identical seeds and states.

Deliverables:

- Expanded GPU parity test coverage.
- Clear device selection policy with deterministic defaults.

## Lane 5: Training Data Governance & Pipeline Gates

Goal: Canonical-only data for new training and clear provenance.

Scope:

- `ai-service/TRAINING_DATA_REGISTRY.md`
- `ai-service/scripts/generate_canonical_selfplay.py`
- `ai-service/scripts/check_canonical_phase_history.py`

Immediate steps:

- Add checks that reject non-canonical DBs in training scripts by default.
- Emit and store canonical gate summaries alongside DBs.
- Update registry guidance for v2 datasets (source, rules version, parity hash).

Deliverables:

- Canonical gating embedded into training scripts.
- Updated registry metadata for auditability.

## Lane 6: UX/Explanation Consistency

Goal: Align player-facing explanations with canonical rules and FE/ANM logic.

Scope:

- `src/shared/engine/gameEndExplanation.ts`
- `docs/ux/UX_RULES_*`
- `tests/unit/GameEndExplanation.*`

Immediate steps:

- Add tests for forced elimination and active-no-move sequences.
- Ensure explanation payloads match teaching overlays and docs.

Deliverables:

- Stronger UX-rule alignment tests.
- Updated documentation references where needed.

## Lane 7: Observability & Operational Readiness

Goal: Operational clarity for AI reliability and parity regressions.

Scope:

- AI service metrics/logging
- Parity harness summaries and alerts

Immediate steps:

- Add per-AI latency/fallback counters in service logs.
- Standardize parity summary output for CI ingestion.
- Draft a minimal runbook for parity failures.

Deliverables:

- Actionable observability signals.
- Short runbook for parity incidents.
