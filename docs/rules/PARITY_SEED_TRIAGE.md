# Parity Seed Triage Matrix

> **Doc Status (2025-11-27): Active (test meta-doc, non-semantics)**
>
> - Tracks **Backend ↔ Sandbox semantic trace parity divergences** for selected seeded AI games and documents which trace-based suites remain diagnostic vs gating.
> - Treats **shared TS orchestrator + contracts + contract vectors** as the rules SSoT:
>   - Orchestrator + aggregates: `src/shared/engine/orchestration/turnOrchestrator.ts`, `src/shared/engine/aggregates/*.ts`.
>   - Contracts & vectors: `src/shared/engine/contracts/*`, `tests/fixtures/contract-vectors/v2/*.json`.
>   - TS runner: `tests/contracts/contractVectorRunner.test.ts`.
>   - Python runner & parity: `ai-service/tests/contracts/test_contract_vectors.py`, `ai-service/tests/parity/*.py`.
> - Seeded `Backend_vs_Sandbox.*` and related parity suites referenced here are **host/adapter diagnostics** over GameEngine / ClientSandboxEngine / BoardManager and Python hosts; they are not a separate rules SSoT and must yield to the shared TS engine + contract vectors + rules docs when they disagree.
> - For rules semantics and lifecycle SSoT, see:
>   - [`RULES_CANONICAL_SPEC.md`](../RULES_CANONICAL_SPEC.md) (RR-CANON-RXXX rules).
>   - [`ringrift_complete_rules.md`](../ringrift_complete_rules.md) and the scenario index in [`RULES_SCENARIO_MATRIX.md`](../RULES_SCENARIO_MATRIX.md).
>   - [`docs/CANONICAL_ENGINE_API.md` §3.9–3.10](./CANONICAL_ENGINE_API.md) for Move / PendingDecision / PlayerChoice / WebSocket.
> - For broader test taxonomy and CI profiles, see `tests/README.md`, `tests/TEST_LAYERS.md`, and `tests/TEST_SUITE_PARITY_PLAN.md`.
> - Last triage update: 2025-11-27; status and seed set should be periodically revalidated as parity work and contract-vector coverage progress.
>
> **Last Updated:** November 27, 2025 \
> **Severity:** P0 – Critical for engine correctness \
> **Related:** [KNOWN_ISSUES.md](../KNOWN_ISSUES.md) (P0.2), [TRACE_PARITY_CONTINUATION_TASK.md](../archive/TRACE_PARITY_CONTINUATION_TASK.md)

---

## 1. Executive Summary

### 1.1 Scope

This document tracks all known **Backend↔Sandbox semantic trace parity divergences**. These are cases where the GameEngine (backend) and ClientSandboxEngine (sandbox) produce different behavior when replaying the same sequence of moves from identical initial states.

### 1.2 Total Seeds with Known Divergences

| Priority | Seed | Board Type | Status | First Divergence | Severity |
| -------- | ---- | ---------- | ------ | ---------------- | -------- |
| **1**    | 5    | square8    | Open   | Move ~12         | Critical |
| **2**    | 17   | square8    | Open   | Move 16          | Critical |
| **3**    | 14   | square8    | Open   | Move ~31         | High     |

> **Note on plateau snapshots (seed‑1 / seed‑18)**
>
> Historical plateau snapshots used by the Python ai‑service parity harnesses
> are exported by archived Jest utilities under:
>
> - `archive/tests/unit/ExportSeed1Snapshot.test.ts`
> - `archive/tests/unit/ExportSeed18Snapshot.test.ts`
>
> These suites are **not** part of normal CI; they are opt‑in diagnostics for
> regenerating:
>
> - `ai-service/tests/parity/square8_2p_seed1_plateau.snapshot.json`
> - `ai-service/tests/parity/square8_2p_seed18_plateau.snapshot.json`
>
> To (re)export a plateau snapshot locally, run for example:
>
> ```bash
> RINGRIFT_EXPORT_PARITY_SNAPSHOTS=1 \
> npx jest archive/tests/unit/ExportSeed1Snapshot.test.ts --runInBand
> ```

### 1.3 Impact Assessment

- **Debugging blockers**: Trace-based debugging is unreliable because sandbox traces may not replay correctly on backend
- **Test confidence**: 10+ parity test files are currently skipped
- **Rule correctness**: Some divergences may indicate genuine rule bugs
- **AI training**: AI trained on sandbox traces may learn behaviors backend rejects

### 1.4 Priority Ranking

1. **Seed 5** – Most investigated, earliest divergence with capture enumeration
2. **Seed 17** – Clear capture/chain-capture divergences at known move numbers
3. **Seed 14** – Placement validation and line processing divergences

### 1.5 Orchestrator + contract-vector backbone vs seed traces

Since the Phase 1–4 rules remediation, the **primary TS↔Python rules parity guarantees** are provided by:

- **Canonical TS orchestrator + aggregates:** `src/shared/engine/orchestration/turnOrchestrator.ts` and `src/shared/engine/aggregates/*.ts`.
- **Cross-language contracts:** `src/shared/engine/contracts/*` and v2 contract vectors under `tests/fixtures/contract-vectors/v2/*.json`.
- **TS contract runner:** `tests/contracts/contractVectorRunner.test.ts`.
- **Python contract runner + parity suites:** `ai-service/tests/contracts/test_contract_vectors.py` and scenario-oriented suites under `ai-service/tests/parity/`.

In that topology:

- **Seeded Backend_vs_Sandbox.\* suites** act as **host/adapter and instrumentation checks** (GameEngine/RuleEngine/BoardManager ↔ ClientSandboxEngine), not as the source of truth for Move legality.
- **`TraceFixtures.sharedEngineParity.test.ts`** exercises orchestrator-level traces (using shared-engine fixtures and v1 traces under `tests/fixtures/rules-parity/v1/*.json`) and is the preferred place to assert orchestrator semantics over long games.
- **Python parity tests** use the same contract vectors and selected traces to ensure the Python `GameEngine` mirrors the shared orchestrator’s behavior.

Use this document when deciding **which seeds/traces are worth preserving** as host/adapter regression tests, and when deciding whether to:

- Fix a discrepancy by changing **GameEngine / ClientSandboxEngine / BoardManager** behaviour;
- Update **trace fixtures / test harnesses** because the orchestrator+contracts semantics have evolved; or
- Defer/retire a divergence as historical once contract-vector and orchestrator suites agree.

---

## 2. Divergence Matrix

| Seed | Board Type | First Diverging Move | Divergence Type | Root Cause Hypothesis                                  | Test File                                                                                                                      | Status       |
| ---- | ---------- | -------------------- | --------------- | ------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------ | ------------ |
| 5    | square8    | Move 12              | capture         | Backend missing overtaking_capture enumeration         | [`TraceParity.seed5.firstDivergence.test.ts`](../tests/unit/TraceParity.seed5.firstDivergence.test.ts)                         | **RESOLVED** |
| 5    | square8    | Move 43              | territory       | Territory region detection/gating differences          | [`Backend_vs_Sandbox.seed5.checkpoints.test.ts`](../tests/unit/Backend_vs_Sandbox.seed5.checkpoints.test.ts)                   | **RESOLVED** |
| 5    | square8    | Move 45              | territory       | Territory decision phase orchestration                 | [`TerritoryDecision.seed5Move45.parity.test.ts`](../tests/unit/TerritoryDecision.seed5Move45.parity.test.ts)                   | **RESOLVED** |
| 5    | square8    | Move 64              | phase           | Late-game move match failure at end of game            | [`Backend_vs_Sandbox.seed5.bisectParity.test.ts`](../tests/unit/Backend_vs_Sandbox.seed5.bisectParity.test.ts)                 | Open         |
| 14   | square8    | Move ~31             | placement       | Multi-ring placement validation (`placementCount > 1`) | [`TraceParity.seed14.firstDivergence.test.ts`](../tests/unit/TraceParity.seed14.firstDivergence.test.ts)                       | Open         |
| 14   | square8    | Move 33              | line            | Line detection/processing differences                  | [`ParityDebug.seed14.trace.test.ts`](../tests/unit/ParityDebug.seed14.trace.test.ts)                                           | Open         |
| 17   | square8    | Move 16              | capture         | Missing `overtaking_capture` (c1×d2→f4)                | [`Seed17Move16And33Parity.GameEngine_vs_Sandbox.test.ts`](../tests/unit/Seed17Move16And33Parity.GameEngine_vs_Sandbox.test.ts) | Open         |
| 17   | square8    | Move 33              | chain_capture   | Chain capture phase out-of-sync (h4×f4→c4)             | [`Seed17Move16And33Parity.GameEngine_vs_Sandbox.test.ts`](../tests/unit/Seed17Move16And33Parity.GameEngine_vs_Sandbox.test.ts) | Open         |
| 17   | square8    | Move 52              | phase           | Phase/player tracking divergence                       | [`TraceParity.seed17.firstDivergence.test.ts`](../tests/unit/TraceParity.seed17.firstDivergence.test.ts)                       | Open         |

---

## 3. Per-Seed Analysis

### 3.1 Seed 5

**Board type:** square8  
**Players:** 2 AI  
**Game length at divergence:** ~12 moves (earliest), game continues to ~60 moves

#### Observed Divergence Points

| Move # | Type                       | Description                                             | Status                                            |
| ------ | -------------------------- | ------------------------------------------------------- | ------------------------------------------------- |
| 12     | `overtaking_capture`       | Sandbox emits capture, backend only offers simple moves | **RESOLVED** - Unified via shared captureLogic.ts |
| 43     | `territory`                | Checkpoint snapshot mismatch                            | **RESOLVED** - Tests pass                         |
| 45     | `process_territory_region` | Territory decision phase differences                    | **RESOLVED** - Tests pass                         |
| 64     | `phase`                    | Late-game move match failure at end of game             | Open                                              |

#### Resolution: Move 12 Capture Enumeration

**Status: RESOLVED** (November 25, 2025)

Both Backend and Sandbox now use the unified `enumerateCaptureMoves()` function from [`captureLogic.ts`](../src/shared/engine/captureLogic.ts):

- **Backend (RuleEngine.ts)**: `getValidCaptures()` delegates to shared `enumerateCaptureMoves()`
- **Sandbox (sandboxCaptures.ts)**: `enumerateCaptureSegmentsFromBoard()` delegates to shared `enumerateCaptureMovesShared()`

The first actual divergence for Seed 5 now occurs at move 64 (end of game), which is a late-game phase/player tracking issue.

#### Related Skipped Tests

- [`Backend_vs_Sandbox.seed5.bisectParity.test.ts`](../tests/unit/Backend_vs_Sandbox.seed5.bisectParity.test.ts) – `TODO-BISECT-PARITY`
- [`Backend_vs_Sandbox.seed5.checkpoints.test.ts`](../tests/unit/Backend_vs_Sandbox.seed5.checkpoints.test.ts)
- [`Backend_vs_Sandbox.seed5.internalStateParity.test.ts`](../tests/unit/Backend_vs_Sandbox.seed5.internalStateParity.test.ts)
- [`Backend_vs_Sandbox.seed5.move1Checkpoints.test.ts`](../tests/unit/Backend_vs_Sandbox.seed5.move1Checkpoints.test.ts)
- [`Backend_vs_Sandbox.seed5.prefixDiagnostics.test.ts`](../tests/unit/Backend_vs_Sandbox.seed5.prefixDiagnostics.test.ts)
- [`TerritoryDecision.seed5Move45.parity.test.ts`](../tests/unit/TerritoryDecision.seed5Move45.parity.test.ts) – `.skip`
- [`TerritoryDetection.seed5Move45.parity.test.ts`](../tests/unit/TerritoryDetection.seed5Move45.parity.test.ts)

#### Suspected Code Paths

- [`src/server/game/RuleEngine.ts`](../src/server/game/RuleEngine.ts) – capture enumeration
- [`src/client/sandbox/sandboxCaptures.ts`](../src/client/sandbox/sandboxCaptures.ts) – sandbox capture enumeration
- [`src/shared/engine/captureLogic.ts`](../src/shared/engine/captureLogic.ts) – shared capture validation
- [`src/shared/engine/territoryDecisionHelpers.ts`](../src/shared/engine/territoryDecisionHelpers.ts) – territory decisions

---

### 3.2 Seed 14

**Board type:** square8  
**Players:** 2 AI  
**Game length at divergence:** ~31 moves

#### Observed Divergence Points

| Move # | Type         | Description                                                     |
| ------ | ------------ | --------------------------------------------------------------- |
| ~31    | `place_ring` | Multi-ring placement (`placementCount > 1`) rejected by backend |
| 33     | `line/hash`  | State hash divergence at line processing                        |
| 35     | `line`       | Line detection parity test failure                              |

#### Backend Behavior at Move 31

- Backend rejects placement as "dead placement" (no legal moves from resulting stack)
- [`hasAnyLegalMoveOrCaptureFrom()`](../src/server/game/RuleEngine.ts) returns false
- Suggests hypothetical board evaluation differs

#### Sandbox Behavior at Move 31

- Sandbox AI attempts `place_ring` with `placementCount > 1`
- Sandbox considers this placement legal
- Indicates different evaluation of post-placement legality

#### Related Skipped Tests

- [`TraceParity.seed14.firstDivergence.test.ts`](../tests/unit/TraceParity.seed14.firstDivergence.test.ts) – `.skip`, `TODO-SEED14-DIVERGENCE`
- [`ParityDebug.seed14.trace.test.ts`](../tests/unit/ParityDebug.seed14.trace.test.ts)
- [`Seed14Move35LineParity.test.ts`](../tests/unit/Seed14Move35LineParity.test.ts)

#### Suspected Code Paths

- [`src/server/game/RuleEngine.ts`](../src/server/game/RuleEngine.ts) – `hasAnyLegalMoveOrCaptureFrom()`
- [`src/shared/engine/placementHelpers.ts`](../src/shared/engine/placementHelpers.ts) – placement validation
- [`src/shared/engine/lineDecisionHelpers.ts`](../src/shared/engine/lineDecisionHelpers.ts) – line processing

---

### 3.3 Seed 17

**Board type:** square8  
**Players:** 2 AI  
**Game length at divergence:** 16 moves (earliest)

#### Observed Divergence Points

| Move # | Type                       | Description                                                                              |
| ------ | -------------------------- | ---------------------------------------------------------------------------------------- |
| 16     | `overtaking_capture`       | Sandbox: c1×d2→f4 capture; Backend: only non-capture moves from c1                       |
| 33     | `continue_capture_segment` | Sandbox: chain_capture for P2 (h4×f4→c4); Backend: already advanced to P1 ring_placement |
| 52     | `phase`                    | Phase/player skipping logic differs                                                      |

#### Backend Behavior at Move 16

- Backend is in movement phase for player 2
- From position c1, backend only enumerates simple moves
- No `overtaking_capture` targeting d2 is offered
- Attacker and target stack geometry appears correct

#### Sandbox Behavior at Move 16

- Sandbox in identical geometric state
- Sandbox AI selects `overtaking_capture` from c1 over d2 to f4
- Move type explicitly `overtaking_capture` with `captureTarget` set

#### Backend Behavior at Move 33

- Backend has already advanced to player 1's `ring_placement` phase
- No `continue_capture_segment` moves are valid
- Suggests chain capture phase was exited prematurely

#### Sandbox Behavior at Move 33

- Sandbox is in `chain_capture` phase for player 2
- Sandbox emits `continue_capture_segment` h4×f4→c4
- Sandbox believes chain capture is still active

#### Related Skipped Tests

- [`TraceParity.seed17.firstDivergence.test.ts`](../tests/unit/TraceParity.seed17.firstDivergence.test.ts) – `.skip`, `TODO-SEED17-STRICT-DIVERGENCE`
- [`Seed17Move16And33Parity.GameEngine_vs_Sandbox.test.ts`](../tests/unit/Seed17Move16And33Parity.GameEngine_vs_Sandbox.test.ts) – `.skip`, `TODO-SEED17-CAPTURE-PARITY`
- [`Seed17Move52Parity.GameEngine_vs_Sandbox.test.ts`](../tests/unit/Seed17Move52Parity.GameEngine_vs_Sandbox.test.ts)
- [`Sandbox_vs_Backend.seed17.traceDebug.test.ts`](../tests/unit/Sandbox_vs_Backend.seed17.traceDebug.test.ts)
- [`SInvariant.seed17FinalBoard.test.ts`](../tests/unit/SInvariant.seed17FinalBoard.test.ts)

#### Suspected Code Paths

- [`src/server/game/RuleEngine.ts`](../src/server/game/RuleEngine.ts) – capture enumeration, chain capture detection
- [`src/shared/engine/captureChainHelpers.ts`](../src/shared/engine/captureChainHelpers.ts) – chain capture logic
- [`src/server/game/GameEngine.ts`](../src/server/game/GameEngine.ts) – phase transitions after captures
- [`src/shared/engine/turnLogic.ts`](../src/shared/engine/turnLogic.ts) – turn advancement

---

## 4. Divergence Categories

### 4.1 Capture Enumeration

**Description:** Backend and sandbox enumerate different legal captures from the same board position.

**Symptoms:**

- Sandbox emits `overtaking_capture` that backend doesn't have
- Move match fails because capture type mismatches

**Root Cause Candidates:**

- [`RuleEngine.getValidMoves()`](../src/server/game/RuleEngine.ts) capture enumeration logic
- Cap height validation differences
- Path blocking calculation differences

**Affected Seeds:** 5, 17

---

### 4.2 Chain Capture Phase Transitions

**Description:** Backend exits chain capture phase at different times than sandbox.

**Symptoms:**

- Sandbox in `chain_capture` phase, backend in `ring_placement`
- `continue_capture_segment` move not available on backend

**Root Cause Candidates:**

- [`GameEngine.advanceGame()`](../src/server/game/GameEngine.ts) phase advancement
- Chain capture termination conditions differ
- Player turn advancement after captures

**Affected Seeds:** 17

---

### 4.3 Territory Processing

**Description:** Territory region detection and processing order differ between engines.

**Symptoms:**

- Different number of disconnected regions detected
- Different gating logic for which regions can be processed
- `process_territory_region` moves differ

**Root Cause Candidates:**

- [`BoardManager.findDisconnectedRegions()`](../src/server/game/BoardManager.ts)
- [`findDisconnectedRegionsOnBoard()`](../src/client/sandbox/sandboxTerritory.ts)
- [`findDisconnectedRegions()`](../src/shared/engine/territoryDetection.ts) (shared)
- Region eligibility gating differences

**Affected Seeds:** 5

---

### 4.4 Placement Validation

**Description:** Backend rejects placements that sandbox considers legal.

**Symptoms:**

- Sandbox attempts `place_ring` with `placementCount > 1`
- Backend rejects as "dead placement"

**Root Cause Candidates:**

- [`hasAnyLegalMoveOrCaptureFrom()`](../src/server/game/RuleEngine.ts)
- Hypothetical board evaluation differences
- Multi-ring placement handling

**Affected Seeds:** 14

---

### 4.5 Phase/Player Tracking

**Description:** Late-game divergence in whose turn it is or what phase.

**Symptoms:**

- Game state hash matches but currentPlayer differs
- Phase mismatch at end of game

**Root Cause Candidates:**

- [`advanceTurnAndPhaseForCurrentPlayerSandbox()`](../src/shared/engine/turnLogic.ts)
- [`GameEngine.advanceGame()`](../src/server/game/GameEngine.ts) defensive skip logic
- Player elimination handling

**Affected Seeds:** 5, 17

---

## 5. Skipped Test Files Summary

| File                                                                                                                           | Seed(s) | TODO Marker                     | Description                  |
| ------------------------------------------------------------------------------------------------------------------------------ | ------- | ------------------------------- | ---------------------------- |
| [`Backend_vs_Sandbox.traceParity.test.ts`](../tests/unit/Backend_vs_Sandbox.traceParity.test.ts)                               | All     | `TODO-TRACE-PARITY`             | Full trace replay parity     |
| [`TraceParity.seed5.firstDivergence.test.ts`](../tests/unit/TraceParity.seed5.firstDivergence.test.ts)                         | 5       | None (active)                   | First divergence helper      |
| [`TraceParity.seed14.firstDivergence.test.ts`](../tests/unit/TraceParity.seed14.firstDivergence.test.ts)                       | 14      | `TODO-SEED14-DIVERGENCE`        | First divergence helper      |
| [`TraceParity.seed17.firstDivergence.test.ts`](../tests/unit/TraceParity.seed17.firstDivergence.test.ts)                       | 17      | `TODO-SEED17-STRICT-DIVERGENCE` | First divergence (strict)    |
| [`Seed17Move16And33Parity.GameEngine_vs_Sandbox.test.ts`](../tests/unit/Seed17Move16And33Parity.GameEngine_vs_Sandbox.test.ts) | 17      | `TODO-SEED17-CAPTURE-PARITY`    | Focused capture parity       |
| [`TerritoryDecision.seed5Move45.parity.test.ts`](../tests/unit/TerritoryDecision.seed5Move45.parity.test.ts)                   | 5       | `.skip`                         | Territory decision parity    |
| [`TraceFixtures.sharedEngineParity.test.ts`](../tests/unit/TraceFixtures.sharedEngineParity.test.ts)                           | N/A     | `TODO-TRACE-FIXTURES-PARITY`    | Shared engine fixture replay |

---

## 6. Remediation Tracking

| ID      | Seed | Divergence                     | Fix Strategy                                            | Owner | Status                          | Est. Effort |
| ------- | ---- | ------------------------------ | ------------------------------------------------------- | ----- | ------------------------------- | ----------- |
| DIV-001 | 5    | Capture enumeration at move 12 | Unified via shared captureLogic.ts                      | -     | **RESOLVED**                    | -           |
| DIV-002 | 5    | Territory at move 43/45        | Align BoardManager.findDisconnectedRegions with sandbox | -     | **RESOLVED**                    | -           |
| DIV-003 | 14   | Placement at move 31           | Review hasAnyLegalMoveOrCaptureFrom for multi-ring      | -     | Open                            | Medium      |
| DIV-004 | 14   | Line processing at move 33     | Compare line detection implementations                  | -     | Open                            | Medium      |
| DIV-005 | 17   | Capture at move 16             | Same root cause as DIV-001                              | -     | Open                            | Medium      |
| DIV-006 | 17   | Chain capture at move 33       | Audit chain capture phase exit conditions               | -     | Open                            | High        |
| DIV-007 | 17   | Phase tracking at move 52      | Align advanceGame with sandbox turn logic               | -     | Open                            | Medium      |
| DIV-008 | All  | Phase/player late-game         | Unify turn advancement semantics                        | -     | **DEFERRED** (within tolerance) | High        |

---

## 7. Investigation Tools & Commands

### 7.1 Run Specific Parity Tests

```bash
# Seed 5 first divergence (not skipped)
npm test -- TraceParity.seed5.firstDivergence

# Seed 5 bisect (skipped, enable manually)
npm test -- Backend_vs_Sandbox.seed5.bisectParity

# Seed 14 parity debug
npm test -- ParityDebug.seed14.trace

# Seed 17 trace debug
npm test -- Sandbox_vs_Backend.seed17.traceDebug
```

### 7.2 Enable Debug Logging

```bash
# Trace debug output
RINGRIFT_TRACE_DEBUG=1 npm test -- <test-name>

# AI debug output
RINGRIFT_AI_DEBUG=1 npm test -- <test-name>
```

### 7.3 Search for Parity-Related Code

```bash
# Find all TODO markers
grep -r "TODO-.*PARITY\|TODO-SEED" tests/unit/

# Find skipped parity tests
grep -r "describe.skip\|it.skip" tests/unit/*Parity*.test.ts

# Find capture enumeration logic
grep -rn "overtaking_capture\|enumerateCapture" src/
```

### 7.4 Dump Trace for Inspection

```bash
npx ts-node tests/scripts/dump_seed5_trace.ts
```

---

## 8. Utilities & Infrastructure

### 8.1 Primary Trace Utilities

| File                                                                | Purpose                                                                                  |
| ------------------------------------------------------------------- | ---------------------------------------------------------------------------------------- |
| [`tests/utils/traces.ts`](../tests/utils/traces.ts)                 | `runSandboxAITrace()`, `replayMovesOnBackend()`, `createBackendEngineFromInitialState()` |
| [`tests/utils/bisectParity.ts`](../tests/utils/bisectParity.ts)     | `runBackendVsSandboxBisect()` – binary search for first divergence                       |
| [`tests/utils/moveMatching.ts`](../tests/utils/moveMatching.ts)     | `findMatchingBackendMove()` – loose move matching for replay                             |
| [`tests/utils/stateSnapshots.ts`](../tests/utils/stateSnapshots.ts) | `snapshotFromGameState()` – comparable state snapshots                                   |
| [`tests/utils/traceReplayer.ts`](../tests/utils/traceReplayer.ts)   | Engine adapters and prefix replay helpers                                                |

### 8.2 Debug Scripts

| File                                                                                                    | Purpose                                       |
| ------------------------------------------------------------------------------------------------------- | --------------------------------------------- |
| [`tests/scripts/dump_seed5_trace.ts`](../tests/scripts/dump_seed5_trace.ts)                             | Dump seed 5 trace moves 40-50 with formatting |
| [`tests/scripts/generate_rules_parity_fixtures.ts`](../tests/scripts/generate_rules_parity_fixtures.ts) | Generate parity test fixtures                 |

---

## 9. Success Criteria

### 9.1 Seed 5 Resolved When:

- [x] **DIV-001 (capture enumeration) RESOLVED** – Unified via shared `captureLogic.ts`
- [x] **DIV-002 (territory processing) RESOLVED** – Territory region detection aligned
- [ ] All `Backend_vs_Sandbox.seed5.*.test.ts` suites pass (late-game divergence at move 64 remains – see DIV-008)
- [ ] Territory decision tests unskipped and passing

### 9.2 Seed 14 Resolved When:

- [ ] `TraceParity.seed14.firstDivergence` unskipped and passing
- [ ] Placement moves with `placementCount > 1` handled consistently
- [ ] Line parity tests passing

### 9.3 Seed 17 Resolved When:

- [ ] `Seed17Move16And33Parity` unskipped and passing
- [ ] `TraceParity.seed17.firstDivergence` unskipped and passing
- [ ] Chain capture phase transitions aligned

### 9.4 Full Parity Achieved When:

- [ ] `Backend_vs_Sandbox.traceParity.test.ts` unskipped and passing for all board types
- [ ] Zero TODO markers with "PARITY" in tests/unit/
- [ ] Trace-based debugging is reliable for any seed
- [x] **DIV-008 (late-game phase/player) DEFERRED** – Within tolerance, does not affect gameplay correctness

---

## 10. Related Documentation

- [KNOWN_ISSUES.md – P0.2](../KNOWN_ISSUES.md) – High-level parity gap documentation
- [archive/TRACE_PARITY_CONTINUATION_TASK.md](../archive/TRACE_PARITY_CONTINUATION_TASK.md) – Historical investigation notes
- [archive/SEED5_TAIL_DIVERGENCE_DIAGNOSTIC_SUMMARY.md](../archive/SEED5_TAIL_DIVERGENCE_DIAGNOSTIC_SUMMARY.md) – Seed 5 late-game analysis
- [archive/SEED5_LPS_ALIGNMENT_PROGRESS.md](../archive/SEED5_LPS_ALIGNMENT_PROGRESS.md) – LPS alignment work
