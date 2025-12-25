# AGENTS Guide for `src/` (TypeScript / Client / Server)

This file refines the root `AGENTS.md` for the TypeScript codebase under
`src/**` (shared types, engine, client UI, backend hosts).

It assumes the rules and engine semantics defined here are the executable
Single Source of Truth (SSoT) for the game.

---

## 1. SSoT: Types & Engine

- **Shared types**:
  - `src/shared/types/game.ts` is the authoritative shape for:
    - `BoardType`, `GamePhase`, `GameStatus`.
    - `MoveType`, `Move`, `GameState`, `GameResult`.
    - Any rules options and AI/control metadata exposed across layers.
- **Engine orchestration**:
  - `src/shared/engine/**`:
    - Turn/phase logic (`orchestration/turnOrchestrator.ts`, `fsm/TurnStateMachine.ts`, `fsm/FSMAdapter.ts`).
    - LPS/ANM/FE evaluation and weird‑state detection.
    - Hashing, invariants, and helpers used by parity tooling.

**Rules for agents:**

- If you change rules semantics (phases, MoveType behavior, LPS, FE, ANM),
  you must:
  - Update `game.ts` types and comments.
  - Update engine logic in `src/shared/engine/**`.
  - Keep Python mirrors (`ai-service/app/models/core.py`, `ai-service/app/game_engine.py`) in sync.
  - Align the canonical spec docs (`RULES_CANONICAL_SPEC.md`, etc.).

Do **not** hide rule changes only in client components or backend routes.

---

## 2. Canonical Phases & Forced Elimination

`GamePhase` in `src/shared/types/game.ts` must always include exactly:

1. `ring_placement`
2. `movement`
3. `capture`
4. `chain_capture`
5. `line_processing`
6. `territory_processing`
7. `forced_elimination`

Key constraints:

- `forced_elimination` is a real phase, not a flag:
  - `currentPhase === 'forced_elimination'` whenever FE is being resolved.
  - `MoveType 'forced_elimination'` is the only legal move type in this phase.
- No **silent** phase transitions:
  - Every phase visit must correspond to a recorded move in canonical history
    (action, voluntary skip, or forced no‑op).

When you modify phase or FE behavior:

- Keep the phase → MoveType mapping documented in comments in `game.ts` aligned
  with `ai-service/app/rules/history_contract.py`.
- Ensure `getValidMoves` / turn orchestrator never emits illegal MoveTypes for a phase.

---

## 3. Client & Backend Hosts

### 3.1 Backend hosts

Key files:

- `src/server/game/GameEngine.ts` and related backend host logic.
- `src/client/pages/BackendGameHost.tsx` – main backend game UI shell.
- Shared engine imports from `src/shared/engine/**`.

Guidelines:

- Backend must treat `src/shared/engine` as the authority for:
  - Legal moves, phase transitions, and victory detection.
  - Invariants like “no ACTIVE state with ANM(currentPlayer)”.
- If you add new outcome types, weird‑state reasons, or explanation hooks:
  - Thread them through a typed “engine view” structure.
  - Use `buildGameEndExplanationFromEngineView` (see below) to transform into UX copy and telemetry.

### 3.2 Client HUD, Victory Modal, and Teaching

Key components:

- `src/client/components/GameHUD.tsx`
- `src/client/components/VictoryModal.tsx`
- `src/client/components/TeachingOverlay.tsx`
- `src/client/adapters/gameViewModels.ts` (HUD + Victory view models)
- Weird‑state banners: `src/client/utils/gameStateWeirdness.ts`

Guidelines:

- Keep UI logic **rules‑aware but engine‑agnostic**:
  - Derive copy from shared types and small adapters, not from internal engine details.
  - Prefer using `GameEndExplanation` and `GameViewModel` adapters as the UX contract.
- Ensure teaching flows and weird‑state banners align with:
  - ANM/FE semantics,
  - Last Player Standing (LPS),
  - Structural stalemate definitions in docs.

---

## 4. GameEnd Explanations & Weird States

Engine‑level explanation:

- `src/shared/engine/gameEndExplanation.ts`:
  - `buildGameEndExplanation` – from a `GameEndExplanationSource`.
  - `buildGameEndExplanationFromEngineView` – from a `GameEndEngineView` plus UX metadata.
  - Output type: `GameEndExplanation` carries:
    - Outcome type and winner.
    - Victory reason codes (ANM, FE, LPS, territory, stalemate).
    - Weird‑state context, rules references, telemetry tags.

Tests:

- `tests/unit/GameEndExplanation.builder.test.ts`
- `tests/unit/GameEndExplanation.fromEngineView.test.ts`

**Rules for agents:**

- When you change how endings are classified (ANM/FE/LPS/territory/stalemate):
  - Update `gameEndExplanation.ts` logic.
  - Update tests to reflect the intended UX/telemetry contract.
  - Wire new reason codes into HUD/Teaching overlays only via explanation or
    view‑model layers, not ad‑hoc flags.

Wiring expectations:

- Backend and sandbox hosts should call `buildGameEndExplanationFromEngineView`
  when a game finishes and:
  - Surface `GameEndExplanation` into:
    - HUD (banners / tooltips),
    - VictoryModal,
    - Telemetry events (rules context tags, weird‑state reason codes).

---

## 5. TS↔Python Parity & Invariants (TS Side)

Parity + soak tooling:

- `scripts/run-orchestrator-soak.ts`:
  - Runs many random self‑play games via TS orchestrator.
  - Checks invariants using `isANMState` and related helpers.
- `scripts/selfplay-db-ts-replay.ts`:
  - Replays games from Python GameReplayDBs using the TS engine.
  - Used by Python parity harnesses as the TS side.

Agent expectations:

- Keep TS invariants strong:
  - No ACTIVE state should satisfy ANM for `currentPlayer`.
  - 7‑phase ordering must hold for all test games and soaks.
  - FE semantics must match canonical spec and Python engine.
- When parity issues appear, **do not** patch around them by weakening TS
  invariants; instead, fix engine or bridge logic.

---

## 6. Testing & Code Style (src/\*\*)

- Use existing test infrastructure:
  - Jest + `tests/unit/**` for unit tests.
  - Playwright/E2E only when necessary.
- Keep components and adapters:
  - Pure and typed, with explicit props.
  - Free of direct GameState mutations or engine calls; go through shared helpers.

When adding new files:

- Co‑locate tests under `tests/unit/**` mirroring the module path.
- Prefer small, focused utilities over monolithic files.

When editing existing files:

- Respect current formatting and naming conventions.
- Avoid introducing new global singletons; prefer dependency injection via props or context.
