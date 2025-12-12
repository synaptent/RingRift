# AGENTS Guide for RingRift

This file is for tool‑driven agents (LLMs, automation) working in this repo.  
It captures the key invariants, sources of truth, and expectations for changes.

If you are making non‑trivial edits anywhere in this repository, read this first.

---

## 1. Project Overview & Structure

- **Domain**: RingRift is a turn‑based board game with:
  - Canonical rules and replay semantics (SSoT in markdown specs).
  - A shared **TypeScript** rules/engine stack (the executable SSoT).
  - A **Python** AI service (replay DBs, parity, training).
- **Key directories**:
  - `src/shared/engine/**` – shared TS game engine + turn orchestrator.
  - `src/shared/types/**` – canonical TS types (`GameState`, `Move`, `GamePhase`, `MoveType`).
  - `src/client/**` – React client UI, HUD, teaching overlays, victory modals.
  - `src/server/**` – backend hosts for the TS engine.
  - `ai-service/**` – Python AI service:
    - `app/models/**` – Python mirrors of TS types.
    - `app/game_engine.py` – Python GameEngine replay semantics.
    - `app/db/**` – GameReplayDB and helpers.
    - `app/rules/**` – Python rules abstraction + history validators.
    - `scripts/**` – parity harnesses, self‑play, training utilities.
    - `tests/**` – pytest suite, including parity tests.
  - `scripts/**` (TS/Node) – orchestrator soaks, TS↔Python parity helpers, deployment checks.
  - `docs/**` – rules, UX, AI, calibration, and improvement loop documentation.

When in doubt about _what something is supposed to do_, prefer reading specs under `RULES_*.md`, `docs/**`, and TS engine code under `src/shared/engine/**` before modifying behavior.

---

## 2. Rules & Engine Single Source of Truth

### 2.1 Canonical Rules Specs

The rules semantics SSoT is:

- `RULES_CANONICAL_SPEC.md`
- `ringrift_complete_rules.md`
- `ringrift_compact_rules.md`
- Supporting UX/rules documents:
  - `docs/UX_RULES_WEIRD_STATES_SPEC.md`
  - `docs/UX_RULES_TEACHING_SCENARIOS.md`
  - `docs/UX_RULES_EXPLANATION_MODEL_SPEC.md`

**Requirements for agents:**

- Treat these as **normative**. If code disagrees with the canonical docs, the code is wrong unless the docs are explicitly being revised in tandem.
- Do not introduce alternative rules “variants” in shared engine paths unless they are:
  - Clearly optional, feature‑flagged, and
  - Documented against the canonical spec.

### 2.2 Executable Engine SSoT

- The executable engine SSoT lives in **TypeScript**:
  - `src/shared/engine/**` (especially `orchestration/turnOrchestrator.ts`, `phaseStateMachine.ts`, and their helpers).
  - Types in `src/shared/types/game.ts`.
- The **Python** side must mirror TS semantics; any divergence is a bug or non‑canonical data.

**Guidelines:**

- When changing rules or turn/phase behavior:
  - Update TS engine + TS types first.
  - Then propagate changes to Python mirrors (`ai-service/app/models/core.py`, `ai-service/app/game_engine.py`) and tests.
  - Keep the specs (`RULES_CANONICAL_SPEC.md`, etc.) in sync.
- Never “fix” parity by weakening canonical rules; instead, fix TS/Python logic or filter out invalid data.

---

## 3. Canonical Phases, Forced Elimination, and Move Semantics

### 3.1 Seven Canonical Phases (+ Terminal)

The canonical **turn** `currentPhase` values are:

1. `ring_placement`
2. `movement`
3. `capture`
4. `chain_capture`
5. `line_processing`
6. `territory_processing`
7. `forced_elimination`

A terminal `game_over` phase is used when the game ends (victory/stalemate). It is not a turn phase and must not be used for move recording.

Locations:

- TS: `src/shared/types/game.ts` (`GamePhase`).
- Python: `ai-service/app/models/core.py` (`GamePhase` enum).

Agents must keep these in lockstep between TS, Python, and any docs.

### 3.2 Canonical Move Types per Phase

The canonical **phase ↔ move_type** contract is encoded in:

- `ai-service/app/rules/history_contract.py`:
  - `phase_move_contract()`
  - `derive_phase_from_move_type(move_type)`
  - `validate_canonical_move(phase, move_type)`

Key points:

- Each phase has a **whitelist** of allowed `MoveType`s (including both “action” and “skip/no‑action” moves such as `no_territory_action`, `skip_capture`).
- `forced_elimination` is:
  - A dedicated phase (`currentPhase = 'forced_elimination'`).
  - A dedicated move type (`MoveType.FORCED_ELIMINATION` / `'forced_elimination'`).
  - Entered only when the canonical preconditions for FE hold.

### 3.3 Canonical Replay Requirements (RR‑CANON‑R075)

Canonically:

1. **Every phase transition must be recorded as a move.**
   - No silent transitions between phases.
   - Even “no action possible” must be recorded with `no_*_action` move types.
2. **All players must traverse all phases** each turn:
   - Even eliminated/blocked players record “no action” or forced‑elimination moves as appropriate.
3. **Voluntary skip vs forced no‑op must be distinguished**:
   - Voluntary skips use `skip_*` move types.
   - Forced no‑ops (no legal actions) use `no_*_action` move types.
4. **Forced elimination**:
   - Is its own phase (`forced_elimination`) with explicit `forced_elimination` moves.
   - Must not be applied silently at territory exit or anywhere else.

**Do not introduce any code paths that:**

- Advance `currentPhase` without recording a move.
- Apply forced elimination implicitly without a `forced_elimination` move.

---

## 4. TS↔Python Parity & Debugging

### 4.1 Core Parity Harness

Python parity harness:

- `ai-service/scripts/check_ts_python_replay_parity.py`
  - Replays games from a `GameReplayDB` in TS and Python and compares:
    - Phase, player, status.
    - Canonical state hash (`hash_game_state` / `simpleHash`).
  - Important flags:
    - `--db <path>` – DB to check.
    - `--emit-fixtures-dir DIR` – writes compact parity fixture JSONs.
    - `--emit-state-bundles-dir DIR` – writes rich TS + Python `GameState` bundles near the first divergence.
    - `--compact` – reduces output verbosity.

State bundles and diffing:

- Bundles are written as `*.state_bundle.json` and contain:
  - Python + TS states at key `k` values around the first divergence.
  - Divergence metadata (`mismatch_kinds`, `diverged_at`, etc.).
- `ai-service/scripts/diff_state_bundle.py`:
  - `--bundle <path>` (required) and optional `--k <ts_k>`.
  - Reconstructs Python `GameState` from bundle JSON and prints:
    - Phase/player/status on both sides.
    - Counts of stacks, collapsed cells, total eliminations.
    - A concise structural diff summary (players, stacks, collapsed ownership).

**Recommended parity debug loop for agents:**

1. Run parity with `--emit-state-bundles-dir` on a target DB.
2. Pick a divergent game’s bundle and run `diff_state_bundle.py`.
3. Inspect phase/turn semantics and structural diffs at `diverged_at`.
4. Fix TS or Python logic (not data) to resolve the semantic mismatch.
5. Rerun parity on that DB until `games_with_semantic_divergence == 0`.

### 4.2 Canonical Phase‑History Validation

Read‑side validator:

- `ai-service/app/rules/history_validation.py`
  - `validate_canonical_history_for_game(db, game_id)`:
    - Uses `validate_canonical_move` for each recorded move.
    - Returns `CanonicalHistoryReport(is_canonical, issues)`.

CLIs:

- `ai-service/scripts/check_canonical_phase_history.py` – checks one DB by replaying games via `GameEngine.apply_move` and enforcing invariants.
- `ai-service/scripts/scan_canonical_phase_dbs.py` – scans many DBs, optionally deleting non‑canonical DBs and any models trained on them.

**Guidance for agents:**

- Use these scripts and helpers to classify DBs as canonical vs legacy.
- Never move a DB into the “canonical” list without running both:
  - TS↔Python parity (`check_ts_python_replay_parity.py` or higher‑level driver).
  - Canonical phase‑history validation (`validate_canonical_history_for_game` or `check_canonical_phase_history.py`).

---

## 5. Replay Databases & Canonical Data Pipeline

### 5.1 GameReplayDB

Location:

- `ai-service/app/db/game_replay.py`:
  - `GameReplayDB` – SQLite‑backed storage for games.
  - `GameWriter`, `GameRecorder`, and helpers in `app/db/recording.py`.

Key invariants:

- Schema version is tracked in `schema_metadata`; current version is `SCHEMA_VERSION = 6`.
- **Write‑time canonical enforcement**:
  - `GameReplayDB.__init__(db_path, snapshot_interval=..., enforce_canonical_history=True)`:
    - When `enforce_canonical_history=True` (default), `_store_move_conn` will:
      - Call `validate_canonical_move("", move.type.value)` for each recorded move.
      - Raise `ValueError` if the move is not canonical.
  - Only set `enforce_canonical_history=False` for:
    - Legacy migration tooling,
    - Explicitly non‑canonical test fixtures (and document why).

State reconstruction:

- `get_state_at_move(game_id, move_number)`:
  - Replays moves from initial state via `GameEngine.apply_move(..., trace_mode=True)`.
  - Ignores old snapshots that may predate current rules.

### 5.2 Canonical vs Legacy DBs

Registry:

- `ai-service/TRAINING_DATA_REGISTRY.md` is the authoritative inventory.
  - `canonical_*` – intended for new training; must pass parity + canonical history gates.
  - `selfplay_*`, `legacy_*`, etc. – **legacy_noncanonical** unless explicitly regenerated and re‑gated.

Important policy:

- **Do not train new models on legacy DBs.**
- Old self‑play data and models are considered toxic; any re‑introduction or reuse must be clearly marked “legacy_noncanonical” and used only for ablation/historical analysis.

### 5.3 Canonical Self‑Play Generator

Unified generator & gate:

- `ai-service/scripts/generate_canonical_selfplay.py`:
  - Runs canonical self‑play via `run_canonical_selfplay_parity_gate.py`.
  - Runs TS↔Python parity on the resulting DB.
  - Runs canonical history validation for every game.
  - Outputs a summary with:
    - `parity_gate` (including `passed_canonical_parity_gate`).
    - `canonical_history.games_checked` and `.non_canonical_games`.
    - `canonical_ok` flag (true only when all checks pass and at least one game exists).

**When regenerating or creating a canonical DB, agents should:**

1. Prefer `generate_canonical_selfplay.py` over ad‑hoc self‑play scripts.
2. Only mark a DB as canonical in `TRAINING_DATA_REGISTRY.md` if `canonical_ok` is true.
3. Keep the gate summary JSON (e.g. `db_health.canonical_square19.json`) alongside the registry.

---

## 6. AI Training Stack & Data

Key locations:

- `ai-service/app/training/**`:
  - `encoding.py` – encoders (must be consistent with 7‑phase / FE semantics).
  - `generate_data.py` – dataset generation from replay DBs.
  - `train.py`, `train_loop.py` – high‑level training entry points.
  - `model_versioning.py` – `ModelVersionManager` for checkpoint metadata/versioning.
- `ai-service/TRAINING_DATA_REGISTRY.md` – which DBs and models are canonical vs legacy.

Agent expectations:

- Do **not** silently re‑introduce or rely on legacy `*.npz`, `*.npy`, or `*.pth` artifacts if they have been removed.
- For any new training‑related feature:
  - Make clear whether it expects canonical data only.
  - Thread board‑type, rules version, and data provenance where appropriate.
  - Prefer small, composable CLIs over large monoliths.

Planned direction (for context):

- A v2 dataset layout that annotates:
  - `source_db`, `canonical: bool`, `rules_version`, `parity_hash`, board type.
- v2 models (`ringrift_v2_square8/19/hex.pth`) trained only on canonical DBs and documented in the registry.

---

## 7. UX, Teaching, and Game‑End Explanations

Core UX/rules alignment:

- `src/shared/engine/gameEndExplanation.ts`:
  - `buildGameEndExplanation` and `buildGameEndExplanationFromEngineView` turn engine outcomes into structured explanations and telemetry.
  - Must reflect:
    - ANM (“active no move”) sequences,
    - Forced elimination (FE),
    - Last player standing (LPS),
    - Structural stalemate, territory end‑games.
- Tests:
  - `tests/unit/GameEndExplanation.builder.test.ts`
  - `tests/unit/GameEndExplanation.fromEngineView.test.ts`
- Teaching & weird‑state specs:
  - `docs/UX_RULES_WEIRD_STATES_SPEC.md`
  - `docs/UX_RULES_TEACHING_SCENARIOS.md`
  - `docs/UX_RULES_EXPLANATION_MODEL_SPEC.md`
- UI components:
  - `src/client/components/GameHUD.tsx`
  - `src/client/components/TeachingOverlay.tsx`

Agent guidance:

- When editing rules about ANM/FE/LPS or structural stalemate:
  - Ensure `gameEndExplanation` logic and tests are updated.
  - Keep UX docs and teaching topics consistent (especially FE as its own phase with explicit moves).
  - Maintain or extend tests instead of loosening them.

---

## 8. General Coding & Tooling Guidelines for Agents

These are **in addition to** any broader system prompts or per‑directory AGENTS files that may be added later.

1. **Use existing helpers and scripts.**
   - Don’t reinvent parity, history validation, or self‑play orchestration; use the tools under `ai-service/scripts/**` and `scripts/**` where possible.

2. **Be conservative with data and models.**
   - Do not create new DBs or model files under ambiguous names; prefer:
     - `canonical_<board>.db` for canonical DBs.
     - Clearly `legacy_*` / `experimental_*` for anything non‑canonical.
   - Never silently promote a DB or model to canonical status; always document in `TRAINING_DATA_REGISTRY.md`.

3. **Respect invariants and specs.**
   - No silent phase transitions.
   - No silent forced eliminations.
   - TS and Python must agree on:
     - Phase sequence,
     - Move types per phase,
     - State hashes for canonical trajectories.

4. **Prefer small, testable changes.**
   - When debugging parity, fix one concrete divergence at a time using state bundles.
   - Add or adapt tests nearby (`ai-service/tests/**`, `tests/unit/**`) instead of only manual scripts.

5. **Do not re‑introduce deleted legacy artifacts.**
   - If you see references to old DBs or models that no longer exist, either:
     - Update them to canonical equivalents, or
     - Clearly mark them as historical in docs/tests (and avoid re‑creating files with the same names unless explicitly requested).

6. **Follow repo formatting/testing patterns.**
   - TypeScript/JS: Jest tests under `tests/**`, lint via existing configs.
   - Python: pytest under `ai-service/tests/**`, no new frameworks.
   - Run focused tests / scripts relevant to your changes; avoid blanket, long‑running soaks unless the user asks.

7. **Document non‑obvious behavior in the right place.**
   - Rules semantics → canonical specs + engine comments.
   - Data provenance → `TRAINING_DATA_REGISTRY.md` and script docstrings.
   - UX behavior → `docs/UX_*` and tests around `GameHUD` / `TeachingOverlay` / `gameEndExplanation`.

If a future AGENTS file appears deeper in the tree (e.g. under `ai-service/` or `src/`), follow that file's more specific instructions for changes within its subtree, using this root document as background context.

---

## 9. Quick Reference: Project Structure

```
RingRift/
├── src/
│   ├── shared/                    # Shared TypeScript rules engine (canonical SSoT)
│   │   ├── engine/                # Core rules logic
│   │   │   ├── orchestration/     # Turn orchestrator & phase state machine
│   │   │   ├── aggregates/        # Domain aggregates (Placement, Movement, Capture, Line, Territory, Victory)
│   │   │   ├── fsm/               # FSM validation (TurnStateMachine, FSMAdapter)
│   │   │   └── contracts/         # Cross-language parity contracts & serialization
│   │   ├── types/                 # Canonical game types (Move, GameState, BoardState, etc.)
│   │   └── decisions/             # Decision trees (PlayerChoice, LineDecision, etc.)
│   │
│   ├── server/                    # Express backend (Node.js)
│   │   ├── game/
│   │   │   ├── turn/TurnEngineAdapter.ts  # Backend adapter wrapping orchestrator
│   │   │   ├── ai/AIEngine.ts              # AI move selection & service integration
│   │   │   └── GameEngine.ts               # Session & turn management
│   │   ├── websocket/             # Socket.IO real-time communication
│   │   ├── routes/                # HTTP REST API
│   │   └── database/              # Prisma schema, migrations
│   │
│   └── client/                    # React frontend (TypeScript/TSX)
│       ├── sandbox/               # ClientSandboxEngine & local game logic
│       ├── components/            # React UI (BoardView, GameHUD, ChoiceDialog, etc.)
│       ├── pages/                 # Main pages (LobbyPage, GamePage, SandboxPage)
│       └── contexts/              # React contexts (GameContext, SandboxContext)
│
├── ai-service/                    # Python FastAPI microservice
│   ├── app/
│   │   ├── main.py                # FastAPI app & difficulty ladder config
│   │   ├── game_engine.py         # Python rules engine (mirrors TS)
│   │   ├── ai/                    # AI implementations (Random, Heuristic, Minimax, MCTS, Descent)
│   │   └── rules/                 # Rules mutators & validators
│   ├── scripts/                   # Training scripts, parity harnesses, self-play
│   └── tests/                     # pytest suites
│
├── tests/                         # Jest + Playwright test suites
│   ├── unit/                      # Unit tests
│   ├── integration/               # Integration tests
│   ├── scenarios/                 # Rules/FAQ scenario tests
│   ├── contracts/                 # Contract vector runner (TS side)
│   └── fixtures/contract-vectors/ # v2 contract vector JSONs
│
├── docs/                          # Comprehensive documentation
├── RULES_CANONICAL_SPEC.md        # Canonical rules SSoT (RR-CANON-RXXX rules)
├── ringrift_complete_rules.md     # Authoritative rulebook (narrative)
├── ringrift_compact_rules.md      # Compact implementation-oriented spec
├── PROJECT_GOALS.md               # Product/technical goals, scope
└── TODO.md                        # Task tracking
```

---

## 10. Quick Reference: Build & Run Commands

### Development

```bash
npm install                    # Install dependencies
npm run dev                    # Start backend + frontend (hot reload)
npm run dev:server             # Backend on :3000
npm run dev:client             # Frontend on :5173

# AI service (Python)
cd ai-service && ./setup.sh    # One-time: create venv
cd ai-service && ./run.sh      # Start uvicorn on :8001
```

### Testing

```bash
npm test                          # All Jest tests
npm run test:core                 # Fast core profile (PR gate)
npm run test:coverage             # Coverage report
npm run test:p0-robustness        # Pre-PR comprehensive gate
npm run test:orchestrator-parity  # Canonical orchestrator + contract tests
npm run test:e2e                  # Playwright E2E tests
cd ai-service && pytest           # Python tests
```

### Build

```bash
npm run build                  # Build server + client
npm start                      # Run production build
docker-compose up -d           # Full stack in Docker
```

---

## 11. Quick Reference: Board Configurations

| Board Type | Size | Total Spaces | Rings/Player | Line Length (2p) | Line Length (3-4p) |
| ---------- | ---- | ------------ | ------------ | ---------------- | ------------------ |
| square8    | 8    | 64           | 18           | 4                | 3                  |
| square19   | 19   | 361          | 72           | 4                | 4                  |
| hexagonal  | 13   | 469          | 96           | 4                | 4                  |

### Victory Thresholds (Ring Elimination per RR-CANON-R061)

Formula: `round(ringsPerPlayer × (2/3 + 1/3 × (numPlayers - 1)))`

| Board Type | 2-player | 3-player | 4-player |
| ---------- | -------- | -------- | -------- |
| square8    | 18       | 24       | 30       |
| square19   | 72       | 96       | 120      |
| hexagonal  | 96       | 128      | 160      |

---

## 12. Quick Reference: Key File Locations

| Need to...                  | File(s)                                                                       |
| --------------------------- | ----------------------------------------------------------------------------- |
| Understand game rules       | `RULES_CANONICAL_SPEC.md`, `ringrift_complete_rules.md`                       |
| Implement a rule            | `src/shared/engine/aggregates/*.ts` + test                                    |
| Add AI difficulty           | `src/server/game/ai/AIEngine.ts`, `ai-service/app/main.py`                    |
| Fix a WebSocket issue       | `src/server/websocket/server.ts`, `src/client/hooks/useGameConnection.ts`     |
| Render the board            | `src/client/components/BoardView.tsx`                                         |
| Handle player choices       | `src/client/components/ChoiceDialog.tsx`                                      |
| Access game state (backend) | `src/server/game/GameEngine.ts`                                               |
| Access game state (client)  | `src/client/contexts/GameContext.tsx` or `SandboxContext.tsx`                 |
| Test rules/parity           | `tests/unit/`, `tests/scenarios/`, `tests/contracts/`                         |
| Run Python tests            | `ai-service/tests/`                                                           |
| Configure boards            | `src/shared/types/game.ts` (BOARD_CONFIGS)                                    |
| Debug orchestrator          | `src/shared/engine/orchestration/turnOrchestrator.ts`                         |
| Debug parity issues         | `ai-service/scripts/check_ts_python_replay_parity.py`, `diff_state_bundle.py` |

---

## 13. Quick Reference: AI Service & Difficulty Ladder

### Difficulty Levels (1-10)

- **1**: RandomAI (random legal moves)
- **2**: HeuristicAI (strategic heuristics)
- **3-6**: MinimaxAI (depth-limited minimax with alpha-beta)
- **7-8**: MCTSAI (Monte Carlo tree search)
- **9-10**: DescentAI (UBFM/Descent-style tree search)

### AI Service Endpoints

- `GET /health` - Health check
- `POST /ai/move` - Get AI move for a game state
- `POST /ai/evaluate` - Position evaluation
- `POST /ai/choice` - AI decision for PlayerChoice

---

## 14. Quick Reference: Environment Variables

Key environment variables:

- `NODE_ENV` - `development`, `test`, `production`
- `DATABASE_URL` - PostgreSQL connection string
- `REDIS_URL` - Redis connection string
- `AI_SERVICE_URL` - Python AI service URL (default: `http://localhost:8001`)
- `RINGRIFT_TRACE_DEBUG` - Enable debug tracing
- `RINGRIFT_SKIP_SHADOW_CONTRACTS` - Skip contract validation (Python)
- `PYTHONPATH` - Set to `ai-service` for Python scripts
