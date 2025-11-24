# RingRift AI Architecture & Strategy

**Last Updated:** November 23, 2025
**Scope:** AI Service, Algorithms, Training Pipeline, and Integration

This document consolidates the architectural overview, technical assessment, and improvement plans for the RingRift AI system. It serves as the **canonical, current** reference for AI behaviour and design.

Historical deep-dive documents:

- `AI In Depth Improvement Analysis.md` – extensive analysis and review (historical; may describe older ladders/behaviour).
- `ai-service/AI_ASSESSMENT_REPORT.md` – Python AI service–focused assessment (historical snapshot).

For an up-to-date, ticket-ready roadmap, see **`AI_IMPROVEMENT_BACKLOG.md`**, which groups concrete tasks by theme (difficulty ladder, RNG/determinism, rules parity, search performance, NN robustness, behaviour tuning, tooling, and documentation).

---

## 1. Architecture Overview

### System Context

The AI system operates as a dedicated microservice (`ai-service`) built with Python/FastAPI, communicating with the main Node.js backend via HTTP.

- **Microservice:** `ai-service/` (Python 3.11+)
- **Communication:** REST API (`/ai/move`, `/ai/evaluate`, `/rules/evaluate_move`)
- **Integration:**
  - [`AIEngine`](src/server/game/ai/AIEngine.ts:135) (TypeScript) delegates to [`AIServiceClient`](src/server/services/AIServiceClient.ts:170) for AI moves.
  - [`RulesBackendFacade`](src/server/game/RulesBackendFacade.ts:1) (TypeScript) delegates to [`PythonRulesClient`](src/server/services/PythonRulesClient.ts:1) for rules validation (shadow/authoritative modes).
- **Resilience:** Multi-tier fallback system ensures games never get stuck due to AI failures.
- **UI Integration:** Full lobby and game UI support for AI opponent configuration and visualization.

### Rules, shared engine, and training topology

At runtime there are three tightly coupled layers that share the **same rules semantics** but serve different roles:

- **Shared TypeScript rules engine (canonical)**
  - Core game types (including `GameState`, `BoardState`, `Move`, and enums for phases and move types) live in [`game.ts`](src/shared/types/game.ts:1) and engine-specific type helpers in [`types.ts`](src/shared/engine/types.ts:1).
  - Movement, capture, lines, and territory are implemented as pure helpers under [`src/shared/engine`](src/shared/engine/core.ts:1):
    - Movement / reachability: [`movementLogic.ts`](src/shared/engine/movementLogic.ts:1), [`movementApplication.ts`](src/shared/engine/movementApplication.ts:1).
    - Capture / chains: [`captureLogic.ts`](src/shared/engine/captureLogic.ts:1), [`captureChainHelpers.ts`](src/shared/engine/captureChainHelpers.ts:1).
    - Lines: [`lineDetection.ts`](src/shared/engine/lineDetection.ts:1), [`lineDecisionHelpers.ts`](src/shared/engine/lineDecisionHelpers.ts:1).
    - Territory: [`territoryDetection.ts`](src/shared/engine/territoryDetection.ts:1), [`territoryProcessing.ts`](src/shared/engine/territoryProcessing.ts:1), [`territoryDecisionHelpers.ts`](src/shared/engine/territoryDecisionHelpers.ts:1).
  - These shared helpers are orchestrated by the TS shared [`GameEngine`](src/shared/engine/GameEngine.ts:37) and are used by:
    - The backend host engine in [`src/server/game/GameEngine.ts`](src/server/game/GameEngine.ts:1).
    - The client-local sandbox engine in [`ClientSandboxEngine.ts`](src/client/sandbox/ClientSandboxEngine.ts:1).
    - Rules/FAQ parity tests under `tests/unit/*shared.test.ts` and `tests/scenarios/RulesMatrix.*.test.ts`.

- **Python rules engine and mutators (AI service)**
  - The Python AI service embeds a rules engine that mirrors the shared TS engine:
    - Canonical Python engine orchestration: [`GameEngine`](ai-service/app/game_engine.py:33).
    - Board-level helpers (including disconnected-region detection): [`BoardManager.find_disconnected_regions()`](ai-service/app/board_manager.py:171).
    - Rules façade and shadow-contract mutators: [`DefaultRulesEngine`](ai-service/app/rules/default_engine.py:23), [`TerritoryMutator`](ai-service/app/rules/mutators/territory.py:6) and the other mutators under `ai-service/app/rules/mutators/`.
  - [`PythonRulesClient`](src/server/services/PythonRulesClient.ts:33) exposes this engine to the TS backend via `/rules/evaluate_move`, and [`RulesBackendFacade`](src/server/game/RulesBackendFacade.ts:1) decides whether to treat the Python engine as **shadow** (parity only) or **authoritative** (`RINGRIFT_RULES_MODE`).
  - Territory semantics are deliberately wired to mirror the TS shared helpers:
    - TS geometry and region detection: [`territoryDetection.ts`](src/shared/engine/territoryDetection.ts:1) ↔ Python [`BoardManager.find_disconnected_regions`](ai-service/app/board_manager.py:171).
    - TS region application and Q23 outside-stack prerequisite: [`territoryProcessing.ts`](src/shared/engine/territoryProcessing.ts:1) and [`territoryDecisionHelpers.ts`](src/shared/engine/territoryDecisionHelpers.ts:1) ↔ Python [`GameEngine._apply_territory_claim()`](ai-service/app/game_engine.py:2718).
    - Explicit territory self-elimination decisions: TS [`enumerateTerritoryEliminationMoves()`](src/shared/engine/territoryDecisionHelpers.ts:402) ↔ Python [`TerritoryMutator`](ai-service/app/rules/mutators/territory.py:6) and [`GameEngine._apply_forced_elimination()`](ai-service/app/game_engine.py:2988).

- **Training and dataset-generation pipelines (Python)**
  - General self-play dataset generation (policy/value, NN-style) is implemented in [`generate_data.py`](ai-service/app/training/generate_data.py:1) using:
    - The same Python [`GameEngine`](ai-service/app/game_engine.py:33) and [`RingRiftEnv`](ai-service/app/training/env.py:6) used by online AI search.
    - `DescentAI` and the neural network encoders from `ai-service/app/ai/`.
  - The **territory/combined-margin dataset generator** for heuristic training is implemented in [`generate_territory_dataset.py`](ai-service/app/training/generate_territory_dataset.py:1):
    - Builds a fresh `GameState` via [`RingRiftEnv`](ai-service/app/training/env.py:6) (which in turn uses [`create_initial_state()`](ai-service/app/training/generate_data.py:21)).
    - Uses [`GameEngine.get_valid_moves()`](ai-service/app/game_engine.py:45) and [`GameEngine.apply_move()`](ai-service/app/game_engine.py:117) as the single source of rules for self-play.
    - Serialises **pre-move** snapshots of the Python `GameState` along each trajectory with per-player scalar targets derived from the final board via [`_final_combined_margin()`](ai-service/app/training/generate_territory_dataset.py:79).
    - Emits one JSONL record per `(state, player)` with `game_state`, `player_number`, `target`, `time_weight`, and engine/AI metadata (`engine_mode`, `num_players`, `ai_type_pN`, `ai_difficulty_pN`).
  - Training jobs and the live AI/rules service therefore share:
    - The same Python `GameEngine` implementation for all rules, including territory and forced elimination.
    - The same `GameState` / `Move` model surface (mirroring TS [`GameState`](src/shared/types/game.ts:1)).
    - The same territory stack, now guarded by TS↔Python parity tests (see [`RULES_ENGINE_ARCHITECTURE.md`](RULES_ENGINE_ARCHITECTURE.md:1) and Python parity suites under `ai-service/tests/parity/` and [`test_territory_forced_elimination_divergence.py`](ai-service/tests/test_territory_forced_elimination_divergence.py:1)).

For a deeper, rules-focused mapping between TS and Python (including how parity fixtures and mutator contracts are enforced), see [`RULES_ENGINE_ARCHITECTURE.md`](RULES_ENGINE_ARCHITECTURE.md:1). For detailed CLI usage and dataset schemas for training, see [`docs/AI_TRAINING_AND_DATASETS.md`](docs/AI_TRAINING_AND_DATASETS.md:1). For the forced-elimination / TerritoryMutator divergence and its guard rails, see [`docs/INCIDENT_TERRITORY_MUTATOR_DIVERGENCE.md`](docs/INCIDENT_TERRITORY_MUTATOR_DIVERGENCE.md:1).

### Difficulty-to-AI-Type Mapping

The system provides a unified difficulty scale (1–10) that automatically selects the appropriate AI type. The **canonical ladder** is implemented in:

- **TypeScript backend:** `AI_DIFFICULTY_PRESETS` and `selectAITypeForDifficulty()` in
  [`src/server/game/ai/AIEngine.ts`](src/server/game/ai/AIEngine.ts)
- **Python AI service:** `_CANONICAL_DIFFICULTY_PROFILES` and `_select_ai_type()` in
  [`ai-service/app/main.py`](ai-service/app/main.py)

These two tables are kept in **lockstep** and are covered by unit tests on both sides
(`tests/unit/AIEngine.serviceClient.test.ts` and `ai-service/tests/test_ai_creation.py`).

#### Canonical Ladder (v1)

| Difficulty | Label       | AI Type     | Backend preset (TS)                              | Service profile (Python)            |
| ---------- | ----------- | ----------- | ------------------------------------------------ | ----------------------------------- |
| 1          | Beginner    | RandomAI    | `aiType: RANDOM`, `randomness: 0.5`, `150 ms`    | `AIType.RANDOM`, `0.5`, `150 ms`    |
| 2          | Easy        | HeuristicAI | `aiType: HEURISTIC`, `randomness: 0.3`, `200 ms` | `AIType.HEURISTIC`, `0.3`, `200 ms` |
| 3          | Level 3     | MinimaxAI   | `aiType: MINIMAX`, `randomness: 0.2`, `250 ms`   | `AIType.MINIMAX`, `0.2`, `250 ms`   |
| 4          | Level 4     | MinimaxAI   | `aiType: MINIMAX`, `randomness: 0.1`, `300 ms`   | `AIType.MINIMAX`, `0.1`, `300 ms`   |
| 5          | Level 5     | MinimaxAI   | `aiType: MINIMAX`, `randomness: 0.05`, `350 ms`  | `AIType.MINIMAX`, `0.05`, `350 ms`  |
| 6          | Level 6     | MinimaxAI   | `aiType: MINIMAX`, `randomness: 0.02`, `400 ms`  | `AIType.MINIMAX`, `0.02`, `400 ms`  |
| 7          | Expert      | MCTSAI      | `aiType: MCTS`, `randomness: 0.0`, `500 ms`      | `AIType.MCTS`, `0.0`, `500 ms`      |
| 8          | Expert+     | MCTSAI      | `aiType: MCTS`, `randomness: 0.0`, `600 ms`      | `AIType.MCTS`, `0.0`, `600 ms`      |
| 9          | Master      | DescentAI   | `aiType: DESCENT`, `randomness: 0.0`, `700 ms`   | `AIType.DESCENT`, `0.0`, `700 ms`   |
| 10         | Grandmaster | DescentAI   | `aiType: DESCENT`, `randomness: 0.0`, `800 ms`   | `AIType.DESCENT`, `0.0`, `800 ms`   |

Key properties:

- **Single source of truth:** any changes to the ladder must be reflected in both
  `AI_DIFFICULTY_PRESETS` (TS) and `_CANONICAL_DIFFICULTY_PROFILES` (Python), and in their
  corresponding tests.
- **AI type auto-selection:** when a profile omits an explicit `aiType`, the backend and
  service both derive it from the ladder via `selectAITypeForDifficulty()` / `_select_ai_type()`.
- **Service mapping:** the server’s internal `AIType` enum is mapped 1:1 onto the
  service’s `AIType` via `mapInternalTypeToServiceType()` in `AIEngine.ts` and the
  FastAPI models in `ai-service/app/models/core.py`.

Users can optionally override the AI type for specific testing or gameplay scenarios, but
**difficulty alone** should always reproduce the same underlying engine and coarse
behaviour across TS and Python.

### AI Implementations

**Production-supported tactical engines (behind the `AIType`/`AIServiceClient.AIType` enum):**

1.  **RandomAI** (`random`): Baseline engine for testing and very low difficulty.
2.  **HeuristicAI** (`heuristic`): Rule-based evaluation using weighted factors (stack control, territory, mobility).
3.  **MinimaxAI** (`minimax`): Alpha–beta search with move ordering and quiescence. Wired into the canonical difficulty ladder for difficulties 3–6 via the Python service’s `_CANONICAL_DIFFICULTY_PROFILES`, using `AIType.MINIMAX` with a bounded `think_time_ms` search budget.
4.  **MCTSAI** (`mcts`): Monte Carlo Tree Search with PUCT and RAVE, using the shared neural network for value/policy where weights are available. Selected by the ladder for difficulties 7–8.
5.  **DescentAI** (`descent`): UBFM/Descent-style tree search that also consumes the shared neural network for guidance and learning logs. Selected by the ladder for difficulties 9–10.

**Supporting / experimental components:**

- **NeuralNetAI:** CNN-based evaluation (value and policy heads) shared across board types (8×8, 19×19, hex) and used internally by `MCTSAI` and `DescentAI`.
- Training-side helpers and analysis tools under `ai-service/app/training/` (self-play data generation, tournaments, overfit tests).

The Python `ai-service` exposes these tactical engines via the `AIType` enum, and the TypeScript backend selects them through [`AIServiceClient.AIType`](src/server/services/AIServiceClient.ts:16) and the profile-driven mapping in [`AIEngine`](src/server/game/ai/AIEngine.ts:26).

### Neural Network Status

- **Architecture:** ResNet-style CNN (10 residual blocks).
- **Input:** 10-channel board representation + 10 global features.
- **Output:** Value (scalar) and Policy (probability distribution over ~55k moves).
- **Training:** Basic training loop implemented (`train.py`), but data generation (`generate_data.py`) needs improvement to use self-play with the current best model.

#### Hexagonal vs square boards

The underlying **geometry** of hexagonal and square boards is fundamentally
different:

- Square grids have 4-fold rotational symmetry and axis-aligned adjacency.
- Hex grids have 6-fold rotational symmetry and 3 principal axes; any
  embedding into a square lattice that preserves neighbourhood structure
  requires **non-linear** coordinate transforms (offset/cube/axial), not a
  single affine transform.

This has direct implications for convolutional networks:

- A single CNN trained jointly on square and hex boards using a naïve
  square-grid embedding would see **different local patterns** and symmetry
  structure for hex vs square positions, even if both are packed into the
  same H×W tensor.
- The network would effectively be asked to learn **two different games** on
  the same convolutional filters, which tends to dilute capacity and makes it
  harder to exploit the rotational/translation regularities specific to each
  board family.

**Conclusion:** for production play and serious training, the hexagonal board
should use a **separate neural network (or at least separate weights and
heads)** from the square (8×8 / 19×19) boards, with its own input encoding and
policy head.

#### Hex-specific network design (side-10 hex, 331 cells)

For the canonical regular hex board with 331 cells (side length N = 10):

- Total cells: `C = 3N² + 3N + 1 = 331` (already used in rules docs/tests).
- Bounding box in offset coordinates: `(2N+1) × (2N+1) = 21 × 21`.

We adopt a **dedicated hex model**:

- **Name:** `HexNeuralNet` (Python) / `HexNN_v1` (model tag)
- **Input tensor:** `[C_hex, 21, 21]` with a **binary mask** channel
  indicating which lattice sites are real hex cells.
  - Board-aligned channels (per 21×21 cell):
    - Current player’s ring height / presence
    - Opponent ring height / presence
    - Current player markers
    - Opponent markers
    - Collapsed / eliminated indicator
    - Territory ownership (if applicable to the variant)
    - Legal-move mask (optional, for policy pruning)
    - Turn / phase encodings (one-hot planes for placement/movement/capture,
      line/territory processing, etc.)
    - **Hex mask** (1 = valid hex cell, 0 = padding)
  - Global feature vector (concatenated later):
    - Player to move, remaining rings, score / line rewards, S-invariant
      summary, time controls, etc. (mirroring the square model’s globals).
- **Backbone:**
  - 8–10 residual blocks with 3×3 convolutions, stride 1, padding that
    preserves 21×21.
  - BatchNorm + ReLU (or GroupNorm for smaller batches).
- **Heads:**
  - **Policy head**:
    - 1×1 conv → BN → ReLU → flatten → linear to `P_hex` logits, where
      `P_hex` is the size of the **hex-only action space** (see
      `ai-service/tests/test_action_encoding.py` for square; hex needs a
      parallel encoder).
    - Mask invalid actions via a hex-specific `ActionEncoderHex`.
  - **Value head**:
    - 1×1 conv → BN → ReLU → global average pool (over 21×21 masked cells).
    - Concatenate with global features.
    - MLP → scalar tanh output in [-1, 1].
  - **Concrete hex action space (P_hex)**:
    - We fix the policy head dimension to `P_hex = 54_244`, computed as:
      - Placement: `21 × 21 × 3` = 1,323 slots (per-cell × 3 placement counts).
      - Movement / capture: `21 × 21 × 6 × 20` = 52,920 slots (origin index × 6
        hex directions × distance bucket up to 20).
      - Special: 1 slot reserved for `skip_placement` (additional special
        actions may be layered on in future versions).
    - The hex action encoder (`ActionEncoderHex`) uses this layout:
      - **Placements**: `pos_idx * 3 + (count - 1)` in `[0, 1,322]`, where
        `pos_idx = cy * 21 + cx` on the canonical 21×21 frame and
        `count ∈ {1,2,3}`.
      - **Movement / captures**: indices in
        `[HEX_MOVEMENT_BASE, HEX_MOVEMENT_BASE + 52_920)`, where
        `HEX_MOVEMENT_BASE = 1,323` and
        `idx = HEX_MOVEMENT_BASE + from_idx * (6 * 20) + dir_idx * 20 + (dist - 1)`,
        with `from_idx = from_cy * 21 + from_cx`, directions from the 6
        canonical hex directions `(1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)`,
        and `1 ≤ dist ≤ 20`.
      - **Special**: `idx == HEX_SPECIAL_BASE` encodes `skip_placement`, where
        `HEX_SPECIAL_BASE = HEX_MOVEMENT_BASE + 52_920`.
    - Any index that decodes to a canonical `(cx, cy)` outside the true hex
      shape (331 valid cells inside the 21×21 frame) is treated as invalid and
      yields `None` from `decode_move`, mirroring the square encoder’s
      handling of off-board positions.

The **square-board model** and **hex model** share high-level design (ResNet
backbone + (policy, value) heads), but they:

- Use different input shapes and masks.
- Use different action encoders and policy dimensions.
- Maintain **separate parameters and checkpoints** (e.g. `SquareNN_v1` vs
  `HexNN_v1`), trained on disjoint datasets.

`MCTSAI` and `DescentAI` should be parameterized by `boardType` and select the
appropriate network + action encoder for:

- `square8`, `square19` → square model/encoder.
- `hexagonal` → hex model/encoder.

### UI Integration

**Lobby (Game Creation)**

- AI opponent configuration panel with visual difficulty selector
- Support for 0-3 AI opponents per game
- Difficulty slider (1-10) with clear labels (Beginner/Intermediate/Advanced/Expert)
- Optional AI type and control mode overrides
- Clear visual feedback showing AI configuration before game creation

**Game Display**

- AI opponent indicator badges in game header and player cards
- Color-coded difficulty labels (green=Beginner, blue=Intermediate, purple=Advanced, red=Expert)
- AI type display (Random/Heuristic/Minimax/MCTS)
- Animated "thinking" indicators during AI turns
- Distinct styling for AI players vs human players

**Game Lifecycle**

- AI games auto-start immediately upon creation (no waiting for human opponents)
- AI moves are automatically triggered by GameSession when it's an AI player's turn
- AI games are unrated by default to prevent rating manipulation

---

## 2. Technical Assessment & Code Review

### Operational Stability

- **Status:** **Stable**
- **Verification:** Environment setup (`setup.sh`, `run.sh`) and dependencies are correct. Unit tests for MCTS (`tests/test_mcts_ai.py`) are passing.
- **Issues:** Runtime warning regarding model architecture mismatch (handled via fallback).

### Component Analysis

#### Heuristic AI (`heuristic_ai.py`)

- **Status:** **Improved**
- **Optimizations:** Mobility evaluation bottleneck resolved using "pseudo-mobility" heuristic.
- **Issues:** Hardcoded weights make dynamic tuning difficult. Redundant line-of-sight logic.

#### Minimax AI (`minimax_ai.py`)

- **Status:** **Significantly Improved**
- **Optimizations:** Safe time management, enhanced move ordering (MVV-LVA), optimized quiescence search.
- **Critical Issue:** Zobrist hashing is O(N) instead of O(1), which still limits the effectiveness of the transposition table.
- **Current Wiring:** MinimaxAI is now instantiated in `_create_ai_instance` and selected by the canonical difficulty ladder for difficulties 3–6, so mid–high difficulties actually exercise the minimax searcher instead of silently falling back to HeuristicAI.

#### MCTS Agent (`mcts_ai.py`)

- **Status:** **Improved**
- **Strengths:** Implements PUCT with RAVE heuristics. Supports batched inference.
- **Weaknesses:** Fallback rollout policy is weak. Tree reuse is not fully implemented. State copying during simulation is expensive.

#### Neural Network (`neural_net.py`)

- **Strengths:** ResNet-style CNN with adaptive pooling.
- **Weaknesses:** Architecture mismatch with saved checkpoint. History handling in training pipeline is flawed (see below).

---

## 5. RNG Determinism & Replay System

### Overview

RingRift implements comprehensive per-game RNG seeding to enable:

- **Deterministic replay:** Same seed + same inputs = same outputs
- **Cross-language parity:** TypeScript and Python produce identical sequences
- **Debugging:** Reproducible AI behavior for troubleshooting
- **Tournament validation:** Verify results by replaying games
- **Testing:** Reliable parity tests between engines

### Architecture

**Canonical Per-Game Seed (Database → Engine → AI):**

- The Prisma [`Game` model](prisma/schema.prisma) includes an optional `rngSeed: Int?` field.
- This **`Game.rngSeed` column is the single source of truth** for a game's RNG seed:
  - New games without a seed have one generated and persisted on first `GameSession.initialize()`.
  - Existing games created before seeding support will have `rngSeed = NULL` and are retrofitted on first load.
- The shared [`GameState`](src/shared/types/game.ts) type exposes an optional `rngSeed?: number` so both backend and sandbox engines can carry the canonical seed in-memory.

**TypeScript Implementation (Backend & Sandbox):**

- Core RNG primitive: [`SeededRNG`](src/shared/utils/rng.ts) (xorshift128+), providing:
  - `next(): number` in `[0, 1)`
  - `nextInt(max: number): number`
  - `shuffle<T>(items: T[]): T[]`
  - `choice<T>(items: T[]): T | undefined`
- **Backend game sessions**:
  - [`GameSession`](src/server/game/GameSession.ts) is responsible for wiring seeds on the server:
    1. On `initialize()` it loads the `Game` row (including `rngSeed`).
    2. If `rngSeed` is `NULL`, it calls `generateGameSeed()` once, persists the result via `prisma.game.update`, and logs any failure.
    3. It constructs a per-game `SeededRNG` instance (`this.rng`) from this `gameSeed`.
    4. It passes the same `gameSeed` into the [`GameEngine`](src/server/game/GameEngine.ts) constructor, which stores it on
       `gameState.rngSeed` but **does not** generate its own seed.
  - AI turns use the per-game RNG exclusively:
    - `GameSession.maybePerformAITurn()` always calls `globalAIEngine.getAIMove(...)` and `getLocalFallbackMove(...)`
      with `rng: () => this.rng.next()`.
    - Any last-resort random selection in `AIEngine` (`randomRng = rng ?? Math.random`) therefore runs on the
      per-game RNG whenever a caller supplies `rng`.
- **Backend AI engine**:
  - [`AIEngine`](src/server/game/ai/AIEngine.ts) does **not** own global RNG state. Instead, it:
    - Accepts an optional `rng: LocalAIRng` (`() => number`) in `getAIMove`, `getLocalFallbackMove`,
      and `chooseLocalMoveFromCandidates`.
    - Delegates all local heuristic move selection to the shared
      [`chooseLocalMoveFromCandidates`](src/shared/engine/localAIMoveSelection.ts), passing through the provided RNG.
  - When no RNG is supplied (legacy or test-only paths), `AIEngine` falls back to `Math.random`, but all production
    entry points (`GameSession` and sandbox) **always inject a per-game RNG**, so global randomness is not used in live games.
- **Shared/local AI policy**:
  - [`localAIMoveSelection`](src/shared/engine/localAIMoveSelection.ts) defines a phase-aware, side-effect-free
    policy for choosing among already-legal candidate moves.
  - It takes an explicit `rng: LocalAIRng = Math.random`, but:
    - Backend callers (`AIEngine`, via `GameSession`) and
    - Frontend sandbox callers (`sandboxAI.ts`, `ClientSandboxEngine.ts`)
      always pass a seeded RNG (`() => seededRng.next()` in both cases) when determinism is required.
  - Internally it sorts candidates deterministically before drawing from buckets, so that different engines
    (backend vs sandbox) see the same ordering and RNG stream.
- **Client sandbox engine**:
  - [`ClientSandboxEngine`](src/client/sandbox/ClientSandboxEngine.ts) maintains its own per-sandbox `SeededRNG`.
  - It initialises from one of:
    - `initialState.rngSeed` when provided (e.g. parity/debug traces), or
    - A fresh `generateGameSeed()` for ad-hoc local games.
  - All sandbox AI helpers (`maybeRunAITurnSandbox`, movement/decision builders) use `rng: () => this.rng.next()` and
    the same `localAIMoveSelection` policy as the backend.

**Python Implementation (AI Service):**

- Python uses `random.Random` instances for determinism:
  - [`BaseAI`](ai-service/app/ai/base.py) initialises `self.rng` from `AIConfig.rng_seed`.
  - All concrete engines (RandomAI, HeuristicAI, MinimaxAI, MCTSAI, DescentAI) draw from `self.rng`, **never** from the
    module-level `random` APIs in production paths.
- The AI config on the Python side is constructed from the TS payload:
  - [`AIServiceClient.getAIMove`](src/server/services/AIServiceClient.ts) accepts an optional `seed` argument.
  - When no explicit seed is provided, it falls back to `gameState.rngSeed` from the payload it serialises.
  - The `/ai/move` FastAPI endpoint accepts `seed: Optional[int]` and instantiates a fresh AI instance seeded from `seed`.

**API & Cross-Language Integration:**

- When `GameSession` calls `globalAIEngine.getAIMove` for a given turn:
  1. The `GameEngine`-backed `GameState` (including `rngSeed`) is passed to `AIEngine`.
  2. `AIEngine` calls `AIServiceClient.getAIMove(gameState, playerNumber, difficulty, aiType)`.
  3. `AIServiceClient` serialises `gameState` (including `rngSeed`) and either uses the
     provided `seed` argument or `gameState.rngSeed` as the canonical seed for the Python request.
  4. Python constructs `AIConfig(rng_seed=seed)` and seeds its internal `Random` instance.

This wiring guarantees that, for a fixed `Game.rngSeed` and identical game history, the TS backend and
Python service see the same RNG stream per game, modulo any non-determinism in low-level NN/GPU ops.

**Legacy / Non-Canonical RNG Usage:**

- `Math.random` is still used in a few **non-semantic** or legacy places:
  - UUID / debug ID helpers in rules modules (`GameEngine.generateUUID`, capture/territory helpers).
  - Unused legacy AI base class [`AIPlayer`](src/server/game/ai/AIPlayer.ts), which is explicitly
    documented as **not part of the production AI pipeline** and must not be reintroduced without an
    injected RNG refactor.
  - Test-only helpers that monkey-patch `Math.random` for deterministic simulations.
- None of these sites influence **which move** is selected in live games when `GameSession` and
  sandbox engines are supplying seeded RNG hooks.

**Reference Tests & Invariants:**

- **TS-side determinism & reconstruction:**
  - [`tests/integration/GameReconnection.test.ts`](tests/integration/GameReconnection.test.ts) builds an authoritative
    baseline via `GameEngine`, then mocks a Prisma `Game` row (including `rngSeed`) and asserts that two independent
    `GameSession.initialize()` instances reconstruct identical `GameState` hashes from the same history + seed.
  - [`tests/integration/GameSession.aiDeterminism.test.ts`](tests/integration/GameSession.aiDeterminism.test.ts)
    constructs two independent `GameSession` instances backed by the same mocked `Game` row (including `rngSeed`
    and `aiOpponents` config), drives a single AI turn in each via `maybePerformAITurn`, and asserts that both the
    last AI move and the resulting `hashGameState` are identical, with diagnostics confirming the move came from the
    local heuristic fallback path rather than the remote service.
- **Sandbox vs backend RNG parity:**
  - [`tests/unit/Sandbox_vs_Backend.aiRngParity.test.ts`](tests/unit/Sandbox_vs_Backend.aiRngParity.test.ts)
  - [`tests/unit/Sandbox_vs_Backend.aiRngFullParity.test.ts`](tests/unit/Sandbox_vs_Backend.aiRngFullParity.test.ts)
  - These suites inject `LocalAIRng` hooks (`() => seededRng.next()`) into both sandbox and backend AI flows and
    assert that, given identical seeds and states, both sides select the same sequence of AI moves.
- **Python-side determinism:**
  - [`ai-service/tests/test_determinism.py`](ai-service/tests/test_determinism.py) verifies that seeded `AIConfig`
    instances for RandomAI/HeuristicAI (and other engines as they are brought under test) produce identical move
    sequences for identical seeds and states.

Together, these tests form the basis of the RNG determinism contract: **same seed + same history ⇒ same moves**
across TS backend, sandbox engine, and Python AI service.

### Determinism Guarantees

**What is deterministic:**

- AI move selection with same seed + same game state
- Random tie-breaking in move evaluation
- MCTS exploration with same seed
- Line reward and territory processing choices

**What is NOT deterministic:**

- Network timing (latency, timeouts)
- Wall-clock timestamps
- Concurrent game execution order
- User input timing

### Testing

**TypeScript Tests:**

- [`RNGDeterminism.test.ts`](tests/unit/RNGDeterminism.test.ts:1): Core SeededRNG algorithm tests
- AI parity tests verify sandbox and backend produce identical sequences

**Python Tests:**

- [`test_determinism.py`](ai-service/tests/test_determinism.py:1): AI determinism with seeded configs
- Verify RandomAI, HeuristicAI produce identical moves with same seed

### Known Limitations

1. **Python Neural Network:** Some NN operations may use non-seeded GPU operations
2. **External Services:** Network calls introduce non-determinism in timing
3. **Process Isolation:** Python global state requires careful seed management

### Migration & Backward Compatibility

- **Existing games:** Migration sets `rngSeed` to NULL (games created before this feature)
- **API:** `seed` parameter is optional in all requests
- **Fallback:** Games without seed generate one automatically and log it for debugging
- **No breaking changes:** All existing code paths continue to work

---

## 6. Error Handling & Resilience

### Tiered Fallback Architecture

The AI system implements a robust three-tier fallback hierarchy to ensure games never get stuck due to AI service failures:

```
Level 1: Python AI Service (RemoteAI)
   ↓ (on failure: timeout, error, invalid move)
Level 2: Local Heuristic AI (TypeScript)
   ↓ (on failure: exception in local selection)
Level 3: Random Valid Move Selection
```

**Implementation:** [`AIEngine.getAIMove()`](src/server/game/ai/AIEngine.ts:228)

### Error Scenarios Handled

#### Network & Service Failures

- **Connection Refused:** AI service unreachable or not started
  - Circuit breaker opens after 5 consecutive failures
  - Automatic fallback to local heuristics
  - Service availability re-tested after 60-second cooldown

- **Timeouts:** AI service taking too long to respond
  - Default timeout: 30 seconds (configured in [`AIServiceClient`](src/server/services/AIServiceClient.ts:179))
  - Automatic fallback to local heuristics
  - Logged with latency metrics for monitoring

- **HTTP Errors:** Server errors (500, 503) from AI service
  - Categorized and logged with error type
  - Immediate fallback without retries
  - Circuit breaker tracks failure patterns

#### Invalid Move Responses

- **Move Validation:** All AI-suggested moves are validated against the legal move list from [`RuleEngine`](src/server/game/RuleEngine.ts:1)
  - Validates move type, player, positions, and special properties
  - Deep equality check including hexagonal coordinates
  - Invalid moves trigger automatic fallback

- **Malformed Responses:** AI service returns null or unparseable moves
  - Handled as service failure
  - Immediate fallback to local heuristics

- **Wrong Phase/Player:** AI suggests moves for incorrect game state
  - Caught by move validation
  - Fallback maintains game flow

### Circuit Breaker Pattern

**Implementation:** [`CircuitBreaker`](src/server/services/AIServiceClient.ts:20) class in AIServiceClient

**Behavior:**

- **Closed:** Normal operation, all requests attempt service
- **Opening:** After 5 consecutive failures within 60 seconds
- **Open:** Rejects requests immediately for 60 seconds
- **Half-Open:** After timeout, allows test request to check recovery

**Benefits:**

- Prevents hammering failing AI service
- Reduces cascade failures
- Automatic recovery detection
- Minimal latency when service is down

### Fallback Strategy

#### Level 1: Remote AI Service

- Uses Python microservice for sophisticated AI
- Supports all AI types (Random, Heuristic, Minimax, MCTS, Descent)
- Provides evaluation scores and thinking time metrics
- Protected by circuit breaker

#### Level 2: Local Heuristic AI

**Implementation:** [`AIEngine.selectLocalHeuristicMove()`](src/server/game/ai/AIEngine.ts:352)

- Uses shared [`chooseLocalMoveFromCandidates()`](src/shared/engine/localAIMoveSelection.ts:1)
- Prioritizes captures over movements
- Prefers moves that advance game state
- Deterministic with provided RNG
- Always produces valid moves

**Shared Policy:**

- Same heuristics used by sandbox AI and backend fallback
- Ensures consistent behavior across test/production
- Maintains game parity for debugging

#### Level 3: Random Selection

- Last resort when both service and heuristics fail
- Selects uniformly from valid moves using provided RNG
- Guarantees game progression
- Logs warning for monitoring

### Diagnostics & Monitoring

#### Per-Player Diagnostics

**[`AIDiagnostics`](src/server/game/ai/AIEngine.ts:50) Interface:**

```typescript
{
  serviceFailureCount: number; // Times AI service failed
  localFallbackCount: number; // Times local heuristic was used
}
```

**Access:** [`AIEngine.getDiagnostics(playerNumber)`](src/server/game/ai/AIEngine.ts:722)

#### Per-Game Quality Mode

[`GameSession`](src/server/game/GameSession.ts:42) tracks aggregate AI quality:

- `normal`: AI service working as expected
- `fallbackLocalAI`: Using local heuristics due to service issues
- `rulesServiceDegraded`: Python rules engine failures detected

**Access:** [`GameSession.getAIDiagnosticsSnapshotForTesting()`](src/server/game/GameSession.ts:832)

#### Logging

All AI failures are logged with context:

```typescript
logger.warn('Remote AI service failed, falling back to local heuristics', {
  error: error.message,
  playerNumber,
  difficulty,
});
```

**Log Levels:**

- `info`: Normal operation, successful fallbacks
- `warn`: Service failures, invalid moves, fallback usage
- `error`: Fatal errors, game abandonment

### Client-Side Error Handling

#### Error Events

[`GameSession`](src/server/game/GameSession.ts:759) emits `game_error` events when AI encounters fatal failures:

```typescript
socket.emit('game_error', {
  message: 'AI encountered a fatal error. Game cannot continue.',
  technical: error.message,
  gameId,
});
```

#### UI Feedback

[`GamePage`](src/client/pages/GamePage.tsx:1) displays error banners:

- User-friendly error message
- Technical details in development mode
- Dismissible notification
- Game marked as completed with abandonment

### Sandbox AI Resilience

[`sandboxAI.ts`](src/client/sandbox/sandboxAI.ts:1) implements comprehensive error handling:

- Top-level try-catch in [`maybeRunAITurnSandbox()`](src/client/sandbox/sandboxAI.ts:437)
- Error recovery in [`selectSandboxMovementMove()`](src/client/sandbox/sandboxAI.ts:392)
- Fallback to random selection on errors
- Never propagates exceptions to game engine
- Logs all errors for debugging

### Testing

#### Unit Tests

[`tests/unit/AIEngine.fallback.test.ts`](tests/unit/AIEngine.fallback.test.ts:1):

- Service failure handling
- Invalid move rejection
- Circuit breaker behavior
- Move validation logic
- Diagnostics tracking
- RNG determinism

#### Integration Tests

[`tests/integration/AIResilience.test.ts`](tests/integration/AIResilience.test.ts:1):

- Complete game with AI service down
- Intermittent failures
- Circuit breaker integration
- Performance under failure
- Error recovery patterns

### Operational Monitoring

**Health Checks:**

Endpoint: `/health/ai-service` (when implemented)

- Checks [`AIServiceClient.healthCheck()`](src/server/services/AIServiceClient.ts:455)
- Returns status: `healthy`, `degraded`, or `unavailable`

**Metrics to Monitor:**

1. **AI Service Availability:** Success rate of AI service calls
2. **Fallback Usage:** Frequency of local heuristic usage
3. **Circuit Breaker State:** Open/closed status and failure counts
4. **Move Validation Failures:** Rate of invalid moves from AI service
5. **Random Fallback Usage:** Should be near zero in production

**Alert Thresholds:**

- Service availability < 95%: Investigate AI service health
- Fallback usage > 20%: Check network or service degradation
- Circuit breaker open: Critical - AI service down
- Invalid moves > 1%: AI service logic issue

### Known Limitations

1. **Fatal Failures:** If all three tiers fail (extremely rare), game is abandoned
2. **Quality Degradation:** Local heuristics are weaker than trained AI
3. **No Retry Logic:** Service failures trigger immediate fallback (by design for responsiveness)
4. **Circuit Breaker State:** Shared across all games (not per-game isolation)

### Future Enhancements

1. **Adaptive Timeout:** Adjust timeout based on AI type and difficulty
2. **Quality Metrics:** Track move quality when using fallbacks
3. **Graceful Degradation:** Warn users when AI quality is degraded
4. **Service Pool:** Load balance across multiple AI service instances
5. **Caching:** Cache positions for common opening/endgame patterns

---

## 3. Improvement Plan & Roadmap

### Phase 1: Enhanced Training Pipeline (Immediate)

1.  **Fix History Handling:** Update `generate_data.py` to store full stacked feature tensors or reconstruct history correctly. Remove hacks in `train.py`.
2.  **Self-Play Loop:** Update `generate_data.py` to use the latest model for self-play data generation.
3.  **Rich Policy Targets:** Modify MCTS to return visit counts/probabilities instead of just the best move.
4.  **Data Augmentation:** Implement dihedral symmetries (rotation/reflection) for board data.

### Phase 2: Engine Optimization (Short-term)

1.  **Incremental Hashing:** Fix Zobrist hashing in `MinimaxAI` to be truly incremental (O(1)).
2.  **Batched Inference:** Ensure MCTS evaluates leaf nodes in batches to maximize throughput.
3.  **Tree Reuse:** Implement MCTS tree persistence between moves.

### Phase 3: Architecture Refinement (Medium-term)

1.  **Input Features:** Add history planes (last 3-5 moves) to capture dynamics.
2.  **Network Size:** Experiment with MobileNet vs ResNet-50 for different difficulty levels.
3.  **In-place State Updates:** Refactor `GameEngine` or create a specialized `FastGameEngine` for MCTS to eliminate copying overhead.

### Phase 4: Production Readiness (Long-term)

1.  **Model Versioning:** Implement a system to manage and serve different model versions.
2.  **Async Inference:** Use an async task queue (e.g., Celery/Redis) for heavy AI computations.

---

## 4. Rules Completeness in AI Service

- **Status:** **Mostly Complete**
- **Implemented:** Ring placement, movement, capturing (including chains), line formation, territory claiming, forced elimination, and victory conditions.
- **Simplifications:**
  - **Line Formation:** Automatically chooses to collapse all markers and eliminate from the largest stack (biasing against "Option 2").
  - **Territory Claim:** Automatically claims territory without user choice nuances.

**Recommendation:** Update `GameEngine` to support branching for Line Formation and Territory Claim choices to fully match the game's strategic depth.

---

## 5. Heuristic Training & Behaviour Tuning

This section describes how we treat `HeuristicAI` as a _trainable, versioned
component_ rather than a one-off hand-tuned evaluator, and how that interacts
with RingRift’s specific rules (ring placement, chain captures, lines,
territory, forced elimination, and dual victory conditions).

### 5.1 Goals

- Provide a **stable, documented evaluation basis** that can be improved over
  time without breaking backwards compatibility.
- Allow **designer-facing personas** (balanced / aggressive / territorial)
  backed by concrete weight profiles and, eventually, small NN correctors.
- Ensure the heuristic “understands” all of RingRift’s **tactically rich
  surfaces**:
  - Ring placement (including "no-dead-placement" constraints).
  - Movement and multi-step **chain captures**.
  - Line formation, including overlength lines and Option 1 vs Option 2
    rewards.
  - Territory regions, collapsed spaces, and forced **self-elimination**.
  - Dual victory conditions (eliminated rings vs territory) and the
    S-invariant.

### 5.2 RingRift-specific signal design

`HeuristicAI` today already encodes many RingRift-specific features; training
should make these signals _quantitatively_ correct and well-balanced:

- **Ring placement & rings in hand**
  - Reward placements that respect the "no-dead-placement" rule (i.e. avoid
    creating stacks with no legal moves).
  - Penalise hoarding too many rings in hand once the board is sufficiently
    developed.
  - Prefer placements that increase future mobility and line/territory
    potential rather than isolated stacks.

- **Movement & chain captures**
  - Reward capture sequences that:
    - Increase eliminated-rings margin.
    - Move toward victory thresholds (few rings remaining for the opponent).
    - Preserve or improve our own stack safety.
  - Penalise over-extended stacks that become easy capture targets on the next
    turn (vulnerability along line-of-sight lanes, especially in hex geometry).
  - Use the S-invariant as a _sanity check_: sequences that reduce our S
    without sufficient compensating gain should be disfavoured.

- **Lines (including Option 2)**
  - Distinguish between:
    - **Option 1**: collapse entire line, eliminate from tallest stack.
    - **Option 2**: collapse just three markers (minimum collapse, no
      elimination).
  - Reward Option 1 when:
    - It creates a decisive elimination advantage or immediate victory.
    - Territory and mobility loss are small.
  - Reward Option 2 when:
    - We are ahead on elimination and want to avoid giving the opponent
      counterplay.
    - Territory structure or future line potential would be badly damaged by a
      full collapse.

- **Territory & region processing**
  - Evaluate regions by:
    - Size (number of cells).
    - Strategic relevance (proximity to stacks and line threats).
    - Safety (distance to enemy stacks, likelihood of future invasion or
      self-elimination).
  - When multiple regions can be processed, score region-order choices by
    expected net gain in:
    - Collapsed spaces for us.
    - Forced elimination of opponent vs self.
    - Future mobility and line opportunities.

- **Forced elimination**
  - Treat forced self-elimination states as _tactically rich_, not merely
    losing:
    - Prefer eliminating rings from stacks that are tactically dead or
      strategically redundant.
    - Avoid eliminating rings from critical stacks that anchor territory
      regions or key lines.

- **Victory and S-invariant**
  - The heuristic’s scalar output should correlate strongly with:
    - Probability of eventual win (`P(win | state)`).
    - Time-to-win (shorter winning lines preferred; slower losses preferred
      when losing is unavoidable).
  - The S-invariant (`markers + collapsed + eliminated`) should be used as a
    coarse monotonicity check:
    - States with strictly better S and clear tactical/strategic advantages
      _should_ have strictly higher heuristic scores.

### 5.3 Training signals and data sources

We use a **teacher-based + outcome-based** strategy for training
`HeuristicAI` weights and any small NN correctors:

- **Teacher-based targets**
  - Use stronger engines (MCTS+NN, deeper Minimax, or Descent) as teachers.
  - For sampled states from real games and curated scenarios, record:
    - A scalar evaluation from the teacher (`v_teacher ≈ P(win | state)`).
    - Auxiliary labels (e.g. expected eliminated-rings margin, territory
      advantage, estimated moves-to-win).

- **Outcome-based targets**
  - For full self-play or human-vs-AI games, derive Monte Carlo estimates of
    `P(win | state)` from observed outcomes.
  - Weight positions closer to the end more strongly to reduce variance.

- **RingRift-specific scenario corpora**
  - Reuse existing test suites as **labelled tactical corpora**:
    - `RULES_SCENARIO_MATRIX.md` and the associated `RulesMatrix.*` tests.
    - FAQ scenario tests (`tests/scenarios/FAQ_*.test.ts`).
    - Plateau/stalemate scenarios (`test_ai_plateau_progress.py`,
      `ForcedEliminationAndStalemate.test.ts`).
    - Line and territory parity fixtures (`Seed14Move35LineParity`,
      `test_line_and_territory_scenario_parity.py`).
  - For each scenario, record states where humans or designers have a
    _preferred move_ or _clear judgement_ (e.g. "Option 2 is safer here").

- **Data pipeline**
  - Utilities under `ai-service/app/training/` and `ai-service/scripts/` turn
    logs + fixtures into `(features, target)` pairs using the same
    `_extract_features` encoder as production.
  - Datasets are sharded and versioned (e.g. `heuristic_v1_train`,
    `heuristic_v1_eval`) to keep training reproducible.

### 5.4 Optimising heuristic weights & personas

`HeuristicAI`’s evaluation is a **linear (or piecewise-linear) combination** of
feature terms. We optimise this in two stages:

1. **Base weight fit**
   - Solve a regularised regression problem:
     - Inputs: feature vectors from `_extract_features` (or higher-level
       aggregates such as stack/territory summaries).
     - Targets: teacher-based scalar values or outcome-based labels.
   - Use simple methods (ridge regression, small MLP with L2 regularisation)
     to obtain a _balanced_ `heuristic_v1_balanced` weight profile.
   - Enforce monotonicity constraints where appropriate (e.g., more enemy
     eliminated rings must not _improve_ our evaluation).

2. **Personas**
   - Define persona-specific deltas on top of the balanced profile:
     - **Aggressive:** upweight capture and elimination terms; slightly
       downweight territory-safety.
     - **Territorial:** upweight territory and line-formation terms; slightly
       downweight short-term elimination.
     - **Defensive:** upweight vulnerability and safety terms; penalise risky
       sacrifices.
   - Each persona becomes a named weight set (`heuristic_v1_aggressive`,
     `heuristic_v1_territorial`, etc.) referenced from `AIConfig`/
     `AIProfile` and selected by difficulty/queue configuration.

Trained weights are stored in a small `heuristic_weights.py` (Python) and
mirrored in TS (for improved fallback) where needed. Versioning is handled
explicitly (e.g. `heuristic_v1`, `heuristic_v2`) and documented in
`AI_IMPROVEMENT_BACKLOG.md` §6.4.

### 5.5 TS fallback AI & sandbox alignment

The TS-side local fallback AI (`AIEngine.getLocalAIMove` +
`chooseLocalMoveFromCandidates`) and sandbox AI are intentionally **much
simpler** than the Python `HeuristicAI`:

- They share a deterministic, RNG-parametrised selection policy that:
  - Buckets moves by type (captures vs quiet, placements vs movements).
  - Samples uniformly within buckets.
- This is ideal for **parity and resilience**, but too weak as a
  long-duration opponent at higher difficulties.

The long-term plan is to **bring TS fallback closer to the trained heuristic**
while keeping it cheap and deterministic:

- Port a _small subset_ of `HeuristicAI`’s features to TS (e.g. stack control,
  basic territory, vulnerability) and expose them via a light-weight
  evaluation function.
- Use the same trained weight profiles (or a coarser TS-only approximation) so
  fallback behaviour roughly matches Python `HeuristicAI` at the same
  difficulty/persona.
- Optionally expose a "remote AI" sandbox mode that calls `/ai/move` directly
  so designers can feel the exact production behaviour in the client.

As of November 2025 the first TS-side pieces are in place:

- `src/shared/engine/heuristicEvaluation.ts` implements
  `evaluateHeuristicState` (stack control/height, simple territory, local
  vulnerability) and now defines a small set of **v1 personas** mirroring the
  Python `heuristic_weights.py` ids:
  - `heuristic_v1_balanced`
  - `heuristic_v1_aggressive`
  - `heuristic_v1_territorial`
  - `heuristic_v1_defensive`
- A TS registry `HEURISTIC_WEIGHT_PROFILES_TS` and helper
  `getHeuristicWeightsTS(profileId)` provide persona lookup with a safe
  fallback to the balanced profile and also expose ladder-linked ids
  (`v1-heuristic-2`…`v1-heuristic-5`).
- Cross-language heuristic **parity fixtures** live under
  `tests/fixtures/heuristic/v1/*.json`. Each file encodes:
  - Minimal `GameState` snapshots.
  - A list of `(better, worse, profileId)` orderings that must hold for both
    the Python `HeuristicAI` and TS `evaluateHeuristicState`.
- Python parity tests in `ai-service/tests/test_heuristic_parity.py` and TS
  tests in `tests/unit/heuristicParity.shared.test.ts` load the same fixtures
  and assert ordering consistency per persona id.

Looking ahead, the TS heuristics module also exposes a **design-time stub**
`scoreMove({ before, move, playerNumber, weights })` that will eventually be
wired to the shared movement/placement mutators
(`movementApplication.ts`, `placementHelpers.ts`). Once those helpers are
fully implemented, `scoreMove` will:

1. Apply a candidate move to a cloned `before` state using shared mutators.
2. Evaluate both `before` and `after` via `evaluateHeuristicState` for the
   given persona.
3. Return a delta score `(after - before)` for use in tie-breaking and
   heuristic move ordering.

This stub is intentionally not called from production code yet; it exists to
lock down the API surface for future heuristic-driven selection
(`AIEngine`/sandbox) while keeping current behaviour and RNG parity unchanged.

These steps are tracked in `AI_IMPROVEMENT_BACKLOG.md` (§1.3 and §6.4) and will
be rolled out incrementally to preserve determinism and test coverage.
