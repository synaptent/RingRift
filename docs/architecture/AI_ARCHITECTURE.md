# RingRift AI Architecture & Strategy

**Last Updated:** December 6, 2025
**Scope:** AI Service, Algorithms, Training Pipeline, and Integration

> **SSoT alignment:** This document is a derived view over the following canonical sources:
>
> - **Single Source of Truth (SSoT):** The canonical rules defined in `RULES_CANONICAL_SPEC.md` (plus `../rules/COMPLETE_RULES.md` / `../rules/COMPACT_RULES.md`) are the **ultimate authority** for RingRift game semantics. All implementations must derive from and faithfully implement these canonical rules.
> - **Implementation hierarchy:**
>   - **TS shared engine** (`src/shared/engine/**`) is the _primary executable derivation_ of the canonical rules spec. If the TS engine and the canonical rules document disagree, that is a bug in the TS engine.
>   - **Python AI service** (`ai-service/app/**`) is a _host adapter_ that must mirror the canonical rules. If Python disagrees with the canonical rules or the validated TS engine behaviour, Python must be updated—never the other way around.
> - **Lifecycle/API SSoT:** `docs/architecture/CANONICAL_ENGINE_API.md`, `src/shared/types/game.ts`, `src/shared/engine/orchestration/types.ts`, `src/shared/types/websocket.ts`, and `src/shared/validation/websocketSchemas.ts` for the Move/orchestrator/WebSocket lifecycle that AI integrates with.
> - **AI/Training SSoT:** Executable Python AI service code under `ai-service/app/**` and its tests under `ai-service/tests/**`, plus CI workflows (`.github/workflows/ci.yml`) that build and validate the AI service.
>   - Offline optimisation & evaluation harnesses: `ai-service/scripts/run_cmaes_optimization.py`, `ai-service/scripts/run_genetic_heuristic_search.py`, and their associated tests (`ai-service/tests/test_heuristic_training_evaluation.py`, `ai-service/tests/test_multi_start_evaluation.py`) are treated as **sanity/diagnostic tooling** over the shared rules SSoT rather than independent sources of semantics.
> - **Precedence:** If this document ever conflicts with those specs, engines, types, or workflows, **code + tests win**, and this doc must be updated to match them.
>
> **Doc Status (2025-12-06): Active (with historical appendix)**
>
> - Canonical reference for the **current AI architecture and integration**: Python AI service, TS AI boundary, RNG/determinism, neural nets, training pipeline, and resilience/fallback behaviour.
> - As of Dec 2025, end‑to‑end GameRecord flow is implemented: online games populate canonical GameRecords via `GameRecordRepository`, and both Python self‑play (`generate_data.py --game-records-jsonl`) and Node (`scripts/export-game-records-jsonl.ts`) can export training‑ready `GameRecord` JSONL datasets.
> - References several historical deep-dive/assessment docs (under `archive/` and `deprecated/`); those are explicitly called out as historical snapshots and should be read as background only.
> - For the canonical rules engine and Move lifecycle, defer to `ARCHITECTURE_ASSESSMENT.md`, `ARCHITECTURE_REMEDIATION_PLAN.md`, `RULES_CANONICAL_SPEC.md`, and `docs/CANONICAL_ENGINE_API.md` (Move + orchestrator + hosts), which this document assumes as its rules SSoT.
>
> This document consolidates the architectural overview, technical assessment, and improvement plans for the RingRift AI system. It serves as the **canonical, current** reference for AI behaviour and design.

Historical deep-dive documents:

- `archive/AI In Depth Improvement Analysis.md` – extensive analysis and review (historical; may describe older ladders/behaviour).
- `deprecated/AI_ASSESSMENT_REPORT.md` – Python AI service–focused assessment (historical snapshot).

For an up-to-date, ticket-ready roadmap, see **`docs/supplementary/AI_IMPROVEMENT_BACKLOG.md`**, which groups concrete tasks by theme (difficulty ladder, RNG/determinism, rules parity, search performance, NN robustness, behaviour tuning, tooling, and documentation).

---

## 0. AI Incident Overview (rules vs AI vs infra)

This section is a shared stub for AI-related incidents; the detailed runbooks
(`docs/runbooks/AI_ERRORS.md`, `AI_PERFORMANCE.md`, `AI_FALLBACK.md`,
`AI_SERVICE_DOWN.md`) link back here for high-level classification.

When something looks “AI-broken” in production, ask **which layer is likely at fault**:

- **Rules engine / orchestrator (shared TS SSoT)**
  - Symptoms:
    - Illegal moves accepted or produced (violating `RULES_CANONICAL_SPEC.md`).
    - Incorrect scoring, victory conditions, or turn sequencing.
    - Disagreements between TS shared-engine tests and contract vectors
      vs all hosts (backend, sandbox, Python).
  - First checks:
    - Run shared rules suites and contract vectors:
      - TS: `tests/unit/*.shared.test.ts`, `tests/contracts/contractVectorRunner.test.ts`.
      - Python: `ai-service/tests/contracts/test_contract_vectors.py`.
    - Inspect orchestrator adapters and flags (now mostly fixed in production):
      `ORCHESTRATOR_ADAPTER_ENABLED`, `RINGRIFT_RULES_MODE`
      (see `docs/ORCHESTRATOR_ROLLOUT_PLAN.md` and
      `docs/ENVIRONMENT_VARIABLES.md` for current rollout semantics).
  - Playbook:
    - If the shared engine or orchestrator looks wrong, follow
      `docs/ORCHESTRATOR_ROLLOUT_PLAN.md` (Safe rollback checklist).
    - Do **not** change rules or flip to legacy engines from AI runbooks.

- **AI behaviour / models (Python service + TS AI boundary)**
  - Symptoms:
    - Moves are legal but obviously weak/odd for the configured difficulty.
    - Certain difficulties feel much slower/stronger/weaker after a model/config change.
    - Errors in `/ai/move` without any evidence of rules violations.
  - First checks:
    - AI-specific metrics and logs:
      `ringrift_ai_requests_total`, `ringrift_ai_request_duration_seconds_bucket`,
      `ringrift_ai_fallback_total`, `AI_MOVE_LATENCY`, `AI_MOVE_REQUESTS`.
    - Status of `AIServiceDown`, `AIRequestHighLatency`, `AIErrorsIncreasing`,
      `AIFallbackRateHigh` alerts.
  - Playbook:
    - Use `AI_ERRORS.md` and `AI_PERFORMANCE.md` to classify internal errors vs
      performance/capacity vs contract/input issues.
    - Use `AI_FALLBACK.md` / `AI_SERVICE_DOWN.md` for quality-preserving
      fallbacks when the remote service is unhealthy.

- **Infrastructure / transport (Node, network, host resources)**
  - Symptoms:
    - Timeouts, connection errors, or very high P95/P99 AI latencies.
    - AI works in isolation but fails or falls back under load.
    - Other services showing concurrent availability/latency/resource alerts.
  - First checks:
    - App readiness (`/ready`) and service status metrics:
      `ringrift_service_status{service="ai_service"}`.
    - Host/container resources (`docker stats ai-service`, CPU/memory dashboards).
    - Global runbooks: `HIGH_LATENCY.md`, `SERVICE_DEGRADATION.md`,
      `SERVICE_OFFLINE.md`.
  - Playbook:
    - Treat these as availability / performance incidents first; scale, tune
      timeouts, or adjust concurrency via `AI_PERFORMANCE.md` and
      `AI_SERVICE_DOWN.md`.

Rule of thumb:

- If a move is **illegal or contradicts the written rules**, suspect the **shared
  rules engine / orchestrator** and follow the orchestrator rollout plan.
- If moves are **legal but slow, weak, or failing intermittently**, suspect the
  **AI service or infra**, and use the AI runbooks; do not alter rules in
  response to AI-only issues.

## 1. Architecture Overview

### System Context

The AI system operates as a dedicated microservice (`ai-service`) built with Python/FastAPI, communicating with the main Node.js backend via HTTP.

- **Microservice:** `ai-service/` (Python 3.11+)
- **Communication:** REST API (`/ai/move`, `/ai/evaluate`, `/rules/evaluate_move`)
- **Integration:**
  - [`AIEngine`](src/server/game/ai/AIEngine.ts) (TypeScript) delegates to [`AIServiceClient`](src/server/services/AIServiceClient.ts) for AI moves.
  - [`RulesBackendFacade`](src/server/game/RulesBackendFacade.ts) (TypeScript) delegates to [`PythonRulesClient`](src/server/services/PythonRulesClient.ts) for rules validation (shadow/authoritative modes).
- **Resilience:** Multi-tier fallback system ensures games never get stuck due to AI failures.
- **UI Integration:** Full lobby and game UI support for AI opponent configuration and visualization.

> **Note (PASS20 - December 2025):** As of PASS20 completion, `ORCHESTRATOR_ADAPTER_ENABLED` is hardcoded to `true` and the former `ORCHESTRATOR_ROLLOUT_PERCENTAGE` flag was removed. The orchestrator is the only production code path. Legacy and shadow modes remain available only for diagnostics and debugging.

### Rules, shared engine, and training topology

> For the **authoritative Move / PendingDecision / PlayerChoice / WebSocket lifecycle**, see [`docs/architecture/CANONICAL_ENGINE_API.md` §3.9–3.10](./CANONICAL_ENGINE_API.md). This section is an AI- and training-centric view over that same orchestrator-centred flow.

At runtime there are three tightly coupled layers that share the **same rules semantics** but serve different roles:

- **Shared TypeScript Rules Engine (canonical)**
  - Core game types (including `GameState`, `BoardState`, `Move`, and enums for phases and move types) live in [`game.ts`](src/shared/types/game.ts) and engine-specific type helpers in [`types.ts`](src/shared/engine/types.ts).
  - Movement, capture, lines, and Territory are implemented as pure helpers under [`src/shared/engine`](src/shared/engine/core.ts):
    - Movement / reachability: [`movementLogic.ts`](src/shared/engine/movementLogic.ts), [`movementApplication.ts`](src/shared/engine/movementApplication.ts).
    - Capture / chains: [`captureLogic.ts`](src/shared/engine/captureLogic.ts), [`captureChainHelpers.ts`](src/shared/engine/captureChainHelpers.ts).
    - Lines: [`lineDetection.ts`](src/shared/engine/lineDetection.ts), [`lineDecisionHelpers.ts`](src/shared/engine/lineDecisionHelpers.ts).
    - Territory: [`territoryBorders.ts`](src/shared/engine/territoryBorders.ts), [`territoryProcessing.ts`](src/shared/engine/territoryProcessing.ts), [`territoryDecisionHelpers.ts`](src/shared/engine/territoryDecisionHelpers.ts).
  - These shared helpers are orchestrated by the **shared turn orchestrator** in
    [`src/shared/engine/orchestration/turnOrchestrator.ts`](src/shared/engine/orchestration/turnOrchestrator.ts),
    working through domain aggregates under
    [`src/shared/engine/aggregates/`](src/shared/engine/aggregates/):
    - `PlacementAggregate.ts`, `MovementAggregate.ts`, `CaptureAggregate.ts`,
      `LineAggregate.ts`, `TerritoryAggregate.ts`, `VictoryAggregate.ts`.
  - These helpers + aggregates are consumed via thin host/adapters:
    - **Backend host engine** [`src/server/game/GameEngine.ts`](src/server/game/GameEngine.ts)
      using [`TurnEngineAdapter`](src/server/game/turn/TurnEngineAdapter.ts) over the shared
      orchestrator (`processTurn` / `processTurnAsync`).
    - **Client-local sandbox engine** [`ClientSandboxEngine.ts`](src/client/sandbox/ClientSandboxEngine.ts)
      using [`SandboxOrchestratorAdapter`](src/client/sandbox/SandboxOrchestratorAdapter.ts) for local
      simulation, FAQ/RulesMatrix scenarios, and AI-debug flows.
    - Rules/FAQ and parity tests under `tests/unit/*shared.test.ts`,
      `tests/unit/Backend_vs_Sandbox.*.test.ts`,
      `tests/unit/Territory*.test.ts`, and
      `tests/unit/TraceFixtures.sharedEngineParity.test.ts`, which all treat the
      shared helpers + aggregates + orchestrator as the canonical rules surface.
  - Host orchestration modes and rollout:
    - Backend `GameEngine.makeMove()` can run in **orchestrator-adapter mode** (delegating to `TurnEngineAdapter` / `processTurnAsync`) or **legacy mode** (validating via `RuleEngine` and applying moves directly). In production, `ORCHESTRATOR_ADAPTER_ENABLED=true` and `ORCHESTRATOR_ROLLOUT_PERCENTAGE=100` make the adapter path the default, with `OrchestratorRolloutService` and its circuit breaker confining the legacy path to tests, diagnostics, and kill-switch/rollback scenarios.
    - `ClientSandboxEngine` similarly offers orchestrator-driven flows (via `SandboxOrchestratorAdapter`) alongside thin legacy UX helpers that call shared aggregates directly for interactive tooling.
    - Python never embeds the TS orchestrator; its `GameEngine` is treated as a host/adapter validated against the TS orchestrator + contracts via v2 contract vectors and parity suites (see [`RULES_ENGINE_ARCHITECTURE.md`](RULES_ENGINE_ARCHITECTURE.md) for a detailed mapping).

- **Python Rules Engine and mutators (AI Service)**
  - The Python AI Service embeds a Rules Engine that mirrors the shared TS engine:
    - Canonical Python engine orchestration: [`GameEngine`](ai-service/app/game_engine/__init__.py).
    - Board-level helpers (including disconnected-region detection): [`BoardManager.find_disconnected_regions()`](ai-service/app/board_manager.py).
    - Rules façade and shadow-contract mutators: [`DefaultRulesEngine`](ai-service/app/rules/default_engine.py), [`TerritoryMutator`](ai-service/app/rules/mutators/territory.py) and the other mutators under `ai-service/app/rules/mutators/`.
  - [`PythonRulesClient`](src/server/services/PythonRulesClient.ts) exposes this engine to the TS backend via `/rules/evaluate_move`, and [`RulesBackendFacade`](src/server/game/RulesBackendFacade.ts) decides whether to treat the Python engine as **shadow** (parity only) or **authoritative** (`RINGRIFT_RULES_MODE`).
  - Territory semantics are deliberately wired to mirror the TS shared helpers:
    - TS geometry and region detection: [`territoryDetection.ts`](src/shared/engine/territoryDetection.ts) ↔ Python [`BoardManager.find_disconnected_regions`](ai-service/app/board_manager.py).
    - TS region application and Q23 outside-stack prerequisite: [`territoryProcessing.ts`](src/shared/engine/territoryProcessing.ts) and [`territoryDecisionHelpers.ts`](src/shared/engine/territoryDecisionHelpers.ts) ↔ Python [`GameEngine._apply_territory_claim()`](ai-service/app/game_engine/__init__.py).
    - Explicit Territory self-elimination decisions: TS [`enumerateTerritoryEliminationMoves()`](src/shared/engine/territoryDecisionHelpers.ts) ↔ Python [`TerritoryMutator`](ai-service/app/rules/mutators/territory.py) and [`GameEngine._apply_forced_elimination()`](ai-service/app/game_engine/__init__.py).

- **Training and dataset-generation pipelines (Python)**
- - General self-play dataset generation (policy/value, NN-style) is implemented in [`generate_data.py`](ai-service/app/training/generate_data.py) using:
-     - The same Python [`GameEngine`](ai-service/app/game_engine/__init__.py) and [`RingRiftEnv`](ai-service/app/training/env.py) used by online AI search.
-     - `DescentAI` and the neural network encoders from `ai-service/app/ai/`.
- - The **Territory/combined-margin dataset generator** for heuristic training is implemented in [`generate_territory_dataset.py`](ai-service/app/training/generate_territory_dataset.py):
-     - Builds a fresh `GameState` via [`RingRiftEnv`](ai-service/app/training/env.py) (which in turn uses [`create_initial_state()`](ai-service/app/training/generate_data.py)).
-     - Uses [`GameEngine.get_valid_moves()`](ai-service/app/game_engine/__init__.py) and [`GameEngine.apply_move()`](ai-service/app/game_engine/__init__.py) as the single source of rules for self-play.
-     - Serialises **pre-move** snapshots of the Python `GameState` along each trajectory with per-player scalar targets derived from the final board via [`_final_combined_margin()`](ai-service/app/training/generate_territory_dataset.py).
-     - Emits one JSONL record per `(state, player)` with `game_state`, `player_number`, `target`, `time_weight`, and engine/AI metadata (`engine_mode`, `num_players`, `ai_type_pN`, `ai_difficulty_pN`).
- - Training jobs and the live AI/Rules Service therefore share:
-     - The same Python `GameEngine` implementation for all rules, including Territory and forced elimination.
-     - The same `GameState` / `Move` model surface (mirroring TS [`GameState`](src/shared/types/game.ts)).
-     - The same Territory stack, now guarded by TS↔Python parity tests (see [`RULES_ENGINE_ARCHITECTURE.md`](RULES_ENGINE_ARCHITECTURE.md) and Python parity suites under `ai-service/tests/parity/` and [`test_territory_forced_elimination_divergence.py`](ai-service/tests/test_territory_forced_elimination_divergence.py)).
-
- For a deeper, rules-focused mapping between TS and Python (including how parity fixtures and mutator contracts are enforced), see [`RULES_ENGINE_ARCHITECTURE.md`](RULES_ENGINE_ARCHITECTURE.md). For detailed CLI usage and dataset schemas for training, see [`docs/ai/AI_TRAINING_AND_DATASETS.md`](../ai/AI_TRAINING_AND_DATASETS.md). For the forced-elimination / TerritoryMutator divergence and its guard rails, see [`docs/incidents/INCIDENT_TERRITORY_MUTATOR_DIVERGENCE.md`](../incidents/INCIDENT_TERRITORY_MUTATOR_DIVERGENCE.md).

Async decision/cancellation semantics at the WebSocket boundary are exercised and guarded on the TS side by `tests/unit/WebSocketServer.sessionTermination.test.ts`, which couples `WebSocketServer.terminateUserSessions`, `GameSession.terminate`, and AI-backed decision flows (including `region_order` and `line_reward_option`) to the shared cancellation model; once the Python parity suites are green, treat that Jest suite as the canonical async/cancellation harness for live AI decisions.

- +### Pre-Training Preparation & Memory Budgeting
- +This section provides a **RingRift-specific pre-training checklist** and explains how the **global memory budget** (16&nbsp;GB by default, configurable via `RINGRIFT_MAX_MEMORY_GB`) is propagated through training jobs, search components, and long-running soaks. It is intentionally high-level and defers detailed procedures to [`docs/ai/AI_TRAINING_PREPARATION_GUIDE.md`](../ai/AI_TRAINING_PREPARATION_GUIDE.md), [`docs/ai/AI_TRAINING_AND_DATASETS.md`](../ai/AI_TRAINING_AND_DATASETS.md), and the assessment in [`docs/ai/AI_TRAINING_ASSESSMENT_FINAL.md`](../ai/AI_TRAINING_ASSESSMENT_FINAL.md).
- +#### High-level pre-training checklist
- +- **Clarify objective & dataset source**
- - Decide what you are training/tuning:
- - **Neural network policy/value** using self-play NPZ datasets from [`generate_data.py`](ai-service/app/training/generate_data.py).
- - **Heuristic scalar evaluators** using JSONL territory/combined-margin datasets from [`generate_territory_dataset.py`](ai-service/app/training/generate_territory_dataset.py).
- - Ensure **leak-free splits by game**, not by individual positions:
- - Use complete games as the unit of train/validation/test splitting (see recommendations in [`docs/ai/AI_TRAINING_PREPARATION_GUIDE.md` §6.1](../ai/AI_TRAINING_PREPARATION_GUIDE.md)).
- - Confirm coverage of critical regimes:
- - Early/mid/late game, forced-elimination / LPS, and near‑victory situations.
- - Draw scenarios from self-play plus rules/FAQ matrices and invariant suites (e.g. [`RULES_SCENARIO_MATRIX.md`](../rules/RULES_SCENARIO_MATRIX.md), `tests/scenarios/RulesMatrix.*.test.ts`, and parity/invariant tests under `ai-service/tests/`).
- +- **Start from stable architectures & hyperparameters**
- - **Heuristics:**
- - Treat the Python weight profiles in [`heuristic_weights.py`](ai-service/app/ai/heuristic_weights.py) and the TS fallback personas in [`heuristicEvaluation.ts`](src/shared/engine/heuristicEvaluation.ts) as the canonical starting point for `HeuristicAI`.
- - When adding new features, keep weights **sign-correct** (e.g. more eliminated opponent rings must not worsen our score) and of **conservative magnitude** relative to existing terms (see [`AI_ARCHITECTURE.md` §5.2–5.4](AI_ARCHITECTURE.md)).
- - **Neural networks:**
- - Start from the current CNNs (`RingRiftCNN` / `HexNeuralNet`) and baseline `TrainConfig` defaults in the training stack under `ai-service/app/training/**` (documented in [`docs/ai/AI_TRAINING_PREPARATION_GUIDE.md` §2](../ai/AI_TRAINING_PREPARATION_GUIDE.md)).
- - Treat architecture changes (depth, width, additional heads) as **second‑order** optimizations after data quality, regularisation, and batch sizing have been tuned (see empirical findings in [`docs/ai/AI_TRAINING_ASSESSMENT_FINAL.md` §6–7](../ai/AI_TRAINING_ASSESSMENT_FINAL.md)).
- +- **Configure and respect the global memory budget**
- - Set a host‑level limit via:
- - `RINGRIFT_MAX_MEMORY_GB` (env var; default `16.0`).
- - [`MemoryConfig`](ai-service/app/utils/memory_config.py) (documented in [`docs/ai/AI_TRAINING_ASSESSMENT_FINAL.md` §2](../ai/AI_TRAINING_ASSESSMENT_FINAL.md) and [`docs/ai/AI_TRAINING_PREPARATION_GUIDE.md` §§3,13](../ai/AI_TRAINING_PREPARATION_GUIDE.md)).
- - Ensure this budget is threaded through:
- - **Search data structures:** use [`BoundedTranspositionTable`](ai-service/app/ai/bounded_transposition_table.py) in `MinimaxAI` and `DescentAI` instead of unbounded dicts, to prevent OOMs during long searches and tournaments.
- - **Training loops:** size batches and (where applicable) DataLoader workers in [`train.py`](ai-service/app/training/train.py) according to `max_memory_gb`, preferring memory‑mapped NPZ datasets for large runs as described in [`docs/ai/AI_TRAINING_PREPARATION_GUIDE.md` §3](../ai/AI_TRAINING_PREPARATION_GUIDE.md).
- - - **Self-play / soaks:** select difficulty bands and GC intervals in [`run_self_play_soak.py`](ai-service/scripts/run_self_play_soak.py) that keep memory usage within the configured limit (see [`docs/testing/STRICT_INVARIANT_SOAKS.md` §2](../testing/STRICT_INVARIANT_SOAKS.md)).
- - **Validate weight initialization and training stability**
- - **NNs:** follow the Xavier/He initialization and validation helpers in [`docs/ai/AI_TRAINING_PREPARATION_GUIDE.md` §4](../ai/AI_TRAINING_PREPARATION_GUIDE.md):
-     - Check initial weight statistics, policy entropy, and value distributions on a small real batch before running long epochs.
- -     - Confirm no NaNs or explosive gradients (NaN/instability bugs in early pipelines are documented and fixed in [`docs/ai/AI_TRAINING_ASSESSMENT_FINAL.md` §4](../ai/AI_TRAINING_ASSESSMENT_FINAL.md)).
- - **Heuristics:** treat new or retuned features conservatively:
-     - Use plateaus/ordering constraints from regression fixtures under `tests/fixtures/heuristic/v1/**` plus parity tests in `ai-service/tests/test_heuristic_parity.py` and `tests/unit/heuristicParity.shared.test.ts` to guard against sign/magnitude mistakes.
- - **Define baselines, evaluation harness, and scenario battery**
- - **Baselines:**
-     - Random AI, canonical heuristic profiles, and the current NN are all available as `AIType`/difficulty presets (see difficulty ladder mapping earlier in this doc).
- - **Evaluation harness:**
- -     - Use [`evaluate_ai_models.py`](ai-service/scripts/evaluate_ai_models.py) for structured head‑to‑head matches and [`generate_statistical_report.py`](ai-service/scripts/generate_statistical_report.py) for CI‑style reports (Wilson CIs, p‑values, effect sizes), as summarized in [`docs/ai/AI_TRAINING_ASSESSMENT_FINAL.md` §§6,10](../ai/AI_TRAINING_ASSESSMENT_FINAL.md).
- - **Scenario batteries:**
-     - Plan targeted tests over:
-       - Rules matrix and FAQ scenarios (`RULES_SCENARIO_MATRIX.md`, `tests/scenarios/RulesMatrix.*.test.ts`).
- -       - Invariant and strict‑no‑move soaks described in [`docs/testing/STRICT_INVARIANT_SOAKS.md`](../testing/STRICT_INVARIANT_SOAKS.md).
-     - Plug these into the same evaluation/reporting scripts so regressions show up in the same statistical pipeline.
- - **Reproducibility and experiment management**
- - Fix seeds for:
-     - Self‑play generators (`--seed` flags in [`generate_data.py`](ai-service/app/training/generate_data.py) and [`generate_territory_dataset.py`](ai-service/app/training/generate_territory_dataset.py)).
-     - Training runs (`TrainConfig.seed` and seeding helpers in [`docs/ai/AI_TRAINING_PREPARATION_GUIDE.md` §7.1](../ai/AI_TRAINING_PREPARATION_GUIDE.md)).
-     - Evaluation batteries (`--seed` options in [`evaluate_ai_models.py`](ai-service/scripts/evaluate_ai_models.py)).
- - Record for each run:
-     - Git commit hash and dirty state.
-     - Dataset manifest/version (see the manifest format in [`docs/ai/AI_TRAINING_PREPARATION_GUIDE.md` §7.4](../ai/AI_TRAINING_PREPARATION_GUIDE.md)).
-     - `MemoryConfig` values and key hyperparameters.
-     - Paths to checkpoints, evaluation JSONs, and generated statistical reports.
- - **Documentation & ethical notes**
- - For **RingRift game AI**, ethical risk is low, but:
-     - Difficulty and AI type must match user expectations.
-     - Behaviour should be reproducible enough to debug and reason about (see RNG determinism section later in this doc).
- - Any reuse of this infrastructure outside game AI must additionally respect the data/privacy and security guidance in [`docs/security/DATA_LIFECYCLE_AND_PRIVACY.md`](../security/DATA_LIFECYCLE_AND_PRIVACY.md) and related security docs.
- - For end‑to‑end, procedural guidance (including concrete CLI examples and more detailed checklists), see:
- - [`docs/ai/AI_TRAINING_PREPARATION_GUIDE.md`](../ai/AI_TRAINING_PREPARATION_GUIDE.md) for the full pre‑flight and infrastructure checklist.
- - [`docs/ai/AI_TRAINING_AND_DATASETS.md`](../ai/AI_TRAINING_AND_DATASETS.md) for dataset schemas and generator usage.
- - [`docs/ai/AI_TRAINING_ASSESSMENT_FINAL.md`](../ai/AI_TRAINING_ASSESSMENT_FINAL.md) for the implemented memory limits, bug fixes, and empirical results that motivated these guidelines.

### Difficulty-to-AI-Type Mapping

The system provides a unified difficulty scale (1–10) that automatically selects the appropriate AI type. The **canonical ladder** is implemented in:

- **TypeScript backend:** `AI_DIFFICULTY_PRESETS` and `selectAITypeForDifficulty()` in
  [`src/server/game/ai/AIEngine.ts`](src/server/game/ai/AIEngine.ts)
- **Python AI Service:** `_CANONICAL_DIFFICULTY_PROFILES` and `_select_ai_type()` in
  [`ai-service/app/main.py`](ai-service/app/main.py)

These two tables are kept in **lockstep** and are covered by unit tests on both sides
(`tests/unit/AIEngine.serviceClient.test.ts` and `ai-service/tests/test_ai_creation.py`).

#### Canonical Ladder (v1)

| Difficulty | Label       | AI Type     | Backend preset (TS)                              | Service profile (Python)            |
| ---------- | ----------- | ----------- | ------------------------------------------------ | ----------------------------------- |
| 1          | Beginner    | RandomAI    | `aiType: RANDOM`, `randomness: 0.5`, `150 ms`    | `AIType.RANDOM`, `0.5`, `150 ms`    |
| 2          | Easy        | HeuristicAI | `aiType: HEURISTIC`, `randomness: 0.3`, `200 ms` | `AIType.HEURISTIC`, `0.3`, `200 ms` |
| 3          | Level 3     | MinimaxAI   | `aiType: MINIMAX`, `randomness: 0.2`, `1250 ms`  | `AIType.MINIMAX`, `0.2`, `1250 ms`  |
| 4          | Level 4     | MinimaxAI   | `aiType: MINIMAX`, `randomness: 0.1`, `2100 ms`  | `AIType.MINIMAX`, `0.1`, `2100 ms`  |
| 5          | Level 5     | MinimaxAI   | `aiType: MINIMAX`, `randomness: 0.05`, `3500 ms` | `AIType.MINIMAX`, `0.05`, `3500 ms` |
| 6          | Level 6     | MinimaxAI   | `aiType: MINIMAX`, `randomness: 0.02`, `4800 ms` | `AIType.MINIMAX`, `0.02`, `4800 ms` |
| 7          | Expert      | MCTSAI      | `aiType: MCTS`, `randomness: 0.0`, `7000 ms`     | `AIType.MCTS`, `0.0`, `7000 ms`     |
| 8          | Expert+     | MCTSAI      | `aiType: MCTS`, `randomness: 0.0`, `9600 ms`     | `AIType.MCTS`, `0.0`, `9600 ms`     |
| 9          | Master      | DescentAI   | `aiType: DESCENT`, `randomness: 0.0`, `12600 ms` | `AIType.DESCENT`, `0.0`, `12600 ms` |
| 10         | Grandmaster | DescentAI   | `aiType: DESCENT`, `randomness: 0.0`, `16000 ms` | `AIType.DESCENT`, `0.0`, `16000 ms` |

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
2.  **HeuristicAI** (`heuristic`): Rule-based evaluation using weighted factors (stack control, Territory, mobility).
3.  **MinimaxAI** (`minimax`): Alpha–beta search with move ordering and quiescence. Wired into the canonical difficulty ladder for difficulties 3–6 via the Python Service’s `_CANONICAL_DIFFICULTY_PROFILES`, using `AIType.MINIMAX` with a bounded `think_time_ms` **search-time** budget (no artificial post‑search delay).
4.  **MCTSAI** (`mcts`): Monte Carlo Tree Search with PUCT and RAVE, using the shared neural network for value/policy where weights are available. Selected by the ladder for difficulties 7–8.
5.  **DescentAI** (`descent`): UBFM/Descent-style tree search that also consumes the shared neural network for guidance and learning logs. Selected by the ladder for difficulties 9–10.

**Supporting / experimental components:**

- **NeuralNetAI:** CNN-based evaluation (value and policy heads) shared across board types (8×8, 19×19, hex) and used internally by `MCTSAI` and `DescentAI`.
- **Research AIs:** EBMO, GMO, and IG-GMO live in the Python AI service and are not part of the canonical difficulty ladder. Use them only via explicit AI type overrides or tournament tooling.
- Training-side helpers and analysis tools under `ai-service/app/training/` (self-play data generation, tournaments, overfit tests).

The Python `ai-service` exposes these tactical engines via the `AIType` enum, and the TypeScript backend selects them through [`AIServiceClient.AIType`](src/server/services/AIServiceClient.ts) and the profile-driven mapping in [`AIEngine`](src/server/game/ai/AIEngine.ts).

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

#### Hex-specific network design (side-13 hex, 469 cells)

For the canonical regular hex board with 469 cells (side length N = 12):

- Total cells: `C = 3N² + 3N + 1 = 469` (rules docs/tests).
- Bounding box in offset coordinates: `(2N+1) × (2N+1) = 25 × 25`.

We adopt a **dedicated hex model**:

- **Name:** `HexNeuralNet` (Python) / `HexNN_v1` (model tag)
- **Input tensor:** `[C_hex, 25, 25]` with a **binary mask** channel
  indicating which lattice sites are real hex cells.
  - Board-aligned channels (per 25×25 cell):
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
    preserves 25×25.
  - BatchNorm + ReLU (or GroupNorm for smaller batches).
- **Heads:**
  - **Policy head**:
    - 1×1 conv → BN → ReLU → flatten → linear to `P_hex` logits, where
      `P_hex` is the size of the **hex-only action space** (see
      `ai-service/tests/test_action_encoding.py` for square; hex needs a
      parallel encoder).
    - Mask invalid actions via a hex-specific `ActionEncoderHex`.
  - **Value head**:
    - 1×1 conv → BN → ReLU → global average pool (over 25×25 masked cells).
    - Concatenate with global features.
    - MLP → scalar tanh output in [-1, 1].
  - **Concrete hex action space (P_hex)**:
    - We fix the policy head dimension to `P_hex = 91_876`, computed as:
      - Placement: `25 × 25 × 3` = 1,875 slots (per-cell × 3 placement counts).
      - Movement / capture: `25 × 25 × 6 × 24` = 90,000 slots (origin index × 6
        hex directions × distance bucket up to 24).
      - Special: 1 slot reserved for `skip_placement` (additional special
        actions may be layered on in future versions).
    - The hex action encoder (`ActionEncoderHex`) uses this layout:
      - **Placements**: `pos_idx * 3 + (count - 1)` in `[0, 1,874]`, where
        `pos_idx = cy * 25 + cx` on the canonical 25×25 frame and
        `count ∈ {1,2,3}`.
      - **Movement / captures**: indices in
        `[HEX_MOVEMENT_BASE, HEX_MOVEMENT_BASE + 90_000)`, where
        `HEX_MOVEMENT_BASE = 1,875` and
        `idx = HEX_MOVEMENT_BASE + from_idx * (6 * 24) + dir_idx * 24 + (dist - 1)`,
        with `from_idx = from_cy * 25 + from_cx`, directions from the 6
        canonical hex directions `(1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)`,
        and `1 ≤ dist ≤ 24`.
      - **Special**: `idx == HEX_SPECIAL_BASE` encodes `skip_placement`, where
        `HEX_SPECIAL_BASE = HEX_MOVEMENT_BASE + 90_000`.
    - Any index that decodes to a canonical `(cx, cy)` outside the true hex
      shape (469 valid cells inside the 25×25 frame) is treated as invalid and
      yields `None` from `decode_move`, mirroring the square encoder's
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
- **Strengths:** Implements PUCT with RAVE heuristics. Supports batched inference. Uses make/unmake pattern via `MCTSNodeLite` and `_search_incremental()` for efficient state manipulation.
- **Weaknesses:** Fallback rollout policy is weak. Tree reuse is not fully implemented.

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

- **Python Implementation (AI Service):**

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
Python Service see the same RNG stream per game, modulo any non-determinism in low-level NN/GPU ops.

**Legacy / Non-Canonical RNG Usage:**

- `Math.random` is still used in a few **non-semantic** or legacy places:
  - UUID / debug ID helpers in rules modules (`GameEngine.generateUUID`, capture/Territory helpers).
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
  - _Historical:_ `ai-service/tests/test_determinism.py` previously verified that
    seeded `AIConfig` instances for RandomAI/HeuristicAI (and other engines as they were brought under test) produced
    identical move sequences for identical seeds and states. That legacy suite has been removed; determinism is now
    covered by [`test_engine_determinism.py`](ai-service/tests/test_engine_determinism.py) and
    [`test_no_random_in_rules_core.py`](ai-service/tests/test_no_random_in_rules_core.py).

Together, these tests form the basis of the RNG determinism contract: **same seed + same history ⇒ same moves**
across TS backend, sandbox engine, and Python AI Service.

### Determinism Guarantees

**What is deterministic:**

- AI move selection with same seed + same game state
- Random tie-breaking in move evaluation
- MCTS exploration with same seed
- Line reward and Territory processing choices

**What is NOT deterministic:**

- Network timing (latency, timeouts)
- Wall-clock timestamps
- Concurrent game execution order
- User input timing

### Testing

**TypeScript Tests:**

- [`EngineDeterminism.shared.test.ts`](tests/unit/EngineDeterminism.shared.test.ts): Shared-engine determinism and turn replay invariants for the canonical TS rules engine.
- [`NoRandomInCoreRules.test.ts`](tests/unit/NoRandomInCoreRules.test.ts): Guards against unseeded randomness in shared-engine helpers/aggregates/orchestrator.
- AI RNG / parity tests (`Sandbox_vs_Backend.aiRngParity.test.ts`, `Sandbox_vs_Backend.aiRngFullParity.test.ts`, and `GameSession.aiDeterminism.test.ts`) verify sandbox and backend produce identical AI move sequences for the same seeds and per-game RNG.
- _Historical:_ an earlier `RNGDeterminism.test.ts` suite exercised the raw `SeededRNG` implementation; its coverage is now subsumed by the integrated determinism and "no random in core" suites above.

**Python Tests:**

- [`test_engine_determinism.py`](ai-service/tests/test_engine_determinism.py): Python rules engine determinism (no divergence under fixed seeds and histories).
- [`test_no_random_in_rules_core.py`](ai-service/tests/test_no_random_in_rules_core.py): Guards against unseeded randomness in the Python rules core.
- AI plateau/trace parity tests under `ai-service/tests/parity/` (e.g. `test_ts_seed_plateau_snapshot_parity.py`, `test_ai_plateau_progress.py`, `test_line_and_territory_scenario_parity.py`) use TS-generated fixtures to validate fully deterministic cross-language behaviour at the scenario/trace level.

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

The AI system implements a robust three-tier fallback hierarchy to ensure games never get stuck due to AI Service failures:

```
Level 1: Python AI Service (RemoteAI)
   ↓ (on failure: timeout, error, invalid move)
Level 2: Local Heuristic AI (TypeScript)
   ↓ (on failure: exception in local selection)
Level 3: Random Valid Move Selection
```

**Implementation:** [`AIEngine.getAIMove()`](src/server/game/ai/AIEngine.ts)

### Error Scenarios Handled

#### Network & Service Failures

- **Connection Refused:** AI Service unreachable or not started
  - Circuit breaker opens after 5 consecutive failures
  - Automatic fallback to local heuristics
  - Service availability re-tested after 60-second cooldown

- **Timeouts:** AI Service taking too long to respond
  - Default timeout: 30 seconds (configured in [`AIServiceClient`](src/server/services/AIServiceClient.ts))
  - Automatic fallback to local heuristics
  - Logged with latency metrics for monitoring

- **HTTP Errors:** Server errors (500, 503) from AI Service
  - Categorized and logged with error type
  - Immediate fallback without retries
  - Circuit breaker tracks failure patterns

#### Invalid Move Responses

- **Move Validation:** All AI-suggested moves are validated against the legal move list from [`RuleEngine`](src/server/game/RuleEngine.ts)
  - Validates move type, player, positions, and special properties
  - Deep equality check including hexagonal coordinates
  - Invalid moves trigger automatic fallback

- **Malformed Responses:** AI Service returns null or unparseable moves
  - Handled as service failure
  - Immediate fallback to local heuristics

- **Wrong Phase/Player:** AI suggests moves for incorrect game state
  - Caught by move validation
  - Fallback maintains game flow

### Circuit Breaker Pattern

**Implementation:** [`CircuitBreaker`](src/server/services/AIServiceClient.ts) class in AIServiceClient

**Behavior:**

- **Closed:** Normal operation, all requests attempt service
- **Opening:** After 5 consecutive failures within 60 seconds
- **Open:** Rejects requests immediately for 60 seconds
- **Half-Open:** After timeout, allows test request to check recovery

**Benefits:**

- Prevents hammering failing AI Service
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

**Implementation:** [`AIEngine.selectLocalHeuristicMove()`](src/server/game/ai/AIEngine.ts)

- Uses shared [`chooseLocalMoveFromCandidates()`](src/shared/engine/localAIMoveSelection.ts)
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

**[`AIDiagnostics`](src/server/game/ai/AIEngine.ts) Interface:**

```typescript
{
  serviceFailureCount: number; // Times AI Service failed
  localFallbackCount: number; // Times local heuristic was used
}
```

**Access:** [`AIEngine.getDiagnostics(playerNumber)`](src/server/game/ai/AIEngine.ts)

#### Per-Game Quality Mode

[`GameSession`](src/server/game/GameSession.ts) tracks aggregate AI quality:

- `normal`: AI Service working as expected
- `fallbackLocalAI`: Using local heuristics due to service issues
- `rulesServiceDegraded`: Python Rules Engine failures detected

**Access:** [`GameSession.getAIDiagnosticsSnapshotForTesting()`](src/server/game/GameSession.ts)

#### Logging

All AI failures are logged with context:

```typescript
logger.warn('Remote AI Service failed, falling back to local heuristics', {
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

[`GameSession`](src/server/game/GameSession.ts) emits `game_error` events when AI encounters fatal failures:

```typescript
socket.emit('game_error', {
  message: 'AI encountered a fatal error. Game cannot continue.',
  technical: error.message,
  gameId,
});
```

#### UI Feedback

[`GamePage`](src/client/pages/GamePage.tsx) displays error banners:

- User-friendly error message
- Technical details in development mode
- Dismissible notification
- Game marked as completed with abandonment

### Sandbox AI Resilience

[`sandboxAI.ts`](src/client/sandbox/sandboxAI.ts) implements comprehensive error handling:

- Top-level try-catch in [`maybeRunAITurnSandbox()`](src/client/sandbox/sandboxAI.ts)
- Error recovery in [`selectSandboxMovementMove()`](src/client/sandbox/sandboxAI.ts)
- Fallback to random selection on errors
- Never propagates exceptions to game engine
- Logs all errors for debugging

### Testing

#### Unit Tests

[`tests/unit/AIEngine.fallback.test.ts`](tests/unit/AIEngine.fallback.test.ts):

- Service failure handling
- Invalid move rejection
- Circuit breaker behavior
- Move validation logic
- Diagnostics tracking
- RNG determinism

#### Integration Tests

[`tests/integration/AIResilience.test.ts`](tests/integration/AIResilience.test.ts):

- Complete game with AI Service down
- Intermittent failures
- Circuit breaker integration
- Performance under failure
- Error recovery patterns

### Operational Monitoring

**Health Checks:**

Endpoint: `/health/ai-service` (when implemented)

- Checks [`AIServiceClient.healthCheck()`](src/server/services/AIServiceClient.ts)
- Returns status: `healthy`, `degraded`, or `unavailable`

**Metrics to Monitor:**

1. **AI Service Availability:** Success rate of AI Service calls
2. **Fallback Usage:** Frequency of local heuristic usage
3. **Circuit Breaker State:** Open/closed status and failure counts
4. **Move Validation Failures:** Rate of invalid moves from AI Service
5. **Random Fallback Usage:** Should be near zero in production

**Alert Thresholds:**

- Service availability < 95%: Investigate AI Service health
- Fallback usage > 20%: Check network or service degradation
- Circuit breaker open: Critical - AI Service down
- Invalid moves > 1%: AI Service logic issue

### Known Limitations

1. **Fatal Failures:** If all three tiers fail (extremely rare), game is abandoned
2. **Quality Degradation:** Local heuristics are weaker than trained AI
3. **No Retry Logic:** Service failures trigger immediate fallback (by design for responsiveness)
4. **Circuit Breaker State:** Shared across all games (not per-game isolation)

### Future Enhancements

1. **Adaptive Timeout:** Adjust timeout based on AI type and difficulty
2. **Quality Metrics:** Track move quality when using fallbacks
3. **Graceful Degradation:** Warn users when AI quality is degraded
4. **Service Pool:** Load balance across multiple AI Service instances
5. **Caching:** Cache positions for common opening/endgame patterns
6. **Recovery Action Support:** Recovery action is implemented (`recovery_slide`). AI must:
   - Enumerate recovery moves for recovery-eligible players (controls no stacks, has markers + buried rings; eligibility is independent of rings in hand)
   - Evaluate recovery moves alongside other actions in move selection
   - Treat recovery as **NOT** a "real action" for LPS tracking (RR-CANON-R172)
   - See `TODO.md` §3.1.1 and `RECOVERY_ACTION_IMPLEMENTATION_PLAN.md`

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
3.  ~~**In-place State Updates:** Refactor `GameEngine` or create a specialized `FastGameEngine` for MCTS to eliminate copying overhead.~~ ✅ **COMPLETE** (as of 2025-11-30) - Implemented via `MutableGameState` with `make_move()`/`unmake_move()` pattern in [`mutable_state.py`](ai-service/app/rules/mutable_state.py). All tactical AI engines (MinimaxAI, MCTSAI, DescentAI) now use this for efficient O(1) state manipulation during search.

### Phase 4: Production Readiness (Long-term)

1.  **Model Versioning:** Implement a system to manage and serve different model versions.
2.  **Async Inference:** Use an async task queue (e.g., Celery/Redis) for heavy AI computations.

---

## 4. Rules Completeness in AI Service

- **Status:** **Mostly Complete**
- **Implemented:** Ring placement, movement, capturing (including chains), line formation, Territory claiming, forced elimination, recovery action (`recovery_slide`), and victory conditions.
- **Not Yet Implemented:** See `KNOWN_ISSUES.md` and `TODO.md` for current gaps.
- **Simplifications:**
  - **Line Formation:** Automatically chooses to collapse all markers and eliminate from the largest stack (biasing against "Option 2").
  - **Territory Claim:** Automatically claims Territory without user choice nuances.

**Recommendation:** Update `GameEngine` to support branching for Line Formation and Territory Claim choices to fully match the game's strategic depth.

---

## 5. Heuristic Training & Behaviour Tuning

This section describes how we treat `HeuristicAI` as a _trainable, versioned
component_ rather than a one-off hand-tuned evaluator, and how that interacts
with RingRift’s specific rules (ring placement, chain captures, lines,
Territory, forced elimination, and dual victory conditions).

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
  - Dual victory conditions (eliminated rings vs Territory) and the
    S-invariant.

### 5.2 RingRift-specific signal design

`HeuristicAI` today already encodes many RingRift-specific features; training
should make these signals _quantitatively_ correct and well-balanced:

- **Ring placement & rings in hand**
  - Reward placements that respect the "no-dead-placement" rule (i.e. avoid
    creating stacks with no legal moves).
  - Penalise hoarding too many rings in hand once the board is sufficiently
    developed.
  - Prefer placements that increase future mobility and line/Territory
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
    - Avoid eliminating rings from critical stacks that anchor Territory
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
    - Auxiliary labels (e.g. expected eliminated-rings margin, Territory
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
    - Line and Territory parity fixtures (`Seed14Move35LineParity`,
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
       aggregates such as stack/Territory summaries).
     - Targets: teacher-based scalar values or outcome-based labels.
   - Use simple methods (ridge regression, small MLP with L2 regularisation)
     to obtain a _balanced_ `heuristic_v1_balanced` weight profile.
   - Enforce monotonicity constraints where appropriate (e.g., more enemy
     eliminated rings must not _improve_ our evaluation).

2. **Personas**
   - Define persona-specific deltas on top of the balanced profile:
     - **Aggressive:** upweight capture and elimination terms; slightly
       downweight Territory-safety.
     - **Territorial:** upweight Territory and line-formation terms; slightly
       downweight short-term elimination.
     - **Defensive:** upweight vulnerability and safety terms; penalise risky
       sacrifices.
   - Each persona becomes a named weight set (`heuristic_v1_aggressive`,
     `heuristic_v1_territorial`, etc.) referenced from `AIConfig`/
     `AIProfile` and selected by difficulty/queue configuration.

Trained weights are stored in a small `heuristic_weights.py` (Python) and
mirrored in TS (for improved fallback) where needed. Versioning is handled
explicitly (e.g. `heuristic_v1`, `heuristic_v2`) and documented in
`docs/supplementary/AI_IMPROVEMENT_BACKLOG.md` §6.4.

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
  basic Territory, vulnerability) and expose them via a light-weight
  evaluation function.
- Use the same trained weight profiles (or a coarser TS-only approximation) so
  fallback behaviour roughly matches Python `HeuristicAI` at the same
  difficulty/persona.
- Optionally expose a "remote AI" sandbox mode that calls `/ai/move` directly
  so designers can feel the exact production behaviour in the client.

As of November 2025 the first TS-side pieces are in place:

- `src/shared/engine/heuristicEvaluation.ts` implements
  `evaluateHeuristicState` (stack control/height, simple Territory, local
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

These steps are tracked in `docs/supplementary/AI_IMPROVEMENT_BACKLOG.md` (§1.3 and §6.4) and will
be rolled out incrementally to preserve determinism and test coverage.

### 5.6 Heuristic Training Sanity & Plateau Diagnostics

Heuristic training/evaluation is **not** part of the canonical rules surface, but it is still subject to explicit
sanity requirements so that optimisation loops (CMA-ES, genetic search, manual tuning) actually explore distinct
policies instead of accidentally re-evaluating the same behaviour.

- Canonical requirements and test strategy are documented in
  `docs/PYTHON_PARITY_REQUIREMENTS.md` (§4.6 “Heuristic Training & Evaluation Sanity”), which treats the Python
  heuristic harness as a parity-adjacent subsystem with its own invariants.
- The dedicated test suite `ai-service/tests/test_heuristic_training_evaluation.py` asserts two key properties:
  - `evaluate_fitness` (CMA-ES/GA fitness harness) distinguishes `heuristic_v1_balanced` from a clearly bad
    all-zero profile under identical conditions and reports non-zero weight-distance diagnostics via `debug_hook`.
  - `HeuristicAI.get_evaluation_breakdown(state)["total"]` on a non-trivial Square8 mid-game state _differs_ between
    the balanced and all-zero profiles, confirming that the scalar evaluation actually depends on the active weights.
- Together with the plateau/trace parity tests in `ai-service/tests/parity/` (especially
  `test_ai_plateau_progress.py` and `test_ts_seed_plateau_snapshot_parity.py`), these sanity checks provide a
  first line of defence against:
  - Wiring bugs where candidate/baseline weights are silently equal.
  - Training configurations that never meaningfully explore the heuristic weight space.
  - Misleading “flat plateau” interpretations caused by harness or evaluation issues rather than true optimisation
    saturation.
    These suites are classified and tracked in `tests/TEST_SUITE_PARITY_PLAN.md` (§4.1) as Python core/behaviour
    tests with a specific focus on heuristic-training and plateau diagnostics.

Any change to `HeuristicAI`, the shared fitness harness (`run_cmaes_optimization.py` / `run_genetic_heuristic_search.py`),
or the evaluation/debug instrumentation that would cause these tests to pass trivially (e.g. equal fitness or identical
evaluations for obviously different profiles) must be treated as a regression and investigated alongside the usual
rules parity suites.

Operationally, two complementary optimisation harnesses exercise these invariants:

- **CMA-ES tuner**: [`ai-service/scripts/run_cmaes_optimization.py`](ai-service/scripts/run_cmaes_optimization.py)
  runs structured CMA-ES searches over `HeuristicWeights` (typically around `heuristic_v1_balanced`), writing
  `run_meta.json`, generation summaries, checkpoints, and `best_weights.json` under `logs/cmaes/runs/<run_id>/`. It
  is orchestrated by `run_heuristic_experiment.py` and its results are integrated into
  `statistical_analysis_report.json` and `AI_TRAINING_ASSESSMENT_FINAL.md` (§10).
- **Genetic search harness**: [`ai-service/scripts/run_genetic_heuristic_search.py`](ai-service/scripts/run_genetic_heuristic_search.py)
  reuses the same `evaluate_fitness(...)` function to run an elitist GA over the same weight space, optionally using
  multi-start / evaluation pools from `app/training/eval_pools.py`. GA runs log `best_weights.json` under
  `logs/ga/runs/<run_id>/` with the same schema as CMA-ES so downstream tooling and reports can compare profiles
  directly.

Both harnesses are intentionally kept **behind test and documentation guardrails** rather than the production AI
API surface; they exist to explore the heuristic landscape and to diagnose plateau behaviour without compromising
the deterministic, orchestrator-aligned canonical rules SSoT (the written rules spec), or its primary executable
derivation in the shared TS engine/orchestrator.

#### 5.6.1 Multi-start Evaluation & State Pools

To mitigate overfitting to a single seed or opening, both CMA-ES and the GA harness support **multi-start evaluation**
driven by precomputed state pools:

- State pools are JSONL files of `GameState` snapshots indexed by `(BoardType, pool_id)` via
  [`ai-service/app/training/eval_pools.py`](ai-service/app/training/eval_pools.py).
- `evaluate_fitness(...)` in `run_cmaes_optimization.py` accepts:
  - `eval_mode="initial-only"` (default) to start games from the standard initial state, or
  - `eval_mode="multi-start"` plus `state_pool_id`, in which case each evaluation game samples its starting
    position from the configured pool for that board type.
- The genetic harness (`run_genetic_heuristic_search.py`) forwards these parameters verbatim, so both CMA-ES and GA
  can be run against the same evaluation pools for apples-to-apples comparisons.

A minimal smoke test for this path lives in
[`ai-service/tests/test_multi_start_evaluation.py`](ai-service/tests/test_multi_start_evaluation.py). It:

- Builds a tiny Square8 pool using `RingRiftEnv` and writes it via `model_dump_json` to a temporary JSONL file.
- Patches `eval_pools.POOL_PATHS` to point `(BoardType.SQUARE8, "test")` at that file.
- Calls `evaluate_fitness` with `eval_mode="multi-start"` / `state_pool_id="test"` and asserts that
  baseline-vs-baseline fitness is well-behaved in `[0.0, 1.0]`.

This ensures the wiring between CLI flags (`--eval-mode`, `--state-pool-id`), the pool loader, and the underlying
fitness loop remains deterministic and test-covered as we evolve training configs.
