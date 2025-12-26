# AI Tier Training and Promotion Pipeline (H-AI-9 – Square-8 2-Player)

> Status: partially implemented. Describes the intended training, evaluation, and promotion loop for Square-8 2-player tiers D1–D10 (D11 reserved for internal benchmarks).  
> Core building blocks (tier eval configs, perf budgets, curriculum harness) and the combined gating CLI are now implemented; remaining orchestration is incremental wiring on top.
> Implementation reference: `ai-service/docs/training/TIER_PROMOTION_SYSTEM.md` (module/API overview).

This document specifies a concrete, repeatable pipeline for training, evaluating, and promoting AI models for the Square-8 2-player difficulty ladder tiers D1–D10 (with D11 treated as internal-only).

### Current runnable pieces (2025-12-25)

- **Tier training + gating:** `ai-service/scripts/run_tier_training_pipeline.py` (D2–D10) + `ai-service/scripts/run_full_tier_gating.py` (wraps `run_tier_gate.py` + perf benchmark). Defaults live in `ai-service/config/tier_training_pipeline.square8_2p.json` and runs emit `training_report.json` + `gate_report.json`.
- **Promotion tracking:** `ai-service/scripts/apply_tier_promotion_plan.py` records candidates in `ai-service/config/tier_candidate_registry.square8_2p.json` and emits `promotion_summary.json` + `promotion_patch_guide.txt`. Runtime ladder overrides (if used) live in `ai-service/data/ladder_runtime_overrides.json` and are applied by `get_effective_ladder_config`.
- **NN baseline demos:** `ai-service/scripts/run_nn_training_baseline.py` with accompanying smokes `ai-service/tests/test_nn_training_baseline_demo.py` and `ai-service/tests/test_neural_net_ai_demo.py` run today but are **demo-scale only** (not production checkpoints). Treat outputs as experimental candidates until slotted into the gate.
- **Distributed self-play soak:** `ai-service/scripts/run_distributed_selfplay_soak.py` is wired for Square-8 2p distributed self-play; use it for dataset generation ahead of tier gates. Promotion criteria below still apply.

The design is constrained to existing infrastructure:

- Ladder definitions in [`python.LadderTierConfig`](../../ai-service/app/config/ladder_config.py:26) and [`python.get_effective_ladder_config`](../../ai-service/app/config/ladder_config.py:1249) (runtime overrides live in `ai-service/data/ladder_runtime_overrides.json`).
- Candidate registry in [`python.tier_promotion_registry`](../../ai-service/app/training/tier_promotion_registry.py:1) with default file `ai-service/config/tier_candidate_registry.square8_2p.json`.
- Tier training defaults in `ai-service/config/tier_training_pipeline.square8_2p.json` (used by `scripts/run_tier_training_pipeline.py`).
- Canonical training environment [`python.RingRiftEnv`](../../ai-service/app/training/env.py:274) via [`python.make_env`](../../ai-service/app/training/env.py:221) and [`python.TrainingEnvConfig`](../../ai-service/app/training/env.py:171).
- Tier evaluation configs in [`python.TierEvaluationConfig`](../../ai-service/app/training/tier_eval_config.py:37) and runner [`python.run_tier_evaluation`](../../ai-service/app/training/tier_eval_runner.py:319).
- Difficulty-tier gating CLI [`run_tier_gate.py`](../../ai-service/scripts/run_tier_gate.py:1).
- Perf budgets in [`python.TierPerfBudget`](../../ai-service/app/config/perf_budgets.py:20) and benchmark helper [`python.run_tier_perf_benchmark`](../../ai-service/app/training/tier_perf_benchmark.py:84).
- Human calibration experiments in [`AI_HUMAN_CALIBRATION_GUIDE.md`](AI_HUMAN_CALIBRATION_GUIDE.md:1) and calibration telemetry in [`difficultyCalibrationEvents.ts`](../../src/shared/telemetry/difficultyCalibrationEvents.ts:1) and [`typescript.MetricsService.recordDifficultyCalibrationEvent`](../../src/server/services/MetricsService.ts:1292).

---

## 1. Scope and invariants

### 1.1 Scope

This pipeline covers only:

- Board type: `square8` (8×8, compact ruleset).
- Players: 2.
- Ladder difficulties: D1–D10 (D11 reserved for internal benchmarks; not exposed via the public API).

Future work can extend the same patterns to other boards and player counts using the multi-board tier configs already in [`python.tier_eval_config`](../../ai-service/app/training/tier_eval_config.py:196) and [`python.ladder_config`](../../ai-service/app/config/ladder_config.py:121).

> **Note (2025-12-17):** Minimax is disabled for square19 and hexagonal boards. The search space on these larger boards causes minimax to take tens of minutes per move. D4/D5 tiers on square19 and hex currently use Descent + NN instead (see `ai-service/app/config/ladder_config.py`).

### 1.2 Non-goals

- No rules or move-semantics changes; all engines must stay aligned with the TS shared rules engine and parity contracts.
- No changes to the public HTTP API of the AI service in this task.
- No new on-disk formats beyond small JSON/Markdown artefacts and model checkpoints under existing `models/` and training log directories.

### 1.3 Tier invariants

For Square-8 2-player, the canonical ladder currently defines (simplified):

| Tier | Difficulty | Ladder ai_type | Ladder model_id      | heuristic_profile_id | Intended strength            |
| ---- | ---------- | -------------- | -------------------- | -------------------- | ---------------------------- |
| D1   | 1          | RANDOM         | —                    | —                    | Entry / baseline             |
| D2   | 2          | HEURISTIC      | heuristic_v1_weak    | heuristic_v1_weak    | Casual / learning            |
| D3   | 3          | HEURISTIC      | heuristic_v1_sq8_2p  | heuristic_v1_sq8_2p  | Lower-mid (tuned heuristic)  |
| D4   | 4          | MINIMAX        | nnue_square8_2p      | heuristic_v1_sq8_2p  | Mid (NNUE minimax)           |
| D5   | 5          | MINIMAX        | nnue_square8_2p      | heuristic_v1_sq8_2p  | Upper-mid (NNUE minimax)     |
| D6   | 6          | DESCENT        | ringrift_best_sq8_2p | heuristic_v1_sq8_2p  | High (neural Descent)        |
| D7   | 7          | MCTS           | —                    | heuristic_v1_sq8_2p  | Expert (heuristic-only MCTS) |
| D8   | 8          | MCTS           | ringrift_best_sq8_2p | heuristic_v1_sq8_2p  | Strong expert (neural MCTS)  |
| D9   | 9          | GUMBEL_MCTS    | ringrift_best_sq8_2p | heuristic_v1_sq8_2p  | Master (Gumbel MCTS)         |
| D10  | 10         | GUMBEL_MCTS    | ringrift_best_sq8_2p | heuristic_v1_sq8_2p  | Grandmaster (Gumbel MCTS)    |
| D11  | 11         | GUMBEL_MCTS    | ringrift_best_sq8_2p | heuristic_v1_sq8_2p  | Internal-only stress tier    |

(from [`python._build_default_square8_two_player_configs`](../../ai-service/app/config/ladder_config.py:47))

Pipeline invariants:

- **Ordering:** D1 < D2 < … < D10 in strength on Square-8 2-player (D11 is internal-only).
- **Non-regression vs previous production version of each tier:** new Dn must not be substantially weaker than the previous Dn against shared baselines.
- **Perf budgets for D3–D8:** average and p95 move latencies must respect [`AI_TIER_PERF_BUDGETS.md`](AI_TIER_PERF_BUDGETS.md:1) and [`python.TierPerfBudget`](../../ai-service/app/config/perf_budgets.py:20).

---

## 2. Building blocks

### 2.1 Environment and rules

- Training and evaluation always use [`python.RingRiftEnv`](../../ai-service/app/training/env.py:274) constructed via [`python.make_env`](../../ai-service/app/training/env.py:221) with:
  - `board_type=BoardType.SQUARE8`.
  - `num_players=2`.
  - `reward_mode="terminal"` for gating and perf benchmarks.
- `max_moves` defaults to 200 for Square-8 2-player, consistent with [`python.make_env`](../../ai-service/app/training/env.py:221) and [`python.get_theoretical_max_moves`](../../ai-service/app/training/env.py:55).
- Rules semantics are provided by [`python.DefaultRulesEngine`](../../ai-service/app/rules/default_engine.py:1) and [`python.GameEngine`](../../ai-service/app/game_engine/__init__.py:1); both are already wired through the env.

### 2.2 Difficulty ladder and online integration

The FastAPI service endpoint [`python.get_ai_move`](../../ai-service/app/main.py:236):

- Uses [`python._get_difficulty_profile`](../../ai-service/app/main.py:1048) for the global 1–10 mapping.
- When possible, refines settings via board-aware [`python.get_effective_ladder_config`](../../ai-service/app/config/ladder_config.py:1249) for `(difficulty, board_type, num_players)` (runtime overrides load from `ai-service/data/ladder_runtime_overrides.json`).
- Constructs [`python.AIConfig`](../../ai-service/app/models/core.py:419) with:
  - `difficulty`, `randomness`, `think_time` from the ladder or profile.
  - `heuristic_profile_id` from ladder or difficulty profile.
  - `nn_model_id` from ladder `model_id` when `ai_type` is MCTS or DESCENT.

Neural-net-backed AIs (Minimax/MCTS/Descent) resolve `nn_model_id` to a checkpoint under `ai-service/models` via [`python.NeuralNetAI`](../../ai-service/app/ai/neural_net/__init__.py:1049).

Heuristic tiers use profiles in [`python.HEURISTIC_WEIGHT_PROFILES`](../../ai-service/app/ai/heuristic_weights.py:1). If `RINGRIFT_TRAINED_HEURISTIC_PROFILES` points at a JSON bundle, trained profiles are loaded at startup and merged into the registry.

### 2.3 Training components

We reuse existing training infrastructure described in [`AI_TRAINING_AND_DATASETS.md`](AI_TRAINING_AND_DATASETS.md:1):

- Self-play NPZ datasets via [`python.generate_dataset`](../../ai-service/app/training/generate_data.py:1).
- Territory / combined-margin JSONL datasets via [`python.generate_territory_dataset`](../../ai-service/app/training/generate_territory_dataset.py:1).
- Heuristic-weight training via [`python.train_heuristic_weights`](../../ai-service/app/training/train_heuristic_weights.py:1).
- Neural net training via [`python.train_model`](../../ai-service/app/training/train.py:1171) and [`python.TrainConfig`](../../ai-service/app/config/training_config.py:176).
- Model versioning via [`python.ModelVersionManager`](../../ai-service/app/training/model_versioning.py:311) and helpers like [`python.save_model_checkpoint`](../../ai-service/app/training/model_versioning.py:865).
- Curriculum-based self-play and promotion via [`python.CurriculumConfig`](../../ai-service/app/training/curriculum.py:152) and [`python.CurriculumTrainer`](../../ai-service/app/training/curriculum.py:247).
- Auto-tournaments and Elo-based comparison via [`python.AutoTournamentPipeline`](../../ai-service/app/training/auto_tournament.py:327).

### 2.4 Evaluation, gating, and perf

- Tier evaluation configs: [`python.TIER_EVAL_CONFIGS`](../../ai-service/app/training/tier_eval_config.py:53) with [`python.TierEvaluationConfig`](../../ai-service/app/training/tier_eval_config.py:37), keyed by `"D1"`–`"D11"`.
- Runner: [`python.run_tier_evaluation`](../../ai-service/app/training/tier_eval_runner.py:319) returning [`python.TierEvaluationResult`](../../ai-service/app/training/tier_eval_runner.py:74).
- Difficulty-tier gate CLI: [`run_tier_gate.py`](../../ai-service/scripts/run_tier_gate.py:1), difficulty mode via [`python._run_difficulty_mode`](../../ai-service/scripts/run_tier_gate.py:252).
  - Emits evaluation JSON via `--output-json`.
  - Emits promotion descriptor via `--promotion-plan-out`.
- Perf budgets:
  - [`python.TIER_PERF_BUDGETS`](../../ai-service/app/config/perf_budgets.py:101) with entries `"D3_SQ8_2P"`–`"D8_SQ8_2P"`.
  - Benchmarks via [`python.run_tier_perf_benchmark`](../../ai-service/app/training/tier_perf_benchmark.py:84) and CLI [`run_tier_perf_benchmark.py`](../../ai-service/scripts/run_tier_perf_benchmark.py:1).

### 2.5 Human calibration and telemetry

- Human calibration experiments: [`AI_HUMAN_CALIBRATION_GUIDE.md`](AI_HUMAN_CALIBRATION_GUIDE.md:1) (Templates A/B/C).
- Telemetry schema: [`difficultyCalibrationEvents.ts`](../../src/shared/telemetry/difficultyCalibrationEvents.ts:1) defining `DifficultyCalibrationEventPayload`.
- Client helper: [`difficultyCalibrationTelemetry.ts`](../../src/client/utils/difficultyCalibrationTelemetry.ts:1).
- Server route: [`difficultyCalibrationTelemetry.ts`](../../src/server/routes/difficultyCalibrationTelemetry.ts:1).
- Metrics sink: [`typescript.MetricsService.recordDifficultyCalibrationEvent`](../../src/server/services/MetricsService.ts:1292), exposing `ringrift_difficulty_calibration_events_total`.

---

## 3. Candidate model shape per tier

We standardise what a _candidate_ is at each tier and how it will be wired into the ladder.

### 3.1 Summary table

| Tier | Engine family (Square-8 2p) | Candidate kind                     | Primary artefact(s)                                                                               | Ladder fields touched on promotion                             |
| ---- | --------------------------- | ---------------------------------- | ------------------------------------------------------------------------------------------------- | -------------------------------------------------------------- |
| D1   | RANDOM                      | None (baseline)                    | —                                                                                                 | —                                                              |
| D2   | HEURISTIC                   | Heuristic profile                  | New entry in [`python.HEURISTIC_WEIGHT_PROFILES`](../../ai-service/app/ai/heuristic_weights.py:1) | `heuristic_profile_id` (and optionally `model_id`)             |
| D3   | HEURISTIC (tuned)           | Heuristic profile                  | New entry in [`python.HEURISTIC_WEIGHT_PROFILES`](../../ai-service/app/ai/heuristic_weights.py:1) | `heuristic_profile_id` (and optionally `model_id`)             |
| D4   | MINIMAX (NNUE)              | NNUE checkpoint + search persona   | Versioned NNUE checkpoint (`models/nnue/<nn_model_id>.pt`) and/or minimax config                  | `model_id` (mapped to `nn_model_id`)                           |
| D5   | MINIMAX (NNUE)              | NNUE checkpoint + search persona   | Versioned NNUE checkpoint (`models/nnue/<nn_model_id>.pt`) and/or minimax config                  | `model_id` (mapped to `nn_model_id`)                           |
| D6   | DESCENT (neural)            | Neural checkpoint                  | Versioned NN checkpoint (`models/<nn_model_id>.pth`)                                              | `model_id` (mapped to `nn_model_id`)                           |
| D7   | MCTS (heuristic-only)       | Search persona + heuristic profile | MCTS config snapshot (search persona) + heuristic profile                                         | `model_id` (persona tag) and optionally `heuristic_profile_id` |
| D8   | MCTS (neural)               | Neural checkpoint + MCTS config    | Versioned NN checkpoint (`models/<nn_model_id>.pth`) and MCTS config                              | `model_id` (mapped to `nn_model_id`)                           |
| D9   | GUMBEL_MCTS (neural)        | Neural checkpoint + Gumbel config  | Versioned NN checkpoint (`models/<nn_model_id>.pth`) and Gumbel MCTS config                       | `model_id` (mapped to `nn_model_id`)                           |
| D10  | GUMBEL_MCTS (neural)        | Neural checkpoint + Gumbel config  | Versioned NN checkpoint (`models/<nn_model_id>.pth`) and Gumbel MCTS config                       | `model_id` (mapped to `nn_model_id`)                           |
| D11  | GUMBEL_MCTS (internal)      | Neural checkpoint + Gumbel config  | Same as D9/D10; reserved for internal benchmarks                                                  | `model_id` (mapped to `nn_model_id`)                           |

Rules:

- All **neural** candidates are identified by a logical `nn_model_id` string and a versioned checkpoint saved via [`python.save_model_checkpoint`](../../ai-service/app/training/model_versioning.py:865).
- All **heuristic** candidates are identified by a `heuristic_profile_id` key in [`python.HEURISTIC_WEIGHT_PROFILES`](../../ai-service/app/ai/heuristic_weights.py:1).
- Search-persona changes must map to distinct ladder `model_id` values (for example `v2-minimax-4`).

---

## 4. Training pipeline per tier (Square-8 2-player)

### 4.1 Shared training conventions

- Board: `BoardType.SQUARE8`, `num_players=2` in [`python.TrainingEnvConfig`](../../ai-service/app/training/env.py:171).
- Reward: `"terminal"` for gating-oriented training (self-play for value/policy can still use shaping when appropriate).
- Seeding: follow the RNG model from [`AI_TRAINING_AND_DATASETS.md`](AI_TRAINING_AND_DATASETS.md:284) and [`python.RingRiftEnv.reset`](../../ai-service/app/training/env.py:371).
- Data provenance: all training / eval datasets generated through the Python rules host described in [`AI_TRAINING_AND_DATASETS.md`](AI_TRAINING_AND_DATASETS.md:19), and only from DBs marked canonical in `ai-service/TRAINING_DATA_REGISTRY.md` (Square-8 2-player uses `ai-service/data/games/canonical_square8_2p.db` by default).
- Tier pipeline defaults: `ai-service/config/tier_training_pipeline.square8_2p.json` is the default config for [`run_tier_training_pipeline.py`](../../ai-service/scripts/run_tier_training_pipeline.py:1) (override with `--config`).
- CMA-ES workers: `cmaes_workers` in the tier config or `RINGRIFT_CMAES_WORKERS` env var.

### 4.2 Tier training modes (summary)

| Tier | Ladder AI type         | `run_tier_training_pipeline.py` mode | Primary artifact                                                   |
| ---- | ---------------------- | ------------------------------------ | ------------------------------------------------------------------ |
| D1   | RANDOM                 | —                                    | —                                                                  |
| D2   | HEURISTIC              | `heuristic_cmaes`                    | Trained heuristic profile (`data/trained_heuristic_profiles.json`) |
| D3   | HEURISTIC (tuned)      | `heuristic_cmaes`                    | Trained heuristic profile                                          |
| D4   | MINIMAX (NNUE)         | `search_persona`                     | Minimax persona + NNUE checkpoint                                  |
| D5   | MINIMAX (NNUE)         | `search_persona`                     | Minimax persona + NNUE checkpoint                                  |
| D6   | DESCENT (neural)       | `neural`                             | NN checkpoint                                                      |
| D7   | MCTS (heuristic-only)  | `search_persona`                     | MCTS persona JSON + heuristic profile                              |
| D8   | MCTS (neural)          | `neural`                             | NN checkpoint + MCTS config                                        |
| D9   | GUMBEL_MCTS (neural)   | `neural`                             | NN checkpoint + Gumbel config                                      |
| D10  | GUMBEL_MCTS (neural)   | `neural`                             | NN checkpoint + Gumbel config                                      |
| D11  | GUMBEL_MCTS (internal) | —                                    | Internal-only                                                      |

> **Note:** `config/tier_training_pipeline.square8_2p.json` lists D5 as `search_persona` to match the minimax ladder mapping.

### 4.3 D2 – heuristic baseline

**Model type**

- Pure heuristic AI using a Square-8 2-player profile in [`python.HEURISTIC_WEIGHT_PROFILES`](../../ai-service/app/ai/heuristic_weights.py:1), currently `heuristic_v1_sq8_2p`.

**Training data & method**

- Primary dataset: territory / combined-margin JSONL as in [`python.generate_territory_dataset`](../../ai-service/app/training/generate_territory_dataset.py:1).
- Optional NPZ self-play dataset via [`python.generate_dataset`](../../ai-service/app/training/generate_data.py:1).
- Run via `run_tier_training_pipeline.py --tier D2` (CMA-ES). Distributed CMA-ES modes are selected by the tier config and `RINGRIFT_CMAES_WORKERS`.

**Outputs & versioning**

- CMA-ES updates `data/trained_heuristic_profiles.json`; set `RINGRIFT_TRAINED_HEURISTIC_PROFILES` to load it (auto-set in `ai-service/run.sh` if the file exists).
- The candidate id recorded in `training_report.json` is used for gating and promotion.

### 4.4 D3 – tuned heuristic tier

**Model type**

- Heuristic AI (`AIType.HEURISTIC`) using CMA-ES tuned weights.

**Training data & method**

- Use CMA-ES (same pipeline as D2) to produce a D3-specific heuristic profile.

**Outputs & versioning**

- Candidate includes `heuristic_profile_id` (and optionally a `model_id` alias).

### 4.5 D4 – mid minimax tier (NNUE)

**Model type**

- Minimax with NNUE evaluation; default checkpoint `nnue_square8_2p`.

**Training data & method**

- `run_tier_training_pipeline.py --tier D4` produces a `search_persona.json` snapshot for minimax settings.
- Optional: train a new NNUE checkpoint and update `model_id` to the candidate `nn_model_id`.

**Outputs & versioning**

- Persona tag (for search config) and/or `nn_model_id` checkpoint.

### 4.6 D5 – upper-mid minimax tier (NNUE)

**Model type**

- Minimax with NNUE evaluation; default checkpoint `nnue_square8_2p`.

**Training data & method**

- `run_tier_training_pipeline.py --tier D5` produces a `search_persona.json` snapshot for minimax settings.
- Optional: train a new NNUE checkpoint and update `model_id` to the candidate `nn_model_id`.

### 4.7 D6 – high Descent tier (neural)

**Model type**

- Descent (`AIType.DESCENT`) with larger budget and stronger NN.

**Training data & method**

- Neural training via `run_tier_training_pipeline.py --tier D6` (GPU required for full runs).
- Optional curriculum loops using [`python.CurriculumTrainer`](../../ai-service/app/training/curriculum.py:247).

**Outputs & versioning**

- `nn_model_id = "sq8_d6_vN"` (or similar) and updated ladder entry.

### 4.8 D7 – heuristic-only MCTS tier

**Model type**

- MCTS without neural guidance; uses heuristic evaluation only.

**Training data & method**

- `run_tier_training_pipeline.py --tier D7` generates a `search_persona.json` with MCTS parameters.
- Heuristic weights can be shared with D2/D3 or tuned separately.

### 4.9 D8 – neural MCTS tier

**Model type**

- MCTS with neural guidance (NN checkpoint + MCTS params).

**Training data & method**

- Neural training via `run_tier_training_pipeline.py --tier D8`.
- Capture MCTS settings in a persona tag; candidate uses `model_id` as `nn_model_id`.

### 4.10 D9/D10 – Gumbel MCTS tiers (neural)

**Model type**

- Gumbel MCTS with neural guidance; strongest search budgets.

**Training data & method**

- Neural training via `run_tier_training_pipeline.py --tier D9/D10`.
- No perf budgets today (see Section 6); rely on tier gating plus regression checks.

### 4.11 Baseline NN demo experiment (A2, non-production)

- For a cheap, smoke-testable Square-8 NN baseline, use the dedicated
  Python CLI:

  ```bash
  cd ai-service
  PYTHONPATH=. python scripts/run_nn_training_baseline.py \
    --board square8 \
    --num-players 2 \
    --run-dir /tmp/nn_baseline_demo \
    --demo \
    --seed 123
  ```

- This runs a tiny training loop on a synthetic dummy dataset (no real
  self-play data) and writes `nn_training_report.json` into `--run-dir`
  with:
  - `board = "square8"`, `num_players = 2`
  - `mode = "demo"`
  - `model_id` (logical NN id)
  - a small `training_params` block and stubbed `metrics`.

- The demo run is intended only for wiring and CI smoke tests; it is
  **not** a production training path and must not be used to update
  ladder tiers or canonical models.

- For sandbox evaluation, callers can route traffic to the experimental
  `AIType.NEURAL_DEMO` engine in the Python AI service:
  - Set `AI_ENGINE_NEURAL_DEMO_ENABLED=1` in the AI service environment.
  - Construct `AIConfig` with `ai_type=NEURAL_DEMO` and an appropriate
    `nn_model_id` (for example the `model_id` emitted by the baseline
    script).
  - The canonical difficulty ladder (D1–D10) does **not** select
    `NEURAL_DEMO`; this path is reserved for private queues and
    experiments.

---

## 5. Automated evaluation and gating

### 5.1 Evaluation profiles per tier

[`python.TIER_EVAL_CONFIGS`](../../ai-service/app/training/tier_eval_config.py:53) includes:

Square-8 2-player tiers D1–D11 plus cross-board configs. Opponent mixes and roles live in the config; summary of Square-8 2-player thresholds:

| Tier           | num_games | min_win_rate_vs_baseline | max_regression_vs_previous_tier |
| -------------- | --------- | ------------------------ | ------------------------------- |
| D1             | 100       | —                        | —                               |
| D2             | 200       | 0.60                     | 0.05                            |
| D3             | 200       | 0.55                     | 0.05                            |
| D4             | 300       | 0.68                     | 0.05                            |
| D5             | 300       | 0.60                     | 0.05                            |
| D6             | 400       | 0.72                     | 0.05                            |
| D7             | 400       | 0.65                     | 0.05                            |
| D8             | 400       | 0.75                     | 0.05                            |
| D9             | 400       | 0.75                     | 0.05                            |
| D10            | 400       | 0.75                     | 0.05                            |
| D11 (internal) | 400       | 0.75                     | 0.05                            |

`min_win_rate_vs_baseline` is applied to the Wilson lower bound at `promotion_confidence=0.95`.

These are consumed by:

- [`run_tier_evaluation.py`](../../ai-service/scripts/run_tier_evaluation.py:1) for ad-hoc tier eval.
- [`run_tier_gate.py`](../../ai-service/scripts/run_tier_gate.py:1) for gating.

### 5.2 Gating & promotion rules

For each tier T ∈ {D2–D10} (D1 is a baseline, D11 is internal), the automated gate is:

1. **Preferred:** run the full wrapper after training completes:

   ```bash
   cd ai-service
   python scripts/run_full_tier_gating.py \
     --tier D6 \
     --candidate-id CANDIDATE_D6_ID \
     --run-dir logs/tier_gate/D6_candidate
   ```

   The wrapper reads `training_report.json`, enforces candidate artefacts, runs the tier evaluation, and executes perf checks when a budget exists.

2. The wrapper invokes [`run_tier_gate.py`](../../ai-service/scripts/run_tier_gate.py:1), whose [`python._run_difficulty_mode`](../../ai-service/scripts/run_tier_gate.py:252) will:
   - Look up the TierEvaluationConfig via [`python.get_tier_config`](../../ai-service/app/training/tier_eval_config.py:436).
   - Resolve current production ladder entry via [`python.get_ladder_tier_config`](../../ai-service/app/config/ladder_config.py:279) to find `current_model_id` (runtime overrides are not applied here).
   - Call [`python.run_tier_evaluation`](../../ai-service/app/training/tier_eval_runner.py:319).
   - Print a human summary and emit:
     - Full TierEvaluationResult JSON (`tier_eval_result.json` in the wrapper run dir).
     - A promotion plan JSON (`promotion_plan.json`) with keys:
       - `tier`, `board_type`, `num_players`.
       - `current_model_id`, `candidate_model_id`.
       - `decision`: `"promote"` or `"reject"`.
       - `reason`: metrics summary (`win_rate_vs_baseline`, `win_rate_vs_previous_tier`, `overall_pass`).

3. Decision semantics:
   - `overall_pass` must be true.
   - Criteria:
     - `win_rate_vs_baseline >= min_win_rate_vs_baseline` when applicable.
     - `win_rate_vs_previous_tier >= 0.5 - max_regression_vs_previous_tier` when applicable.

Process-level expectations added by H-AI-9:

- **Non-regression vs previous production version of the same tier:**
  - When available, compare candidate vs previous production T (e.g. using `evaluate_ai_models.py`) and require that the candidate is not clearly worse (for example no extreme <45% win rate).
- **Tier ordering:**
  - After promotion of any tier, run a brief cross-tier smoke evaluation (e.g. D2 vs D4, D4 vs D6, D6 vs D8) and sanity-check that tier strengths remain ordered as intended.

### 5.3 Ladder integration

When a candidate passes automated gating (and perf, see below) and is approved:

- Record the promotion plan in the candidate registry:

  ```bash
  cd ai-service
  python scripts/apply_tier_promotion_plan.py \
    --plan-path logs/tier_gate/D6_candidate/promotion_plan.json
  ```

  This updates `ai-service/config/tier_candidate_registry.square8_2p.json` and emits a `promotion_patch_guide.txt` next to the plan.

- Apply the ladder update:
  - **Runtime override:** write/update `ai-service/data/ladder_runtime_overrides.json` via `update_tier_model` / `update_tier_heuristic_profile` (no code change, immediate effect).
  - **Permanent change:** update [`python._build_default_square8_two_player_configs`](../../ai-service/app/config/ladder_config.py:47) and commit the change.
  - For heuristic tiers, ensure the promoted profile exists in `data/trained_heuristic_profiles.json` and is wired via `RINGRIFT_TRAINED_HEURISTIC_PROFILES`.

Promotion plans written by `run_tier_gate.py` should be archived under something like `ai-service/data/promotions/square8_2p/` for traceability.

---

## 6. Performance constraints in the loop

### 6.1 Running perf benchmarks

For D3–D8, perf budgets are defined in [`python._build_square8_two_player_budgets`](../../ai-service/app/config/perf_budgets.py:51) and described in [`AI_TIER_PERF_BUDGETS.md`](AI_TIER_PERF_BUDGETS.md:1).

For each candidate that _passes_ the tier gate, the pipeline must also run:

```bash
cd ai-service
python scripts/run_tier_perf_benchmark.py \
  --tier D6 \
  --num-games 4 \
  --moves-per-game 16 \
  --seed 1 \
  --output-json logs/tier_perf/D6_candidate.json
```

[`python.run_tier_perf_benchmark`](../../ai-service/app/training/tier_perf_benchmark.py:84) returns:

- `average_ms`, `p95_ms`.
- Associated `TierPerfBudget` with `max_avg_move_ms`, `max_p95_move_ms`.

The CLI evaluates budgets via [`python._eval_budget`](../../ai-service/scripts/run_tier_perf_benchmark.py:95):

- `within_avg = average_ms <= max_avg_move_ms`.
- `within_p95 = p95_ms <= max_p95_move_ms`.
- `overall_pass = within_avg and within_p95`.

H-AI-9 treats `overall_pass` for perf as **required** for D3–D8 promotions unless there is an explicit, documented adjustment of budgets.

In practice, instead of invoking the perf benchmark separately, the
combined wrapper [`run_full_tier_gating.py`](../../ai-service/scripts/run_full_tier_gating.py:1) can be used to run both the difficulty-tier gate and perf check in one step:

```bash
cd ai-service
python scripts/run_full_tier_gating.py \
  --tier D6 \
  --candidate-id CANDIDATE_D6_ID \
  --run-dir logs/tier_gate/D6_candidate
```

This writes:

- `tier_eval_result.json` and `promotion_plan.json` – difficulty-tier gate outputs.
- `tier_perf_report.json` – perf benchmark metrics and budget evaluation (when a budget exists).
- `gate_report.json` – combined summary suitable for CI and manual review.

### 6.2 Trade-offs and escalation

If a candidate is stronger but exceeds perf budgets:

1. **Preferred path:** adjust search parameters (depth, iteration caps, or reroll heuristics) to bring perf under budget, then re-run tier evaluation and gating.
2. **If regression is minor and strength gain is large:**
   - Optionally propose a new `think_time_ms` ladder value and recompute perf budgets in [`perf_budgets.py`](../../ai-service/app/config/perf_budgets.py:51) via a coordinated change.
   - This must be justified by product/UX decisions (for example “we accept D8 being slower for significantly more depth”).
3. **Otherwise:** reject the candidate and keep current production tier.

---

## 7. Human calibration integration

### 7.1 Mapping calibration experiments to tiers

[`AI_HUMAN_CALIBRATION_GUIDE.md`](AI_HUMAN_CALIBRATION_GUIDE.md:1) defines three experiment templates for Square-8 2-player:

- Template A – New-player quick check (D2/D4, 3-game sets).
- Template B – Intermediate validation (D4/D6, 5-game blocks).
- Template C – Strong-player session (D6/D8, 10+ games).

H-AI-9 integrates these as follows:

- **D2:** Template A with true beginners; ensure at least one win in 3–5 games is common and D2 is not overwhelmingly strong.
- **D4:** Template A (when D2 trivial) and Template B for intermediates; confirm 30–70% win-rate bands and “about right” perceived difficulty.
- **D6:** Template B and C for strong players; aim for 40–60% win rates among strong testers.
- **D8:** Template C; even strong testers should struggle to exceed ~60% win rate.

### 7.2 Telemetry and aggregation

Calibration-mode games in the client should:

- Use Square-8, 2 players, **anchor tiers** D2/D4/D6/D8 by default (the public ladder is D1–D10); expand only when a specific tier is under study, as recommended in [`AI_HUMAN_CALIBRATION_GUIDE.md`](AI_HUMAN_CALIBRATION_GUIDE.md:26).
- Attach `isCalibrationOptIn=true` in the payload to [`difficultyCalibrationEvents.ts`](../../src/shared/telemetry/difficultyCalibrationEvents.ts:1).

During such games, the client sends:

- `difficulty_calibration_game_started`.
- `difficulty_calibration_game_completed` with:
  - `result`: win / loss / draw / abandoned.
  - `movesPlayed`.
  - `perceivedDifficulty` (1–5).

On the server, [`typescript.MetricsService.recordDifficultyCalibrationEvent`](../../src/server/services/MetricsService.ts:1292) increments a counter labelled by `type`, `board_type`, `num_players`, `difficulty`, and `result`, but **only when `isCalibrationOptIn=true`**.

For H-AI-9, a lightweight aggregation (outside this doc) should periodically compute, per `(board_type, num_players, difficulty)`:

- Number of completed calibration games.
- Human win rate (fraction of `result="win"`).
- Distribution of `perceivedDifficulty`.

This can use Prometheus queries or downstream metric export.

### 7.3 Decision rules using human data

Human calibration is **advisory but important**:

- **Promotion only on automated + perf success:**
  - A candidate must pass automated gate and perf first; human data refines whether we roll out broadly or treat as experimental.
- **Adjusting tiers based on human data:**
  - If automated gate passes but humans consistently find a tier “far too easy” with >80% win rate for the target profile, consider:
    - Promoting the candidate to a higher logical difficulty (e.g. D4→D5) while leaving the old D4 intact.
    - Tightening automated gate thresholds for future candidates.
  - If automated gate passes but humans find a tier “far too hard” with <20% win rate and many 4–5 ratings, consider:
    - Weakening the tier (adjust ladder randomness or search budget, then re-run gates).
    - Hiding the tier from default UX or labelling it as “experimental” or “expert only”.
- **Avoid overfitting to small samples:**
  - Treat sample sizes <30 calibration games per `(player_profile, tier)` as inconclusive; use them only to detect gross issues (e.g. AI never loses).
  - Require multiple waves of data before permanently repositioning tiers or changing displayed difficulty descriptions.

---

## 8. Orchestration and automation plan

### 8.1 High-level pipeline stages

For each tier T ∈ {D2–D10}, the pipeline is:

1. **Train candidate for tier T:**
   - Run the appropriate training jobs (heuristic or NN) for Square-8 2-player using [`python.train_heuristic_weights`](../../ai-service/app/training/train_heuristic_weights.py:1), [`python.train_model`](../../ai-service/app/training/train.py:1171), or [`python.CurriculumTrainer`](../../ai-service/app/training/curriculum.py:247).
   - Produce candidate artefacts and assign a candidate id `CANDIDATE_T_ID`.

2. **Run automated evaluation and gating:**
   - `python scripts/run_full_tier_gating.py --tier T --candidate-id CANDIDATE_T_ID --run-dir RUN_DIR`.
   - Inspect `tier_eval_result.json`, `promotion_plan.json`, and `gate_report.json`.

3. **Run perf benchmarks (where configured):**
   - For D3–D8, run `python scripts/run_tier_perf_benchmark.py --tier T ...` (or rely on the wrapper).
   - Ensure `overall_pass=true`.

4. **Optional cross-tier sanity evaluation:**
   - Use [`run_tournament.py`](../../ai-service/scripts/run_tournament.py:1) in `basic` mode or [`evaluate_ai_models.py`](../../ai-service/scripts/evaluate_ai_models.py:1) to play:
     - New D2 vs old D2 vs random.
     - New D4 vs D2 / D6.
     - New D6 vs D4 / D8.
     - New D8 vs D6 and, if appropriate, some D9/D10 experiments.

5. **Optional calibration phase:**
   - Roll the candidate to calibration cohorts via feature flags or controlled environments.
   - Monitor aggregated telemetry for win-rate and perceived difficulty.

6. **Approve and apply ladder update:**
   - If all required stages succeed, apply `promotion_plan.json` via `apply_tier_promotion_plan.py`.
   - Promote via runtime override or by updating [`ladder_config.py`](../../ai-service/app/config/ladder_config.py:47).

### 8.2 Orchestration scripts / configs

To make the above reproducible, H-AI-9 defines the following orchestration entrypoints:

1. **`ai-service/scripts/run_tier_training_pipeline.py`** (implemented)
   - Arguments:
     - `--tier {D2,D3,D4,D5,D6,D7,D8,D9,D10}`.
     - `--board square8`.
     - `--num-players 2`.
     - `--output-dir PATH` (script creates a timestamped run dir inside).
   - Behaviour (conceptual):
     - Generate training data for the tier (self-play, territory, etc.).
     - Run heuristic or NN training as appropriate.
     - Optionally register NN models into [`python.AutoTournamentPipeline`](../../ai-service/app/training/auto_tournament.py:327).
     - Emit `training_report.json` with:
       - `candidate_id`.
       - Data sources.
       - Key hyperparameters.
       - Training metrics (losses, etc.).

2. **`ai-service/scripts/run_full_tier_gating.py`** (implemented wrapper around existing CLIs)
   - Arguments (current implementation):
     - `--tier T` (e.g. `D3`, `D5`, `D7`, `D9`).
     - `--candidate-id CANDIDATE_T_ID`.
     - `--run-dir PATH` (path containing `training_report.json`).
     - `--seed`, `--num-games` for the tier evaluation step.
     - `--no-perf` to skip perf even when a budget exists, `--demo` for CI-scale runs.
   - Behaviour:
     - Runs [`run_tier_gate.py`](../../ai-service/scripts/run_tier_gate.py:1) in difficulty-tier mode and writes `tier_eval_result.json` + `promotion_plan.json` under `--run-dir`.
     - Runs the tier perf benchmark via [`run_tier_perf_benchmark.py`](../../ai-service/scripts/run_tier_perf_benchmark.py:1) (when a `TierPerfBudget` exists) and writes `tier_perf_report.json`.
     - Aggregates into `gate_report.json`:
       - Embedded TierEvaluationResult metrics (`overall_pass`, `win_rate_vs_baseline`, `win_rate_vs_previous_tier`, etc.).
       - Perf metrics (`average_ms`, `p95_ms`) and budget evaluation (`within_avg`, `within_p95`, `overall_pass`) when a perf budget is defined for the tier.
   - Exit semantics:
     - Returns exit code `0` only when **both** the difficulty gate and perf budget pass; otherwise returns `1` so CI pipelines can key off a single orchestration step.

3. **`ai-service/scripts/unified_promotion_daemon.py parity-gate`** (cross-board candidate-vs-baseline parity gate)
   - Purpose:
     - Run a small evaluation matrix of _candidate vs baseline_ AIs across one or more boards using the generic `evaluate_ai_models` harness.
     - Aggregate win-rate and Wilson CI metrics and emit a single JSON summary with `overall_pass` plus per-board breakdowns.
     - Provide an additional, optional non-regression check that can be run **before** tier-specific gating and perf budgets.
   - Note:
     - `scripts/run_parity_promotion_gate.py` remains for backward compatibility but is deprecated (see its module docstring).
   - Arguments (current implementation):
     - `--player1`, `--player2`: AI types for candidate and baseline (for example `neural_network` vs `neural_network`).
     - `--checkpoint`, `--checkpoint2`: optional checkpoints for candidate and baseline.
     - `--boards`: list of boards to evaluate (defaults to `["square8"]`).
     - `--games-per-matrix`, `--max-moves`, `--seed`: evaluation controls.
     - `--min-ci-lower-bound`: minimum acceptable lower bound of the 95% CI for the candidate win rate on each matrix (default `0.5`).
     - `--output-json PATH`: optional JSON report output.
   - Behaviour:
     - For each requested board, call [`python.run_evaluation`](../../ai-service/scripts/evaluate_ai_models.py:260) and [`python.format_results_json`](../../ai-service/scripts/evaluate_ai_models.py:777) to obtain win-rate and positional stats.
     - Evaluate a simple non-inferiority rule per board: candidate passes that matrix when the _lower_ bound of its 95% Wilson CI is ≥ `min-ci-lower-bound`.
     - Emit a JSON report with:
       - `gate.overall_pass`: true only if **all** matrices pass.
       - `gate.matrices[board]`: per-board `player1_win_rate`, CI, `piece_advantage_p1`, and `passes` flag.
   - Integration:
     - Can be wired into CI pipelines or manual workflows as a fast “sanity check” ahead of running the heavier tier-specific gate + perf budgets.

4. **Minimal config snippet**
   - Optional `ai-service/config/tier_training_pipeline.square8_2p.json` describing:
     - Per-tier training modes (`heuristic_cmaes`, `search_persona`, `neural`).
     - CMA-ES preferences and gating thresholds.
     - Seeds and per-tier override knobs used by the tier pipeline.

### 8.3 Pipeline status artefacts

Within each `--run-dir`, it is useful to maintain a small status JSON, for example:

```json
{
  "tier": "D6",
  "board_type": "square8",
  "num_players": 2,
  "candidate_id": "sq8_d6_v2",
  "training": {
    "status": "completed",
    "report": "training_report.json"
  },
  "automated_gate": {
    "status": "passed",
    "eval_json": "tier_eval_result.json",
    "promotion_plan": "promotion_plan.json"
  },
  "perf": {
    "status": "passed",
    "perf_json": "tier_perf_report.json"
  },
  "human_calibration": {
    "required": true,
    "status": "pending",
    "min_games": 50
  }
}
```

This status file can be updated incrementally by the orchestration scripts and consumed by CI or dashboards.

### 8.4 Running the tier training pipeline (Square-8 2-player)

This section shows a concrete end-to-end invocation for a mid-tier on Square-8 2-player using the D4 minimax tier as an example. The same pattern applies to D2–D10 by changing the `--tier` argument.

#### 8.4.1 Demo / smoke run (no heavy training or evaluation)

From `ai-service/`, run the training orchestrator in demo mode:

```bash
cd ai-service
PYTHONPATH=. python scripts/run_tier_training_pipeline.py \
  --tier D4 \
  --board square8 \
  --num-players 2 \
  --output-dir logs/tier_gate \
  --demo \
  --seed 123
```

This:

- Writes `training_report.json` and `status.json` into a timestamped run dir under `logs/tier_gate/`.
- Records a `candidate_id` (for example `sq8_2p_d4_demo_YYYYMMDD_HHMMSS`) and a snapshot of the env + tier-specific training parameters.
- Uses only stubbed, demo-friendly training logic (no long self-play or neural training); this is the safe path for CI and local smoke tests.

Next, run the combined tier gate + perf wrapper (also in demo mode) for the same run directory:

```bash
cd ai-service
PYTHONPATH=. python scripts/run_full_tier_gating.py \
  --tier D4 \
  --candidate-id "<CANDIDATE_ID_FROM_TRAINING_REPORT>" \
  --run-dir "<RUN_DIR_WITH_TRAINING_REPORT>" \
  --demo
```

You obtain:

- `tier_eval_result.json` – difficulty-tier evaluation summary for D4.
- `promotion_plan.json` – promotion descriptor (`tier`, `current_model_id`, `candidate_model_id`, `decision`, `reason`).
- `tier_perf_report.json` – perf benchmark metrics and budget evaluation for D4 (since a `TierPerfBudget` exists).
- `gate_report.json` – combined summary with `final_decision`, referenced from `status.json` as described in §8.3.

This pair of commands exercises the full tier training + gating pipeline for a mid-tier (D4 on Square-8 2-player) in a way that is fast enough for automated tests.

#### 8.4.2 Full pipeline run (when canonical data is available)

When canonical Square-8 2-player replay DBs exist and have passed the unified gate in [`TRAINING_DATA_REGISTRY.md`](../../ai-service/TRAINING_DATA_REGISTRY.md:18):

1. Run [`run_tier_training_pipeline.py`](../../ai-service/scripts/run_tier_training_pipeline.py:1) **without** `--demo` so that it can:
   - Generate or select training datasets derived **only** from DBs marked `canonical` in the registry.
   - Invoke the appropriate heuristic / neural training loops for the tier (for example via [`python.train_model`](../../ai-service/app/training/train.py:1171) for neural tiers D6–D10, plus D5 when refreshing NNUE checkpoints).
   - Emit a richer `training_report.json` with explicit data-source and hyperparameter fields alongside training metrics.

2. Run [`run_full_tier_gating.py`](../../ai-service/scripts/run_full_tier_gating.py:1) **without** `--demo` (and without `--no-perf`) so that it:
   - Plays the full number of games from the tier’s [`TierEvaluationConfig`](../../ai-service/app/training/tier_eval_config.py:37).
   - Enforces the relevant [`TierPerfBudget`](../../ai-service/app/config/perf_budgets.py:20) for D3–D8.
   - Produces a `gate_report.json` whose `final_decision` drives promotion proposals and ladder updates as described in §5.3.

In both cases you should keep each candidate’s artefacts under a dedicated `--run-dir` such as `logs/tier_gate/D4_candidate_YYYYMMDD` so that `training_report.json`, `gate_report.json`, and `status.json` can be archived together.

#### 8.4.3 Canonical data and compute constraints

- **Canonical-only training data:** Real training runs for D2–D10 must be wired so that all self-play or replay-derived datasets are generated from DBs whose status is `canonical` in [`TRAINING_DATA_REGISTRY.md`](../../ai-service/TRAINING_DATA_REGISTRY.md:18). DBs marked `legacy_noncanonical`, `pending_gate`, or `DEPRECATED_R10` must not be used for new ladder training.
- **Default to demo in CI:** CI and quick local checks should use the `--demo` pipeline described above; heavy training runs are reserved for dedicated training environments with explicit operator sign-off.
- **Alignment with tier config JSON:** The small JSON descriptor [`tier_training_pipeline.square8_2p.json`](../../ai-service/config/tier_training_pipeline.square8_2p.json:1) mirrors the demo defaults (seeds, gating overrides) and is used by tests as a shape contract. Future Code-mode work can thread this config through the orchestrators to centralise per-tier settings.

---

## 9. Tier-by-tier summary (public ladder)

| Tier | Ladder engine        | Training mode     | Gate baseline threshold | Perf budget | Calibration role       |
| ---- | -------------------- | ----------------- | ----------------------- | ----------- | ---------------------- |
| D1   | Random               | —                 | —                       | —           | Baseline only          |
| D2   | Heuristic            | `heuristic_cmaes` | 0.60                    | No          | Anchor (Template A)    |
| D3   | Heuristic (tuned)    | `heuristic_cmaes` | 0.55                    | Yes         | Optional               |
| D4   | Minimax (NNUE)       | `search_persona`  | 0.68                    | Yes         | Anchor (Templates A/B) |
| D5   | Minimax (NNUE)       | `search_persona`  | 0.60                    | Yes         | Optional               |
| D6   | Descent (neural)     | `neural`          | 0.72                    | Yes         | Anchor (Templates B/C) |
| D7   | MCTS (heuristic)     | `search_persona`  | 0.65                    | Yes         | Optional               |
| D8   | MCTS (neural)        | `neural`          | 0.75                    | Yes         | Anchor (Template C)    |
| D9   | Gumbel MCTS (neural) | `neural`          | 0.75                    | No          | Optional / expert      |
| D10  | Gumbel MCTS (neural) | `neural`          | 0.75                    | No          | Optional / expert      |

The internal D11 tier follows the same Gumbel MCTS configuration but is not exposed in the public ladder.

---

## 10. Cross-references and doc updates

Follow-up doc work (in Code/Docs-mode) should:

- Add a short “Training & promotion pipeline” subsection to the future `AI_DIFFICULTY_SPEC.md` that points at this document as the process SSoT for D1–D10 (public ladder) on Square-8 2-player, noting that calibration anchors focus on D2/D4/D6/D8.
- Cross-link the calibration analysis process in [`AI_DIFFICULTY_CALIBRATION_ANALYSIS.md`](AI_DIFFICULTY_CALIBRATION_ANALYSIS.md:1) as the **human-facing difficulty tuning SSoT**, sitting on top of the automated training and gating loop defined here.
- Add a note to [`AI_HUMAN_CALIBRATION_GUIDE.md`](AI_HUMAN_CALIBRATION_GUIDE.md:1) clarifying that:
  - calibration templates A/B/C feed into the H-AI-9 promotion loop; and
  - calibration telemetry and experiment results are interpreted using the workflow and decision rules in [`AI_DIFFICULTY_CALIBRATION_ANALYSIS.md`](AI_DIFFICULTY_CALIBRATION_ANALYSIS.md:1); and
  - calibration is run on candidates that have already passed automated gates and perf budgets.

This completes the architectural design for H-AI-9 – Tiered Model Training & Promotion Loop (Square-8 2-Player). Implementation work is limited to adding orchestration scripts, wiring model ids / heuristic profiles into the existing training code, and updating ladder configs in controlled, well-documented promotions.

---

## Appendix A: GPU Cluster Calibration Runbook (2025-12-17)

This section provides concrete commands for running full AI ladder calibration on GPU infrastructure.

### A.1 Prerequisites

1. **GPU Access**: D6–D10 neural tiers require GPU (D6, D8–D10). D5 is minimax+NNUE (CPU OK for sq8), and D7 is heuristic MCTS but still CPU-intensive. Available hosts in `config/distributed_hosts.yaml`:
   - Lambda Cloud: GH200 (96GB), H100 (80GB), A10 (23GB)
   - Vast.ai: Various RTX cards for selfplay

2. **Environment Check**:

```bash
# On GPU host
cd ~/ringrift/ai-service
source venv/bin/activate
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"
```

3. **Canonical Dataset**: Verify training data exists (or export from the canonical DB):

```bash
ls -la data/training/canonical_square8_2p.npz
# If missing: python scripts/export_replay_dataset.py --db data/games/canonical_square8_2p.db \
#   --output data/training/canonical_square8_2p.npz
```

### A.2 Full Calibration Workflow

**Step 1: Training (per tier)**

```bash
OUTPUT_DIR=./runs/calibration_$(date +%Y%m%d)

# D2 (Heuristic) - CPU OK
python scripts/run_tier_training_pipeline.py --tier D2 \
  --output-dir "$OUTPUT_DIR" \
  --board square8 \
  --num-players 2

# D4 (Minimax/NNUE) - CPU OK for sq8
python scripts/run_tier_training_pipeline.py --tier D4 \
  --output-dir "$OUTPUT_DIR" \
  --board square8 \
  --num-players 2

# D6 (Neural) - GPU REQUIRED
python scripts/run_tier_training_pipeline.py --tier D6 \
  --output-dir "$OUTPUT_DIR" \
  --board square8 \
  --num-players 2

# D8 (Neural/MCTS) - GPU REQUIRED
python scripts/run_tier_training_pipeline.py --tier D8 \
  --output-dir "$OUTPUT_DIR" \
  --board square8 \
  --num-players 2

# Repeat for D3/D5/D7/D9/D10 as needed.
```

**Step 2: Gating (per tier)**

```bash
# Full gating includes evaluation + perf benchmark + promotion plan
python scripts/run_full_tier_gating.py --tier D2 \
  --candidate-id <from_training_report> \
  --run-dir <run_dir_with_training_report>

# Repeat for D3, D4, D5, D6, D7, D8, D9, D10 as needed
```

**Step 3: Promotion (if gates pass)**

```bash
python scripts/apply_tier_promotion_plan.py \
  --promotion-plan ./runs/calibration_d6_$(date +%Y%m%d)/promotion_plan.json \
  --registry config/tier_candidate_registry.square8_2p.json
```

### A.3 Time Estimates

> Note: Estimates below are for anchor tiers D2/D4/D6/D8. D3/D5/D7 are typically comparable to adjacent tiers; D9/D10 are longer due to Gumbel MCTS budgets.

| Tier | Training  | Evaluation (400 games) | Total |
| ---- | --------- | ---------------------- | ----- |
| D2   | 30-60 min | 10-20 min              | ~1h   |
| D4   | 15-30 min | 20-40 min              | ~1h   |
| D6   | 2-4 hours | 30-60 min              | ~4h   |
| D8   | 3-6 hours | 40-80 min              | ~6h   |

**Anchor-tier calibration: 8-12 hours** (can parallelize D2/D4 on CPU while D6/D8 run on GPU)

### A.4 Board-Specific Notes

| Board     | D4/D5 AI | Notes                              |
| --------- | -------- | ---------------------------------- |
| square8   | MINIMAX  | Standard pipeline                  |
| square19  | MCTS     | Minimax too slow (commit 44bf4400) |
| hexagonal | MCTS     | Minimax too slow (commit 44bf4400) |

### A.5 Monitoring

```bash
# Watch training progress
tail -f ./runs/calibration_d6_*/training.log

# Check GPU utilization
nvidia-smi -l 5

# Monitor P2P sync status (if distributed)
python scripts/vast_p2p_sync.py --check
```

### A.6 Troubleshooting

| Issue           | Solution                                           |
| --------------- | -------------------------------------------------- |
| OOM on training | Reduce batch size via `--batch-size 32`            |
| Eval games slow | Check think_time_ms in ladder_config.py            |
| Gate fails      | Check win rate vs threshold in tier_eval_config.py |
| P2P sync errors | Use Tailscale IPs (100.x.x.x) for reliability      |
