# RingRift AI Training & Datasets

> **Doc Status (2025-11-27): Active (with historical/aspirational content)**
>
> **Role:** Canonical reference for current offline AI training and dataset-generation flows in the Python AI service (self-play generators, territory/combined-margin datasets, and heuristic-weight training). Describes how training jobs reuse the Python host over the shared TS rules engine and how datasets are structured and generated.
>
> **Not a semantics SSoT:** This document does not define core game rules or lifecycle semantics. Rules semantics are owned by the shared TypeScript rules engine under `src/shared/engine/**` plus contracts and vectors (see `RULES_CANONICAL_SPEC.md`, `RULES_ENGINE_ARCHITECTURE.md`, `RULES_IMPLEMENTATION_MAPPING.md`, `docs/RULES_ENGINE_SURFACE_AUDIT.md`). Lifecycle semantics are owned by `docs/CANONICAL_ENGINE_API.md` together with shared types/schemas in `src/shared/types/game.ts`, `src/shared/engine/orchestration/types.ts`, `src/shared/types/websocket.ts`, and `src/shared/validation/websocketSchemas.ts`. The Python engine and training stack are **hosts** that must match the TS SSoT via the parity backbone.
>
> **Related docs:** `AI_ARCHITECTURE.md`, `docs/PYTHON_PARITY_REQUIREMENTS.md`, `docs/PARITY_SEED_TRIAGE.md`, `ai-service/README.md`, `docs/AI_TRAINING_PREPARATION_GUIDE.md`, `docs/STRICT_INVARIANT_SOAKS.md`, `tests/TEST_SUITE_PARITY_PLAN.md`, and `DOCUMENTATION_INDEX.md`.

**Scope:** Python AI service training pipelines, self-play generators, and territory/combined-margin datasets.

This document is the canonical reference for the current **offline training and dataset generation** flows in the Python AI service. It complements the high-level overview in [`AI_ARCHITECTURE.md`](../AI_ARCHITECTURE.md) and the rules-engine mapping in [`RULES_ENGINE_ARCHITECTURE.md`](../RULES_ENGINE_ARCHITECTURE.md).

---

## 1. Components & shared logic

All training entrypoints reuse the same Python rules stack as the live AI service:

- Python rules engine (host implementation mirroring the TS shared engine): [`GameEngine`](../ai-service/app/game_engine.py).
- Board helpers and disconnected-region detection: [`BoardManager`](../ai-service/app/board_manager.py) and [`BoardManager.find_disconnected_regions()`](../ai-service/app/board_manager.py).
- Rules façade and mutators: [`DefaultRulesEngine`](../ai-service/app/rules/default_engine.py) and [`TerritoryMutator`](../ai-service/app/rules/mutators/territory.py).
- RL-style environment wrapper: [`RingRiftEnv`](../ai-service/app/training/env.py), which internally calls [`create_initial_state()`](../ai-service/app/training/generate_data.py), [`GameEngine.get_valid_moves()`](../ai-service/app/game_engine.py), and [`GameEngine.apply_move()`](../ai-service/app/game_engine.py).

**Rules SSoT and parity safeguards:**

- The **canonical rules semantics** live in the shared TypeScript engine under [`src/shared/engine/`](../src/shared/engine/) (helpers → aggregates → orchestrator) together with the v2 **contract vectors** under [`tests/fixtures/contract-vectors/v2/`](../tests/fixtures/contract-vectors/v2/). See [`docs/CANONICAL_ENGINE_API.md`](./CANONICAL_ENGINE_API.md) and [`docs/PYTHON_PARITY_REQUIREMENTS.md`](./PYTHON_PARITY_REQUIREMENTS.md) for details.
- The Python engine and mutators above are treated as a **host/adapter implementation** that must match the TS SSoT; they are validated by:
  - Contract-vector runners (`tests/contracts/contractVectorRunner.test.ts`, `ai-service/tests/contracts/test_contract_vectors.py`).
  - Parity/plateau/territory suites under `tests/unit/*Parity*` and `ai-service/tests/parity/`.
  - Mutator shadow contracts and divergence guards described in [`RULES_ENGINE_ARCHITECTURE.md`](../RULES_ENGINE_ARCHITECTURE.md) and [`docs/INCIDENT_TERRITORY_MUTATOR_DIVERGENCE.md`](./INCIDENT_TERRITORY_MUTATOR_DIVERGENCE.md).

Because training jobs and the live AI/rules service share this stack, **any divergence between Python and the canonical TS rules engine** will affect both online play and offline datasets. The parity suites and mutator contracts listed above are therefore critical safeguards.

---

## 2. Running the AI service for training

Most training workflows expect the Python AI service to be installable and runnable on its own. For HTTP API details and the full Docker stack see [`ai-service/README.md`](../ai-service/README.md); this section focuses on quick local usage.

### 2.1 Local virtualenv (recommended)

```bash
cd ai-service
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Start the FastAPI service (development hot reload)
uvicorn app.main:app --reload --port 8001
```

Key endpoints:

- Base URL: `http://localhost:8001`
- Health: `http://localhost:8001/health`
- Swagger UI: `http://localhost:8001/docs`

Ensure the Node backend (if you run it alongside training experiments) points to the same URL:

```env
AI_SERVICE_URL=http://localhost:8001
```

### 2.2 Docker

For containerised experiments, reuse the standard image:

```bash
cd ai-service
docker build -t ringrift-ai-service .
docker run --rm -p 8001:8001 ringrift-ai-service
```

Again, point any backend-client experiments at `http://localhost:8001` via `AI_SERVICE_URL`.

---

## 3. General self-play dataset generator (`generate_data.py`)

The legacy NN-style self-play generator lives in [`generate_dataset()`](../ai-service/app/training/generate_data.py). It uses:

- [`RingRiftEnv`](../ai-service/app/training/env.py) backed by the Python [`GameEngine`](../ai-service/app/game_engine.py).
- [`DescentAI`](../ai-service/app/ai/descent_ai.py) (and its neural network, when configured) for tree search.
- Feature extraction and action encoding from the NN stack under `ai-service/app/ai/`.

### 3.1 Invocation

`generate_data.py` now exposes both:

- A reusable function [`generate_dataset()`](../ai-service/app/training/generate_data.py) for programmatic use.
- A CLI entrypoint wired through [`_parse_args()`](../ai-service/app/training/generate_data.py) and [`main()`](../ai-service/app/training/generate_data.py).

From the `ai-service` root you can invoke the generator as a module:

```bash
cd ai-service
python -m app.training.generate_data \
  --num-games 100 \
  --output logs/training_data.npz \
  --board-type square8 \
  --seed 42 \
  --max-moves 200
```

If you omit flags, the CLI uses these defaults:

- `--num-games` (`int`, default `100`): number of self-play games to generate.
- `--output` (`str`, default `logs/training_data.npz`): output NPZ path (directories are created if needed).
- `--board-type` (`square8` | `square19` | `hexagonal`, default `square8`): board geometry for self-play games; converted to [`BoardType`](../ai-service/app/models/core.py) via [`_board_type_from_str()`](../ai-service/app/training/generate_data.py).
- `--seed` (`int`, default `42`): base RNG seed for reproducible runs. When provided, per-game seeds are derived as `seed + game_idx`.
- `--max-moves` (`int`, default `200`): maximum number of moves per game before the environment forces termination.
- `--batch-size` (`int`, optional): reserved for future streaming/flush behaviour; currently accepted but not used.

Under the hood, `main()` validates numeric arguments and then calls:

```python
generate_dataset(
    num_games=args.num_games,
    output_file=args.output,
    board_type=board_type,
    seed=args.seed,
    max_moves=args.max_moves,
    batch_size=args.batch_size,
)
```

The older hard-coded `if __name__ == "__main__": generate_dataset(num_games=2)` path has been replaced by this CLI, so you no longer need to edit the file to change `num_games`, `board_type`, or the output location.

---

## 4. Territory / combined-margin dataset generator

The **territory dataset generator** is implemented in [`generate_territory_dataset.py`](../ai-service/app/training/generate_territory_dataset.py) and exposed as a CLI module:

- Core function: [`generate_territory_dataset()`](../ai-service/app/training/generate_territory_dataset.py).
- CLI entry: [`main()`](../ai-service/app/training/generate_territory_dataset.py) via `python -m app.training.generate_territory_dataset`.
- Smoke test: [`test_generate_territory_dataset_mixed_smoke`](../ai-service/tests/test_generate_territory_dataset_smoke.py) runs a small mixed-engine job and asserts exit code 0, no `TerritoryMutator diverged from GameEngine.apply_move` stderr, and a non-empty output JSONL file.

### 4.1 Semantics and targets

For each finished self-play game with final `GameState` (denoted `S_T` in the docstring), the generator computes, for each player `i`:

```text
territory_margin_i = T_i - max_{j != i} T_j
elim_margin_i      = E_i - max_{j != i} E_j
target_i           = territory_margin_i + elim_margin_i
```

where:

- `T_p` is `players[p].territory_spaces` at the final state.
- `E_p` is `players[p].eliminated_rings` at the final state.
- The implementation is [`_final_combined_margin()`](../ai-service/app/training/generate_territory_dataset.py).

Along each self-play trajectory, [`generate_territory_dataset()`](../ai-service/app/training/generate_territory_dataset.py) records **pre-move** snapshots `S_t` and emits one JSONL record per `(S_t, player)` with:

- `target` equal to that player’s final combined margin `target_i` from the finished game.
- `time_weight = gamma^(T - t)` where `gamma = 0.99` is currently a **fixed constant** in the implementation.

These examples are intended for scalar-regression training of heuristic-style evaluators (for example, via [`train_heuristic_weights.py`](../ai-service/app/training/train_heuristic_weights.py)).

### 4.2 CLI usage

The module exposes a straightforward CLI via [`_parse_args()`](../ai-service/app/training/generate_territory_dataset.py). From the `ai-service` root:

```bash
cd ai-service
python -m app.training.generate_territory_dataset \
  --num-games 10 \
  --output logs/debug.square8.descent2p.10.jsonl \
  --board-type square8 \
  --engine-mode descent-only \
  --num-players 2 \
  --max-moves 200 \
  --seed 123
```

A more demanding **mixed-engine** example (mirroring the smoke test and typical training runs):

```bash
cd ai-service
python -m app.training.generate_territory_dataset \
  --num-games 10 \
  --output logs/debug.square8.mixed2p.10.jsonl \
  --board-type square8 \
  --engine-mode mixed \
  --num-players 2 \
  --max-moves 200 \
  --seed 42
```

#### 4.2.1 CLI arguments

- `--num-games` (`int`, default `10`): number of self-play games to generate.
- `--output` (`str`, required): path to the output `.jsonl` file. Parent directories are created if needed.
- `--board-type` (`square8` | `square19` | `hexagonal`, default `square8`): board geometry for self-play games, mapped to [`BoardType`](../ai-service/app/models/core.py) via [`_board_type_from_str()`](../ai-service/app/training/generate_territory_dataset.py).
- `--max-moves` (`int`, default `200`): maximum number of moves per game before forcibly terminating the trajectory.
- `--seed` (`int`, optional): base RNG seed for deterministic runs. When provided, per-game seeds are derived as `seed + game_idx` so that each game sees a distinct but reproducible RNG stream.
- `--engine-mode` (`descent-only` | `mixed`, default `descent-only`):
  - `descent-only` – all players use fixed [`DescentAI`](../ai-service/app/ai/descent_ai.py) instances with deterministic configs.
  - `mixed` – for each game and player, difficulty and AI type are sampled from the canonical ladder profiles via [`_get_difficulty_profile()`](../ai-service/app/main.py) and [`_create_ai_instance()`](../ai-service/app/main.py).
- `--num-players` (`int`, default `2`): number of active players per game (2–4). Used when constructing the initial [`GameState`](../ai-service/app/training/generate_data.py) via [`RingRiftEnv.reset()`](../ai-service/app/training/env.py).
- `--gamma` (`float`, default `0.99`): discount factor for time weighting. Controls how much earlier positions are weighted relative to later positions in the trajectory.

### 4.3 JSONL record schema

Each line of the output file is a single JSON object with at least the following fields:

- `game_state`: a serialised `GameState` snapshot using `model_dump(by_alias=True, mode="json")`, compatible with `GameState.model_validate(...)` on reload.
- `player_number` (`int`): 1-based player index whose perspective the target is defined for.
- `target` (`float`): final combined margin (territory + eliminated rings) for `player_number` at the end of the game, as computed by [`_final_combined_margin()`](../ai-service/app/training/generate_territory_dataset.py).
- `time_weight` (`float`): discount weight `gamma^(T - t)` with `gamma = 0.99`, where `T` is the trajectory length and `t` is the 1-based index of this snapshot along the trajectory.
- `engine_mode` (`"descent-only"` | `"mixed"`): copied directly from the CLI argument.
- `num_players` (`int`): number of active players in this game.
- `ai_type_pN` (`str`, for each active player `N`): string label derived from the per-player [`AIType`](../ai-service/app/models/core.py) enum, recorded for analysis (for example `"random"`, `"heuristic"`, `"minimax"`, `"mcts"`, `"descent"`).
- `ai_difficulty_pN` (`int`, for each active player `N`): numeric difficulty assigned to that player when the game was initialised.

A typical consumer expects to iterate over the JSONL file line-by-line, parse each object, and then feed `(game_state, player_number, target, time_weight, engine_mode, num_players, ai_*_pN)` into a training pipeline such as [`train_heuristic_weights.py`](../ai-service/app/training/train_heuristic_weights.py).

---

## 5. Seeds, determinism, and mixed-engine selection

### 5.1 Environment seeding

[`RingRiftEnv.reset()`](../ai-service/app/training/env.py) applies the provided `seed` (when non-`None`) to:

- Python’s `random` module.
- NumPy’s RNG.
- PyTorch’s RNG.

This ensures that, for a fixed `(board_type, num_players, seed)`, the initial `GameState` and any stochastic components inside AI search that rely on these RNGs are reproducible.

### 5.2 Territory generator seeding

Within [`generate_territory_dataset()`](../ai-service/app/training/generate_territory_dataset.py):

- A **base seed** is taken directly from `--seed` (if provided).
- For each game index `game_idx`, a per-game `game_seed` is derived as `seed + game_idx` and threaded into [`RingRiftEnv.reset(seed=game_seed)`](../ai-service/app/training/env.py).
- In `engine_mode == "mixed"`, a local `random.Random` instance (`game_rng`) is also initialised from `base_seed + game_idx` and used exclusively for:
  - Sampling difficulties from `difficulty_choices`.
  - Drawing per-player RNG seeds for `AIConfig.rngSeed`.

As a result:

- For fixed arguments `(--board-type, --engine-mode, --num-players, --max-moves, --seed)` and fixed code, the JSONL output of [`generate_territory_dataset.py`](../ai-service/app/training/generate_territory_dataset.py) is **reproducible**.
- Changing `--seed` changes both **which AI profiles** are chosen in mixed mode and the internal RNG streams inside those AIs.

---

## 6. LPS and Ring Cap Alignment (R172, CLAR-003)

### 6.1 LPS (Last Player Standing) Victory Handling

The training code correctly handles LPS victories (R172) through the following mechanisms:

- **Reward computation** ([`env.py`](../ai-service/app/training/env.py)): Uses `state.winner` to assign rewards (+1 for winner, -1 for loser). Since LPS victories set `state.winner`, they receive the same appropriate rewards as elimination or territory victories.

- **Tournament statistics** ([`tournament.py`](../ai-service/app/training/tournament.py)): The `infer_victory_reason()` function categorizes victories by type:
  - `"elimination"`: Player reached `victory_threshold` for eliminated rings.
  - `"territory"`: Player reached `territory_victory_threshold` for collapsed spaces.
  - `"last_player_standing"`: R172 LPS victory where `lps_exclusive_player_for_completed_round` matches the winner.
  - `"structural"`: Global stalemate resolved by tie-breakers.
  - `"unknown"`: Catch-all for edge cases.

- **Data generation** ([`generate_data.py`](../ai-service/app/training/generate_data.py)): Uses `state.winner` for outcome labels, correctly capturing LPS victories in training data.

### 6.2 Own-Colour Ring Caps (CLAR-003)

The `ringsPerPlayer` cap applies only to **own-colour rings** in play, not captured opponent rings:

- **Canonical helper** ([`rules/core.py`](../ai-service/app/rules/core.py)): The `count_rings_in_play_for_player()` function correctly counts only rings where `ring_owner == player_number`, plus `rings_in_hand`.

- **Move generation** ([`game_engine.py`](../ai-service/app/game_engine.py)): Ring placement validation uses `count_rings_in_play_for_player()` to enforce the per-player cap correctly.

- **State encoding**: The neural network encoding in [`neural_net.py`](../ai-service/app/ai/neural_net.py) includes:
  - Stack heights (normalized) for my/opponent stacks (channels 0/1)
  - `rings_in_hand` per player (global features)
  - `eliminated_rings` per player (global features)

  While the encoding doesn't directly track "own-colour rings on board" as a separate feature, the cap is enforced at move generation time, so only legal moves are presented to the AI.

### 6.3 Test Coverage

The alignment tests in [`test_training_lps_alignment.py`](../ai-service/tests/test_training_lps_alignment.py) verify:

- LPS victories give appropriate rewards.
- Victory reason inference works correctly.
- Own-colour ring counting excludes captured opponent rings.

---

## 7. Relationship to live rules and parity tests

The territory generator is deliberately wired to the **same canonical rules logic** used by live games and TS parity fixtures:

- All move legality and state transitions come from Python [`GameEngine.get_valid_moves()`](../ai-service/app/game_engine.py) and [`GameEngine.apply_move()`](../ai-service/app/game_engine.py), which in turn mirror the TS shared engine modules such as [`territoryDetection.ts`](../src/shared/engine/territoryDetection.ts), [`territoryProcessing.ts`](../src/shared/engine/territoryProcessing.ts), and [`territoryDecisionHelpers.ts`](../src/shared/engine/territoryDecisionHelpers.ts).
- The rules façade [`DefaultRulesEngine`](../ai-service/app/rules/default_engine.py) and [`TerritoryMutator`](../ai-service/app/rules/mutators/territory.py) enforce **shadow contracts** against [`GameEngine.apply_move()`](../ai-service/app/game_engine.py) for territory moves, with a targeted escape hatch when host-level forced elimination for the next player occurs (see [`docs/INCIDENT_TERRITORY_MUTATOR_DIVERGENCE.md`](./INCIDENT_TERRITORY_MUTATOR_DIVERGENCE.md)).
- The CLI smoke test [`test_generate_territory_dataset_mixed_smoke`](../ai-service/tests/test_generate_territory_dataset_smoke.py) is the end-to-end guard that exercises the module in `engine_mode="mixed"`, asserts no `TerritoryMutator diverged from GameEngine.apply_move` messages appear on stderr, and verifies that a non-empty JSONL file is produced.

For a deeper discussion of how TS and Python engines are kept in sync (trace parity, mutator equivalence tests, and shadow modes), see [`RULES_ENGINE_ARCHITECTURE.md`](../RULES_ENGINE_ARCHITECTURE.md) and the incident report in [`docs/INCIDENT_TERRITORY_MUTATOR_DIVERGENCE.md`](./INCIDENT_TERRITORY_MUTATOR_DIVERGENCE.md).
