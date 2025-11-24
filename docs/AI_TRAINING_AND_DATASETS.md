# RingRift AI Training & Datasets

**Scope:** Python AI service training pipelines, self-play generators, and territory/combined-margin datasets.

This document is the canonical reference for the current **offline training and dataset generation** flows in the Python AI service. It complements the high-level overview in [`AI_ARCHITECTURE.md`](AI_ARCHITECTURE.md:17) and the rules-engine mapping in [`RULES_ENGINE_ARCHITECTURE.md`](RULES_ENGINE_ARCHITECTURE.md:1).

---

## 1. Components & shared logic

All training entrypoints reuse the same Python rules stack as the live AI service:

- Canonical Python rules engine: [`GameEngine`](ai-service/app/game_engine.py:33).
- Board helpers and disconnected-region detection: [`BoardManager`](ai-service/app/board_manager.py:1) and [`BoardManager.find_disconnected_regions()`](ai-service/app/board_manager.py:171).
- Rules façade and mutators: [`DefaultRulesEngine`](ai-service/app/rules/default_engine.py:23) and [`TerritoryMutator`](ai-service/app/rules/mutators/territory.py:6).
- RL-style environment wrapper: [`RingRiftEnv`](ai-service/app/training/env.py:6), which internally calls [`create_initial_state()`](ai-service/app/training/generate_data.py:21), [`GameEngine.get_valid_moves()`](ai-service/app/game_engine.py:45), and [`GameEngine.apply_move()`](ai-service/app/game_engine.py:117).

Because training jobs and the live AI/rules service share this stack, **any divergence between Python and the canonical TS rules engine** will affect both online play and offline datasets. Parity suites and the mutator contracts described in [`RULES_ENGINE_ARCHITECTURE.md`](RULES_ENGINE_ARCHITECTURE.md:158) are therefore critical safeguards.

---

## 2. Running the AI service for training

Most training workflows expect the Python AI service to be installable and runnable on its own. For HTTP API details and the full Docker stack see [`ai-service/README.md`](ai-service/README.md:1); this section focuses on quick local usage.

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

The legacy NN-style self-play generator lives in [`generate_dataset()`](ai-service/app/training/generate_data.py:337). It uses:

- [`RingRiftEnv`](ai-service/app/training/env.py:6) backed by the Python [`GameEngine`](ai-service/app/game_engine.py:33).
- [`DescentAI`](ai-service/app/ai/descent_ai.py:1) (and its neural network, when configured) for tree search.
- Feature extraction and action encoding from the NN stack under `ai-service/app/ai/`.

### 3.1 Invocation

At present, [`generate_data.py`](ai-service/app/training/generate_data.py:1) exposes `generate_dataset()` as a module-level function and wires a small default invocation under:

```python
if __name__ == "__main__":
    generate_dataset(num_games=2)
```

You can run a quick smoke generation from the `ai-service` root:

```bash
cd ai-service
python -m app.training.generate_data
```

This will:

- Play a small number of self-play games (currently hard-coded `num_games=2`).
- Append training samples to the default `.npz` dataset path inside [`generate_dataset()`](ai-service/app/training/generate_data.py:337).

To change parameters such as `num_games`, `board_type`, or output location you currently need to **edit the call site in** [`generate_data.py`](ai-service/app/training/generate_data.py:337). There is no public CLI parser for this generator yet; introducing one would be a good follow-up task in the AI/NN roadmap.

---

## 4. Territory / combined-margin dataset generator

The **territory dataset generator** is implemented in [`generate_territory_dataset.py`](ai-service/app/training/generate_territory_dataset.py:1) and exposed as a CLI module:

- Core function: [`generate_territory_dataset()`](ai-service/app/training/generate_territory_dataset.py:107).
- CLI entry: [`main()`](ai-service/app/training/generate_territory_dataset.py:442) via `python -m app.training.generate_territory_dataset`.
- Smoke test: [`test_generate_territory_dataset_mixed_smoke`](ai-service/tests/test_generate_territory_dataset_smoke.py:15) runs a small mixed-engine job and asserts exit code 0, no `TerritoryMutator diverged from GameEngine.apply_move` stderr, and a non-empty output JSONL file.

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
- The implementation is [`_final_combined_margin()`](ai-service/app/training/generate_territory_dataset.py:79).

Along each self-play trajectory, [`generate_territory_dataset()`](ai-service/app/training/generate_territory_dataset.py:107) records **pre-move** snapshots `S_t` and emits one JSONL record per `(S_t, player)` with:

- `target` equal to that player’s final combined margin `target_i` from the finished game.
- `time_weight = gamma^(T - t)` where `gamma = 0.99` is currently a **fixed constant** in the implementation.

These examples are intended for scalar-regression training of heuristic-style evaluators (for example, via [`train_heuristic_weights.py`](ai-service/app/training/train_heuristic_weights.py:1)).

### 4.2 CLI usage

The module exposes a straightforward CLI via [`_parse_args()`](ai-service/app/training/generate_territory_dataset.py:372). From the `ai-service` root:

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
- `--board-type` (`square8` | `square19` | `hexagonal`, default `square8`): board geometry for self-play games, mapped to [`BoardType`](ai-service/app/models/core.py:1) via [`_board_type_from_str()`](ai-service/app/training/generate_territory_dataset.py:432).
- `--max-moves` (`int`, default `200`): maximum number of moves per game before forcibly terminating the trajectory.
- `--seed` (`int`, optional): base RNG seed for deterministic runs. When provided, per-game seeds are derived as `seed + game_idx` so that each game sees a distinct but reproducible RNG stream.
- `--engine-mode` (`descent-only` | `mixed`, default `descent-only`):
  - `descent-only` – all players use fixed [`DescentAI`](ai-service/app/ai/descent_ai.py:1) instances with deterministic configs.
  - `mixed` – for each game and player, difficulty and AI type are sampled from the canonical ladder profiles via [`_get_difficulty_profile()`](ai-service/app/main.py:1) and [`_create_ai_instance()`](ai-service/app/main.py:1).
- `--num-players` (`int`, default `2`): number of active players per game (2–4). Used when constructing the initial [`GameState`](ai-service/app/training/generate_data.py:21) via [`RingRiftEnv.reset()`](ai-service/app/training/env.py:29).

The **discount factor `gamma`** used for `time_weight` is currently a hard-coded constant (`gamma = 0.99` inside [`generate_territory_dataset()`](ai-service/app/training/generate_territory_dataset.py:328)). There is **no `--gamma` CLI flag yet**; changing this value requires editing the implementation.

### 4.3 JSONL record schema

Each line of the output file is a single JSON object with at least the following fields:

- `game_state`: a serialised `GameState` snapshot using `model_dump(by_alias=True, mode="json")`, compatible with `GameState.model_validate(...)` on reload.
- `player_number` (`int`): 1-based player index whose perspective the target is defined for.
- `target` (`float`): final combined margin (territory + eliminated rings) for `player_number` at the end of the game, as computed by [`_final_combined_margin()`](ai-service/app/training/generate_territory_dataset.py:79).
- `time_weight` (`float`): discount weight `gamma^(T - t)` with `gamma = 0.99`, where `T` is the trajectory length and `t` is the 1-based index of this snapshot along the trajectory.
- `engine_mode` (`"descent-only"` | `"mixed"`): copied directly from the CLI argument.
- `num_players` (`int`): number of active players in this game.
- `ai_type_pN` (`str`, for each active player `N`): string label derived from the per-player [`AIType`](ai-service/app/models/core.py:1) enum, recorded for analysis (for example `"random"`, `"heuristic"`, `"minimax"`, `"mcts"`, `"descent"`).
- `ai_difficulty_pN` (`int`, for each active player `N`): numeric difficulty assigned to that player when the game was initialised.

A typical consumer expects to iterate over the JSONL file line-by-line, parse each object, and then feed `(game_state, player_number, target, time_weight, engine_mode, num_players, ai_*_pN)` into a training pipeline such as [`train_heuristic_weights.py`](ai-service/app/training/train_heuristic_weights.py:1).

---

## 5. Seeds, determinism, and mixed-engine selection

### 5.1 Environment seeding

[`RingRiftEnv.reset()`](ai-service/app/training/env.py:29) applies the provided `seed` (when non-`None`) to:

- Python’s `random` module.
- NumPy’s RNG.
- PyTorch’s RNG.

This ensures that, for a fixed `(board_type, num_players, seed)`, the initial `GameState` and any stochastic components inside AI search that rely on these RNGs are reproducible.

### 5.2 Territory generator seeding

Within [`generate_territory_dataset()`](ai-service/app/training/generate_territory_dataset.py:107):

- A **base seed** is taken directly from `--seed` (if provided).
- For each game index `game_idx`, a per-game `game_seed` is derived as `seed + game_idx` and threaded into [`RingRiftEnv.reset(seed=game_seed)`](ai-service/app/training/env.py:29).
- In `engine_mode == "mixed"`, a local `random.Random` instance (`game_rng`) is also initialised from `base_seed + game_idx` and used exclusively for:
  - Sampling difficulties from `difficulty_choices`.
  - Drawing per-player RNG seeds for `AIConfig.rngSeed`.

As a result:

- For fixed arguments `(--board-type, --engine-mode, --num-players, --max-moves, --seed)` and fixed code, the JSONL output of [`generate_territory_dataset.py`](ai-service/app/training/generate_territory_dataset.py:1) is **reproducible**.
- Changing `--seed` changes both **which AI profiles** are chosen in mixed mode and the internal RNG streams inside those AIs.

---

## 6. Relationship to live rules and parity tests

The territory generator is deliberately wired to the **same canonical rules logic** used by live games and TS parity fixtures:

- All move legality and state transitions come from Python [`GameEngine.get_valid_moves()`](ai-service/app/game_engine.py:45) and [`GameEngine.apply_move()`](ai-service/app/game_engine.py:117), which in turn mirror the TS shared engine modules such as [`territoryDetection.ts`](src/shared/engine/territoryDetection.ts:1), [`territoryProcessing.ts`](src/shared/engine/territoryProcessing.ts:1), and [`territoryDecisionHelpers.ts`](src/shared/engine/territoryDecisionHelpers.ts:1).
- The rules façade [`DefaultRulesEngine`](ai-service/app/rules/default_engine.py:23) and [`TerritoryMutator`](ai-service/app/rules/mutators/territory.py:6) enforce **shadow contracts** against [`GameEngine.apply_move()`](ai-service/app/game_engine.py:117) for territory moves, with a targeted escape hatch when host-level forced elimination for the next player occurs (see [`docs/INCIDENT_TERRITORY_MUTATOR_DIVERGENCE.md`](docs/INCIDENT_TERRITORY_MUTATOR_DIVERGENCE.md:1)).
- The CLI smoke test [`test_generate_territory_dataset_mixed_smoke`](ai-service/tests/test_generate_territory_dataset_smoke.py:15) is the end-to-end guard that exercises the module in `engine_mode="mixed"`, asserts no `TerritoryMutator diverged from GameEngine.apply_move` messages appear on stderr, and verifies that a non-empty JSONL file is produced.

For a deeper discussion of how TS and Python engines are kept in sync (trace parity, mutator equivalence tests, and shadow modes), see [`RULES_ENGINE_ARCHITECTURE.md`](RULES_ENGINE_ARCHITECTURE.md:180) and the incident report in [`docs/INCIDENT_TERRITORY_MUTATOR_DIVERGENCE.md`](docs/INCIDENT_TERRITORY_MUTATOR_DIVERGENCE.md:1).
