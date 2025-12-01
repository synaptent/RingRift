# RingRift AI Service

> **Doc Status (2025-11-27): Active (Python AI microservice, AI host/adapter only)**
>
> - Role: primary reference for the Python AI microservice API surface, difficulty ladder, and integration with the Node.js backend. It describes the AI host, not the game rules themselves.
> - Not a semantics or lifecycle SSoT: the Python rules engine inside this service is a **host/adapter** over the shared TypeScript rules SSoT under `src/shared/engine/**` and the engine contracts under `src/shared/engine/contracts/**`, with cross-language behaviour anchored by the v2 contract vectors in `tests/fixtures/contract-vectors/v2/**`. For canonical rules semantics and lifecycle/API contracts, defer to [`RULES_CANONICAL_SPEC.md`](../RULES_CANONICAL_SPEC.md), [`ringrift_complete_rules.md`](../ringrift_complete_rules.md), [`RULES_ENGINE_ARCHITECTURE.md`](../RULES_ENGINE_ARCHITECTURE.md), [`RULES_IMPLEMENTATION_MAPPING.md`](../RULES_IMPLEMENTATION_MAPPING.md), and [`docs/CANONICAL_ENGINE_API.md`](../docs/CANONICAL_ENGINE_API.md).
> - Related docs: high-level AI architecture and roadmap in [`AI_ARCHITECTURE.md`](../AI_ARCHITECTURE.md), host-focused analysis in [`ai-service/AI_ASSESSMENT_REPORT.md`](./AI_ASSESSMENT_REPORT.md), improvement roadmap in [`ai-service/AI_IMPROVEMENT_PLAN.md`](./AI_IMPROVEMENT_PLAN.md), training workflows in [`docs/AI_TRAINING_AND_DATASETS.md`](../docs/AI_TRAINING_AND_DATASETS.md) and [`docs/AI_TRAINING_PREPARATION_GUIDE.md`](../docs/AI_TRAINING_PREPARATION_GUIDE.md), TS↔Python rules parity details in [`docs/PYTHON_PARITY_REQUIREMENTS.md`](../docs/PYTHON_PARITY_REQUIREMENTS.md) and [`docs/STRICT_INVARIANT_SOAKS.md`](../docs/STRICT_INVARIANT_SOAKS.md), and dependency/upgrade details in [`ai-service/DEPENDENCY_UPDATES.md`](./DEPENDENCY_UPDATES.md) together with the wave-based cross-repo plan in [`docs/DEPENDENCY_UPGRADE_PLAN.md`](../docs/DEPENDENCY_UPGRADE_PLAN.md).

Python-based FastAPI microservice for AI move generation and position evaluation.

## Architecture

This service provides AI capabilities for the RingRift game through a RESTful API. It's designed as a microservice to:

- Enable ML/AI libraries (TensorFlow, PyTorch, etc.) without adding them to the Node.js backend
- Allow independent scaling of AI computation
- Facilitate future ML model integration
- Provide language-appropriate AI implementations (Python for ML/AI)

## Features

- **Currently supported AI types (canonical ladder)**:
  - RandomAI: Selects random valid moves (difficulty 1)
  - HeuristicAI: Uses strategic heuristics (difficulty 2)
  - MinimaxAI: Depth-limited minimax with alpha-beta pruning (difficulty 3–6)
  - MCTSAI: Monte Carlo tree search with PUCT/RAVE (difficulty 7–8)
  - DescentAI: UBFM/Descent-style tree search (difficulty 9–10)

  These types are wired through the **canonical 1–10 difficulty ladder** defined in [`app/main.py`](ai-service/app/main.py) and mirrored in the TypeScript backend’s `AI_DIFFICULTY_PRESETS` in [`AIEngine.ts`](../src/server/game/ai/AIEngine.ts:1). For a given numeric difficulty, both the backend and service agree on the underlying AI type, randomness, and think-time budget.

- **Experimental / ML-focused work (in progress)**:
  - NeuralNetAI scaffolding lives in [`neural_net.py`](ai-service/app/ai/neural_net.py:1) but is **not** yet part of the production `/ai/move` path.
  - Future ML-backed engines (policy/value networks) will be added as new `AIType` variants and corresponding difficulty profiles.

- **Difficulty Levels**: 1–10, mapped consistently to AI profiles in both the backend and service via:
  - Python: `_CANONICAL_DIFFICULTY_PROFILES` in [`app/main.py`](ai-service/app/main.py)
  - TypeScript: `AI_DIFFICULTY_PRESETS` in [`AIEngine.ts`](../src/server/game/ai/AIEngine.ts:1)
- **Position Evaluation**: Heuristic evaluation with detailed breakdown
- **AI Caching**: Instance caching for performance
- **Health Checks**: Container orchestration support

For end-to-end training, self-play, and dataset-generation workflows that consume this service and its embedded rules engine (including the territory/combined-margin generator), see [`docs/AI_TRAINING_AND_DATASETS.md`](docs/AI_TRAINING_AND_DATASETS.md:1). For the territory forced-elimination / `TerritoryMutator` incident and its fix, see [`docs/INCIDENT_TERRITORY_MUTATOR_DIVERGENCE.md`](docs/INCIDENT_TERRITORY_MUTATOR_DIVERGENCE.md:1).

## Offline Training & Evaluation Scripts

The AI service repo also contains **offline training/diagnostic scripts** that reuse the same
`HeuristicAI` and rules/parity backbone as the production service. These are not part of the
FastAPI surface, but they are important for experimentation and regression analysis.

### CMA-ES Heuristic Optimisation

The canonical CMA-ES harness lives in [`scripts/run_cmaes_optimization.py`](scripts/run_cmaes_optimization.py).
It exposes both a library-style `run_cmaes_optimization(config)` entry point and a CLI:

```bash
cd ai-service
python scripts/run_cmaes_optimization.py \
  --board square8 \
  --generations 4 \
  --population-size 16 \
  --games-per-eval 32 \
  --sigma 0.5 \
  --baseline-profile-id heuristic_v1_balanced \
  --output-dir logs/cmaes \
  --run-id demo_cmaes_square8_v1
```

- Baseline weights are taken from `BASE_V1_BALANCED_WEIGHTS` in
  [`app/ai/heuristic_weights.py`](app/ai/heuristic_weights.py).
- Fitness is computed by `evaluate_fitness(...)` (also used by the genetic harness), which
  plays self-play matches vs the baseline heuristic under the shared TS↔Python rules engine.
- Results are written under `logs/cmaes/runs/<run_id>/`:
  - `run_meta.json`, `baseline_weights.json`, `best_weights.json`.
  - Per-generation summaries in `generations/generation_00N.json`.
  - Checkpoints in `checkpoints/checkpoint_gen00N.json`.

For a higher-level orchestration wrapper that also generates statistical reports, see
[`scripts/run_heuristic_experiment.py`](scripts/run_heuristic_experiment.py) and the
summary in `docs/AI_TRAINING_ASSESSMENT_FINAL.md` (§10 “Extended CMA-ES Tuning Run #1”).

### Genetic Search over Heuristic Weights

An experimental genetic search harness lives in
[`scripts/run_genetic_heuristic_search.py`](scripts/run_genetic_heuristic_search.py). It
explores the same heuristic weight space as CMA-ES but with a simple GA:

- Individuals are `HeuristicWeights` dicts (same keys as `BASE_V1_BALANCED_WEIGHTS`).
- Fitness is delegated to `evaluate_fitness(...)` from `run_cmaes_optimization.py`, so CMA-ES
  and GA share a single fitness definition and plateau diagnostics.
- Selection uses elitism (top-K per generation), with Gaussian per-weight mutation.

Typical invocation (from `ai-service/`):

```bash
python scripts/run_genetic_heuristic_search.py \
  --generations 3 \
  --population-size 8 \
  --elite-count 3 \
  --games-per-eval 16 \
  --sigma 2.0 \
  --board square8 \
  --output-dir logs/ga \
  --run-id ga_v1_square8_demo \
  --seed 12345
```

Key implementation details:

- Supports `--eval-mode` (`initial-only` vs `multi-start`) and `--state-pool-id` to reuse the
  same evaluation-pool infrastructure as CMA-ES (`ai-service/app/training/eval_pools.py`).
- Prints per-individual diagnostics via the `debug_hook` from `evaluate_fitness`, including
  wins/draws/losses and `weight_l2` vs baseline, so you can quickly see whether a run is
  actually exploring distinct policies.
- Logs are written under `logs/ga/runs/<run_id>/`, with `best_weights.json` matching the
  CMA-ES schema (`{"weights": { ... }}`) so downstream tooling can treat GA outputs and
  CMA-ES outputs uniformly.

Both CMA-ES and GA harnesses are covered by the heuristic-training sanity suite
(`ai-service/tests/test_heuristic_training_evaluation.py`) and statistical reporting
pipelines (`scripts/generate_statistical_report.py`), as described in
[`AI_ARCHITECTURE.md`](../AI_ARCHITECTURE.md#56-heuristic-training-sanity--plateau-diagnostics).

### Multi-board heuristic evaluation sanity check

For a quick end-to-end sanity check of the heuristic evaluation stack across all supported
board types, use:

```bash
cd ai-service
python scripts/sanity_check_multiboard_eval.py
```

This script runs a canonical multi-board, multi-start evaluation using
`build_training_eval_kwargs` and compares:

- baseline vs baseline (`BASE_V1_BALANCED_WEIGHTS` vs itself), and
- zero vs baseline (all heuristic weights set to 0.0 vs the baseline).

It should confirm that:

- baseline vs baseline fitness is not structurally pinned at `0.5`, and
- clearly bad weights (all zeros) score significantly worse than the baseline,
  with per-board fitness summaries printed for `square8`, `square19`, and `hexagonal`.

## API Endpoints

### `GET /`

Service information and status

### `GET /health`

Health check endpoint for container orchestration

### `POST /ai/move`

Get AI-selected move for current game state

**Request Body:**

```json
{
  "game_state": {
    /* GameState object */
  },
  "player_number": 1,
  "difficulty": 5,
  "ai_type": "heuristic" // optional
}
```

**Response:**

```json
{
  "move": {
    /* Move object */
  },
  "evaluation": 12.5,
  "thinking_time_ms": 850,
  "ai_type": "heuristic",
  "difficulty": 5
}
```

### `POST /ai/evaluate`

Evaluate position from player's perspective.

**Request Body:**

```json
{
  "game_state": {
    /* GameState object */
  },
  "player_number": 1
}
```

**Response:**

```json
{
  "score": 12.5,
  "breakdown": {
    "total": 12.5,
    "stack_control": 20.0,
    "territory": 5.0,
    "rings_in_hand": 9.0,
    "center_control": 4.0,
    "opponent_threats": -25.5
  }
}
```

### `POST /ai/choice/line_reward_option`

Select a line reward option for an AI-controlled player. This corresponds to the
`line_reward_option` PlayerChoice in the TypeScript engine.

The service currently mirrors the backend heuristic by preferring **Option 2**
(minimum collapse, no elimination) when available, while still accepting
difficulty/ai_type metadata for future smarter policies.

**Request Body (simplified):**

```json
{
  "game_state": {
    /* Optional GameState object; may be null for simple heuristics */
  },
  "player_number": 1,
  "difficulty": 5,
  "ai_type": "heuristic",
  "options": ["option_1_collapse_all_and_eliminate", "option_2_min_collapse_no_elimination"]
}
```

**Response:**

```json
{
  "selectedOption": "option_2_min_collapse_no_elimination",
  "aiType": "heuristic",
  "difficulty": 5
}
```

### `POST /ai/choice/ring_elimination`

Select which stack to eliminate rings from when a line collapse or territory
processing step produces a `ring_elimination` PlayerChoice.

**Request Body (simplified):**

```json
{
  "game_state": {
    /* Optional GameState object; not strictly required for current heuristic */
  },
  "player_number": 1,
  "difficulty": 5,
  "ai_type": "heuristic",
  "options": [
    {
      "stackPosition": { "x": 4, "y": 4 },
      "capHeight": 3,
      "totalHeight": 4
    },
    {
      "stackPosition": { "x": 10, "y": 10 },
      "capHeight": 2,
      "totalHeight": 5
    }
  ]
}
```

The current heuristic prefers the option with the **smallest `capHeight`**, then
breaks ties on **smallest `totalHeight`**.

**Response:**

```json
{
  "selectedOption": {
    "stackPosition": { "x": 10, "y": 10 },
    "capHeight": 2,
    "totalHeight": 5
  },
  "aiType": "heuristic",
  "difficulty": 5
}
```

### `POST /ai/choice/region_order`

Select which disconnected region to process first for a `region_order`
PlayerChoice during territory processing.

**Request Body (simplified):**

```json
{
  "game_state": {
    /* Optional GameState object; used to look for nearby enemy stacks */
  },
  "player_number": 1,
  "difficulty": 5,
  "ai_type": "heuristic",
  "options": [
    {
      "regionId": "A",
      "size": 5,
      "representativePosition": { "x": 10, "y": 10 }
    },
    {
      "regionId": "B",
      "size": 3,
      "representativePosition": { "x": 2, "y": 2 }
    }
  ]
}
```

The current heuristic scores each region by:

- Base score = `size`
- Bonus for nearby enemy-controlled stacks (within a small radius of
  `representativePosition`), with closer stacks contributing more

The region with the highest score is selected; ties are broken in favour of the
larger region.

**Response:**

```json
{
  "selectedOption": {
    "regionId": "A",
    "size": 5,
    "representativePosition": { "x": 10, "y": 10 }
  },
  "aiType": "heuristic",
  "difficulty": 5
}
```

### `DELETE /ai/cache`

Clear cached AI instances

## Development Setup

### Local Development (without Docker)

1. **Install dependencies:**

   ```bash
   cd ai-service
   pip install -r requirements.txt
   ```

2. **Run the service:**

   ```bash
   python -m app.main
   # or
   uvicorn app.main:app --reload --port 8001
   ```

3. **Service will be available at:**
   - http://localhost:8001
   - API docs: http://localhost:8001/docs
   - ReDoc: http://localhost:8001/redoc

### Docker Development

1. **Build the image:**

   ```bash
   docker build -t ringrift-ai-service .
   ```

2. **Run the container:**
   ```bash
   docker run -p 8001:8001 ringrift-ai-service
   ```

### Full Stack with Docker Compose

The root `docker-compose.yml` defines the main application stack, **including this AI service**:

- `app` – Node.js backend (builds from the root [`Dockerfile`](Dockerfile))
- `nginx` – reverse proxy
- `postgres` – PostgreSQL
- `redis` – Redis
- `ai-service` – Python FastAPI AI microservice (exposes port `8001`)
- `prometheus`, `grafana` – observability

To run the full stack in Docker, from the project root:

```bash
docker compose up
# or
docker compose up -d
```

By default:

- The `ai-service` container is built from [`ai-service/Dockerfile`](ai-service/Dockerfile) and started alongside the `app` container.
- The `app` container is configured with `AI_SERVICE_URL=http://ai-service:8001` (see the `AI_SERVICE_URL` entry under the `app` service in `docker-compose.yml`), so you do **not** need to run the AI service separately when using `docker compose up`.

You should only run the AI service on its own (via `uvicorn` or `docker run`) when you are developing or profiling it in isolation from the rest of the stack.

## Environment Variables

- `PYTHON_ENV`: Environment mode (development/production)
- `LOG_LEVEL`: Logging level (default: INFO)
- `RINGRIFT_TRAINED_HEURISTIC_PROFILES` (optional): Path to a JSON file
  produced by `app.training.train_heuristic_weights`. When set and
  `load_trained_profiles_if_available` is called with `mode="override"`, the
  in-memory heuristic profiles will be replaced by the trained values.

## Project Structure

```
ai-service/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI application
│   ├── models/              # Pydantic models mirroring src/shared/types/game.ts
│   │   ├── __init__.py      # Import surface for GameState, Move, choice payloads
│   │   └── core.py          # Core model definitions (BoardState, GameState, choices)
│   └── ai/
│       ├── __init__.py
│       ├── base.py          # Base AI class
│       ├── random_ai.py     # Random AI implementation
│       ├── heuristic_ai.py  # Heuristic AI implementation
│       └── ...              # Future AI implementations (MCTS, neural net, descent, etc.)
├── app/training/
│   ├── generate_territory_dataset.py  # Self-play dataset generation for heuristic training
│   ├── heuristic_features.py          # Feature extraction for heuristic regression
│   ├── train_heuristic_weights.py     # Offline training for heuristic weight profiles
│   └── ...                            # Additional training loops and helpers
├── scripts/
│   ├── run_ai_tournament.py           # Generic AI-vs-AI tournament driver
│   └── run_heuristic_experiment.py    # Baseline-vs-trained and A/B heuristic experiments
├── Dockerfile
├── requirements.txt
└── README.md
```

## AI Implementation Details

### Base AI Class (`base.py`)

- Abstract base class for all AI implementations
- Provides common utilities: thinking simulation, randomness, position helpers
- Defines interface: `select_move()`, `evaluate_position()`

### Random AI (`random_ai.py`)

- Selects random valid moves
- Minimal evaluation (neutral with small variance)
- Used for difficulty levels 1-2

### Heuristic AI (`heuristic_ai.py`)

- Evaluates positions using weighted heuristics:
  - Stack control (10.0)
  - Stack height (5.0)
  - Territory control (8.0)
  - Rings in hand (3.0)
  - Center control (4.0)
  - Opponent threats (6.0)
- Selects moves with highest evaluation
- Used for difficulty levels 3-5

### Offline heuristic training and experiments

The `app/training` and `scripts` modules provide an offline pipeline for
training and evaluating heuristic weight profiles for `HeuristicAI`. High-level CLI usage, dataset schemas, and guidance for these training workflows (including how [`generate_territory_dataset.py`](ai-service/app/training/generate_territory_dataset.py:1) is used to produce combined-margin targets) are documented centrally in [`docs/AI_TRAINING_AND_DATASETS.md`](docs/AI_TRAINING_AND_DATASETS.md:1).

#### 1. Generate self-play datasets

Use `generate_territory_dataset.py` to produce JSONL datasets with
per-state targets for heuristic training. For example, from the `ai-service`
root:

```bash
python -m app.training.generate_territory_dataset \
  --num-games 100 \
  --output logs/heuristic/combined_margin.square8.mixed2p.jsonl \
  --board-type square8 \
  --max-moves 200 \
  --seed 123 \
  --engine-mode mixed \
  --num-players 2
```

This produces a JSONL file where each line has:

- `game_state`: a serialised `GameState` snapshot
- `player_number`: perspective (1..N)
- `target`: scalar combined-margin target for that player
- `time_weight`: optional per-example weight

#### 2. Train heuristic weight profiles

Use `train_heuristic_weights.py` to fit the scalar weights used by
`HeuristicAI` to a dataset:

```bash
python -m app.training.train_heuristic_weights \
  --dataset logs/heuristic/combined_margin.square8.mixed2p.jsonl \
  --output  logs/heuristic/heuristic_profiles.v1.trained.json \
  --lambda  0.001
```

The output JSON contains a mapping from profile ids (e.g.
`"heuristic_v1_balanced"`) to updated weight dictionaries. These are
compatible with the runtime registry in `app.ai.heuristic_weights`.

#### 3. Loading trained profiles at runtime

The helper `load_trained_profiles_if_available` in
`app.ai.heuristic_weights` can be used to register trained profiles:

```python
from app.ai.heuristic_weights import load_trained_profiles_if_available

# Experiment-only: keep baselines and add trained copies under *_trained ids
load_trained_profiles_if_available(
    path="logs/heuristic/heuristic_profiles.v1.trained.json",
    mode="suffix",
    suffix="_trained",
)

# Production override: replace existing ids with trained values
load_trained_profiles_if_available(
    path="logs/heuristic/heuristic_profiles.v1.trained.json",
    mode="override",
)
```

When using the `suffix` mode with the default `"_trained"` suffix, a
baseline profile like `"heuristic_v1_balanced"` will get a trained
counterpart `"heuristic_v1_balanced_trained"` that can be referenced from
`AIConfig.heuristic_profile_id`.

#### 4. Automated baseline-vs-trained and A/B experiments

The script `scripts/run_heuristic_experiment.py` provides a thin CLI for
pitting heuristic profiles against each other using the canonical rules
engine and collecting aggregated stats.

**Baseline vs trained (single file):**

```bash
cd ai-service

python scripts/run_heuristic_experiment.py \
  --mode baseline-vs-trained \
  --trained-profiles-a logs/heuristic/heuristic_profiles.v1.trained.json \
  --base-profile-id-a heuristic_v1_balanced \
  --difficulties 5 \
  --boards Square8 \
  --games-per-match 200 \
  --out-json logs/heuristic/experiments.baseline_vs_trained.json \
  --out-csv  logs/heuristic/experiments.baseline_vs_trained.csv
```

This registers `heuristic_v1_balanced_trained` from the given JSON and runs
matches between:

- Profile A: `heuristic_v1_balanced` (baseline)
- Profile B: `heuristic_v1_balanced_trained` (trained)

for each requested `(difficulty, board)` pairing, swapping sides every other
game for fairness. A summary is printed to stdout and written to the optional
JSON/CSV outputs.

**A/B between two different trained files:**

```bash
cd ai-service

python scripts/run_heuristic_experiment.py \
  --mode ab-trained \
  --trained-profiles-a logs/heuristic/heuristic_profiles.v1.expA.json \
  --trained-profiles-b logs/heuristic/heuristic_profiles.v1.expB.json \
  --base-profile-id-a heuristic_v1_balanced \
  --base-profile-id-b heuristic_v1_balanced \
  --difficulties 3,5,7 \
  --boards Square8,Square19 \
  --games-per-match 200 \
  --out-json logs/heuristic/experiments.expA_vs_expB.json \
  --out-csv  logs/heuristic/experiments.expA_vs_expB.csv
```

In this mode, the script registers profiles like
`"heuristic_v1_balanced_A"` and `"heuristic_v1_balanced_B"` from the two
trained files and reports relative win-rates across the requested grid of
(difficulty, board) conditions.

## Testing

### Manual Testing

```bash
# Health check
curl http://localhost:8001/health

# Get service info
curl http://localhost:8001/

# Test AI move (requires valid game state JSON)
curl -X POST http://localhost:8001/ai/move \
  -H "Content-Type: application/json" \
  -d @test_game_state.json
```

### Interactive API Documentation

Visit http://localhost:8001/docs for Swagger UI with interactive testing

## Distributed Training

The training infrastructure supports distributed training using PyTorch's DistributedDataParallel (DDP) for multi-GPU training on a single machine or across multiple nodes.

### Quick Start

```bash
# Single-node multi-GPU training with torchrun
cd ai-service
./scripts/run_distributed_training.sh 4 \
  --data path/to/data.npz \
  --epochs 100 \
  --scale-lr

# Or use torchrun directly
torchrun --nproc_per_node=4 \
  app/training/train.py \
  --distributed \
  --data path/to/data.npz \
  --epochs 100
```

### Key Features

1. **Automatic Data Sharding**: StreamingDataLoader automatically shards data
   across workers so each GPU processes unique samples:

   ```python
   from app.training.data_loader import StreamingDataLoader
   from app.training.distributed import get_rank, get_world_size

   loader = StreamingDataLoader(
       data_paths=["data1.npz", "data2.npz"],
       batch_size=64,
       rank=get_rank(),
       world_size=get_world_size(),
   )
   # Each worker gets ~total_samples/world_size unique samples
   ```

2. **Versioned Checkpoints**: Uses ModelVersionManager for checkpoint
   management with architecture validation and checksum verification:

   ```python
   from app.training.model_versioning import ModelVersionManager

   manager = ModelVersionManager()

   # Save with full metadata
   metadata = manager.create_metadata(model, training_info={...})
   manager.save_checkpoint(model, metadata, "checkpoint.pth")

   # Load with validation
   state, metadata = manager.load_checkpoint("checkpoint.pth")
   ```

3. **High-Level DistributedTrainer**: Coordinates all components:

   ```python
   from app.training.distributed import DistributedTrainer
   from app.training.config import TrainConfig

   config = TrainConfig(epochs_per_iter=100, batch_size=64)

   trainer = DistributedTrainer(
       config=config,
       data_paths=["data1.npz", "data2.npz"],
       model=my_model,
   )
   trainer.setup()      # Initialize distributed, wrap model, create loaders
   trainer.train()      # Run training loop with metrics
   trainer.cleanup()    # Clean up resources
   ```

### Command Line Arguments

The `train.py` script supports the following distributed training options:

| Argument                   | Description                                |
| -------------------------- | ------------------------------------------ |
| `--distributed`            | Enable distributed training with DDP       |
| `--local-rank`             | Local rank (set automatically by torchrun) |
| `--scale-lr`               | Scale learning rate based on world size    |
| `--lr-scale-mode`          | LR scaling mode: `linear` or `sqrt`        |
| `--find-unused-parameters` | Enable for models with unused params       |

### Checkpoint Synchronization

- **Only rank 0 saves checkpoints** to avoid file conflicts
- **All ranks load checkpoints** on resume for consistency
- **Barrier synchronization** before and after checkpoint operations

```python
# In DistributedTrainer.checkpoint():
synchronize()  # All ranks wait here
if is_main_process():
    # Only rank 0 saves
    manager.save_checkpoint(model, metadata, path)
synchronize()  # All ranks wait until save is complete
```

### Streaming Data Sharding

The StreamingDataLoader partitions samples across workers:

```
Total samples: 1000
World size: 4

Rank 0 gets: samples 0, 4, 8, 12, ... (250 samples)
Rank 1 gets: samples 1, 5, 9, 13, ... (250 samples)
Rank 2 gets: samples 2, 6, 10, 14, ... (250 samples)
Rank 3 gets: samples 3, 7, 11, 15, ... (250 samples)
```

Each epoch shuffles with a deterministic seed so all ranks see consistent
ordering when using the same base seed.

### Multi-Node Training

For multi-node training, set the appropriate environment variables:

```bash
# On node 0 (master)
export MASTER_ADDR=192.168.1.100
export MASTER_PORT=29500
torchrun --nnodes=2 --node_rank=0 --nproc_per_node=4 \
  app/training/train.py --distributed ...

# On node 1
export MASTER_ADDR=192.168.1.100
export MASTER_PORT=29500
torchrun --nnodes=2 --node_rank=1 --nproc_per_node=4 \
  app/training/train.py --distributed ...
```

### Best Practices

1. **Scale learning rate** with `--scale-lr` when using multiple GPUs
2. **Use linear scaling** for small world sizes (2-4 GPUs)
3. **Use sqrt scaling** for larger setups (8+ GPUs)
4. **Monitor per-rank metrics** using DistributedMetrics class
5. **Set consistent seeds** across ranks for reproducibility

## Future Enhancements

- [ ] Productionize Minimax AI with alpha-beta pruning and integrate it into the main `/ai/move` path (building on [`MinimaxAI`](ai-service/app/ai/minimax_ai.py:1) and its tests).
- [ ] Implement full MCTS (Monte Carlo Tree Search) AI and wire it through [`MCTSAI`](ai-service/app/ai/mcts_ai.py:1) in a way that is compatible with the existing REST contract.
- [ ] Train and deploy a neural network–based evaluation for [`NeuralNetAI`](ai-service/app/ai/neural_net.py:1), replacing the current heuristic-style placeholder.
- [ ] Opening book support
- [ ] Endgame tablebases
- [ ] Parallel move generation
- [ ] GPU acceleration for ML models
- [ ] Performance metrics and monitoring

## Integration with TypeScript Backend

The TypeScript backend communicates with this service via the `AIServiceClient`:

```typescript
import { getAIServiceClient } from './services/AIServiceClient';

const client = getAIServiceClient();
const response = await client.getAIMove(gameState, playerNumber, difficulty);
```

The service URL is configured via environment variable:

```
AI_SERVICE_URL=http://ai-service:8001  # Docker
AI_SERVICE_URL=http://localhost:8001   # Local
```

## Canonical Difficulty Ladder

The AI service implements a 10-level difficulty ladder that maps numeric difficulty values to specific AI algorithms and parameters. This mapping is defined in [`_CANONICAL_DIFFICULTY_PROFILES`](app/main.py:1) and mirrored in the TypeScript backend's [`AI_DIFFICULTY_PRESETS`](../src/server/game/ai/AIEngine.ts:1).

| Difficulty | AI Type   | Randomness | Profile ID     | Description                     |
| ---------- | --------- | ---------- | -------------- | ------------------------------- |
| 1          | RANDOM    | 0.5        | v1-random-1    | Random valid moves              |
| 2          | HEURISTIC | 0.3        | v1-heuristic-2 | Basic heuristic with noise      |
| 3          | MINIMAX   | 0.2        | v1-minimax-3   | Shallow search, some randomness |
| 4          | MINIMAX   | 0.15       | v1-minimax-4   | Medium depth                    |
| 5          | MINIMAX   | 0.1        | v1-minimax-5   | Deeper search                   |
| 6          | MINIMAX   | 0.0        | v1-minimax-6   | Deterministic minimax           |
| 7          | MCTS      | 0.0        | v1-mcts-7      | Monte Carlo tree search         |
| 8          | MCTS      | 0.0        | v1-mcts-8      | Extended MCTS                   |
| 9          | DESCENT   | 0.0        | v1-descent-9   | Local descent optimizer         |
| 10         | DESCENT   | 0.0        | v1-descent-10  | Full strength descent           |

## RNG Seeding for Determinism

The AI service supports deterministic behavior through RNG seeding, enabling reproducible AI moves for testing, replay, and parity verification.

### Seed Flow Architecture

```
gameState.rngSeed
    └─> AIServiceClient.getAIMove()
            └─> POST /ai/move { seed: <value> }
                    └─> AIConfig.rng_seed
                            └─> BaseAI.rng (per-instance Random)
```

### API Usage

Pass the `seed` field in the `/ai/move` request body:

```json
{
  "game_state": {
    /* GameState object */
  },
  "player_number": 1,
  "difficulty": 5,
  "seed": 42
}
```

### Python Implementation

Each AI instance in [`base.py`](app/ai/base.py:1) creates a per-instance `random.Random` seeded as follows:

```python
if config.rng_seed is not None:
    seed = config.rng_seed
else:
    # Deterministic fallback from difficulty + player
    seed = (difficulty * 1_000_003) ^ (player_number * 97_911)
self.rng = random.Random(seed)
```

### TypeScript Alignment

The TypeScript side uses the [`SeededRNG`](../src/shared/utils/rng.ts:1) class (xorshift128+) for client-side sandbox AI, and passes seeds through `AIServiceClient` for server-side requests. Key files:

- [`localAIMoveSelection.ts`](../src/shared/engine/localAIMoveSelection.ts:1) - Accepts injectable `LocalAIRng` for deterministic selection
- [`ClientSandboxEngine.ts`](../src/client/sandbox/ClientSandboxEngine.ts:1) - Creates `SeededRNG` from `generateGameSeed()`
- [`AIServiceClient.ts`](../src/server/services/AIServiceClient.ts:1) - Passes `gameState.rngSeed` as `seed` in API requests

### Testing Determinism

**Python tests**:

- [`test_engine_determinism.py`](tests/test_engine_determinism.py:1) - applies a fixed scripted move sequence from a
  canonical initial `GameState` and asserts identical final snapshots and `hash_game_state` values on repeated runs.
- [`test_no_random_in_rules_core.py`](tests/test_no_random_in_rules_core.py:1) - guards against unseeded randomness in
  the Python rules core, mirroring the TS `NoRandomInCoreRules.test.ts` invariant.

**TypeScript tests**:

- [`Sandbox_vs_Backend.aiRngParity.test.ts`](../tests/unit/Sandbox_vs_Backend.aiRngParity.test.ts:1) - RNG plumbing verification and shared seeded-RNG injection across backend and sandbox.
- [`Sandbox_vs_Backend.aiRngFullParity.test.ts`](../tests/unit/Sandbox_vs_Backend.aiRngFullParity.test.ts:1) - Deeper RNG parity coverage (diagnostic/opt-in).
- [`GameSession.aiDeterminism.test.ts`](../tests/integration/GameSession.aiDeterminism.test.ts:1) - End-to-end AI determinism for server-side game sessions.
- [`EngineDeterminism.shared.test.ts`](../tests/unit/EngineDeterminism.shared.test.ts:1) - Shared-engine determinism and turn replay invariants for the canonical TS rules engine.
- [`NoRandomInCoreRules.test.ts`](../tests/unit/NoRandomInCoreRules.test.ts:1) - Guards against unseeded randomness in core shared-engine helpers/aggregates/orchestrator.
- _Historical:_ an earlier `RNGDeterminism.test.ts` suite exercised the raw `SeededRNG` implementation; its coverage is now subsumed by the integrated determinism and "no random in core" suites above.

## Performance Considerations

- AI instances are cached by `{ai_type}-{difficulty}-{player_number}`
- Thinking time simulated for natural feel
- Timeout: 30 seconds for complex AI calculations
- Memory limit: 512MB (configurable in docker-compose.yml)

## Troubleshooting

### Service won't start

- Check Python version (requires 3.11+)
- Verify all dependencies installed: `pip install -r requirements.txt`
- Check port 8001 is not in use

### Connection refused from main app

- Ensure AI service is running
- Check `AI_SERVICE_URL` environment variable
- Verify Docker network if using containers

### Slow AI responses

- Check CPU/memory resources
- Review difficulty setting (higher = slower)
- Consider adjusting timeout in AIServiceClient

## License

Part of the RingRift project. See main project LICENSE.
