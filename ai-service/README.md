# RingRift AI Service

Python-based FastAPI microservice for AI move generation and position evaluation.

## Architecture

This service provides AI capabilities for the RingRift game through a RESTful API. It's designed as a microservice to:

- Enable ML/AI libraries (TensorFlow, PyTorch, etc.) without adding them to the Node.js backend
- Allow independent scaling of AI computation
- Facilitate future ML model integration
- Provide language-appropriate AI implementations (Python for ML/AI)

## Features

- **Currently supported AI types (canonical ladder)**:
  - RandomAI: Selects random valid moves (difficulty 1–2)
  - HeuristicAI: Uses strategic heuristics (difficulty 3–5)
  - MinimaxAI: Depth-limited search with evaluation, used for difficulty 6–8
  - MCTSAI: Monte Carlo tree search implementation, used for difficulty 9–10
  - DescentAI: Local-descent evaluator used in some parity and evaluation tests

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
