# RingRift AI Service

Python-based FastAPI microservice for AI move generation and position evaluation.

## Architecture

This service provides AI capabilities for the RingRift game through a RESTful API. It's designed as a microservice to:

- Enable ML/AI libraries (TensorFlow, PyTorch, etc.) without adding them to the Node.js backend
- Allow independent scaling of AI computation
- Facilitate future ML model integration
- Provide language-appropriate AI implementations (Python for ML/AI)

## Features

- **Currently supported AI types (production)**:
  - RandomAI: Selects random valid moves (difficulty 1–2)
  - HeuristicAI: Uses strategic heuristics (difficulty 3–5)

- **Experimental / in-progress AI types**:
  - MinimaxAI: Prototype implementation under test in [`minimax_ai.py`](ai-service/app/ai/minimax_ai.py:1) and [`test_minimax_ai.py`](ai-service/tests/test_minimax_ai.py:1).
  - MCTS / NeuralNetAI: Prototype Monte Carlo tree search + neural net scaffolding in [`mcts_ai.py`](ai-service/app/ai/mcts_ai.py:1) and [`neural_net.py`](ai-service/app/ai/neural_net.py:1).

  > **Note:** These experimental types are not yet fully wired into the main `/ai/move` endpoint for all difficulty levels. The service currently relies primarily on `RandomAI` and `HeuristicAI` for production gameplay.

- **Difficulty Levels**: 1–10, mapped to AI profiles in the backend via [`AIServiceClient`](src/server/services/AIServiceClient.ts:1) and [`AIEngine`](src/server/game/ai/AIEngine.ts:1)
- **Position Evaluation**: Heuristic evaluation with detailed breakdown
- **AI Caching**: Instance caching for performance
- **Health Checks**: Container orchestration support

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
