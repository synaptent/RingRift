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
  - MinimaxAI: Prototype implementation under test in [`minimax_ai.py`](ai-service/app/ai/minimax_ai.py:1) and [`test_minimax_ai.py`](ai-service/tests/test_minimax_ai.py:1); not yet wired into the `/ai/move` path by default.
  - MCTS / NeuralNetAI: Prototype Monte Carlo tree search + neural net scaffolding in [`mcts_ai.py`](ai-service/app/ai/mcts_ai.py:1) and [`neural_net.py`](ai-service/app/ai/neural_net.py:1). Current `/ai/move` requests at higher difficulties still fall back to heuristic-style evaluation; these types are not considered production-ready.

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

Evaluate position from player's perspective

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

The root `docker-compose.yml` currently defines the main application stack:

- `app` – Node.js backend (builds from the root [`Dockerfile`](Dockerfile))
- `nginx` – reverse proxy
- `postgres` – PostgreSQL
- `redis` – Redis
- `prometheus`, `grafana` – observability

It does **not yet include** an `ai-service` service. To use this Python AI service alongside the compose stack:

1. Start the main stack from the project root:

```bash
docker-compose up
```

2. Build and run the AI service container separately as shown above (or run it via `uvicorn` on your host).

If you prefer, you can add an `ai-service` section to `docker-compose.yml` that uses [`ai-service/Dockerfile`](ai-service/Dockerfile) and exposes port `8001`, then update `AI_SERVICE_URL` for the `app` container accordingly.

## Environment Variables

- `PYTHON_ENV`: Environment mode (development/production)
- `LOG_LEVEL`: Logging level (default: INFO)

## Project Structure

```
ai-service/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI application
│   ├── models.py            # Pydantic models
│   └── ai/
│       ├── __init__.py
│       ├── base.py          # Base AI class
│       ├── random_ai.py     # Random AI implementation
│       ├── heuristic_ai.py  # Heuristic AI implementation
│       └── ...              # Future AI implementations
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
