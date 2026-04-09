# RingRift Quick Start

This guide gets the supported RingRift path running locally:

1. Play the web app.
2. Run the Python AI service.
3. Launch one of the proven training configurations.
4. Inspect the resulting artifacts.

For the current research evidence, see [docs/RESULTS.md](/Users/armand/Development/RingRift/docs/RESULTS.md).

## Prerequisites

- Node.js `18+`
- npm `9+`
- Python `3.10+` (`3.11` recommended)
- Docker and Docker Compose for local Postgres and Redis

## Run the web app

```bash
git clone https://github.com/synaptent/RingRift.git
cd RingRift
npm install
cp .env.example .env
docker compose up -d postgres redis
npm run db:migrate
npm run db:generate
npm run dev
```

Local endpoints:

- Client: `http://localhost:5173`
- Server: `http://localhost:3000`
- Health: `http://localhost:3000/health`

## Run the AI service

```bash
cd ai-service
./setup.sh
./run.sh
```

Local endpoints:

- Health: `http://localhost:8001/health`
- Docs: `http://localhost:8001/docs`

Return to the repo root before the next step:

```bash
cd ..
```

## Reproduce a proven experiment

The simplest supported way to run the same minimal-loop configurations used for the published results is:

```bash
./scripts/run_proven_experiment.sh square8_2p --print-only
./scripts/run_proven_experiment.sh square8_2p --iterations 10
```

Available presets:

- `hex8_2p`
- `square8_2p`

Notes:

- The script writes artifacts to `ai-service/data/proven_experiments/<config>/` by default.
- A single iteration proves the pipeline; multiple iterations are needed before promotions become likely.
- The published results came from longer GH200 cluster runs, so local runtime depends heavily on your GPU.

## Inspect the output

```bash
cat ai-service/data/proven_experiments/square8_2p/summary.json
tail -n 5 ai-service/data/proven_experiments/square8_2p/metrics.jsonl
```

Expected artifacts:

- `metrics.jsonl`: one row per completed iteration
- `summary.json`: a compact machine-readable summary of the latest result
- `models/best.pth`: the best model checkpoint inside the work directory

## Useful validation commands

```bash
npm run test:ts-rules-engine
npm run test:orchestrator-parity
cd ai-service && PYTHONPATH=. .venv/bin/pytest tests/unit/scripts/test_minimal_alphazero_loop.py
```

## Next reads

- [README.md](/Users/armand/Development/RingRift/README.md)
- [docs/RESULTS.md](/Users/armand/Development/RingRift/docs/RESULTS.md)
- [docs/ARCHITECTURE_OVERVIEW.md](/Users/armand/Development/RingRift/docs/ARCHITECTURE_OVERVIEW.md)
- [docs/REPOSITORY_MAP.md](/Users/armand/Development/RingRift/docs/REPOSITORY_MAP.md)
