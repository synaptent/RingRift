# Contributing to RingRift

RingRift is both a playable strategy game and a research codebase. The repo contains a large amount of product, rules, AI, training, and operational history, so the first contribution rule is simple:

Work from the supported path first.

If you are new to the project, start here:

1. [README.md](/Users/armand/Development/RingRift/README.md)
2. [docs/PROJECT_BRIEF.md](/Users/armand/Development/RingRift/docs/PROJECT_BRIEF.md)
3. [QUICKSTART.md](/Users/armand/Development/RingRift/QUICKSTART.md)
4. [docs/RESULTS.md](/Users/armand/Development/RingRift/docs/RESULTS.md)
5. [docs/ARCHITECTURE_OVERVIEW.md](/Users/armand/Development/RingRift/docs/ARCHITECTURE_OVERVIEW.md)
6. [docs/REPOSITORY_MAP.md](/Users/armand/Development/RingRift/docs/REPOSITORY_MAP.md)

## What Is Canonical

The project has clear sources of truth.

- Rules semantics:
  - [RULES_CANONICAL_SPEC.md](/Users/armand/Development/RingRift/RULES_CANONICAL_SPEC.md)
  - [docs/rules/COMPLETE_RULES.md](/Users/armand/Development/RingRift/docs/rules/COMPLETE_RULES.md)
  - [docs/rules/COMPACT_RULES.md](/Users/armand/Development/RingRift/docs/rules/COMPACT_RULES.md)
- Executable rules engine:
  - [src/shared/engine](/Users/armand/Development/RingRift/src/shared/engine)
  - [src/shared/types](/Users/armand/Development/RingRift/src/shared/types)
- Python parity mirror:
  - [ai-service/app](/Users/armand/Development/RingRift/ai-service/app)
- Supported training engine:
  - [ai-service/scripts/minimal_alphazero_loop.py](/Users/armand/Development/RingRift/ai-service/scripts/minimal_alphazero_loop.py)

If TypeScript and Python disagree on rules behavior, TypeScript is authoritative and Python must be fixed to match.

## Supported Contribution Areas

These are the best contribution targets for most engineers:

- web product improvements under [src/client](/Users/armand/Development/RingRift/src/client) and [src/server](/Users/armand/Development/RingRift/src/server)
- canonical rules-engine changes under [src/shared/engine](/Users/armand/Development/RingRift/src/shared/engine)
- Python AI/parity improvements under [ai-service/app](/Users/armand/Development/RingRift/ai-service/app)
- training-loop and experiment-harness work under [ai-service/scripts](/Users/armand/Development/RingRift/ai-service/scripts)
- documentation improvements along the supported path

These areas exist but are not the best first stop:

- `archive/`
- `docs/archive/`
- older cluster orchestration and daemon surfaces
- internal assistant memory files such as `CLAUDE.md`

## Local Setup

### Web app

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

### AI service

```bash
cd ai-service
./setup.sh
./run.sh
```

For the supported training entry point, use:

```bash
./scripts/run_proven_experiment.sh square8_2p --print-only
```

## Change Expectations

### Rules changes

If you change rules semantics:

1. Update the canonical TypeScript engine first.
2. Update Python mirrors second.
3. Keep rule docs in sync.
4. Preserve canonical replay semantics:
   - no silent phase transitions
   - no silent forced eliminations
   - explicit canonical move recording

### Training or parity changes

If you change training or replay validation behavior:

1. Keep TypeScript ↔ Python parity intact.
2. Prefer canonical replay data only.
3. Do not silently reintroduce legacy training artifacts or non-canonical databases.

### Documentation changes

For docs-only changes, prefer tightening the supported public path rather than expanding historical coverage.

## Validation

Run the smallest sensible validation set for the area you changed.

### Documentation-only changes

Usually enough:

```bash
git diff --check
```

### TypeScript product or rules-engine changes

Core checks:

```bash
npm run test:ts-rules-engine
npm run test:orchestrator-parity
```

If you touched rules, AI integration, WebSocket lifecycle, or backend turn execution, run the heavier gate:

```bash
npm run test:p0-robustness
```

### Python AI or training changes

Run focused pytest coverage near the files you changed. Common examples:

```bash
cd ai-service
PYTHONPATH=. .venv/bin/pytest tests/unit/scripts/test_minimal_alphazero_loop.py
PYTHONPATH=. .venv/bin/pytest tests/unit/training/test_train_cli.py
```

For replay correctness or parity work, use:

```bash
cd ai-service
PYTHONPATH=. python scripts/check_ts_python_replay_parity.py --db <path-to-db>
```

## Pull Request Guidance

Good PRs in this repo are narrow and explicit.

- Explain what changed and why.
- State whether the change affects canonical rules, parity, training, or only docs/UI.
- List the exact validation you ran.
- If the behavior is non-obvious, link the relevant rules or architecture docs.

## Documentation And Reader Experience

If you improve the repo for external readers, prioritize:

- [docs/PROJECT_BRIEF.md](/Users/armand/Development/RingRift/docs/PROJECT_BRIEF.md)
- [README.md](/Users/armand/Development/RingRift/README.md)
- [QUICKSTART.md](/Users/armand/Development/RingRift/QUICKSTART.md)
- [docs/RESULTS.md](/Users/armand/Development/RingRift/docs/RESULTS.md)
- [docs/ARCHITECTURE_OVERVIEW.md](/Users/armand/Development/RingRift/docs/ARCHITECTURE_OVERVIEW.md)
- [docs/REPOSITORY_MAP.md](/Users/armand/Development/RingRift/docs/REPOSITORY_MAP.md)

The goal is not to make every historical subsystem look equally current. The goal is to make the proven path understandable and reproducible.

## Automation Notes

Human contributors can stop here.

If you are using an automation agent, also read:

- [AGENTS.md](/Users/armand/Development/RingRift/AGENTS.md)
- [src/AGENTS.md](/Users/armand/Development/RingRift/src/AGENTS.md)
- [ai-service/AGENTS.md](/Users/armand/Development/RingRift/ai-service/AGENTS.md)
