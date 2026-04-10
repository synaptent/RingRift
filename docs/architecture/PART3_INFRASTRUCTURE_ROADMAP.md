# Part 3 Infrastructure Roadmap

Updated: April 10, 2026

This document records the Part 3 deep infrastructure improvement session so future work can resume without relying on chat history. The goal is to make RingRift easier to understand, verify, operate, and evolve without SSH archaeology or tribal knowledge.

## Constraints

- Do not modify `ai-service/scripts/minimal_alphazero_loop.py` or its support files.
- Do not modify `ai-service/config/distributed_hosts.yaml`.
- Do not modify database files under `data/` or `ai-service/data/`.
- Keep Python 3.10 compatibility: no `match`, no `datetime.UTC`, no `tomllib`.
- Run Python tests from `ai-service/` with `PYTHONPATH=.`.
- Follow `ai-service/AGENTS.md` for AI-service changes.
- Do not change `ai-service/app/main.py` `eval_mode` logic from commit `a1f8c80ff`.
- Leave `ai-service/archive/deprecated_ai/_game_engine_legacy.py` alone if it is the only dirty file.
- Commit every 3-5 tasks and push frequently.

## Current Baseline

- Branch: `main`.
- Known dirty file to leave untouched: `ai-service/archive/deprecated_ai/_game_engine_legacy.py`.
- P2P orchestrator baseline before Part 3: about 4,913 LOC with 14 extracted mixins.
- Coordination module baseline before Part 3: about 297K LOC across 314 files, with 20+ files over 3,000 LOC.
- Training status baseline: `hex8_2p` plateaued near 1968 Elo after 31 iterations and 6 promotions; `square8_2p` near 1602 Elo with 2 promotions but node health unstable; `square8_3p` near 1535 Elo and regressing under seat-fair evaluation; `square8_4p` still baseline near 1500 Elo.

## Roadmap

| Phase | Focus                                           | Target Outcome                                                                                                         | Status      |
| ----- | ----------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------- | ----------- |
| 0     | Roadmap capture                                 | Durable document for the Part 3 goals and remaining work                                                               | In progress |
| 1     | CI fix                                          | Supported-path workflow no longer fails when optional lint tools are absent; contract tests pass locally               | Pending     |
| 2     | P2P orchestrator                                | Extract state, peer discovery, job, process, HTTP, and game-count mixins; reduce `p2p_orchestrator.py` below 3,000 LOC | Pending     |
| 3     | Coordination module                             | Audit >3,000 LOC coordination files and extract repeated execution/lifecycle/strategy patterns                         | Pending     |
| 4     | Script consolidation                            | Inventory 602 scripts, archive deprecated scripts, and add a unified operational CLI                                   | Pending     |
| 5     | Training pipeline quality                       | Document minimal loop contracts, compare legacy behavior, add training pipeline contract tests                         | Pending     |
| 6     | Client code quality                             | Document extraction plans for large client files, run TypeScript checks, and reduce easy `as any` usage                | Pending     |
| 7     | Server code quality                             | Extract major route handlers and document server decomposition targets                                                 | Pending     |
| 8     | Test infrastructure                             | Remove empty tests, detect broken imports, add test-coverage meta-contracts, and clean conftest fixtures               | Pending     |
| 9     | Documentation cleanup                           | Archive stale 2025 docs and refresh current status, results, architecture, developer guide, and repository map         | Pending     |
| 10    | Type safety                                     | Audit `# type: ignore`, narrow bare `except`, add type-safety contracts                                                | Pending     |
| 11    | Config/environment cleanup                      | Archive unused cluster/hyperparameter configs and refresh `.env.*.example` files                                       | Pending     |
| 12    | Archive cleanup                                 | Audit active imports from archive modules and archive unused lambda scripts safely                                     | Pending     |
| 13    | Event system completion                         | Migrate remaining active `emit_event` calls to `safe_emit_event` and add canonical event contracts                     | Pending     |
| 14    | Large file decomposition: `app/ai` and `app/db` | Extract board encoding, MCTS tree logic, and replay validation modules with size contracts                             | Pending     |
| 15    | Large file decomposition: `app/training`        | Extract training data pipeline, checkpointing, and Elo algorithms with size contracts                                  | Pending     |
| 16    | CI workflow consolidation                       | Add composite setup actions for Python AI and Node workflows                                                           | Pending     |
| 17    | Dead code and import cleanup                    | Detect unused app modules, circular imports, star imports, and obvious unused arguments                                | Pending     |
| 18    | Operational resilience                          | Add dead-loop restart and cluster health scripts plus supervisor heartbeat tests                                       | Pending     |
| 19    | Rules engine quality                            | Add parity coverage and rules completeness contracts across all supported board/player configs                         | Pending     |
| 20    | Final verification                              | Run Python tests, TypeScript checks, supported path checks, and update final architecture/audit docs                   | Pending     |

## Verification Rhythm

After each completed phase, run the targeted verification requested for the phase. The default phase gate is:

```bash
cd ai-service && PYTHONPATH=. python -m pytest tests/unit/ tests/contracts/ -x -q --timeout=120
```

For P2P-specific extraction, also run:

```bash
cd ai-service && PYTHONPATH=. python -m pytest tests/unit/p2p/ -x -q
```

For final verification, run:

```bash
cd ai-service && PYTHONPATH=. python -m pytest tests/ -x -q --timeout=300
npx tsc --noEmit
cd ai-service && PYTHONPATH=. python scripts/check_supported_path.py
```

## Resume Notes

If this session pauses before all phases are complete, resume from the first phase marked `Pending` or `In progress`. Do not infer completion from the roadmap alone; verify with git history, tests, and the referenced architecture documents.
