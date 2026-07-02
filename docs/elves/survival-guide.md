# ELVES SURVIVAL GUIDE — READ THIS FILE FIRST

If you are an AI agent resuming after compaction: read this file, then `.elves-session.json`,
then `docs/elves/learnings.md`, then the plan
(`docs/planning/PROJECT_IMPROVEMENT_EPICS_2026-07.md`), then `docs/elves/execution-log.md`.

## Mission

Record 11 project-improvement epics as docs + GitHub issues, then execute them in priority
order (E2 → E1 → E5 → E11 → E10 → …) per the plan. Each batch: contract → implement →
validate → review → document → commit → push → poll PR.

## Run Control

- Run mode: **finite**
- Scope this run: Batch 0 (setup + epics + issues + PR), Batch 1 (E2 write-up),
  Batch 2 (E1 ringrift-env MVP), Batch 3 (E5 puzzle miner MVP) — then scout/Final Completion.
  Remaining epics live as GitHub issues for future runs.
- Runaway threshold: 5 modifications of same file without progress → stop, log, rethink.
- Review-cycle threshold: 3 cycles on a non-actionable finding → resolve with assessment.

## Stop Gate

- Stop allowed right now: **no**
- Stop conditions: all scoped batches complete + Readiness Gate passed, OR genuine blocker,
  OR explicit user stop.

## Non-negotiables

- NEVER modify: `ai-service/scripts/minimal_alphazero_loop.py`, its support libs,
  `ai-service/config/distributed_hosts.yaml`, anything under `ai-service/data/`.
- NEVER commit the user's in-flight files:
  `ai-service/archive/deprecated_ai/_game_engine_legacy.py`,
  `ai-service/scripts/export_replay_dataset.py`, `.security/`,
  `ai-service/scripts/experiments/cmaes_nnue_shadow.py`,
  `docs/planning/CMAES_ONLINE_NNUE_ADAPTATION_PLAN.md`. Stage specific files only.
- TypeScript engine is rules source of truth. Never weaken canonical rules or parity gates.
- Never merge the PR. Never force-push, reset --hard, or rebase pushed branches.
- Python compatibility target: 3.10.
- Databases may be READ for puzzle mining; never written.

## Current Phase

Batch 0 in progress: session setup, epics doc written; next: GitHub labels + 11 epic issues,
commit, push, open PR.

## Next Exact Batch

Batch 0 remainder: create `epic` label; create issues E1–E11 referencing the plan doc; commit
planning + session docs; push; `gh pr create`; record PR number in `.elves-session.json`.

## Tooling

- Branch: `feat/improvement-epics` (PR target: `main`, repo synaptent/RingRift via origin)
- Validation gates: `npm run test:core` (TS, touched surfaces), `cd ai-service && pytest
tests/unit -x -q -k <touched>` (Python, touched surfaces). Full suites are huge — use
  touched-surface proof per batch; broader runs at entropy checks.
- Lint/typecheck: `npm run lint` / `npx tsc --noEmit` for TS work; `ruff check` if configured
  for Python work.
- PYTHONPATH=. required for ai-service scripts; use `python3`.

## Batch Sizing

- Default sprint-sized batches; epics E1/E5 are MVP-scoped in this run (see plan).

## Session Budget

- Started: 2026-07-01. No user return time given; user is offline and asked for autonomous
  execution. Pace: complete scoped batches, then Final Completion with report.
