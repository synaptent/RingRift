#!/usr/bin/env python3
"""Preflight + sync AI artifacts (NN, NNUE, CMA-ES heuristics) to staging.

This script is a convenience wrapper intended for day-to-day use when wiring
the sandbox against staging:

1) Runs fast local checks (optional):
   - Python AI smoke across all boards (Minimax/MCTS/Descent variants).
   - TypeScript sandbox AI contract tests (route payload + ladder health proxy).
2) Runs a local ladder/artifact health preflight (missing_* counts must be 0).
3) Syncs artifacts to staging via SSH using scripts/sync_staging_ai_artifacts.py.
4) Validates /internal/ladder/health inside the staging ai-service container.

Required environment variables for syncing:
  - RINGRIFT_STAGING_SSH_HOST
  - RINGRIFT_STAGING_ROOT

Optional:
  - RINGRIFT_STAGING_SSH_USER
  - RINGRIFT_STAGING_SSH_PORT
  - RINGRIFT_STAGING_SSH_KEY
  - RINGRIFT_STAGING_COMPOSE_FILE
"""

from __future__ import annotations

import argparse
import asyncio
import os
import subprocess
import sys
from pathlib import Path
from collections.abc import Mapping


AI_SERVICE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = AI_SERVICE_ROOT.parent


def _run(cmd: list[str], *, cwd: Path, env: Mapping[str, str] | None = None) -> None:
    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        env=None if env is None else dict(env),
        text=True,
    )
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


def _preflight_local_ladder_health() -> None:
    """Fail-fast when local artifacts are missing for any canonical tier."""
    trained_profiles = AI_SERVICE_ROOT / "data" / "trained_heuristic_profiles.json"
    if "RINGRIFT_TRAINED_HEURISTIC_PROFILES" not in os.environ and trained_profiles.exists():
        os.environ["RINGRIFT_TRAINED_HEURISTIC_PROFILES"] = str(trained_profiles)

    sys.path.insert(0, str(AI_SERVICE_ROOT))
    from app.main import ladder_health  # type: ignore

    payload = asyncio.run(ladder_health())
    summary = payload.get("summary") if isinstance(payload, dict) else None
    if not isinstance(summary, dict):
        raise SystemExit("Local ladder health returned unexpected payload")

    missing = (
        int(summary.get("missing_heuristic_profiles") or 0)
        + int(summary.get("missing_nnue_checkpoints") or 0)
        + int(summary.get("missing_neural_checkpoints") or 0)
    )
    if missing > 0:
        raise SystemExit(f"Local ladder health reports missing artifacts: {summary}")


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-ts-checks", action="store_true")
    parser.add_argument("--skip-py-checks", action="store_true")
    parser.add_argument("--skip-local-health", action="store_true")
    parser.add_argument("--no-restart", action="store_true")
    parser.add_argument("--include-snapshot-checkpoints", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    if not args.dry_run:
        host = os.environ.get("RINGRIFT_STAGING_SSH_HOST")
        remote_root = os.environ.get("RINGRIFT_STAGING_ROOT")
        if not host or not remote_root:
            print(
                "Missing staging SSH config. Set RINGRIFT_STAGING_SSH_HOST and RINGRIFT_STAGING_ROOT.",
                file=sys.stderr,
            )
            return 2

    if not args.skip_py_checks:
        _run(
            [sys.executable, "-m", "pytest", "-q", "tests/test_ai_smoke_all_boards.py"],
            cwd=AI_SERVICE_ROOT,
        )

    if not args.skip_ts_checks:
        _run(
            [
                "npx",
                "jest",
                "--runInBand",
                "tests/contracts/contractVectorRunner.test.ts",
                "tests/unit/sandboxAiMove.routes.test.ts",
                "tests/unit/sandboxAiLadderHealth.routes.test.ts",
            ],
            cwd=PROJECT_ROOT,
        )

    if not args.skip_local_health:
        _preflight_local_ladder_health()

    sync_cmd = [sys.executable, "scripts/sync_staging_ai_artifacts.py", "--validate-health", "--fail-on-missing"]
    if not args.no_restart:
        sync_cmd.append("--restart")
    if args.include_snapshot_checkpoints:
        sync_cmd.append("--include-snapshot-checkpoints")
    if args.dry_run:
        sync_cmd.append("--dry-run")

    _run(sync_cmd, cwd=AI_SERVICE_ROOT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
