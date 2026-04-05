#!/usr/bin/env python3
"""Post-deploy smoke test for cluster nodes.

Verifies that deployed code is importable and functional. Run on each
node after code update, before restarting services.

Exit code 0 = pass, 1 = fail.

Usage:
    cd ~/ringrift/ai-service && PYTHONPATH=. python3 scripts/deploy_smoke_test.py
    python3 scripts/deploy_smoke_test.py --expected-commit abc123
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

# Allow execution either from the tracked file on disk or via stdin from the
# deploy smoke runner. The runner `cd`s into the ai-service dir first.
_THIS_FILE = globals().get("__file__")
if _THIS_FILE and _THIS_FILE != "<stdin>":
    ROOT = Path(_THIS_FILE).resolve().parent.parent
else:
    ROOT = Path.cwd()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CRITICAL_IMPORTS = [
    "app.models",
    "app.config.env",
    "app.ai.gumbel_mcts_ai",
    "app.training.train",
    "app.training.env",
    "app.board_manager",
    "app.coordination.handler_base",
    "app.coordination.event_router",
    "app.coordination.daemon_manager",
    "scripts.p2p.work_executors.training_executor",
]

OPTIONAL_IMPORTS = [
    "app.coordination.auto_promotion_daemon",
    "app.coordination.evaluation_daemon",
    "app.coordination.training_trigger_daemon",
]


def check_imports() -> list[str]:
    failures = []
    for mod in CRITICAL_IMPORTS:
        try:
            __import__(mod)
        except Exception as e:
            failures.append(f"CRITICAL import {mod}: {e}")

    for mod in OPTIONAL_IMPORTS:
        try:
            __import__(mod)
        except Exception as e:
            # Log but don't fail on optional imports
            print(f"  [WARN] Optional import {mod}: {e}")

    return failures


def check_commit(expected: str | None) -> list[str]:
    if not expected:
        return []
    try:
        r = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10, cwd=str(ROOT)
        )
        actual = r.stdout.strip()[:12]
        exp = expected[:12]
        if actual != exp:
            return [f"Commit mismatch: expected {exp}, got {actual}"]
    except Exception as e:
        return [f"Git check failed: {e}"]
    return []


def check_models(required: bool = False) -> list[str]:
    failures = []
    models_dir = ROOT / "models"
    if not models_dir.exists():
        if required:
            return ["Models directory not found"]
        print("  [WARN] Models directory not found")
        return []

    canonical = list(models_dir.glob("canonical_*.pth"))
    if len(canonical) < 4:
        if required:
            failures.append(f"Only {len(canonical)} canonical models (expected 12)")
        else:
            print(f"  [WARN] Only {len(canonical)} canonical models (expected 12)")
    return failures


def check_device() -> list[str]:
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            print(f"  Device: CUDA ({name})")
            return []
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            print("  Device: MPS")
            return []
        else:
            print("  Device: CPU only")
            return []  # CPU is valid for Hetzner nodes
    except Exception as e:
        return [f"PyTorch device check failed: {e}"]


def check_quick_game() -> list[str]:
    """Run a single fast game to verify the engine works."""
    try:
        from app.models import BoardType, GameStatus
        from app.training.env import TrainingEnvConfig, make_env
        import random

        env = make_env(TrainingEnvConfig(
            board_type=BoardType.HEX8, num_players=2, max_moves=100
        ))
        state = env.reset(seed=42)
        moves = 0
        while state.game_status == GameStatus.ACTIVE and moves < 100:
            legal = env.legal_moves()
            if not legal:
                break
            mv = legal[random.randint(0, len(legal) - 1)]
            state, _, done, _ = env.step(mv)
            moves += 1
            if done:
                break

        if moves < 5:
            return [f"Game too short ({moves} moves)"]
        print(f"  Quick game: {moves} moves, status={state.game_status.value}")
        return []
    except Exception as e:
        return [f"Quick game failed: {e}"]


def main():
    ap = argparse.ArgumentParser(description="Post-deploy smoke test")
    ap.add_argument("--expected-commit", default=None)
    ap.add_argument("--require-models", action="store_true")
    args = ap.parse_args()

    print("=" * 50)
    print("  DEPLOY SMOKE TEST")
    print("=" * 50)

    all_failures = []

    checks = [
        ("Imports", lambda: check_imports()),
        ("Commit", lambda: check_commit(args.expected_commit)),
        ("Models", lambda: check_models(required=args.require_models)),
        ("Device", lambda: check_device()),
        ("Quick game", lambda: check_quick_game()),
    ]

    for name, fn in checks:
        t0 = time.time()
        try:
            failures = fn()
        except Exception as e:
            failures = [f"Check crashed: {e}"]
        elapsed = time.time() - t0

        if failures:
            print(f"  [FAIL] {name} ({elapsed:.1f}s): {failures[0]}")
            all_failures.extend(failures)
        else:
            print(f"  [OK]   {name} ({elapsed:.1f}s)")

    print()
    if all_failures:
        print(f"  FAILED: {len(all_failures)} issues")
        for f in all_failures:
            print(f"    - {f}")
        sys.exit(1)
    else:
        print("  ALL CHECKS PASSED")
        sys.exit(0)


if __name__ == "__main__":
    main()
