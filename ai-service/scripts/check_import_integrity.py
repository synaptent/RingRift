#!/usr/bin/env python3
"""Check that all Python modules under app/ can be imported without errors.

Catches deleted symbols, renamed modules, broken imports, and circular
dependencies BEFORE they reach the cluster.

Exit code 0 = all imports pass, 1 = failures found.

Usage:
    cd ai-service && PYTHONPATH=. python3 scripts/check_import_integrity.py
"""
from __future__ import annotations

import importlib
import os
import pkgutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Modules that are known to fail on CI (missing native deps, etc.)
SKIP_MODULES = {
    "app.ai.gpu_parallel_games",  # Requires CUDA
    "app.ai.gpu_batch_state",     # Requires CUDA
}


def main():
    t0 = time.time()
    failures = []
    success = 0
    skipped = 0

    for info in pkgutil.walk_packages(
        [str(ROOT / "app")], prefix="app."
    ):
        if info.name in SKIP_MODULES:
            skipped += 1
            continue
        try:
            importlib.import_module(info.name)
            success += 1
        except Exception as e:
            err_type = type(e).__name__
            failures.append((info.name, f"{err_type}: {e}"))

    elapsed = time.time() - t0

    print(f"Import integrity check: {success} passed, {len(failures)} failed, "
          f"{skipped} skipped ({elapsed:.1f}s)")

    if failures:
        print(f"\nFAILED IMPORTS ({len(failures)}):")
        for name, err in sorted(failures):
            print(f"  {name}: {err}")
        sys.exit(1)
    else:
        print("All imports OK.")
        sys.exit(0)


if __name__ == "__main__":
    main()
