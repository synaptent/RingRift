#!/usr/bin/env python3
"""
Wrapper script to run both TS↔Python replay parity and canonical phase history
checks for a given replay DB. This packages the common gate used when adding or
modifying DBs so callers do not have to remember both commands.

Usage:
  python -m scripts.run_parity_and_history_gate --db /path/to/db.sqlite \
      --emit-state-bundles-dir /tmp/state_bundles

Notes:
- This script is a convenience wrapper; it delegates to:
    - check_ts_python_replay_parity.py
    - check_canonical_phase_history.py
- Any extra args after `--` are forwarded to both underlying scripts.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List, Sequence


def run_cmd(cmd: Sequence[str], cwd: Path) -> int:
    proc = subprocess.run(cmd, cwd=cwd, text=True)
    return proc.returncode


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description="Run both parity and canonical history gates for a replay DB.")
    parser.add_argument(
        "--db",
        required=True,
        help="Path to the SQLite replay DB to validate.",
    )
    parser.add_argument(
        "--emit-state-bundles-dir",
        help="Optional directory to emit state bundles from parity script.",
    )
    parser.add_argument(
        "extra_args",
        nargs="*",
        help="Optional extra args forwarded to both scripts.",
    )

    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parents[2]
    scripts_dir = Path(__file__).resolve().parent

    parity_cmd = [
        sys.executable,
        str(scripts_dir / "check_ts_python_replay_parity.py"),
        "--db",
        args.db,
    ]
    if args.emit_state_bundles_dir:
        parity_cmd += ["--emit-state-bundles-dir", args.emit_state_bundles_dir]
    parity_cmd += args.extra_args

    history_cmd = [
        sys.executable,
        str(scripts_dir / "check_canonical_phase_history.py"),
        "--db",
        args.db,
    ]
    history_cmd += args.extra_args

    print(f"Running parity gate: {' '.join(parity_cmd)}")
    parity_rc = run_cmd(parity_cmd, cwd=repo_root)
    if parity_rc != 0:
        print("Parity gate failed.")
        return parity_rc

    print(f"Running canonical history gate: {' '.join(history_cmd)}")
    history_rc = run_cmd(history_cmd, cwd=repo_root)
    if history_rc != 0:
        print("Canonical history gate failed.")
        return history_rc

    print("Parity + canonical history gates passed.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main(sys.argv[1:]))
