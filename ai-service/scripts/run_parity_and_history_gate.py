#!/usr/bin/env python3
"""
Wrapper script to run both TS↔Python replay parity and canonical phase history
checks for a given replay DB. This packages the common gate used when adding or
modifying DBs so callers do not have to remember both commands.

Canonical vs legacy modes
-------------------------

This wrapper primarily targets **canonical gating** runs:

  - Parity is executed via ``check_ts_python_replay_parity.py`` in
    ``--mode canonical --view post_move``.
  - Canonical history is enforced via
    ``check_canonical_phase_history.py``.

For canonical runs, a DB is considered to have passed the combined gate only
when:

  - The parity script reports ``passed_canonical_parity_gate = true`` in its
    JSON summary, and
  - The canonical history checker exits with code 0.

For diagnostic or legacy-analysis runs, you can select ``--parity-mode legacy``
to run the parity harness in non-gating mode while still invoking the history
checker. In legacy mode, structural issues and non-canonical histories are
reported by the parity script but do not by themselves force a non-zero exit
code from this wrapper (unless the underlying tools fail).

Usage:
  python -m scripts.run_parity_and_history_gate --db /path/to/db.sqlite \
      --emit-state-bundles-dir /tmp/state_bundles \
      --summary-json /tmp/db_health.parity_history.json

Notes:
- This script is a convenience wrapper; it delegates to:
    - check_ts_python_replay_parity.py
    - check_canonical_phase_history.py
- Any extra args are forwarded to both underlying scripts.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import List
from collections.abc import Sequence


def run_cmd(cmd: Sequence[str], cwd: Path) -> subprocess.CompletedProcess:
    """Run a subprocess and return the completed process."""
    proc = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True)
    return proc


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Run both TS↔Python replay parity and canonical phase-history gates for a replay DB."
    )
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
        "--summary-json",
        help=(
            "Optional path to write a combined JSON summary containing the parity "
            "summary, canonical history result, and a top-level canonical_ok flag."
        ),
    )
    parser.add_argument(
        "--parity-mode",
        choices=["canonical", "legacy"],
        default="canonical",
        help=(
            "Parity mode to use when invoking check_ts_python_replay_parity.py. "
            "'canonical' (default) enforces the canonical parity gate using "
            "post_move semantics only. 'legacy' runs the harness in diagnostic "
            "mode without treating structural issues as a hard gate."
        ),
    )
    parser.add_argument(
        "extra_args",
        nargs="*",
        help="Optional extra args forwarded to both scripts.",
    )

    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parents[2]
    scripts_dir = Path(__file__).resolve().parent

    # Parity command: always thread through an explicit --mode, and when acting
    # as a canonical gate we also force --view post_move regardless of any view
    # passed via extra_args (our arguments are appended last so they win).
    parity_cmd: list[str] = [
        sys.executable,
        str(scripts_dir / "check_ts_python_replay_parity.py"),
        "--db",
        args.db,
    ]
    if args.emit_state_bundles_dir:
        parity_cmd += ["--emit-state-bundles-dir", args.emit_state_bundles_dir]
    # Forward caller-supplied extras first so our enforced flags take precedence.
    parity_cmd += list(args.extra_args or [])
    parity_cmd += ["--mode", args.parity_mode]
    if args.parity_mode == "canonical":
        parity_cmd += ["--view", "post_move"]

    history_cmd: list[str] = [
        sys.executable,
        str(scripts_dir / "check_canonical_phase_history.py"),
        "--db",
        args.db,
    ]
    history_cmd += list(args.extra_args or [])

    print(f"Running parity gate: {' '.join(parity_cmd)}")
    parity_proc = run_cmd(parity_cmd, cwd=repo_root)

    try:
        parity_summary = json.loads(parity_proc.stdout)
    except Exception:
        parity_summary = {
            "error": "failed_to_parse_parity_summary",
            "stdout": parity_proc.stdout,
            "stderr": parity_proc.stderr,
        }
    parity_summary["returncode"] = parity_proc.returncode

    if parity_proc.returncode != 0:
        print("Parity gate failed.")
    else:
        print("Parity gate passed.")

    print(f"Running canonical history gate: {' '.join(history_cmd)}")
    history_proc = run_cmd(history_cmd, cwd=repo_root)
    history_rc = history_proc.returncode
    canonical_history_ok = history_rc == 0

    if history_rc != 0:
        print("Canonical history gate failed.")
    else:
        print("Canonical history gate passed.")

    # Determine overall canonical_ok status. We only consider the canonical
    # parity gate when parity_mode == 'canonical'.
    passed_canonical_parity_gate = bool(
        args.parity_mode == "canonical"
        and parity_summary.get("passed_canonical_parity_gate")
    )
    canonical_ok = bool(passed_canonical_parity_gate and canonical_history_ok)

    gate_summary = {
        "db_path": args.db,
        "parity_mode": args.parity_mode,
        "parity_summary": parity_summary,
        "canonical_history": {
            "returncode": history_rc,
        },
        "canonical_ok": canonical_ok,
    }

    if args.summary_json:
        summary_path = Path(args.summary_json).resolve()
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(gate_summary, f, indent=2, sort_keys=True)

    if not canonical_ok and args.parity_mode == "canonical":
        # Prefer the parity return code when it failed; otherwise fall back to
        # the history return code or a generic failure code.
        if parity_proc.returncode != 0:
            return parity_proc.returncode
        if history_rc != 0:
            return history_rc
        return 1

    # In legacy mode we simply propagate tool failures but do not treat the
    # parity summary itself as a canonical gate.
    if parity_proc.returncode != 0:
        return parity_proc.returncode
    if history_rc != 0:
        return history_rc

    print("Parity + canonical history gates passed.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main(sys.argv[1:]))
