#!/usr/bin/env python3
"""Preflight check for Slurm-backed RingRift execution."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

def _run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, capture_output=True, text=True, check=False)


AI_SERVICE_ROOT = Path(__file__).resolve().parents[1]
if str(AI_SERVICE_ROOT) not in sys.path:
    sys.path.insert(0, str(AI_SERVICE_ROOT))


def _resolve_repo_root(slurm_config, fallback_root: Path) -> Path:
    shared_root = getattr(slurm_config, "shared_root", None)
    if shared_root:
        return Path(shared_root) / getattr(slurm_config, "repo_subdir", "ai-service")
    return fallback_root


def _resolve_path(path: str, repo_root: Path) -> Path:
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = repo_root / resolved
    return resolved


def _print_check(label: str, ok: bool, detail: str | None = None) -> None:
    status = "OK" if ok else "MISSING"
    suffix = f" ({detail})" if detail else ""
    print(f"[{status}] {label}{suffix}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Slurm preflight checks for RingRift.")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional path to unified_loop.yaml (defaults to config/unified_loop.yaml).",
    )
    args = parser.parse_args()

    try:
        from app.config.unified_config import UnifiedConfig, get_config
    except Exception as exc:
        print(f"[ERROR] Failed to import config: {exc}")
        return 2

    if args.config:
        config = UnifiedConfig.from_yaml(args.config)
    else:
        config = get_config()

    slurm = config.slurm
    ai_root = Path(__file__).parent.parent
    repo_root = _resolve_repo_root(slurm, ai_root)

    print("Slurm preflight checks")
    print(f"- execution_backend: {config.execution_backend}")
    print(f"- slurm.enabled: {slurm.enabled}")
    print(f"- repo_root: {repo_root}")

    missing = False
    binaries = ("sbatch", "squeue", "sacct", "sinfo")
    binary_paths = {}
    for binary in binaries:
        path = shutil.which(binary)
        binary_paths[binary] = path
        ok = path is not None
        _print_check(f"{binary}", ok, path)
        if not ok:
            missing = True

    shared_root = getattr(slurm, "shared_root", None)
    if shared_root:
        shared_root_path = Path(shared_root)
        _print_check("shared_root exists", shared_root_path.exists(), str(shared_root_path))
        _print_check("shared_root mounted", os.path.ismount(shared_root_path), str(shared_root_path))
        if not shared_root_path.exists():
            missing = True
    else:
        print("[WARN] slurm.shared_root is unset; assuming local filesystem.")

    _print_check("repo_root exists", repo_root.exists(), str(repo_root))
    if not repo_root.exists():
        missing = True

    job_dir = _resolve_path(slurm.job_dir, repo_root)
    log_dir = _resolve_path(slurm.log_dir, repo_root)
    job_parent = job_dir if job_dir.exists() else job_dir.parent
    log_parent = log_dir if log_dir.exists() else log_dir.parent

    _print_check("job_dir writable", os.access(job_parent, os.W_OK), str(job_dir))
    _print_check("log_dir writable", os.access(log_parent, os.W_OK), str(log_dir))

    if slurm.partition_training or slurm.partition_selfplay or slurm.partition_tournament:
        if not binary_paths.get("sinfo"):
            print("[WARN] sinfo unavailable; skipping partition validation.")
            return 1 if missing else 0
        result = _run(["sinfo", "-h", "-o", "%P"])
        if result.returncode == 0:
            partitions = {line.strip().strip("*") for line in result.stdout.splitlines() if line.strip()}
            print(f"- partitions: {', '.join(sorted(partitions))}")
            for label, part in (
                ("partition_training", slurm.partition_training),
                ("partition_selfplay", slurm.partition_selfplay),
                ("partition_tournament", slurm.partition_tournament),
            ):
                if part:
                    _print_check(label, part in partitions, part)
        else:
            print(f"[WARN] sinfo failed: {result.stderr.strip()}")
            missing = True

    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
