#!/usr/bin/env python3
"""Submit and monitor a minimal Slurm job for RingRift."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

def _run(cmd: List[str]) -> subprocess.CompletedProcess[str]:
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


def _normalize_job_name(name: str) -> str:
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)
    return safe[:128] if safe else "ringrift"


def _build_script(repo_root: Path, slurm_config, command: str) -> str:
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        f"cd {repo_root}",
    ]
    venv_activate = getattr(slurm_config, "venv_activate", None)
    if venv_activate:
        lines.append(f"source {venv_activate}")
    for cmd in getattr(slurm_config, "setup_commands", []) or []:
        lines.append(str(cmd))
    lines.append(command)
    return "\n".join(lines) + "\n"


def _build_sbatch_args(
    slurm_config,
    work_type: str,
    job_name: str,
    log_dir: Path,
    partition: Optional[str],
    time_limit: Optional[str],
    gpus: Optional[int],
    cpus: Optional[int],
    mem: Optional[str],
) -> List[str]:
    args = [
        "--job-name",
        job_name,
        "--output",
        str(log_dir / f"{job_name}.%j.out"),
        "--error",
        str(log_dir / f"{job_name}.%j.err"),
    ]
    account = getattr(slurm_config, "account", None)
    if account:
        args.extend(["--account", str(account)])
    qos = getattr(slurm_config, "qos", None)
    if qos:
        args.extend(["--qos", str(qos)])
    if partition:
        args.extend(["--partition", partition])
    if time_limit:
        args.extend(["--time", time_limit])
    if gpus and gpus > 0:
        args.extend(["--gres", f"gpu:{gpus}"])
    if cpus:
        args.extend(["--cpus-per-task", str(cpus)])
    if mem:
        args.extend(["--mem", str(mem)])
    args.extend([str(a) for a in getattr(slurm_config, "extra_sbatch_args", []) or []])
    return args


def _get_status(job_id: str) -> Tuple[Optional[str], Optional[str]]:
    result = _run(["squeue", "-j", job_id, "-h", "-o", "%T"])
    if result.returncode == 0:
        state = result.stdout.strip().splitlines()[0] if result.stdout.strip() else ""
        if state:
            return state, None

    result = _run(["sacct", "-j", job_id, "--format=State,ExitCode", "-n", "-P"])
    if result.returncode != 0:
        return None, result.stderr.strip() or result.stdout.strip()

    for line in result.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split("|")
        state = parts[0].strip()
        exit_code = parts[1].strip() if len(parts) > 1 else ""
        if state:
            return state, exit_code or None

    return None, "sacct returned no state"


def main() -> int:
    parser = argparse.ArgumentParser(description="Submit a Slurm smoke test job.")
    parser.add_argument("--config", type=str, default=None, help="Path to unified_loop.yaml override.")
    parser.add_argument("--work-type", choices=["training", "selfplay", "tournament"], default="training")
    parser.add_argument("--partition", type=str, default=None)
    parser.add_argument("--time", dest="time_limit", type=str, default=None)
    parser.add_argument("--cpus", type=int, default=None)
    parser.add_argument("--gpus", type=int, default=None)
    parser.add_argument("--mem", type=str, default=None)
    parser.add_argument(
        "--command",
        type=str,
        default="python -c \"print('slurm_smoke_test_ok')\"",
        help="Command to run inside the Slurm job.",
    )
    parser.add_argument("--timeout", type=int, default=600)
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

    if shutil.which("sbatch") is None:
        print("[ERROR] sbatch not found on PATH. Install Slurm or run on a Slurm head node.")
        return 1

    job_dir = _resolve_path(slurm.job_dir, repo_root)
    log_dir = _resolve_path(slurm.log_dir, repo_root)
    job_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    work_type = args.work_type
    partition = args.partition or getattr(slurm, f"partition_{work_type}", None)
    time_limit = args.time_limit or getattr(slurm, f"default_time_{work_type}", None)
    gpus = args.gpus if args.gpus is not None else getattr(slurm, f"gpus_{work_type}", 0)
    cpus = args.cpus if args.cpus is not None else getattr(slurm, f"cpus_{work_type}", None)
    mem = args.mem if args.mem is not None else getattr(slurm, f"mem_{work_type}", None)

    job_name = _normalize_job_name(f"ringrift-smoke-{work_type}")
    script_body = _build_script(repo_root, slurm, args.command)
    script_path = job_dir / f"{job_name}.{int(time.time())}.sh"
    script_path.write_text(script_body, encoding="utf-8")
    script_path.chmod(0o755)

    sbatch_args = _build_sbatch_args(slurm, work_type, job_name, log_dir, partition, time_limit, gpus, cpus, mem)
    cmd = ["sbatch", "--parsable", *sbatch_args, str(script_path)]
    result = _run(cmd)
    if result.returncode != 0:
        print(f"[ERROR] sbatch failed: {result.stderr.strip()}")
        return 1

    job_id = result.stdout.strip().split(";")[0].strip()
    if not job_id:
        print("[ERROR] sbatch returned no job id.")
        return 1

    print(f"[INFO] Submitted job {job_id}")
    stdout_path = log_dir / f"{job_name}.{job_id}.out"
    stderr_path = log_dir / f"{job_name}.{job_id}.err"

    deadline = time.time() + args.timeout
    while time.time() < deadline:
        state, detail = _get_status(job_id)
        if state:
            normalized = state.split("+")[0].split(":")[0]
            if normalized in ("COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL"):
                print(f"[INFO] Job state: {normalized}")
                print(f"[INFO] stdout: {stdout_path}")
                print(f"[INFO] stderr: {stderr_path}")
                return 0 if normalized == "COMPLETED" else 1
        time.sleep(getattr(slurm, "poll_interval_seconds", 20))

    print(f"[ERROR] Timeout waiting for job {job_id}")
    print(f"[INFO] stdout: {stdout_path}")
    print(f"[INFO] stderr: {stderr_path}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
