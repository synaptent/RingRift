#!/usr/bin/env python3
"""Sync promoted AI artifacts (NN, NNUE, heuristic weights) to staging.

This script is intended to be called from training/promotion automation
(for example scripts/continuous_improvement_daemon.py) after new artifacts are
published locally:

- Neural net best-alias checkpoints: ai-service/models/ringrift_best_*.pth
- NNUE checkpoints: ai-service/models/nnue/*.pt
- CMA-ES heuristic overrides: ai-service/data/trained_heuristic_profiles.json
- Ladder runtime overrides (optional): ai-service/data/ladder_runtime_overrides.json

The staging host should run docker-compose.staging.yml with volumes that mount:
  - ./ai-service/models -> /app/models
  - ./ai-service/data   -> /app/data

Configuration is driven by CLI flags or environment variables:
  - RINGRIFT_STAGING_SSH_HOST
  - RINGRIFT_STAGING_SSH_USER
  - RINGRIFT_STAGING_SSH_PORT
  - RINGRIFT_STAGING_SSH_KEY
  - RINGRIFT_STAGING_ROOT
  - RINGRIFT_STAGING_COMPOSE_FILE
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import tarfile
import tempfile
import time
from collections.abc import Iterable
from pathlib import Path


def _env_bool(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


_BEST_ALIAS_PTH_RE = re.compile(
    r"^ringrift_best_[a-z0-9]+_[234]p(?:_[a-z0-9]+)?\.pth$"
)


def _is_best_alias_checkpoint(path: Path) -> bool:
    """Return True when path looks like a runtime `ringrift_best_*` alias file.

    This intentionally excludes timestamped snapshot checkpoints like:
      ringrift_best_sq8_2p_20251214_010416.pth
    """

    return bool(_BEST_ALIAS_PTH_RE.match(path.name))


def _gather_files(project_root: Path, *, include_snapshot_checkpoints: bool) -> list[Path]:
    candidates: list[Path] = []

    models_dir = project_root / "ai-service" / "models"
    if models_dir.exists():
        best_checkpoints = sorted(models_dir.glob("ringrift_best_*.pth"))
        if not include_snapshot_checkpoints:
            best_checkpoints = [p for p in best_checkpoints if _is_best_alias_checkpoint(p)]
        candidates.extend(best_checkpoints)
        candidates.extend(sorted(models_dir.glob("ringrift_best_*.meta.json")))

        nnue_dir = models_dir / "nnue"
        if nnue_dir.exists():
            candidates.extend(sorted(nnue_dir.glob("*.pt")))

    data_dir = project_root / "ai-service" / "data"
    for rel in (
        "trained_heuristic_profiles.json",
        "ladder_runtime_overrides.json",
    ):
        path = data_dir / rel
        if path.exists():
            candidates.append(path)

    # Promoted model mapping is written to ai-service/runs/promotion/ at runtime
    # (avoid dirtying git checkouts on worker nodes). Fall back to the seed file.
    promotion_runtime = project_root / "ai-service" / "runs" / "promotion" / "promoted_models.json"
    promotion_seed = data_dir / "promoted_models.json"
    if promotion_runtime.exists():
        candidates.append(promotion_runtime)
    elif promotion_seed.exists():
        candidates.append(promotion_seed)

    # Promotion history log (optional).
    promotion_log_runtime = (
        project_root / "ai-service" / "runs" / "promotion" / "model_promotion_history.json"
    )
    promotion_log_seed = data_dir / "model_promotion_history.json"
    if promotion_log_runtime.exists():
        candidates.append(promotion_log_runtime)
    elif promotion_log_seed.exists():
        candidates.append(promotion_log_seed)

    # De-dup while preserving deterministic ordering.
    seen: set[Path] = set()
    files: list[Path] = []
    for path in candidates:
        try:
            resolved = path.resolve()
        except Exception:
            resolved = path
        if resolved in seen:
            continue
        seen.add(resolved)
        if path.is_file():
            files.append(path)

    return files


def _make_tarball(
    *,
    project_root: Path,
    files: Iterable[Path],
    tar_path: Path,
) -> None:
    with tarfile.open(tar_path, "w:gz") as tar:
        for file_path in files:
            rel = file_path.resolve().relative_to(project_root.resolve())
            tar.add(file_path, arcname=str(rel))


def _run(cmd: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        check=check,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )


def _ssh_base_args(args: argparse.Namespace) -> list[str]:
    ssh_cmd = ["ssh", "-o", "BatchMode=yes"]
    if args.port:
        ssh_cmd.extend(["-p", str(args.port)])
    if args.key:
        ssh_cmd.extend(["-i", str(args.key)])
    dest = args.host if not args.user else f"{args.user}@{args.host}"
    ssh_cmd.append(dest)
    return ssh_cmd


def _scp_base_args(args: argparse.Namespace) -> list[str]:
    scp_cmd = ["scp", "-o", "BatchMode=yes"]
    if args.port:
        scp_cmd.extend(["-P", str(args.port)])
    if args.key:
        scp_cmd.extend(["-i", str(args.key)])
    return scp_cmd


_SAFE_TILDE_PATH_RE = re.compile(r"^~[a-zA-Z0-9_.-]*(/[a-zA-Z0-9_.-]+)*$")  # ~, ~/foo, ~user/foo


def _shell_quote_path(raw: str) -> str:
    """Shell-quote a path, allowing safe ~ expansion when possible."""
    value = str(raw or "").strip()
    if not value:
        return "''"
    if any(ch in value for ch in ("\0", "\n", "\r")):
        raise ValueError("Path contains control characters")
    # Allow tilde expansion for simple, injection-safe paths (no spaces, no metacharacters).
    if value.startswith("~") and _SAFE_TILDE_PATH_RE.match(value):
        return value
    return shlex.quote(value)


def _docker_compose_file(remote_root: str, compose_file: str) -> str:
    # Allow passing an absolute compose file, otherwise treat as remote_root-relative.
    if os.path.isabs(compose_file):
        return compose_file
    return compose_file


def _validate_remote_ladder_health(
    *,
    args: argparse.Namespace,
    remote_root: str,
) -> tuple[bool, str]:
    """Run a ladder/artifact health check inside the staging ai-service container."""
    compose_file = _docker_compose_file(remote_root, str(args.compose_file))
    url = str(args.validate_health_url or "http://localhost:8001/internal/ladder/health")
    timeout = float(args.validate_health_timeout_seconds or 8)
    fail_on_missing = bool(args.fail_on_missing)

    python_code = (
        "import json, sys\n"
        "import httpx\n"
        f"url={json.dumps(url)}\n"
        f"timeout={timeout}\n"
        "resp=httpx.get(url, timeout=timeout)\n"
        "resp.raise_for_status()\n"
        "payload=resp.json() if resp.content else {}\n"
        "summary=payload.get('summary') or {}\n"
        "print(json.dumps(summary, indent=2, sort_keys=True))\n"
        "missing=(\n"
        "  int(summary.get('missing_heuristic_profiles') or 0)\n"
        "  + int(summary.get('missing_nnue_checkpoints') or 0)\n"
        "  + int(summary.get('missing_neural_checkpoints') or 0)\n"
        ")\n"
        f"sys.exit(2 if ({'True' if fail_on_missing else 'False'} and missing>0) else 0)\n"
    )

    remote_cmd = (
        f"cd {_shell_quote_path(remote_root)} && "
        f"docker compose -f {shlex.quote(compose_file)} exec -T ai-service "
        f"python -c {shlex.quote(python_code)}"
    )
    result = _run([*_ssh_base_args(args), remote_cmd], check=False)
    ok = result.returncode == 0
    return ok, result.stdout


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=os.getenv("RINGRIFT_STAGING_SSH_HOST"))
    parser.add_argument("--user", default=os.getenv("RINGRIFT_STAGING_SSH_USER"))
    parser.add_argument("--port", type=int, default=int(os.getenv("RINGRIFT_STAGING_SSH_PORT", "22")))
    parser.add_argument("--key", default=os.getenv("RINGRIFT_STAGING_SSH_KEY"))
    parser.add_argument("--remote-root", default=os.getenv("RINGRIFT_STAGING_ROOT"))
    parser.add_argument(
        "--compose-file",
        default=os.getenv("RINGRIFT_STAGING_COMPOSE_FILE", "docker-compose.staging.yml"),
    )
    parser.add_argument("--restart", action="store_true", help="Restart docker compose services after sync.")
    parser.add_argument(
        "--restart-services",
        default=os.getenv("RINGRIFT_STAGING_RESTART_SERVICES", "ai-service"),
        help="Comma-separated compose service names (default: ai-service).",
    )
    parser.add_argument(
        "--validate-health",
        action="store_true",
        help="After syncing (and optional restart), query /internal/ladder/health inside the ai-service container.",
    )
    parser.add_argument(
        "--validate-health-url",
        default=os.getenv("RINGRIFT_STAGING_LADDER_HEALTH_URL"),
        help="Override the URL used for ladder health validation (default: http://localhost:8001/internal/ladder/health).",
    )
    parser.add_argument(
        "--validate-health-timeout-seconds",
        type=float,
        default=float(os.getenv("RINGRIFT_STAGING_LADDER_HEALTH_TIMEOUT_SECONDS", "8")),
        help="Timeout (seconds) for ladder health validation (default: 8).",
    )
    parser.add_argument(
        "--fail-on-missing",
        action="store_true",
        help="When used with --validate-health, exit non-zero if any missing artifact counts are reported.",
    )
    parser.add_argument(
        "--include-snapshot-checkpoints",
        action="store_true",
        help=(
            "Also sync timestamped ringrift_best_* snapshot checkpoints (large). "
            "Default behaviour only syncs runtime ringrift_best_* alias files."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args(argv)

    project_root = Path(__file__).resolve().parents[2]
    files = _gather_files(
        project_root,
        include_snapshot_checkpoints=bool(args.include_snapshot_checkpoints),
    )
    if not files:
        print("No artifacts found to sync.", file=sys.stderr)
        return 1

    if args.dry_run:
        for path in files:
            print(str(path))
        return 0

    if not args.host:
        print("Missing --host (or RINGRIFT_STAGING_SSH_HOST)", file=sys.stderr)
        return 2
    if not args.remote_root:
        print("Missing --remote-root (or RINGRIFT_STAGING_ROOT)", file=sys.stderr)
        return 2

    remote_root = str(args.remote_root)
    timestamp = int(time.time())
    remote_tar = f"/tmp/ringrift_ai_artifacts_{timestamp}.tar.gz"

    with tempfile.TemporaryDirectory(prefix="ringrift_ai_sync_") as tmp_dir:
        tar_path = Path(tmp_dir) / "artifacts.tar.gz"
        _make_tarball(project_root=project_root, files=files, tar_path=tar_path)

        if args.verbose:
            print(f"[sync] Packaging {len(files)} files -> {tar_path}")

        scp_cmd = [*_scp_base_args(args), str(tar_path), f"{args.user + '@' if args.user else ''}{args.host}:{remote_tar}"]
        result = _run(scp_cmd, check=False)
        if result.returncode != 0:
            print(result.stdout, file=sys.stderr)
            return result.returncode

        extract_cmd = (
            f"mkdir -p {_shell_quote_path(remote_root)} && "
            f"tar -xzf {shlex.quote(remote_tar)} -C {_shell_quote_path(remote_root)} && "
            f"rm -f {shlex.quote(remote_tar)}"
        )
        ssh_cmd = [*_ssh_base_args(args), extract_cmd]
        result = _run(ssh_cmd, check=False)
        if result.returncode != 0:
            print(result.stdout, file=sys.stderr)
            return result.returncode

        if args.restart:
            services = [s.strip() for s in str(args.restart_services or "").split(",") if s.strip()]
            if services:
                compose_file = _docker_compose_file(remote_root, str(args.compose_file))
                restart_cmd = (
                    f"cd {_shell_quote_path(remote_root)} && "
                    f"docker compose -f {shlex.quote(compose_file)} up -d --force-recreate "
                    + shlex.join(services)
                )
                ssh_cmd = [*_ssh_base_args(args), restart_cmd]
                result = _run(ssh_cmd, check=False)
                if result.returncode != 0:
                    print(result.stdout, file=sys.stderr)
                    return result.returncode

        if args.validate_health:
            ok, output = _validate_remote_ladder_health(args=args, remote_root=remote_root)
            if args.verbose:
                print(output)
            if not ok:
                print(output, file=sys.stderr)
                return 3

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
