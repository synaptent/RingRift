#!/usr/bin/env python3
"""Ensure remote cluster hosts have a usable Python environment.

This is a lightweight health check for SSH-based tournament/selfplay tooling.
By default it is read-only: it only reports missing Python dependencies.

Optional remediation is explicit:
  - ``--install`` runs ``python -m pip install -r requirements.txt`` on hosts.
  - ``--bootstrap-venv`` creates/refreshes ``ai-service/venv`` and installs deps.

The host list is read from ``ai-service/config/distributed_hosts.yaml``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

AI_SERVICE_ROOT = Path(__file__).resolve().parents[1]
if str(AI_SERVICE_ROOT) not in sys.path:
    sys.path.insert(0, str(AI_SERVICE_ROOT))

from app.distributed.hosts import HostConfig, SSHExecutor, load_remote_hosts

DEFAULT_REQUIRED_MODULES = (
    "numpy",
    "pydantic",
    "torch",
    "yaml",
    "aiohttp",
)


@dataclass(frozen=True)
class HostEnvReport:
    host: str
    ok: bool
    ssh_targets: list[str]
    python: str
    missing: dict[str, str]
    versions: dict[str, str]
    error: str | None = None
    attempted_install: bool = False
    attempted_bootstrap: bool = False


def _parse_hosts_list(raw: str | None) -> list[str] | None:
    if raw is None:
        return None
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    return parts or None


def _build_check_command(modules: Sequence[str]) -> str:
    # Emit a JSON blob to stdout, even on partial failures.
    mod_list = ",".join(modules)
    return (
        "python -c \""
        "import importlib, json, sys; "
        f"mods='{mod_list}'.split(','); "
        "missing={}; versions={}; "
        "for m in mods: "
        "  try: "
        "    mod=importlib.import_module(m); "
        "    ver=getattr(mod,'__version__',None) or getattr(getattr(mod,'version',None),'__version__',None) or ''; "
        "    versions[m]=str(ver); "
        "  except Exception as e: "
        "    missing[m]=f'{type(e).__name__}:{e}'; "
        "payload={'ok': (len(missing)==0), 'missing': missing, 'versions': versions}; "
        "print(json.dumps(payload)); "
        "sys.exit(0 if payload['ok'] else 10)\""
    )


def _run_remote(host: HostConfig, command: str, timeout: int) -> tuple[int, str, str]:
    executor = SSHExecutor(host)
    result = executor.run(command, timeout=timeout, capture_output=True)
    return result.returncode, result.stdout or "", result.stderr or ""


def _bootstrap_venv(host: HostConfig, *, timeout_sec: int) -> tuple[bool, str]:
    # Best-effort: create ai-service/venv and install requirements.
    cmd = (
        "set -euo pipefail; "
        "if [ ! -d venv ]; then "
        "  (command -v python3 >/dev/null 2>&1 && python3 -m venv venv) "
        "  || python -m venv venv; "
        "fi; "
        ". venv/bin/activate; "
        "python -m pip install --upgrade pip wheel setuptools; "
        "python -m pip install -r requirements.txt"
    )
    rc, out, err = _run_remote(host, cmd, timeout=timeout_sec)
    if rc == 0:
        return True, out.strip()
    msg = (err or out).strip()
    return False, msg[-800:]


def _install_requirements(host: HostConfig, *, timeout_sec: int) -> tuple[bool, str]:
    cmd = (
        "set -euo pipefail; "
        "python -m pip install --upgrade pip wheel setuptools; "
        "python -m pip install -r requirements.txt"
    )
    rc, out, err = _run_remote(host, cmd, timeout=timeout_sec)
    if rc == 0:
        return True, out.strip()
    msg = (err or out).strip()
    return False, msg[-800:]


def _filter_hosts(
    hosts: dict[str, HostConfig],
    *,
    only: Sequence[str] | None,
    include_nonready: bool,
) -> dict[str, HostConfig]:
    selected = hosts
    if only:
        allow = {name.strip() for name in only if name.strip()}
        selected = {k: v for k, v in hosts.items() if k in allow}

    filtered: dict[str, HostConfig] = {}
    for name, cfg in selected.items():
        status = str(cfg.properties.get("status") or "").strip().lower()
        if include_nonready or not status or status == "ready":
            filtered[name] = cfg
    return filtered


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hosts",
        type=str,
        default=None,
        help="Comma-separated host names to check (default: all configured ready hosts).",
    )
    parser.add_argument(
        "--include-nonready",
        action="store_true",
        help="Include hosts whose status is not 'ready'.",
    )
    parser.add_argument(
        "--required",
        type=str,
        default=",".join(DEFAULT_REQUIRED_MODULES),
        help=f"Comma-separated python modules to require (default: {','.join(DEFAULT_REQUIRED_MODULES)}).",
    )
    parser.add_argument(
        "--timeout-sec",
        type=int,
        default=60,
        help="SSH timeout per host for checks (default: 60).",
    )
    parser.add_argument(
        "--install",
        action="store_true",
        help="Attempt to install missing deps via pip (explicit; not default).",
    )
    parser.add_argument(
        "--bootstrap-venv",
        action="store_true",
        help="Create/refresh ai-service/venv and install deps (explicit; not default).",
    )
    parser.add_argument(
        "--json",
        type=str,
        default=None,
        help="Optional path to write a JSON report.",
    )
    args = parser.parse_args(argv)

    required = [m.strip() for m in str(args.required).split(",") if m.strip()]
    if not required:
        raise SystemExit("--required must be non-empty")

    hosts = load_remote_hosts()
    selected = _filter_hosts(
        hosts,
        only=_parse_hosts_list(args.hosts),
        include_nonready=bool(args.include_nonready),
    )
    if not selected:
        raise SystemExit("No eligible hosts found (check --hosts / distributed_hosts.yaml).")

    check_cmd = _build_check_command(required)
    reports: list[HostEnvReport] = []

    started = time.time()
    for name, host in selected.items():
        rc, stdout, stderr = _run_remote(host, check_cmd, timeout=int(args.timeout_sec))
        payload: dict[str, Any] = {}
        if stdout.strip():
            try:
                payload = json.loads(stdout.strip().splitlines()[-1])
            except Exception:
                payload = {}

        missing = payload.get("missing") if isinstance(payload.get("missing"), dict) else {}
        versions = payload.get("versions") if isinstance(payload.get("versions"), dict) else {}
        ok = bool(payload.get("ok")) and rc == 0
        error = None

        attempted_bootstrap = False
        attempted_install = False

        if not ok and (args.bootstrap_venv or args.install):
            if args.bootstrap_venv:
                attempted_bootstrap = True
                success, msg = _bootstrap_venv(host, timeout_sec=int(args.timeout_sec) * 10)
                if not success:
                    error = f"bootstrap_failed: {msg}"
                else:
                    # Re-check after bootstrap.
                    rc, stdout, stderr = _run_remote(host, check_cmd, timeout=int(args.timeout_sec))
                    try:
                        payload = json.loads(stdout.strip().splitlines()[-1]) if stdout.strip() else {}
                    except Exception:
                        payload = {}
            elif args.install:
                attempted_install = True
                success, msg = _install_requirements(host, timeout_sec=int(args.timeout_sec) * 10)
                if not success:
                    error = f"install_failed: {msg}"
                else:
                    rc, stdout, stderr = _run_remote(host, check_cmd, timeout=int(args.timeout_sec))
                    try:
                        payload = json.loads(stdout.strip().splitlines()[-1]) if stdout.strip() else {}
                    except Exception:
                        payload = {}

            missing = payload.get("missing") if isinstance(payload.get("missing"), dict) else {}
            versions = payload.get("versions") if isinstance(payload.get("versions"), dict) else {}
            ok = bool(payload.get("ok")) and rc == 0 and not missing

        if not ok and error is None and rc != 0:
            error = (stderr or stdout).strip()[-400:] or f"rc={rc}"

        python_version = ""
        try:
            _v_rc, v_out, v_err = _run_remote(host, "python -V", timeout=15)
            python_version = (v_out or v_err).strip().splitlines()[-1] if (v_out or v_err) else ""
        except Exception:
            python_version = ""

        report = HostEnvReport(
            host=name,
            ok=ok,
            ssh_targets=list(host.ssh_targets),
            python=python_version,
            missing={str(k): str(v) for k, v in (missing or {}).items()},
            versions={str(k): str(v) for k, v in (versions or {}).items()},
            error=error,
            attempted_install=attempted_install,
            attempted_bootstrap=attempted_bootstrap,
        )
        reports.append(report)

        status = "OK" if report.ok else "FAIL"
        print(f"{name}: {status} {report.python}")
        if report.missing:
            missing_keys = ", ".join(sorted(report.missing.keys()))
            print(f"  missing: {missing_keys}")
        if report.error:
            print(f"  error: {report.error}")

    summary = {
        "checked_hosts": len(reports),
        "ok_hosts": sum(1 for r in reports if r.ok),
        "failed_hosts": sum(1 for r in reports if not r.ok),
        "elapsed_sec": round(time.time() - started, 2),
        "reports": [asdict(r) for r in reports],
    }

    if args.json:
        out_path = Path(args.json).expanduser()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    return 0 if summary["failed_hosts"] == 0 else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

