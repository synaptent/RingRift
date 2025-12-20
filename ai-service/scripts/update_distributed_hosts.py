#!/usr/bin/env python3
"""Update `config/distributed_hosts.yaml` with one or more hosts.

This script is intended for local operator use. `distributed_hosts.yaml` is
gitignored and often contains private IPs/ports, so keeping edits out of git
diffs reduces accidental leakage.

Examples:
  python3 scripts/update_distributed_hosts.py \\
    --node-id vast-5090-a \\
    --ssh-host 104.188.118.187 \\
    --ssh-user root \\
    --ssh-port 45180 \\
    --ringrift-path /root/ringrift/ai-service \\
    --gpu "RTX 5090" \\
    --cpus 48 \\
    --memory-gb 128 \\
    --role mixed \\
    --write
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Optional


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml  # type: ignore
    except Exception:
        raise RuntimeError("PyYAML is required: pip install pyyaml")

    if not path.exists():
        return {"hosts": {}}

    data = yaml.safe_load(path.read_text()) or {}
    if not isinstance(data, dict):
        raise RuntimeError(f"Invalid YAML root in {path}")
    data.setdefault("hosts", {})
    if not isinstance(data["hosts"], dict):
        raise RuntimeError(f"Invalid 'hosts' mapping in {path}")
    return data


def _dump_yaml(data: dict[str, Any]) -> str:
    try:
        import yaml  # type: ignore
    except Exception:
        raise RuntimeError("PyYAML is required: pip install pyyaml")

    return yaml.safe_dump(data, sort_keys=False)


def _infer_role(explicit_role: str | None, gpu: str) -> str:
    if explicit_role:
        return explicit_role
    return "mixed" if gpu else "selfplay"


def main() -> None:
    parser = argparse.ArgumentParser(description="Update ai-service/config/distributed_hosts.yaml")
    parser.add_argument("--config", default="config/distributed_hosts.yaml", help="Config path (relative to ai-service/)")
    parser.add_argument("--node-id", required=True, help="Host key under hosts: (e.g. vast-5090-a)")
    parser.add_argument("--ssh-host", required=True, help="SSH host/IP (no user@ prefix)")
    parser.add_argument("--ssh-user", default="root", help="SSH user")
    parser.add_argument("--ssh-port", type=int, default=22, help="SSH port")
    parser.add_argument("--ssh-key", default="", help="SSH key path (optional)")
    parser.add_argument("--ringrift-path", default="~/ringrift/ai-service", help="Remote ai-service path")
    parser.add_argument("--venv-activate", default="", help="Remote venv activate snippet (optional)")
    parser.add_argument("--cpus", type=int, default=0, help="CPU cores (optional)")
    parser.add_argument("--memory-gb", type=int, default=0, help="RAM in GB (optional)")
    parser.add_argument("--gpu", default="", help="GPU name/label (optional)")
    parser.add_argument("--role", default=None, help="Role (selfplay/mixed/nn_training/...)")
    parser.add_argument("--status", default="ready", help="Status (ready/disabled/maintenance)")
    parser.add_argument("--write", action="store_true", help="Write changes to disk")
    args = parser.parse_args()

    ai_service_dir = Path(__file__).resolve().parent.parent
    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = ai_service_dir / cfg_path

    data = _load_yaml(cfg_path)
    hosts = data["hosts"]

    entry: dict[str, Any] = {
        "ssh_host": args.ssh_host,
        "ssh_user": args.ssh_user,
        "ssh_port": int(args.ssh_port),
        "ringrift_path": args.ringrift_path,
        "status": args.status,
        "role": _infer_role(args.role, args.gpu),
    }

    if args.ssh_key:
        entry["ssh_key"] = args.ssh_key
    if args.venv_activate:
        entry["venv_activate"] = args.venv_activate
    if args.cpus:
        entry["cpus"] = int(args.cpus)
    if args.memory_gb:
        entry["memory_gb"] = int(args.memory_gb)
    if args.gpu:
        entry["gpu"] = args.gpu

    hosts[args.node_id] = entry
    rendered = _dump_yaml(data)

    if not args.write:
        sys.stdout.write(rendered)
        return

    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(rendered)
    print(f"Wrote {cfg_path}")


if __name__ == "__main__":
    main()

