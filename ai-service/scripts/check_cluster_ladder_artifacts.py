#!/usr/bin/env python3
"""Check ladder artifact health across cluster hosts via SSH.

This is a thin orchestrator around ``scripts/check_ladder_artifacts.py`` that:
  - Loads hosts from ``config/distributed_hosts.yaml`` (same config used by other
    distributed scripts).
  - Runs the local ladder-artifact check on each host over SSH (preferring
    tailscale_ip when configured).
  - Aggregates results into a single JSON report and exit code.

Example:

  PYTHONPATH=. python scripts/check_cluster_ladder_artifacts.py \
    --board-type square8 \
    --load-checkpoints \
    --fail-on-missing \
    --fail-on-corrupt
"""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional
from collections.abc import Sequence

AI_SERVICE_ROOT = Path(__file__).resolve().parents[1]
if str(AI_SERVICE_ROOT) not in sys.path:
    sys.path.insert(0, str(AI_SERVICE_ROOT))

from app.distributed.hosts import HostConfig, SSHExecutor, load_remote_hosts  # noqa: E402


def _shell_join(parts: Sequence[str]) -> str:
    return " ".join(shlex.quote(p) for p in parts)


def _parse_board(value: str | None) -> str | None:
    if value is None:
        return None
    key = value.strip().lower()
    if not key:
        return None
    mapping = {
        "square8": "square8",
        "sq8": "square8",
        "square19": "square19",
        "sq19": "square19",
        "hexagonal": "hexagonal",
        "hex": "hexagonal",
    }
    if key not in mapping:
        raise ValueError(f"Unsupported board_type={value!r}; expected one of {sorted(mapping)}")
    return mapping[key]


def _host_is_ready(host: HostConfig) -> bool:
    status = str(host.properties.get("status", "") or "").strip().lower()
    if not status:
        return True
    return status == "ready"


@dataclass(frozen=True)
class HostCheckResult:
    host: str
    ok: bool
    exit_code: int
    error: str | None
    payload: dict[str, Any] | None


def _run_check_on_host(
    *,
    host_name: str,
    host: HostConfig,
    timeout_sec: int,
    board_type: str | None,
    num_players: int | None,
    difficulty: int | None,
    load_checkpoints: bool,
) -> HostCheckResult:
    executor = SSHExecutor(host)
    cmd = [
        "python",
        "scripts/check_ladder_artifacts.py",
        "--json",
    ]
    if board_type:
        cmd += ["--board-type", board_type]
    if num_players is not None:
        cmd += ["--num-players", str(int(num_players))]
    if difficulty is not None:
        cmd += ["--difficulty", str(int(difficulty))]
    if load_checkpoints:
        cmd.append("--load-checkpoints")

    result = executor.run(_shell_join(cmd), timeout=timeout_sec, capture_output=True)
    if result.returncode != 0:
        combined = (result.stdout or "") + ("\n" + result.stderr if getattr(result, "stderr", None) else "")
        message = combined.strip()[:500] if combined else "remote_check_failed"
        return HostCheckResult(
            host=host_name,
            ok=False,
            exit_code=int(result.returncode),
            error=message,
            payload=None,
        )

    raw = (result.stdout or "").strip()
    try:
        payload = json.loads(raw) if raw else {}
    except Exception as exc:
        return HostCheckResult(
            host=host_name,
            ok=False,
            exit_code=2,
            error=f"failed_to_parse_json: {exc}",
            payload={"raw_stdout": raw[:5000]},
        )

    return HostCheckResult(
        host=host_name,
        ok=True,
        exit_code=0,
        error=None,
        payload=payload,
    )


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/distributed_hosts.yaml")
    parser.add_argument("--hosts", default=None, help="Comma-separated host names (default: all ready hosts).")
    parser.add_argument("--include-nonready", action="store_true", help="Include hosts not marked as ready.")
    parser.add_argument("--timeout-sec", type=int, default=60, help="Per-host SSH timeout (default: 60).")

    parser.add_argument("--board-type", default=None, help="Filter by board type (square8/square19/hexagonal).")
    parser.add_argument("--num-players", type=int, default=None, help="Filter by num players (2/3/4).")
    parser.add_argument("--difficulty", type=int, default=None, help="Filter by difficulty (1..10).")
    parser.add_argument(
        "--load-checkpoints",
        action="store_true",
        help="Attempt to torch.load CNN checkpoints (via scripts/check_ladder_artifacts.py).",
    )
    parser.add_argument("--fail-on-missing", action="store_true")
    parser.add_argument("--fail-on-corrupt", action="store_true")
    parser.add_argument("--output", default=None, help="Optional JSON output path.")
    args = parser.parse_args(argv)

    board_type = _parse_board(args.board_type)
    num_players = args.num_players
    difficulty = args.difficulty

    if num_players is not None and num_players not in (2, 3, 4):
        raise ValueError("num_players must be 2, 3, or 4")
    if difficulty is not None and not (1 <= difficulty <= 10):
        raise ValueError("difficulty must be between 1 and 10")

    hosts = load_remote_hosts(args.config)
    if not hosts:
        print("No hosts loaded (check config/distributed_hosts.yaml)", file=sys.stderr)
        return 2

    requested = None
    if args.hosts:
        requested = [h.strip() for h in args.hosts.split(",") if h.strip()]

    selected: dict[str, HostConfig] = {}
    for name, host in hosts.items():
        if requested is not None and name not in requested:
            continue
        if not args.include_nonready and not _host_is_ready(host):
            continue
        selected[name] = host

    if not selected:
        print("No eligible hosts selected.", file=sys.stderr)
        return 2

    results: list[HostCheckResult] = []
    for name, host in sorted(selected.items(), key=lambda kv: kv[0]):
        res = _run_check_on_host(
            host_name=name,
            host=host,
            timeout_sec=int(args.timeout_sec),
            board_type=board_type,
            num_players=num_players,
            difficulty=difficulty,
            load_checkpoints=bool(args.load_checkpoints),
        )
        results.append(res)
        if not res.ok:
            print(f"[cluster-health] {name}: FAIL ({res.exit_code}) {res.error}", file=sys.stderr)
        else:
            summary = (res.payload or {}).get("summary") or {}
            missing = int(summary.get("missing_neural_checkpoints") or 0) + int(summary.get("missing_nnue_checkpoints") or 0)
            corrupt = int(summary.get("corrupt_neural_checkpoints") or 0)
            print(f"[cluster-health] {name}: ok missing={missing} corrupt={corrupt}", file=sys.stderr)

    aggregate = {
        "filters": {
            "board_type": board_type,
            "num_players": num_players,
            "difficulty": difficulty,
            "load_checkpoints": bool(args.load_checkpoints),
        },
        "hosts": [
            {
                "host": r.host,
                "ok": r.ok,
                "exit_code": r.exit_code,
                "error": r.error,
                "summary": (r.payload or {}).get("summary") if r.payload else None,
            }
            for r in results
        ],
    }

    if args.output:
        out_path = Path(args.output).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(aggregate, indent=2, sort_keys=True), encoding="utf-8")

    hard_fail = False
    for r in results:
        if not r.ok:
            hard_fail = True
            continue
        summary = (r.payload or {}).get("summary") or {}
        missing_total = (
            int(summary.get("missing_heuristic_profiles") or 0)
            + int(summary.get("missing_nnue_checkpoints") or 0)
            + int(summary.get("missing_neural_checkpoints") or 0)
        )
        corrupt_total = int(summary.get("corrupt_neural_checkpoints") or 0)
        if args.fail_on_missing and missing_total > 0:
            hard_fail = True
        if args.fail_on_corrupt and corrupt_total > 0:
            hard_fail = True

    print(json.dumps(aggregate, indent=2, sort_keys=True))
    return 1 if hard_fail else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

