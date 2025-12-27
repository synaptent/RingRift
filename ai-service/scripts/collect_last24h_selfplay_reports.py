#!/usr/bin/env python3
"""Collect last-N-hours selfplay JSONL stats across all configured hosts.

This script is designed for distributed selfplay setups:
1) Each host generates a TSV manifest of JSONL files modified within the last N hours.
2) Each host runs `scripts/analyze_game_statistics.py` locally (fast, low bandwidth).
3) The coordinator (this script) SCPs the per-host manifests + JSON reports to a single
   local run directory and merges them via `scripts/merge_game_statistics_reports.py`.

By default, hosts are sourced from:
- `config/distributed_hosts.yaml` (SSH-based cluster nodes; gitignored)
- `config/remote_hosts.yaml` (legacy fallback; gitignored)

Output layout (local):
  logs/selfplay/collected_last24h/<timestamp>/
    manifests/<host>.tsv
    reports/<host>.json
    combined.json
    combined.md
    collection_summary.json

Usage:
  PYTHONPATH=. python scripts/collect_last24h_selfplay_reports.py
  PYTHONPATH=. python scripts/collect_last24h_selfplay_reports.py --hours 24 --workers 4
  PYTHONPATH=. python scripts/collect_last24h_selfplay_reports.py --hosts lambda-h100 lambda-a10
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

AI_SERVICE_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = AI_SERVICE_ROOT / "config"


@dataclass(frozen=True)
class HostConfig:
    name: str
    ssh_host: str
    ssh_user: str
    ssh_port: int = 22
    ssh_key: str | None = None
    ringrift_path: str = "~/ringrift"
    venv_activate: str | None = None

    @property
    def ssh_target(self) -> str:
        return f"{self.ssh_user}@{self.ssh_host}"

    @property
    def ai_service_dir(self) -> str:
        base = self.ringrift_path.rstrip("/")
        if base.endswith("/ai-service"):
            return base
        return f"{base}/ai-service"


@dataclass
class HostCollectionResult:
    host: str
    ok: bool
    manifest_remote: str | None = None
    report_remote: str | None = None
    error: str | None = None


def _now_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _ssh_base_args(host: HostConfig) -> list[str]:
    args = [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "ConnectTimeout=20",
    ]
    if host.ssh_key:
        args.extend(["-i", os.path.expanduser(host.ssh_key)])
    if host.ssh_port != 22:
        args.extend(["-p", str(host.ssh_port)])
    args.append(host.ssh_target)
    return args


def _scp_base_args(host: HostConfig) -> list[str]:
    args = [
        "scp",
        "-o",
        "BatchMode=yes",
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "ConnectTimeout=20",
    ]
    if host.ssh_key:
        args.extend(["-i", os.path.expanduser(host.ssh_key)])
    if host.ssh_port != 22:
        args.extend(["-P", str(host.ssh_port)])
    return args


def _run(cmd: list[str], *, timeout: int) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def _ssh(host: HostConfig, command: str, *, timeout: int = 600) -> subprocess.CompletedProcess[str]:
    cmd = [*_ssh_base_args(host), command]
    return _run(cmd, timeout=timeout)


def _scp_from(host: HostConfig, remote_path: str, local_path: Path, *, timeout: int = 300) -> None:
    local_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [*_scp_base_args(host), f"{host.ssh_target}:{remote_path}", str(local_path)]
    proc = _run(cmd, timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or f"scp failed: {remote_path}")


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(f"PyYAML is required to parse {path}: {e}") from e
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _load_hosts_from_distributed_config() -> list[HostConfig]:
    path = CONFIG_DIR / "distributed_hosts.yaml"
    if not path.exists():
        return []
    data = _load_yaml(path)
    result: list[HostConfig] = []
    for name, cfg in (data.get("hosts", {}) or {}).items():
        if cfg.get("status") != "ready":
            continue
        ssh_host = str(cfg.get("tailscale_ip") or cfg.get("ssh_host", "")).strip()
        if not ssh_host:
            continue
        result.append(
            HostConfig(
                name=str(name),
                ssh_host=ssh_host,
                ssh_user=str(cfg.get("ssh_user", "ubuntu")),
                ssh_port=int(cfg.get("ssh_port", 22) or 22),
                ssh_key=str(cfg.get("ssh_key")) if cfg.get("ssh_key") else None,
                ringrift_path=str(cfg.get("ringrift_path", "~/ringrift")),
                venv_activate=str(cfg.get("venv_activate")) if cfg.get("venv_activate") else None,
            )
        )
    return result


def _infer_ringrift_path_from_remote_path(remote_path: str) -> str:
    marker = "/ai-service/"
    if marker in remote_path:
        return remote_path.split(marker, 1)[0]
    return os.path.dirname(remote_path) or "~/ringrift"


def _load_aws_extra_from_remote_hosts() -> list[HostConfig]:
    path = CONFIG_DIR / "remote_hosts.yaml"
    if not path.exists():
        return []
    data = _load_yaml(path)
    std = (data.get("standard_hosts", {}) or {}).get("aws_extra")
    if not isinstance(std, dict):
        return []
    ssh_host = str(std.get("ssh_host", "")).strip()
    if not ssh_host:
        return []
    ssh_user = str(std.get("ssh_user", "ubuntu"))
    ssh_key = str(std.get("ssh_key")) if std.get("ssh_key") else None
    ringrift_path = _infer_ringrift_path_from_remote_path(str(std.get("remote_path", "~/ringrift/ai-service")))
    return [
        HostConfig(
            name="aws-selfplay-extra",
            ssh_host=ssh_host,
            ssh_user=ssh_user,
            ssh_key=ssh_key,
            ringrift_path=ringrift_path,
            venv_activate=f"source {ringrift_path}/ai-service/venv/bin/activate",
        )
    ]


def load_all_hosts() -> list[HostConfig]:
    hosts = _load_hosts_from_distributed_config()
    names = {h.name for h in hosts}
    for h in _load_aws_extra_from_remote_hosts():
        if h.name not in names:
            hosts.append(h)
            names.add(h.name)
    return sorted(hosts, key=lambda h: h.name)


def _build_remote_collection_command(*, host: HostConfig, ts: str, hours: float, scan_profile: str) -> str:
    out_dir = f"logs/selfplay/collected_last24h/{ts}"
    manifest_rel = f"{out_dir}/manifests/{host.name}.tsv"
    report_rel = f"{out_dir}/reports/{host.name}.json"

    activate = host.venv_activate.strip() if host.venv_activate else ""
    if activate:
        activate = f"{activate} && "

    python_manifest = """python - <<'PY'
import os
import time
from pathlib import Path

ts = os.environ["RR_TS"]
host = os.environ["RR_HOST"]
hours = float(os.environ["RR_HOURS"])
scan_profile = os.environ.get("RR_SCAN_PROFILE", "broad")

root = Path.cwd()
out_dir = root / "logs" / "selfplay" / "collected_last24h" / ts
cutoff = time.time() - hours * 3600

scan_specs = [
    (root / "data" / "selfplay", True),  # recursive
    (root / "logs" / "selfplay", False),  # top-level only (avoid archived copies)
]
entries = []
for scan_root, recursive in scan_specs:
    if not scan_root.exists():
        continue
    iterator = scan_root.rglob("*.jsonl") if recursive else scan_root.glob("*.jsonl")
    for p in iterator:
        try:
            st = p.stat()
        except OSError:
            continue
        if st.st_size <= 0 or st.st_mtime < cutoff:
            continue
        try:
            rel = str(p.relative_to(root))
        except ValueError:
            rel = str(p)

        if scan_profile == "recent":
            parts = Path(rel).parts
            # Exclude top-level ad-hoc files under data/selfplay/*.jsonl. These are
            # often merged/backfilled archives whose mtime doesn't reflect when
            # the contained games were played.
            if len(parts) == 3 and parts[0] == "data" and parts[1] == "selfplay":
                continue
            # Exclude known legacy/analysis buckets that tend to contain older
            # rulesets (e.g. 48/72 ring supplies) even when recently rewritten.
            if (
                len(parts) >= 4
                and parts[0] == "data"
                and parts[1] == "selfplay"
                and parts[2] in {"new_ruleset", "vast_sync", "combined_gpu", "toxic_archives", "imported"}
            ):
                continue

        entries.append((int(st.st_mtime), int(st.st_size), rel))

entries.sort(key=lambda t: (-t[0], -t[1], t[2]))
out_path = out_dir / "manifests" / f"{host}.tsv"
out_path.parent.mkdir(parents=True, exist_ok=True)
with out_path.open("w", encoding="utf-8") as f:
    for mtime, size, rel in entries:
        f.write(f"{mtime}\\t{size}\\t{rel}\\n")

print(str(out_path))
PY
"""
    return (
        "bash -lc "
        + sh_quote(
            "set -euo pipefail; "
            f"cd {host.ai_service_dir}; "
            f"{activate}"
            f"export RR_TS={sh_quote(ts)} RR_HOST={sh_quote(host.name)} RR_HOURS={sh_quote(str(hours))} "
            f"RR_SCAN_PROFILE={sh_quote(scan_profile)}; "
            f"mkdir -p {sh_quote(out_dir + '/manifests')} {sh_quote(out_dir + '/reports')}; "
            f"{python_manifest}\n"
            "python scripts/analyze_game_statistics.py "
            "--data-dir __no_stats__ "
            f"--jsonl-filelist {sh_quote(manifest_rel)} "
            "--format json "
            f"--output {sh_quote(report_rel)} "
            "--allow-empty --quiet "
            f"--max-age-hours {sh_quote(str(hours))} "
            f"--game-max-age-hours {sh_quote(str(hours))}"
        )
    )


def sh_quote(s: str) -> str:
    return "'" + s.replace("'", "'\"'\"'") + "'"


def collect_from_host(
    host: HostConfig, *, ts: str, hours: float, local_out: Path, deploy: bool, scan_profile: str
) -> HostCollectionResult:
    out_dir = f"logs/selfplay/collected_last24h/{ts}"
    manifest_remote = f"{host.ai_service_dir}/{out_dir}/manifests/{host.name}.tsv"
    report_remote = f"{host.ai_service_dir}/{out_dir}/reports/{host.name}.json"

    try:
        ping = _ssh(host, "echo ok", timeout=20)
        if ping.returncode != 0 or "ok" not in (ping.stdout or ""):
            return HostCollectionResult(host=host.name, ok=False, error=ping.stderr.strip() or "unreachable")

        if deploy:
            local_script = AI_SERVICE_ROOT / "scripts" / "analyze_game_statistics.py"
            remote_script = f"{host.ai_service_dir}/scripts/analyze_game_statistics.py"
            cmd = [*_scp_base_args(host), str(local_script), f"{host.ssh_target}:{remote_script}"]
            proc = _run(cmd, timeout=120)
            if proc.returncode != 0:
                return HostCollectionResult(
                    host=host.name,
                    ok=False,
                    error=proc.stderr.strip() or "failed to deploy analyze_game_statistics.py",
                )

        cmd = _build_remote_collection_command(host=host, ts=ts, hours=hours, scan_profile=scan_profile)
        proc = _ssh(host, cmd, timeout=1800)
        if proc.returncode != 0:
            return HostCollectionResult(
                host=host.name,
                ok=False,
                error=(proc.stderr.strip() or proc.stdout.strip() or "remote command failed"),
            )

        # Fetch artifacts locally.
        _scp_from(host, manifest_remote, local_out / "manifests" / f"{host.name}.tsv")
        _scp_from(host, report_remote, local_out / "reports" / f"{host.name}.json")

        return HostCollectionResult(
            host=host.name, ok=True, manifest_remote=manifest_remote, report_remote=report_remote
        )
    except Exception as e:
        return HostCollectionResult(host=host.name, ok=False, error=str(e))


def collect_local(*, ts: str, hours: float, local_out: Path, scan_profile: str) -> HostCollectionResult:
    out_dir = local_out
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "manifests").mkdir(parents=True, exist_ok=True)
    (out_dir / "reports").mkdir(parents=True, exist_ok=True)

    manifest_path = out_dir / "manifests" / "local.tsv"
    report_path = out_dir / "reports" / "local.json"

    cutoff = datetime.now(timezone.utc).timestamp() - hours * 3600
    scan_specs = [
        (AI_SERVICE_ROOT / "data" / "selfplay", True),  # recursive
        (AI_SERVICE_ROOT / "logs" / "selfplay", False),  # top-level only (avoid archived copies)
    ]

    entries: list[tuple[int, int, str]] = []
    for scan_root, recursive in scan_specs:
        if not scan_root.exists():
            continue
        iterator = scan_root.rglob("*.jsonl") if recursive else scan_root.glob("*.jsonl")
        for p in iterator:
            try:
                st = p.stat()
            except OSError:
                continue
            if st.st_size <= 0 or st.st_mtime < cutoff:
                continue
            rel = str(p.relative_to(AI_SERVICE_ROOT))
            if scan_profile == "recent":
                parts = Path(rel).parts
                if len(parts) == 3 and parts[0] == "data" and parts[1] == "selfplay":
                    continue
                if (
                    len(parts) >= 4
                    and parts[0] == "data"
                    and parts[1] == "selfplay"
                    and parts[2] in {"new_ruleset", "vast_sync", "combined_gpu", "toxic_archives", "imported"}
                ):
                    continue
            entries.append((int(st.st_mtime), int(st.st_size), rel))

    entries.sort(key=lambda t: (-t[0], -t[1], t[2]))
    manifest_path.write_text(
        "\n".join(f"{mtime}\t{size}\t{rel}" for mtime, size, rel in entries) + ("\n" if entries else ""),
        encoding="utf-8",
    )

    analyze = [
        sys.executable,
        str(AI_SERVICE_ROOT / "scripts" / "analyze_game_statistics.py"),
        "--data-dir",
        "__no_stats__",
        "--jsonl-filelist",
        str(manifest_path),
        "--format",
        "json",
        "--output",
        str(report_path),
        "--allow-empty",
        "--quiet",
        "--max-age-hours",
        str(hours),
        "--game-max-age-hours",
        str(hours),
    ]
    proc = _run(analyze, timeout=1800)
    if proc.returncode != 0:
        return HostCollectionResult(host="local", ok=False, error=proc.stderr.strip() or "local analysis failed")

    return HostCollectionResult(host="local", ok=True)


def merge_reports(local_out: Path) -> None:
    reports_dir = local_out / "reports"
    merged_base = local_out / "combined"
    cmd = [
        sys.executable,
        str(AI_SERVICE_ROOT / "scripts" / "merge_game_statistics_reports.py"),
        "--inputs",
        str(reports_dir),
        "--output",
        str(merged_base),
        "--format",
        "both",
    ]
    proc = _run(cmd, timeout=300)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or "merge failed")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hours", type=float, default=24.0, help="Lookback window in hours (default: 24).")
    parser.add_argument("--workers", type=int, default=4, help="Parallel SSH workers (default: 4).")
    parser.add_argument(
        "--scan-profile",
        choices=["broad", "recent"],
        default="broad",
        help=(
            "File selection profile: 'broad' scans all data/selfplay JSONLs (recursive) + top-level logs/selfplay JSONLs; "
            "'recent' excludes top-level data/selfplay/*.jsonl and known legacy/sync buckets (data/selfplay/new_ruleset/**, "
            "data/selfplay/vast_sync/**, data/selfplay/combined_gpu/**, data/selfplay/toxic_archives/**, data/selfplay/imported/**) to reduce stale/backfilled drift."
        ),
    )
    parser.add_argument(
        "--hosts",
        nargs="*",
        help="Optional subset of hosts to collect (names from distributed_hosts.yaml + aws-selfplay-extra).",
    )
    parser.add_argument(
        "--no-deploy",
        action="store_true",
        help="Do not SCP the local analyze_game_statistics.py to remote hosts before running.",
    )
    parser.add_argument(
        "--timestamp",
        default=None,
        help="Override run timestamp (UTC). Default: now.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    ts = str(args.timestamp or _now_ts())
    hours = float(args.hours)
    scan_profile = str(args.scan_profile)

    hosts = load_all_hosts()
    if args.hosts:
        selected = set(args.hosts)
        hosts = [h for h in hosts if h.name in selected]
        missing = sorted(selected - {h.name for h in hosts})
        if missing:
            print(f"Warning: unknown hosts skipped: {', '.join(missing)}", file=sys.stderr)

    if not hosts:
        print("Error: no hosts configured (check config/distributed_hosts.yaml)", file=sys.stderr)
        return 1

    local_out = AI_SERVICE_ROOT / "logs" / "selfplay" / "collected_last24h" / ts
    (local_out / "manifests").mkdir(parents=True, exist_ok=True)
    (local_out / "reports").mkdir(parents=True, exist_ok=True)

    results: list[HostCollectionResult] = []
    results.append(collect_local(ts=ts, hours=hours, local_out=local_out, scan_profile=scan_profile))

    deploy = not args.no_deploy

    from concurrent.futures import ThreadPoolExecutor, as_completed

    with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as pool:
        futures = [
            pool.submit(
                collect_from_host,
                host,
                ts=ts,
                hours=hours,
                local_out=local_out,
                deploy=deploy,
                scan_profile=scan_profile,
            )
            for host in hosts
        ]
        for fut in as_completed(futures):
            results.append(fut.result())

    summary_path = local_out / "collection_summary.json"
    summary = {
        "timestamp": ts,
        "hours": hours,
        "hosts": [asdict(r) for r in sorted(results, key=lambda r: r.host)],
        "ok": [r.host for r in results if r.ok],
        "failed": [r.host for r in results if not r.ok],
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    try:
        merge_reports(local_out)
    except Exception as e:
        print(f"Warning: merge failed: {e}", file=sys.stderr)
        return 2

    ok = [r.host for r in results if r.ok]
    failed = [r.host for r in results if not r.ok]
    print(f"Collected reports: {len(ok)} ok, {len(failed)} failed")
    if failed:
        print("Failed hosts:", ", ".join(sorted(failed)), file=sys.stderr)
    print(f"Output: {local_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
