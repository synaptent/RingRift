#!/usr/bin/env python3
"""Deploy RingRift node resilience setup across a cluster via SSH.

This script is intentionally conservative:
- Defaults to dry-run (prints what would run).
- Makes no destructive changes to game DBs or model artifacts.

It installs/updates:
- `/etc/ringrift/node.conf`
- `ringrift-p2p` + `ringrift-resilience` systemd units when available, else cron/watchdog fallback

Usage (dry-run by default):
  python3 ai-service/deploy/deploy_cluster_resilience.py \\
    --coordinator-url "http://192.222.53.22:8770,http://54.198.219.106:8770"

Apply:
  python3 ai-service/deploy/deploy_cluster_resilience.py --apply \\
    --coordinator-url "http://192.222.53.22:8770,http://54.198.219.106:8770"

Force-sync code (overwrites local changes) for broken/dirty workers:
  python3 ai-service/deploy/deploy_cluster_resilience.py --apply --force-sync \\
    --coordinator-url "http://192.222.53.22:8770,http://54.198.219.106:8770"

Optionally distribute the cluster auth token securely (stdin → sudo tee):
  python3 ai-service/deploy/deploy_cluster_resilience.py --apply \\
    --cluster-auth-token-file /path/to/token \\
    --coordinator-url "http://192.222.53.22:8770,http://54.198.219.106:8770"
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


@dataclass(frozen=True)
class HostTarget:
    node_id: str
    ssh_host: str
    ssh_user: str
    ssh_port: int
    ssh_key: Optional[str]
    ringrift_path: str
    p2p_port: Optional[int] = None

    def ssh_target(self) -> str:
        if "@" in self.ssh_host:
            return self.ssh_host
        return f"{self.ssh_user}@{self.ssh_host}"


def _load_hosts(config_path: Path) -> Dict[str, HostTarget]:
    try:
        import yaml  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("PyYAML is required (pip install pyyaml)") from exc

    if not config_path.exists():
        raise FileNotFoundError(f"Missing config: {config_path}")

    data = yaml.safe_load(config_path.read_text()) or {}
    hosts = data.get("hosts", {}) or {}

    targets: Dict[str, HostTarget] = {}
    for node_id, cfg in hosts.items():
        if not isinstance(cfg, dict):
            continue
        status = str(cfg.get("status", "ready") or "ready")
        if status not in {"ready", "setup"}:
            continue
        ssh_host = str(cfg.get("ssh_host", "") or "").strip()
        if not ssh_host:
            continue
        ssh_user = str(cfg.get("ssh_user", "ubuntu") or "ubuntu").strip() or "ubuntu"
        ssh_port = int(cfg.get("ssh_port", 22) or 22)
        ssh_key = cfg.get("ssh_key")
        ringrift_path = str(cfg.get("ringrift_path", "") or "").strip() or "~/ringrift/ai-service"
        p2p_port_raw = cfg.get("p2p_port", None)
        p2p_port: Optional[int] = None
        if p2p_port_raw is not None and str(p2p_port_raw).strip():
            try:
                p2p_port = int(p2p_port_raw)
            except Exception:
                p2p_port = None
        targets[node_id] = HostTarget(
            node_id=node_id,
            ssh_host=ssh_host,
            ssh_user=ssh_user,
            ssh_port=ssh_port,
            ssh_key=ssh_key,
            ringrift_path=ringrift_path,
            p2p_port=p2p_port,
        )
    return targets


def _default_p2p_port_for_node(node_id: str) -> int:
    return 8770


def _derive_p2p_voters(config_path: Path) -> List[str]:
    """Derive the stable voter set from distributed_hosts.yaml.

    This avoids depending on the config file being present on remote nodes. The
    deploy step propagates the voter set via RINGRIFT_P2P_VOTERS into node.conf.
    """
    try:
        import yaml  # type: ignore
    except Exception:
        return []

    if not config_path.exists():
        return []

    try:
        data = yaml.safe_load(config_path.read_text()) or {}
    except Exception:
        return []

    hosts = data.get("hosts", {}) or {}
    voters: List[str] = []
    for node_id, cfg in hosts.items():
        if not isinstance(cfg, dict):
            continue
        raw = cfg.get("p2p_voter", False)
        if raw is True:
            voters.append(str(node_id))
            continue
        if isinstance(raw, (int, float)) and int(raw) == 1:
            voters.append(str(node_id))
            continue
        if isinstance(raw, str) and raw.strip().lower() in {"1", "true", "yes", "y"}:
            voters.append(str(node_id))
    return sorted(set(voters))


def _remote_path_assignment(path: str) -> str:
    """Return a POSIX-shell assignment RHS that expands ~ on the remote."""
    p = (path or "").strip()
    if p == "~":
        return "\"$HOME\""
    if p.startswith("~/"):
        return f"\"$HOME/{p[2:]}\""
    return shlex.quote(p)


def _ssh_base_cmd(target: HostTarget) -> List[str]:
    cmd = [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=15",
        "-o",
        "ServerAliveInterval=30",
        "-o",
        "StrictHostKeyChecking=accept-new",
    ]
    if target.ssh_key:
        cmd.extend(["-o", "IdentitiesOnly=yes"])
        cmd.extend(["-i", os.path.expanduser(target.ssh_key)])
    if int(target.ssh_port) != 22:
        cmd.extend(["-p", str(int(target.ssh_port))])
    cmd.append(target.ssh_target())
    return cmd


def _run_ssh(
    target: HostTarget,
    script: str,
    *,
    stdin_bytes: Optional[bytes] = None,
    dry_run: bool = True,
    login_shell: bool = True,
    wrap_in_bash: bool = True,
) -> None:
    if wrap_in_bash:
        mode_flag = "-lc" if login_shell else "-c"
        # IMPORTANT: ssh executes the remote command through a shell; do not
        # rely on argument boundaries to keep multi-line scripts intact.
        remote_cmd = f"bash {mode_flag} {shlex.quote(script)}"
        cmd = _ssh_base_cmd(target) + [remote_cmd]
    else:
        # Let ssh run the command through the remote user's default shell.
        cmd = _ssh_base_cmd(target) + [script]
    printable = " ".join(shlex.quote(p) for p in cmd)
    if dry_run:
        print(f"[DRY-RUN] {printable}")
        return
    # Vast.ai and some cloud networks can transiently reject SSH auth or drop
    # connections; retry a few times on ssh's generic failure code (255).
    max_attempts = 3
    base_delay_seconds = 2.0
    for attempt in range(1, max_attempts + 1):
        try:
            subprocess.run(cmd, input=stdin_bytes, check=True)
            return
        except subprocess.CalledProcessError as exc:
            if exc.returncode != 255 or attempt >= max_attempts:
                raise
            delay = base_delay_seconds * attempt
            print(f"[WARN] {target.node_id}: ssh failed (255), retrying in {delay:.1f}s...")
            time.sleep(delay)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    default_config = repo_root / "ai-service" / "config" / "distributed_hosts.yaml"

    parser = argparse.ArgumentParser(description="Deploy RingRift resilience setup across cluster hosts")
    parser.add_argument("--config", type=str, default=str(default_config), help="Path to distributed_hosts.yaml")
    parser.add_argument(
        "--coordinator-url",
        type=str,
        required=True,
        help="Comma-separated seed peers for --peers / node.conf COORDINATOR_URL",
    )
    parser.add_argument(
        "--include",
        type=str,
        default="",
        help="Comma-separated node ids to include (default: all ready/setup nodes)",
    )
    parser.add_argument(
        "--exclude",
        type=str,
        default="",
        help="Comma-separated node ids to exclude",
    )
    parser.add_argument(
        "--cluster-auth-token-file",
        type=str,
        default="",
        help="Optional local token file to push to /etc/ringrift/cluster_auth_token (stdin → sudo tee)",
    )
    parser.add_argument(
        "--p2p-voters",
        type=str,
        default="",
        help="Optional comma-separated voter node_ids to propagate as RINGRIFT_P2P_VOTERS (default: derive from config p2p_voter: true).",
    )
    parser.add_argument(
        "--include-macos",
        action="store_true",
        help="Include macOS nodes (runs setup_node_resilience_macos.sh). Default skips mac-studio/mbp-* entries.",
    )
    parser.add_argument(
        "--force-sync",
        action="store_true",
        help="Force `git checkout -f main && git reset --hard origin/main` on targets (overwrites local changes).",
    )
    parser.add_argument("--apply", action="store_true", help="Actually run commands (default: dry-run)")
    args = parser.parse_args()

    config_path = Path(args.config).expanduser()
    targets = _load_hosts(config_path)

    include = {s.strip() for s in (args.include or "").split(",") if s.strip()}
    exclude = {s.strip() for s in (args.exclude or "").split(",") if s.strip()}
    if include:
        targets = {k: v for k, v in targets.items() if k in include}
    if exclude:
        targets = {k: v for k, v in targets.items() if k not in exclude}

    if not args.include_macos:
        targets = {
            k: v
            for k, v in targets.items()
            if not (k == "mac-studio" or k.startswith("mbp-") or k.startswith("mac-"))
        }

    if not targets:
        raise RuntimeError("No matching hosts found in config")

    token_bytes: Optional[bytes] = None
    token_path = (args.cluster_auth_token_file or "").strip()
    if token_path:
        token_bytes = Path(token_path).expanduser().read_text(encoding="utf-8").strip().encode("utf-8")

    explicit_voters = (args.p2p_voters or "").strip()
    if explicit_voters:
        voter_list = [t.strip() for t in explicit_voters.split(",") if t.strip()]
        voter_env = ",".join(sorted(set(voter_list)))
    else:
        voter_env = ",".join(_derive_p2p_voters(config_path))
    voter_prefix = f"RINGRIFT_P2P_VOTERS={shlex.quote(voter_env)} " if voter_env else ""

    failures: List[str] = []

    for node_id, target in sorted(targets.items(), key=lambda kv: kv[0]):
        is_macos = node_id == "mac-studio" or node_id.startswith("mbp-") or node_id.startswith("mac-")
        p2p_port = int(target.p2p_port or _default_p2p_port_for_node(node_id))
        is_root = target.ssh_user == "root" or target.ssh_target().startswith("root@")
        sudo = "" if is_root else "sudo "

        print(f"\n=== {node_id} ({target.ssh_target()}:{target.ssh_port}) ===")

        try:
            if token_bytes:
                if is_macos:
                    token_cmd = (
                        "mkdir -p \"$HOME/Library/Application Support/RingRift\" && "
                        "tee \"$HOME/Library/Application Support/RingRift/cluster_auth_token\" >/dev/null && "
                        "chmod 600 \"$HOME/Library/Application Support/RingRift/cluster_auth_token\""
                    )
                else:
                    # Push token securely via stdin (avoid putting it in argv/env).
                    token_cmd = (
                        f"{sudo}mkdir -p /etc/ringrift && "
                        f"{sudo}tee /etc/ringrift/cluster_auth_token >/dev/null && "
                        f"{sudo}chmod 600 /etc/ringrift/cluster_auth_token"
                    )
                # Avoid wrapping in `bash -c` when piping stdin; on some hosts
                # that combination can cause unexpected sudo noise/failures.
                _run_ssh(
                    target,
                    token_cmd,
                    stdin_bytes=token_bytes,
                    dry_run=not args.apply,
                    login_shell=False,
                    wrap_in_bash=False,
                )

            ringrift_dir_rhs = _remote_path_assignment(target.ringrift_path)
            if args.force_sync:
                git_sync = (
                    "git fetch origin main\n"
                    "git checkout -f main\n"
                    "git reset --hard origin/main\n"
                )
            else:
                git_sync = (
                    "git fetch origin main\n"
                    "git checkout main 2>/dev/null || true\n"
                    "BR=\"$(git rev-parse --abbrev-ref HEAD 2>/dev/null || true)\"\n"
                    "if [ \"$BR\" != \"main\" ]; then\n"
                    "  echo \"[WARN] repo not on main (branch=$BR); skipping git sync (use --force-sync)\" >&2\n"
                    "elif [ -n \"$(git status --porcelain 2>/dev/null || true)\" ]; then\n"
                    "  echo \"[WARN] repo has local changes; skipping git sync (use --force-sync)\" >&2\n"
                    "else\n"
                    "  git merge --ff-only origin/main 2>/dev/null || echo \"[WARN] non-fast-forward; skipping git sync (use --force-sync)\" >&2\n"
                    "fi\n"
                )
            if is_macos:
                remote_setup = (
                    f"set -euo pipefail\n"
                    f"RINGRIFT_DIR={ringrift_dir_rhs}\n"
                    f"RINGRIFT_ROOT=\"$(dirname \"$RINGRIFT_DIR\")\"\n"
                    f"cd \"$RINGRIFT_ROOT\"\n"
                    f"{git_sync}"
                    f"chmod +x \"$RINGRIFT_DIR/deploy/setup_node_resilience_macos.sh\"\n"
                    f"P2P_PORT={int(p2p_port)} RINGRIFT_ROOT=\"$RINGRIFT_ROOT\" "
                    f"{voter_prefix}"
                    f"bash \"$RINGRIFT_DIR/deploy/setup_node_resilience_macos.sh\" "
                    f"{shlex.quote(node_id)} {shlex.quote(args.coordinator_url)}\n"
                )
            else:
                # Ensure the SSH user can update the repo even if root-owned git
                # objects were created by long-running services.
                remote_setup = (
                    f"set -euo pipefail\n"
                    f"RINGRIFT_DIR={ringrift_dir_rhs}\n"
                    f"RINGRIFT_ROOT=\"$(dirname \"$RINGRIFT_DIR\")\"\n"
                    f"mkdir -p \"$RINGRIFT_ROOT\"\n"
                    f"if [ ! -d \"$RINGRIFT_ROOT/.git\" ]; then\n"
                    f"  if [ -n \"$(ls -A \"$RINGRIFT_ROOT\" 2>/dev/null || true)\" ]; then\n"
                    f"    echo \"Error: $RINGRIFT_ROOT exists and is not a git repo\" >&2\n"
                    f"    exit 1\n"
                    f"  fi\n"
                    f"  git clone https://github.com/an0mium/RingRift.git \"$RINGRIFT_ROOT\"\n"
                    f"fi\n"
                    f"{sudo}chown -R \"$(id -un)\":\"$(id -gn)\" \"$RINGRIFT_ROOT/.git\" 2>/dev/null || true\n"
                    f"cd \"$RINGRIFT_ROOT\"\n"
                    f"{git_sync}"
                    f"if [ ! -x \"$RINGRIFT_DIR/venv/bin/python\" ] && [ -x \"$RINGRIFT_DIR/setup.sh\" ] && [ ! -f /.launch ] && [ -z \"${{VAST_CONTAINERLABEL:-}}\" ]; then\n"
                    f"  if command -v apt-get >/dev/null 2>&1; then\n"
                    f"    {sudo}apt-get update -y\n"
                    f"    {sudo}apt-get install -y python3-venv python3-pip\n"
                    f"  fi\n"
                    f"  cd \"$RINGRIFT_DIR\"\n"
                    f"  chmod +x setup.sh 2>/dev/null || true\n"
                    f"  ./setup.sh\n"
                    f"  cd \"$RINGRIFT_ROOT\"\n"
                    f"fi\n"
                    f"chmod +x \"$RINGRIFT_DIR/deploy/setup_node_resilience.sh\" || true\n"
                    f"{sudo}RINGRIFT_DIR=\"$RINGRIFT_DIR\" "
                    f"P2P_PORT={int(p2p_port)} "
                    f"SSH_PORT={int(target.ssh_port)} "
                    f"{voter_prefix}"
                    f"bash \"$RINGRIFT_DIR/deploy/setup_node_resilience.sh\" "
                    f"{shlex.quote(node_id)} {shlex.quote(args.coordinator_url)}\n"
                    # Best-effort restart.
                    # - On systemd hosts: rely on systemd units (avoid duplicate daemons).
                    # - On non-systemd hosts (e.g. Vast.ai): kill + watchdog + nohup daemon.
                    f"HAS_SYSTEMD=0\n"
                    f"if command -v systemctl >/dev/null 2>&1 && [ -d /etc/systemd/system ]; then\n"
                    f"  STATE=\"$(systemctl is-system-running 2>/dev/null || true)\"\n"
                    f"  if [ \"$STATE\" = \"running\" ] || [ \"$STATE\" = \"degraded\" ]; then\n"
                    f"    HAS_SYSTEMD=1\n"
                    f"  fi\n"
                    f"fi\n"
                    f"if [ \"$HAS_SYSTEMD\" = \"1\" ]; then\n"
                    f"  {sudo}systemctl restart ringrift-p2p ringrift-resilience 2>/dev/null || true\n"
                    f"  {sudo}systemctl status ringrift-p2p --no-pager -n 0 2>/dev/null || true\n"
                    f"else\n"
                    f"  {sudo}pkill -f '[p]2p_orchestrator.py' 2>/dev/null || true\n"
                    f"  {sudo}pkill -f '[n]ode_resilience.py' 2>/dev/null || true\n"
                    f"  {sudo}/usr/local/bin/ringrift-watchdog 2>/dev/null || true\n"
                    f"  if [ -f /etc/ringrift/node.conf ]; then\n"
                    f"    set -a\n"
                    f"    source /etc/ringrift/node.conf\n"
                    f"    set +a\n"
                    f"    cd \"$RINGRIFT_DIR\" 2>/dev/null || exit 0\n"
                    f"    {sudo}PYTHONPATH=\"$RINGRIFT_DIR\" nohup python3 scripts/node_resilience.py "
                    f"--node-id \"$NODE_ID\" --coordinator \"$COORDINATOR_URL\" "
                    f"--ai-service-dir \"$RINGRIFT_DIR\" --p2p-port \"$P2P_PORT\" "
                    f">> /var/log/ringrift/resilience.log 2>&1 &\n"
                    f"  fi\n"
                    f"fi\n"
                )

            try:
                _run_ssh(target, remote_setup, dry_run=not args.apply)
            except subprocess.CalledProcessError as exc:
                # Vast.ai sometimes executes the remote command but drops the
                # connection mid-flight, yielding ssh's generic 255 failure code.
                # Treat this as a soft failure iff the local health endpoint is
                # reachable after the deploy.
                if (
                    exc.returncode == 255
                    and args.apply
                    and node_id.lower().startswith("vast-")
                ):
                    print(f"[WARN] {node_id}: ssh returned 255; verifying via localhost /health...")
                    health_check = (
                        f"set -euo pipefail\n"
                        f"PORT=8770\n"
                        f"if [ -f /etc/ringrift/node.conf ]; then\n"
                        f"  set -a\n"
                        f"  source /etc/ringrift/node.conf\n"
                        f"  set +a\n"
                        f"  PORT=\"${{P2P_PORT:-8770}}\"\n"
                        f"fi\n"
                        f"/usr/local/bin/ringrift-watchdog >/dev/null 2>&1 || true\n"
                        f"for i in 1 2 3 4 5 6 7 8 9 10; do\n"
                        f"  if curl -s --connect-timeout 5 \"http://localhost:${{PORT}}/health\" | grep -q '\"node_id\"'; then\n"
                        f"    exit 0\n"
                        f"  fi\n"
                        f"  sleep 2\n"
                        f"done\n"
                        f"exit 1\n"
                    )
                    try:
                        _run_ssh(
                            target,
                            health_check,
                            dry_run=False,
                            login_shell=False,
                            wrap_in_bash=True,
                        )
                        print(f"[WARN] {node_id}: /health OK; continuing despite ssh 255")
                    except Exception:
                        failures.append(node_id)
                        print(f"[ERROR] {node_id}: deploy may not have applied (ssh 255, /health check failed)")
                    continue
                failures.append(node_id)
                print(f"[ERROR] {node_id}: command failed with exit code {exc.returncode}")
                continue
        except subprocess.CalledProcessError as exc:
            failures.append(node_id)
            print(f"[ERROR] {node_id}: command failed with exit code {exc.returncode}")
            continue

    if failures:
        raise SystemExit(f"Deploy failed on {len(failures)} host(s): {', '.join(failures)}")


if __name__ == "__main__":
    main()
