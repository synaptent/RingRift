#!/usr/bin/env python3
"""Cluster Health Check Script.

Validates connectivity, PyTorch installation, CUDA availability, and disk space
for all nodes defined in cluster.yaml.

Usage:
    python scripts/cluster_health_check.py
    python scripts/cluster_health_check.py --nodes lambda-gh200-f lambda-gh200-g
    python scripts/cluster_health_check.py --fix  # Attempt to fix common issues
"""

from __future__ import annotations

import argparse
import concurrent.futures
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml

SCRIPT_DIR = Path(__file__).parent
AI_SERVICE_ROOT = SCRIPT_DIR.parent
CONFIG_PATH = AI_SERVICE_ROOT / "config" / "cluster.yaml"

# SSH key mapping
SSH_KEYS = {
    "id_ed25519": os.path.expanduser("~/.ssh/id_ed25519"),
    "id_cluster": os.path.expanduser("~/.ssh/id_cluster"),
}

# Default key if not specified
DEFAULT_SSH_KEY = "id_ed25519"


@dataclass
class NodeHealth:
    """Health status for a single node."""
    name: str
    reachable: bool = False
    ssh_error: Optional[str] = None
    pytorch_ok: bool = False
    pytorch_version: Optional[str] = None
    pytorch_error: Optional[str] = None
    cuda_available: bool = False
    cuda_version: Optional[str] = None
    gpu_count: int = 0
    disk_free_gb: float = 0.0
    disk_usage_pct: float = 0.0
    venv_ok: bool = False
    screens_active: list[str] = None

    def __post_init__(self):
        if self.screens_active is None:
            self.screens_active = []

    @property
    def healthy(self) -> bool:
        """Node is fully healthy."""
        return self.reachable and self.pytorch_ok and self.venv_ok

    @property
    def status_emoji(self) -> str:
        if self.healthy:
            return "✅"
        elif self.reachable:
            return "⚠️"
        else:
            return "❌"


def load_cluster_config() -> dict:
    """Load cluster configuration from YAML."""
    if not CONFIG_PATH.exists():
        print(f"Error: Cluster config not found at {CONFIG_PATH}")
        sys.exit(1)

    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def get_ssh_command(node_config: dict) -> list[str]:
    """Build SSH command for a node."""
    host = node_config.get("tailscale_ip") or node_config.get("host")
    user = node_config.get("ssh_user", "ubuntu")
    key_name = node_config.get("ssh_key", DEFAULT_SSH_KEY)
    key_path = SSH_KEYS.get(key_name, SSH_KEYS[DEFAULT_SSH_KEY])
    port = node_config.get("ssh_port", 22)

    cmd = ["ssh", "-i", key_path, "-o", "ConnectTimeout=10",
           "-o", "StrictHostKeyChecking=no", "-o", "BatchMode=yes"]
    if port != 22:
        cmd.extend(["-p", str(port)])
    cmd.append(f"{user}@{host}")
    return cmd


def check_node_health(name: str, config: dict) -> NodeHealth:
    """Check health of a single node."""
    health = NodeHealth(name=name)

    ssh_base = get_ssh_command(config)

    # Check SSH connectivity
    try:
        result = subprocess.run(
            ssh_base + ["echo", "ok"],
            capture_output=True, text=True, timeout=15
        )
        if result.returncode == 0:
            health.reachable = True
        else:
            health.ssh_error = result.stderr.strip()[:100]
            return health
    except subprocess.TimeoutExpired:
        health.ssh_error = "Connection timeout"
        return health
    except Exception as e:
        health.ssh_error = str(e)[:100]
        return health

    # Check PyTorch and CUDA
    pytorch_check = """
cd ~/ringrift/ai-service 2>/dev/null || cd ~/ringrift 2>/dev/null || true
if [ -d "venv" ]; then
    source venv/bin/activate 2>/dev/null
    python3 -c "
import torch
print('PYTORCH_VERSION=' + torch.__version__)
print('CUDA_AVAILABLE=' + str(torch.cuda.is_available()))
if torch.cuda.is_available():
    print('CUDA_VERSION=' + torch.version.cuda)
    print('GPU_COUNT=' + str(torch.cuda.device_count()))
" 2>&1
    echo "VENV_OK=True"
else
    echo "VENV_OK=False"
fi
"""
    try:
        result = subprocess.run(
            ssh_base + ["bash", "-c", pytorch_check],
            capture_output=True, text=True, timeout=30
        )
        output = result.stdout + result.stderr

        for line in output.split("\n"):
            if line.startswith("PYTORCH_VERSION="):
                health.pytorch_ok = True
                health.pytorch_version = line.split("=", 1)[1]
            elif line.startswith("CUDA_AVAILABLE="):
                health.cuda_available = line.split("=", 1)[1] == "True"
            elif line.startswith("CUDA_VERSION="):
                health.cuda_version = line.split("=", 1)[1]
            elif line.startswith("GPU_COUNT="):
                health.gpu_count = int(line.split("=", 1)[1])
            elif line.startswith("VENV_OK="):
                health.venv_ok = line.split("=", 1)[1] == "True"
            elif "Error" in line or "error" in line:
                health.pytorch_error = line[:100]
    except Exception as e:
        health.pytorch_error = str(e)[:100]

    # Check disk space
    try:
        result = subprocess.run(
            ssh_base + ["df", "-BG", "/home"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split("\n")
            if len(lines) >= 2:
                parts = lines[1].split()
                if len(parts) >= 5:
                    health.disk_free_gb = float(parts[3].rstrip("G"))
                    health.disk_usage_pct = float(parts[4].rstrip("%"))
    except Exception:
        pass

    # Check active screens
    try:
        result = subprocess.run(
            ssh_base + ["screen", "-ls"],
            capture_output=True, text=True, timeout=10
        )
        for line in result.stdout.split("\n"):
            if "." in line and ("Detached" in line or "Attached" in line):
                parts = line.strip().split("\t")
                if parts:
                    screen_name = parts[0].split(".", 1)[-1] if "." in parts[0] else parts[0]
                    health.screens_active.append(screen_name)
    except Exception:
        pass

    return health


def print_health_report(nodes: list[NodeHealth]) -> None:
    """Print formatted health report."""
    print("\n" + "=" * 80)
    print("RINGRIFT CLUSTER HEALTH REPORT")
    print("=" * 80 + "\n")

    # Summary
    total = len(nodes)
    healthy = sum(1 for n in nodes if n.healthy)
    reachable = sum(1 for n in nodes if n.reachable)

    print(f"Summary: {healthy}/{total} healthy, {reachable}/{total} reachable\n")

    # Detailed status
    print(f"{'Node':<20} {'Status':<8} {'PyTorch':<12} {'CUDA':<8} {'GPUs':<5} {'Disk':<10} {'Screens'}")
    print("-" * 80)

    for n in sorted(nodes, key=lambda x: (not x.healthy, not x.reachable, x.name)):
        pytorch = n.pytorch_version or ("ERROR" if n.pytorch_error else "N/A")
        cuda = n.cuda_version or ("N/A" if not n.cuda_available else "?")
        disk = f"{n.disk_free_gb:.0f}GB ({n.disk_usage_pct:.0f}%)" if n.disk_free_gb else "N/A"
        screens = ", ".join(n.screens_active[:3]) if n.screens_active else "-"
        if len(n.screens_active) > 3:
            screens += f" +{len(n.screens_active) - 3}"

        print(f"{n.name:<20} {n.status_emoji:<8} {pytorch:<12} {cuda:<8} {n.gpu_count:<5} {disk:<10} {screens}")

    # Issues
    issues = [n for n in nodes if not n.healthy]
    if issues:
        print("\n" + "-" * 80)
        print("ISSUES DETECTED:")
        for n in issues:
            if not n.reachable:
                print(f"  ❌ {n.name}: SSH failed - {n.ssh_error}")
            elif not n.pytorch_ok:
                print(f"  ⚠️  {n.name}: PyTorch issue - {n.pytorch_error}")
            elif not n.venv_ok:
                print(f"  ⚠️  {n.name}: Virtual environment not found")

    print()


def fix_pytorch(name: str, config: dict) -> bool:
    """Attempt to fix PyTorch installation on a node."""
    print(f"Attempting to fix PyTorch on {name}...")
    ssh_base = get_ssh_command(config)

    fix_cmd = """
cd ~/ringrift/ai-service && source venv/bin/activate && \
pip install torch --force-reinstall --quiet && \
python -c "import torch; print('Fixed! PyTorch', torch.__version__)"
"""
    try:
        result = subprocess.run(
            ssh_base + ["bash", "-c", fix_cmd],
            capture_output=True, text=True, timeout=300
        )
        if "Fixed!" in result.stdout:
            print(f"  ✅ {name}: PyTorch reinstalled successfully")
            return True
        else:
            print(f"  ❌ {name}: Fix failed - {result.stderr[:100]}")
            return False
    except Exception as e:
        print(f"  ❌ {name}: Fix failed - {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Check cluster node health")
    parser.add_argument("--nodes", nargs="+", help="Specific nodes to check")
    parser.add_argument("--fix", action="store_true", help="Attempt to fix issues")
    parser.add_argument("--parallel", type=int, default=8, help="Parallel checks")
    args = parser.parse_args()

    config = load_cluster_config()
    nodes_config = config.get("nodes", {})

    # Filter nodes if specified
    if args.nodes:
        nodes_config = {k: v for k, v in nodes_config.items() if k in args.nodes}

    # Filter to active nodes only
    nodes_config = {k: v for k, v in nodes_config.items()
                    if v.get("status") == "active" and v.get("gpu_type") != "none"}

    print(f"Checking {len(nodes_config)} nodes...")

    # Check nodes in parallel
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.parallel) as executor:
        futures = {
            executor.submit(check_node_health, name, cfg): name
            for name, cfg in nodes_config.items()
        }
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())

    print_health_report(results)

    # Attempt fixes if requested
    if args.fix:
        broken = [n for n in results if n.reachable and not n.pytorch_ok]
        if broken:
            print("Attempting to fix broken nodes...")
            for node in broken:
                fix_pytorch(node.name, nodes_config[node.name])
        else:
            print("No fixable issues found.")

    # Exit code based on health
    if all(n.healthy for n in results):
        return 0
    elif any(n.reachable for n in results):
        return 1
    else:
        return 2


if __name__ == "__main__":
    sys.exit(main())
