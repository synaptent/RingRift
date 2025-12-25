#!/usr/bin/env python3
"""Deploy Universal Keepalive Daemon to cluster nodes.

This script deploys the universal_keepalive.py daemon and systemd/launchd
service to all configured cluster nodes.

Usage:
    # Deploy to all nodes
    python scripts/deploy_keepalive.py --all

    # Deploy to specific node
    python scripts/deploy_keepalive.py --node lambda-gh200-a

    # Deploy to all nodes of a type
    python scripts/deploy_keepalive.py --type lambda
    python scripts/deploy_keepalive.py --type vast
    python scripts/deploy_keepalive.py --type hetzner

    # Dry run (show what would be deployed)
    python scripts/deploy_keepalive.py --all --dry-run

    # Check status of deployed keepalive daemons
    python scripts/deploy_keepalive.py --status
"""

import argparse
import logging
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Paths
SCRIPT_DIR = Path(__file__).parent
AI_SERVICE_ROOT = SCRIPT_DIR.parent
CONFIG_PATH = AI_SERVICE_ROOT / "config" / "distributed_hosts.yaml"
KEEPALIVE_SCRIPT = SCRIPT_DIR / "universal_keepalive.py"
SYSTEMD_SERVICE = AI_SERVICE_ROOT / "deploy" / "systemd" / "ringrift-keepalive.service"
VAST_SYSTEMD_SERVICE = AI_SERVICE_ROOT / "deploy" / "systemd" / "ringrift-keepalive-vast.service"

# SSH options
SSH_OPTIONS = [
    "-o", "StrictHostKeyChecking=accept-new",
    "-o", "ConnectTimeout=10",
    "-o", "BatchMode=yes",
]


@dataclass
class NodeConfig:
    """Configuration for a cluster node."""
    name: str
    ssh_host: str
    ssh_port: int = 22
    ssh_user: str = "ubuntu"
    ssh_key: str = "~/.ssh/id_cluster"
    node_type: str = "linux"
    ringrift_path: str = "~/ringrift/ai-service"
    tailscale_ip: Optional[str] = None
    status: str = "ready"


def load_node_configs() -> dict[str, NodeConfig]:
    """Load node configurations from distributed_hosts.yaml."""
    if not CONFIG_PATH.exists():
        logger.error(f"Config file not found: {CONFIG_PATH}")
        return {}

    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)

    nodes = {}
    for name, host_config in config.get("hosts", {}).items():
        # Skip coordinator (mac-studio) for deployment
        if host_config.get("role") == "coordinator":
            continue

        # Skip retired nodes
        if host_config.get("status") == "retired":
            continue

        # Determine node type
        node_type = "linux"
        if "vast" in name.lower():
            node_type = "vast"
        elif "lambda" in name.lower() or "gh200" in name.lower() or "h100" in name.lower():
            node_type = "lambda"
        elif "hetzner" in name.lower() or "cpu" in name.lower():
            node_type = "hetzner"

        # Get SSH connection info
        ssh_host = host_config.get("tailscale_ip") or host_config.get("public_ip", name)
        ssh_port = host_config.get("ssh_port", 22)
        ssh_user = host_config.get("ssh_user", "ubuntu")
        if node_type == "vast":
            ssh_user = "root"
            # Vast uses ssh gateway
            ssh_host = host_config.get("ssh_host", ssh_host)

        nodes[name] = NodeConfig(
            name=name,
            ssh_host=ssh_host,
            ssh_port=ssh_port,
            ssh_user=ssh_user,
            ssh_key=host_config.get("ssh_key", "~/.ssh/id_cluster"),
            node_type=node_type,
            ringrift_path=host_config.get("ringrift_path", "~/ringrift/ai-service"),
            tailscale_ip=host_config.get("tailscale_ip"),
            status=host_config.get("status", "ready"),
        )

    return nodes


def run_ssh_command(node: NodeConfig, command: str, timeout: int = 30) -> tuple[bool, str]:
    """Run SSH command on a node.

    Returns:
        (success, output)
    """
    ssh_key = Path(node.ssh_key).expanduser()
    ssh_cmd = [
        "ssh",
        "-i", str(ssh_key),
        "-p", str(node.ssh_port),
        *SSH_OPTIONS,
        f"{node.ssh_user}@{node.ssh_host}",
        command
    ]

    try:
        result = subprocess.run(
            ssh_cmd,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        return result.returncode == 0, result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        return False, "SSH timeout"
    except Exception as e:
        return False, str(e)


def rsync_file(node: NodeConfig, local_path: Path, remote_path: str, timeout: int = 60) -> bool:
    """Rsync a file to a node."""
    ssh_key = Path(node.ssh_key).expanduser()
    ssh_cmd = f"ssh -i {ssh_key} -p {node.ssh_port} " + " ".join(SSH_OPTIONS)

    rsync_cmd = [
        "rsync",
        "-avz",
        "--progress",
        "-e", ssh_cmd,
        str(local_path),
        f"{node.ssh_user}@{node.ssh_host}:{remote_path}"
    ]

    try:
        result = subprocess.run(
            rsync_cmd,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        return result.returncode == 0
    except Exception as e:
        logger.error(f"rsync failed: {e}")
        return False


def deploy_to_node(node: NodeConfig, dry_run: bool = False) -> bool:
    """Deploy keepalive daemon to a single node.

    Steps:
    1. Test SSH connectivity
    2. Create directories
    3. Copy keepalive script
    4. Copy appropriate systemd service
    5. Create node config file
    6. Enable and start service

    Returns:
        True if deployment succeeded
    """
    logger.info(f"Deploying to {node.name} ({node.node_type})...")

    if dry_run:
        logger.info(f"  [DRY RUN] Would deploy to {node.ssh_user}@{node.ssh_host}:{node.ssh_port}")
        logger.info(f"  [DRY RUN] Script: {KEEPALIVE_SCRIPT}")
        logger.info(f"  [DRY RUN] Service: {SYSTEMD_SERVICE if node.node_type != 'vast' else VAST_SYSTEMD_SERVICE}")
        return True

    # Step 1: Test SSH connectivity
    success, output = run_ssh_command(node, "echo ok", timeout=15)
    if not success:
        logger.error(f"  Failed to connect to {node.name}: {output}")
        return False
    logger.info(f"  SSH connection OK")

    # Step 2: Create directories
    remote_scripts_dir = f"{node.ringrift_path}/scripts"
    success, _ = run_ssh_command(node, f"mkdir -p {remote_scripts_dir}")
    if not success:
        logger.error(f"  Failed to create directories on {node.name}")
        return False

    # Step 3: Copy keepalive script
    if not rsync_file(node, KEEPALIVE_SCRIPT, f"{remote_scripts_dir}/"):
        logger.error(f"  Failed to copy keepalive script to {node.name}")
        return False
    logger.info(f"  Copied keepalive script")

    # Step 4: Copy appropriate systemd service
    service_file = VAST_SYSTEMD_SERVICE if node.node_type == "vast" else SYSTEMD_SERVICE
    if service_file.exists():
        # Copy to temp location first, then move with sudo
        if not rsync_file(node, service_file, "/tmp/ringrift-keepalive.service"):
            logger.warning(f"  Failed to copy service file to {node.name}")
        else:
            success, _ = run_ssh_command(
                node,
                "sudo mv /tmp/ringrift-keepalive.service /etc/systemd/system/ringrift-keepalive.service && "
                "sudo chmod 644 /etc/systemd/system/ringrift-keepalive.service"
            )
            if success:
                logger.info(f"  Installed systemd service")
            else:
                logger.warning(f"  Failed to install systemd service (may need manual setup)")

    # Step 5: Create node config file
    config_content = f"""# RingRift Keepalive Node Configuration
# Auto-generated by deploy_keepalive.py
NODE_ID={node.name}
NODE_TYPE={node.node_type}
P2P_PORT=8770
RINGRIFT_PATH={node.ringrift_path}
ESCALATION_ENABLED=true
"""
    config_cmd = f"sudo mkdir -p /etc/ringrift && echo '{config_content}' | sudo tee /etc/ringrift/node.conf > /dev/null"
    success, _ = run_ssh_command(node, config_cmd)
    if success:
        logger.info(f"  Created node config")
    else:
        logger.warning(f"  Failed to create node config (non-critical)")

    # Step 6: Enable and start service
    success, output = run_ssh_command(
        node,
        "sudo systemctl daemon-reload && "
        "sudo systemctl enable ringrift-keepalive.service && "
        "sudo systemctl restart ringrift-keepalive.service"
    )
    if success:
        logger.info(f"  Started keepalive service")
    else:
        # Try starting directly if systemd fails (e.g., container without systemd)
        logger.warning(f"  Systemd failed, trying direct start...")
        success, _ = run_ssh_command(
            node,
            f"cd {node.ringrift_path} && "
            f"nohup python3 scripts/universal_keepalive.py --node-id {node.name} --daemon "
            f"> logs/keepalive.log 2>&1 &"
        )
        if success:
            logger.info(f"  Started keepalive directly (no systemd)")
        else:
            logger.error(f"  Failed to start keepalive on {node.name}")
            return False

    logger.info(f"  Deployment to {node.name} complete")
    return True


def check_status(nodes: dict[str, NodeConfig]) -> None:
    """Check status of keepalive daemons on all nodes."""
    logger.info("Checking keepalive status on all nodes...")
    print()
    print(f"{'Node':<25} {'Type':<10} {'Status':<12} {'Uptime':<15} {'Escalations'}")
    print("-" * 80)

    for name, node in sorted(nodes.items()):
        # Check if process is running
        success, output = run_ssh_command(
            node,
            "pgrep -f 'universal_keepalive' > /dev/null && echo 'running' || echo 'stopped'",
            timeout=15
        )

        if not success:
            print(f"{name:<25} {node.node_type:<10} {'unreachable':<12} {'-':<15} -")
            continue

        status = output.strip()

        # Get uptime if running
        uptime = "-"
        if "running" in status:
            success, output = run_ssh_command(
                node,
                "ps -o etime= -p $(pgrep -f 'universal_keepalive' | head -1) 2>/dev/null || echo 'unknown'",
                timeout=10
            )
            if success:
                uptime = output.strip()

        # Get escalation status
        escalations = "-"
        success, output = run_ssh_command(
            node,
            "cat /tmp/ringrift_escalation_state.json 2>/dev/null | python3 -c 'import sys,json; d=json.load(sys.stdin); print(len([k for k,v in d.items() if v.get(\"current_tier\",0)>0]))' 2>/dev/null || echo '0'",
            timeout=10
        )
        if success and output.strip().isdigit():
            count = int(output.strip())
            escalations = f"{count} active" if count > 0 else "none"

        print(f"{name:<25} {node.node_type:<10} {status:<12} {uptime:<15} {escalations}")

    print()


def main():
    parser = argparse.ArgumentParser(
        description="Deploy Universal Keepalive Daemon to cluster nodes"
    )
    parser.add_argument("--all", action="store_true", help="Deploy to all nodes")
    parser.add_argument("--node", help="Deploy to specific node")
    parser.add_argument("--type", choices=["lambda", "vast", "hetzner", "linux"],
                        help="Deploy to all nodes of a type")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be deployed")
    parser.add_argument("--status", action="store_true", help="Check status of deployed daemons")
    parser.add_argument("--parallel", type=int, default=5,
                        help="Number of parallel deployments (default: 5)")
    args = parser.parse_args()

    # Load node configurations
    nodes = load_node_configs()
    if not nodes:
        logger.error("No nodes found in configuration")
        sys.exit(1)

    logger.info(f"Loaded {len(nodes)} node configurations")

    # Check status only
    if args.status:
        check_status(nodes)
        return

    # Determine which nodes to deploy to
    target_nodes = {}

    if args.all:
        target_nodes = nodes
    elif args.node:
        if args.node in nodes:
            target_nodes = {args.node: nodes[args.node]}
        else:
            logger.error(f"Node '{args.node}' not found in configuration")
            sys.exit(1)
    elif args.type:
        target_nodes = {
            name: node for name, node in nodes.items()
            if node.node_type == args.type
        }
    else:
        parser.print_help()
        sys.exit(1)

    if not target_nodes:
        logger.error("No target nodes selected")
        sys.exit(1)

    logger.info(f"Deploying to {len(target_nodes)} nodes...")

    # Deploy to each node
    success_count = 0
    failed_nodes = []

    for name, node in target_nodes.items():
        try:
            if deploy_to_node(node, dry_run=args.dry_run):
                success_count += 1
            else:
                failed_nodes.append(name)
        except Exception as e:
            logger.error(f"Error deploying to {name}: {e}")
            failed_nodes.append(name)

        # Small delay between deployments to avoid overload
        if not args.dry_run:
            time.sleep(1)

    # Summary
    print()
    logger.info(f"Deployment complete: {success_count}/{len(target_nodes)} succeeded")
    if failed_nodes:
        logger.warning(f"Failed nodes: {', '.join(failed_nodes)}")


if __name__ == "__main__":
    main()
