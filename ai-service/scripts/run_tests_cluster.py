#!/usr/bin/env python3
"""
Run pytest on cluster nodes instead of locally.

This script enables running the test suite on remote cluster nodes,
preferring Hetzner CPU nodes which are dedicated to lightweight tasks.

Usage:
    # Run all tests on best available CPU node
    python scripts/run_tests_cluster.py

    # Run specific test file
    python scripts/run_tests_cluster.py tests/rules/test_capture_chain.py

    # Run with specific pytest args
    python scripts/run_tests_cluster.py -- -v -k "test_capture"

    # Run on specific node
    python scripts/run_tests_cluster.py --node hetzner-cpu1

    # Run on multiple nodes in parallel (for test sharding)
    python scripts/run_tests_cluster.py --parallel --num-nodes 3

    # Dry run (show what would be executed)
    python scripts/run_tests_cluster.py --dry-run

    # Check which nodes are available
    python scripts/run_tests_cluster.py --list-nodes
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config.cluster_config import (
    ClusterNode,
    get_cluster_nodes,
    get_cpu_only_nodes,
)
from app.core.ssh import SSHClient, SSHConfig, SSHResult

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

# Node preference order for test execution
PREFERRED_PROVIDERS = ["hetzner", "vultr", "nebius"]

# Test execution timeout (10 minutes)
DEFAULT_TEST_TIMEOUT = 600

# Path mappings per provider
PATH_MAPPINGS = {
    "runpod": "/workspace/ringrift/ai-service",
    "vast": "~/ringrift/ai-service",
    "nebius": "~/ringrift/ai-service",
    "vultr": "/root/ringrift/ai-service",
    "hetzner": "/root/ringrift/ai-service",
    "mac-studio": "~/Development/RingRift/ai-service",
    "local-mac": "/Users/armand/Development/RingRift/ai-service",
}


@dataclass
class TestResult:
    """Result of a test execution on a remote node."""

    node_name: str
    success: bool
    exit_code: int
    stdout: str
    stderr: str
    duration_seconds: float
    test_args: list[str]
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def summary(self) -> str:
        """Get a one-line summary of the result."""
        status = "PASSED" if self.success else "FAILED"
        return f"[{self.node_name}] {status} in {self.duration_seconds:.1f}s (exit code: {self.exit_code})"


@dataclass
class ClusterTestConfig:
    """Configuration for cluster test execution."""

    test_paths: list[str] = field(default_factory=list)
    pytest_args: list[str] = field(default_factory=list)
    target_node: str | None = None
    parallel: bool = False
    num_nodes: int = 1
    timeout: int = DEFAULT_TEST_TIMEOUT
    dry_run: bool = False
    verbose: bool = False
    sync_first: bool = True  # Sync code before running tests


# =============================================================================
# Node Selection
# =============================================================================


def get_node_path(node: ClusterNode) -> str:
    """Get the ringrift path for a node."""
    # Use explicit path from node config if available
    if node.ringrift_path:
        return node.ringrift_path

    # Determine provider from node name
    for provider, path in PATH_MAPPINGS.items():
        if node.name.startswith(provider):
            return path

    # Default fallback
    return "~/ringrift/ai-service"


def get_node_provider(node: ClusterNode) -> str:
    """Extract provider from node name."""
    for provider in PREFERRED_PROVIDERS:
        if node.name.startswith(provider):
            return provider
    # Fallback: first part of name before hyphen
    return node.name.split("-")[0] if "-" in node.name else "unknown"


def select_test_nodes(
    config: ClusterTestConfig,
) -> list[ClusterNode]:
    """Select nodes for test execution.

    Selection priority:
    1. If specific node requested, use that
    2. Prefer CPU-only nodes (Hetzner)
    3. Fall back to any active node
    """
    if config.target_node:
        # Specific node requested
        all_nodes = get_cluster_nodes()
        if config.target_node not in all_nodes:
            available = list(all_nodes.keys())
            raise ValueError(
                f"Node '{config.target_node}' not found. "
                f"Available: {', '.join(sorted(available)[:10])}..."
            )
        node = all_nodes[config.target_node]
        if not node.is_active:
            logger.warning(f"Node {config.target_node} is not active (status: {node.status})")
        return [node]

    # Get CPU-only nodes (preferred for tests)
    cpu_nodes = get_cpu_only_nodes(only_active=True)

    # Sort by preference
    def node_priority(node: ClusterNode) -> tuple:
        provider = get_node_provider(node)
        try:
            priority = PREFERRED_PROVIDERS.index(provider)
        except ValueError:
            priority = 999
        return (priority, node.name)

    cpu_nodes.sort(key=node_priority)

    if not cpu_nodes:
        # Fall back to any active node
        logger.warning("No CPU-only nodes available, checking all nodes")
        all_nodes = list(get_cluster_nodes().values())
        active_nodes = [n for n in all_nodes if n.is_active]
        if not active_nodes:
            raise RuntimeError("No active nodes available for test execution")
        active_nodes.sort(key=node_priority)
        cpu_nodes = active_nodes

    # Select requested number of nodes
    num_nodes = config.num_nodes if config.parallel else 1
    selected = cpu_nodes[:num_nodes]

    if len(selected) < num_nodes:
        logger.warning(
            f"Only {len(selected)} nodes available, requested {num_nodes}"
        )

    return selected


# =============================================================================
# Remote Execution
# =============================================================================


async def check_node_available(node: ClusterNode) -> bool:
    """Check if a node is reachable via SSH."""
    try:
        client = SSHClient(
            SSHConfig(
                host=node.best_ip or "",
                port=node.ssh_port,
                user=node.ssh_user,
                key_path=node.ssh_key,
                connect_timeout=10,
                command_timeout=15,
            )
        )
        result = await client.run_async("echo ok", timeout=15)
        return result.returncode == 0
    except Exception as e:
        logger.debug(f"Node {node.name} not reachable: {e}")
        return False


async def sync_code_to_node(node: ClusterNode, dry_run: bool = False) -> bool:
    """Ensure node has latest code via git pull."""
    node_path = get_node_path(node)
    cmd = f"cd {node_path} && git pull --ff-only"

    if dry_run:
        logger.info(f"[DRY RUN] Would sync {node.name}: {cmd}")
        return True

    try:
        client = SSHClient(
            SSHConfig(
                host=node.best_ip or "",
                port=node.ssh_port,
                user=node.ssh_user,
                key_path=node.ssh_key,
                connect_timeout=10,
                command_timeout=120,
            )
        )
        result = await client.run_async(cmd, timeout=120)
        if result.returncode == 0:
            logger.info(f"[{node.name}] Code synced successfully")
            return True
        else:
            logger.warning(
                f"[{node.name}] Sync failed: {result.stderr or result.stdout}"
            )
            return False
    except Exception as e:
        logger.error(f"[{node.name}] Sync error: {e}")
        return False


async def run_tests_on_node(
    node: ClusterNode,
    config: ClusterTestConfig,
) -> TestResult:
    """Run pytest on a remote node."""
    start_time = datetime.now()
    node_path = get_node_path(node)

    # Build pytest command
    pytest_cmd_parts = ["python", "-m", "pytest"]

    # Add test paths
    if config.test_paths:
        pytest_cmd_parts.extend(config.test_paths)

    # Add pytest args
    if config.pytest_args:
        pytest_cmd_parts.extend(config.pytest_args)

    # Add default flags if not verbose
    if not config.verbose and "-v" not in config.pytest_args:
        pytest_cmd_parts.append("-q")  # Quiet by default

    pytest_cmd = " ".join(pytest_cmd_parts)

    # Build full command with PYTHONPATH
    full_cmd = f"cd {node_path} && PYTHONPATH=. {pytest_cmd}"

    if config.dry_run:
        logger.info(f"[DRY RUN] Would execute on {node.name}:")
        logger.info(f"  {full_cmd}")
        return TestResult(
            node_name=node.name,
            success=True,
            exit_code=0,
            stdout="[dry run]",
            stderr="",
            duration_seconds=0,
            test_args=config.test_paths + config.pytest_args,
        )

    logger.info(f"[{node.name}] Running tests: {pytest_cmd}")

    try:
        client = SSHClient(
            SSHConfig(
                host=node.best_ip or "",
                port=node.ssh_port,
                user=node.ssh_user,
                key_path=node.ssh_key,
                connect_timeout=30,
                command_timeout=config.timeout,
            )
        )

        result = await client.run_async(full_cmd, timeout=config.timeout)
        duration = (datetime.now() - start_time).total_seconds()

        return TestResult(
            node_name=node.name,
            success=result.returncode == 0,
            exit_code=result.returncode,
            stdout=result.stdout or "",
            stderr=result.stderr or "",
            duration_seconds=duration,
            test_args=config.test_paths + config.pytest_args,
        )

    except asyncio.TimeoutError:
        duration = (datetime.now() - start_time).total_seconds()
        return TestResult(
            node_name=node.name,
            success=False,
            exit_code=-1,
            stdout="",
            stderr=f"Test execution timed out after {config.timeout}s",
            duration_seconds=duration,
            test_args=config.test_paths + config.pytest_args,
        )
    except Exception as e:
        duration = (datetime.now() - start_time).total_seconds()
        return TestResult(
            node_name=node.name,
            success=False,
            exit_code=-1,
            stdout="",
            stderr=str(e),
            duration_seconds=duration,
            test_args=config.test_paths + config.pytest_args,
        )


# =============================================================================
# Main Entry Points
# =============================================================================


async def run_tests_cluster(config: ClusterTestConfig) -> list[TestResult]:
    """Run tests on cluster node(s).

    Returns:
        List of TestResult objects, one per node.
    """
    # Select nodes
    try:
        nodes = select_test_nodes(config)
    except (ValueError, RuntimeError) as e:
        logger.error(f"Node selection failed: {e}")
        return []

    logger.info(f"Selected node(s): {', '.join(n.name for n in nodes)}")

    # Check availability
    available_nodes = []
    for node in nodes:
        if await check_node_available(node):
            available_nodes.append(node)
        else:
            logger.warning(f"Node {node.name} is not reachable, skipping")

    if not available_nodes:
        logger.error("No reachable nodes available")
        return []

    # Sync code if requested
    if config.sync_first and not config.dry_run:
        logger.info("Syncing code to nodes...")
        sync_tasks = [
            sync_code_to_node(node, config.dry_run)
            for node in available_nodes
        ]
        sync_results = await asyncio.gather(*sync_tasks)

        # Filter to successfully synced nodes
        synced_nodes = [
            node
            for node, success in zip(available_nodes, sync_results)
            if success
        ]
        if not synced_nodes:
            logger.error("Failed to sync code to any node")
            return []
        available_nodes = synced_nodes

    # Run tests
    if config.parallel and len(available_nodes) > 1:
        logger.info(f"Running tests in parallel on {len(available_nodes)} nodes")
        tasks = [
            run_tests_on_node(node, config)
            for node in available_nodes
        ]
        results = await asyncio.gather(*tasks)
    else:
        # Sequential execution on first node
        node = available_nodes[0]
        results = [await run_tests_on_node(node, config)]

    return list(results)


def list_available_nodes() -> None:
    """Print available nodes for test execution."""
    print("\n=== Available Nodes for Test Execution ===\n")

    # CPU-only nodes (preferred)
    cpu_nodes = get_cpu_only_nodes(only_active=True)
    if cpu_nodes:
        print("CPU-only nodes (preferred for tests):")
        for node in sorted(cpu_nodes, key=lambda n: n.name):
            provider = get_node_provider(node)
            print(f"  {node.name:25} ({provider:10}) - {node.status}")
    else:
        print("No CPU-only nodes available")

    # All active nodes
    print("\nAll active nodes:")
    all_nodes = get_cluster_nodes()
    active = [n for n in all_nodes.values() if n.is_active]
    for node in sorted(active, key=lambda n: n.name):
        gpu_info = f"GPU: {node.gpu}" if node.gpu else "CPU only"
        print(f"  {node.name:25} - {gpu_info}")

    print(f"\nTotal: {len(active)} active nodes")


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run pytest on cluster nodes",
        epilog="""
Examples:
  %(prog)s                                    # Run all tests on best CPU node
  %(prog)s tests/rules/                       # Run specific test directory
  %(prog)s --node hetzner-cpu1 -- -v          # Run on specific node with verbose
  %(prog)s --parallel --num-nodes 3           # Run on 3 nodes in parallel
  %(prog)s --list-nodes                       # Show available nodes
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "test_paths",
        nargs="*",
        help="Test paths to run (passed to pytest)",
    )
    parser.add_argument(
        "--node", "-n",
        dest="target_node",
        help="Specific node to run on (default: auto-select)",
    )
    parser.add_argument(
        "--parallel", "-p",
        action="store_true",
        help="Run on multiple nodes in parallel",
    )
    parser.add_argument(
        "--num-nodes",
        type=int,
        default=3,
        help="Number of nodes for parallel execution (default: 3)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TEST_TIMEOUT,
        help=f"Test timeout in seconds (default: {DEFAULT_TEST_TIMEOUT})",
    )
    parser.add_argument(
        "--no-sync",
        action="store_true",
        help="Skip git pull before running tests",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be executed without running",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output",
    )
    parser.add_argument(
        "--list-nodes",
        action="store_true",
        help="List available nodes and exit",
    )

    # Everything after -- goes to pytest
    args, pytest_args = parser.parse_known_args()

    if args.list_nodes:
        list_available_nodes()
        return 0

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    config = ClusterTestConfig(
        test_paths=args.test_paths,
        pytest_args=pytest_args,
        target_node=args.target_node,
        parallel=args.parallel,
        num_nodes=args.num_nodes,
        timeout=args.timeout,
        dry_run=args.dry_run,
        verbose=args.verbose,
        sync_first=not args.no_sync,
    )

    # Run tests
    results = asyncio.run(run_tests_cluster(config))

    if not results:
        print("\nNo test results (execution failed)")
        return 1

    # Print results
    print("\n" + "=" * 70)
    print("TEST RESULTS")
    print("=" * 70)

    all_passed = True
    for result in results:
        print(f"\n{result.summary}")

        if result.stdout.strip():
            print("-" * 50)
            print(result.stdout)

        if result.stderr.strip() and not result.success:
            print("STDERR:")
            print(result.stderr)

        if not result.success:
            all_passed = False

    # Summary
    print("\n" + "=" * 70)
    passed = sum(1 for r in results if r.success)
    total = len(results)
    print(f"SUMMARY: {passed}/{total} nodes passed")
    print("=" * 70)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
