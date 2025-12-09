#!/usr/bin/env python
"""
Cluster management utilities for local Mac cluster CMA-ES training.

This script provides commands for managing a cluster of worker machines:
- Discover workers on the network (mDNS/Bonjour)
- Check worker health via HTTP
- Preload state pools on workers
- Run test evaluations
- SSH-based worker lifecycle management (start/stop/restart)
- Memory-aware cluster status and monitoring
- Run CMA-ES or self-play soaks on the cluster

Usage:
------
    # Discover workers on the network (mDNS)
    python scripts/cluster_manager.py discover

    # Check health of specific workers
    python scripts/cluster_manager.py health --workers 192.168.1.10:8765,192.168.1.11:8765

    # Check health with auto-discovery
    python scripts/cluster_manager.py health --discover

    # Preload state pools on workers
    python scripts/cluster_manager.py preload --discover --board square8 --num-players 2 --pool-id v1

    # Run a test evaluation across workers
    python scripts/cluster_manager.py test --discover --games 4

    # Show cluster stats
    python scripts/cluster_manager.py stats --workers 192.168.1.10:8765

    # SSH-based commands (require config/distributed_hosts.yaml):

    # Show full cluster status via SSH
    python scripts/cluster_manager.py ssh-status
    python scripts/cluster_manager.py ssh-status --json

    # Start workers on configured hosts via SSH
    python scripts/cluster_manager.py ssh-start
    python scripts/cluster_manager.py ssh-start --hosts mac-studio,mac-mini

    # Stop workers on configured hosts via SSH
    python scripts/cluster_manager.py ssh-stop

    # Restart workers
    python scripts/cluster_manager.py ssh-restart

    # Monitor cluster memory in real-time
    python scripts/cluster_manager.py ssh-monitor --interval 5

    # Run CMA-ES optimization on cluster
    python scripts/cluster_manager.py run-cmaes --board square8 --generations 50

    # Run self-play soak on cluster
    python scripts/cluster_manager.py run-soak --board square8 --games 1000

    # Benchmark cluster memory usage
    python scripts/cluster_manager.py benchmark --board square8 --ai minimax,mcts
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Allow imports from app/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.distributed.discovery import (
    WorkerInfo,
    WorkerDiscovery,
    discover_workers,
    wait_for_workers,
    parse_manual_workers,
    filter_healthy_workers,
)
from app.distributed.client import (
    WorkerClient,
    DistributedEvaluator,
)
from app.distributed.hosts import (
    HostConfig,
    HostMemoryInfo,
    SSHExecutor,
    load_remote_hosts,
    detect_host_memory,
    detect_all_host_memory,
    get_eligible_hosts_for_board,
    get_high_memory_hosts,
    get_ssh_executor,
    clear_memory_cache,
    BOARD_MEMORY_REQUIREMENTS,
)
from app.ai.heuristic_weights import BASE_V1_BALANCED_WEIGHTS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# =============================================================================
# SSH-based Cluster Management (via distributed_hosts.yaml)
# =============================================================================


@dataclass
class SSHWorkerStatus:
    """Status of a worker on a host via SSH."""
    host_name: str
    is_running: bool
    pid: Optional[int] = None
    port: int = 8765
    memory_mb: int = 0
    last_health_check: Optional[datetime] = None
    error: Optional[str] = None


@dataclass
class SSHClusterStatus:
    """Overall cluster status via SSH."""
    timestamp: datetime
    workers: List[SSHWorkerStatus] = field(default_factory=list)
    total_memory_gb: int = 0
    available_memory_gb: int = 0
    healthy_workers: int = 0
    unhealthy_workers: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "workers": [
                {
                    "host": w.host_name,
                    "running": w.is_running,
                    "pid": w.pid,
                    "port": w.port,
                    "memory_mb": w.memory_mb,
                    "error": w.error,
                }
                for w in self.workers
            ],
            "total_memory_gb": self.total_memory_gb,
            "available_memory_gb": self.available_memory_gb,
            "healthy_workers": self.healthy_workers,
            "unhealthy_workers": self.unhealthy_workers,
        }


class SSHClusterManager:
    """Manages distributed RingRift cluster via SSH."""

    WORKER_SCRIPT = "scripts/cluster_worker.py"
    WORKER_LOG = "logs/cluster_worker.log"
    DEFAULT_PORT = 8765

    def __init__(
        self,
        hosts: Optional[List[str]] = None,
        worker_port: int = 8765,
    ):
        """Initialize SSH cluster manager.

        Args:
            hosts: List of host names to manage. If None, uses all configured hosts.
            worker_port: Port for worker HTTP server.
        """
        self.worker_port = worker_port
        self._hosts_config = load_remote_hosts()

        if hosts:
            self._hosts = {
                name: cfg for name, cfg in self._hosts_config.items()
                if name in hosts
            }
        else:
            self._hosts = self._hosts_config

        self._executors: Dict[str, SSHExecutor] = {}
        self._stop_event = threading.Event()

        for name, config in self._hosts.items():
            self._executors[name] = SSHExecutor(config)

    @property
    def host_names(self) -> List[str]:
        """Get list of managed host names."""
        return list(self._hosts.keys())

    def check_worker_status(self, host_name: str) -> SSHWorkerStatus:
        """Check worker status on a host via SSH."""
        executor = self._executors.get(host_name)
        if not executor:
            return SSHWorkerStatus(
                host_name=host_name,
                is_running=False,
                error=f"Unknown host: {host_name}",
            )

        status = SSHWorkerStatus(host_name=host_name, is_running=False, port=self.worker_port)

        try:
            result = executor.run(
                f"pgrep -f 'cluster_worker.py.*--port {self.worker_port}'",
                timeout=10,
            )

            if result.returncode == 0 and result.stdout.strip():
                pids = result.stdout.strip().split('\n')
                status.pid = int(pids[0])
                status.is_running = True

                mem_result = executor.run(f"ps -o rss= -p {status.pid}", timeout=10)
                if mem_result.returncode == 0 and mem_result.stdout.strip():
                    status.memory_mb = int(mem_result.stdout.strip()) // 1024

                health_result = executor.run(
                    f"curl -s --connect-timeout 2 http://localhost:{self.worker_port}/health",
                    timeout=10,
                )
                if health_result.returncode == 0:
                    status.last_health_check = datetime.now()

        except Exception as e:
            status.error = str(e)

        return status

    def get_cluster_status(self) -> SSHClusterStatus:
        """Get status of entire cluster via SSH."""
        status = SSHClusterStatus(timestamp=datetime.now())

        with ThreadPoolExecutor(max_workers=max(len(self._hosts), 1)) as pool:
            futures = {
                pool.submit(self.check_worker_status, name): name
                for name in self._hosts
            }

            for future in as_completed(futures):
                worker_status = future.result()
                status.workers.append(worker_status)

                if worker_status.is_running and not worker_status.error:
                    status.healthy_workers += 1
                else:
                    status.unhealthy_workers += 1

        memory_info = detect_all_host_memory(list(self._hosts.keys()))
        for info in memory_info.values():
            status.total_memory_gb += info.total_gb
            status.available_memory_gb += info.available_gb

        return status

    def start_worker(self, host_name: str, extra_args: str = "") -> bool:
        """Start a worker on a host via SSH."""
        executor = self._executors.get(host_name)
        if not executor:
            logger.error(f"Unknown host: {host_name}")
            return False

        status = self.check_worker_status(host_name)
        if status.is_running:
            logger.info(f"Worker already running on {host_name} (PID {status.pid})")
            return True

        try:
            executor.run("mkdir -p logs", timeout=10)

            cmd = f"python {self.WORKER_SCRIPT} --port {self.worker_port}"
            if extra_args:
                cmd += f" {extra_args}"

            result = executor.run_async(cmd, self.WORKER_LOG)

            if result.returncode != 0:
                logger.error(f"Failed to start worker on {host_name}: {result.stderr}")
                return False

            time.sleep(2)
            status = self.check_worker_status(host_name)

            if status.is_running:
                logger.info(f"Started worker on {host_name} (PID {status.pid})")
                return True
            else:
                logger.error(f"Worker failed to start on {host_name}")
                return False

        except Exception as e:
            logger.error(f"Error starting worker on {host_name}: {e}")
            return False

    def stop_worker(self, host_name: str) -> bool:
        """Stop a worker on a host via SSH."""
        executor = self._executors.get(host_name)
        if not executor:
            logger.error(f"Unknown host: {host_name}")
            return False

        try:
            executor.run(
                f"pkill -f 'cluster_worker.py.*--port {self.worker_port}'",
                timeout=10,
            )

            time.sleep(1)
            status = self.check_worker_status(host_name)

            if not status.is_running:
                logger.info(f"Stopped worker on {host_name}")
                return True
            else:
                executor.run(
                    f"pkill -9 -f 'cluster_worker.py.*--port {self.worker_port}'",
                    timeout=10,
                )
                logger.info(f"Force-killed worker on {host_name}")
                return True

        except Exception as e:
            logger.error(f"Error stopping worker on {host_name}: {e}")
            return False

    def start_all_workers(self, extra_args: str = "") -> int:
        """Start workers on all hosts."""
        started = 0

        with ThreadPoolExecutor(max_workers=max(len(self._hosts), 1)) as pool:
            futures = {
                pool.submit(self.start_worker, name, extra_args): name
                for name in self._hosts
            }

            for future in as_completed(futures):
                if future.result():
                    started += 1

        logger.info(f"Started {started}/{len(self._hosts)} workers")
        return started

    def stop_all_workers(self) -> int:
        """Stop workers on all hosts."""
        stopped = 0

        with ThreadPoolExecutor(max_workers=max(len(self._hosts), 1)) as pool:
            futures = {
                pool.submit(self.stop_worker, name): name
                for name in self._hosts
            }

            for future in as_completed(futures):
                if future.result():
                    stopped += 1

        logger.info(f"Stopped {stopped}/{len(self._hosts)} workers")
        return stopped

    def monitor_cluster(
        self,
        interval: float = 5.0,
        duration: Optional[float] = None,
        callback: Optional[callable] = None,
    ) -> List[SSHClusterStatus]:
        """Monitor cluster status over time."""
        statuses = []
        start_time = time.time()
        self._stop_event.clear()

        logger.info(f"Starting cluster monitoring (interval={interval}s)")

        try:
            while not self._stop_event.is_set():
                status = self.get_cluster_status()
                statuses.append(status)

                if callback:
                    callback(status)
                else:
                    print(
                        f"[{status.timestamp.strftime('%H:%M:%S')}] "
                        f"Workers: {status.healthy_workers}/{len(status.workers)} healthy, "
                        f"Memory: {status.available_memory_gb}/{status.total_memory_gb} GB available"
                    )

                if duration and (time.time() - start_time) >= duration:
                    break

                time.sleep(interval)

        except KeyboardInterrupt:
            logger.info("Monitoring stopped by user")

        return statuses

    def stop_monitoring(self) -> None:
        """Signal monitor to stop."""
        self._stop_event.set()

    def get_eligible_hosts(self, board_type: str) -> List[str]:
        """Get hosts with enough memory for a board type."""
        return get_eligible_hosts_for_board(board_type, list(self._hosts.keys()))

    def run_distributed_cmaes(
        self,
        board_type: str = "square8",
        generations: int = 50,
        population_size: int = 16,
        extra_args: str = "",
    ) -> subprocess.CompletedProcess:
        """Run CMA-ES optimization using the cluster."""
        status = self.get_cluster_status()
        if status.healthy_workers == 0:
            logger.info("No healthy workers, starting cluster...")
            self.start_all_workers()
            time.sleep(3)

        eligible = self.get_eligible_hosts(board_type)
        if not eligible:
            logger.error(f"No hosts have enough memory for {board_type}")
            raise RuntimeError(f"No hosts eligible for {board_type}")

        logger.info(f"Running CMA-ES on {len(eligible)} hosts: {eligible}")

        worker_urls = []
        for host_name in eligible:
            host = self._hosts[host_name]
            worker_urls.append(host.http_worker_url)

        cmd = [
            "python", "scripts/run_cmaes_optimization.py",
            "--board", board_type,
            "--generations", str(generations),
            "--population-size", str(population_size),
            "--distributed",
            "--workers", ",".join(worker_urls),
        ]

        if extra_args:
            cmd.extend(extra_args.split())

        logger.info(f"Executing: {' '.join(cmd)}")

        return subprocess.run(
            cmd,
            cwd=str(Path(__file__).parent.parent),
        )

    def run_distributed_soak(
        self,
        board_type: str = "square8",
        num_games: int = 100,
        ai_difficulty: int = 3,
        extra_args: str = "",
    ) -> subprocess.CompletedProcess:
        """Run self-play soak test using the cluster."""
        status = self.get_cluster_status()
        if status.healthy_workers == 0:
            logger.info("No healthy workers, starting cluster...")
            self.start_all_workers()
            time.sleep(3)

        eligible = self.get_eligible_hosts(board_type)
        if not eligible:
            logger.error(f"No hosts have enough memory for {board_type}")
            raise RuntimeError(f"No hosts eligible for {board_type}")

        logger.info(f"Running soak on {len(eligible)} hosts: {eligible}")

        worker_urls = []
        for host_name in eligible:
            host = self._hosts[host_name]
            worker_urls.append(host.http_worker_url)

        cmd = [
            "python", "scripts/run_distributed_selfplay_soak.py",
            "--board-type", board_type,
            "--games", str(num_games),
            "--difficulty", str(ai_difficulty),
            "--workers", ",".join(worker_urls),
        ]

        if extra_args:
            cmd.extend(extra_args.split())

        logger.info(f"Executing: {' '.join(cmd)}")

        return subprocess.run(
            cmd,
            cwd=str(Path(__file__).parent.parent),
        )

    def benchmark_cluster_memory(
        self,
        board_type: str = "square8",
        ai_types: Optional[List[str]] = None,
        output_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Benchmark memory usage across the cluster."""
        ai_types = ai_types or ["minimax", "mcts"]
        results = {}

        for host_name in self._hosts:
            executor = self._executors[host_name]
            host_results = {"host": host_name, "benchmarks": []}

            for ai_type in ai_types:
                logger.info(f"Benchmarking {ai_type} on {host_name}...")

                cmd = (
                    f"PYTHONPATH=. python scripts/benchmark_ai_memory.py "
                    f"--board {board_type} --ai {ai_type} --quick --json"
                )

                try:
                    result = executor.run(cmd, timeout=300)

                    if result.returncode == 0:
                        for line in result.stdout.split('\n'):
                            if line.strip().startswith('{'):
                                try:
                                    data = json.loads(line)
                                    host_results["benchmarks"].append(data)
                                except json.JSONDecodeError:
                                    pass
                    else:
                        host_results["benchmarks"].append({
                            "ai_type": ai_type,
                            "error": result.stderr,
                        })

                except Exception as e:
                    host_results["benchmarks"].append({
                        "ai_type": ai_type,
                        "error": str(e),
                    })

            results[host_name] = host_results

        if output_path:
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Saved benchmark results to {output_path}")

        return results


def print_ssh_status(status: SSHClusterStatus) -> None:
    """Print SSH cluster status to console."""
    print(f"\n{'=' * 60}")
    print(f"Cluster Status - {status.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'=' * 60}")
    print(f"Total Memory: {status.total_memory_gb} GB")
    print(f"Available Memory: {status.available_memory_gb} GB")
    print(f"Workers: {status.healthy_workers} healthy, {status.unhealthy_workers} unhealthy")
    print(f"{'-' * 60}")

    for worker in sorted(status.workers, key=lambda w: w.host_name):
        state = "RUNNING" if worker.is_running else "STOPPED"
        if worker.error:
            state = f"ERROR: {worker.error}"

        print(
            f"  {worker.host_name:20} {state:20} "
            f"PID={worker.pid or '-':>6} "
            f"Mem={worker.memory_mb:>6} MB"
        )

    print()


# =============================================================================
# SSH Command Handlers
# =============================================================================


def cmd_ssh_status(args) -> None:
    """Show cluster status via SSH."""
    hosts = None
    if hasattr(args, 'hosts') and args.hosts:
        hosts = [h.strip() for h in args.hosts.split(',')]

    manager = SSHClusterManager(hosts=hosts)

    if not manager.host_names:
        print("No hosts configured. Create config/distributed_hosts.yaml")
        print("from config/distributed_hosts.template.yaml")
        return

    status = manager.get_cluster_status()

    if args.json:
        print(json.dumps(status.to_dict(), indent=2))
    else:
        print_ssh_status(status)


def cmd_ssh_start(args) -> None:
    """Start workers via SSH."""
    hosts = None
    if hasattr(args, 'hosts') and args.hosts:
        hosts = [h.strip() for h in args.hosts.split(',')]

    manager = SSHClusterManager(hosts=hosts)

    if not manager.host_names:
        print("No hosts configured.")
        return

    manager.start_all_workers(getattr(args, 'extra_args', ''))


def cmd_ssh_stop(args) -> None:
    """Stop workers via SSH."""
    hosts = None
    if hasattr(args, 'hosts') and args.hosts:
        hosts = [h.strip() for h in args.hosts.split(',')]

    manager = SSHClusterManager(hosts=hosts)

    if not manager.host_names:
        print("No hosts configured.")
        return

    manager.stop_all_workers()


def cmd_ssh_restart(args) -> None:
    """Restart workers via SSH."""
    hosts = None
    if hasattr(args, 'hosts') and args.hosts:
        hosts = [h.strip() for h in args.hosts.split(',')]

    manager = SSHClusterManager(hosts=hosts)

    if not manager.host_names:
        print("No hosts configured.")
        return

    manager.stop_all_workers()
    time.sleep(2)
    manager.start_all_workers(getattr(args, 'extra_args', ''))


def cmd_ssh_monitor(args) -> None:
    """Monitor cluster via SSH."""
    hosts = None
    if hasattr(args, 'hosts') and args.hosts:
        hosts = [h.strip() for h in args.hosts.split(',')]

    manager = SSHClusterManager(hosts=hosts)

    if not manager.host_names:
        print("No hosts configured.")
        return

    manager.monitor_cluster(
        interval=args.interval,
        duration=getattr(args, 'duration', None),
    )


def cmd_run_cmaes(args) -> None:
    """Run CMA-ES on cluster."""
    hosts = None
    if hasattr(args, 'hosts') and args.hosts:
        hosts = [h.strip() for h in args.hosts.split(',')]

    manager = SSHClusterManager(hosts=hosts)

    if not manager.host_names:
        print("No hosts configured.")
        return

    result = manager.run_distributed_cmaes(
        board_type=args.board,
        generations=args.generations,
        population_size=args.population_size,
        extra_args=getattr(args, 'extra_args', ''),
    )
    sys.exit(result.returncode)


def cmd_run_soak(args) -> None:
    """Run self-play soak on cluster."""
    hosts = None
    if hasattr(args, 'hosts') and args.hosts:
        hosts = [h.strip() for h in args.hosts.split(',')]

    manager = SSHClusterManager(hosts=hosts)

    if not manager.host_names:
        print("No hosts configured.")
        return

    result = manager.run_distributed_soak(
        board_type=args.board,
        num_games=args.games,
        ai_difficulty=args.difficulty,
        extra_args=getattr(args, 'extra_args', ''),
    )
    sys.exit(result.returncode)


def cmd_benchmark(args) -> None:
    """Benchmark cluster memory."""
    hosts = None
    if hasattr(args, 'hosts') and args.hosts:
        hosts = [h.strip() for h in args.hosts.split(',')]

    manager = SSHClusterManager(hosts=hosts)

    if not manager.host_names:
        print("No hosts configured.")
        return

    ai_types = [t.strip() for t in args.ai.split(',')]
    results = manager.benchmark_cluster_memory(
        board_type=args.board,
        ai_types=ai_types,
        output_path=getattr(args, 'output', None),
    )

    if not getattr(args, 'output', None):
        print(json.dumps(results, indent=2))


# =============================================================================
# mDNS/HTTP-based Discovery Commands (original)
# =============================================================================


def get_workers(args) -> List[WorkerInfo]:
    """Get worker list from args (manual or discovery)."""
    if args.workers:
        return parse_manual_workers(args.workers)
    elif getattr(args, "discover", False):
        print("Discovering workers on network...")
        workers = wait_for_workers(
            min_workers=getattr(args, "min_workers", 1),
            timeout=getattr(args, "timeout", 10.0),
        )
        if not workers:
            print("No workers found on network.")
            print("Ensure workers are running: python scripts/cluster_worker.py --register-bonjour")
        return workers
    else:
        print("Error: Specify --workers or --discover")
        return []


def cmd_discover(args) -> None:
    """Discover workers on the network."""
    print(f"Scanning network for {args.timeout}s...")
    workers = wait_for_workers(
        min_workers=args.min_workers,
        timeout=args.timeout,
    )

    if not workers:
        print("No workers found.")
        print("\nTo register a worker, run on each Mac:")
        print("  python scripts/cluster_worker.py --register-bonjour")
        return

    print(f"\nFound {len(workers)} worker(s):")
    print("-" * 60)
    for w in workers:
        print(f"  {w.worker_id:20s}  {w.url:20s}  ({w.hostname})")
    print("-" * 60)

    # Verify health
    print("\nVerifying worker health...")
    healthy = filter_healthy_workers(workers)
    print(f"  Healthy: {len(healthy)}/{len(workers)}")


def cmd_health(args) -> None:
    """Check worker health."""
    workers = get_workers(args)
    if not workers:
        return

    print(f"\nChecking health of {len(workers)} worker(s)...")
    print("-" * 70)

    healthy_count = 0
    for worker in workers:
        client = WorkerClient(worker.url)
        result = client.health_check()

        if result.get("status") == "healthy":
            healthy_count += 1
            tasks = result.get("tasks_completed", 0)
            print(f"  [OK]    {worker.url:20s}  tasks_completed={tasks}")
        else:
            error = result.get("error", "unknown error")
            print(f"  [FAIL]  {worker.url:20s}  {error}")

    print("-" * 70)
    print(f"Healthy: {healthy_count}/{len(workers)}")


def cmd_stats(args) -> None:
    """Show detailed worker statistics."""
    workers = get_workers(args)
    if not workers:
        return

    print(f"\nGathering stats from {len(workers)} worker(s)...")

    for worker in workers:
        client = WorkerClient(worker.url)
        stats = client.get_stats()

        print(f"\n{'='*60}")
        print(f"Worker: {worker.url}")
        print(f"{'='*60}")

        if "error" in stats:
            print(f"  Error: {stats['error']}")
            continue

        print(f"  Worker ID:         {stats.get('worker_id', 'unknown')}")
        print(f"  Tasks completed:   {stats.get('tasks_completed', 0)}")
        print(f"  Tasks failed:      {stats.get('tasks_failed', 0)}")
        print(f"  Total games:       {stats.get('total_games_played', 0)}")
        print(f"  Total eval time:   {stats.get('total_evaluation_time_sec', 0):.1f}s")
        print(f"  Uptime:            {stats.get('uptime_sec', 0):.0f}s")
        print(f"  Cached pools:      {stats.get('cached_pools', [])}")


def cmd_preload(args) -> None:
    """Preload state pools on workers."""
    workers = get_workers(args)
    if not workers:
        return

    print(f"\nPreloading pool on {len(workers)} worker(s)...")
    print(f"  Board: {args.board}")
    print(f"  Players: {args.num_players}")
    print(f"  Pool ID: {args.pool_id}")
    print()

    for worker in workers:
        client = WorkerClient(worker.url)
        result = client.preload_pool(args.board, args.num_players, args.pool_id)

        if result.get("status") == "success":
            size = result.get("pool_size", 0)
            print(f"  [OK]    {worker.url:20s}  loaded {size} states")
        else:
            error = result.get("error", "unknown error")
            print(f"  [FAIL]  {worker.url:20s}  {error}")


def cmd_test(args) -> None:
    """Run a test evaluation across workers."""
    workers = get_workers(args)
    if not workers:
        return

    healthy = filter_healthy_workers(workers)
    if not healthy:
        print("No healthy workers available for testing")
        return

    print(f"\nRunning test evaluation on {len(healthy)} worker(s)...")
    print(f"  Board: {args.board}")
    print(f"  Players: {args.num_players}")
    print(f"  Games per worker: {args.games}")
    print()

    evaluator = DistributedEvaluator(
        workers=[w.url for w in healthy],
        board_type=args.board,
        num_players=args.num_players,
        games_per_eval=args.games,
        eval_mode="multi-start",
        state_pool_id=args.pool_id,
        max_moves=200,
        eval_randomness=0.02,
    )

    # Preload pools
    print("Preloading state pools...")
    evaluator.preload_pools()

    # Create test population (baseline weights)
    population = [BASE_V1_BALANCED_WEIGHTS for _ in range(len(healthy))]

    print(f"\nEvaluating {len(population)} candidates...")
    start_time = time.time()

    def progress_callback(completed: int, total: int) -> None:
        print(f"  Progress: {completed}/{total}")

    fitness_scores, stats = evaluator.evaluate_population(
        population=population,
        progress_callback=progress_callback,
    )

    elapsed = time.time() - start_time

    print(f"\nResults:")
    print(f"  Candidates evaluated: {stats.total_candidates}")
    print(f"  Successful: {stats.successful_evaluations}")
    print(f"  Failed: {stats.failed_evaluations}")
    print(f"  Total games: {stats.total_games}")
    print(f"  Time: {elapsed:.1f}s")
    print(f"  Fitness scores: {fitness_scores}")

    print("\nWorker task distribution:")
    for worker, count in stats.worker_task_counts.items():
        print(f"  {worker}: {count} tasks")


def cmd_kill_workers(args) -> None:
    """Show command to stop workers (workers must be stopped manually)."""
    print("To stop workers, press Ctrl+C in each worker terminal.")
    print("\nAlternatively, use SSH to stop workers remotely:")
    print("  ssh user@worker-ip 'pkill -f cluster_worker.py'")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cluster management utilities for local Mac cluster",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # =========================================================================
    # mDNS/HTTP-based discovery commands
    # =========================================================================

    # discover command
    p_discover = subparsers.add_parser("discover", help="Discover workers on network (mDNS)")
    p_discover.add_argument(
        "--timeout", type=float, default=10.0,
        help="Discovery timeout in seconds (default: 10)"
    )
    p_discover.add_argument(
        "--min-workers", type=int, default=0,
        help="Minimum workers to wait for (default: 0)"
    )

    # health command
    p_health = subparsers.add_parser("health", help="Check worker health (HTTP)")
    p_health.add_argument("--workers", type=str, help="Comma-separated worker URLs")
    p_health.add_argument("--discover", action="store_true", help="Auto-discover workers")
    p_health.add_argument("--timeout", type=float, default=10.0)
    p_health.add_argument("--min-workers", type=int, default=1)

    # stats command
    p_stats = subparsers.add_parser("stats", help="Show worker statistics (HTTP)")
    p_stats.add_argument("--workers", type=str, help="Comma-separated worker URLs")
    p_stats.add_argument("--discover", action="store_true", help="Auto-discover workers")
    p_stats.add_argument("--timeout", type=float, default=10.0)
    p_stats.add_argument("--min-workers", type=int, default=1)

    # preload command
    p_preload = subparsers.add_parser("preload", help="Preload state pools on workers (HTTP)")
    p_preload.add_argument("--workers", type=str, help="Comma-separated worker URLs")
    p_preload.add_argument("--discover", action="store_true", help="Auto-discover workers")
    p_preload.add_argument("--board", type=str, default="square8", help="Board type")
    p_preload.add_argument("--num-players", type=int, default=2, help="Number of players")
    p_preload.add_argument("--pool-id", type=str, default="v1", help="State pool ID")
    p_preload.add_argument("--timeout", type=float, default=10.0)
    p_preload.add_argument("--min-workers", type=int, default=1)

    # test command
    p_test = subparsers.add_parser("test", help="Run test evaluation (HTTP)")
    p_test.add_argument("--workers", type=str, help="Comma-separated worker URLs")
    p_test.add_argument("--discover", action="store_true", help="Auto-discover workers")
    p_test.add_argument("--board", type=str, default="square8", help="Board type")
    p_test.add_argument("--num-players", type=int, default=2, help="Number of players")
    p_test.add_argument("--pool-id", type=str, default="v1", help="State pool ID")
    p_test.add_argument("--games", type=int, default=4, help="Games per candidate")
    p_test.add_argument("--timeout", type=float, default=10.0)
    p_test.add_argument("--min-workers", type=int, default=1)

    # =========================================================================
    # SSH-based cluster management commands
    # =========================================================================

    # ssh-status command
    p_ssh_status = subparsers.add_parser("ssh-status", help="Show cluster status via SSH")
    p_ssh_status.add_argument("--json", action="store_true", help="Output as JSON")
    p_ssh_status.add_argument("--hosts", help="Comma-separated list of hosts")

    # ssh-start command
    p_ssh_start = subparsers.add_parser("ssh-start", help="Start workers via SSH")
    p_ssh_start.add_argument("--hosts", help="Comma-separated list of hosts")
    p_ssh_start.add_argument("--extra-args", default="", help="Extra args for worker")

    # ssh-stop command
    p_ssh_stop = subparsers.add_parser("ssh-stop", help="Stop workers via SSH")
    p_ssh_stop.add_argument("--hosts", help="Comma-separated list of hosts")

    # ssh-restart command
    p_ssh_restart = subparsers.add_parser("ssh-restart", help="Restart workers via SSH")
    p_ssh_restart.add_argument("--hosts", help="Comma-separated list of hosts")
    p_ssh_restart.add_argument("--extra-args", default="", help="Extra args for worker")

    # ssh-monitor command
    p_ssh_monitor = subparsers.add_parser("ssh-monitor", help="Monitor cluster via SSH")
    p_ssh_monitor.add_argument("--interval", type=float, default=5.0, help="Check interval")
    p_ssh_monitor.add_argument("--duration", type=float, help="Monitoring duration")
    p_ssh_monitor.add_argument("--hosts", help="Comma-separated list of hosts")

    # run-cmaes command
    p_cmaes = subparsers.add_parser("run-cmaes", help="Run CMA-ES on cluster")
    p_cmaes.add_argument("--board", default="square8", help="Board type")
    p_cmaes.add_argument("--generations", type=int, default=50, help="Generations")
    p_cmaes.add_argument("--population-size", type=int, default=16, help="Population size")
    p_cmaes.add_argument("--hosts", help="Comma-separated list of hosts")
    p_cmaes.add_argument("--extra-args", default="", help="Extra args")

    # run-soak command
    p_soak = subparsers.add_parser("run-soak", help="Run self-play soak on cluster")
    p_soak.add_argument("--board", default="square8", help="Board type")
    p_soak.add_argument("--games", type=int, default=100, help="Number of games")
    p_soak.add_argument("--difficulty", type=int, default=3, help="AI difficulty")
    p_soak.add_argument("--hosts", help="Comma-separated list of hosts")
    p_soak.add_argument("--extra-args", default="", help="Extra args")

    # benchmark command
    p_bench = subparsers.add_parser("benchmark", help="Benchmark cluster memory")
    p_bench.add_argument("--board", default="square8", help="Board type")
    p_bench.add_argument("--ai", default="minimax,mcts", help="AI types to benchmark")
    p_bench.add_argument("--output", help="Output JSON file")
    p_bench.add_argument("--hosts", help="Comma-separated list of hosts")

    args = parser.parse_args()

    # mDNS/HTTP commands
    if args.command == "discover":
        cmd_discover(args)
    elif args.command == "health":
        cmd_health(args)
    elif args.command == "stats":
        cmd_stats(args)
    elif args.command == "preload":
        cmd_preload(args)
    elif args.command == "test":
        cmd_test(args)
    # SSH commands
    elif args.command == "ssh-status":
        cmd_ssh_status(args)
    elif args.command == "ssh-start":
        cmd_ssh_start(args)
    elif args.command == "ssh-stop":
        cmd_ssh_stop(args)
    elif args.command == "ssh-restart":
        cmd_ssh_restart(args)
    elif args.command == "ssh-monitor":
        cmd_ssh_monitor(args)
    elif args.command == "run-cmaes":
        cmd_run_cmaes(args)
    elif args.command == "run-soak":
        cmd_run_soak(args)
    elif args.command == "benchmark":
        cmd_benchmark(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
