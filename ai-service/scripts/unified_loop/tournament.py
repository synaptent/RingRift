"""Unified Loop Tournament Services.

This module contains tournament-related services for the unified AI loop:
- ShadowTournamentService: Lightweight continuous evaluation on remote hosts

Extracted from unified_ai_loop.py for better modularity (Phase 2 refactoring).
"""

from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List

from .config import DataEvent, DataEventType, EvaluationConfig

# Path to config directory
AI_SERVICE_ROOT = Path(__file__).resolve().parents[2]

if TYPE_CHECKING:
    from unified_ai_loop import EventBus, UnifiedLoopState

# Import improvement optimizer for adaptive intervals
try:
    from app.training.improvement_optimizer import (
        get_improvement_optimizer,
        get_evaluation_interval,
    )
    HAS_IMPROVEMENT_OPTIMIZER = True
except ImportError:
    HAS_IMPROVEMENT_OPTIMIZER = False
    get_improvement_optimizer = None
    get_evaluation_interval = None


def _load_tournament_hosts() -> List[Dict[str, Any]]:
    """Load tournament hosts from config/distributed_hosts.yaml."""
    config_path = AI_SERVICE_ROOT / "config" / "distributed_hosts.yaml"

    if not config_path.exists():
        print("[Tournament] Warning: No config found, using empty host list")
        return []

    try:
        import yaml
        with open(config_path) as f:
            config = yaml.safe_load(f) or {}

        hosts = []
        for name, info in config.get("hosts", {}).items():
            # Use Tailscale IP preferentially
            ip = info.get("tailscale_ip") or info.get("ssh_host")
            if not ip or info.get("status") != "ready":
                continue

            user = info.get("ssh_user", "root")
            hosts.append({
                "name": name,
                "ssh": f"{user}@{ip}",
                "cpus": info.get("cpus", 8),
                "ringrift_path": info.get("ringrift_path", "~/ringrift/ai-service"),
            })

        # Sort by CPU count descending (prioritize high-CPU hosts)
        hosts.sort(key=lambda h: h["cpus"], reverse=True)
        return hosts
    except Exception as e:
        print(f"[Tournament] Error loading config: {e}")
        return []


class ShadowTournamentService:
    """Runs lightweight continuous evaluation."""

    # Tournament hosts loaded from config/distributed_hosts.yaml
    # Prioritized by CPU count for CPU-intensive tournament evaluation
    TOURNAMENT_HOSTS = _load_tournament_hosts()

    def __init__(self, config: EvaluationConfig, state: "UnifiedLoopState", event_bus: "EventBus"):
        self.config = config
        self.state = state
        self.event_bus = event_bus
        # Tracking for adaptive intervals
        self._eval_durations: List[float] = []  # Recent evaluation durations
        self._eval_success_rate: float = 1.0  # Rolling success rate
        self._last_interval_adjustment: float = 0.0
        # Round-robin index for tournament host selection
        self._host_index: int = 0
        # Track host availability for load balancing
        self._host_busy: Dict[str, bool] = {h["name"]: False for h in self.TOURNAMENT_HOSTS}

    def _get_next_tournament_host(self) -> Dict[str, Any]:
        """Get next available tournament host using round-robin."""
        # Try to find an available host
        for _ in range(len(self.TOURNAMENT_HOSTS)):
            host = self.TOURNAMENT_HOSTS[self._host_index]
            self._host_index = (self._host_index + 1) % len(self.TOURNAMENT_HOSTS)
            if not self._host_busy.get(host["name"], False):
                return host
        # All busy - just use round-robin regardless
        host = self.TOURNAMENT_HOSTS[self._host_index]
        self._host_index = (self._host_index + 1) % len(self.TOURNAMENT_HOSTS)
        return host

    async def _run_remote_tournament(self, host: Dict[str, Any], config_key: str) -> Dict[str, Any]:
        """Run tournament on remote host via SSH."""
        parts = config_key.rsplit("_", 1)
        board_type = parts[0]
        num_players = int(parts[1].replace("p", ""))

        ssh_target = host["ssh"]
        ringrift_path = host["ringrift_path"]
        host_name = host["name"]

        # Build remote command
        remote_cmd = f'''cd {ringrift_path} && source venv/bin/activate && \\
            python scripts/run_model_elo_tournament.py \\
            --board {board_type} \\
            --players {num_players} \\
            --games {self.config.shadow_games_per_config} \\
            --quick --include-baselines 2>&1'''

        try:
            self._host_busy[host_name] = True
            print(f"[ShadowTournament] Dispatching {config_key} to {host_name} ({ssh_target})")

            proc = await asyncio.create_subprocess_exec(
                "ssh", "-i", os.path.expanduser("~/.ssh/id_cluster"),
                "-o", "ConnectTimeout=10", "-o", "BatchMode=yes",
                "-o", "StrictHostKeyChecking=no", ssh_target, remote_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=900)

            success = proc.returncode == 0
            if not success:
                stderr_text = stderr.decode()[:500] if stderr else ""
                print(f"[ShadowTournament] {config_key} on {host_name} failed: {stderr_text}")

            return {
                "config": config_key,
                "games_played": self.config.shadow_games_per_config,
                "success": success,
                "host": host_name,
            }
        except asyncio.TimeoutError:
            print(f"[ShadowTournament] {config_key} on {host_name} timed out after 900s")
            return {"config": config_key, "error": "timeout", "success": False, "host": host_name}
        except Exception as e:
            print(f"[ShadowTournament] {config_key} on {host_name} error: {e}")
            return {"config": config_key, "error": str(e), "success": False, "host": host_name}
        finally:
            self._host_busy[host_name] = False

    async def run_shadow_tournament(self, config_key: str) -> Dict[str, Any]:
        """Run a quick shadow tournament for a configuration on remote hosts."""
        parts = config_key.rsplit("_", 1)
        board_type = parts[0]
        num_players = int(parts[1].replace("p", ""))

        await self.event_bus.publish(DataEvent(
            event_type=DataEventType.EVALUATION_STARTED,
            payload={"config": config_key, "type": "shadow"}
        ))

        try:
            # Get next available tournament host
            host = self._get_next_tournament_host()

            # Run on remote host
            result = await self._run_remote_tournament(host, config_key)

            if config_key in self.state.configs:
                self.state.configs[config_key].last_evaluation_time = time.time()

            self.state.total_evaluations += 1

            await self.event_bus.publish(DataEvent(
                event_type=DataEventType.EVALUATION_COMPLETED,
                payload=result
            ))

            return result

        except Exception as e:
            print(f"[ShadowTournament] Error running tournament for {config_key}: {e}")
            return {"config": config_key, "error": str(e), "success": False}

    async def run_full_tournament(self) -> Dict[str, Any]:
        """Run a full tournament across all configurations on best remote host."""
        await self.event_bus.publish(DataEvent(
            event_type=DataEventType.EVALUATION_STARTED,
            payload={"type": "full"}
        ))

        try:
            # Use the most powerful host (vast-5090-quad with 512 CPUs) for full tournaments
            host = self.TOURNAMENT_HOSTS[0]  # vast-5090-quad
            ssh_target = host["ssh"]
            ringrift_path = host["ringrift_path"]
            host_name = host["name"]

            remote_cmd = f'''cd {ringrift_path} && source venv/bin/activate && \\
                python scripts/run_model_elo_tournament.py \\
                --all-configs \\
                --games {self.config.full_tournament_games} \\
                --include-baselines 2>&1'''

            print(f"[ShadowTournament] Running full tournament on {host_name} ({ssh_target})")

            proc = await asyncio.create_subprocess_exec(
                "ssh", "-i", os.path.expanduser("~/.ssh/id_cluster"),
                "-o", "ConnectTimeout=10", "-o", "BatchMode=yes",
                "-o", "StrictHostKeyChecking=no", ssh_target, remote_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=3600)

            result = {
                "type": "full",
                "success": proc.returncode == 0,
                "host": host_name,
            }

            self.state.total_evaluations += 1

            await self.event_bus.publish(DataEvent(
                event_type=DataEventType.EVALUATION_COMPLETED,
                payload=result
            ))

            return result

        except Exception as e:
            print(f"[ShadowTournament] Error running full tournament: {e}")
            return {"type": "full", "error": str(e), "success": False}

    async def run_parallel_shadow_tournaments(
        self,
        config_keys: List[str],
        max_concurrent: int = 4
    ) -> List[Dict[str, Any]]:
        """Run shadow tournaments for multiple configs in parallel on remote hosts.

        Dispatches tournaments to Vast hosts with high CPU counts for efficient
        CPU-intensive evaluation. Uses round-robin host selection with 4 hosts.

        Args:
            config_keys: List of config keys to evaluate
            max_concurrent: Maximum concurrent tournaments (default 4 = num tournament hosts)

        Returns:
            List of tournament results
        """
        if not config_keys:
            return []

        # Use semaphore to limit concurrency
        semaphore = asyncio.Semaphore(max_concurrent)

        async def run_with_semaphore(config_key: str) -> Dict[str, Any]:
            async with semaphore:
                return await self.run_shadow_tournament(config_key)

        # Run all tournaments in parallel (limited by semaphore)
        start_time = time.time()
        print(f"[ShadowTournament] Running {len(config_keys)} tournaments in parallel (max {max_concurrent} concurrent)")

        results = await asyncio.gather(
            *[run_with_semaphore(ck) for ck in config_keys],
            return_exceptions=True
        )

        # Process results and handle exceptions
        processed_results = []
        for config_key, result in zip(config_keys, results):
            if isinstance(result, Exception):
                processed_results.append({
                    "config": config_key,
                    "error": str(result),
                    "success": False
                })
            else:
                processed_results.append(result)

        elapsed = time.time() - start_time
        successful = sum(1 for r in processed_results if r.get("success", False))
        print(f"[ShadowTournament] Completed {len(config_keys)} tournaments in {elapsed:.1f}s ({successful} successful)")

        # Track evaluation performance for adaptive intervals
        self._eval_durations.append(elapsed)
        if len(self._eval_durations) > 20:
            self._eval_durations = self._eval_durations[-20:]

        # Update rolling success rate (exponential moving average)
        success_rate = successful / max(1, len(config_keys))
        self._eval_success_rate = 0.7 * self._eval_success_rate + 0.3 * success_rate

        return processed_results

    async def run_batched_parallel_tournaments(
        self,
        config_keys: List[str],
        games_per_config: int = 50,
        batch_size: int = 3,
    ) -> List[Dict[str, Any]]:
        """Phase 3.3: Run tournaments in batches for more efficient resource utilization.

        This method batches multiple configs together on powerful hosts, running
        larger batches of games per tournament for better throughput.

        Strategy:
        1. Assign high-CPU hosts (512, 384 CPUs) to run batches of configs
        2. Each batch runs multiple configs sequentially with more games per config
        3. Lower-CPU hosts run individual configs in parallel

        Args:
            config_keys: List of config keys to evaluate
            games_per_config: Number of games per config (default 50)
            batch_size: Number of configs per batch on high-CPU hosts (default 3)

        Returns:
            List of tournament results
        """
        if not config_keys:
            return []

        start_time = time.time()
        all_results = []

        # Separate high-CPU and standard hosts
        high_cpu_hosts = [h for h in self.TOURNAMENT_HOSTS if h.get("cpus", 0) >= 128]
        standard_hosts = [h for h in self.TOURNAMENT_HOSTS if h.get("cpus", 0) < 128]

        # Create batches for high-CPU hosts
        batches = []
        remaining = list(config_keys)

        # Assign batches to high-CPU hosts
        for host in high_cpu_hosts:
            if not remaining:
                break
            batch = remaining[:batch_size]
            remaining = remaining[batch_size:]
            batches.append((host, batch))

        print(f"[ShadowTournament] Batched tournament: {len(batches)} batches on high-CPU hosts, "
              f"{len(remaining)} individual configs remaining")

        async def run_batch(host: Dict[str, Any], batch_configs: List[str]) -> List[Dict[str, Any]]:
            """Run a batch of configs on a single host."""
            ssh_target = host["ssh"]
            ringrift_path = host["ringrift_path"]
            host_name = host["name"]

            batch_results = []
            for config_key in batch_configs:
                parts = config_key.rsplit("_", 1)
                board_type = parts[0]
                num_players = int(parts[1].replace("p", ""))

                remote_cmd = f'''cd {ringrift_path} && source venv/bin/activate && \\
                    python scripts/run_model_elo_tournament.py \\
                    --board {board_type} \\
                    --players {num_players} \\
                    --games {games_per_config} \\
                    --quick --include-baselines 2>&1'''

                try:
                    proc = await asyncio.create_subprocess_exec(
                        "ssh", "-i", os.path.expanduser("~/.ssh/id_cluster"),
                        "-o", "ConnectTimeout=10", "-o", "BatchMode=yes",
                        "-o", "StrictHostKeyChecking=no", ssh_target, remote_cmd,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE
                    )
                    await asyncio.wait_for(proc.communicate(), timeout=600)

                    batch_results.append({
                        "config": config_key,
                        "games_played": games_per_config,
                        "success": proc.returncode == 0,
                        "host": host_name,
                        "batch": True,
                    })
                except Exception as e:
                    batch_results.append({
                        "config": config_key,
                        "error": str(e),
                        "success": False,
                        "host": host_name,
                    })

            return batch_results

        # Run batches in parallel
        batch_tasks = [run_batch(host, configs) for host, configs in batches]

        # Run remaining configs individually on standard hosts
        async def run_individual(config_key: str) -> Dict[str, Any]:
            return await self.run_shadow_tournament(config_key)

        individual_tasks = [run_individual(ck) for ck in remaining]

        # Gather all results
        if batch_tasks or individual_tasks:
            results = await asyncio.gather(
                *batch_tasks,
                *individual_tasks,
                return_exceptions=True
            )

            # Flatten batch results
            for result in results[:len(batch_tasks)]:
                if isinstance(result, Exception):
                    all_results.append({"error": str(result), "success": False})
                elif isinstance(result, list):
                    all_results.extend(result)
                else:
                    all_results.append(result)

            # Add individual results
            for result in results[len(batch_tasks):]:
                if isinstance(result, Exception):
                    all_results.append({"error": str(result), "success": False})
                else:
                    all_results.append(result)

        elapsed = time.time() - start_time
        successful = sum(1 for r in all_results if r.get("success", False))
        total_games = successful * games_per_config
        print(f"[ShadowTournament] Batched tournament complete: "
              f"{successful}/{len(config_keys)} configs, "
              f"{total_games} games in {elapsed:.1f}s "
              f"({total_games / max(0.1, elapsed):.1f} games/sec)")

        return all_results

    def get_adaptive_interval(self, promotion_velocity: float = 0.0) -> int:
        """Calculate adaptive evaluation interval based on cluster health and performance.

        The interval adapts based on:
        1. Evaluation success rate - faster when evals succeed consistently
        2. Evaluation duration - faster when evals complete quickly
        3. Promotion velocity - faster when we're getting good results
        4. Time since last adjustment - smoothing to prevent oscillation

        Returns:
            Adaptive interval in seconds (between min and max configured values)
        """
        if not self.config.adaptive_interval_enabled:
            return self.config.shadow_interval_seconds

        base_interval = self.config.shadow_interval_seconds
        min_interval = self.config.adaptive_interval_min_seconds
        max_interval = self.config.adaptive_interval_max_seconds

        # Start with base adjustment factor
        adjustment = 1.0

        # Factor 1: Success rate (faster when successful)
        if self._eval_success_rate >= 0.95:
            adjustment *= 0.7  # 30% faster when nearly all succeed
        elif self._eval_success_rate >= 0.80:
            adjustment *= 0.85  # 15% faster when most succeed
        elif self._eval_success_rate < 0.50:
            adjustment *= 1.5  # Slow down if many failures

        # Factor 2: Evaluation duration (faster when quick)
        if self._eval_durations:
            avg_duration = sum(self._eval_durations) / len(self._eval_durations)
            if avg_duration < 60:  # Under 1 minute average
                adjustment *= 0.8  # 20% faster
            elif avg_duration < 120:  # Under 2 minutes
                adjustment *= 0.9  # 10% faster
            elif avg_duration > 300:  # Over 5 minutes
                adjustment *= 1.3  # Slow down

        # Factor 3: Promotion velocity (faster when getting results)
        if promotion_velocity > 0.5:  # More than 0.5 promotions/hour
            adjustment *= 0.8  # Ride the momentum - evaluate more often
        elif promotion_velocity < 0.1 and promotion_velocity > 0:
            adjustment *= 0.9  # Try to break plateau with more evaluation

        # Factor 4: Improvement optimizer acceleration
        # When on a promotion streak or high data quality, evaluate faster
        if HAS_IMPROVEMENT_OPTIMIZER:
            try:
                optimizer = get_improvement_optimizer()
                metrics = optimizer.get_improvement_metrics()

                # Consecutive promotions - strong positive signal
                consecutive = metrics.get('consecutive_promotions', 0)
                if consecutive >= 3:
                    adjustment *= 0.7  # 30% faster for strong streak
                elif consecutive >= 2:
                    adjustment *= 0.85  # 15% faster for building streak

                # High data quality means more likely to find promotable models
                if metrics.get('parity_success_rate', 0) >= 0.98:
                    adjustment *= 0.9  # 10% faster with clean data

                # Use optimizer's dynamic evaluation interval
                optimizer_interval = get_evaluation_interval(base_interval)
                if optimizer_interval < base_interval:
                    # Blend with optimizer recommendation
                    adjustment = min(adjustment, optimizer_interval / base_interval)
            except Exception:
                pass  # Don't fail evaluation for optimizer errors

        # Calculate final interval
        final_interval = int(base_interval * adjustment)

        # Clamp to configured bounds
        final_interval = max(min_interval, min(max_interval, final_interval))

        # Log significant changes
        now = time.time()
        if now - self._last_interval_adjustment > 300:  # Log at most every 5 min
            if final_interval != base_interval:
                print(f"[ShadowTournament] Adaptive interval: {final_interval}s "
                      f"(base={base_interval}s, success_rate={self._eval_success_rate:.1%}, "
                      f"promotion_velocity={promotion_velocity:.2f}/hr)")
                self._last_interval_adjustment = now

        return final_interval
