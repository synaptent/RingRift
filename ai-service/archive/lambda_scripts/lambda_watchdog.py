#!/usr/bin/env python3
"""Lambda Cluster Watchdog - Monitor and restart terminated instances.

This script monitors Lambda cloud instances and can automatically restart
terminated instances to maintain cluster capacity for AI training.

Usage:
    # Check cluster status (dry run)
    python scripts/lambda_watchdog.py

    # Auto-restart terminated instances
    python scripts/lambda_watchdog.py --auto-restart

    # Cron entry (hourly)
    0 * * * * cd ~/ringrift/ai-service && python scripts/lambda_watchdog.py --auto-restart >> logs/lambda_watchdog.log 2>&1

Configuration:
    Environment variables:
        LAMBDA_API_KEY: Required for API access
        LAMBDA_MAX_HOURLY_BUDGET: Max $/hr to spend (default: 100.0)
        LAMBDA_RESTART_GPU_TYPES: Comma-separated GPU types to restart (default: gh200,h100)
        LAMBDA_SSH_KEY_NAME: SSH key name in Lambda account (default: ringrift-cluster)

Events Emitted:
    - CLUSTER_CAPACITY_CHANGED: When instances are started/stopped
    - CLUSTER_HEALTH_WARNING: When capacity drops below threshold
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.coordination.providers.lambda_provider import LambdaProvider, LAMBDA_INSTANCE_TYPES
from app.coordination.providers.base import InstanceStatus, GPUType

logger = logging.getLogger(__name__)


class LambdaWatchdog:
    """Monitor and manage Lambda cluster instances."""

    def __init__(
        self,
        max_hourly_budget: float = 100.0,
        restart_gpu_types: list[str] | None = None,
        ssh_key_name: str = "ringrift-cluster",
    ):
        """Initialize watchdog.

        Args:
            max_hourly_budget: Maximum hourly spending allowed ($/hr)
            restart_gpu_types: GPU types to restart (e.g., ["gh200", "h100"])
            ssh_key_name: SSH key name registered in Lambda account
        """
        self.provider = LambdaProvider()
        self.max_hourly_budget = max_hourly_budget
        self.restart_gpu_types = restart_gpu_types or ["gh200", "h100"]
        self.ssh_key_name = ssh_key_name

        # Map short names to GPUType enum
        self._gpu_type_map = {
            "a10": GPUType.A10,
            "a100": GPUType.A100_80GB,
            "a100_40": GPUType.A100_40GB,
            "a100_80": GPUType.A100_80GB,
            "h100": GPUType.H100_80GB,
            "gh200": GPUType.GH200_96GB,
        }

    async def get_cluster_status(self) -> dict:
        """Get current cluster status.

        Returns:
            Dictionary with instance counts, GPU memory, and costs
        """
        if not self.provider.is_configured():
            logger.error("Lambda API key not configured (LAMBDA_API_KEY)")
            return {"error": "API key not configured"}

        instances = await self.provider.list_instances()

        running = [i for i in instances if i.status == InstanceStatus.RUNNING]
        terminated = [i for i in instances if i.status == InstanceStatus.TERMINATED]
        other = [i for i in instances if i.status not in (InstanceStatus.RUNNING, InstanceStatus.TERMINATED)]

        # Calculate totals
        running_gpu_memory = sum(i.gpu_memory_gb for i in running)
        terminated_gpu_memory = sum(i.gpu_memory_gb for i in terminated)
        running_hourly_cost = sum(i.cost_per_hour for i in running)

        return {
            "timestamp": datetime.now().isoformat(),
            "running_count": len(running),
            "terminated_count": len(terminated),
            "other_count": len(other),
            "running_gpu_memory_gb": running_gpu_memory,
            "terminated_gpu_memory_gb": terminated_gpu_memory,
            "running_hourly_cost": running_hourly_cost,
            "budget_remaining": self.max_hourly_budget - running_hourly_cost,
            "running_instances": [
                {
                    "id": i.id,
                    "name": i.name,
                    "gpu_type": i.gpu_type.value,
                    "gpu_memory_gb": i.gpu_memory_gb,
                    "ip": i.ip_address,
                    "cost_per_hour": i.cost_per_hour,
                }
                for i in running
            ],
            "terminated_instances": [
                {
                    "id": i.id,
                    "name": i.name,
                    "gpu_type": i.gpu_type.value,
                    "gpu_memory_gb": i.gpu_memory_gb,
                }
                for i in terminated
            ],
        }

    async def restart_terminated_instances(
        self,
        dry_run: bool = False,
        max_to_restart: int | None = None,
    ) -> dict:
        """Restart terminated instances within budget.

        Args:
            dry_run: If True, don't actually restart instances
            max_to_restart: Maximum number of instances to restart

        Returns:
            Dictionary with restart results
        """
        status = await self.get_cluster_status()

        if "error" in status:
            return status

        terminated = status.get("terminated_instances", [])
        if not terminated:
            logger.info("No terminated instances to restart")
            return {"restarted": [], "skipped": [], "message": "No terminated instances"}

        budget_remaining = status.get("budget_remaining", 0)
        logger.info(
            f"Found {len(terminated)} terminated instances, "
            f"${budget_remaining:.2f}/hr budget remaining"
        )

        # Filter to restartable GPU types
        restartable = []
        for inst in terminated:
            gpu_short = inst["gpu_type"].lower().replace("_", "")
            if any(t in gpu_short for t in self.restart_gpu_types):
                restartable.append(inst)

        if not restartable:
            logger.info(f"No instances match restart GPU types: {self.restart_gpu_types}")
            return {
                "restarted": [],
                "skipped": terminated,
                "message": f"No instances match GPU types: {self.restart_gpu_types}",
            }

        # Sort by cost (cheapest first) to maximize capacity within budget
        restartable.sort(key=lambda i: i.get("cost_per_hour", 999))

        restarted = []
        skipped = []
        current_spend = status.get("running_hourly_cost", 0)

        for inst in restartable:
            if max_to_restart and len(restarted) >= max_to_restart:
                skipped.append(inst)
                continue

            # Check budget
            inst_cost = self._estimate_instance_cost(inst["gpu_type"])
            if current_spend + inst_cost > self.max_hourly_budget:
                logger.warning(
                    f"Skipping {inst['name']} (${inst_cost:.2f}/hr) - would exceed budget"
                )
                skipped.append(inst)
                continue

            if dry_run:
                logger.info(f"[DRY RUN] Would restart: {inst['name']} ({inst['gpu_type']})")
                restarted.append({"instance": inst, "dry_run": True})
            else:
                # Restart the instance
                success = await self._restart_instance(inst)
                if success:
                    restarted.append({"instance": inst, "success": True})
                    current_spend += inst_cost
                    logger.info(f"Restarted: {inst['name']} ({inst['gpu_type']})")
                else:
                    skipped.append(inst)
                    logger.error(f"Failed to restart: {inst['name']}")

        # Emit event if instances were restarted
        if restarted and not dry_run:
            await self._emit_capacity_changed(restarted)

        return {
            "restarted": restarted,
            "skipped": skipped,
            "new_hourly_cost": current_spend,
            "budget_remaining": self.max_hourly_budget - current_spend,
        }

    def _estimate_instance_cost(self, gpu_type: str) -> float:
        """Estimate hourly cost for GPU type."""
        cost_map = {
            "a10": 0.75,
            "a100_40gb": 1.29,
            "a100_80gb": 1.69,
            "h100_80gb": 2.49,
            "gh200_96gb": 2.99,
        }
        normalized = gpu_type.lower().replace(" ", "_")
        for key, cost in cost_map.items():
            if key in normalized:
                return cost
        return 3.0  # Default to GH200 cost

    async def _restart_instance(self, inst: dict) -> bool:
        """Restart a terminated instance.

        Note: Lambda doesn't support restart - we need to launch a new instance.
        This attempts to launch a new instance of the same type.
        """
        try:
            # Map GPU type string back to enum
            gpu_type = None
            for key, enum_val in self._gpu_type_map.items():
                if key in inst["gpu_type"].lower():
                    gpu_type = enum_val
                    break

            if not gpu_type:
                logger.error(f"Unknown GPU type: {inst['gpu_type']}")
                return False

            # Launch new instance
            new_instances = await self.provider.scale_up(
                gpu_type=gpu_type,
                count=1,
                name_prefix="ringrift",
            )

            return len(new_instances) > 0

        except Exception as e:
            logger.error(f"Failed to restart instance: {e}")
            return False

    async def _emit_capacity_changed(self, restarted: list) -> None:
        """Emit CLUSTER_CAPACITY_CHANGED event."""
        try:
            from app.coordination.event_router import get_router
            from app.distributed.data_events import DataEventType

            router = get_router()
            if router is None:
                return

            await router.publish(
                DataEventType.REGISTRY_UPDATED,  # Use existing event type
                {
                    "event_subtype": "cluster_capacity_changed",
                    "instances_restarted": len(restarted),
                    "gpu_types": [r["instance"]["gpu_type"] for r in restarted],
                    "timestamp": datetime.now().isoformat(),
                },
            )
            logger.info("Emitted cluster capacity change event")

        except ImportError:
            pass  # Event system not available
        except Exception as e:
            logger.debug(f"Failed to emit event: {e}")

    async def close(self) -> None:
        """Clean up resources."""
        await self.provider.close()


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Lambda Cluster Watchdog - Monitor and restart terminated instances"
    )
    parser.add_argument(
        "--auto-restart",
        action="store_true",
        help="Automatically restart terminated instances",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be restarted without actually restarting",
    )
    parser.add_argument(
        "--max-restart",
        type=int,
        default=None,
        help="Maximum number of instances to restart",
    )
    parser.add_argument(
        "--max-budget",
        type=float,
        default=float(os.environ.get("LAMBDA_MAX_HOURLY_BUDGET", "100.0")),
        help="Maximum hourly budget in dollars (default: $100/hr)",
    )
    parser.add_argument(
        "--gpu-types",
        type=str,
        default=os.environ.get("LAMBDA_RESTART_GPU_TYPES", "gh200,h100"),
        help="Comma-separated GPU types to restart (default: gh200,h100)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output in JSON format",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output",
    )

    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    # Parse GPU types
    gpu_types = [t.strip().lower() for t in args.gpu_types.split(",")]

    watchdog = LambdaWatchdog(
        max_hourly_budget=args.max_budget,
        restart_gpu_types=gpu_types,
    )

    try:
        # Get current status
        status = await watchdog.get_cluster_status()

        if "error" in status:
            print(f"Error: {status['error']}")
            return 1

        if args.json:
            import json
            print(json.dumps(status, indent=2))
        else:
            print(f"\n=== Lambda Cluster Status ({status['timestamp']}) ===")
            print(f"Running instances:    {status['running_count']}")
            print(f"Terminated instances: {status['terminated_count']}")
            print(f"Running GPU memory:   {status['running_gpu_memory_gb']} GB")
            print(f"Idle GPU memory:      {status['terminated_gpu_memory_gb']} GB")
            print(f"Current hourly cost:  ${status['running_hourly_cost']:.2f}/hr")
            print(f"Budget remaining:     ${status['budget_remaining']:.2f}/hr")

            if status['terminated_instances']:
                print(f"\nTerminated instances ({len(status['terminated_instances'])}):")
                for inst in status['terminated_instances']:
                    print(f"  - {inst['name']}: {inst['gpu_type']} ({inst['gpu_memory_gb']}GB)")

        # Auto-restart if requested
        if args.auto_restart or args.dry_run:
            print("\n=== Restart Operation ===")
            result = await watchdog.restart_terminated_instances(
                dry_run=args.dry_run,
                max_to_restart=args.max_restart,
            )

            if args.json:
                import json
                print(json.dumps(result, indent=2))
            else:
                if result.get("restarted"):
                    prefix = "[DRY RUN] " if args.dry_run else ""
                    print(f"{prefix}Restarted {len(result['restarted'])} instances:")
                    for r in result["restarted"]:
                        print(f"  - {r['instance']['name']} ({r['instance']['gpu_type']})")

                if result.get("skipped"):
                    print(f"Skipped {len(result['skipped'])} instances (budget/type mismatch)")

                if not args.dry_run and result.get("restarted"):
                    print(f"\nNew hourly cost: ${result.get('new_hourly_cost', 0):.2f}/hr")
                    print(f"Budget remaining: ${result.get('budget_remaining', 0):.2f}/hr")

        return 0

    finally:
        await watchdog.close()


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
