#!/usr/bin/env python3
"""Vast.ai Demand-Based Autoscaler - Scale instances based on P2P workload.

Monitors the P2P network and scales Vast instances up/down based on:
- Selfplay job queue depth
- Idle instance count
- Budget constraints
- Time-of-day policies
- Vast autoscaler groups for automatic replacement

Usage:
    python scripts/vast_autoscaler.py --status          # Check scaling status
    python scripts/vast_autoscaler.py --scale           # Execute scaling decisions
    python scripts/vast_autoscaler.py --auto            # Full auto-scaling cycle
    python scripts/vast_autoscaler.py --dry-run         # Preview without changes
    python scripts/vast_autoscaler.py --create-group    # Create autoscaler group
    python scripts/vast_autoscaler.py --list-groups     # List autoscaler groups

Designed to run via cron every 10-15 minutes.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

AI_SERVICE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AI_SERVICE_ROOT))

LOG_DIR = AI_SERVICE_ROOT / "logs"
LOG_FILE = LOG_DIR / "vast_autoscaler.log"
STATE_FILE = AI_SERVICE_ROOT / "data" / "autoscaler_state.json"

LOG_DIR.mkdir(parents=True, exist_ok=True)
STATE_FILE.parent.mkdir(parents=True, exist_ok=True)

from scripts.lib.logging_config import setup_script_logging
from scripts.lib.state_manager import StateManager

logger = setup_script_logging("vast_autoscaler")

# =============================================================================
# Configuration
# =============================================================================

# SAFETY: Disable automatic instance termination to prevent data loss
SCALE_DOWN_DISABLED = True

@dataclass
class ScalingConfig:
    """Autoscaling configuration."""
    # Budget limits
    max_hourly_spend: float = 10.0          # Max $/hr total
    max_daily_spend: float = 200.0          # Max $/day total
    max_instances: int = 20                  # Max concurrent instances
    min_instances: int = 5                   # Min instances to keep running

    # Scale-up triggers
    scale_up_queue_threshold: int = 500     # Queue depth to trigger scale-up
    scale_up_utilization_threshold: float = 0.8  # Avg utilization to scale up
    scale_up_cooldown_minutes: int = 15     # Wait between scale-ups

    # Scale-down triggers
    scale_down_idle_minutes: int = 30       # Idle time before scale-down
    scale_down_utilization_threshold: float = 0.3  # Avg util to scale down
    scale_down_cooldown_minutes: int = 10   # Wait between scale-downs

    # Instance preferences (ordered by cost-effectiveness)
    preferred_gpus: list[dict[str, Any]] = None

    def __post_init__(self):
        if self.preferred_gpus is None:
            self.preferred_gpus = [
                {"name": "RTX 3070", "max_price": 0.08, "min_reliability": 0.95},
                {"name": "RTX 3060", "max_price": 0.06, "min_reliability": 0.95},
                {"name": "RTX 2080 Ti", "max_price": 0.10, "min_reliability": 0.95},
                {"name": "RTX 4060 Ti", "max_price": 0.12, "min_reliability": 0.90},
                {"name": "RTX 3060 Ti", "max_price": 0.08, "min_reliability": 0.95},
            ]


@dataclass
class ScalingState:
    """Persistent autoscaler state."""
    last_scale_up: datetime | None = None
    last_scale_down: datetime | None = None
    instances_created_today: int = 0
    instances_terminated_today: int = 0
    daily_spend: float = 0.0
    last_reset: datetime | None = None

    def to_dict(self) -> dict:
        return {
            "last_scale_up": self.last_scale_up.isoformat() if self.last_scale_up else None,
            "last_scale_down": self.last_scale_down.isoformat() if self.last_scale_down else None,
            "instances_created_today": self.instances_created_today,
            "instances_terminated_today": self.instances_terminated_today,
            "daily_spend": self.daily_spend,
            "last_reset": self.last_reset.isoformat() if self.last_reset else None,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ScalingState":
        return cls(
            last_scale_up=datetime.fromisoformat(data["last_scale_up"]) if data.get("last_scale_up") else None,
            last_scale_down=datetime.fromisoformat(data["last_scale_down"]) if data.get("last_scale_down") else None,
            instances_created_today=data.get("instances_created_today", 0),
            instances_terminated_today=data.get("instances_terminated_today", 0),
            daily_spend=data.get("daily_spend", 0.0),
            last_reset=datetime.fromisoformat(data["last_reset"]) if data.get("last_reset") else None,
        )


# =============================================================================
# P2P Workload Monitoring
# =============================================================================

def _load_p2p_leaders_from_config() -> list[str]:
    """Load P2P leader endpoints from config or environment."""
    config_path = Path(__file__).parent.parent / "config" / "distributed_hosts.yaml"
    leaders = []

    if not config_path.exists():
        logger.warning("[Autoscaler] Warning: No config found at %s", config_path)
        return []

    try:
        import yaml
        with open(config_path) as f:
            config = yaml.safe_load(f) or {}

        hosts = config.get("hosts", {})
        # Get hosts with p2p_voter role or primary training nodes
        for _host_id, host_info in hosts.items():
            if host_info.get("status") != "ready":
                continue
            tailscale_ip = host_info.get("tailscale_ip")
            if tailscale_ip and tailscale_ip.startswith("100.") and host_info.get("p2p_voter") or "primary" in host_info.get("role", ""):
                leaders.append(f"http://{tailscale_ip}:8770")

        return leaders[:5]  # Limit to top 5
    except Exception as e:
        logger.warning("[Autoscaler] Error loading config: %s", e)
        return []


def get_p2p_status() -> dict | None:
    """Get P2P network status from leader."""
    import urllib.request

    leaders = _load_p2p_leaders_from_config()
    if not leaders:
        logger.warning("No P2P leaders configured, cannot get P2P status")
        return None

    for leader in leaders:
        try:
            with urllib.request.urlopen(f"{leader}/status", timeout=10) as resp:
                return json.loads(resp.read().decode())
        except Exception:
            continue
    return None


def analyze_workload(p2p_status: dict) -> dict[str, Any]:
    """Analyze P2P workload to determine scaling needs."""
    peers = p2p_status.get("peers", {})

    # Count active (non-retired) peers
    active_peers = {k: v for k, v in peers.items() if not v.get("retired", False)}
    vast_peers = {k: v for k, v in active_peers.items() if "vast" in k.lower()}

    # Calculate metrics
    total_selfplay = sum(p.get("selfplay_jobs", 0) for p in active_peers.values())
    vast_selfplay = sum(p.get("selfplay_jobs", 0) for p in vast_peers.values())

    idle_vast = sum(1 for p in vast_peers.values() if p.get("selfplay_jobs", 0) == 0)
    busy_vast = len(vast_peers) - idle_vast

    # Calculate utilization (jobs per instance, normalized)
    avg_utilization = total_selfplay / max(len(active_peers), 1) / 100  # Normalize to 0-1
    vast_utilization = vast_selfplay / max(len(vast_peers), 1) / 100

    return {
        "total_peers": len(active_peers),
        "vast_peers": len(vast_peers),
        "total_selfplay_jobs": total_selfplay,
        "vast_selfplay_jobs": vast_selfplay,
        "idle_vast_instances": idle_vast,
        "busy_vast_instances": busy_vast,
        "avg_utilization": min(avg_utilization, 1.0),
        "vast_utilization": min(vast_utilization, 1.0),
    }


# =============================================================================
# Vast.ai Instance Management
# =============================================================================

def get_vast_instances() -> list[dict]:
    """Get all Vast instances."""
    try:
        result = subprocess.run(
            ["vastai", "show", "instances", "--raw"],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode == 0:
            return json.loads(result.stdout)
    except Exception as e:
        logger.error(f"Failed to get instances: {e}")
    return []


def get_running_instances() -> list[dict]:
    """Get running Vast instances with cost info."""
    instances = get_vast_instances()
    return [i for i in instances if i.get("actual_status") == "running"]


def get_current_hourly_cost() -> float:
    """Get current hourly spend across all instances."""
    instances = get_running_instances()
    return sum(i.get("dph_total", 0) or 0 for i in instances)


def search_offers(gpu_name: str, max_price: float, min_reliability: float = 0.95) -> list[dict]:
    """Search for GPU offers."""
    try:
        query = f"gpu_name={gpu_name} reliability>{min_reliability} dph<{max_price} rentable=true"
        result = subprocess.run(
            ["vastai", "search", "offers", query, "-o", "dph", "--raw"],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode == 0:
            return json.loads(result.stdout)[:10]  # Top 10 offers
    except Exception as e:
        logger.debug(f"Search failed for {gpu_name}: {e}")
    return []


def create_instance(offer_id: int, disk_gb: int = 50) -> str | None:
    """Create a new Vast instance."""
    try:
        result = subprocess.run(
            [
                "vastai", "create", "instance", str(offer_id),
                "--image", "pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime",
                "--disk", str(disk_gb),
                "--ssh",
                "--onstart-cmd", "apt-get update && apt-get install -y git curl",
            ],
            capture_output=True, text=True, timeout=60
        )
        if result.returncode == 0:
            output = result.stdout + result.stderr
            for word in output.split():
                if word.isdigit() and len(word) > 6:
                    return word
    except Exception as e:
        logger.error(f"Failed to create instance: {e}")
    return None


def destroy_instance(instance_id: str) -> bool:
    """Destroy a Vast instance."""
    # Safety check: never destroy instances if scale-down is disabled
    if SCALE_DOWN_DISABLED:
        logger.info(f"[SCALE_DOWN_DISABLED] Would destroy instance {instance_id} but termination is disabled")
        return False

    try:
        result = subprocess.run(
            ["vastai", "destroy", "instance", instance_id],
            capture_output=True, text=True, timeout=30
        )
        return result.returncode == 0
    except Exception as e:
        logger.error(f"Failed to destroy instance {instance_id}: {e}")
    return False


def stop_instance(instance_id: str) -> bool:
    """Stop (but don't destroy) a Vast instance."""
    # Safety check: never stop instances if scale-down is disabled
    if SCALE_DOWN_DISABLED:
        logger.info(f"[SCALE_DOWN_DISABLED] Would stop instance {instance_id} but termination is disabled")
        return False

    try:
        result = subprocess.run(
            ["vastai", "stop", "instance", instance_id],
            capture_output=True, text=True, timeout=30
        )
        return result.returncode == 0
    except Exception as e:
        logger.error(f"Failed to stop instance {instance_id}: {e}")
    return False


# =============================================================================
# Vast Autoscaler Groups
# =============================================================================

def get_autoscaler_groups() -> list[dict]:
    """Get all autoscaler groups."""
    try:
        result = subprocess.run(
            ["vastai", "show", "autoscalers", "--raw"],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode == 0:
            return json.loads(result.stdout)
    except Exception as e:
        logger.debug(f"Failed to get autoscaler groups: {e}")
    return []


def create_autoscaler_group(
    name: str,
    gpu_name: str,
    min_instances: int = 2,
    max_instances: int = 10,
    target_instances: int = 5,
    max_price: float = 0.10,
    disk_gb: int = 50,
) -> str | None:
    """Create a new autoscaler group.

    Autoscaler groups automatically maintain a target number of instances,
    replacing terminated instances and scaling based on demand.
    """
    # Build search query for autoscaler
    search_query = f"gpu_name={gpu_name} rentable=true dph<{max_price}"

    try:
        result = subprocess.run(
            [
                "vastai", "create", "autoscaler",
                "--search-query", search_query,
                "--image", "pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime",
                "--disk", str(disk_gb),
                "--min-instances", str(min_instances),
                "--max-instances", str(max_instances),
                "--target-instances", str(target_instances),
                "--onstart-cmd", "apt-get update && apt-get install -y git curl",
                "--name", name,
            ],
            capture_output=True, text=True, timeout=60
        )

        if result.returncode == 0:
            output = result.stdout + result.stderr
            logger.info(f"Created autoscaler group '{name}': {output}")
            # Parse group ID from output
            for word in output.split():
                if word.isdigit():
                    return word
            return output
        else:
            logger.error(f"Failed to create autoscaler group: {result.stderr}")
            return None
    except Exception as e:
        logger.error(f"Failed to create autoscaler group: {e}")
        return None


def update_autoscaler_group(
    group_id: str,
    target_instances: int | None = None,
    min_instances: int | None = None,
    max_instances: int | None = None,
) -> bool:
    """Update an autoscaler group's settings."""
    args = ["vastai", "change", "autoscaler", group_id]

    if target_instances is not None:
        args.extend(["--target-instances", str(target_instances)])
    if min_instances is not None:
        args.extend(["--min-instances", str(min_instances)])
    if max_instances is not None:
        args.extend(["--max-instances", str(max_instances)])

    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=30)
        return result.returncode == 0
    except Exception as e:
        logger.error(f"Failed to update autoscaler group: {e}")
        return False


def delete_autoscaler_group(group_id: str) -> bool:
    """Delete an autoscaler group."""
    try:
        result = subprocess.run(
            ["vastai", "destroy", "autoscaler", group_id],
            capture_output=True, text=True, timeout=30
        )
        return result.returncode == 0
    except Exception as e:
        logger.error(f"Failed to delete autoscaler group: {e}")
        return False


def cmd_list_groups():
    """List all autoscaler groups."""
    groups = get_autoscaler_groups()
    if not groups:
        print("No autoscaler groups found")
        return

    print(f"\n{'ID':<10} {'Name':<20} {'Min':<5} {'Target':<7} {'Max':<5} {'Running':<8} {'Status'}")
    print("-" * 70)

    for g in groups:
        print(f"{g.get('id', '?'):<10} {g.get('name', '?'):<20} "
              f"{g.get('min_instances', '?'):<5} {g.get('target_instances', '?'):<7} "
              f"{g.get('max_instances', '?'):<5} {g.get('cur_instances', '?'):<8} "
              f"{g.get('status', '?')}")


def cmd_create_group(name: str, gpu: str, target: int, max_price: float):
    """Create a new autoscaler group."""
    logger.info(f"Creating autoscaler group '{name}' for {gpu}...")
    group_id = create_autoscaler_group(
        name=name,
        gpu_name=gpu,
        min_instances=2,
        max_instances=target * 2,
        target_instances=target,
        max_price=max_price,
    )
    if group_id:
        print(f"Created autoscaler group: {group_id}")
    else:
        print("Failed to create autoscaler group")


# =============================================================================
# Scaling Logic
# =============================================================================

# Use StateManager for persistent state
_state_manager = StateManager(STATE_FILE, ScalingState)


def load_state() -> ScalingState:
    """Load persistent state."""
    state = _state_manager.load()
    # Reset daily counters if new day
    if state.last_reset:
        if state.last_reset.date() < datetime.now().date():
            state.instances_created_today = 0
            state.instances_terminated_today = 0
            state.daily_spend = 0.0
            state.last_reset = datetime.now()
    elif state.last_reset is None:
        state.last_reset = datetime.now()
    return state


def save_state(state: ScalingState):
    """Save persistent state."""
    _state_manager.save(state)


def should_scale_up(
    workload: dict[str, Any],
    config: ScalingConfig,
    state: ScalingState,
    current_cost: float,
) -> tuple[bool, str]:
    """Determine if we should scale up."""
    # Check cooldown
    if state.last_scale_up:
        elapsed = (datetime.now() - state.last_scale_up).total_seconds() / 60
        if elapsed < config.scale_up_cooldown_minutes:
            return False, f"Scale-up cooldown ({config.scale_up_cooldown_minutes - elapsed:.0f}m remaining)"

    # Check budget
    if current_cost >= config.max_hourly_spend:
        return False, f"At max hourly spend (${current_cost:.2f}/${config.max_hourly_spend:.2f})"

    # Check instance count
    running = len(get_running_instances())
    if running >= config.max_instances:
        return False, f"At max instances ({running}/{config.max_instances})"

    # Check utilization
    if workload["vast_utilization"] >= config.scale_up_utilization_threshold:
        return True, f"High utilization ({workload['vast_utilization']:.0%} >= {config.scale_up_utilization_threshold:.0%})"

    # Check if all instances are busy
    if workload["idle_vast_instances"] == 0 and workload["vast_peers"] > 0:
        return True, "All Vast instances busy"

    # Check queue depth (high selfplay jobs = need more capacity)
    if workload["total_selfplay_jobs"] >= config.scale_up_queue_threshold:
        return True, f"High queue depth ({workload['total_selfplay_jobs']} >= {config.scale_up_queue_threshold})"

    return False, "No scale-up triggers met"


def should_scale_down(
    workload: dict[str, Any],
    config: ScalingConfig,
    state: ScalingState,
) -> tuple[bool, str, list[dict]]:
    """Determine if we should scale down. Returns (should_scale, reason, candidates)."""
    # Check cooldown
    if state.last_scale_down:
        elapsed = (datetime.now() - state.last_scale_down).total_seconds() / 60
        if elapsed < config.scale_down_cooldown_minutes:
            return False, f"Scale-down cooldown ({config.scale_down_cooldown_minutes - elapsed:.0f}m remaining)", []

    # Check minimum instances
    running = get_running_instances()
    vast_running = [i for i in running if "vast" in str(i.get("id", "")).lower() or i.get("ssh_host", "").startswith("ssh")]

    if len(vast_running) <= config.min_instances:
        return False, f"At min instances ({len(vast_running)}/{config.min_instances})", []

    # Check utilization - only scale down if low
    if workload["vast_utilization"] > config.scale_down_utilization_threshold:
        return False, f"Utilization too high ({workload['vast_utilization']:.0%} > {config.scale_down_utilization_threshold:.0%})", []

    # Check for idle instances
    if workload["idle_vast_instances"] == 0:
        return False, "No idle instances", []

    # Find candidates (prefer most expensive idle instances)
    # We need to match Vast instances to P2P status
    idle_candidates = []
    for inst in vast_running:
        inst_id = str(inst.get("id", ""))
        cost = inst.get("dph_total", 0) or 0
        idle_candidates.append({"instance": inst, "cost": cost})

    # Sort by cost (highest first)
    idle_candidates.sort(key=lambda x: -x["cost"])

    if idle_candidates:
        return True, f"Found {workload['idle_vast_instances']} idle instances", idle_candidates[:2]

    return False, "No suitable candidates", []


def execute_scale_up(config: ScalingConfig, state: ScalingState, dry_run: bool = False) -> int:
    """Execute scale-up by creating new instances."""
    created = 0
    current_cost = get_current_hourly_cost()
    budget_remaining = config.max_hourly_spend - current_cost

    for gpu_pref in config.preferred_gpus:
        if budget_remaining <= 0:
            break

        offers = search_offers(
            gpu_pref["name"],
            min(gpu_pref["max_price"], budget_remaining),
            gpu_pref.get("min_reliability", 0.95),
        )

        for offer in offers[:2]:  # Max 2 per GPU type
            offer_id = offer.get("id")
            price = offer.get("dph_total", 0)

            if price > budget_remaining:
                continue

            if dry_run:
                logger.info(f"[DRY-RUN] Would create {gpu_pref['name']} (${price:.3f}/hr)")
                created += 1
                budget_remaining -= price
            else:
                logger.info(f"Creating {gpu_pref['name']} from offer {offer_id} (${price:.3f}/hr)...")
                instance_id = create_instance(offer_id)
                if instance_id:
                    created += 1
                    budget_remaining -= price
                    state.instances_created_today += 1
                    state.last_scale_up = datetime.now()

            if created >= 2:  # Max 2 instances per scale-up
                break

        if created >= 2:
            break

    return created


def execute_scale_down(candidates: list[dict], state: ScalingState, dry_run: bool = False) -> int:
    """Execute scale-down by stopping/destroying idle instances."""
    # Safety check: never terminate instances if disabled
    if SCALE_DOWN_DISABLED:
        logger.info("[SCALE_DOWN_DISABLED] Instance termination is disabled - skipping scale-down")
        return 0

    terminated = 0

    for candidate in candidates[:1]:  # Only terminate 1 at a time
        inst = candidate["instance"]
        inst_id = str(inst.get("id", ""))
        cost = candidate["cost"]

        if dry_run:
            logger.info(f"[DRY-RUN] Would stop instance {inst_id} (${cost:.3f}/hr)")
            terminated += 1
        else:
            logger.info(f"Stopping instance {inst_id} (${cost:.3f}/hr)...")
            if stop_instance(inst_id):
                terminated += 1
                state.instances_terminated_today += 1
                state.last_scale_down = datetime.now()

    return terminated


# =============================================================================
# Main Commands
# =============================================================================

def cmd_status(config: ScalingConfig):
    """Show current autoscaling status."""
    logger.info("=" * 70)
    logger.info("VAST AUTOSCALER STATUS")
    logger.info("=" * 70)

    # Get P2P status
    p2p_status = get_p2p_status()
    if not p2p_status:
        logger.warning("Could not reach P2P network")
        workload = {
            "total_peers": 0, "vast_peers": 0, "total_selfplay_jobs": 0,
            "vast_selfplay_jobs": 0, "idle_vast_instances": 0, "busy_vast_instances": 0,
            "avg_utilization": 0, "vast_utilization": 0,
        }
    else:
        workload = analyze_workload(p2p_status)

    # Get Vast status
    instances = get_running_instances()
    current_cost = sum(i.get("dph_total", 0) or 0 for i in instances)

    # Load state
    state = load_state()

    print(f"\n{'Metric':<30} {'Value':<20}")
    print("-" * 50)
    print(f"{'Running Vast instances':<30} {len(instances):<20}")
    print(f"{'Hourly cost':<30} ${current_cost:.2f}")
    print(f"{'Daily cost (projected)':<30} ${current_cost * 24:.2f}")
    print(f"{'Budget (hourly)':<30} ${config.max_hourly_spend:.2f}")
    print(f"{'Budget remaining':<30} ${max(0, config.max_hourly_spend - current_cost):.2f}")
    print()
    print(f"{'P2P Peers (total)':<30} {workload['total_peers']:<20}")
    print(f"{'Vast peers in P2P':<30} {workload['vast_peers']:<20}")
    print(f"{'Total selfplay jobs':<30} {workload['total_selfplay_jobs']:<20}")
    print(f"{'Idle Vast instances':<30} {workload['idle_vast_instances']:<20}")
    print(f"{'Vast utilization':<30} {workload['vast_utilization']:.0%}")
    print()
    print(f"{'Instances created today':<30} {state.instances_created_today:<20}")
    print(f"{'Instances terminated today':<30} {state.instances_terminated_today:<20}")

    # Check scaling decisions
    scale_up, up_reason = should_scale_up(workload, config, state, current_cost)
    scale_down, down_reason, _ = should_scale_down(workload, config, state)

    print()
    print(f"{'Scale-up decision':<30} {'YES' if scale_up else 'NO'} - {up_reason}")
    print(f"{'Scale-down decision':<30} {'YES' if scale_down else 'NO'} - {down_reason}")


def cmd_scale(config: ScalingConfig, dry_run: bool = False):
    """Execute scaling decisions."""
    logger.info("=" * 70)
    logger.info(f"VAST AUTOSCALER - {'DRY RUN' if dry_run else 'SCALING'}")
    logger.info("=" * 70)

    # Get P2P status
    p2p_status = get_p2p_status()
    if not p2p_status:
        logger.warning("Could not reach P2P network, using instance-only metrics")
        workload = {
            "total_peers": 0, "vast_peers": 0, "total_selfplay_jobs": 0,
            "vast_selfplay_jobs": 0, "idle_vast_instances": 0, "busy_vast_instances": 0,
            "avg_utilization": 0, "vast_utilization": 0,
        }
    else:
        workload = analyze_workload(p2p_status)
        logger.info(f"Workload: {workload['total_selfplay_jobs']} jobs, {workload['vast_utilization']:.0%} util, {workload['idle_vast_instances']} idle")

    # Load state
    state = load_state()
    current_cost = get_current_hourly_cost()

    # Check scale-up
    scale_up, up_reason = should_scale_up(workload, config, state, current_cost)
    if scale_up:
        logger.info(f"Scale-up triggered: {up_reason}")
        created = execute_scale_up(config, state, dry_run)
        logger.info(f"Created {created} instances")
    else:
        logger.info(f"No scale-up: {up_reason}")

    # Check scale-down
    scale_down, down_reason, candidates = should_scale_down(workload, config, state)
    if scale_down and candidates:
        logger.info(f"Scale-down triggered: {down_reason}")
        terminated = execute_scale_down(candidates, state, dry_run)
        logger.info(f"Terminated {terminated} instances")
    else:
        logger.info(f"No scale-down: {down_reason}")

    # Save state
    if not dry_run:
        save_state(state)

    logger.info("=" * 70)


def cmd_auto(config: ScalingConfig):
    """Full auto-scaling cycle."""
    logger.info(f"Auto-scaling cycle at {datetime.now().isoformat()}")
    cmd_scale(config, dry_run=False)


def main():
    parser = argparse.ArgumentParser(description="Vast.ai Demand-Based Autoscaler")
    parser.add_argument("--status", action="store_true", help="Show scaling status")
    parser.add_argument("--scale", action="store_true", help="Execute scaling decisions")
    parser.add_argument("--auto", action="store_true", help="Full auto-scaling cycle")
    parser.add_argument("--dry-run", action="store_true", help="Preview without changes")

    # Autoscaler groups
    parser.add_argument("--list-groups", action="store_true", help="List autoscaler groups")
    parser.add_argument("--create-group", type=str, metavar="NAME", help="Create autoscaler group")
    parser.add_argument("--gpu", type=str, default="RTX 3070", help="GPU type for group")
    parser.add_argument("--target", type=int, default=5, help="Target instances for group")
    parser.add_argument("--max-price", type=float, default=0.10, help="Max price for group")

    # Config overrides
    parser.add_argument("--max-hourly", type=float, default=10.0, help="Max hourly spend")
    parser.add_argument("--max-instances", type=int, default=20, help="Max instances")
    parser.add_argument("--min-instances", type=int, default=5, help="Min instances")

    args = parser.parse_args()

    config = ScalingConfig(
        max_hourly_spend=args.max_hourly,
        max_instances=args.max_instances,
        min_instances=args.min_instances,
    )

    if args.list_groups:
        cmd_list_groups()
    elif args.create_group:
        cmd_create_group(args.create_group, args.gpu, args.target, args.max_price)
    elif args.status:
        cmd_status(config)
    elif args.scale or args.dry_run:
        cmd_scale(config, dry_run=args.dry_run)
    elif args.auto:
        cmd_auto(config)
    else:
        cmd_status(config)


if __name__ == "__main__":
    main()
