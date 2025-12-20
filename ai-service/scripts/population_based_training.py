#!/usr/bin/env python3
"""Population-Based Training (PBT) for RingRift AI.

Implements population-based training where multiple models train in parallel,
with periodic exploitation (copying weights from better performers) and
exploration (perturbing hyperparameters).

Benefits:
- Automatically discovers good hyperparameter schedules
- Adapts hyperparameters during training, not just at start
- More efficient than grid/random search
- Robust to local optima

Reference: Jaderberg et al., "Population Based Training of Neural Networks" (2017)

Usage:
    # Run PBT with 8 population members
    python scripts/population_based_training.py \
        --population-size 8 \
        --board square8 --players 2

    # Resume PBT run
    python scripts/population_based_training.py --resume pbt_run_123

    # PBT with custom hyperparameters to tune
    python scripts/population_based_training.py \
        --tune learning_rate,batch_size,temperature
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

AI_SERVICE_ROOT = Path(__file__).resolve().parents[1]

# Unified logging setup
from scripts.lib.logging_config import setup_script_logging

logger = setup_script_logging("population_based_training")


# Default hyperparameter ranges for exploration
HYPERPARAMETER_RANGES = {
    "learning_rate": {"min": 1e-5, "max": 1e-2, "scale": "log"},
    "batch_size": {"min": 32, "max": 256, "scale": "linear", "type": "int"},
    "weight_decay": {"min": 1e-6, "max": 1e-3, "scale": "log"},
    "temperature": {"min": 0.5, "max": 2.0, "scale": "linear"},
    "mcts_simulations": {"min": 100, "max": 800, "scale": "linear", "type": "int"},
    "dropout": {"min": 0.0, "max": 0.3, "scale": "linear"},
}

# PBT configuration
DEFAULT_POPULATION_SIZE = 8
DEFAULT_EXPLOIT_INTERVAL = 1000  # Steps between exploit/explore
DEFAULT_READY_THRESHOLD = 0.8  # Fraction of interval before eligible for exploit


@dataclass
class PopulationMember:
    """A single member of the PBT population."""
    member_id: str
    hyperparams: dict[str, float]
    model_path: str | None = None
    performance: float = 0.0  # Current Elo or evaluation metric
    steps: int = 0
    generations: int = 0
    parent_id: str | None = None
    created_at: str = ""
    last_exploit_step: int = 0


@dataclass
class PBTState:
    """Complete state of a PBT run."""
    run_id: str
    board_type: str
    num_players: int
    population: list[PopulationMember]
    tunable_params: list[str]
    total_steps: int = 0
    exploit_interval: int = DEFAULT_EXPLOIT_INTERVAL
    best_performance: float = 0.0
    best_hyperparams: dict[str, float] = field(default_factory=dict)
    history: list[dict[str, Any]] = field(default_factory=list)
    created_at: str = ""


def sample_hyperparameter(name: str, current: float | None = None, perturb: bool = False) -> float:
    """Sample or perturb a hyperparameter value.

    Args:
        name: Hyperparameter name
        current: Current value (for perturbation)
        perturb: If True, perturb current value; else sample fresh

    Returns:
        New hyperparameter value
    """
    if name not in HYPERPARAMETER_RANGES:
        return current if current is not None else 0.0

    spec = HYPERPARAMETER_RANGES[name]
    min_val = spec["min"]
    max_val = spec["max"]
    scale = spec.get("scale", "linear")
    is_int = spec.get("type") == "int"

    if perturb and current is not None:
        # Perturb by multiplying by 0.8 or 1.2
        factor = random.choice([0.8, 1.2])
        if scale == "log":
            new_val = current * factor
        else:
            new_val = current * factor
        new_val = max(min_val, min(max_val, new_val))
    else:
        # Sample fresh
        if scale == "log":
            log_min = np.log10(min_val)
            log_max = np.log10(max_val)
            new_val = 10 ** np.random.uniform(log_min, log_max)
        else:
            new_val = np.random.uniform(min_val, max_val)

    if is_int:
        new_val = int(round(new_val))

    return new_val


def create_population(
    population_size: int,
    tunable_params: list[str],
    board_type: str,
    num_players: int,
) -> list[PopulationMember]:
    """Create initial population with random hyperparameters."""
    population = []

    for i in range(population_size):
        member_id = f"member_{i:03d}"
        hyperparams = {}

        for param in tunable_params:
            hyperparams[param] = sample_hyperparameter(param)

        member = PopulationMember(
            member_id=member_id,
            hyperparams=hyperparams,
            created_at=datetime.utcnow().isoformat() + "Z",
        )
        population.append(member)

    return population


def exploit(
    population: list[PopulationMember],
    member: PopulationMember,
    exploit_fraction: float = 0.2,
) -> PopulationMember | None:
    """Exploit: potentially copy weights from a better performer.

    Returns the member to copy from, or None if no exploitation.
    """
    # Sort population by performance
    sorted_pop = sorted(population, key=lambda m: m.performance, reverse=True)

    # Find member's rank
    member_rank = next(i for i, m in enumerate(sorted_pop) if m.member_id == member.member_id)

    # If in bottom 20%, copy from top 20%
    top_cutoff = int(len(population) * exploit_fraction)
    bottom_cutoff = len(population) - int(len(population) * exploit_fraction)

    if member_rank >= bottom_cutoff and top_cutoff > 0:
        # Select random member from top performers
        top_members = sorted_pop[:top_cutoff]
        return random.choice(top_members)

    return None


def explore(hyperparams: dict[str, float], tunable_params: list[str]) -> dict[str, float]:
    """Explore: perturb hyperparameters."""
    new_hyperparams = copy.deepcopy(hyperparams)

    for param in tunable_params:
        if param in new_hyperparams:
            new_hyperparams[param] = sample_hyperparameter(
                param, current=new_hyperparams[param], perturb=True
            )

    return new_hyperparams


def run_pbt_step(
    state: PBTState,
    member: PopulationMember,
    training_steps: int = 100,
) -> float:
    """Run one PBT step for a population member.

    In a real implementation, this would:
    1. Train the model for `training_steps` steps
    2. Evaluate performance
    3. Return the new performance metric

    Returns:
        New performance metric (e.g., Elo)
    """
    # Simulate training (in production, would call actual training)
    member.steps += training_steps

    # Simulate performance improvement with noise
    base_improvement = 0.1 * training_steps / 1000
    lr_factor = np.log10(member.hyperparams.get("learning_rate", 1e-3)) + 4  # Normalize around 1
    noise = np.random.normal(0, 0.5)

    new_performance = member.performance + base_improvement * lr_factor + noise
    member.performance = max(0, new_performance)

    return member.performance


def should_exploit_explore(member: PopulationMember, state: PBTState) -> bool:
    """Check if member is ready for exploit/explore."""
    steps_since_last = member.steps - member.last_exploit_step
    return steps_since_last >= state.exploit_interval * DEFAULT_READY_THRESHOLD


def pbt_iteration(state: PBTState, training_steps: int = 100) -> PBTState:
    """Run one iteration of PBT across all population members.

    Steps:
    1. Train each member for some steps
    2. Check if ready for exploit/explore
    3. If ready, potentially copy from better member and perturb hyperparams
    """
    for member in state.population:
        # Train
        performance = run_pbt_step(state, member, training_steps)

        # Check for exploit/explore
        if should_exploit_explore(member, state):
            # Exploit: maybe copy from better member
            source = exploit(state.population, member)
            if source is not None:
                logger.info(
                    f"Member {member.member_id} exploiting from {source.member_id} "
                    f"(perf: {member.performance:.1f} -> {source.performance:.1f})"
                )
                member.model_path = source.model_path  # Would copy weights in production
                member.performance = source.performance
                member.parent_id = source.member_id

                # Explore: perturb hyperparameters
                member.hyperparams = explore(member.hyperparams, state.tunable_params)
                member.generations += 1

            member.last_exploit_step = member.steps

        # Track best
        if performance > state.best_performance:
            state.best_performance = performance
            state.best_hyperparams = copy.deepcopy(member.hyperparams)
            logger.info(f"New best performance: {performance:.2f} with {member.hyperparams}")

    state.total_steps += training_steps

    # Record history
    state.history.append({
        "step": state.total_steps,
        "best_performance": state.best_performance,
        "performances": [m.performance for m in state.population],
        "timestamp": datetime.utcnow().isoformat() + "Z",
    })

    return state


def save_pbt_state(state: PBTState, output_dir: Path):
    """Save PBT state to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)

    state_file = output_dir / "pbt_state.json"
    with open(state_file, "w") as f:
        json.dump({
            "run_id": state.run_id,
            "board_type": state.board_type,
            "num_players": state.num_players,
            "tunable_params": state.tunable_params,
            "total_steps": state.total_steps,
            "exploit_interval": state.exploit_interval,
            "best_performance": state.best_performance,
            "best_hyperparams": state.best_hyperparams,
            "created_at": state.created_at,
            "population": [asdict(m) for m in state.population],
            "history": state.history[-100:],  # Keep last 100 entries
        }, f, indent=2)

    logger.info(f"Saved PBT state to {state_file}")


def load_pbt_state(state_file: Path) -> PBTState:
    """Load PBT state from disk."""
    with open(state_file) as f:
        data = json.load(f)

    population = [PopulationMember(**m) for m in data["population"]]

    return PBTState(
        run_id=data["run_id"],
        board_type=data["board_type"],
        num_players=data["num_players"],
        population=population,
        tunable_params=data["tunable_params"],
        total_steps=data["total_steps"],
        exploit_interval=data["exploit_interval"],
        best_performance=data["best_performance"],
        best_hyperparams=data["best_hyperparams"],
        history=data.get("history", []),
        created_at=data["created_at"],
    )


def print_population_status(state: PBTState):
    """Print current population status."""
    print("\n" + "=" * 70)
    print(f"PBT STATUS - Step {state.total_steps}")
    print("=" * 70)
    print(f"Best Performance: {state.best_performance:.2f}")
    print(f"Best Hyperparams: {state.best_hyperparams}")

    print("\nPopulation:")
    sorted_pop = sorted(state.population, key=lambda m: m.performance, reverse=True)
    for i, member in enumerate(sorted_pop):
        lr = member.hyperparams.get("learning_rate", 0)
        print(f"  {i+1}. {member.member_id}: perf={member.performance:.2f}, lr={lr:.2e}, gen={member.generations}")


def main():
    parser = argparse.ArgumentParser(
        description="Population-Based Training for RingRift AI"
    )

    parser.add_argument(
        "--population-size",
        type=int,
        default=DEFAULT_POPULATION_SIZE,
        help="Number of population members",
    )
    parser.add_argument(
        "--board",
        type=str,
        default="square8",
        help="Board type",
    )
    parser.add_argument(
        "--players",
        type=int,
        default=2,
        help="Number of players",
    )
    parser.add_argument(
        "--tune",
        type=str,
        default="learning_rate,batch_size,temperature",
        help="Comma-separated list of hyperparameters to tune",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=100,
        help="Number of PBT iterations",
    )
    parser.add_argument(
        "--steps-per-iteration",
        type=int,
        default=100,
        help="Training steps per iteration",
    )
    parser.add_argument(
        "--exploit-interval",
        type=int,
        default=DEFAULT_EXPLOIT_INTERVAL,
        help="Steps between exploit/explore checks",
    )
    parser.add_argument(
        "--resume",
        type=str,
        help="Resume from existing PBT run",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(AI_SERVICE_ROOT / "logs" / "pbt"),
        help="Output directory",
    )

    args = parser.parse_args()

    tunable_params = [p.strip() for p in args.tune.split(",")]
    output_dir = Path(args.output_dir)

    # Initialize or resume
    if args.resume:
        state_file = Path(args.resume) / "pbt_state.json"
        if state_file.exists():
            state = load_pbt_state(state_file)
            logger.info(f"Resumed PBT run {state.run_id}")
        else:
            logger.error(f"State file not found: {state_file}")
            return 1
    else:
        run_id = f"pbt_{int(time.time())}"
        population = create_population(
            args.population_size, tunable_params, args.board, args.players
        )
        state = PBTState(
            run_id=run_id,
            board_type=args.board,
            num_players=args.players,
            population=population,
            tunable_params=tunable_params,
            exploit_interval=args.exploit_interval,
            created_at=datetime.utcnow().isoformat() + "Z",
        )
        output_dir = output_dir / run_id
        logger.info(f"Created new PBT run: {run_id}")

    # Run PBT
    logger.info(f"Running {args.iterations} PBT iterations...")
    for i in range(args.iterations):
        state = pbt_iteration(state, args.steps_per_iteration)

        if (i + 1) % 10 == 0:
            print_population_status(state)
            save_pbt_state(state, output_dir)

    # Final save
    save_pbt_state(state, output_dir)
    print_population_status(state)

    print("\n" + "=" * 70)
    print("PBT COMPLETE")
    print("=" * 70)
    print(f"Final best performance: {state.best_performance:.2f}")
    print(f"Best hyperparameters: {state.best_hyperparams}")
    print(f"Output saved to: {output_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
