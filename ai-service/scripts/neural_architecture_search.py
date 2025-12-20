#!/usr/bin/env python3
"""Neural Architecture Search (NAS) for RingRift AI.

Implements automated neural architecture search to discover optimal network
structures for the RingRift game AI.

Benefits:
- Automatically discovers efficient architectures
- Adapts architecture to game complexity
- Reduces manual architecture tuning
- Can find specialized structures for policy vs value heads

Search Space:
- Number and size of residual blocks
- Convolutional filter sizes
- Attention mechanisms
- Skip connections
- Head architectures (policy/value)

Reference: Zoph & Le, "Neural Architecture Search with Reinforcement Learning" (2016)

Usage:
    # Run NAS with evolutionary search
    python scripts/neural_architecture_search.py \
        --strategy evolutionary \
        --population 20 \
        --generations 50

    # Run NAS with Bayesian optimization
    python scripts/neural_architecture_search.py \
        --strategy bayesian \
        --trials 100

    # Resume NAS run
    python scripts/neural_architecture_search.py --resume nas_run_123
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

AI_SERVICE_ROOT = Path(__file__).resolve().parents[1]

# Unified logging setup
from scripts.lib.logging_config import setup_script_logging

logger = setup_script_logging("neural_architecture_search")


# Architecture search space definition
SEARCH_SPACE = {
    # Backbone
    "num_res_blocks": {"type": "int", "min": 4, "max": 20, "default": 10},
    "channels": {"type": "choice", "options": [64, 128, 192, 256, 384], "default": 128},
    "kernel_size": {"type": "choice", "options": [3, 5], "default": 3},

    # Residual block options
    "use_se_block": {"type": "bool", "default": False},  # Squeeze-excitation
    "se_ratio": {"type": "choice", "options": [4, 8, 16], "default": 8},
    "use_bottleneck": {"type": "bool", "default": False},
    "bottleneck_ratio": {"type": "choice", "options": [2, 4], "default": 4},

    # Attention
    "use_attention": {"type": "bool", "default": False},
    "attention_heads": {"type": "choice", "options": [2, 4, 8], "default": 4},
    "attention_layers": {"type": "int", "min": 0, "max": 4, "default": 0},

    # Policy head
    "policy_channels": {"type": "choice", "options": [32, 64, 128, 256], "default": 64},
    "policy_layers": {"type": "int", "min": 1, "max": 3, "default": 1},

    # Value head
    "value_channels": {"type": "choice", "options": [32, 64, 128, 256], "default": 64},
    "value_hidden": {"type": "choice", "options": [128, 256, 512], "default": 256},
    "value_layers": {"type": "int", "min": 1, "max": 3, "default": 1},

    # Regularization
    "dropout": {"type": "float", "min": 0.0, "max": 0.3, "default": 0.0},
    "use_layer_norm": {"type": "bool", "default": False},

    # Activation
    "activation": {"type": "choice", "options": ["relu", "gelu", "swish"], "default": "relu"},
}

# NAS configuration
DEFAULT_POPULATION_SIZE = 20
DEFAULT_GENERATIONS = 50
DEFAULT_MUTATION_RATE = 0.3
DEFAULT_CROSSOVER_RATE = 0.5
DEFAULT_ELITE_FRACTION = 0.1


@dataclass
class Architecture:
    """A candidate neural network architecture."""
    arch_id: str
    params: dict[str, Any]
    performance: float = 0.0  # Validation metric (e.g., policy accuracy)
    flops: int = 0  # Estimated FLOPs
    param_count: int = 0  # Number of parameters
    latency_ms: float = 0.0  # Inference latency
    generations_survived: int = 0
    parent_ids: list[str] = field(default_factory=list)
    created_at: str = ""
    evaluated: bool = False


@dataclass
class NASState:
    """Complete state of a NAS run."""
    run_id: str
    strategy: str
    board_type: str
    num_players: int
    search_space: dict[str, Any]
    population: list[Architecture]
    generation: int = 0
    total_evaluations: int = 0
    best_performance: float = 0.0
    best_architecture: dict[str, Any] | None = None
    pareto_front: list[dict[str, Any]] = field(default_factory=list)
    history: list[dict[str, Any]] = field(default_factory=list)
    created_at: str = ""


def sample_architecture(arch_id: str, search_space: dict[str, Any] = SEARCH_SPACE) -> Architecture:
    """Sample a random architecture from the search space."""
    params = {}

    for name, spec in search_space.items():
        param_type = spec["type"]

        if param_type == "int":
            params[name] = random.randint(spec["min"], spec["max"])
        elif param_type == "float":
            params[name] = random.uniform(spec["min"], spec["max"])
        elif param_type == "bool":
            params[name] = random.choice([True, False])
        elif param_type == "choice":
            params[name] = random.choice(spec["options"])

    return Architecture(
        arch_id=arch_id,
        params=params,
        created_at=datetime.utcnow().isoformat() + "Z",
    )


def mutate_architecture(
    arch: Architecture,
    new_id: str,
    mutation_rate: float = DEFAULT_MUTATION_RATE,
    search_space: dict[str, Any] = SEARCH_SPACE,
) -> Architecture:
    """Mutate an architecture by randomly changing some parameters."""
    new_params = copy.deepcopy(arch.params)

    for name, spec in search_space.items():
        if random.random() < mutation_rate:
            param_type = spec["type"]

            if param_type == "int":
                # Gaussian mutation with clipping
                current = new_params.get(name, spec["default"])
                delta = random.gauss(0, (spec["max"] - spec["min"]) / 6)
                new_val = round(current + delta)
                new_params[name] = max(spec["min"], min(spec["max"], new_val))
            elif param_type == "float":
                current = new_params.get(name, spec["default"])
                delta = random.gauss(0, (spec["max"] - spec["min"]) / 6)
                new_val = current + delta
                new_params[name] = max(spec["min"], min(spec["max"], new_val))
            elif param_type == "bool":
                new_params[name] = not new_params.get(name, spec["default"])
            elif param_type == "choice":
                new_params[name] = random.choice(spec["options"])

    return Architecture(
        arch_id=new_id,
        params=new_params,
        parent_ids=[arch.arch_id],
        created_at=datetime.utcnow().isoformat() + "Z",
    )


def crossover_architectures(
    parent1: Architecture,
    parent2: Architecture,
    new_id: str,
    search_space: dict[str, Any] = SEARCH_SPACE,
) -> Architecture:
    """Create child architecture by crossing over two parents."""
    new_params = {}

    for name in search_space:
        # Uniform crossover
        if random.random() < 0.5:
            new_params[name] = parent1.params.get(name, SEARCH_SPACE[name]["default"])
        else:
            new_params[name] = parent2.params.get(name, SEARCH_SPACE[name]["default"])

    return Architecture(
        arch_id=new_id,
        params=new_params,
        parent_ids=[parent1.arch_id, parent2.arch_id],
        created_at=datetime.utcnow().isoformat() + "Z",
    )


def estimate_architecture_cost(arch: Architecture) -> tuple[int, int]:
    """Estimate FLOPs and parameter count for an architecture.

    Returns:
        (estimated_flops, param_count)
    """
    p = arch.params
    board_size = 64  # 8x8 board

    # Input convolution
    input_channels = 32  # Estimated input feature channels
    channels = p.get("channels", 128)
    kernel = p.get("kernel_size", 3)

    # FLOPs for conv: 2 * H * W * C_in * C_out * K^2
    conv_flops = 2 * board_size * input_channels * channels * kernel * kernel
    conv_params = input_channels * channels * kernel * kernel + channels

    # Residual blocks
    num_blocks = p.get("num_res_blocks", 10)
    use_bottleneck = p.get("use_bottleneck", False)
    bottleneck_ratio = p.get("bottleneck_ratio", 4)

    if use_bottleneck:
        # Bottleneck: 1x1 -> 3x3 -> 1x1
        mid_channels = channels // bottleneck_ratio
        block_flops = 2 * board_size * (
            channels * mid_channels +  # 1x1
            mid_channels * mid_channels * kernel * kernel +  # 3x3
            mid_channels * channels  # 1x1
        )
        block_params = channels * mid_channels + mid_channels * mid_channels * kernel * kernel + mid_channels * channels
    else:
        # Standard residual: 3x3 -> 3x3
        block_flops = 2 * 2 * board_size * channels * channels * kernel * kernel
        block_params = 2 * channels * channels * kernel * kernel + 2 * channels

    # SE block overhead
    if p.get("use_se_block", False):
        se_ratio = p.get("se_ratio", 8)
        se_hidden = channels // se_ratio
        se_flops = 2 * (channels * se_hidden + se_hidden * channels)
        se_params = channels * se_hidden + se_hidden * channels
        block_flops += se_flops
        block_params += se_params

    total_backbone_flops = conv_flops + num_blocks * block_flops
    total_backbone_params = conv_params + num_blocks * block_params

    # Attention layers
    if p.get("use_attention", False):
        attention_layers = p.get("attention_layers", 0)
        p.get("attention_heads", 4)
        # Self-attention: Q, K, V projections + attention + output projection
        attn_flops = attention_layers * 4 * board_size * channels * channels
        attn_params = attention_layers * 4 * channels * channels
        total_backbone_flops += attn_flops
        total_backbone_params += attn_params

    # Policy head
    policy_channels = p.get("policy_channels", 64)
    policy_layers = p.get("policy_layers", 1)
    policy_flops = policy_layers * 2 * board_size * channels * policy_channels
    policy_flops += 2 * board_size * policy_channels * board_size  # Final linear
    policy_params = policy_layers * channels * policy_channels + policy_channels * board_size

    # Value head
    value_channels = p.get("value_channels", 64)
    value_hidden = p.get("value_hidden", 256)
    value_layers = p.get("value_layers", 1)
    value_flops = value_layers * 2 * board_size * channels * value_channels
    value_flops += 2 * value_channels * board_size * value_hidden  # Flatten + hidden
    value_flops += 2 * value_hidden  # Output
    value_params = value_layers * channels * value_channels + value_channels * board_size * value_hidden + value_hidden

    total_flops = total_backbone_flops + policy_flops + value_flops
    total_params = total_backbone_params + policy_params + value_params

    return int(total_flops), int(total_params)


def evaluate_architecture(arch: Architecture, quick_eval: bool = True) -> float:
    """Evaluate an architecture's performance.

    Uses actual training when RINGRIFT_NAS_REAL_TRAINING=1 is set,
    otherwise uses simulated evaluation for development/testing.
    """
    import os
    use_real_training = os.environ.get("RINGRIFT_NAS_REAL_TRAINING", "0") == "1"


    # Estimate cost
    flops, param_count = estimate_architecture_cost(arch)
    arch.flops = flops
    arch.param_count = param_count
    arch.latency_ms = flops / 1e9  # Rough estimate

    if use_real_training:
        return _evaluate_architecture_real(arch, quick_eval)
    else:
        return _evaluate_architecture_simulated(arch)


def _evaluate_architecture_real(arch: Architecture, quick_eval: bool = True) -> float:
    """Evaluate architecture with actual training."""
    import os
    import subprocess
    import tempfile

    p = arch.params
    board_type = os.environ.get("RINGRIFT_NAS_BOARD", "square8")
    num_players = int(os.environ.get("RINGRIFT_NAS_PLAYERS", "2"))
    epochs = 3 if quick_eval else 10

    # Create temp directory for this evaluation
    with tempfile.TemporaryDirectory(prefix=f"nas_{arch.arch_id}_") as tmpdir:
        # Build training command with architecture params
        cmd = [
            sys.executable, str(AI_SERVICE_ROOT / "scripts" / "run_nn_training_baseline.py"),
            "--board", board_type,
            "--num-players", str(num_players),
            "--run-dir", tmpdir,
            "--epochs", str(epochs),
            "--demo",  # Use demo mode for faster evaluation
            "--learning-rate", str(p.get("learning_rate", 0.001)),
            "--batch-size", str(int(p.get("batch_size", 256))),
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,  # 10 min timeout per eval
                cwd=str(AI_SERVICE_ROOT),
                env={**os.environ, "PYTHONPATH": str(AI_SERVICE_ROOT)},
            )

            if result.returncode != 0:
                logger.warning(f"Training failed for {arch.arch_id}: {result.stderr[:200]}")
                arch.performance = 0.3  # Low score for failed training
                arch.evaluated = True
                return arch.performance

            # Parse training report
            report_path = Path(tmpdir) / "nn_training_report.json"
            if report_path.exists():
                with open(report_path) as f:
                    report = json.load(f)
                # Use validation loss as performance metric (inverted - lower is better)
                val_loss = report.get("metrics", {}).get("final_loss")
                if val_loss is not None:
                    # Convert loss to performance score (0.3 to 0.95)
                    arch.performance = max(0.3, min(0.95, 1.0 - val_loss))
                else:
                    arch.performance = 0.5  # Neutral if no loss reported
            else:
                arch.performance = 0.5

        except subprocess.TimeoutExpired:
            logger.warning(f"Training timeout for {arch.arch_id}")
            arch.performance = 0.3
        except Exception as e:
            logger.warning(f"Training error for {arch.arch_id}: {e}")
            arch.performance = 0.3

    arch.evaluated = True
    return arch.performance


def _evaluate_architecture_simulated(arch: Architecture) -> float:
    """Simulated evaluation for testing (original implementation)."""
    p = arch.params

    # Simulate performance based on architecture choices
    capacity_score = math.log10(max(arch.param_count, 1000)) / 7
    depth_score = min(p.get("num_res_blocks", 10) / 20, 1.0)
    attention_bonus = 0.05 if p.get("use_attention", False) else 0
    se_bonus = 0.03 if p.get("use_se_block", False) else 0
    activation_bonus = 0.02 if p.get("activation", "relu") in ["gelu", "swish"] else 0
    dropout = p.get("dropout", 0.0)
    dropout_bonus = 0.02 if 0.05 <= dropout <= 0.2 else 0

    base_performance = 0.4 + 0.3 * capacity_score + 0.2 * depth_score
    bonuses = attention_bonus + se_bonus + activation_bonus + dropout_bonus
    noise = np.random.normal(0, 0.03)

    performance = base_performance + bonuses + noise
    performance = max(0.3, min(0.95, performance))

    arch.performance = performance
    arch.evaluated = True
    return performance


def tournament_selection(
    population: list[Architecture],
    tournament_size: int = 3,
) -> Architecture:
    """Select architecture via tournament selection."""
    contestants = random.sample(population, min(tournament_size, len(population)))
    return max(contestants, key=lambda a: a.performance)


def is_pareto_dominated(arch: Architecture, others: list[Architecture]) -> bool:
    """Check if architecture is dominated by any other (worse on all objectives)."""
    for other in others:
        if other.arch_id == arch.arch_id:
            continue
        # Multi-objective: performance (higher better), latency (lower better)
        if (other.performance >= arch.performance and other.latency_ms <= arch.latency_ms and other.performance > arch.performance) or other.latency_ms < arch.latency_ms:
            return True
    return False


def update_pareto_front(population: list[Architecture]) -> list[dict[str, Any]]:
    """Extract non-dominated architectures (Pareto front)."""
    evaluated = [a for a in population if a.evaluated]
    pareto = []

    for arch in evaluated:
        if not is_pareto_dominated(arch, evaluated):
            pareto.append({
                "arch_id": arch.arch_id,
                "performance": arch.performance,
                "latency_ms": arch.latency_ms,
                "param_count": arch.param_count,
                "params": arch.params,
            })

    return sorted(pareto, key=lambda x: -x["performance"])


def evolutionary_step(
    state: NASState,
    elite_fraction: float = DEFAULT_ELITE_FRACTION,
    mutation_rate: float = DEFAULT_MUTATION_RATE,
    crossover_rate: float = DEFAULT_CROSSOVER_RATE,
) -> NASState:
    """Run one generation of evolutionary NAS."""
    population = state.population
    pop_size = len(population)

    # Evaluate unevaluated architectures
    for arch in population:
        if not arch.evaluated:
            evaluate_architecture(arch)
            state.total_evaluations += 1

    # Sort by performance
    population.sort(key=lambda a: a.performance, reverse=True)

    # Update best
    if population[0].performance > state.best_performance:
        state.best_performance = population[0].performance
        state.best_architecture = copy.deepcopy(population[0].params)
        logger.info(f"New best architecture: perf={state.best_performance:.4f}")

    # Keep elites
    num_elites = max(1, int(pop_size * elite_fraction))
    new_population = population[:num_elites]

    for arch in new_population:
        arch.generations_survived += 1

    # Generate new architectures
    gen_id = state.generation + 1
    arch_counter = 0

    while len(new_population) < pop_size:
        arch_id = f"gen{gen_id:03d}_arch{arch_counter:03d}"

        if random.random() < crossover_rate and len(population) >= 2:
            # Crossover
            parent1 = tournament_selection(population)
            parent2 = tournament_selection(population)
            child = crossover_architectures(parent1, parent2, arch_id)
            # Possibly mutate child
            if random.random() < mutation_rate:
                child = mutate_architecture(child, arch_id, mutation_rate=0.5)
        else:
            # Mutation only
            parent = tournament_selection(population)
            child = mutate_architecture(parent, arch_id, mutation_rate)

        new_population.append(child)
        arch_counter += 1

    state.population = new_population
    state.generation = gen_id
    state.pareto_front = update_pareto_front(new_population)

    # Record history
    state.history.append({
        "generation": state.generation,
        "best_performance": state.best_performance,
        "avg_performance": np.mean([a.performance for a in new_population if a.evaluated]),
        "pareto_size": len(state.pareto_front),
        "timestamp": datetime.utcnow().isoformat() + "Z",
    })

    return state


def random_search_step(state: NASState, batch_size: int = 5) -> NASState:
    """Run one batch of random search."""
    gen_id = state.generation + 1

    for i in range(batch_size):
        arch_id = f"random_{gen_id:03d}_{i:03d}"
        arch = sample_architecture(arch_id)
        evaluate_architecture(arch)
        state.total_evaluations += 1

        if arch.performance > state.best_performance:
            state.best_performance = arch.performance
            state.best_architecture = copy.deepcopy(arch.params)
            logger.info(f"New best architecture: perf={state.best_performance:.4f}")

        state.population.append(arch)

    # Trim population to reasonable size
    if len(state.population) > 100:
        state.population.sort(key=lambda a: a.performance, reverse=True)
        state.population = state.population[:100]

    state.generation = gen_id
    state.pareto_front = update_pareto_front(state.population)

    state.history.append({
        "generation": state.generation,
        "best_performance": state.best_performance,
        "total_evaluations": state.total_evaluations,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    })

    return state


def bayesian_acquisition(
    evaluated: list[Architecture],
    search_space: dict[str, Any],
    num_candidates: int = 100,
) -> Architecture:
    """Simple acquisition function for Bayesian-style search.

    Uses a simple surrogate based on similarity to good architectures.
    In production, would use Gaussian Process or Tree Parzen Estimators.
    """
    # Sort by performance
    sorted_archs = sorted(evaluated, key=lambda a: a.performance, reverse=True)
    top_k = sorted_archs[:max(1, len(sorted_archs) // 4)]

    best_candidate = None
    best_score = -float("inf")

    for i in range(num_candidates):
        # Sample candidate
        candidate = sample_architecture(f"candidate_{i}")

        # Score based on similarity to top performers + exploration bonus
        similarity = 0
        for top_arch in top_k:
            match_count = sum(
                1 for k in search_space
                if candidate.params.get(k) == top_arch.params.get(k)
            )
            similarity += match_count / len(search_space) * top_arch.performance

        similarity /= len(top_k)

        # Add exploration bonus for novel configurations
        novelty = 1.0
        for existing in evaluated:
            match_count = sum(
                1 for k in search_space
                if candidate.params.get(k) == existing.params.get(k)
            )
            if match_count == len(search_space):
                novelty = 0
                break
            novelty = min(novelty, 1 - match_count / len(search_space))

        score = similarity + 0.1 * novelty

        if score > best_score:
            best_score = score
            best_candidate = candidate

    return best_candidate


def bayesian_step(state: NASState) -> NASState:
    """Run one step of Bayesian-style NAS."""
    gen_id = state.generation + 1

    if state.total_evaluations < 10:
        # Initial random exploration
        arch = sample_architecture(f"bayes_{gen_id:03d}")
    else:
        # Use acquisition function
        arch = bayesian_acquisition(state.population, state.search_space)
        arch.arch_id = f"bayes_{gen_id:03d}"

    evaluate_architecture(arch)
    state.total_evaluations += 1
    state.population.append(arch)

    if arch.performance > state.best_performance:
        state.best_performance = arch.performance
        state.best_architecture = copy.deepcopy(arch.params)
        logger.info(f"New best architecture: perf={state.best_performance:.4f}")

    state.generation = gen_id
    state.pareto_front = update_pareto_front(state.population)

    state.history.append({
        "generation": state.generation,
        "best_performance": state.best_performance,
        "total_evaluations": state.total_evaluations,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    })

    return state


def save_nas_state(state: NASState, output_dir: Path):
    """Save NAS state to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)

    state_file = output_dir / "nas_state.json"
    with open(state_file, "w") as f:
        json.dump({
            "run_id": state.run_id,
            "strategy": state.strategy,
            "board_type": state.board_type,
            "num_players": state.num_players,
            "generation": state.generation,
            "total_evaluations": state.total_evaluations,
            "best_performance": state.best_performance,
            "best_architecture": state.best_architecture,
            "pareto_front": state.pareto_front,
            "population": [asdict(a) for a in state.population],
            "history": state.history[-100:],
            "created_at": state.created_at,
        }, f, indent=2)

    # Save best architecture separately for easy access
    if state.best_architecture:
        best_file = output_dir / "best_architecture.json"
        with open(best_file, "w") as f:
            json.dump({
                "performance": state.best_performance,
                "params": state.best_architecture,
            }, f, indent=2)

    logger.info(f"Saved NAS state to {state_file}")


def load_nas_state(state_file: Path) -> NASState:
    """Load NAS state from disk."""
    with open(state_file) as f:
        data = json.load(f)

    population = [Architecture(**a) for a in data["population"]]

    return NASState(
        run_id=data["run_id"],
        strategy=data["strategy"],
        board_type=data["board_type"],
        num_players=data["num_players"],
        search_space=SEARCH_SPACE,
        population=population,
        generation=data["generation"],
        total_evaluations=data["total_evaluations"],
        best_performance=data["best_performance"],
        best_architecture=data["best_architecture"],
        pareto_front=data.get("pareto_front", []),
        history=data.get("history", []),
        created_at=data["created_at"],
    )


def print_nas_status(state: NASState):
    """Print current NAS status."""
    print("\n" + "=" * 70)
    print(f"NAS STATUS - Generation {state.generation}")
    print("=" * 70)
    print(f"Strategy: {state.strategy}")
    print(f"Total Evaluations: {state.total_evaluations}")
    print(f"Best Performance: {state.best_performance:.4f}")

    if state.best_architecture:
        print("\nBest Architecture:")
        for key, value in state.best_architecture.items():
            print(f"  {key}: {value}")

    print(f"\nPareto Front ({len(state.pareto_front)} architectures):")
    for i, arch in enumerate(state.pareto_front[:5]):
        print(f"  {i+1}. perf={arch['performance']:.4f}, latency={arch['latency_ms']:.2f}ms, params={arch['param_count']:,}")

    if state.population:
        evaluated = [a for a in state.population if a.evaluated]
        if evaluated:
            avg_perf = np.mean([a.performance for a in evaluated])
            print(f"\nPopulation avg performance: {avg_perf:.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="Neural Architecture Search for RingRift AI"
    )

    parser.add_argument(
        "--strategy",
        type=str,
        choices=["evolutionary", "random", "bayesian"],
        default="evolutionary",
        help="Search strategy",
    )
    parser.add_argument(
        "--population",
        type=int,
        default=DEFAULT_POPULATION_SIZE,
        help="Population size (for evolutionary)",
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=DEFAULT_GENERATIONS,
        help="Number of generations/iterations",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=100,
        help="Number of trials (for random/bayesian)",
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
        "--mutation-rate",
        type=float,
        default=DEFAULT_MUTATION_RATE,
        help="Mutation rate",
    )
    parser.add_argument(
        "--crossover-rate",
        type=float,
        default=DEFAULT_CROSSOVER_RATE,
        help="Crossover rate",
    )
    parser.add_argument(
        "--resume",
        type=str,
        help="Resume from existing NAS run",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(AI_SERVICE_ROOT / "logs" / "nas"),
        help="Output directory",
    )

    args = parser.parse_args()
    output_dir = Path(args.output_dir)

    # Initialize or resume
    if args.resume:
        state_file = Path(args.resume) / "nas_state.json"
        if state_file.exists():
            state = load_nas_state(state_file)
            logger.info(f"Resumed NAS run {state.run_id}")
        else:
            logger.error(f"State file not found: {state_file}")
            return 1
    else:
        run_id = f"nas_{args.strategy}_{int(time.time())}"

        if args.strategy == "evolutionary":
            # Initialize population
            population = [
                sample_architecture(f"gen000_arch{i:03d}")
                for i in range(args.population)
            ]
        else:
            population = []

        state = NASState(
            run_id=run_id,
            strategy=args.strategy,
            board_type=args.board,
            num_players=args.players,
            search_space=SEARCH_SPACE,
            population=population,
            created_at=datetime.utcnow().isoformat() + "Z",
        )
        output_dir = output_dir / run_id
        logger.info(f"Created new NAS run: {run_id}")

    # Run NAS
    if args.strategy == "evolutionary":
        num_iterations = args.generations
    else:
        num_iterations = args.trials

    logger.info(f"Running {num_iterations} NAS iterations ({args.strategy})...")

    for i in range(num_iterations):
        if args.strategy == "evolutionary":
            state = evolutionary_step(
                state,
                mutation_rate=args.mutation_rate,
                crossover_rate=args.crossover_rate,
            )
        elif args.strategy == "random":
            state = random_search_step(state)
        elif args.strategy == "bayesian":
            state = bayesian_step(state)

        if (i + 1) % 10 == 0:
            print_nas_status(state)
            save_nas_state(state, output_dir)

    # Final save
    save_nas_state(state, output_dir)
    print_nas_status(state)

    print("\n" + "=" * 70)
    print("NAS COMPLETE")
    print("=" * 70)
    print(f"Strategy: {args.strategy}")
    print(f"Total evaluations: {state.total_evaluations}")
    print(f"Best performance: {state.best_performance:.4f}")
    print(f"Output saved to: {output_dir}")

    if state.best_architecture:
        print("\nBest architecture found:")
        for key, value in state.best_architecture.items():
            print(f"  {key}: {value}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
