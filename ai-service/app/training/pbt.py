"""Population-Based Training (PBT) for RingRift AI.

PBT is a hyperparameter optimization technique that trains multiple "agents"
in parallel, periodically replacing poor performers with copies of top
performers and mutating their hyperparameters.

Key advantages:
1. Discovers good hyperparameter schedules (not just fixed values)
2. More efficient than grid/random search
3. Can adapt to changing training dynamics

Usage:
    from app.training.pbt import PBTController, PBTWorker, PBTConfig

    config = PBTConfig(
        population_size=8,
        exploit_interval_steps=5000,
        exploit_fraction=0.2,
    )

    controller = PBTController(config)

    # Add workers with different initial hyperparameters
    for i in range(config.population_size):
        controller.add_worker(PBTWorker(
            worker_id=i,
            hyperparameters=controller.sample_initial_hyperparameters(),
        ))

    # Training loop
    for step in range(total_steps):
        for worker in controller.workers:
            worker.train_step(...)

        if step % config.exploit_interval_steps == 0:
            controller.exploit_and_explore()
"""

from __future__ import annotations

import copy
import json
import logging
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class HyperparameterSpec:
    """Specification for a hyperparameter to be optimized.

    Defines the range, scale, and mutation behavior for a hyperparameter.
    """
    name: str
    min_value: float
    max_value: float
    scale: str = "linear"  # "linear" or "log"
    mutation_strength: float = 0.2  # Fraction of range to perturb
    discrete: bool = False  # If True, round to integer


@dataclass
class PBTConfig:
    """Configuration for Population-Based Training."""
    population_size: int = 8  # Number of parallel workers
    exploit_interval_steps: int = 5000  # Steps between exploit/explore
    exploit_fraction: float = 0.2  # Bottom 20% are replaced
    explore_mutation_prob: float = 0.8  # Prob of mutating each hyperparam
    explore_resample_prob: float = 0.2  # Prob of resampling from scratch
    performance_metric: str = "elo"  # "elo", "loss", "win_rate"
    higher_is_better: bool = True  # True for Elo, False for loss
    checkpoint_dir: str = "data/pbt_checkpoints"
    # Hyperparameter specifications
    hyperparameter_specs: list[HyperparameterSpec] = field(default_factory=list)


@dataclass
class WorkerState:
    """State of a PBT worker."""
    worker_id: int
    hyperparameters: dict[str, float]
    performance: float = 0.0
    steps: int = 0
    generation: int = 0  # How many times this worker has been replaced
    parent_id: int | None = None  # Worker this was copied from
    checkpoint_path: str | None = None


class PBTWorker:
    """A single worker in the PBT population.

    Each worker has its own set of hyperparameters and trains independently.
    Periodically, workers are evaluated and poor performers are replaced.
    """

    def __init__(
        self,
        worker_id: int,
        hyperparameters: dict[str, float],
        checkpoint_dir: Path | None = None,
    ):
        """Initialize a PBT worker.

        Args:
            worker_id: Unique identifier for this worker
            hyperparameters: Initial hyperparameters
            checkpoint_dir: Directory for saving checkpoints
        """
        self.worker_id = worker_id
        self.hyperparameters = hyperparameters.copy()
        self.performance = 0.0
        self.steps = 0
        self.generation = 0
        self.parent_id: int | None = None
        self.checkpoint_dir = checkpoint_dir or Path("data/pbt_checkpoints")

        # Model state (to be set externally)
        self.model_state_dict: dict | None = None
        self.optimizer_state_dict: dict | None = None

    def update_performance(self, performance: float):
        """Update the worker's performance metric."""
        self.performance = performance

    def increment_steps(self, steps: int = 1):
        """Increment the step counter."""
        self.steps += steps

    def get_state(self) -> WorkerState:
        """Get current worker state."""
        return WorkerState(
            worker_id=self.worker_id,
            hyperparameters=self.hyperparameters.copy(),
            performance=self.performance,
            steps=self.steps,
            generation=self.generation,
            parent_id=self.parent_id,
            checkpoint_path=str(self.get_checkpoint_path()),
        )

    def get_checkpoint_path(self) -> Path:
        """Get path for worker checkpoint."""
        return self.checkpoint_dir / f"worker_{self.worker_id}_gen{self.generation}.pt"

    def save_checkpoint(self) -> Path:
        """Save worker checkpoint (model + optimizer + hyperparameters)."""
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = self.get_checkpoint_path()

        checkpoint = {
            "worker_id": self.worker_id,
            "hyperparameters": self.hyperparameters,
            "performance": self.performance,
            "steps": self.steps,
            "generation": self.generation,
            "parent_id": self.parent_id,
            "model_state_dict": self.model_state_dict,
            "optimizer_state_dict": self.optimizer_state_dict,
        }

        # Use torch if available, otherwise pickle
        try:
            import torch
            torch.save(checkpoint, checkpoint_path)
        except ImportError:
            import pickle
            with open(checkpoint_path, "wb") as f:
                pickle.dump(checkpoint, f)

        logger.info(f"[PBT] Worker {self.worker_id} checkpoint saved to {checkpoint_path}")
        return checkpoint_path

    def load_checkpoint(self, checkpoint_path: Path) -> bool:
        """Load worker checkpoint."""
        if not checkpoint_path.exists():
            return False

        try:
            try:
                import torch
                checkpoint = torch.load(checkpoint_path, map_location="cpu")
            except ImportError:
                import pickle
                with open(checkpoint_path, "rb") as f:
                    checkpoint = pickle.load(f)

            self.hyperparameters = checkpoint["hyperparameters"]
            self.performance = checkpoint["performance"]
            self.steps = checkpoint["steps"]
            self.generation = checkpoint["generation"]
            self.parent_id = checkpoint.get("parent_id")
            self.model_state_dict = checkpoint.get("model_state_dict")
            self.optimizer_state_dict = checkpoint.get("optimizer_state_dict")

            logger.info(f"[PBT] Worker {self.worker_id} loaded from {checkpoint_path}")
            return True

        except Exception as e:
            logger.error(f"[PBT] Failed to load checkpoint: {e}")
            return False

    def copy_from(self, other: PBTWorker):
        """Copy model weights and hyperparameters from another worker."""
        self.hyperparameters = other.hyperparameters.copy()
        self.model_state_dict = copy.deepcopy(other.model_state_dict)
        self.optimizer_state_dict = copy.deepcopy(other.optimizer_state_dict)
        self.parent_id = other.worker_id
        self.generation += 1
        # Don't copy performance - it needs to be re-evaluated
        self.performance = 0.0


class PBTController:
    """Controller for Population-Based Training.

    Manages the population of workers, handles exploit/explore cycles,
    and tracks the training history.
    """

    def __init__(self, config: PBTConfig):
        """Initialize the PBT controller.

        Args:
            config: PBT configuration
        """
        self.config = config
        self.workers: dict[int, PBTWorker] = {}
        self.history: list[dict[str, Any]] = []  # History of exploit/explore events
        self.step_count = 0
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Set up default hyperparameter specs if none provided
        if not config.hyperparameter_specs:
            self._setup_default_specs()

    def _setup_default_specs(self):
        """Set up default hyperparameter specifications for training."""
        self.config.hyperparameter_specs = [
            HyperparameterSpec(
                name="learning_rate",
                min_value=1e-5,
                max_value=1e-2,
                scale="log",
                mutation_strength=0.2,
            ),
            HyperparameterSpec(
                name="weight_decay",
                min_value=1e-6,
                max_value=1e-2,
                scale="log",
                mutation_strength=0.2,
            ),
            HyperparameterSpec(
                name="batch_size",
                min_value=32,
                max_value=512,
                scale="log",
                mutation_strength=0.2,
                discrete=True,
            ),
            HyperparameterSpec(
                name="value_loss_weight",
                min_value=0.1,
                max_value=2.0,
                scale="linear",
                mutation_strength=0.2,
            ),
            HyperparameterSpec(
                name="policy_loss_weight",
                min_value=0.1,
                max_value=2.0,
                scale="linear",
                mutation_strength=0.2,
            ),
            HyperparameterSpec(
                name="gradient_clip",
                min_value=0.1,
                max_value=10.0,
                scale="log",
                mutation_strength=0.3,
            ),
        ]

    def sample_initial_hyperparameters(self) -> dict[str, float]:
        """Sample initial hyperparameters for a new worker."""
        hp = {}
        for spec in self.config.hyperparameter_specs:
            if spec.scale == "log":
                value = np.exp(np.random.uniform(
                    np.log(spec.min_value),
                    np.log(spec.max_value)
                ))
            else:
                value = np.random.uniform(spec.min_value, spec.max_value)

            if spec.discrete:
                value = round(value)

            hp[spec.name] = value

        return hp

    def add_worker(self, worker: PBTWorker):
        """Add a worker to the population."""
        self.workers[worker.worker_id] = worker
        logger.info(f"[PBT] Added worker {worker.worker_id} with HP: {worker.hyperparameters}")

    def remove_worker(self, worker_id: int):
        """Remove a worker from the population."""
        if worker_id in self.workers:
            del self.workers[worker_id]

    def get_worker(self, worker_id: int) -> PBTWorker | None:
        """Get a worker by ID."""
        return self.workers.get(worker_id)

    def get_ranked_workers(self) -> list[PBTWorker]:
        """Get workers sorted by performance (best first)."""
        workers = list(self.workers.values())
        reverse = self.config.higher_is_better
        return sorted(workers, key=lambda w: w.performance, reverse=reverse)

    def exploit_and_explore(self):
        """Perform exploit/explore cycle.

        1. Rank workers by performance
        2. Bottom workers copy weights from top workers (exploit)
        3. Mutate hyperparameters of copied workers (explore)
        """
        workers = self.get_ranked_workers()
        n_workers = len(workers)

        if n_workers < 2:
            return

        # Calculate how many workers to replace
        n_replace = max(1, int(n_workers * self.config.exploit_fraction))

        # Top performers to copy from
        top_workers = workers[:n_replace]
        # Bottom performers to replace
        bottom_workers = workers[-n_replace:]

        replacements = []
        for bottom, top in zip(bottom_workers, top_workers, strict=False):
            if bottom.worker_id == top.worker_id:
                continue

            logger.info(
                f"[PBT] Worker {bottom.worker_id} (perf={bottom.performance:.4f}) "
                f"copying from {top.worker_id} (perf={top.performance:.4f})"
            )

            # Exploit: copy weights and hyperparameters
            bottom.copy_from(top)

            # Explore: mutate hyperparameters
            mutated_hp = self._mutate_hyperparameters(bottom.hyperparameters)
            bottom.hyperparameters = mutated_hp

            logger.info(f"[PBT] Worker {bottom.worker_id} new HP: {mutated_hp}")

            replacements.append({
                "bottom_id": bottom.worker_id,
                "top_id": top.worker_id,
                "old_performance": bottom.performance,
                "top_performance": top.performance,
                "new_hyperparameters": mutated_hp.copy(),
            })

        # Record history
        self.history.append({
            "step": self.step_count,
            "timestamp": time.time(),
            "replacements": replacements,
            "rankings": [
                {"id": w.worker_id, "perf": w.performance, "hp": w.hyperparameters}
                for w in workers
            ],
        })

        # Save history
        self._save_history()

    def _mutate_hyperparameters(self, hp: dict[str, float]) -> dict[str, float]:
        """Mutate hyperparameters using explore strategy."""
        mutated = hp.copy()

        for spec in self.config.hyperparameter_specs:
            if spec.name not in mutated:
                continue

            # Decide whether to mutate this hyperparameter
            if random.random() > self.config.explore_mutation_prob:
                continue

            # Decide whether to resample from scratch or perturb
            if random.random() < self.config.explore_resample_prob:
                # Resample from scratch
                if spec.scale == "log":
                    value = np.exp(np.random.uniform(
                        np.log(spec.min_value),
                        np.log(spec.max_value)
                    ))
                else:
                    value = np.random.uniform(spec.min_value, spec.max_value)
            else:
                # Perturb existing value
                current = mutated[spec.name]

                if spec.scale == "log":
                    # Perturb in log space
                    log_current = np.log(current)
                    log_range = np.log(spec.max_value) - np.log(spec.min_value)
                    perturbation = np.random.normal(0, spec.mutation_strength * log_range)
                    value = np.exp(log_current + perturbation)
                else:
                    # Perturb in linear space
                    range_size = spec.max_value - spec.min_value
                    perturbation = np.random.normal(0, spec.mutation_strength * range_size)
                    value = current + perturbation

            # Clip to valid range
            value = np.clip(value, spec.min_value, spec.max_value)

            if spec.discrete:
                value = round(value)

            mutated[spec.name] = value

        return mutated

    def should_exploit_explore(self, step: int) -> bool:
        """Check if it's time for an exploit/explore cycle."""
        return step > 0 and step % self.config.exploit_interval_steps == 0

    def update_step(self, step: int):
        """Update the step count and trigger exploit/explore if needed."""
        self.step_count = step
        if self.should_exploit_explore(step):
            self.exploit_and_explore()

    def get_best_worker(self) -> PBTWorker | None:
        """Get the best performing worker."""
        ranked = self.get_ranked_workers()
        return ranked[0] if ranked else None

    def get_best_hyperparameters(self) -> dict[str, float] | None:
        """Get hyperparameters of the best worker."""
        best = self.get_best_worker()
        return best.hyperparameters.copy() if best else None

    def get_population_stats(self) -> dict[str, Any]:
        """Get statistics about the current population."""
        workers = list(self.workers.values())
        if not workers:
            return {}

        performances = [w.performance for w in workers]

        return {
            "population_size": len(workers),
            "step_count": self.step_count,
            "mean_performance": np.mean(performances),
            "std_performance": np.std(performances),
            "min_performance": np.min(performances),
            "max_performance": np.max(performances),
            "best_worker_id": self.get_best_worker().worker_id if workers else None,
            "exploit_explore_events": len(self.history),
        }

    def _save_history(self):
        """Save history to disk."""
        history_path = self.checkpoint_dir / "pbt_history.json"
        try:
            with open(history_path, "w") as f:
                json.dump(self.history, f, indent=2, default=str)
        except Exception as e:
            logger.warning(f"[PBT] Failed to save history: {e}")

    def save_state(self) -> Path:
        """Save full PBT state."""
        state_path = self.checkpoint_dir / "pbt_state.json"

        state = {
            "step_count": self.step_count,
            "config": {
                "population_size": self.config.population_size,
                "exploit_interval_steps": self.config.exploit_interval_steps,
                "exploit_fraction": self.config.exploit_fraction,
                "explore_mutation_prob": self.config.explore_mutation_prob,
                "explore_resample_prob": self.config.explore_resample_prob,
                "performance_metric": self.config.performance_metric,
                "higher_is_better": self.config.higher_is_better,
            },
            "workers": [w.get_state().__dict__ for w in self.workers.values()],
        }

        with open(state_path, "w") as f:
            json.dump(state, f, indent=2, default=str)

        # Save worker checkpoints
        for worker in self.workers.values():
            worker.save_checkpoint()

        logger.info(f"[PBT] State saved to {state_path}")
        return state_path

    def load_state(self, state_path: Path | None = None) -> bool:
        """Load PBT state from disk."""
        state_path = state_path or (self.checkpoint_dir / "pbt_state.json")

        if not state_path.exists():
            return False

        try:
            with open(state_path) as f:
                state = json.load(f)

            self.step_count = state["step_count"]

            # Load workers
            self.workers.clear()
            for ws in state["workers"]:
                worker = PBTWorker(
                    worker_id=ws["worker_id"],
                    hyperparameters=ws["hyperparameters"],
                    checkpoint_dir=self.checkpoint_dir,
                )
                worker.performance = ws["performance"]
                worker.steps = ws["steps"]
                worker.generation = ws["generation"]
                worker.parent_id = ws.get("parent_id")

                # Load checkpoint if available
                if ws.get("checkpoint_path"):
                    worker.load_checkpoint(Path(ws["checkpoint_path"]))

                self.workers[worker.worker_id] = worker

            logger.info(f"[PBT] State loaded from {state_path}")
            return True

        except Exception as e:
            logger.error(f"[PBT] Failed to load state: {e}")
            return False


def create_pbt_controller(
    population_size: int = 8,
    exploit_interval_steps: int = 5000,
    checkpoint_dir: str = "data/pbt_checkpoints",
) -> PBTController:
    """Factory function to create a PBT controller.

    Args:
        population_size: Number of workers in population
        exploit_interval_steps: Steps between exploit/explore cycles
        checkpoint_dir: Directory for checkpoints

    Returns:
        Configured PBTController
    """
    config = PBTConfig(
        population_size=population_size,
        exploit_interval_steps=exploit_interval_steps,
        checkpoint_dir=checkpoint_dir,
    )
    return PBTController(config)


def run_pbt_training_example():
    """Example of how to use PBT for training."""
    logger.info("[PBT] Starting example PBT training")

    # Create controller
    controller = create_pbt_controller(
        population_size=4,
        exploit_interval_steps=100,
    )

    # Initialize population
    for i in range(4):
        hp = controller.sample_initial_hyperparameters()
        worker = PBTWorker(
            worker_id=i,
            hyperparameters=hp,
            checkpoint_dir=Path(controller.config.checkpoint_dir),
        )
        controller.add_worker(worker)

    # Simulate training
    for step in range(500):
        # Simulate training step for each worker
        for worker in controller.workers.values():
            # Simulate performance based on hyperparameters
            # In reality this would be actual training and evaluation
            lr = worker.hyperparameters.get("learning_rate", 0.001)
            wd = worker.hyperparameters.get("weight_decay", 0.0001)

            # Optimal is around lr=0.001, wd=0.0001
            optimal_lr = 0.001
            optimal_wd = 0.0001
            lr_penalty = abs(np.log(lr / optimal_lr)) * 10
            wd_penalty = abs(np.log(wd / optimal_wd)) * 5

            performance = 100 - lr_penalty - wd_penalty + np.random.normal(0, 2)
            worker.update_performance(performance)
            worker.increment_steps()

        # Update controller (handles exploit/explore)
        controller.update_step(step)

        if step % 50 == 0:
            stats = controller.get_population_stats()
            logger.info(f"[PBT] Step {step}: mean_perf={stats['mean_performance']:.2f}, "
                       f"max_perf={stats['max_performance']:.2f}")

    # Report results
    best = controller.get_best_worker()
    if best:
        logger.info(f"[PBT] Best worker: {best.worker_id} with HP: {best.hyperparameters}")

    return controller


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_pbt_training_example()
