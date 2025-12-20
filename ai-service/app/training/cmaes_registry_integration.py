"""CMA-ES Model Registry Integration.

Bridges CMA-ES hyperparameter optimization results with the model registry
for tracking, versioning, and promotion of optimized heuristic weights.

Features:
- Register CMA-ES optimization runs as versioned models
- Track hyperparameters (population size, sigma, generations) in TrainingConfig
- Store fitness scores and improvement metrics
- Auto-promote based on fitness improvement thresholds
- Tag models with board type and configuration

Usage:
    from app.training.cmaes_registry_integration import register_cmaes_result

    # After CMA-ES optimization completes:
    model_id, version = register_cmaes_result(
        weights_path=Path("logs/cmaes/optimized_weights.json"),
        board_type="square8",
        num_players=2,
        fitness=0.85,
        generation=50,
        cmaes_config={
            "population_size": 20,
            "sigma": 0.5,
            "games_per_eval": 10,
        },
    )
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.utils.paths import AI_SERVICE_ROOT

logger = logging.getLogger(__name__)
DEFAULT_REGISTRY_DIR = AI_SERVICE_ROOT / "data" / "model_registry"


@dataclass
class CMAESRunConfig:
    """Configuration from a CMA-ES optimization run."""
    population_size: int
    sigma: float
    generations: int
    games_per_eval: int
    board_type: str
    num_players: int
    opponent_mode: str = "baseline-only"
    eval_mode: str = "sequential"
    seed: int | None = None
    run_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "population_size": self.population_size,
            "sigma": self.sigma,
            "generations": self.generations,
            "games_per_eval": self.games_per_eval,
            "board_type": self.board_type,
            "num_players": self.num_players,
            "opponent_mode": self.opponent_mode,
            "eval_mode": self.eval_mode,
            "seed": self.seed,
            "run_id": self.run_id,
        }


def register_cmaes_result(
    weights_path: Path,
    board_type: str,
    num_players: int,
    fitness: float,
    generation: int,
    cmaes_config: dict[str, Any] | None = None,
    registry_dir: Path | None = None,
    auto_promote: bool = True,
    min_fitness_improvement: float = 0.02,
) -> tuple[str, int]:
    """Register a CMA-ES optimization result in the model registry.

    Parameters
    ----------
    weights_path:
        Path to the optimized weights JSON file.
    board_type:
        Board type (e.g., "square8", "hex8").
    num_players:
        Number of players (2, 3, or 4).
    fitness:
        Final fitness score achieved (0.0 to 1.0 scale).
    generation:
        Final generation number.
    cmaes_config:
        CMA-ES hyperparameters (population_size, sigma, etc.).
    registry_dir:
        Path to model registry directory.
    auto_promote:
        If True, automatically promote to staging if fitness improves.
    min_fitness_improvement:
        Minimum fitness improvement required for auto-promotion.

    Returns
    -------
    Tuple[str, int]:
        (model_id, version) of the registered model.
    """
    # Import here to avoid circular imports
    from app.training.model_registry import (
        ModelMetrics,
        ModelRegistry,
        ModelStage,
        ModelType,
        TrainingConfig,
    )

    weights_path = Path(weights_path)
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights file not found: {weights_path}")

    registry_dir = registry_dir or DEFAULT_REGISTRY_DIR
    registry = ModelRegistry(registry_dir)

    # Generate model ID from board config
    config_key = f"{board_type}_{num_players}p"
    model_id = f"heuristic_{config_key}"
    model_name = f"CMA-ES Heuristic ({config_key})"

    # Load weights file to get metadata
    with open(weights_path) as f:
        json.load(f)

    # Build metrics - convert fitness to approximate Elo equivalent
    # Fitness of 0.5 = 50% win rate = 0 Elo diff, higher fitness = higher Elo
    # Using Elo formula: win_prob = 1 / (1 + 10^(-elo_diff/400))
    # Solving for elo_diff: elo_diff = -400 * log10((1/fitness) - 1)
    # For fitness > 0.5, this gives positive Elo
    elo_estimate = None
    if 0.01 < fitness < 0.99:
        import math
        elo_estimate = -400 * math.log10((1.0 / fitness) - 1)

    metrics = ModelMetrics(
        elo=elo_estimate,
        win_rate=fitness,  # CMA-ES fitness is essentially win rate vs baseline
        games_played=cmaes_config.get("games_per_eval", 0) * generation if cmaes_config else 0,
    )

    # Build training config with CMA-ES hyperparameters in extra_config
    extra_config = {
        "optimizer_type": "cmaes",
        "final_fitness": fitness,
        "final_generation": generation,
        "board_type": board_type,
        "num_players": num_players,
    }
    if cmaes_config:
        extra_config.update({
            "cmaes_population_size": cmaes_config.get("population_size"),
            "cmaes_sigma": cmaes_config.get("sigma"),
            "cmaes_generations": cmaes_config.get("generations"),
            "cmaes_games_per_eval": cmaes_config.get("games_per_eval"),
            "cmaes_opponent_mode": cmaes_config.get("opponent_mode"),
            "cmaes_eval_mode": cmaes_config.get("eval_mode"),
            "cmaes_run_id": cmaes_config.get("run_id"),
        })

    training_config = TrainingConfig(
        optimizer="cmaes",
        epochs=generation,  # Use generation as "epochs"
        extra_config=extra_config,
    )

    # Check existing best fitness for this config
    existing_best_fitness = 0.0
    existing_models = registry.list_models(model_type=ModelType.HEURISTIC)
    for model in existing_models:
        if model["model_id"] == model_id:
            model_metrics = model.get("metrics", {})
            existing_fitness = model_metrics.get("win_rate", 0.0)
            if existing_fitness and existing_fitness > existing_best_fitness:
                existing_best_fitness = existing_fitness

    # Determine initial stage based on fitness improvement
    initial_stage = ModelStage.DEVELOPMENT
    fitness_improved = fitness > existing_best_fitness + min_fitness_improvement

    # Register the model
    model_id, version = registry.register_model(
        name=model_name,
        model_path=weights_path,
        model_type=ModelType.HEURISTIC,
        description=f"CMA-ES optimized heuristic weights for {config_key}, fitness={fitness:.4f}",
        metrics=metrics,
        training_config=training_config,
        tags=[
            f"board:{board_type}",
            f"players:{num_players}",
            f"generation:{generation}",
            "cmaes",
        ],
        initial_stage=initial_stage,
        model_id=model_id,
    )

    logger.info(
        f"Registered CMA-ES result: {model_id}:v{version}, "
        f"fitness={fitness:.4f}, generation={generation}"
    )

    # Emit Prometheus metrics if available
    try:
        from prometheus_client import REGISTRY, Counter, Gauge

        def _get_metric(name, metric_type, description, labels):
            """Get or create a Prometheus metric."""
            try:
                return REGISTRY._names_to_collectors.get(name)
            except Exception:
                return None

        # Try to get existing metrics (defined in unified_ai_loop.py)
        cmaes_runs = _get_metric('ringrift_cmaes_runs_total', Counter, '', [])
        cmaes_fitness = _get_metric('ringrift_cmaes_best_fitness', Gauge, '', [])
        cmaes_generations = _get_metric('ringrift_cmaes_generations_total', Counter, '', [])

        if cmaes_runs:
            cmaes_runs.labels(config=config_key, board_type=board_type).inc()
        if cmaes_fitness:
            cmaes_fitness.labels(config=config_key, board_type=board_type).set(fitness)
        if cmaes_generations:
            cmaes_generations.labels(config=config_key).inc(generation)
    except ImportError:
        pass  # Prometheus not available
    except Exception as e:
        logger.debug(f"Failed to emit Prometheus metrics: {e}")

    # Auto-promote to staging if fitness improved significantly
    if auto_promote and fitness_improved:
        try:
            registry.promote(
                model_id,
                version,
                ModelStage.STAGING,
                reason=f"Fitness improved from {existing_best_fitness:.4f} to {fitness:.4f}",
                promoted_by="cmaes_auto_promote",
            )
            logger.info(
                f"Auto-promoted {model_id}:v{version} to STAGING "
                f"(fitness improvement: {fitness - existing_best_fitness:.4f})"
            )
        except ValueError as e:
            logger.warning(f"Auto-promotion failed: {e}")

    return model_id, version


def get_best_heuristic_model(
    board_type: str,
    num_players: int,
    registry_dir: Path | None = None,
    stage: str | None = None,
) -> dict[str, Any] | None:
    """Get the best heuristic model for a board configuration.

    Parameters
    ----------
    board_type:
        Board type (e.g., "square8").
    num_players:
        Number of players.
    registry_dir:
        Path to model registry directory.
    stage:
        If specified, only return models in this stage.

    Returns
    -------
    Optional[Dict[str, Any]]:
        Model data dict or None if not found.
    """
    from app.training.model_registry import (
        ModelRegistry,
        ModelStage,
        ModelType,
    )

    registry_dir = registry_dir or DEFAULT_REGISTRY_DIR
    if not registry_dir.exists():
        return None

    registry = ModelRegistry(registry_dir)
    config_key = f"{board_type}_{num_players}p"
    model_id = f"heuristic_{config_key}"

    # Get models filtered by stage if specified
    stage_filter = ModelStage(stage) if stage else None
    models = registry.list_models(stage=stage_filter, model_type=ModelType.HEURISTIC)

    # Find best model for this config by fitness (win_rate)
    best_model = None
    best_fitness = -1.0

    for model in models:
        if model["model_id"] == model_id:
            metrics = model.get("metrics", {})
            fitness = metrics.get("win_rate", 0.0)
            if fitness is not None and fitness > best_fitness:
                best_fitness = fitness
                best_model = model

    return best_model


def load_heuristic_weights_from_registry(
    board_type: str,
    num_players: int,
    registry_dir: Path | None = None,
    stage: str = "production",
) -> dict[str, float] | None:
    """Load heuristic weights from the best registered model.

    Parameters
    ----------
    board_type:
        Board type (e.g., "square8").
    num_players:
        Number of players.
    registry_dir:
        Path to model registry directory.
    stage:
        Stage to load from (default: production).

    Returns
    -------
    Optional[Dict[str, float]]:
        Heuristic weights dict or None if not found.
    """
    model = get_best_heuristic_model(board_type, num_players, registry_dir, stage)
    if not model:
        return None

    weights_path = Path(model["file_path"])
    if not weights_path.exists():
        logger.warning(f"Weights file not found: {weights_path}")
        return None

    with open(weights_path) as f:
        data = json.load(f)

    return data.get("weights")


def list_cmaes_runs(
    board_type: str | None = None,
    num_players: int | None = None,
    registry_dir: Path | None = None,
    limit: int = 10,
) -> list[dict[str, Any]]:
    """List recent CMA-ES optimization runs from the registry.

    Parameters
    ----------
    board_type:
        Filter by board type.
    num_players:
        Filter by number of players.
    registry_dir:
        Path to model registry directory.
    limit:
        Maximum number of results.

    Returns
    -------
    list[Dict[str, Any]]:
        List of model data dicts.
    """
    from app.training.model_registry import (
        ModelRegistry,
        ModelType,
    )

    registry_dir = registry_dir or DEFAULT_REGISTRY_DIR
    if not registry_dir.exists():
        return []

    registry = ModelRegistry(registry_dir)
    models = registry.list_models(model_type=ModelType.HEURISTIC)

    # Filter by config if specified
    # Model IDs follow pattern: heuristic_{board_type}_{num_players}p
    if board_type or num_players:
        filtered = []
        for model in models:
            model_id = model.get("model_id", "")
            # Parse board_type and num_players from model_id
            # Format: heuristic_{board_type}_{num_players}p
            if not model_id.startswith("heuristic_"):
                continue
            parts = model_id[len("heuristic_"):].rsplit("_", 1)
            if len(parts) != 2:
                continue
            id_board_type = parts[0]
            id_num_players_str = parts[1].rstrip("p")
            try:
                id_num_players = int(id_num_players_str)
            except ValueError:
                continue
            if board_type and id_board_type != board_type:
                continue
            if num_players and id_num_players != num_players:
                continue
            filtered.append(model)
        models = filtered

    # Sort by updated_at descending and limit
    models.sort(key=lambda m: m.get("updated_at", ""), reverse=True)
    return models[:limit]
