"""Unified Loop Local Selfplay Generation.

This module provides local selfplay data generation for the unified AI loop.
Uses the parallel_selfplay module for efficient multi-process game generation.

Supports:
- Descent AI (default)
- MCTS AI (standard tree search)
- Gumbel-MCTS AI (soft policy targets for improved training)
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional

if TYPE_CHECKING:
    from scripts.unified_ai_loop import EventBus, UnifiedLoopState

from app.utils.paths import AI_SERVICE_ROOT

# Import centralized Elo constants
from .config import INITIAL_ELO_RATING

# Import event types for data pipeline integration
try:
    from .config import DataEvent, DataEventType
    HAS_DATA_EVENTS = True
except ImportError:
    HAS_DATA_EVENTS = False
    DataEvent = None
    DataEventType = None

# Coordinator-only mode - skip local CPU-intensive work
DISABLE_LOCAL_TASKS = os.environ.get("RINGRIFT_DISABLE_LOCAL_TASKS", "").lower() in ("1", "true", "yes", "on")

logger = logging.getLogger(__name__)

# Import parallel selfplay module
try:
    from app.training.parallel_selfplay import (
        generate_dataset_parallel,
    )
    HAS_PARALLEL_SELFPLAY = True
except ImportError:
    HAS_PARALLEL_SELFPLAY = False
    generate_dataset_parallel = None

# Prometheus metrics - avoid duplicate registration
try:
    from prometheus_client import Counter, Histogram, REGISTRY
    HAS_PROMETHEUS = True

    if 'ringrift_local_selfplay_games_total' in REGISTRY._names_to_collectors:
        LOCAL_SELFPLAY_GAMES = REGISTRY._names_to_collectors['ringrift_local_selfplay_games_total']
    else:
        LOCAL_SELFPLAY_GAMES = Counter(
            'ringrift_local_selfplay_games_total',
            'Total local selfplay games generated',
            ['config', 'engine']
        )

    if 'ringrift_local_selfplay_samples_total' in REGISTRY._names_to_collectors:
        LOCAL_SELFPLAY_SAMPLES = REGISTRY._names_to_collectors['ringrift_local_selfplay_samples_total']
    else:
        LOCAL_SELFPLAY_SAMPLES = Counter(
            'ringrift_local_selfplay_samples_total',
            'Total local selfplay samples generated',
            ['config', 'engine']
        )

    if 'ringrift_local_selfplay_duration_seconds' in REGISTRY._names_to_collectors:
        LOCAL_SELFPLAY_DURATION = REGISTRY._names_to_collectors['ringrift_local_selfplay_duration_seconds']
    else:
        LOCAL_SELFPLAY_DURATION = Histogram(
            'ringrift_local_selfplay_duration_seconds',
            'Local selfplay generation duration',
            ['config', 'engine'],
            buckets=[10, 30, 60, 120, 300, 600, 1200, 1800, 3600]
        )
except ImportError:
    HAS_PROMETHEUS = False
    LOCAL_SELFPLAY_GAMES = None
    LOCAL_SELFPLAY_SAMPLES = None
    LOCAL_SELFPLAY_DURATION = None


class LocalSelfplayGenerator:
    """Generates selfplay data locally using parallel workers.

    Supports multiple AI engines:
    - descent: Fast descent-based AI (default)
    - mcts: Standard MCTS tree search
    - gumbel: Gumbel-MCTS with soft policy targets (best for training)

    Enhanced features (2025-12):
    - PFSP opponent selection for diverse training
    - Priority-based config selection based on training proximity
    - Curriculum weight integration for adaptive generation
    """

    def __init__(
        self,
        state: "UnifiedLoopState",
        event_bus: "EventBus",
        output_dir: Optional[Path] = None,
        num_workers: Optional[int] = None,
        training_scheduler: Optional[Any] = None,
    ):
        self.state = state
        self.event_bus = event_bus
        self.output_dir = output_dir or AI_SERVICE_ROOT / "data" / "games" / "local_selfplay"
        self.num_workers = num_workers
        self._running = False
        self._generation_task: Optional[asyncio.Task] = None
        # Reference to training scheduler for PFSP and priority access
        self._training_scheduler = training_scheduler
        # Reference to signal computer for evaluation feedback loop
        self._signal_computer = None

        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if not HAS_PARALLEL_SELFPLAY:
            logger.warning("Parallel selfplay module not available")

    def set_training_scheduler(self, scheduler: Any) -> None:
        """Set reference to training scheduler for PFSP integration."""
        self._training_scheduler = scheduler

    def set_signal_computer(self, signal_computer: Any) -> None:
        """Set reference to signal computer for evaluation feedback loop.

        This enables selfplay priority to account for ELO regression/momentum,
        giving more attention to configs that are underperforming.
        """
        self._signal_computer = signal_computer

    def get_pfsp_opponent(self, config_key: str, current_elo: float = INITIAL_ELO_RATING) -> Optional[str]:
        """Get PFSP-selected opponent for selfplay.

        Args:
            config_key: Config identifier
            current_elo: Current model Elo for matchmaking

        Returns:
            Opponent model path or None if PFSP not available
        """
        if self._training_scheduler is not None:
            return self._training_scheduler.get_pfsp_opponent(config_key, current_elo)
        return None

    def get_prioritized_config(self) -> Optional[str]:
        """Get the config that should have highest selfplay priority.

        Priority is based on:
        1. Proximity to training threshold (closer = higher priority)
        2. Curriculum weights (higher weight = higher priority)
        3. Time since last training (longer = higher priority)
        4. Evaluation feedback (regressing configs get priority boost)

        Returns:
            Config key with highest priority, or None if no configs
        """
        if not self.state.configs:
            return None

        best_config = None
        best_priority = -1.0

        for config_key, config_state in self.state.configs.items():
            priority = 0.0

            # Factor 1: Proximity to training threshold (0-1)
            # Closer to threshold = higher priority
            # Weight: 40% (reduced from 50% to make room for evaluation feedback)
            if self._training_scheduler:
                threshold = self._training_scheduler._get_dynamic_threshold(config_key)
                if threshold > 0:
                    proximity = min(1.0, config_state.games_since_training / threshold)
                    priority += proximity * 0.4

            # Factor 2: Curriculum weight (0.5-2.0 normalized to 0-1)
            # Weight: 25% (reduced from 30%)
            curriculum_weight = getattr(config_state, 'curriculum_weight', 1.0)
            normalized_curriculum = (curriculum_weight - 0.5) / 1.5
            priority += max(0, normalized_curriculum * 0.25)

            # Factor 3: Time since last training (0-1)
            # Weight: 15% (reduced from 20%)
            time_since_training = time.time() - config_state.last_training_time
            hours_since = time_since_training / 3600
            staleness_factor = min(1.0, hours_since / 6.0)  # Cap at 6 hours
            priority += staleness_factor * 0.15

            # Factor 4: Evaluation feedback - boost regressing configs
            # Weight: 20% (new factor)
            # Regressing/plateauing configs get more selfplay attention
            if self._signal_computer:
                try:
                    current_elo = getattr(config_state, 'current_elo', None) or 1500
                    signals = self._signal_computer.compute_signals(
                        current_games=config_state.games_since_training,
                        current_elo=current_elo,
                        config_key=config_key,
                    )
                    if signals.elo_regression_detected:
                        # Full boost for regression - needs urgent attention
                        priority += 0.20
                        logger.debug(f"[Priority] {config_key}: +0.20 (ELO regression detected)")
                    elif hasattr(signals, 'elo_trend') and signals.elo_trend < 0:
                        # Partial boost for declining performance
                        priority += 0.10
                        logger.debug(f"[Priority] {config_key}: +0.10 (ELO declining)")
                    elif hasattr(signals, 'elo_trend') and signals.elo_trend > 20:
                        # Slight reduction for configs doing very well (+20 Elo/hour)
                        # They need less attention, let others catch up
                        priority -= 0.05
                except Exception as e:
                    # Don't let signal computation errors break priority
                    logger.debug(f"[Priority] Signal computation failed for {config_key}: {e}")

            if priority > best_priority:
                best_priority = priority
                best_config = config_key

        return best_config

    def get_config_priorities(self) -> Dict[str, float]:
        """Get priority scores for all configs.

        Returns:
            Dict mapping config_key to priority score (0-1+)
            Note: Score can exceed 1.0 when evaluation feedback boosts priority.
        """
        priorities = {}

        for config_key, config_state in self.state.configs.items():
            priority = 0.0

            # Factor 1: Proximity to threshold (40%)
            if self._training_scheduler:
                threshold = self._training_scheduler._get_dynamic_threshold(config_key)
                if threshold > 0:
                    proximity = min(1.0, config_state.games_since_training / threshold)
                    priority += proximity * 0.4

            # Factor 2: Curriculum weight (25%)
            curriculum_weight = getattr(config_state, 'curriculum_weight', 1.0)
            normalized_curriculum = (curriculum_weight - 0.5) / 1.5
            priority += max(0, normalized_curriculum * 0.25)

            # Factor 3: Staleness (15%)
            time_since_training = time.time() - config_state.last_training_time
            hours_since = time_since_training / 3600
            staleness_factor = min(1.0, hours_since / 6.0)
            priority += staleness_factor * 0.15

            # Factor 4: Evaluation feedback (20%)
            if self._signal_computer:
                try:
                    current_elo = getattr(config_state, 'current_elo', None) or 1500
                    signals = self._signal_computer.compute_signals(
                        current_games=config_state.games_since_training,
                        current_elo=current_elo,
                        config_key=config_key,
                    )
                    if signals.elo_regression_detected:
                        priority += 0.20
                    elif hasattr(signals, 'elo_trend') and signals.elo_trend < 0:
                        priority += 0.10
                    elif hasattr(signals, 'elo_trend') and signals.elo_trend > 20:
                        priority -= 0.05
                except Exception:
                    pass  # Silent fallback if signal computation fails

            priorities[config_key] = priority

        return priorities

    def _get_diversity_need(self, config_key: str) -> float:
        """Get diversity need from training metrics (0-1).

        Returns higher values when training shows signs of:
        - Loss plateau (training not improving)
        - Overfitting (train/val loss divergence)

        In these cases, more diverse/exploratory selfplay is needed.

        Args:
            config_key: Config identifier

        Returns:
            Diversity need score (0-1), where higher = need more exploration
        """
        if not self._training_scheduler:
            return 0.0

        try:
            # Try to get training quality metrics from scheduler
            if hasattr(self._training_scheduler, 'get_training_quality'):
                quality = self._training_scheduler.get_training_quality(config_key)
                if quality:
                    if quality.get('overfit_detected'):
                        return 0.9  # High diversity need
                    if quality.get('loss_plateau'):
                        return 0.6  # Moderate diversity need
        except Exception as e:
            logger.debug(f"[DiversityNeed] Failed to get training quality for {config_key}: {e}")

        return 0.0

    def get_adaptive_engine(self, config_key: str, quality_threshold: float = 0.7) -> str:
        """Select selfplay engine based on training proximity and diversity needs.

        Engine selection priority:
        1. If diversity needed (overfit/plateau): 'mcts' for exploration
        2. If close to training threshold: 'gumbel' for quality
        3. Otherwise: 'descent' for throughput

        Args:
            config_key: Config identifier
            quality_threshold: Proximity threshold to switch to gumbel (0-1)

        Returns:
            Engine name: 'mcts' for diversity, 'gumbel' for quality, 'descent' for throughput
        """
        # Check for diversity need from training feedback
        diversity_needed = self._get_diversity_need(config_key)

        if diversity_needed > 0.7:
            # High diversity need: use MCTS for more exploration
            logger.info(f"[AdaptiveEngine] {config_key} diversity={diversity_needed:.2f} > 0.7 -> using 'mcts'")
            return "mcts"

        priorities = self.get_config_priorities()
        priority = priorities.get(config_key, 0.0)

        # Higher priority = closer to training threshold = use higher quality engine
        if priority >= quality_threshold:
            logger.info(f"[AdaptiveEngine] {config_key} priority={priority:.2f} >= {quality_threshold} -> using 'gumbel'")
            return "gumbel"
        else:
            logger.debug(f"[AdaptiveEngine] {config_key} priority={priority:.2f} < {quality_threshold} -> using 'descent'")
            return "descent"

    def get_all_adaptive_engines(self) -> Dict[str, str]:
        """Get recommended engine for all configs.

        Returns:
            Dict mapping config_key to recommended engine
        """
        return {
            config_key: self.get_adaptive_engine(config_key)
            for config_key in self.state.configs
        }

    async def generate_games(
        self,
        num_games: int,
        config_key: str,
        engine: str = "descent",
        nn_model_id: Optional[str] = None,
        gumbel_simulations: int = 64,
        gumbel_top_k: int = 16,
        progress_callback: Optional[callable] = None,
        use_pfsp_opponent: bool = False,
        current_elo: float = INITIAL_ELO_RATING,
        temperature: float = 1.0,
        use_temperature_decay: bool = True,
        opening_temperature: float = 1.5,
    ) -> Dict[str, Any]:
        """Generate selfplay games locally.

        Args:
            num_games: Number of games to generate
            config_key: Config identifier (e.g., "square8_2p")
            engine: AI engine ("descent", "mcts", or "gumbel")
            nn_model_id: Neural network model ID for AI
            gumbel_simulations: Simulations per move for Gumbel-MCTS
            gumbel_top_k: Top-k for sequential halving
            progress_callback: Optional callback(completed, total)
            use_pfsp_opponent: Whether to use PFSP for opponent selection
            current_elo: Current model Elo for PFSP matchmaking
            temperature: Base temperature for move selection (1.0 = standard)
            use_temperature_decay: Decay temperature from opening to base
            opening_temperature: Higher temperature for opening moves

        Returns:
            Dict with generation results
        """
        # Guard against local work in coordinator-only mode
        if DISABLE_LOCAL_TASKS:
            logger.info("[LocalSelfplay] Skipping local selfplay (RINGRIFT_DISABLE_LOCAL_TASKS=true)")
            return {
                "success": False,
                "error": "Coordinator-only mode",
                "games": 0,
                "samples": 0,
            }

        if not HAS_PARALLEL_SELFPLAY:
            return {
                "success": False,
                "error": "Parallel selfplay module not available",
                "games": 0,
                "samples": 0,
            }

        # Parse config key
        parts = config_key.rsplit("_", 1)
        board_type = parts[0]
        num_players = int(parts[1].replace("p", "")) if len(parts) > 1 else 2

        # Convert board_type to BoardType enum
        from app.models import BoardType
        board_type_enum = BoardType(board_type)

        # PFSP opponent selection (2025-12 enhancement)
        opponent_model = None
        if use_pfsp_opponent:
            opponent_model = self.get_pfsp_opponent(config_key, current_elo)
            if opponent_model:
                logger.info(f"[PFSP] Selected opponent: {opponent_model} for {config_key}")
            else:
                logger.debug(f"[PFSP] No opponent available, using self-play")

        # Generate output filename
        timestamp = int(time.time())
        output_file = self.output_dir / config_key / f"selfplay_{engine}_{timestamp}.npz"
        output_file.parent.mkdir(parents=True, exist_ok=True)

        start_time = time.time()

        try:
            opponent_info = f", opponent={opponent_model}" if opponent_model else ""
            logger.info(f"Starting local selfplay: {num_games} games, config={config_key}, engine={engine}{opponent_info}")

            # Run parallel selfplay in executor to avoid blocking event loop
            # Temperature scheduling: higher temp early in game for diverse positions
            temp_info = f", temp={temperature:.2f}" if temperature != 1.0 else ""
            if use_temperature_decay:
                temp_info = f", temp={opening_temperature:.1f}->{temperature:.2f}"
            logger.info(f"[Temperature] {config_key}: {temp_info or 'default (1.0)'}")

            loop = asyncio.get_event_loop()
            total_samples = await loop.run_in_executor(
                None,
                lambda: generate_dataset_parallel(
                    num_games=num_games,
                    output_file=str(output_file),
                    num_workers=self.num_workers,
                    board_type=board_type_enum,
                    num_players=num_players,
                    engine=engine,
                    nn_model_id=nn_model_id,
                    multi_player_values=(num_players > 2),
                    max_players=max(4, num_players),
                    progress_callback=progress_callback,
                    gumbel_simulations=gumbel_simulations,
                    gumbel_top_k=gumbel_top_k,
                    temperature=temperature,
                    use_temperature_decay=use_temperature_decay,
                    opening_temperature=opening_temperature,
                )
            )

            duration = time.time() - start_time
            games_per_sec = num_games / duration if duration > 0 else 0

            logger.info(
                f"Local selfplay complete: {num_games} games, {total_samples} samples "
                f"in {duration:.1f}s ({games_per_sec:.1f} games/sec)"
            )

            # Update Prometheus metrics
            if HAS_PROMETHEUS:
                LOCAL_SELFPLAY_GAMES.labels(config=config_key, engine=engine).inc(num_games)
                LOCAL_SELFPLAY_SAMPLES.labels(config=config_key, engine=engine).inc(total_samples)
                LOCAL_SELFPLAY_DURATION.labels(config=config_key, engine=engine).observe(duration)

            # Emit NEW_GAMES_AVAILABLE event for data pipeline integration
            # This allows the training scheduler and data sync services to react
            if HAS_DATA_EVENTS and self.event_bus:
                try:
                    import socket
                    host_name = socket.gethostname()
                    await self.event_bus.publish(DataEvent(
                        event_type=DataEventType.NEW_GAMES_AVAILABLE,
                        payload={
                            "host": host_name,
                            "new_games": num_games,
                            "samples": total_samples,
                            "config": config_key,
                            "engine": engine,
                            "output_file": str(output_file),
                            "quality_estimate": 0.7 if engine == "gumbel" else 0.5,
                        },
                        source="local_selfplay",
                    ))
                except Exception as e:
                    logger.debug(f"Failed to emit NEW_GAMES_AVAILABLE event: {e}")

            return {
                "success": True,
                "games": num_games,
                "samples": total_samples,
                "output_file": str(output_file),
                "duration_seconds": duration,
                "games_per_second": games_per_sec,
                "engine": engine,
                "config": config_key,
                "pfsp_opponent": opponent_model,
            }

        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"Local selfplay failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "games": 0,
                "samples": 0,
                "duration_seconds": duration,
                "engine": engine,
                "config": config_key,
            }

    async def generate_gumbel_games(
        self,
        num_games: int,
        config_key: str,
        nn_model_id: Optional[str] = None,
        simulations: int = 64,
        top_k: int = 16,
    ) -> Dict[str, Any]:
        """Convenience method to generate games using Gumbel-MCTS.

        Gumbel-MCTS produces higher quality training data with soft policy
        targets based on visit counts from sequential halving search.

        Args:
            num_games: Number of games to generate
            config_key: Config identifier
            nn_model_id: Neural network model ID
            simulations: Simulations per move
            top_k: Top-k actions for sequential halving

        Returns:
            Generation results
        """
        return await self.generate_games(
            num_games=num_games,
            config_key=config_key,
            engine="gumbel",
            nn_model_id=nn_model_id,
            gumbel_simulations=simulations,
            gumbel_top_k=top_k,
        )

    async def run_continuous_generation(
        self,
        config_key: str,
        target_games_per_hour: int = 100,
        engine: str = "descent",
        batch_size: int = 20,
        nn_model_id: Optional[str] = None,
    ) -> None:
        """Run continuous selfplay generation in the background.

        Generates games in batches to maintain the target rate while
        allowing for model updates between batches.

        Args:
            config_key: Config identifier
            target_games_per_hour: Target generation rate
            engine: AI engine to use
            batch_size: Games per batch
            nn_model_id: Neural network model ID
        """
        self._running = True
        games_per_batch = batch_size

        # Calculate delay between batches to achieve target rate
        batches_per_hour = target_games_per_hour / games_per_batch
        seconds_per_batch = 3600 / batches_per_hour if batches_per_hour > 0 else 60

        logger.info(
            f"Starting continuous selfplay: target={target_games_per_hour}/hour, "
            f"batch_size={games_per_batch}, interval={seconds_per_batch:.1f}s"
        )

        while self._running:
            try:
                batch_start = time.time()

                result = await self.generate_games(
                    num_games=games_per_batch,
                    config_key=config_key,
                    engine=engine,
                    nn_model_id=nn_model_id,
                )

                if result["success"]:
                    # Update state with new games
                    if config_key in self.state.configs:
                        self.state.configs[config_key].games_since_training += result["games"]
                        self.state.total_games_pending += result["games"]

                # Wait for next batch
                elapsed = time.time() - batch_start
                wait_time = max(0, seconds_per_batch - elapsed)
                if wait_time > 0:
                    await asyncio.sleep(wait_time)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Continuous selfplay error: {e}")
                await asyncio.sleep(60)  # Back off on error

        logger.info("Continuous selfplay stopped")

    def stop_continuous_generation(self) -> None:
        """Stop continuous selfplay generation."""
        self._running = False
        if self._generation_task and not self._generation_task.done():
            self._generation_task.cancel()

    def start_continuous_generation_task(
        self,
        config_key: str,
        target_games_per_hour: int = 100,
        engine: str = "descent",
    ) -> asyncio.Task:
        """Start continuous generation as an async task.

        Returns:
            The asyncio Task for the generation loop
        """
        self._generation_task = asyncio.create_task(
            self.run_continuous_generation(
                config_key=config_key,
                target_games_per_hour=target_games_per_hour,
                engine=engine,
            )
        )
        return self._generation_task

    def get_generation_stats(self) -> Dict[str, Any]:
        """Get statistics about local selfplay generation."""
        stats = {
            "output_dir": str(self.output_dir),
            "num_workers": self.num_workers,
            "running": self._running,
            "parallel_selfplay_available": HAS_PARALLEL_SELFPLAY,
            "gpu_selfplay_available": self._has_gpu_selfplay(),
        }

        # Count generated files per config
        if self.output_dir.exists():
            for config_dir in self.output_dir.iterdir():
                if config_dir.is_dir():
                    npz_files = list(config_dir.glob("*.npz"))
                    stats[f"{config_dir.name}_files"] = len(npz_files)

        return stats

    def _has_gpu_selfplay(self) -> bool:
        """Check if GPU selfplay is available."""
        try:
            import torch
            return torch.cuda.is_available() or (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available())
        except ImportError:
            return False

    async def generate_games_gpu(
        self,
        num_games: int,
        config_key: str,
        batch_size: int = 64,
        nn_model_path: Optional[str] = None,
        temperature: float = 1.0,
    ) -> Dict[str, Any]:
        """Generate selfplay games using GPU-accelerated parallel game runner.

        This uses app/ai/gpu_parallel_games.py for maximum throughput on GPU.
        Best for generating large volumes of games quickly.

        Args:
            num_games: Number of games to generate
            config_key: Config identifier (e.g., "square8_2p")
            batch_size: Number of games to run in parallel on GPU
            nn_model_path: Optional path to policy model for move selection
            temperature: Move selection temperature

        Returns:
            Dict with generation results
        """
        if DISABLE_LOCAL_TASKS:
            return {
                "success": False,
                "error": "Coordinator-only mode",
                "games": 0,
            }

        if not self._has_gpu_selfplay():
            # Fall back to CPU parallel selfplay
            logger.info("[GPUSelfplay] No GPU available, falling back to CPU parallel selfplay")
            return await self.generate_games(
                num_games=num_games,
                config_key=config_key,
                engine="descent",
                temperature=temperature,
            )

        try:
            from app.ai.gpu_parallel_games import ParallelGameRunner
            import torch

            # Parse config key
            parts = config_key.rsplit("_", 1)
            board_type = parts[0]
            num_players = int(parts[1].replace("p", "")) if len(parts) > 1 else 2

            # Determine board size
            board_size = 8 if "8" in board_type else 19

            # Determine device
            if torch.cuda.is_available():
                device = torch.device("cuda")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                device = torch.device("cpu")

            start_time = time.time()

            runner = ParallelGameRunner(
                batch_size=min(batch_size, num_games),
                board_size=board_size,
                num_players=num_players,
                device=device,
                temperature=temperature,
            )

            # Load policy model if provided
            if nn_model_path:
                runner.load_policy_model(nn_model_path)

            # Run games
            games_completed = 0
            output_dir = AI_SERVICE_ROOT / "data" / "games" / "gpu_selfplay" / config_key
            output_dir.mkdir(parents=True, exist_ok=True)

            # Run in batches
            while games_completed < num_games:
                batch_count = min(batch_size, num_games - games_completed)
                runner.batch_size = batch_count
                results = runner.run_games()
                games_completed += batch_count

            duration = time.time() - start_time
            games_per_sec = num_games / duration if duration > 0 else 0

            logger.info(
                f"[GPUSelfplay] Generated {num_games} games on {device} "
                f"in {duration:.1f}s ({games_per_sec:.1f} games/sec)"
            )

            # Emit NEW_GAMES_AVAILABLE event for data pipeline integration
            if HAS_DATA_EVENTS and self.event_bus:
                try:
                    import socket
                    host_name = socket.gethostname()
                    await self.event_bus.publish(DataEvent(
                        event_type=DataEventType.NEW_GAMES_AVAILABLE,
                        payload={
                            "host": host_name,
                            "new_games": num_games,
                            "config": config_key,
                            "engine": "gpu_parallel",
                            "device": str(device),
                            "quality_estimate": 0.6,  # GPU games have decent quality
                        },
                        source="gpu_selfplay",
                    ))
                except Exception as e:
                    logger.debug(f"Failed to emit NEW_GAMES_AVAILABLE event: {e}")

            return {
                "success": True,
                "games": num_games,
                "device": str(device),
                "duration_seconds": duration,
                "games_per_second": games_per_sec,
                "config": config_key,
            }

        except Exception as e:
            logger.error(f"[GPUSelfplay] Failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "games": 0,
            }
