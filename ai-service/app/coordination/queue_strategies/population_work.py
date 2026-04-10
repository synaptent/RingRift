"""Work-item creation and population strategies."""

from __future__ import annotations

import logging
import asyncio
import time
from typing import TYPE_CHECKING

from app.coordination.event_emission_helpers import safe_emit_event
from app.coordination.event_utils import make_config_key
from app.coordination.queue_strategies.common import (
    EXPLORATION_CONFIGS_PER_CYCLE,
    LARGE_BOARDS,
    MINIMUM_EXPLORATION_GAMES,
)

if TYPE_CHECKING:
    from app.coordination.work_queue import WorkItem

logger = logging.getLogger(__name__)


class QueuePopulationWorkMixin:
    """Extracted queue population behavior."""

    # Jan 1, 2026: Engine modes for diversity in selfplay
    # Format: (engine_mode, requires_model, player_restriction, requires_nnue)
    _ENGINE_MODES_2P = [
        ("gumbel", True, None, False),
        ("heuristic-only", False, None, False),
        ("minimax", False, 2, True),
        ("policy-only", True, None, False),
        ("descent", True, None, False),
    ]
    _ENGINE_MODES_MP = [
        ("gumbel", True, None, False),
        ("heuristic-only", False, None, False),
        ("maxn", False, (3, 4), True),
        ("brs", False, (3, 4), True),
        ("policy-only", True, None, False),
    ]
    _engine_mode_counter = 0
    _nnue_cache: dict[str, bool] = {}

    _BOARD_TRAINING_TIMEOUTS: dict[str, float] = {
        "hex8": 7200,
        "square8": 7200,
        "square19": 14400,
        "hexagonal": 18000,
    }
    _BOARD_TOURNAMENT_TIMEOUTS: dict[str, float] = {
        "hex8": 3600,
        "square8": 7200,
        "square19": 10800,
        "hexagonal": 14400,
    }
    _BOARD_TOURNAMENT_GAMES: dict[str, int] = {
        "hex8": 50,
        "square8": 50,
        "square19": 30,
        "hexagonal": 20,
    }

    def _get_scheduler_priorities(self) -> dict[str, float]:
        """Get priority scores from the SelfplayScheduler."""
        if self._selfplay_scheduler is None:
            return {}

        try:
            # Jan 2026: Use the sync method which reads from _config_priorities cache
            # This works in both sync and async contexts without blocking
            if hasattr(self._selfplay_scheduler, "get_priority_configs_sync"):
                priorities_list = self._selfplay_scheduler.get_priority_configs_sync(top_n=12)
                if priorities_list:
                    return dict(priorities_list)
                return {}

            # Fallback: Check if we're in a running event loop
            try:
                asyncio.get_running_loop()
                # We're in a running loop - can't use asyncio.run()
                # Try sync method via _config_priorities directly
                config_priorities = getattr(self._selfplay_scheduler, "_config_priorities", {})
                if config_priorities:
                    return {
                        cfg: p.priority_score
                        for cfg, p in config_priorities.items()
                    }
                return {}
            except RuntimeError:
                # No running loop - safe to create one and run async
                priorities_list = asyncio.run(
                    self._selfplay_scheduler.get_priority_configs(top_n=12)
                )
                return dict(priorities_list)
        except (AttributeError, TypeError, RuntimeError) as e:
            logger.debug(f"[QueuePopulator] Could not get scheduler priorities: {e}")
            return {}
    def _compute_work_priority(
        self,
        base_priority: int,
        config_key: str,
        scheduler_priorities: dict[str, float],
    ) -> int:
        """Compute adjusted work priority based on scheduler priorities."""
        if not scheduler_priorities:
            return base_priority

        scheduler_score = scheduler_priorities.get(config_key, 0.0)
        if scheduler_score <= 0:
            return base_priority

        max_score = max(scheduler_priorities.values()) if scheduler_priorities else 1.0
        normalized = scheduler_score / max(max_score, 0.01)
        priority_boost = int(normalized * 50)

        return base_priority + priority_boost
    @classmethod
    def _has_nnue_model(cls, board_type: str, num_players: int) -> bool:
        """Check if an NNUE model exists for the given configuration.

        Jan 14, 2026: Added for NNUE harness availability checks.
        Uses caching to avoid repeated filesystem checks.
        """
        cache_key = f"{board_type}_{num_players}p"
        if cache_key in cls._nnue_cache:
            return cls._nnue_cache[cache_key]

        # Check for NNUE model in standard locations
        from pathlib import Path

        nnue_paths = [
            Path(f"models/nnue/nnue_canonical_{board_type}_{num_players}p.pt"),
            Path(f"models/nnue/nnue_{board_type}_{num_players}p.pt"),
        ]

        exists = any(p.exists() for p in nnue_paths)
        cls._nnue_cache[cache_key] = exists

        if not exists:
            logger.debug(
                f"[QueuePopulator] No NNUE model found for {cache_key}, "
                f"NNUE harnesses will be skipped"
            )

        return exists
    def _should_force_queue_add(self, config_key: str) -> bool:
        """Determine if we should bypass backpressure for this config.

        January 2026: 4-player configs were being starved because the queue
        filled with 2-player jobs. When backpressure is hit, use force=True
        for underrepresented configs so they can still enter the queue.

        Force-add criteria:
        - 3p/4p configs with < 10,000 games (training data poverty)
        - Configs with < 1,000 games (severe data poverty)
        """
        target = self._targets.get(config_key)
        if not target:
            return False

        game_count = target.games_played

        # Severe data poverty - force add regardless of player count
        if game_count < 1000:
            logger.debug(
                f"[QueuePopulator] Force-add {config_key}: severe data poverty ({game_count} games)"
            )
            return True

        # 3p/4p configs are underrepresented - force add if < 10,000 games
        if config_key.endswith("_3p") or config_key.endswith("_4p"):
            if game_count < 10000:
                logger.debug(
                    f"[QueuePopulator] Force-add {config_key}: multiplayer data poverty ({game_count} games)"
                )
                return True

        return False
    def _create_selfplay_item(
        self, board_type: str, num_players: int
    ) -> "WorkItem":
        """Create a selfplay work item with diverse engine types.

        Jan 1, 2026: Added engine mode rotation for training data diversity.
        Cycles through available harness types (gumbel, heuristic, minimax, etc.)
        to ensure training data comes from varied play styles.

        Jan 13, 2026: Added canonical model fallback to enable gumbel-mcts even
        when no model has Elo rating yet. This fixes the bug where 98% of selfplay
        was heuristic-only because best_model_id was None.
        """
        from app.coordination.work_queue import WorkItem, WorkType

        work_id = f"selfplay_{board_type}_{num_players}p_{int(time.time() * 1000)}"

        key = make_config_key(board_type, num_players)
        target = self._targets.get(key)
        best_model = target.best_model_id if target else None
        model_elo = target.current_best_elo if target else 1500.0

        # Jan 13, 2026: Fallback to canonical model if no Elo-rated model found
        # This enables gumbel-mcts mode even before first gauntlet evaluation
        if not best_model:
            best_model = self._find_canonical_model(board_type, num_players)
            if best_model:
                logger.debug(
                    f"[QueuePopulator] Using canonical model fallback for {board_type}_{num_players}p: {best_model}"
                )

        # Select engine mode with diversity rotation
        # Jan 1, 2026: Rotate through multiple engine modes instead of always using one
        # Jan 14, 2026: Allow gumbel for large boards with reduced simulations when model exists
        if board_type in LARGE_BOARDS:
            if best_model:
                engine_mode = "gumbel"  # Use reduced simulations (set below)
            else:
                engine_mode = "heuristic-only"
        else:
            # Select engine mode based on player count and rotation counter
            modes = self._ENGINE_MODES_2P if num_players == 2 else self._ENGINE_MODES_MP

            # Find valid modes for this configuration
            # Jan 14, 2026: Added requires_nnue check to enable NNUE harness diversity
            valid_modes = []
            for mode, requires_model, player_restrict, requires_nnue in modes:
                # Check player restriction
                if player_restrict is not None:
                    if isinstance(player_restrict, int) and num_players != player_restrict:
                        continue
                    if isinstance(player_restrict, tuple) and num_players not in player_restrict:
                        continue
                # Check model requirement
                if requires_model and not best_model:
                    continue
                # Check NNUE requirement (Jan 14, 2026: for minimax/maxn/brs)
                if requires_nnue and not self._has_nnue_model(board_type, num_players):
                    continue
                valid_modes.append(mode)

            if valid_modes:
                # Rotate through valid modes
                type(self)._engine_mode_counter += 1
                engine_mode = valid_modes[type(self)._engine_mode_counter % len(valid_modes)]
            else:
                # Fallback to heuristic if no valid modes
                engine_mode = "heuristic-only"

        # Jan 5, 2026 (Phase 3): Make requires_gpu conditional to utilize CPU nodes
        # - heuristic-only selfplay doesn't need GPU
        # - small boards (hex8, square8) can run on CPU with heuristic mode
        # This allows Hetzner CPU nodes to contribute to P2P quorum + evaluation
        requires_gpu = True
        if engine_mode == "heuristic-only":
            requires_gpu = False
        elif engine_mode in ("nnue-guided", "policy-only") and board_type in ("hex8", "square8"):
            # NNUE and policy-only are CPU-friendly for small boards
            requires_gpu = False

        config = {
            "board_type": board_type,
            "num_players": num_players,
            "games": self.config.selfplay_games_per_item,
            "source": "queue_populator",
            "engine_mode": engine_mode,
            "requires_gpu": requires_gpu,
        }

        # Jan 14, 2026: Reduce simulations for large boards to make gumbel feasible
        if board_type in LARGE_BOARDS and engine_mode == "gumbel":
            config["gumbel_simulations"] = 64  # Reduced from default 800

        if best_model and model_elo >= 1600:
            config["model_id"] = best_model
            config["model_elo"] = model_elo

        return WorkItem(
            work_id=work_id,
            work_type=WorkType.SELFPLAY,
            priority=self.config.selfplay_priority,
            config=config,
        )
    def _is_training_ready(
        self, board_type: str, num_players: int, min_samples: int | None = None
    ) -> tuple[bool, int]:
        """Check if training data is available for a config.

        Dec 31, 2025: Added to prevent adding TRAINING work items when no
        training data exists. This was causing training jobs to complete
        instantly with loss=0.0000 because nodes had nothing to train on.

        Jan 5, 2026: Fixed to use config.min_games_for_training (100) instead
        of hardcoded 5000. The 50x gap was preventing training from triggering
        for new configs.

        Args:
            board_type: Board type (e.g., "hex8")
            num_players: Number of players (2, 3, or 4)
            min_samples: Minimum samples required for training. If None, uses
                         config.min_games_for_training (default 100).

        Returns:
            Tuple of (is_ready, sample_count). is_ready is True if sufficient
            training data exists.
        """
        # Use config value instead of hardcoded 5000
        if min_samples is None:
            min_samples = self.config.min_games_for_training

        config_key = make_config_key(board_type, num_players)

        try:
            from app.distributed.data_catalog import DataCatalog

            catalog = DataCatalog()
            npz_sources = catalog.discover_npz_files(
                board_type=board_type,
                num_players=num_players,
                min_samples=min_samples,
            )

            if npz_sources:
                total_samples = sum(s.sample_count for s in npz_sources)
                if total_samples >= min_samples:
                    return True, total_samples

            # Also check TrainingTriggerDaemon state if available
            try:
                from app.coordination.training_trigger_daemon import TrainingTriggerDaemon

                daemon = TrainingTriggerDaemon.get_instance_if_exists()
                if daemon:
                    state = daemon._training_states.get(config_key)
                    if state and state.npz_sample_count >= min_samples:
                        return True, state.npz_sample_count
            except (ImportError, AttributeError):
                pass

            return False, 0
        except (ImportError, OSError, AttributeError) as e:
            logger.debug(f"Training readiness check failed for {config_key}: {e}")
            return False, 0
    def _create_training_item(
        self, board_type: str, num_players: int
    ) -> "WorkItem":
        """Create a training work item."""
        from app.coordination.work_queue import WorkItem, WorkType

        work_id = f"training_{board_type}_{num_players}p_{int(time.time() * 1000)}"
        is_hex = board_type.startswith("hex")

        # Mar 2026: Use board-size-based timeout to prevent premature re-queue.
        base_timeout = self._BOARD_TRAINING_TIMEOUTS.get(board_type, 7200)
        player_multiplier = {2: 1.0, 3: 1.25, 4: 1.5}.get(num_players, 1.0)
        timeout = base_timeout * player_multiplier

        return WorkItem(
            work_id=work_id,
            work_type=WorkType.TRAINING,
            priority=self.config.training_priority,
            timeout_seconds=timeout,
            config={
                "board_type": board_type,
                "num_players": num_players,
                "source": "queue_populator",
                "enable_augmentation": True,
                "use_integrated_enhancements": True,
                "augment_hex_symmetry": is_hex,
                # Feb 2026: Training requires GPU - prevents CPU-only nodes
                # (hetzner-cpu*) from claiming training work they can't complete
                "requires_gpu": True,
            },
        )
    def _create_tournament_item(
        self, board_type: str, num_players: int
    ) -> "WorkItem":
        """Create a tournament work item."""
        from app.coordination.work_queue import WorkItem, WorkType

        work_id = f"tournament_{board_type}_{num_players}p_{int(time.time() * 1000)}"

        # Jan 5, 2026 (Phase 6): Small board tournament/evaluation can run on CPU nodes
        # This allows Hetzner CPU nodes to contribute to evaluation work
        requires_gpu = board_type not in ("hex8", "square8")

        # Feb 1, 2026: Use graduated timeouts based on board size.
        # 4-player games need 2x time, 3-player needs 1.5x.
        base_timeout = self._BOARD_TOURNAMENT_TIMEOUTS.get(board_type, 3600)
        player_multiplier = {2: 1.0, 3: 1.5, 4: 2.0}.get(num_players, 1.0)
        timeout = base_timeout * player_multiplier

        # Feb 1, 2026: Reduce game count for large boards to fit within timeout.
        games = self._BOARD_TOURNAMENT_GAMES.get(
            board_type, self.config.tournament_games
        )

        return WorkItem(
            work_id=work_id,
            work_type=WorkType.TOURNAMENT,
            priority=self.config.tournament_priority,
            timeout_seconds=timeout,
            config={
                "board_type": board_type,
                "num_players": num_players,
                "games": games,
                "source": "queue_populator",
                "requires_gpu": requires_gpu,
            },
        )
    def _create_sweep_item(
        self,
        board_type: str,
        num_players: int,
        base_model_id: str,
        base_elo: float,
    ) -> "WorkItem":
        """Create a hyperparameter sweep work item."""
        from app.coordination.work_queue import WorkItem, WorkType

        work_id = f"sweep_{board_type}_{num_players}p_{int(time.time() * 1000)}"

        if base_elo >= 1900:
            strategy = "bayesian"
            trials = 20
        else:
            strategy = "random"
            trials = 30

        return WorkItem(
            work_id=work_id,
            work_type=WorkType.HYPERPARAM_SWEEP,
            priority=60,
            config={
                "board_type": board_type,
                "num_players": num_players,
                "base_model_id": base_model_id,
                "base_elo": base_elo,
                "strategy": strategy,
                "trials": trials,
                "source": "queue_populator",
                "search_params": ["learning_rate", "batch_size", "weight_decay"],
            },
        )
    def populate(self) -> int:
        """Populate the work queue to maintain minimum depth.

        Returns:
            Number of items added
        """
        if not self.config.enabled:
            return 0

        if self._work_queue is None:
            logger.warning("No work queue set, cannot populate")
            return 0

        self._prune_stale_queued_work_ids()

        # Mar 2026: Lazy-load game counts (deferred from __init__)
        self.ensure_game_counts_loaded()

        # === January 14, 2026: Backoff check ===
        # If we're in a backoff period, skip population
        if self._is_backing_off():
            remaining = self._backoff_until - time.time()
            logger.debug(
                f"[QueuePopulator] In backoff period, {remaining:.1f}s remaining"
            )
            self._log_health_status()  # Log health during backoff
            return 0

        # === January 14, 2026: Circuit breaker check ===
        if not self._circuit_breaker_allow():
            logger.warning("[QueuePopulator] Circuit breaker OPEN, skipping population")
            self._log_health_status()
            return 0

        # January 13, 2026: Check if any workers can claim work
        # If all circuit breakers are open, skip population to prevent queue accumulation
        claimable_workers = self._count_claimable_workers()
        if claimable_workers == 0:
            logger.warning(
                "[QueuePopulator] No claimable workers (all circuit breakers open), "
                "skipping population"
            )
            safe_emit_event(
                "QUEUE_POPULATION_SKIPPED",
                {
                    "reason": "no_claimable_workers",
                    "circuit_breaker_blocking_all": True,
                    "dead_nodes_count": len(self._dead_nodes),
                },
            )
            return 0

        if self.all_targets_met():
            # Phase 1.2: Even when all targets met, add exploration work for stale configs
            # This prevents cluster idling and maintains training data diversity
            exploration_added = self._populate_exploration_work()
            if exploration_added > 0:
                logger.info(f"All Elo targets met - added {exploration_added} exploration items")
                self._maybe_emit_queue_exhausted_event()
                return exploration_added

            # January 7, 2026: Force minimum selfplay to prevent complete stall
            # When exploration also returns 0, add some selfplay to keep cluster active
            minimum_added = self._populate_minimum_selfplay(min_items=10)
            if minimum_added > 0:
                self._maybe_emit_queue_exhausted_event()
                return minimum_added

            logger.info("All Elo targets met, no population needed")
            self._maybe_emit_queue_exhausted_event()
            return 0

        # Check backpressure
        bp_level, reduction_factor = self._check_backpressure()

        # Emit backpressure events on state changes (Jan 2026)
        try:
            from app.coordination.queue_monitor import get_queue_monitor

            monitor = get_queue_monitor()
            status = monitor.get_overall_status() if monitor else {}
        except ImportError:
            status = {}
        self._maybe_emit_backpressure_event(bp_level, status)

        if bp_level.should_stop():
            # Phase 15.1.2: Trickle mode - never completely stop population
            # This prevents the pipeline from starving when backpressure is high
            if self.config.trickle_mode_enabled:
                logger.warning(
                    f"[QueuePopulator] Backpressure {bp_level.value} - TRICKLE MODE: "
                    f"adding {self.config.trickle_min_items} items to prevent starvation"
                )
                trickle_result = self._populate_trickle_items()
                self._maybe_emit_queue_exhausted_event()
                return trickle_result
            else:
                logger.info(
                    f"[QueuePopulator] Backpressure {bp_level.value} - skipping population"
                )
                self._maybe_emit_queue_exhausted_event()
                return 0

        items_needed = self.calculate_items_needed()
        if items_needed <= 0:
            return 0

        # Apply backpressure reduction
        if reduction_factor < 1.0:
            original_needed = items_needed
            items_needed = max(1, int(items_needed * reduction_factor))
            logger.info(
                f"[QueuePopulator] Backpressure {bp_level.value}: {original_needed} -> {items_needed}"
            )

        # Apply cluster health factor
        if self._cluster_health_factor < 1.0:
            original = items_needed
            items_needed = max(1, int(items_needed * self._cluster_health_factor))
            logger.debug(
                f"[QueuePopulator] Cluster health {self._cluster_health_factor:.2f}: "
                f"{original} -> {items_needed}"
            )

        # January 10, 2026: Apply worker capacity limit to prevent queue overfilling
        # This prevents the queue from hitting the hard limit (4000) by not adding
        # more work than workers can reasonably process.
        worker_capacity = self._get_worker_capacity()
        if items_needed > worker_capacity:
            original = items_needed
            items_needed = worker_capacity
            logger.info(
                f"[QueuePopulator] Worker capacity limit: {original} -> {items_needed} "
                f"(capacity={worker_capacity})"
            )

        # Get scheduler priorities
        scheduler_priorities = self._get_scheduler_priorities()

        # === January 30, 2026: Per-type limit enforcement ===
        # Get current pending counts by type to prevent backlog accumulation
        pending_by_type = self.get_pending_by_type()

        # Calculate distribution
        selfplay_count = int(items_needed * self.config.selfplay_ratio)
        training_count = int(items_needed * self.config.training_ratio)
        tournament_count = items_needed - selfplay_count - training_count

        # Reduce counts if over per-type limits
        pending_selfplay = pending_by_type.get("selfplay", 0)
        pending_training = pending_by_type.get("training", 0)
        pending_tournament = pending_by_type.get("tournament", 0)

        if pending_selfplay >= self.config.max_pending_selfplay:
            logger.info(
                f"[QueuePopulator] Selfplay over limit ({pending_selfplay}/{self.config.max_pending_selfplay}), skipping"
            )
            selfplay_count = 0

        if pending_training >= self.config.max_pending_training:
            logger.info(
                f"[QueuePopulator] Training over limit ({pending_training}/{self.config.max_pending_training}), skipping"
            )
            training_count = 0

        if pending_tournament >= self.config.max_pending_tournament:
            # Feb 2026: Instead of blanket-blocking all tournaments, check per-config.
            # Previously, configs like hex8/hexagonal got zero evaluation for days
            # because sq8/sq19 had 40+ pending each, exceeding the 200 global limit.
            per_config = self._get_pending_tournament_by_config()
            all_config_keys = set()
            for t in list(self._targets.values()):
                all_config_keys.add(t.config_key)
            starved = [k for k in all_config_keys if per_config.get(k, 0) == 0]
            if starved:
                tournament_count = len(starved)
                self._tournament_starved_configs = set(starved)
                logger.info(
                    f"[QueuePopulator] Tournament over limit ({pending_tournament}/"
                    f"{self.config.max_pending_tournament}), but {len(starved)} configs "
                    f"have 0 pending - adding for: {sorted(starved)}"
                )
            else:
                tournament_count = 0
                self._tournament_starved_configs = None
                logger.info(
                    f"[QueuePopulator] Tournament over limit ({pending_tournament}/"
                    f"{self.config.max_pending_tournament}), all configs have pending, skipping"
                )

        # Early exit if all types are over limit
        if selfplay_count == 0 and training_count == 0 and tournament_count == 0:
            logger.info(
                f"[QueuePopulator] All work types over limits: "
                f"selfplay={pending_selfplay}/{self.config.max_pending_selfplay}, "
                f"training={pending_training}/{self.config.max_pending_training}, "
                f"tournament={pending_tournament}/{self.config.max_pending_tournament}"
            )
            return 0

        # Get unmet targets sorted by priority
        unmet = self.get_unmet_targets()
        if not unmet:
            # Feb 2026: Even when all targets are met, still create maintenance
            # tournaments so met configs continue to get evaluated. Without this,
            # configs that reached target Elo stop getting gauntlet matches entirely.
            met_targets = [t for t in self._targets.values() if t.target_met]
            maintenance_added = 0
            for target in met_targets:
                try:
                    item = self._create_tournament_item(target.board_type, target.num_players)
                    item.priority = max(item.priority - 5, 1)  # Slight depriority only
                    self._work_queue.add_work(item)
                    self._track_queued_work_id(item.work_id)
                    maintenance_added += 1
                except Exception:
                    pass
            if maintenance_added > 0:
                logger.info(
                    f"[QueuePopulator] All targets met. Added {maintenance_added} "
                    f"maintenance tournaments for continued evaluation."
                )
                self._last_populate_time = time.time()
            return maintenance_added

        if scheduler_priorities:
            unmet.sort(
                key=lambda t: scheduler_priorities.get(t.config_key, 0.0),
                reverse=True,
            )

        # Feb 2026: Cap per-type counts at number of unique configs to prevent
        # creating duplicate work items for the same config in a single cycle.
        # Previously, when selfplay_count > len(unmet), the modulo wrap-around
        # (unmet[i % len(unmet)]) created multiple identical items per config,
        # leading to duplicate training/selfplay processes on the same node.
        selfplay_count = min(selfplay_count, len(unmet))
        training_count = min(training_count, len(unmet))
        tournament_count = min(tournament_count, len(unmet))

        added = 0

        # Add selfplay items
        for i in range(selfplay_count):
            target = unmet[i % len(unmet)]
            try:
                item = self._create_selfplay_item(target.board_type, target.num_players)
                if scheduler_priorities:
                    item.priority = self._compute_work_priority(
                        item.priority, target.config_key, scheduler_priorities
                    )
                # January 2026: Force-add for starved configs to bypass backpressure
                force_add = self._should_force_queue_add(target.config_key)
                self._work_queue.add_work(item, force=force_add)
                self._track_queued_work_id(item.work_id)
                target.pending_selfplay_count += 1
                added += 1
            except RuntimeError as e:
                # January 14, 2026: Detect hard limit hit and apply backoff
                if "hard limit" in str(e).lower() or "BACKPRESSURE" in str(e):
                    self._apply_backoff()
                    self._circuit_breaker_record_failure()
                    logger.warning(f"[QueuePopulator] Queue hard limit hit, entering backoff: {e}")
                    break  # Stop trying to add more items
                logger.error(f"Failed to add selfplay item: {e}")
            except Exception as e:
                logger.error(f"Failed to add selfplay item: {e}")

        # Add training items (only if training data exists)
        # Dec 31, 2025: Check training readiness before adding TRAINING work.
        # Previously, training items were added blindly at a 30% ratio, causing
        # jobs to complete instantly with loss=0.0000 when no data existed.
        training_added = 0
        training_skipped = 0
        for i in range(training_count):
            target = unmet[i % len(unmet)]
            try:
                # Check if training data exists before creating work item
                is_ready, sample_count = self._is_training_ready(
                    target.board_type, target.num_players
                )
                if not is_ready:
                    # No training data - skip this config and add selfplay instead
                    training_skipped += 1
                    logger.debug(
                        f"[QueuePopulator] Skipping training for {target.config_key}: "
                        f"insufficient data ({sample_count} samples)"
                    )
                    # Add selfplay item instead to generate more training data
                    try:
                        selfplay_item = self._create_selfplay_item(
                            target.board_type, target.num_players
                        )
                        if scheduler_priorities:
                            selfplay_item.priority = self._compute_work_priority(
                                selfplay_item.priority, target.config_key, scheduler_priorities
                            )
                        # January 2026: Force-add for starved configs to bypass backpressure
                        force_add = self._should_force_queue_add(target.config_key)
                        self._work_queue.add_work(selfplay_item, force=force_add)
                        self._track_queued_work_id(selfplay_item.work_id)
                        target.pending_selfplay_count += 1
                        added += 1
                    except RuntimeError as sp_err:
                        if "hard limit" in str(sp_err).lower() or "BACKPRESSURE" in str(sp_err):
                            self._apply_backoff()
                            self._circuit_breaker_record_failure()
                            break
                        logger.error(f"Failed to add replacement selfplay item: {sp_err}")
                    except Exception as sp_err:
                        logger.error(f"Failed to add replacement selfplay item: {sp_err}")
                    continue

                item = self._create_training_item(target.board_type, target.num_players)
                if scheduler_priorities:
                    item.priority = self._compute_work_priority(
                        item.priority, target.config_key, scheduler_priorities
                    )
                # January 2026: Force-add for starved configs to bypass backpressure
                force_add = self._should_force_queue_add(target.config_key)
                self._work_queue.add_work(item, force=force_add)
                self._track_queued_work_id(item.work_id)
                added += 1
                training_added += 1
            except RuntimeError as e:
                if "hard limit" in str(e).lower() or "BACKPRESSURE" in str(e):
                    self._apply_backoff()
                    self._circuit_breaker_record_failure()
                    break
                logger.error(f"Failed to add training item: {e}")
            except Exception as e:
                logger.error(f"Failed to add training item: {e}")

        if training_skipped > 0:
            logger.info(
                f"[QueuePopulator] Training skipped for {training_skipped} configs "
                f"(no data), added {training_added} training + {training_skipped} extra selfplay"
            )

        # Add tournament items
        # Feb 2026: When global limit is hit but some configs are starved,
        # only create tournaments for starved configs to ensure fair evaluation.
        starved_configs = getattr(self, "_tournament_starved_configs", None)
        if starved_configs and tournament_count > 0:
            # Build target list from all targets (met + unmet) that are starved
            tournament_targets = [
                t for t in self._targets.values()
                if t.config_key in starved_configs
            ]
        elif tournament_count > 0 and unmet:
            tournament_targets = [unmet[i % len(unmet)] for i in range(tournament_count)]
        else:
            tournament_targets = []
        # Reset starved config tracking
        self._tournament_starved_configs = None

        for target in tournament_targets:
            try:
                item = self._create_tournament_item(target.board_type, target.num_players)
                if scheduler_priorities:
                    item.priority = self._compute_work_priority(
                        item.priority, target.config_key, scheduler_priorities
                    )
                # Force-add for starved configs to bypass backpressure
                force_add = starved_configs is not None or self._should_force_queue_add(target.config_key)
                self._work_queue.add_work(item, force=force_add)
                self._track_queued_work_id(item.work_id)
                added += 1
            except RuntimeError as e:
                if "hard limit" in str(e).lower() or "BACKPRESSURE" in str(e):
                    self._apply_backoff()
                    self._circuit_breaker_record_failure()
                    break
                logger.error(f"Failed to add tournament item: {e}")
            except Exception as e:
                logger.error(f"Failed to add tournament item: {e}")

        # Feb 2026: Add maintenance tournaments for "met" configs.
        # Configs that reached target Elo were excluded from tournament work,
        # causing evaluation to stall (hexagonal_2p/3p, square19_3p/4p had
        # no gauntlet matches for 2-4 weeks). This ensures all configs get
        # periodic re-evaluation to track regressions and maintain Elo freshness.
        met_targets = [t for t in self._targets.values() if t.target_met]
        maintenance_added = 0
        for target in met_targets:
            # One tournament item per met config per populate cycle
            try:
                item = self._create_tournament_item(target.board_type, target.num_players)
                item.priority = max(item.priority - 5, 1)  # Slight depriority only
                self._work_queue.add_work(item)
                self._track_queued_work_id(item.work_id)
                added += 1
                maintenance_added += 1
            except RuntimeError as e:
                if "hard limit" in str(e).lower() or "BACKPRESSURE" in str(e):
                    break
                logger.debug(f"Failed to add maintenance tournament: {e}")
            except Exception as e:
                logger.debug(f"Failed to add maintenance tournament: {e}")

        if maintenance_added > 0:
            logger.info(
                f"[QueuePopulator] Added {maintenance_added} maintenance tournaments "
                f"for {len(met_targets)} met configs"
            )

        # Add hyperparameter sweeps opportunistically
        sweep_added = 0
        for target in unmet:
            if target.current_best_elo >= 1600 and target.best_model_id:
                sweep_key = f"sweep_{target.board_type}_{target.num_players}p_"
                if not any(wid.startswith(sweep_key) for wid in self._queued_work_ids):
                    try:
                        item = self._create_sweep_item(
                            target.board_type,
                            target.num_players,
                            target.best_model_id,
                            target.current_best_elo,
                        )
                        # January 2026: Force-add for starved configs to bypass backpressure
                        force_add = self._should_force_queue_add(target.config_key)
                        self._work_queue.add_work(item, force=force_add)
                        self._track_queued_work_id(item.work_id)
                        added += 1
                        sweep_added += 1
                        break
                    except RuntimeError as e:
                        if "hard limit" in str(e).lower() or "BACKPRESSURE" in str(e):
                            self._apply_backoff()
                            self._circuit_breaker_record_failure()
                            break
                        logger.error(f"Failed to add sweep item: {e}")
                    except Exception as e:
                        logger.error(f"Failed to add sweep item: {e}")

        # Phase 1.2: Also add exploration work for stale configs
        # This ensures diversity even when focusing on unmet targets
        exploration_added = self._populate_exploration_work()
        added += exploration_added

        self._last_populate_time = time.time()
        # Dec 31, 2025: Show actual training items added vs planned
        # Training items may be skipped if no training data exists
        logger.info(
            f"Populated queue with {added} items "
            f"(selfplay={selfplay_count + training_skipped}, training={training_added}, "
            f"tournament={tournament_count}, sweeps={sweep_added}, exploration={exploration_added})"
        )

        # January 2026 - Phase 3 Task 4: Check if queue is exhausted after population
        # This triggers UnderutilizationRecoveryHandler if the queue is empty
        self._maybe_emit_queue_exhausted_event()

        # === January 14, 2026: Post-populate checks ===
        # Record success with circuit breaker
        if added > 0:
            self._circuit_breaker_record_success()
            self._reset_backoff()  # Reset backoff on successful population

        # Check for cluster partition (queue not draining)
        self._check_partition()

        # Log health status periodically
        self._log_health_status()

        return added
    def populate_queue(self) -> int:
        """Backward-compatible alias for populate()."""
        return self.populate()
    def _populate_exploration_work(self) -> int:
        """Add exploration work for stale configs (Phase 1.2 - Jan 2026).

        This ensures the cluster never completely idles, even when all Elo
        targets are met. It maintains training data diversity by exploring
        configs that haven't had recent activity.

        Returns:
            Number of exploration items added
        """
        if self._work_queue is None:
            return 0

        # Get stale configs that need exploration
        stale_configs = self.get_least_recent_configs(EXPLORATION_CONFIGS_PER_CYCLE)
        if not stale_configs:
            return 0

        added = 0
        for target in stale_configs:
            pending_games = self.get_pending_selfplay_games(target.config_key)

            if pending_games >= MINIMUM_EXPLORATION_GAMES:
                # Already have enough pending work
                continue

            try:
                item = self._create_selfplay_item(target.board_type, target.num_players)
                # Slightly boost priority for exploration items
                item.priority = self.config.selfplay_priority + 10
                # January 2026: Force-add for starved configs to bypass backpressure
                force_add = self._should_force_queue_add(target.config_key)
                self._work_queue.add_work(item, force=force_add)
                self._track_queued_work_id(item.work_id)
                target.pending_selfplay_count += 1
                added += 1
            except Exception as e:
                logger.error(f"[Exploration] Failed to add item for {target.config_key}: {e}")

        if added > 0:
            logger.info(
                f"[Exploration] Added {added} exploration items for stale configs: "
                f"{', '.join(t.config_key for t in stale_configs[:added])}"
            )

        return added
    def _populate_minimum_selfplay(self, min_items: int = 10) -> int:
        """Add minimum selfplay items to prevent complete pipeline stall.

        January 7, 2026: Added to fix stale Elo ratings. When all_targets_met()
        returns true and _populate_exploration_work() returns 0, the pipeline
        completely stalls. This method ensures at least some selfplay continues
        to keep the cluster active and generate fresh training data.

        Args:
            min_items: Minimum number of selfplay items to add (default 10)

        Returns:
            Number of items added
        """
        if self._work_queue is None:
            return 0

        # Pick configs to populate, prioritizing underserved ones
        all_targets = list(self._targets.values())
        if not all_targets:
            return 0

        # Sort by pending count (ascending) to prioritize configs with less work
        sorted_targets = sorted(
            all_targets,
            key=lambda t: t.pending_selfplay_count,
        )

        added = 0
        target_idx = 0
        while added < min_items and target_idx < len(sorted_targets):
            target = sorted_targets[target_idx % len(sorted_targets)]
            try:
                item = self._create_selfplay_item(target.board_type, target.num_players)
                # Use slightly lower priority to not compete with normal work
                item.priority = max(10, self.config.selfplay_priority - 10)
                # January 2026: Force-add for starved configs to bypass backpressure
                force_add = self._should_force_queue_add(target.config_key)
                self._work_queue.add_work(item, force=force_add)
                self._track_queued_work_id(item.work_id)
                target.pending_selfplay_count += 1
                added += 1
            except Exception as e:
                logger.error(f"[MinSelfplay] Failed to add item for {target.config_key}: {e}")
            target_idx += 1

        if added > 0:
            logger.info(
                f"[MinSelfplay] All targets met, added {added} minimum selfplay items "
                "to prevent pipeline stall"
            )

        return added
    def _get_dynamic_trickle_count(self) -> int:
        """Calculate dynamic trickle item count based on cluster size.

        January 2026 - Phase 3 Task 5: Dynamic trickle mode scales with cluster size.
        This ensures larger clusters get more work items to keep nodes utilized.

        Scale:
        - 10 nodes → 10 items (minimum)
        - 20 nodes → 20 items
        - 40 nodes → 40 items
        - 100+ nodes → 100 items (capped at max)

        Returns:
            Number of trickle items based on active node count
        """
        if not self.config.trickle_dynamic_scaling:
            return self.config.trickle_min_items

        # Get active node count from cluster status
        active_nodes = self._get_active_node_count()

        # Scale: roughly 1 item per active node, with min/max bounds
        dynamic_count = max(
            self.config.trickle_min_items,  # At least min_items
            min(active_nodes, self.config.trickle_max_items),  # At most max_items
        )

        return dynamic_count
    def _get_active_node_count(self) -> int:
        """Get count of active nodes in the cluster.

        Returns:
            Number of active nodes (at least 1)
        """
        try:
            from app.coordination.cluster_status_monitor import ClusterMonitor

            monitor = ClusterMonitor()
            status = monitor.get_cluster_status(
                include_game_counts=False,
                include_training_status=False,
                include_disk_usage=False,
            )
            return max(1, status.active_nodes)
        except (ImportError, AttributeError, RuntimeError) as e:
            logger.debug(f"[QueuePopulator] Could not get active node count: {e}")
            # Fallback: use dead nodes tracking to estimate
            # If we have 40 configured nodes and 5 dead, return 35
            configured_nodes = 40  # Default cluster size
            dead_count = len(self._dead_nodes)
            return max(1, configured_nodes - dead_count)
    def _get_worker_capacity(self) -> int:
        """Estimate worker capacity based on completion rate and pending work.

        January 10, 2026: Added to fix queue backpressure issues on 40+ node clusters.
        Prevents overfilling the queue by estimating how many items workers can process.

        Returns:
            Estimated number of items workers can process per population cycle.
        """
        if self._work_queue is None:
            return self.config.min_queue_depth

        try:
            # Get queue statistics
            stats = self._work_queue.get_stats()
            pending_count = stats.get("pending_count", 0)
            completed_count = stats.get("completed_count", 0)
            failed_count = stats.get("failed_count", 0)

            # Get completion rate (items per minute) from recent activity
            # Use a sliding window of completions to estimate throughput
            total_completed = completed_count + failed_count
            active_nodes = self._get_active_node_count()

            # Estimate: each active node can process ~2 items per minute on average
            # (accounting for job duration, network overhead, etc.)
            estimated_throughput_per_minute = active_nodes * 2.0

            # Scale by check interval (default 5 seconds)
            items_per_cycle = estimated_throughput_per_minute * (
                self.config.check_interval_seconds / 60.0
            )

            # Add headroom: we want queue to have enough work for 2-3 cycles
            headroom_multiplier = 3.0
            target_queue_size = int(items_per_cycle * headroom_multiplier)

            # Calculate capacity as gap between target and current pending
            capacity = max(0, target_queue_size - pending_count)

            logger.debug(
                f"[QueuePopulator] Worker capacity: {capacity} "
                f"(pending={pending_count}, target={target_queue_size}, "
                f"nodes={active_nodes}, throughput={items_per_cycle:.1f}/cycle)"
            )

            return max(1, capacity)  # Always allow at least 1 item

        except Exception as e:
            logger.debug(f"[QueuePopulator] Could not estimate worker capacity: {e}")
            # Fallback to config-based calculation
            return self.config.min_queue_depth
    def _count_claimable_workers(self) -> int:
        """Count workers with open circuits who could claim work.

        January 13, 2026: Added to fix queue accumulation when circuit breaker
        blocks all claims. Without this check, the populator would add items
        even when no workers could claim them.

        Returns:
            Number of workers with CLOSED or HALF_OPEN circuits (can claim work).
        """
        try:
            from app.coordination.node_circuit_breaker import get_node_circuit_breaker
            from app.config.cluster_config import get_gpu_nodes

            breaker = get_node_circuit_breaker()
            gpu_nodes = get_gpu_nodes()

            claimable = 0
            for node in gpu_nodes:
                if breaker.can_check(node.name):
                    claimable += 1

            return claimable

        except ImportError as e:
            logger.debug(f"[QueuePopulator] Could not check circuit breaker status: {e}")
            # Fallback: assume all non-dead nodes can claim
            active = self._get_active_node_count()
            dead = len(self._dead_nodes)
            return max(0, active - dead)
        except Exception as e:
            logger.warning(f"[QueuePopulator] Error counting claimable workers: {e}")
            return self._get_active_node_count()  # Fallback: assume all can claim
    def _populate_trickle_items(self) -> int:
        """Add minimal items under extreme backpressure (Phase 15.1.2).

        This prevents complete pipeline starvation when backpressure is at
        CRITICAL or STOP levels. We add a small number of selfplay items
        focusing on the highest priority config.

        January 2026 - Phase 3 Task 5: Now uses dynamic count based on cluster size.

        Returns:
            Number of items added (dynamic based on cluster size)
        """
        if self._work_queue is None:
            return 0

        unmet = self.get_unmet_targets()
        if not unmet:
            return 0

        # Sort by curriculum weight (highest priority first)
        scheduler_priorities = self._get_scheduler_priorities()
        if scheduler_priorities:
            unmet.sort(
                key=lambda t: scheduler_priorities.get(t.config_key, 0.0),
                reverse=True,
            )
        else:
            unmet.sort(key=lambda t: t.curriculum_weight, reverse=True)

        added = 0
        # January 2026 - Phase 3: Use dynamic count based on cluster size
        trickle_count = self._get_dynamic_trickle_count()
        items_to_add = min(trickle_count, len(unmet))

        # Add selfplay items for highest priority configs only
        for i in range(items_to_add):
            target = unmet[i % len(unmet)]
            try:
                item = self._create_selfplay_item(target.board_type, target.num_players)
                # Boost priority for trickle items to ensure they get processed
                item.priority = self.config.selfplay_priority + 50
                # January 2026: Force-add for starved configs to bypass backpressure
                force_add = self._should_force_queue_add(target.config_key)
                self._work_queue.add_work(item, force=force_add)
                self._track_queued_work_id(item.work_id)
                added += 1
            except Exception as e:
                logger.error(f"[TrickleMode] Failed to add item: {e}")

        if added > 0:
            active_nodes = self._get_active_node_count()
            logger.info(
                f"[TrickleMode] Added {added} emergency items to prevent starvation "
                f"(dynamic trickle: {trickle_count} for {active_nodes} active nodes)"
            )

        return added
