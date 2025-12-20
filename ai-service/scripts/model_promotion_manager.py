#!/usr/bin/env python
"""Model Promotion Manager - Manages promoted model symlinks and cluster-wide deployment.

This script provides:
1. Stable alias publishing: Creates/updates aliases like `ringrift_best_sq8_2p.pth`
2. Symlink management: Optionally maintains `models/promoted/*_best.pth` links
3. Cluster sync: Rsyncs published aliases to all cluster nodes
4. Optional sandbox config emission (opt-in)

Usage:
    # Update symlinks and config for all promoted models
    python scripts/model_promotion_manager.py --update-all

    # Update specific config
    python scripts/model_promotion_manager.py --update square8 2

    # Sync models to cluster
    python scripts/model_promotion_manager.py --sync-cluster

    # Full pipeline: promote, update symlinks, sync cluster
    python scripts/model_promotion_manager.py --full-pipeline
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add project root to path
SCRIPT_DIR = Path(__file__).parent
AI_SERVICE_ROOT = SCRIPT_DIR.parent
PROJECT_ROOT = AI_SERVICE_ROOT.parent
sys.path.insert(0, str(AI_SERVICE_ROOT))

# Import model lineage tracking
try:
    from scripts.model_lineage import register_model, update_performance
    HAS_LINEAGE = True
except ImportError:
    HAS_LINEAGE = False

# Import unified ELO database for alias sync
try:
    from app.tournament.unified_elo_db import get_elo_database, UnifiedEloRating
    HAS_UNIFIED_ELO = True
except ImportError:
    HAS_UNIFIED_ELO = False

# Import event bus helpers (consolidated imports)
from app.distributed.event_helpers import (
    has_event_bus,
    get_event_bus_safe,
    emit_model_promoted_safe,
    emit_error_safe,
    DataEventType,
    DataEvent,
)
HAS_EVENT_BUS = has_event_bus()

# For backwards compatibility, get the raw functions if available
if HAS_EVENT_BUS:
    from app.distributed.data_events import get_event_bus, emit_model_promoted, emit_error
else:
    get_event_bus = get_event_bus_safe
    emit_model_promoted = emit_model_promoted_safe
    emit_error = emit_error_safe

# Import coordination helpers (consolidated imports)
from app.coordination.helpers import (
    has_coordination,
    get_registry_safe,
    can_spawn_safe,
    OrchestratorRole,
    TaskCoordinator,
    TaskType,
)
HAS_COORDINATION = has_coordination()

# For backwards compatibility
if HAS_COORDINATION:
    from app.coordination import get_registry, can_spawn_safe
else:
    get_registry = get_registry_safe

# Import canonical config helpers
try:
    from app.config.unified_config import (
        get_promotion_elo_threshold,
        get_promotion_min_games,
        get_promotion_check_interval,
        get_rollback_elo_threshold,
        get_rollback_min_games,
    )
    HAS_UNIFIED_CONFIG = True
except ImportError:
    HAS_UNIFIED_CONFIG = False

# Paths
MODELS_DIR = AI_SERVICE_ROOT / "models"
PROMOTED_DIR = MODELS_DIR / "promoted"
PROMOTION_RUNTIME_DIR = AI_SERVICE_ROOT / "runs" / "promotion"
PROMOTED_CONFIG_SEED_PATH = AI_SERVICE_ROOT / "data" / "promoted_models.json"
PROMOTION_LOG_SEED_PATH = AI_SERVICE_ROOT / "data" / "model_promotion_history.json"
PROMOTED_CONFIG_PATH = PROMOTION_RUNTIME_DIR / "promoted_models.json"
ELO_DB_PATH = AI_SERVICE_ROOT / "data" / "unified_elo.db"
PROMOTION_LOG_PATH = PROMOTION_RUNTIME_DIR / "model_promotion_history.json"

# Sandbox config path (TypeScript side)
SANDBOX_CONFIG_PATH = PROJECT_ROOT / "src" / "shared" / "config" / "ai_models.json"

# All board/player configurations
ALL_CONFIGS = [
    ("square8", 2),
    ("square8", 3),
    ("square8", 4),
    ("square19", 2),
    ("square19", 3),
    ("square19", 4),
    ("hexagonal", 2),
    ("hexagonal", 3),
    ("hexagonal", 4),
]

BOARD_ALIAS_TOKENS = {
    "square8": "sq8",
    "square19": "sq19",
    "hexagonal": "hex",
}


@dataclass
class PromotedModel:
    """Information about a promoted model."""
    board_type: str
    num_players: int
    model_path: str  # Relative path from ai-service/models/
    model_id: str
    elo_rating: float
    games_played: int
    promoted_at: str
    symlink_name: str  # e.g., "square8_2p_best.pth"
    alias_id: str  # e.g., "ringrift_best_sq8_2p"
    alias_paths: list[str]  # absolute file paths under ai-service/models


def _get_default_elo_threshold() -> float:
    """Get default Elo threshold from canonical config or fallback."""
    if HAS_UNIFIED_CONFIG:
        return get_promotion_elo_threshold()
    return 20.0


def _get_default_min_games() -> int:
    """Get default min games from canonical config or fallback."""
    if HAS_UNIFIED_CONFIG:
        return get_promotion_min_games()
    return 50


def _get_default_check_interval() -> int:
    """Get default check interval from canonical config or fallback."""
    if HAS_UNIFIED_CONFIG:
        return get_promotion_check_interval()
    return 300


@dataclass
class AutoPromotionConfig:
    """Configuration for automatic model promotion."""
    elo_threshold: float = field(default_factory=_get_default_elo_threshold)
    min_games: int = field(default_factory=_get_default_min_games)
    significance_level: float = 0.05  # Statistical significance requirement
    sync_to_cluster: bool = True  # Sync promoted models to all hosts
    check_interval_seconds: int = field(default_factory=_get_default_check_interval)
    run_regression: bool = True  # Run regression tests before promotion


@dataclass
class PromotionCandidate:
    """A model candidate for automatic promotion."""
    board_type: str
    num_players: int
    model_id: str
    model_path: Path
    elo_rating: float
    games_played: int
    current_best_elo: float
    elo_gain: float
    is_significant: bool


class AutoPromotionTrigger:
    """Automatic promotion trigger for the unified AI improvement loop.

    Monitors Elo ratings and automatically promotes models that:
    1. Beat the current best by a configurable Elo threshold
    2. Have sufficient games played
    3. Pass statistical significance tests
    4. Pass regression tests (optional)

    Usage:
        trigger = AutoPromotionTrigger(config)
        candidates = trigger.check_promotion_candidates()
        for candidate in candidates:
            trigger.execute_promotion(candidate)

    Event-driven usage:
        trigger = AutoPromotionTrigger(config)
        trigger.setup_event_subscriptions()  # Subscribe to TRAINING_COMPLETED, EVALUATION_COMPLETED
    """

    def __init__(self, config: AutoPromotionConfig = None):
        self.config = config or AutoPromotionConfig()
        self._last_check: dict[str, float] = {}  # config -> timestamp
        self._promoted_models: dict[str, PromotedModel] = {}  # config -> last promoted
        self._event_bus = None  # Lazy initialization

    def setup_event_subscriptions(self):
        """Subscribe to events that might trigger promotion checks.

        Call this method to enable event-driven promotion checking.
        Events handled:
        - TRAINING_COMPLETED: Check if newly trained model should be promoted
        - EVALUATION_COMPLETED: Check if Elo improvements warrant promotion
        """
        if not HAS_EVENT_BUS:
            print("[AutoPromotion] Event bus not available, skipping event subscriptions")
            return

        try:
            self._event_bus = get_event_bus()

            # Subscribe to training completion - check if new model beats current best
            self._event_bus.subscribe(
                DataEventType.TRAINING_COMPLETED,
                self._on_training_completed
            )

            # Subscribe to evaluation completion - check if Elo changes warrant promotion
            self._event_bus.subscribe(
                DataEventType.EVALUATION_COMPLETED,
                self._on_evaluation_completed
            )

            print("[AutoPromotion] Event subscriptions set up: TRAINING_COMPLETED, EVALUATION_COMPLETED")
        except Exception as e:
            print(f"[AutoPromotion] Failed to set up event subscriptions: {e}")

    async def _on_training_completed(self, event: DataEvent):
        """Handle TRAINING_COMPLETED events - check for promotion opportunity."""
        try:
            payload = event.payload
            config_key = payload.get('config', '')

            # Parse config_key (e.g., "square8_2p") into board_type and num_players
            if '_' in config_key and config_key.endswith('p'):
                parts = config_key.rsplit('_', 1)
                board_type = parts[0]
                num_players = int(parts[1].rstrip('p'))

                print(f"[AutoPromotion] Training completed for {config_key}, checking for promotion")

                # Check promotion candidates for this config
                candidates = self.check_promotion_candidates(board_type, num_players)
                if candidates:
                    print(f"[AutoPromotion] Found {len(candidates)} promotion candidate(s) after training")
                    for candidate in candidates:
                        self.execute_promotion(candidate, verbose=True)

        except Exception as e:
            print(f"[AutoPromotion] Error handling TRAINING_COMPLETED: {e}")

    async def _on_evaluation_completed(self, event: DataEvent):
        """Handle EVALUATION_COMPLETED events - check if Elo changes warrant promotion."""
        try:
            payload = event.payload
            config_key = payload.get('config', '')
            elo = payload.get('elo', 0)
            win_rate = payload.get('win_rate', 0)

            # Only check if significant Elo improvement
            if elo < 1520:  # Skip if Elo is not notably above baseline
                return

            # Parse config_key
            if '_' in config_key and config_key.endswith('p'):
                parts = config_key.rsplit('_', 1)
                board_type = parts[0]
                num_players = int(parts[1].rstrip('p'))

                # Throttle checks - don't re-check within check_interval
                last_check = self._last_check.get(config_key, 0)
                if time.time() - last_check < self.config.check_interval_seconds:
                    return

                self._last_check[config_key] = time.time()

                print(f"[AutoPromotion] Evaluation completed for {config_key} (Elo={elo:.0f}), checking for promotion")

                candidates = self.check_promotion_candidates(board_type, num_players)
                if candidates:
                    print(f"[AutoPromotion] Found {len(candidates)} promotion candidate(s) after evaluation")
                    for candidate in candidates:
                        self.execute_promotion(candidate, verbose=True)

        except Exception as e:
            print(f"[AutoPromotion] Error handling EVALUATION_COMPLETED: {e}")

    def check_promotion_candidates(
        self,
        board_type: str = None,
        num_players: int = None,
    ) -> list[PromotionCandidate]:
        """Check for models that should be promoted.

        Args:
            board_type: Specific board type to check (None for all)
            num_players: Specific player count to check (None for all)

        Returns:
            List of promotion candidates
        """
        candidates = []

        # Determine configurations to check
        if board_type and num_players:
            configs = [(board_type, num_players)]
        else:
            configs = ALL_CONFIGS

        for bt, np in configs:
            candidate = self._check_config(bt, np)
            if candidate:
                candidates.append(candidate)

        return candidates

    def _check_config(self, board_type: str, num_players: int) -> PromotionCandidate | None:
        """Check a specific configuration for promotion candidates."""
        config_key = f"{board_type}_{num_players}p"

        # Get current best from promoted models
        current_best = self._get_current_best(board_type, num_players)
        current_best_elo = current_best.elo_rating if current_best else 1500.0

        # Get challenger from Elo leaderboard
        challenger = get_best_model_from_elo(board_type, num_players)
        if not challenger:
            return None

        # Check minimum games
        if challenger["games_played"] < self.config.min_games:
            return None

        # Check Elo threshold
        elo_gain = challenger["elo_rating"] - current_best_elo
        if elo_gain < self.config.elo_threshold:
            return None

        # Check if this is the same model already promoted
        if current_best and challenger["model_id"] == current_best.model_id:
            return None

        # Find model file
        model_path = find_model_file(challenger["model_id"])
        if not model_path:
            return None

        # Check statistical significance
        is_significant = self._check_significance(
            challenger["elo_rating"],
            current_best_elo,
            challenger["games_played"],
        )

        if not is_significant and self.config.significance_level < 1.0:
            return None

        return PromotionCandidate(
            board_type=board_type,
            num_players=num_players,
            model_id=challenger["model_id"],
            model_path=model_path,
            elo_rating=challenger["elo_rating"],
            games_played=challenger["games_played"],
            current_best_elo=current_best_elo,
            elo_gain=elo_gain,
            is_significant=is_significant,
        )

    def _get_current_best(self, board_type: str, num_players: int) -> PromotedModel | None:
        """Get the currently promoted model for a configuration."""
        config_key = f"{board_type}_{num_players}p"

        # Check cached promoted models
        if config_key in self._promoted_models:
            return self._promoted_models[config_key]

        # Load from config file
        try:
            if PROMOTED_CONFIG_PATH.exists():
                config_path = PROMOTED_CONFIG_PATH
            elif PROMOTED_CONFIG_SEED_PATH.exists():
                config_path = PROMOTED_CONFIG_SEED_PATH
            else:
                return None

            with open(config_path) as f:
                config = json.load(f)

            models = config.get("models", {})
            if config_key in models:
                m = models[config_key]
                return PromotedModel(
                    board_type=board_type,
                    num_players=num_players,
                    model_path=m.get("path", ""),
                    model_id=m.get("model_id", ""),
                    elo_rating=m.get("elo_rating", 1500.0),
                    games_played=m.get("games_played", 0),
                    promoted_at=m.get("promoted_at", ""),
                    symlink_name=f"{board_type}_{num_players}p_best.pth",
                    alias_id=m.get("alias_id", ""),
                    alias_paths=m.get("alias_paths", []),
                )
        except Exception:
            pass

        return None

    def _check_significance(
        self,
        new_elo: float,
        old_elo: float,
        games: int,
    ) -> bool:
        """Check if Elo difference is statistically significant.

        Uses simplified significance check based on Elo confidence intervals.
        A typical Elo system has ~200 points = 1 standard deviation for
        well-played games.
        """
        if games < 10:
            return False

        # Approximate standard error based on number of games
        # SE decreases with sqrt(games)
        se = 200.0 / (games ** 0.5)

        # Z-score for significance
        z_score = (new_elo - old_elo) / se

        # Common significance levels
        # 0.05 -> z > 1.645
        # 0.01 -> z > 2.326
        if self.config.significance_level >= 0.05:
            return z_score > 1.645
        elif self.config.significance_level >= 0.01:
            return z_score > 2.326
        else:
            return z_score > 2.576  # 0.005 level

    def execute_promotion(
        self,
        candidate: PromotionCandidate,
        verbose: bool = True,
    ) -> PromotedModel | None:
        """Execute promotion for a candidate model.

        Args:
            candidate: The promotion candidate
            verbose: Print progress

        Returns:
            PromotedModel if successful, None otherwise
        """
        config_key = f"{candidate.board_type}_{candidate.num_players}p"

        # Check coordination - ensure no conflicting promotion in progress
        if HAS_COORDINATION and TaskCoordinator is not None:
            import socket
            node_id = socket.gethostname()
            allowed, reason = can_spawn_safe(TaskType.TRAINING, node_id)
            if not allowed:
                if verbose:
                    print(f"[AutoPromotion] {config_key}: Promotion delayed - {reason}")
                return None

        if verbose:
            print(f"[AutoPromotion] Promoting {candidate.model_id} for {config_key}")
            print(f"  Elo: {candidate.elo_rating:.0f} (gain: +{candidate.elo_gain:.0f})")
            print(f"  Games: {candidate.games_played}")

        # Run regression tests if enabled
        if self.config.run_regression:
            if not run_regression_gate(
                candidate.model_path,
                candidate.board_type,
                candidate.num_players,
                verbose,
            ):
                if verbose:
                    print(f"  Regression tests FAILED - skipping promotion")
                return None

        # Publish alias
        alias_id = best_alias_id(candidate.board_type, candidate.num_players)
        alias_paths = publish_best_alias(
            board_type=candidate.board_type,
            num_players=candidate.num_players,
            best_model_path=candidate.model_path,
            best_model_id=candidate.model_id,
            elo_rating=candidate.elo_rating,
            games_played=candidate.games_played,
            verbose=verbose,
        )

        # Create symlink
        symlink_name = f"{candidate.board_type}_{candidate.num_players}p_best.pth"
        create_symlink(candidate.model_path, symlink_name)

        # Create PromotedModel record
        promoted = PromotedModel(
            board_type=candidate.board_type,
            num_players=candidate.num_players,
            model_path=str(candidate.model_path.relative_to(MODELS_DIR)),
            model_id=candidate.model_id,
            elo_rating=candidate.elo_rating,
            games_played=candidate.games_played,
            promoted_at=datetime.utcnow().isoformat() + "Z",
            symlink_name=symlink_name,
            alias_id=alias_id,
            alias_paths=[str(p) for p in alias_paths],
        )

        # Log promotion
        log_promotion(promoted)

        # Update cache
        self._promoted_models[config_key] = promoted

        # Update config file
        all_promoted = list(self._promoted_models.values())
        update_promoted_config(all_promoted)

        # Sync to cluster if enabled
        if self.config.sync_to_cluster:
            if verbose:
                print(f"  Syncing to cluster...")
            sync_to_cluster_ssh([promoted], verbose=verbose)

        # Emit event for pipeline integration
        if HAS_EVENT_BUS:
            import asyncio
            try:
                asyncio.run(emit_model_promoted(
                    model_id=candidate.model_id,
                    config=config_key,
                    elo=candidate.elo_rating,
                    elo_gain=candidate.elo_gain,
                    source="model_promotion_manager.py",
                ))
            except Exception as e:
                if verbose:
                    print(f"  Failed to emit promotion event: {e}")

        if verbose:
            print(f"  Promotion complete!")

        return promoted


def run_auto_promotion_daemon(config: AutoPromotionConfig = None, verbose: bool = True):
    """Run automatic promotion as a background daemon.

    Continuously monitors Elo ratings and promotes models automatically.
    Integrates with the unified AI improvement loop.

    Args:
        config: Auto-promotion configuration
        verbose: Print progress
    """
    import signal

    # Check coordination - warn if unified orchestrator is running
    if HAS_COORDINATION and OrchestratorRole is not None:
        registry = get_registry()
        if registry.is_role_held(OrchestratorRole.UNIFIED_LOOP):
            # If unified orchestrator is running, it handles promotions
            holder = registry.get_role_holder(OrchestratorRole.UNIFIED_LOOP)
            existing_pid = holder.pid if holder else "unknown"
            print(f"[AutoPromotion] WARNING: Unified orchestrator is running (PID {existing_pid})")
            print(f"[AutoPromotion] The orchestrator handles promotions - this daemon may duplicate work")
            print(f"[AutoPromotion] Consider using --full-pipeline instead of --daemon")

    config = config or AutoPromotionConfig()
    trigger = AutoPromotionTrigger(config)

    # Set up event subscriptions for reactive promotion checks
    trigger.setup_event_subscriptions()

    running = True

    def signal_handler(sig, frame):
        nonlocal running
        print("\n[AutoPromotion] Shutdown requested")
        running = False

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    print(f"[AutoPromotion] Starting auto-promotion daemon")
    print(f"  Check interval: {config.check_interval_seconds}s")
    print(f"  Elo threshold: +{config.elo_threshold}")
    print(f"  Min games: {config.min_games}")
    print(f"  Sync to cluster: {config.sync_to_cluster}")
    if HAS_COORDINATION:
        print(f"  Coordination: enabled")
    if HAS_EVENT_BUS:
        print(f"  Event-driven: enabled (TRAINING_COMPLETED, EVALUATION_COMPLETED)")

    iteration = 0
    while running:
        iteration += 1

        if verbose:
            print(f"\n[AutoPromotion] Check #{iteration} at {datetime.now().isoformat()}")

        try:
            candidates = trigger.check_promotion_candidates()

            if candidates:
                print(f"[AutoPromotion] Found {len(candidates)} promotion candidate(s)")
                for candidate in candidates:
                    trigger.execute_promotion(candidate, verbose=verbose)
            elif verbose:
                print(f"[AutoPromotion] No promotion candidates")

        except Exception as e:
            print(f"[AutoPromotion] Error: {e}")
            import traceback
            traceback.print_exc()

            if HAS_EVENT_BUS:
                import asyncio
                try:
                    asyncio.run(emit_error(
                        component="auto_promotion",
                        error=str(e),
                        source="model_promotion_manager.py",
                    ))
                except Exception:
                    pass

        # Wait for next check
        if running:
            try:
                time.sleep(config.check_interval_seconds)
            except KeyboardInterrupt:
                running = False

    print("[AutoPromotion] Stopped")


def best_alias_id(board_type: str, num_players: int) -> str:
    token = BOARD_ALIAS_TOKENS.get(board_type)
    if not token:
        raise ValueError(f"Unsupported board_type for alias: {board_type!r}")
    return f"ringrift_best_{token}_{num_players}p"


def _atomic_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + f".tmp_{uuid.uuid4().hex}")
    shutil.copy2(src, tmp)
    os.replace(tmp, dst)


def _write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp_{uuid.uuid4().hex}")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
    os.replace(tmp, path)


def get_best_model_from_elo(board_type: str, num_players: int) -> dict[str, Any] | None:
    """Get the best model from Elo leaderboard for a given config.

    Supports both unified_elo.db (participant_id schema) and
    elo_leaderboard.db (model_id schema).
    """
    if not ELO_DB_PATH.exists():
        return None

    try:
        conn = sqlite3.connect(str(ELO_DB_PATH))
        cursor = conn.cursor()

        # Check which schema we have
        cursor.execute("PRAGMA table_info(elo_ratings)")
        columns = {col[1] for col in cursor.fetchall()}

        if "participant_id" in columns:
            # unified_elo.db schema - join with participants
            cursor.execute("""
                SELECT p.model_path, e.rating, e.games_played, e.wins, e.losses, e.draws
                FROM elo_ratings e
                JOIN participants p ON e.participant_id = p.participant_id
                WHERE e.board_type = ? AND e.num_players = ?
                  AND p.use_neural_net = 1 AND p.model_path IS NOT NULL
                ORDER BY e.rating DESC
                LIMIT 1
            """, (board_type, num_players))
        else:
            # elo_leaderboard.db schema - direct model_id
            cursor.execute("""
                SELECT model_id, rating, games_played, wins, losses, draws
                FROM elo_ratings
                WHERE board_type = ? AND num_players = ?
                ORDER BY rating DESC
                LIMIT 1
            """, (board_type, num_players))

        row = cursor.fetchone()
        conn.close()

        if row:
            model_id = row[0]
            # Extract model name from path if it's a full path
            if model_id and "/" in model_id:
                model_id = Path(model_id).stem
            return {
                "model_id": model_id,
                "elo_rating": row[1],
                "games_played": row[2],
                "wins": row[3],
                "losses": row[4],
                "draws": row[5],
            }
        return None
    except Exception as e:
        print(f"[model_promotion] Error querying Elo DB: {e}")
        return None


# Rollback configuration - use canonical config with env var override
def _get_rollback_threshold() -> float:
    """Get rollback Elo threshold from env or canonical config."""
    env_val = os.environ.get("RINGRIFT_ROLLBACK_ELO_DROP")
    if env_val:
        return float(env_val)
    if HAS_UNIFIED_CONFIG:
        return get_rollback_elo_threshold()
    return 50.0


def _get_rollback_games() -> int:
    """Get rollback min games from env or canonical config."""
    env_val = os.environ.get("RINGRIFT_ROLLBACK_MIN_GAMES")
    if env_val:
        return int(env_val)
    if HAS_UNIFIED_CONFIG:
        return get_rollback_min_games()
    return 20


ROLLBACK_ELO_DROP_THRESHOLD = _get_rollback_threshold()
ROLLBACK_MIN_GAMES = _get_rollback_games()
ROLLBACK_ENABLED = os.environ.get("RINGRIFT_AUTO_ROLLBACK", "1").lower() in ("1", "true", "yes")


@dataclass
class RollbackCandidate:
    """Model that should be rolled back due to Elo regression."""
    board_type: str
    num_players: int
    current_model: str
    current_elo: float
    previous_model: str
    previous_elo: float
    elo_drop: float
    games_since_promotion: int


def check_for_elo_regression(board_type: str, num_players: int) -> RollbackCandidate | None:
    """Check if current promoted model has significant Elo regression.

    Returns RollbackCandidate if rollback is recommended, None otherwise.
    """
    if not ROLLBACK_ENABLED:
        return None

    try:
        # Get promotion history
        if not PROMOTION_LOG_PATH.exists():
            return None

        with open(PROMOTION_LOG_PATH) as f:
            history = json.load(f)

        # Find recent promotions for this config
        config_key = f"{board_type}_{num_players}p"
        config_promotions = [
            h for h in history
            if h.get("board_type") == board_type and h.get("num_players") == num_players
        ]

        if len(config_promotions) < 2:
            return None  # Need at least 2 promotions to compare

        # Sort by timestamp descending
        config_promotions.sort(key=lambda x: x.get("promoted_at", ""), reverse=True)

        current = config_promotions[0]
        previous = config_promotions[1]

        # Get current Elo from database
        current_model_elo = get_best_model_from_elo(board_type, num_players)
        if not current_model_elo:
            return None

        # Check games since promotion
        games_since = current_model_elo.get("games_played", 0)
        if games_since < ROLLBACK_MIN_GAMES:
            return None  # Not enough games to judge

        # Check Elo drop
        original_elo = current.get("elo_rating", 1500.0)
        current_elo = current_model_elo.get("elo_rating", original_elo)
        elo_drop = original_elo - current_elo

        if elo_drop >= ROLLBACK_ELO_DROP_THRESHOLD:
            return RollbackCandidate(
                board_type=board_type,
                num_players=num_players,
                current_model=current.get("model_path", ""),
                current_elo=current_elo,
                previous_model=previous.get("model_path", ""),
                previous_elo=previous.get("elo_rating", 1500.0),
                elo_drop=elo_drop,
                games_since_promotion=games_since,
            )

        return None

    except Exception as e:
        print(f"[model_promotion] Error checking Elo regression: {e}")
        return None


def perform_rollback(candidate: RollbackCandidate, *, verbose: bool = True) -> bool:
    """Perform a model rollback to the previous version.

    This:
    1. Restores the previous model as the best alias
    2. Logs the rollback
    3. Emits rollback event for pipeline integration
    """
    try:
        if verbose:
            print(f"[model_promotion] Rolling back {candidate.board_type}_{candidate.num_players}p")
            print(f"  Current model: {candidate.current_model} (Elo: {candidate.current_elo:.0f})")
            print(f"  Previous model: {candidate.previous_model} (Elo: {candidate.previous_elo:.0f})")
            print(f"  Elo drop: -{candidate.elo_drop:.0f} over {candidate.games_since_promotion} games")

        # Find previous model file
        previous_path = Path(candidate.previous_model)
        if not previous_path.exists():
            # Try to find it
            model_id = previous_path.stem
            found = find_model_file(model_id)
            if found:
                previous_path = found
            else:
                print(f"[model_promotion] ERROR: Previous model not found: {candidate.previous_model}")
                return False

        # Republish previous model as best
        published = publish_best_alias(
            board_type=candidate.board_type,
            num_players=candidate.num_players,
            best_model_path=previous_path,
            best_model_id=previous_path.stem,
            elo_rating=candidate.previous_elo,
            games_played=0,  # Reset for new baseline
            verbose=verbose,
        )

        if not published:
            print(f"[model_promotion] ERROR: Failed to publish rollback alias")
            return False

        # Log the rollback
        rollback_entry = {
            "type": "rollback",
            "board_type": candidate.board_type,
            "num_players": candidate.num_players,
            "rolled_back_model": candidate.current_model,
            "restored_model": candidate.previous_model,
            "elo_drop": candidate.elo_drop,
            "games_since_promotion": candidate.games_since_promotion,
            "rolled_back_at": datetime.utcnow().isoformat() + "Z",
        }

        # Append to promotion log (rollbacks are logged too)
        try:
            history = []
            if PROMOTION_LOG_PATH.exists():
                with open(PROMOTION_LOG_PATH) as f:
                    history = json.load(f)
            history.append(rollback_entry)
            _write_json_atomic(PROMOTION_LOG_PATH, history)
        except Exception as e:
            print(f"[model_promotion] Warning: Could not log rollback: {e}")

        # Emit rollback event if event bus is available
        if HAS_EVENT_BUS:
            try:
                emit_model_promoted(
                    candidate.previous_model,
                    candidate.board_type,
                    candidate.num_players,
                    candidate.previous_elo,
                    source="rollback",
                )
            except Exception as e:
                print(f"[model_promotion] Event emission failed: {e}")

        if verbose:
            print(f"[model_promotion] Rollback complete: restored {candidate.previous_model}")

        return True

    except Exception as e:
        print(f"[model_promotion] Rollback error: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_and_rollback_all(*, verbose: bool = True) -> list[RollbackCandidate]:
    """Check all configurations for Elo regression and perform rollbacks.

    Returns list of rollback candidates that were processed.
    """
    rollbacks = []

    # Check all board configs
    configs = [
        ("square8", 2), ("square8", 3), ("square8", 4),
        ("hexagonal", 2), ("hexagonal", 3), ("hexagonal", 4),
        ("square19", 2), ("square19", 3), ("square19", 4),
    ]

    for board_type, num_players in configs:
        candidate = check_for_elo_regression(board_type, num_players)
        if candidate:
            if verbose:
                print(f"\n[model_promotion] Elo regression detected for {board_type}_{num_players}p!")

            if perform_rollback(candidate, verbose=verbose):
                rollbacks.append(candidate)

    return rollbacks


def find_model_file(model_id: str) -> Path | None:
    """Find the actual model file for a given model ID."""
    # Prefer exact filenames, then fall back to best-effort prefix matches.
    candidates: list[Path] = [
        MODELS_DIR / f"{model_id}_mps.pth",
        MODELS_DIR / f"{model_id}.pth",
        MODELS_DIR / model_id,
    ]

    matches = sorted(MODELS_DIR.glob(f"{model_id}_*.pth"), key=lambda p: p.stat().st_mtime)
    candidates.extend(reversed(matches[-25:]))  # newest-first, cap for speed

    for candidate in candidates:
        try:
            if candidate.exists() and candidate.is_file() and candidate.stat().st_size > 0:
                return candidate
        except OSError:
            continue

    return None


def sync_alias_elo(
    *,
    alias_id: str,
    source_model_id: str,
    board_type: str,
    num_players: int,
    verbose: bool = False,
) -> bool:
    """Sync ELO rating from source model to alias participant.

    This ensures that when a model is promoted to an alias (e.g., ringrift_best_sq8_2p),
    the alias participant inherits the same ELO rating as the source model. This prevents
    ELO fragmentation where the same model is tracked as two separate participants.

    Args:
        alias_id: The alias participant ID (e.g., "ringrift_best_sq8_2p")
        source_model_id: The source model ID (e.g., "sq8_2p_nn_baseline_20251214_092235")
        board_type: Board type (e.g., "square8")
        num_players: Number of players (e.g., 2)
        verbose: Whether to print status messages

    Returns:
        True if sync was successful, False otherwise
    """
    if not HAS_UNIFIED_ELO:
        if verbose:
            print(f"[model_promotion] Skipping ELO sync - unified_elo_db not available")
        return False

    try:
        db = get_elo_database()

        # Get source model's rating
        source_rating = db.get_rating(source_model_id, board_type, num_players)

        if source_rating.games_played == 0:
            if verbose:
                print(f"[model_promotion] Source model {source_model_id} has no games, skipping ELO sync")
            return False

        # Ensure alias participant exists
        db.ensure_participant(
            alias_id,
            participant_type="alias",
            use_neural_net=True,
            model_path=str(MODELS_DIR / f"{alias_id}.pth"),
            metadata={"source_model_id": source_model_id, "is_alias": True},
        )

        # Create alias rating with same stats as source
        alias_rating = UnifiedEloRating(
            participant_id=alias_id,
            board_type=board_type,
            num_players=num_players,
            rating=source_rating.rating,
            games_played=source_rating.games_played,
            wins=source_rating.wins,
            losses=source_rating.losses,
            draws=source_rating.draws,
            rating_deviation=source_rating.rating_deviation,
        )

        db.update_rating(alias_rating)

        if verbose:
            print(f"[model_promotion] Synced ELO: {alias_id} <- {source_model_id} "
                  f"(rating={source_rating.rating:.1f}, games={source_rating.games_played})")

        return True

    except Exception as e:
        if verbose:
            print(f"[model_promotion] ELO sync failed: {e}")
        return False


def publish_best_alias(
    *,
    board_type: str,
    num_players: int,
    best_model_path: Path,
    best_model_id: str,
    elo_rating: float,
    games_played: int,
    verbose: bool,
) -> list[Path]:
    alias = best_alias_id(board_type, num_players)
    published_at = datetime.utcnow().isoformat() + "Z"

    # Prefer an explicit _mps variant for the MPS alias if present.
    mps_src = MODELS_DIR / f"{best_model_id}_mps.pth"
    if not (mps_src.exists() and mps_src.is_file() and mps_src.stat().st_size > 0):
        mps_src = best_model_path

    cpu_dst = MODELS_DIR / f"{alias}.pth"
    mps_dst = MODELS_DIR / f"{alias}_mps.pth"
    meta_dst = MODELS_DIR / f"{alias}.meta.json"

    _atomic_copy(best_model_path, cpu_dst)
    _atomic_copy(mps_src, mps_dst)
    _write_json_atomic(
        meta_dst,
        {
            "alias_id": alias,
            "board_type": board_type,
            "num_players": int(num_players),
            "source_model_id": best_model_id,
            "source_checkpoint": str(best_model_path),
            "source_checkpoint_mps": str(mps_src),
            "elo_rating": float(elo_rating),
            "games_played": int(games_played),
            "published_at": published_at,
        },
    )

    # Sync ELO rating from source model to alias participant
    # This prevents ELO fragmentation where same model tracked as separate participants
    sync_alias_elo(
        alias_id=alias,
        source_model_id=best_model_id,
        board_type=board_type,
        num_players=num_players,
        verbose=verbose,
    )

    if verbose:
        print(f"[model_promotion] Published alias {alias} -> {best_model_path.name}")

    # Track in lineage database
    if HAS_LINEAGE:
        try:
            model_id = register_model(
                model_path=str(best_model_path),
                board_type=board_type,
                num_players=num_players,
                architecture="neural_net",
            )
            update_performance(model_id, "elo", elo_rating, context="promotion")
            update_performance(model_id, "games_played", games_played, context="promotion")
            if verbose:
                print(f"[model_promotion] Tracked in lineage: {model_id}")
        except Exception as e:
            if verbose:
                print(f"[model_promotion] Lineage tracking failed: {e}")

    return [cpu_dst, mps_dst, meta_dst]


def create_symlink(model_path: Path, symlink_name: str) -> bool:
    """Create or update a symlink in the promoted directory."""
    PROMOTED_DIR.mkdir(parents=True, exist_ok=True)
    symlink_path = PROMOTED_DIR / symlink_name

    # Remove existing symlink if it exists
    if symlink_path.exists() or symlink_path.is_symlink():
        symlink_path.unlink()

    # Create relative symlink
    try:
        rel_path = os.path.relpath(model_path, PROMOTED_DIR)
        symlink_path.symlink_to(rel_path)
        print(f"[model_promotion] Created symlink: {symlink_name} -> {rel_path}")
        return True
    except Exception as e:
        print(f"[model_promotion] Error creating symlink: {e}")
        return False


def update_promoted_config(promoted_models: list[PromotedModel]) -> bool:
    """Update the promoted_models.json config file."""
    config = {
        "updated_at": datetime.utcnow().isoformat() + "Z",
        "models": {
            f"{m.board_type}_{m.num_players}p": {
                "path": f"promoted/{m.symlink_name}",
                "alias_id": m.alias_id,
                "alias_paths": m.alias_paths,
                "model_id": m.model_id,
                "elo_rating": m.elo_rating,
                "games_played": m.games_played,
                "promoted_at": m.promoted_at,
            }
            for m in promoted_models
        }
    }

    try:
        _write_json_atomic(PROMOTED_CONFIG_PATH, config)
        print(f"[model_promotion] Updated config: {PROMOTED_CONFIG_PATH}")
        if PROMOTED_CONFIG_SEED_PATH.exists():
            print(f"[model_promotion] Note: seed file remains at {PROMOTED_CONFIG_SEED_PATH} (runtime writes to runs/)")
        return True
    except Exception as e:
        print(f"[model_promotion] Error updating config: {e}")
        return False


def update_sandbox_config(promoted_models: list[PromotedModel]) -> bool:
    """Update the TypeScript sandbox config with promoted models."""
    config = {
        "_comment": "Auto-generated by model_promotion_manager.py - DO NOT EDIT",
        "updated_at": datetime.utcnow().isoformat() + "Z",
        "models": {
            f"{m.board_type}_{m.num_players}p": {
                "path": f"ai-service/models/{m.alias_id}.pth",
                "elo_rating": m.elo_rating,
            }
            for m in promoted_models
        }
    }

    try:
        SANDBOX_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(SANDBOX_CONFIG_PATH, "w") as f:
            json.dump(config, f, indent=2)
        print(f"[model_promotion] Updated sandbox config: {SANDBOX_CONFIG_PATH}")
        return True
    except Exception as e:
        print(f"[model_promotion] Error updating sandbox config: {e}")
        return False


def log_promotion(promoted_model: PromotedModel) -> None:
    """Append promotion to history log."""
    try:
        history = []
        source_path = PROMOTION_LOG_PATH
        if not source_path.exists() and PROMOTION_LOG_SEED_PATH.exists():
            source_path = PROMOTION_LOG_SEED_PATH
        if source_path.exists():
            with open(source_path) as f:
                history = json.load(f)

        history.append(asdict(promoted_model))

        # Keep last 1000 entries
        if len(history) > 1000:
            history = history[-1000:]

        _write_json_atomic(PROMOTION_LOG_PATH, history)
    except Exception as e:
        print(f"[model_promotion] Warning: Could not log promotion: {e}")


def sync_to_cluster_ssh(
    promoted_models: list[PromotedModel],
    *,
    verbose: bool = True,
    restart_p2p: bool = False,
) -> bool:
    """Sync published best-model aliases to cluster nodes via SSH+rsync."""
    # Read cluster hosts from config (gitignored, local).
    hosts_file = AI_SERVICE_ROOT / "config" / "distributed_hosts.yaml"
    if not hosts_file.exists():
        if verbose:
            print("[model_promotion] No distributed_hosts.yaml found, skipping cluster sync")
        return False

    try:
        import yaml
        with open(hosts_file) as f:
            config = yaml.safe_load(f)

        hosts = config.get("hosts", {})
        success_count = 0
        files: list[Path] = []
        for m in promoted_models:
            for raw in m.alias_paths:
                p = Path(raw)
                if p.exists() and p.is_file() and p.stat().st_size > 0:
                    files.append(p)

        # De-duplicate by resolved path.
        uniq: list[Path] = []
        seen: set[Path] = set()
        for p in files:
            rp = p.resolve()
            if rp in seen:
                continue
            seen.add(rp)
            uniq.append(rp)
        files = uniq

        if not files:
            if verbose:
                print("[model_promotion] No published alias files found to sync")
            return False

        ready_hosts = [
            (name, cfg)
            for name, cfg in hosts.items()
            if cfg.get("status", "ready") == "ready"
        ]
        for host_name, host_config in ready_hosts:
            ssh_host = host_config.get("ssh_host")
            tailscale_ip = host_config.get("tailscale_ip")
            ssh_user = host_config.get("ssh_user", "root")
            ssh_port = host_config.get("ssh_port", 22)
            ssh_key = host_config.get("ssh_key")
            ringrift_path = host_config.get("ringrift_path", "~/ringrift")
            host_candidates: list[str] = []
            for candidate in (tailscale_ip, ssh_host):
                if candidate and candidate not in host_candidates:
                    host_candidates.append(str(candidate))
            if not host_candidates:
                continue

            # Normalize ringrift_path: allow configs that point at .../ai-service.
            ringrift_path_str = str(ringrift_path).rstrip("/")
            if ringrift_path_str.endswith("/ai-service"):
                ringrift_path_str = ringrift_path_str[: -len("/ai-service")]
            remote_models_dir = f"{ringrift_path_str}/ai-service/models"

            def _build_ssh_base_args() -> list[str]:
                args = [
                    "-o", "ConnectTimeout=10",
                    "-o", "BatchMode=yes",
                    "-o", "StrictHostKeyChecking=no",
                ]
                if ssh_port != 22:
                    args.extend(["-p", str(ssh_port)])
                if ssh_key:
                    args.extend(["-i", os.path.expanduser(str(ssh_key))])
                return args

            synced = False
            last_error: str | None = None
            for candidate_host in host_candidates:
                ssh_cmd = ["ssh", *_build_ssh_base_args(), f"{ssh_user}@{candidate_host}"]
                ssh_opts = _build_ssh_base_args()

                method_note = " (via tailscale)" if candidate_host == str(tailscale_ip) else ""
                mkdir_cmd = f"mkdir -p {remote_models_dir}"

                try:
                    mkdir_res = subprocess.run(
                        ssh_cmd + [mkdir_cmd],
                        capture_output=True,
                        timeout=45,
                        text=True,
                    )
                    if mkdir_res.returncode != 0:
                        last_error = (mkdir_res.stderr or mkdir_res.stdout or "").strip()[:200]
                        continue

                    rsync_base = [
                        "rsync",
                        "-az",
                        "--timeout=600",
                        "--partial",
                        "--delay-updates",
                        "-e",
                        "ssh " + " ".join(ssh_opts),
                    ]

                    for attempt in range(2):
                        rsync_cmd = [
                            *rsync_base,
                            *[str(p) for p in files],
                            f"{ssh_user}@{candidate_host}:{remote_models_dir}/",
                        ]
                        rsync_res = subprocess.run(
                            rsync_cmd,
                            capture_output=True,
                            timeout=7200,
                            text=True,
                        )
                        if rsync_res.returncode == 0:
                            synced = True
                            break
                        last_error = (rsync_res.stderr or rsync_res.stdout or "").strip()[:200]
                        # Brief backoff for transient network resets.
                        time.sleep(2.0)

                    if synced:
                        if restart_p2p:
                            restart_cmd = (
                                "if command -v systemctl >/dev/null 2>&1; then "
                                "sudo systemctl restart ringrift-p2p.service ringrift-resilience.service >/dev/null 2>&1 || true; "
                                "sudo systemctl restart ringrift-p2p-orchestrator.service >/dev/null 2>&1 || true; "
                                "fi; "
                                "if command -v launchctl >/dev/null 2>&1; then "
                                "launchctl kickstart -k gui/$(id -u)/com.ringrift.p2p-orchestrator >/dev/null 2>&1 || true; "
                                "launchctl kickstart -k system/com.ringrift.p2p-orchestrator >/dev/null 2>&1 || true; "
                                "fi"
                            )
                            subprocess.run(
                                ssh_cmd + [restart_cmd],
                                capture_output=True,
                                timeout=90,
                                text=True,
                            )

                        if verbose:
                            print(f"[model_promotion] Synced aliases to: {host_name}{method_note}")
                        success_count += 1
                        break
                except subprocess.TimeoutExpired:
                    last_error = "timeout"
                except Exception as e:
                    last_error = f"{type(e).__name__}: {e}"

            if not synced and verbose and last_error:
                print(f"[model_promotion] Failed to sync {host_name}: {last_error}")

        if verbose:
            print(f"[model_promotion] Cluster sync complete: {success_count}/{len(ready_hosts)} ready hosts")
        return success_count > 0
    except Exception as e:
        if verbose:
            print(f"[model_promotion] Cluster sync error: {e}")
        return False


def run_regression_gate(model_path: Path, board_type: str, num_players: int, verbose: bool = True) -> bool:
    """Run regression tests on a model before promotion.

    Returns True if tests pass or regression tests are disabled.
    Set RINGRIFT_REGRESSION_HARD_BLOCK=1 to block promotion on test failure.
    """
    # Skip if regression tests are disabled
    if os.environ.get("RINGRIFT_SKIP_REGRESSION_TESTS", "").lower() in ("1", "true", "yes"):
        if verbose:
            print(f"  Regression tests skipped (RINGRIFT_SKIP_REGRESSION_TESTS=1)")
        return True

    # Check if hard blocking is enabled (default: enabled)
    hard_block = os.environ.get("RINGRIFT_REGRESSION_HARD_BLOCK", "1").lower() in ("1", "true", "yes")

    regression_script = AI_SERVICE_ROOT / "scripts" / "model_regression_tests.py"
    if not regression_script.exists():
        if verbose:
            print(f"  Regression tests skipped (script not found)")
        return True

    try:
        # Run minimal regression tests (fewer games for speed)
        result = subprocess.run(
            [
                sys.executable,
                str(regression_script),
                "--model", str(model_path),
                "--games", "5",  # Quick validation
                "--threshold", "0.3",  # Lower threshold for fast check
            ],
            cwd=str(AI_SERVICE_ROOT),
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout
            env={**os.environ, "PYTHONPATH": str(AI_SERVICE_ROOT)},
        )

        passed = result.returncode == 0
        if verbose:
            if passed:
                print(f"  Regression tests: PASSED")
            else:
                if hard_block:
                    print(f"  Regression tests: FAILED - blocking promotion")
                    _log_regression_failure(model_path, board_type, num_players, result)
                else:
                    print(f"  Regression tests: FAILED (set RINGRIFT_REGRESSION_HARD_BLOCK=1 to block)")

        # Block promotion if tests failed and hard_block is enabled
        if not passed and hard_block:
            return False
        return True

    except subprocess.TimeoutExpired:
        if verbose:
            print(f"  Regression tests: TIMEOUT")
        # Block on timeout if hard_block is enabled
        return not hard_block
    except Exception as e:
        if verbose:
            print(f"  Regression tests error: {e}")
        return True


def _log_regression_failure(model_path: Path, board_type: str, num_players: int, result) -> None:
    """Log detailed regression failure for debugging."""
    from datetime import datetime
    log_dir = AI_SERVICE_ROOT / "logs" / "regression_failures"
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"regression_{board_type}_{num_players}p_{timestamp}.log"

    with open(log_file, "w") as f:
        f.write(f"Model: {model_path}\n")
        f.write(f"Board: {board_type}, Players: {num_players}\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Return code: {result.returncode}\n")
        f.write(f"\n--- stdout ---\n{result.stdout}\n")
        f.write(f"\n--- stderr ---\n{result.stderr}\n")

    print(f"  Failure details logged to: {log_file}")


def update_all_promotions(
    min_games: int = 20,
    *,
    verbose: bool = True,
    update_sandbox: bool = False,
    run_regression: bool = True,
) -> list[PromotedModel]:
    """Publish best-model aliases (and optional symlinks/config) for all configs.

    Args:
        min_games: Minimum games required for promotion
        verbose: Print progress
        update_sandbox: Update TypeScript sandbox config
        run_regression: Run regression tests before promotion
    """
    promoted_models = []

    for board_type, num_players in ALL_CONFIGS:
        if verbose:
            print(f"\n[model_promotion] Checking {board_type} {num_players}p...")

        best = get_best_model_from_elo(board_type, num_players)
        if not best:
            if verbose:
                print(f"  No Elo data available")
            continue

        if best["games_played"] < min_games:
            if verbose:
                print(f"  Best model has only {best['games_played']} games (min: {min_games})")
            continue

        model_path = find_model_file(best["model_id"])
        if not model_path:
            if verbose:
                print(f"  Model file not found: {best['model_id']}")
            continue

        # Run regression tests before promotion
        if run_regression:
            if not run_regression_gate(model_path, board_type, num_players, verbose):
                if verbose:
                    print(f"  Skipping promotion due to failed regression tests")
                continue

        alias_id = best_alias_id(board_type, int(num_players))
        alias_paths = publish_best_alias(
            board_type=board_type,
            num_players=int(num_players),
            best_model_path=model_path,
            best_model_id=best["model_id"],
            elo_rating=best["elo_rating"],
            games_played=best["games_played"],
            verbose=verbose,
        )

        symlink_name = f"{board_type}_{num_players}p_best.pth"

        if create_symlink(model_path, symlink_name):
            promoted = PromotedModel(
                board_type=board_type,
                num_players=num_players,
                model_path=str(model_path.relative_to(MODELS_DIR)),
                model_id=best["model_id"],
                elo_rating=best["elo_rating"],
                games_played=best["games_played"],
                promoted_at=datetime.utcnow().isoformat() + "Z",
                symlink_name=symlink_name,
                alias_id=alias_id,
                alias_paths=[str(p) for p in alias_paths],
            )
            promoted_models.append(promoted)
            log_promotion(promoted)

            if verbose:
                print(f"  Promoted: {best['model_id']} (Elo: {best['elo_rating']:.0f}, games: {best['games_played']})")

    if promoted_models:
        update_promoted_config(promoted_models)
        if update_sandbox:
            update_sandbox_config(promoted_models)

    return promoted_models


def run_full_pipeline(
    min_games: int = 20,
    sync_cluster: bool = True,
    update_sandbox: bool = False,
    restart_p2p: bool = False,
    verbose: bool = True,
    run_regression: bool = True,
) -> bool:
    """Run the full promotion pipeline: update symlinks, config, and sync cluster."""
    if verbose:
        print("[model_promotion] Starting full promotion pipeline...")

    # Step 1: Update all promotions
    promoted = update_all_promotions(
        min_games=min_games, verbose=verbose, update_sandbox=update_sandbox, run_regression=run_regression
    )

    if not promoted:
        if verbose:
            print("[model_promotion] No models promoted")
        return False

    if verbose:
        print(f"\n[model_promotion] Promoted {len(promoted)} models")

    # Step 2: Sync to cluster
    if sync_cluster:
        if verbose:
            print("\n[model_promotion] Syncing to cluster...")
        sync_to_cluster_ssh(promoted, verbose=verbose, restart_p2p=restart_p2p)

    if verbose:
        print("\n[model_promotion] Pipeline complete!")

    return True


def sync_staging_artifacts(
    *,
    restart: bool,
    validate_health: bool,
    fail_on_missing: bool,
    verbose: bool,
) -> bool:
    """Best-effort sync of published artifacts to staging via SSH."""
    host = os.environ.get("RINGRIFT_STAGING_SSH_HOST")
    remote_root = os.environ.get("RINGRIFT_STAGING_ROOT")
    if not host or not remote_root:
        if verbose:
            print(
                "[model_promotion] Staging sync requested but missing "
                "RINGRIFT_STAGING_SSH_HOST / RINGRIFT_STAGING_ROOT"
            )
        return False

    cmd = [sys.executable, "scripts/sync_staging_ai_artifacts.py"]
    if restart:
        cmd.append("--restart")
    if validate_health:
        cmd.append("--validate-health")
    if fail_on_missing:
        cmd.append("--fail-on-missing")

    proc = subprocess.run(
        cmd,
        cwd=str(AI_SERVICE_ROOT),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if verbose and proc.stdout:
        print(proc.stdout)
    return proc.returncode == 0


def main():
    parser = argparse.ArgumentParser(
        description="Model Promotion Manager - Manage promoted model symlinks and deployment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--update-all",
        action="store_true",
        help="Update symlinks and config for all configurations",
    )
    parser.add_argument(
        "--update",
        nargs=2,
        metavar=("BOARD", "PLAYERS"),
        help="Update specific configuration (e.g., --update square8 2)",
    )
    parser.add_argument(
        "--sync-cluster",
        action="store_true",
        help="Sync models to all cluster nodes",
    )
    parser.add_argument(
        "--restart-p2p",
        action="store_true",
        help="After syncing, best-effort restart of the P2P orchestrator on each host.",
    )
    parser.add_argument(
        "--full-pipeline",
        action="store_true",
        help="Run full pipeline: update all, sync cluster",
    )
    parser.add_argument(
        "--min-games",
        type=int,
        default=20,
        help="Minimum games required for promotion (default: 20)",
    )
    parser.add_argument(
        "--update-sandbox-config",
        action="store_true",
        help="Also write src/shared/config/ai_models.json (opt-in).",
    )
    parser.add_argument(
        "--sync-staging",
        action="store_true",
        help="After publishing aliases, sync the promoted artifacts to staging via scripts/sync_staging_ai_artifacts.py.",
    )
    parser.add_argument(
        "--sync-staging-no-restart",
        action="store_true",
        help="When used with --sync-staging, do not restart docker compose services after syncing.",
    )
    parser.add_argument(
        "--sync-staging-validate-health",
        action="store_true",
        help="When used with --sync-staging, validate /internal/ladder/health on staging after sync.",
    )
    parser.add_argument(
        "--sync-staging-fail-on-missing",
        action="store_true",
        help="When used with --sync-staging-validate-health, exit non-zero if staging reports missing artifacts.",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress verbose output",
    )
    parser.add_argument(
        "--no-regression",
        action="store_true",
        help="Skip regression tests before promotion",
    )

    # Auto-promotion daemon mode (for unified AI loop integration)
    parser.add_argument(
        "--daemon",
        action="store_true",
        help="Run as auto-promotion daemon (continuous monitoring)",
    )
    parser.add_argument(
        "--check-interval",
        type=int,
        default=300,
        help="Seconds between promotion checks in daemon mode (default: 300)",
    )
    parser.add_argument(
        "--elo-threshold",
        type=float,
        default=20.0,
        help="Elo gain required for auto-promotion (default: 20)",
    )
    parser.add_argument(
        "--check-candidates",
        action="store_true",
        help="Check for promotion candidates and exit (one-shot mode)",
    )
    parser.add_argument(
        "--auto-promote",
        action="store_true",
        help="Automatically promote any eligible candidates (one-shot mode)",
    )

    # Rollback options
    parser.add_argument(
        "--check-rollback",
        action="store_true",
        help="Check for Elo regression and rollback if needed (one-shot mode)",
    )
    parser.add_argument(
        "--rollback-threshold",
        type=float,
        default=50.0,
        help="Elo drop threshold for rollback (default: 50)",
    )
    parser.add_argument(
        "--rollback-min-games",
        type=int,
        default=20,
        help="Minimum games before rollback is considered (default: 20)",
    )

    args = parser.parse_args()
    verbose = not args.quiet
    run_regression = not args.no_regression

    # Handle daemon mode
    if args.daemon:
        config = AutoPromotionConfig(
            elo_threshold=args.elo_threshold,
            min_games=args.min_games,
            check_interval_seconds=args.check_interval,
            sync_to_cluster=True,
            run_regression=run_regression,
        )
        run_auto_promotion_daemon(config, verbose=verbose)
        return

    # Handle one-shot candidate check
    if args.check_candidates or args.auto_promote:
        config = AutoPromotionConfig(
            elo_threshold=args.elo_threshold,
            min_games=args.min_games,
            run_regression=run_regression,
        )
        trigger = AutoPromotionTrigger(config)
        candidates = trigger.check_promotion_candidates()

        if not candidates:
            print("[AutoPromotion] No promotion candidates found")
        else:
            print(f"[AutoPromotion] Found {len(candidates)} candidate(s):")
            for c in candidates:
                print(f"  {c.board_type}_{c.num_players}p: {c.model_id}")
                print(f"    Elo: {c.elo_rating:.0f} (gain: +{c.elo_gain:.0f})")
                print(f"    Games: {c.games_played}")

            if args.auto_promote:
                print("\n[AutoPromotion] Executing promotions...")
                for candidate in candidates:
                    trigger.execute_promotion(candidate, verbose=verbose)
        return

    # Handle rollback check
    if args.check_rollback:
        # Set environment variables from args
        global ROLLBACK_ELO_DROP_THRESHOLD, ROLLBACK_MIN_GAMES
        ROLLBACK_ELO_DROP_THRESHOLD = args.rollback_threshold
        ROLLBACK_MIN_GAMES = args.rollback_min_games

        if verbose:
            print("[Rollback] Checking for Elo regression...")
            print(f"  Threshold: -{args.rollback_threshold} Elo")
            print(f"  Min games: {args.rollback_min_games}")

        rollbacks = check_and_rollback_all(verbose=verbose)

        if not rollbacks:
            print("[Rollback] No Elo regressions detected")
        else:
            print(f"\n[Rollback] Performed {len(rollbacks)} rollback(s)")
            for r in rollbacks:
                print(f"  {r.board_type}_{r.num_players}p: rolled back to {r.previous_model}")
        return

    promoted_models: list[PromotedModel] = []
    did_publish = False

    if args.full_pipeline:
        promoted_models = update_all_promotions(
            min_games=args.min_games,
            verbose=verbose,
            update_sandbox=bool(args.update_sandbox_config),
            run_regression=run_regression,
        )
        did_publish = bool(promoted_models)
        if did_publish and verbose:
            print(f"\n[model_promotion] Promoted {len(promoted_models)} models")

        if did_publish:
            if verbose:
                print("\n[model_promotion] Syncing to cluster...")
            sync_to_cluster_ssh(promoted_models, verbose=verbose, restart_p2p=bool(args.restart_p2p))

            if verbose:
                print("\n[model_promotion] Pipeline complete!")
    elif args.update_all:
        promoted_models = update_all_promotions(
            min_games=args.min_games,
            verbose=verbose,
            update_sandbox=bool(args.update_sandbox_config),
            run_regression=run_regression,
        )
        did_publish = bool(promoted_models)
    elif args.sync_cluster:
        promoted_models = update_all_promotions(
            min_games=args.min_games,
            verbose=False,
            update_sandbox=False,
            run_regression=run_regression,
        )
        did_publish = bool(promoted_models)
        sync_to_cluster_ssh(promoted_models, verbose=verbose, restart_p2p=bool(args.restart_p2p))
    elif args.update:
        board_type, num_players = args.update
        promoted_models = update_all_promotions(
            min_games=args.min_games,
            verbose=verbose,
            update_sandbox=bool(args.update_sandbox_config),
            run_regression=run_regression,
        )
        did_publish = bool(promoted_models)
        for p in promoted_models:
            if p.board_type == board_type and p.num_players == int(num_players):
                print(f"Updated: {p.symlink_name}")
    else:
        parser.print_help()

    if args.sync_staging and did_publish:
        restart = not bool(args.sync_staging_no_restart)
        validate = bool(args.sync_staging_validate_health)
        fail_on_missing = bool(args.sync_staging_fail_on_missing)
        if fail_on_missing:
            validate = True
        ok = sync_staging_artifacts(
            restart=restart,
            validate_health=validate,
            fail_on_missing=fail_on_missing,
            verbose=verbose,
        )
        if not ok:
            raise SystemExit(3)


if __name__ == "__main__":
    main()
