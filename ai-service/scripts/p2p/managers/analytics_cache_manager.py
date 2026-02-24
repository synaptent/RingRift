"""Analytics Cache Manager for P2P Orchestrator.

January 2026: Phase 1 of P2P Orchestrator Aggressive Decomposition

This module extracts analytics caching and computed metrics from
p2p_orchestrator.py (~1,070 LOC) for better modularity and testability.

Responsibilities:
- Victory type statistics aggregation
- Game analytics (lengths, throughput, openings)
- Training metrics (loss, epochs)
- Holdout validation metrics
- MCTS search statistics
- Tournament matchup matrix
- Model lineage tracking
- Data quality metrics
- Training efficiency calculation
- Rollback condition checking

Usage:
    from scripts.p2p.managers.analytics_cache_manager import (
        AnalyticsCacheManager,
        AnalyticsCacheConfig,
        get_analytics_cache_manager,
    )

    # Create with callbacks
    manager = AnalyticsCacheManager(
        config=AnalyticsCacheConfig(),
        get_ai_service_path=lambda: "/path/to/ai-service",
        is_in_startup_grace_period=lambda: False,
    )

    # Get cached analytics
    victory_stats = await manager.get_victory_type_stats()
    game_analytics = await manager.get_game_analytics_cached()
    training_metrics = await manager.get_training_metrics_cached()
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import re
import shutil
import sqlite3
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from app.core.async_context import safe_create_task
from scripts.p2p.db_helpers import p2p_db_connection

if TYPE_CHECKING:
    from typing import Callable

logger = logging.getLogger(__name__)

# Default Elo rating for new participants
INITIAL_ELO_RATING = 1500.0

# Module-level singleton
_analytics_cache_manager: AnalyticsCacheManager | None = None
_manager_lock = threading.Lock()


def safe_db_connection(db_path: Path):
    """Create a SQLite connection with safe defaults via centralized limiter.

    Note: This is a thin wrapper around p2p_db_connection for backward compatibility.
    Returns a context manager.
    """
    return p2p_db_connection(db_path, timeout=30.0)


def open_jsonl_file(path: Path):
    """Open a JSONL file, handling gzip if needed."""
    import gzip

    if path.suffix == ".gz" or str(path).endswith(".jsonl.gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, encoding="utf-8")


@dataclass
class AnalyticsCacheConfig:
    """Configuration for analytics cache manager.

    Attributes:
        victory_stats_ttl: TTL for victory stats cache in seconds
        game_analytics_ttl: TTL for game analytics cache in seconds
        training_metrics_ttl: TTL for training metrics cache in seconds
        holdout_metrics_ttl: TTL for holdout metrics cache in seconds
        mcts_stats_ttl: TTL for MCTS stats cache in seconds
        matchup_matrix_ttl: TTL for matchup matrix cache in seconds
        model_lineage_ttl: TTL for model lineage cache in seconds
        data_quality_ttl: TTL for data quality cache in seconds
        training_efficiency_ttl: TTL for training efficiency cache in seconds
        data_lookback_hours: Hours to look back for game data
        mcts_lookback_hours: Hours to look back for MCTS data
        matchup_lookback_days: Days to look back for matchup data
    """

    victory_stats_ttl: float = 300.0  # 5 minutes
    game_analytics_ttl: float = 300.0  # 5 minutes
    training_metrics_ttl: float = 120.0  # 2 minutes
    holdout_metrics_ttl: float = 300.0  # 5 minutes
    mcts_stats_ttl: float = 120.0  # 2 minutes
    matchup_matrix_ttl: float = 300.0  # 5 minutes
    model_lineage_ttl: float = 600.0  # 10 minutes
    data_quality_ttl: float = 300.0  # 5 minutes
    training_efficiency_ttl: float = 300.0  # 5 minutes
    data_lookback_hours: int = 24
    mcts_lookback_hours: int = 1
    matchup_lookback_days: int = 7


@dataclass
class CacheEntry:
    """Generic cache entry with data and timestamp."""

    data: Any = None
    timestamp: float = 0.0

    def is_valid(self, ttl: float) -> bool:
        """Check if cache entry is still valid."""
        return time.time() - self.timestamp < ttl


@dataclass
class RollbackCandidate:
    """Represents a model that may need rollback.

    Attributes:
        config: Board config (e.g., "square8_2p")
        rollback_recommended: Whether rollback is recommended
        reasons: List of reasons for recommendation
    """

    config: str
    rollback_recommended: bool = False
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "config": self.config,
            "rollback_recommended": self.rollback_recommended,
            "reasons": self.reasons,
        }


@dataclass
class RollbackResult:
    """Result of a rollback operation.

    Attributes:
        success: Whether rollback succeeded
        config: Board config
        dry_run: Whether this was a dry run
        message: Status message
        details: Additional details
    """

    success: bool = False
    config: str = ""
    dry_run: bool = False
    message: str = ""
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "config": self.config,
            "dry_run": self.dry_run,
            "message": self.message,
            "details": self.details,
        }


class AnalyticsCacheManager:
    """Manages cached analytics and computed metrics for P2P orchestrator.

    Provides caching layer for expensive analytics computations with
    configurable TTLs. Uses callbacks to access orchestrator state
    for loose coupling.
    """

    def __init__(
        self,
        config: AnalyticsCacheConfig | None = None,
        get_ai_service_path: Callable[[], str] | None = None,
        is_in_startup_grace_period: Callable[[], bool] | None = None,
        increment_rollback_counter: Callable[[], None] | None = None,
        send_notification: Callable[..., Any] | None = None,
        node_id: str = "",
    ):
        """Initialize analytics cache manager.

        Args:
            config: Cache configuration
            get_ai_service_path: Callback to get AI service path
            is_in_startup_grace_period: Callback to check startup grace
            increment_rollback_counter: Callback to increment rollback counter
            send_notification: Callback to send notifications
            node_id: Node identifier for notifications
        """
        self.config = config or AnalyticsCacheConfig()
        self._get_ai_service_path = get_ai_service_path or (lambda: ".")
        self._is_in_startup_grace_period = is_in_startup_grace_period or (lambda: False)
        self._increment_rollback_counter = increment_rollback_counter
        self._send_notification = send_notification
        self._node_id = node_id

        # Cache storage
        self._victory_stats_cache = CacheEntry()
        self._game_analytics_cache = CacheEntry()
        self._training_metrics_cache = CacheEntry()
        self._holdout_metrics_cache = CacheEntry()
        self._mcts_stats_cache = CacheEntry()
        self._matchup_matrix_cache = CacheEntry()
        self._model_lineage_cache = CacheEntry()
        self._data_quality_cache = CacheEntry()
        self._training_efficiency_cache = CacheEntry()

        self._lock = threading.Lock()

    # =========================================================================
    # Victory Type Statistics
    # =========================================================================

    async def get_victory_type_stats(self) -> dict[tuple[str, int, str], int]:
        """Aggregate victory types from recent game data.

        Returns dict mapping (board_type, num_players, victory_type) -> count.
        """
        with self._lock:
            if self._victory_stats_cache.is_valid(self.config.victory_stats_ttl):
                return self._victory_stats_cache.data or {}

        if self._is_in_startup_grace_period():
            return {}

        stats = await asyncio.to_thread(self._get_victory_type_stats_sync)

        with self._lock:
            self._victory_stats_cache = CacheEntry(data=stats, timestamp=time.time())

        return stats

    def _get_victory_type_stats_sync(self) -> dict[tuple[str, int, str], int]:
        """Synchronous helper for victory stats (runs in thread pool)."""
        stats: dict[tuple[str, int, str], int] = defaultdict(int)
        ai_root = Path(self._get_ai_service_path())
        data_dirs = [
            ai_root / "data" / "games" / "daemon_sync",
            ai_root / "data" / "selfplay",
        ]
        cutoff_time = time.time() - (self.config.data_lookback_hours * 3600)

        for data_dir in data_dirs:
            if not data_dir.exists():
                continue
            for jsonl_path in data_dir.rglob("*.jsonl"):
                try:
                    if jsonl_path.stat().st_mtime < cutoff_time:
                        continue
                    with open_jsonl_file(jsonl_path) as f:
                        for line in f:
                            try:
                                game = json.loads(line)
                                board_type = game.get("board_type", "unknown")
                                num_players = game.get("num_players", 0)
                                victory_type = game.get("victory_type", "unknown")
                                if victory_type and victory_type != "unknown":
                                    stats[(board_type, num_players, victory_type)] += 1
                            except json.JSONDecodeError:
                                continue
                except (json.JSONDecodeError, AttributeError, OSError):
                    continue

        return dict(stats)

    # =========================================================================
    # Game Analytics
    # =========================================================================

    async def get_game_analytics_cached(self) -> dict[str, Any]:
        """Get game analytics with caching."""
        with self._lock:
            if self._game_analytics_cache.is_valid(self.config.game_analytics_ttl):
                return self._game_analytics_cache.data or {"configs": {}}

        if self._is_in_startup_grace_period():
            return {"configs": {}}

        ai_root = Path(self._get_ai_service_path())
        cutoff = time.time() - (self.config.data_lookback_hours * 3600)
        analytics = await asyncio.to_thread(self._get_game_analytics_sync, ai_root, cutoff)

        with self._lock:
            self._game_analytics_cache = CacheEntry(data=analytics, timestamp=time.time())

        return analytics

    def _get_game_analytics_sync(self, ai_root: Path, cutoff: float) -> dict[str, Any]:
        """Synchronous helper for game analytics (runs in thread pool)."""
        data_dirs = [
            ai_root / "data" / "games" / "daemon_sync",
            ai_root / "data" / "selfplay",
        ]

        game_lengths: dict[str, list[int]] = defaultdict(list)
        games_by_hour: dict[str, dict[int, int]] = defaultdict(lambda: defaultdict(int))
        opening_moves: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

        for data_dir in data_dirs:
            if not data_dir.exists():
                continue
            try:
                jsonl_files = list(data_dir.rglob("*.jsonl"))
            except OSError:
                continue

            for jsonl_path in jsonl_files:
                try:
                    mtime = jsonl_path.stat().st_mtime
                    if mtime < cutoff:
                        continue
                    with open_jsonl_file(jsonl_path) as f:
                        for line in f:
                            try:
                                game = json.loads(line)
                                board_type = game.get("board_type", "unknown")
                                num_players = game.get("num_players", 0)
                                config = f"{board_type}_{num_players}p"

                                length = game.get("length", 0)
                                if length > 0:
                                    game_lengths[config].append(length)

                                hour_bucket = int(mtime // 3600)
                                games_by_hour[config][hour_bucket] += 1

                                moves = game.get("moves", [])
                                if moves and len(moves) >= 1:
                                    first_move = str(moves[0].get("action", ""))[:20]
                                    if first_move:
                                        opening_moves[config][first_move] += 1
                            except json.JSONDecodeError:
                                continue
                except (json.JSONDecodeError, ValueError, AttributeError, OSError):
                    continue

        analytics: dict[str, Any] = {"configs": {}}
        for config in set(list(game_lengths.keys()) + list(games_by_hour.keys())):
            lengths = game_lengths.get(config, [])
            hourly = games_by_hour.get(config, {})
            openings = opening_moves.get(config, {})
            throughput = sum(hourly.values()) / max(len(hourly), 1) if hourly else 0

            analytics["configs"][config] = {
                "avg_length": round(sum(lengths) / len(lengths), 1) if lengths else 0,
                "throughput_per_hour": round(throughput, 1),
                "opening_diversity": len(openings),
            }

        return analytics

    # =========================================================================
    # Training Metrics
    # =========================================================================

    async def get_training_metrics_cached(self) -> dict[str, Any]:
        """Get training metrics with caching."""
        with self._lock:
            if self._training_metrics_cache.is_valid(self.config.training_metrics_ttl):
                return self._training_metrics_cache.data or {"configs": {}}

        ai_root = Path(self._get_ai_service_path())
        logs_dir = ai_root / "logs" / "training"
        metrics = await asyncio.to_thread(self._get_training_metrics_sync, logs_dir)

        with self._lock:
            self._training_metrics_cache = CacheEntry(data=metrics, timestamp=time.time())

        return metrics

    def _get_training_metrics_sync(self, logs_dir: Path) -> dict[str, Any]:
        """Synchronous helper for training metrics (runs in thread pool)."""
        metrics: dict[str, Any] = {"configs": {}}

        if not logs_dir.exists():
            return metrics

        try:
            log_files = sorted(logs_dir.glob("*.log"), key=lambda f: f.stat().st_mtime, reverse=True)[:10]
        except OSError:
            return metrics

        for log_file in log_files:
            try:
                content = log_file.read_text()
                config_match = re.search(r"(square\d+|hexagonal|hex)_(\d+)p", log_file.name)
                if not config_match:
                    continue
                config = f"{config_match.group(1)}_{config_match.group(2)}p"

                loss_pattern = re.compile(r"[Ee]poch\s+(\d+).*?loss[=:]\s*([\d.]+)")
                epochs = []
                for match in loss_pattern.finditer(content):
                    epochs.append({
                        "epoch": int(match.group(1)),
                        "loss": float(match.group(2)),
                    })

                if epochs:
                    metrics["configs"][config] = {
                        "latest_loss": epochs[-1]["loss"],
                        "latest_epoch": epochs[-1]["epoch"],
                    }
            except (OSError, ValueError, KeyError, IndexError, AttributeError):
                continue

        return metrics

    # =========================================================================
    # Holdout Validation Metrics
    # =========================================================================

    async def get_holdout_metrics_cached(self) -> dict[str, Any]:
        """Get holdout validation metrics with caching."""
        with self._lock:
            if self._holdout_metrics_cache.is_valid(self.config.holdout_metrics_ttl):
                return self._holdout_metrics_cache.data or {"configs": {}, "evaluations": [], "summary": {}}

        ai_root = Path(self._get_ai_service_path())
        db_path = ai_root / "data" / "holdouts" / "holdout_validation.db"
        metrics = await asyncio.to_thread(self._get_holdout_metrics_sync, db_path)

        with self._lock:
            self._holdout_metrics_cache = CacheEntry(data=metrics, timestamp=time.time())

        return metrics

    def _get_holdout_metrics_sync(self, db_path: Path) -> dict[str, Any]:
        """Synchronous helper for holdout metrics (runs in thread pool)."""
        metrics: dict[str, Any] = {"configs": {}, "evaluations": [], "summary": {}}

        if not db_path.exists():
            return metrics

        try:
            with safe_db_connection(db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()

                # Get holdout game counts by config
                cursor.execute("""
                    SELECT board_type, num_players, COUNT(*) as game_count, SUM(num_positions) as total_positions
                    FROM holdout_games
                    GROUP BY board_type, num_players
                """)
                for row in cursor.fetchall():
                    config = f"{row['board_type']}_{row['num_players']}p"
                    metrics["configs"][config] = {
                        "holdout_games": row["game_count"],
                        "holdout_positions": row["total_positions"] or 0,
                    }

                # Get latest evaluations per config
                cursor.execute("""
                    SELECT model_path, board_type, num_players, holdout_loss, holdout_accuracy,
                           train_loss, num_samples, evaluated_at, overfit_gap
                    FROM evaluations
                    WHERE id IN (
                        SELECT MAX(id) FROM evaluations
                        GROUP BY board_type, num_players
                    )
                    ORDER BY evaluated_at DESC
                """)
                for row in cursor.fetchall():
                    config = f"{row['board_type']}_{row['num_players']}p"
                    eval_data = {
                        "config": config,
                        "model": row["model_path"],
                        "holdout_loss": row["holdout_loss"],
                        "holdout_accuracy": row["holdout_accuracy"],
                        "train_loss": row["train_loss"],
                        "overfit_gap": row["overfit_gap"],
                        "num_samples": row["num_samples"],
                        "evaluated_at": row["evaluated_at"],
                    }
                    metrics["evaluations"].append(eval_data)
                    if config in metrics["configs"]:
                        metrics["configs"][config].update({
                            "holdout_loss": row["holdout_loss"],
                            "holdout_accuracy": row["holdout_accuracy"],
                            "overfit_gap": row["overfit_gap"],
                        })

                # Get summary stats
                cursor.execute("SELECT COUNT(*) FROM holdout_games")
                metrics["summary"]["total_holdout_games"] = cursor.fetchone()[0]
                cursor.execute("SELECT COUNT(*) FROM evaluations")
                metrics["summary"]["total_evaluations"] = cursor.fetchone()[0]
        except (sqlite3.Error, OSError, KeyError, IndexError, TypeError):
            pass

        return metrics

    # =========================================================================
    # MCTS Statistics
    # =========================================================================

    async def get_mcts_stats_cached(self) -> dict[str, Any]:
        """Get MCTS search statistics with caching."""
        with self._lock:
            if self._mcts_stats_cache.is_valid(self.config.mcts_stats_ttl):
                return self._mcts_stats_cache.data or {"configs": {}, "summary": {}}

        if self._is_in_startup_grace_period():
            return {"configs": {}, "summary": {}}

        ai_root = Path(self._get_ai_service_path())
        cutoff = time.time() - (self.config.mcts_lookback_hours * 3600)
        stats = await asyncio.to_thread(self._get_mcts_stats_sync, ai_root, cutoff)

        with self._lock:
            self._mcts_stats_cache = CacheEntry(data=stats, timestamp=time.time())

        return stats

    def _get_mcts_stats_sync(self, ai_root: Path, cutoff: float) -> dict[str, Any]:
        """Synchronous helper for MCTS stats (runs in thread pool)."""
        stats: dict[str, Any] = {"configs": {}, "summary": {}}

        # Parse selfplay logs for MCTS stats
        logs_dir = ai_root / "logs" / "selfplay"
        if logs_dir.exists():
            try:
                log_files = sorted(logs_dir.glob("*.log"), key=lambda f: f.stat().st_mtime, reverse=True)[:20]
            except OSError:
                log_files = []

            nodes_per_move: list[int] = []
            depth_stats: list[int] = []
            time_per_move: list[float] = []

            for log_file in log_files:
                try:
                    content = log_file.read_text(errors='ignore')
                    for match in re.finditer(r'nodes[_\s]*(?:visited)?[:\s]*(\d+)', content, re.I):
                        nodes_per_move.append(int(match.group(1)))
                    for match in re.finditer(r'(?:search_)?depth[:\s]*(\d+)', content, re.I):
                        depth_stats.append(int(match.group(1)))
                    for match in re.finditer(r'(?:move_)?time[:\s]*([\d.]+)\s*(?:s|ms)?', content, re.I):
                        time_per_move.append(float(match.group(1)))
                except (ValueError, KeyError, IndexError, AttributeError, OSError):
                    continue

            if nodes_per_move:
                stats["summary"]["avg_nodes_per_move"] = sum(nodes_per_move) / len(nodes_per_move)
                stats["summary"]["max_nodes_per_move"] = max(nodes_per_move)
            if depth_stats:
                stats["summary"]["avg_search_depth"] = sum(depth_stats) / len(depth_stats)
                stats["summary"]["max_search_depth"] = max(depth_stats)
            if time_per_move:
                stats["summary"]["avg_time_per_move"] = sum(time_per_move) / len(time_per_move)

        # Also check game JSONL files for MCTS metadata
        data_dirs = [
            ai_root / "data" / "games" / "daemon_sync",
            ai_root / "data" / "selfplay",
        ]

        for data_dir in data_dirs:
            if not data_dir.exists():
                continue
            try:
                jsonl_files = list(data_dir.rglob("*.jsonl"))
            except OSError:
                continue

            for jsonl_path in jsonl_files:
                try:
                    if jsonl_path.stat().st_mtime < cutoff:
                        continue
                    with open_jsonl_file(jsonl_path) as f:
                        for line in f:
                            try:
                                game = json.loads(line)
                                board_type = game.get("board_type", "unknown")
                                num_players = game.get("num_players", 0)
                                config = f"{board_type}_{num_players}p"

                                mcts_data = game.get("mcts_stats", {})
                                if mcts_data:
                                    if config not in stats["configs"]:
                                        stats["configs"][config] = {
                                            "nodes_samples": [],
                                            "depth_samples": [],
                                        }
                                    if "avg_nodes" in mcts_data:
                                        stats["configs"][config]["nodes_samples"].append(mcts_data["avg_nodes"])
                                    if "avg_depth" in mcts_data:
                                        stats["configs"][config]["depth_samples"].append(mcts_data["avg_depth"])
                            except json.JSONDecodeError:
                                continue
                except (json.JSONDecodeError, AttributeError, OSError):
                    continue

        # Compute per-config averages
        for _config, data in stats["configs"].items():
            if data.get("nodes_samples"):
                data["avg_nodes"] = sum(data["nodes_samples"]) / len(data["nodes_samples"])
            if data.get("depth_samples"):
                data["avg_depth"] = sum(data["depth_samples"]) / len(data["depth_samples"])
            data.pop("nodes_samples", None)
            data.pop("depth_samples", None)

        return stats

    # =========================================================================
    # Tournament Matchup Matrix
    # =========================================================================

    async def get_matchup_matrix_cached(self) -> dict[str, Any]:
        """Get head-to-head matchup statistics with caching."""
        with self._lock:
            if self._matchup_matrix_cache.is_valid(self.config.matchup_matrix_ttl):
                return self._matchup_matrix_cache.data or {"matchups": [], "models": [], "configs": {}}

        ai_root = Path(self._get_ai_service_path())
        db_path = ai_root / "data" / "unified_elo.db"
        cutoff = time.time() - (self.config.matchup_lookback_days * 86400)
        matrix = await asyncio.to_thread(self._get_matchup_matrix_sync, db_path, cutoff)

        with self._lock:
            self._matchup_matrix_cache = CacheEntry(data=matrix, timestamp=time.time())

        return matrix

    def _get_matchup_matrix_sync(self, db_path: Path, cutoff: float) -> dict[str, Any]:
        """Synchronous helper for matchup matrix (runs in thread pool)."""
        matrix: dict[str, Any] = {"matchups": [], "models": [], "configs": {}}

        if not db_path.exists():
            return matrix

        try:
            with safe_db_connection(db_path) as conn:
                conn.row_factory = sqlite3.Row

                rows = conn.execute("""
                    SELECT participant_a, participant_b, winner, board_type, num_players,
                           game_length, duration_sec, timestamp
                    FROM match_history
                    WHERE timestamp > ?
                    ORDER BY timestamp DESC
                    LIMIT 10000
                """, (cutoff,)).fetchall()

                h2h: dict[str, dict[str, dict[str, int]]] = defaultdict(lambda: defaultdict(lambda: {"wins": 0, "losses": 0, "draws": 0}))
                models: set[str] = set()
                config_stats: dict[str, Any] = defaultdict(lambda: {"total_matches": 0, "avg_game_length": [], "avg_duration": []})

                for row in rows:
                    a = row["participant_a"]
                    b = row["participant_b"]
                    winner = row["winner"]
                    config = f"{row['board_type']}_{row['num_players']}p"

                    if a and b:
                        models.add(a)
                        models.add(b)

                        if winner == a:
                            h2h[a][b]["wins"] += 1
                            h2h[b][a]["losses"] += 1
                        elif winner == b:
                            h2h[b][a]["wins"] += 1
                            h2h[a][b]["losses"] += 1
                        else:
                            h2h[a][b]["draws"] += 1
                            h2h[b][a]["draws"] += 1

                        config_stats[config]["total_matches"] += 1
                        if row["game_length"]:
                            config_stats[config]["avg_game_length"].append(row["game_length"])
                        if row["duration_sec"]:
                            config_stats[config]["avg_duration"].append(row["duration_sec"])

                # Convert to matchup list
                matchups = []
                for model_a in sorted(models):
                    for model_b in sorted(models):
                        if model_a < model_b:
                            stats = h2h[model_a][model_b]
                            total = stats["wins"] + stats["losses"] + stats["draws"]
                            if total > 0:
                                matchups.append({
                                    "model_a": model_a,
                                    "model_b": model_b,
                                    "a_wins": stats["wins"],
                                    "b_wins": stats["losses"],
                                    "draws": stats["draws"],
                                    "total": total,
                                    "a_win_rate": round(stats["wins"] / total, 3) if total > 0 else 0,
                                })

                # Compute config averages
                for _config, data in config_stats.items():
                    if data["avg_game_length"]:
                        data["avg_game_length"] = round(sum(data["avg_game_length"]) / len(data["avg_game_length"]), 1)
                    else:
                        data["avg_game_length"] = 0
                    if data["avg_duration"]:
                        data["avg_duration"] = round(sum(data["avg_duration"]) / len(data["avg_duration"]), 2)
                    else:
                        data["avg_duration"] = 0

                matrix["matchups"] = matchups
                matrix["models"] = sorted(models)
                matrix["configs"] = dict(config_stats)
                matrix["total_matches"] = sum(c["total_matches"] for c in config_stats.values())
        except (sqlite3.Error, OSError, KeyError, ValueError, TypeError):
            pass

        return matrix

    # =========================================================================
    # Model Lineage Tracking
    # =========================================================================

    async def get_model_lineage_cached(self) -> dict[str, Any]:
        """Get model lineage and ancestry with caching."""
        with self._lock:
            if self._model_lineage_cache.is_valid(self.config.model_lineage_ttl):
                return self._model_lineage_cache.data or {"models": [], "generations": {}, "configs": {}}

        lineage = await asyncio.to_thread(self._get_model_lineage_sync)

        with self._lock:
            self._model_lineage_cache = CacheEntry(data=lineage, timestamp=time.time())

        return lineage

    def _get_model_lineage_sync(self) -> dict[str, Any]:
        """Synchronous helper for model lineage (runs in thread pool)."""
        ai_root = Path(self._get_ai_service_path())
        models_dir = ai_root / "models"
        now = time.time()

        lineage: dict[str, Any] = {"models": [], "generations": {}, "configs": {}}

        if not models_dir.exists():
            return lineage

        try:
            model_files = list(models_dir.glob("**/*.pt")) + list(models_dir.glob("**/*.pth"))

            for model_path in model_files:
                model_name = model_path.stem
                model_stat = model_path.stat()

                config_match = re.search(
                    r"(square\d+|sq\d+|hexagonal|hex)[\W_]*(\d+)p",
                    model_name,
                    re.I
                )
                gen_match = re.search(r"gen(\d+)|v(\d+)|epoch(\d+)", model_name, re.I)

                if config_match:
                    board = config_match.group(1).lower()
                    players = config_match.group(2)
                    if board.startswith("sq") and not board.startswith("square"):
                        board = f"square{board[2:]}"
                    elif board == "hex":
                        board = "hexagonal"
                    config = f"{board}_{players}p"
                else:
                    config = "unknown"
                generation = int(gen_match.group(1) or gen_match.group(2) or gen_match.group(3) or 0) if gen_match else 0

                model_info = {
                    "name": model_name,
                    "path": str(model_path.relative_to(ai_root)),
                    "config": config,
                    "generation": generation,
                    "size_mb": round(model_stat.st_size / 1024 / 1024, 2),
                    "created_at": model_stat.st_mtime,
                    "age_hours": round((now - model_stat.st_mtime) / 3600, 1),
                }
                lineage["models"].append(model_info)

                if config not in lineage["generations"]:
                    lineage["generations"][config] = []
                lineage["generations"][config].append(model_info)

            # Sort models by generation within each config
            for config in lineage["generations"]:
                lineage["generations"][config].sort(key=lambda m: m["generation"])

            # Summary per config
            for config, models in lineage["generations"].items():
                lineage["configs"][config] = {
                    "total_models": len(models),
                    "latest_generation": max(m["generation"] for m in models) if models else 0,
                    "latest_model": models[-1]["name"] if models else None,
                    "total_size_mb": round(sum(m["size_mb"] for m in models), 1),
                }

            lineage["total_models"] = len(lineage["models"])

        except (sqlite3.Error, OSError, KeyError, ValueError, TypeError):
            pass

        return lineage

    # =========================================================================
    # Data Quality Metrics
    # =========================================================================

    async def get_data_quality_cached(self) -> dict[str, Any]:
        """Get data quality metrics with caching."""
        with self._lock:
            if self._data_quality_cache.is_valid(self.config.data_quality_ttl):
                return self._data_quality_cache.data or {"configs": {}, "issues": [], "summary": {}}

        if self._is_in_startup_grace_period():
            return {"configs": {}, "issues": [], "summary": {}}

        ai_root = Path(self._get_ai_service_path())
        cutoff = time.time() - (self.config.data_lookback_hours * 3600)
        quality = await asyncio.to_thread(self._get_data_quality_sync, ai_root, cutoff)

        with self._lock:
            self._data_quality_cache = CacheEntry(data=quality, timestamp=time.time())

        return quality

    def _get_data_quality_sync(self, ai_root: Path, cutoff: float) -> dict[str, Any]:
        """Synchronous helper for data quality metrics (runs in thread pool)."""
        quality: dict[str, Any] = {"configs": {}, "issues": [], "summary": {}}

        data_dirs = [
            ai_root / "data" / "games" / "daemon_sync",
            ai_root / "data" / "selfplay",
        ]

        try:
            config_stats: dict[str, Any] = defaultdict(lambda: {
                "total_games": 0,
                "game_lengths": [],
                "short_games": 0,
                "long_games": 0,
                "stalemates": 0,
                "unique_openings": set(),
                "player_wins": defaultdict(int),
                "parse_errors": 0,
            })

            for data_dir in data_dirs:
                if not data_dir.exists():
                    continue
                try:
                    jsonl_files = list(data_dir.rglob("*.jsonl"))
                except OSError:
                    continue

                for jsonl_path in jsonl_files:
                    try:
                        if jsonl_path.stat().st_mtime < cutoff:
                            continue
                        with open_jsonl_file(jsonl_path) as f:
                            for line in f:
                                try:
                                    game = json.loads(line)
                                    board_type = game.get("board_type", "unknown")
                                    num_players = game.get("num_players", 0)
                                    config = f"{board_type}_{num_players}p"

                                    stats = config_stats[config]
                                    stats["total_games"] += 1

                                    length = game.get("length", 0)
                                    if length > 0:
                                        stats["game_lengths"].append(length)
                                        if length < 10:
                                            stats["short_games"] += 1
                                        elif length > 500:
                                            stats["long_games"] += 1

                                    victory_type = game.get("victory_type", "")
                                    if victory_type == "stalemate":
                                        stats["stalemates"] += 1

                                    moves = game.get("moves", [])
                                    if moves and len(moves) >= 2:
                                        opening = str(moves[0].get("action", ""))[:15] + "-" + str(moves[1].get("action", ""))[:15]
                                        stats["unique_openings"].add(opening)

                                    winner = game.get("winner")
                                    if winner is not None:
                                        stats["player_wins"][winner] += 1

                                except json.JSONDecodeError:
                                    config_stats["unknown"]["parse_errors"] += 1
                    except (OSError, ValueError, KeyError):
                        continue

            # Convert to quality metrics
            issues: list[dict[str, Any]] = []
            for config, stats in config_stats.items():
                total = stats["total_games"]
                if total == 0:
                    continue

                lengths = stats["game_lengths"]
                avg_length = sum(lengths) / len(lengths) if lengths else 0
                length_std = (sum((length - avg_length) ** 2 for length in lengths) / len(lengths)) ** 0.5 if len(lengths) > 1 else 0

                short_rate = stats["short_games"] / total
                long_rate = stats["long_games"] / total
                stalemate_rate = stats["stalemates"] / total
                opening_diversity = len(stats["unique_openings"])

                # Detect issues
                if short_rate > 0.1:
                    issues.append({"config": config, "issue": "high_short_game_rate", "value": round(short_rate * 100, 1), "severity": "warning"})
                if stalemate_rate > 0.3:
                    issues.append({"config": config, "issue": "high_stalemate_rate", "value": round(stalemate_rate * 100, 1), "severity": "warning"})
                if opening_diversity < 5 and total > 50:
                    issues.append({"config": config, "issue": "low_opening_diversity", "value": opening_diversity, "severity": "warning"})

                # Check for player bias
                wins = stats["player_wins"]
                if len(wins) >= 2 and total > 20:
                    max_win_rate = max(wins.values()) / total
                    if max_win_rate > 0.7:
                        issues.append({"config": config, "issue": "player_bias", "value": round(max_win_rate * 100, 1), "severity": "info"})

                quality["configs"][config] = {
                    "total_games": total,
                    "avg_length": round(avg_length, 1),
                    "length_std": round(length_std, 1),
                    "short_game_rate": round(short_rate * 100, 1),
                    "long_game_rate": round(long_rate * 100, 1),
                    "stalemate_rate": round(stalemate_rate * 100, 1),
                    "opening_diversity": opening_diversity,
                    "parse_errors": stats["parse_errors"],
                }

            quality["issues"] = issues
            quality["summary"] = {
                "total_configs": len(quality["configs"]),
                "total_issues": len(issues),
                "critical_issues": len([i for i in issues if i["severity"] == "critical"]),
                "warning_issues": len([i for i in issues if i["severity"] == "warning"]),
            }

        except (OSError, ValueError, KeyError, TypeError):
            pass

        return quality

    # =========================================================================
    # Training Efficiency
    # =========================================================================

    async def get_training_efficiency_cached(self) -> dict[str, Any]:
        """Get training efficiency metrics with caching."""
        with self._lock:
            if self._training_efficiency_cache.is_valid(self.config.training_efficiency_ttl):
                return self._training_efficiency_cache.data or {"configs": {}, "summary": {}, "cost_tracking": {}}

        ai_root = Path(self._get_ai_service_path())
        efficiency = {"configs": {}, "summary": {}, "cost_tracking": {}}

        try:
            # Get Elo history to track improvements
            db_path = ai_root / "data" / "unified_elo.db"
            elo_history = await asyncio.to_thread(
                self._get_elo_history_sync, db_path, time.time() - 86400 * 7
            )

            # Parse training logs for GPU hours
            logs_dir = ai_root / "logs" / "training"
            gpu_hours_per_config = {}

            if logs_dir.exists():
                for log_file in logs_dir.glob("*.log"):
                    try:
                        content = log_file.read_text(errors='ignore')
                        config_match = re.search(r"(square\d+|hex\w*)_(\d+)p", log_file.name)
                        if not config_match:
                            continue
                        config = f"{config_match.group(1)}_{config_match.group(2)}p"

                        duration_match = re.search(r"(?:total[_\s]?time|duration)[:\s]*([\d.]+)\s*(?:s|sec|min|h)", content, re.I)
                        if duration_match:
                            duration = float(duration_match.group(1))
                            if duration > 100:
                                duration = duration / 3600
                            elif duration < 24:
                                duration = duration / 60

                            if config not in gpu_hours_per_config:
                                gpu_hours_per_config[config] = 0
                            gpu_hours_per_config[config] += duration
                    except (ValueError, KeyError, IndexError, AttributeError):
                        continue

            # Calculate efficiency metrics per config
            for config in set(list(elo_history.keys()) + list(gpu_hours_per_config.keys())):
                elo_data = elo_history.get(config, {"ratings": [], "timestamps": []})
                gpu_hours = gpu_hours_per_config.get(config, 0)

                if elo_data["ratings"]:
                    initial_elo = elo_data["ratings"][0] if elo_data["ratings"] else INITIAL_ELO_RATING
                    current_elo = elo_data["ratings"][-1] if elo_data["ratings"] else INITIAL_ELO_RATING
                    elo_gain = current_elo - initial_elo
                else:
                    initial_elo = current_elo = INITIAL_ELO_RATING
                    elo_gain = 0

                elo_per_hour = elo_gain / gpu_hours if gpu_hours > 0 else 0
                estimated_cost = gpu_hours * 2.0

                efficiency["configs"][config] = {
                    "gpu_hours": round(gpu_hours, 2),
                    "initial_elo": round(initial_elo, 1),
                    "current_elo": round(current_elo, 1),
                    "elo_gain": round(elo_gain, 1),
                    "elo_per_gpu_hour": round(elo_per_hour, 2),
                    "estimated_cost_usd": round(estimated_cost, 2),
                    "cost_per_elo_point": round(estimated_cost / max(elo_gain, 1), 2) if elo_gain > 0 else None,
                }

            # Summary
            total_gpu_hours = sum(c.get("gpu_hours", 0) for c in efficiency["configs"].values())
            total_elo_gain = sum(c.get("elo_gain", 0) for c in efficiency["configs"].values())
            total_cost = sum(c.get("estimated_cost_usd", 0) for c in efficiency["configs"].values())

            efficiency["summary"] = {
                "total_gpu_hours": round(total_gpu_hours, 2),
                "total_elo_gain": round(total_elo_gain, 1),
                "total_estimated_cost_usd": round(total_cost, 2),
                "overall_elo_per_gpu_hour": round(total_elo_gain / max(total_gpu_hours, 1), 2),
            }

        except (sqlite3.Error, OSError, ValueError, KeyError, TypeError):
            pass

        with self._lock:
            self._training_efficiency_cache = CacheEntry(data=efficiency, timestamp=time.time())

        return efficiency

    def _get_elo_history_sync(self, db_path: Path, cutoff_time: float) -> dict[str, dict]:
        """Get Elo history from database synchronously."""
        elo_history: dict[str, dict] = {}
        if not db_path.exists():
            return elo_history

        try:
            with safe_db_connection(db_path) as conn:
                rows = conn.execute("""
                    SELECT board_type, num_players, participant_id, rating, timestamp
                    FROM rating_history
                    WHERE timestamp > ?
                    ORDER BY timestamp ASC
                """, (cutoff_time,)).fetchall()

                for row in rows:
                    config = f"{row[0]}_{row[1]}p"
                    if config not in elo_history:
                        elo_history[config] = {"ratings": [], "timestamps": []}
                    elo_history[config]["ratings"].append(row[3])
                    elo_history[config]["timestamps"].append(row[4])
        except (sqlite3.Error, OSError):
            pass

        return elo_history

    # =========================================================================
    # Rollback Checking
    # =========================================================================

    async def check_rollback_conditions(self) -> dict[str, Any]:
        """Check if any models should be rolled back based on metrics."""
        rollback_status: dict[str, Any] = {"candidates": [], "recent_rollbacks": [], "config_status": {}}

        try:
            holdout = await self.get_holdout_metrics_cached()
            ai_root = Path(self._get_ai_service_path())
            db_path = ai_root / "data" / "unified_elo.db"
            elo_data = await asyncio.to_thread(self._get_rollback_elo_data_sync, db_path)

            for config, holdout_data in holdout.get("configs", {}).items():
                status = {"config": config, "rollback_recommended": False, "reasons": []}

                # Check 1: Overfitting (overfit_gap > 0.15)
                overfit_gap = holdout_data.get("overfit_gap", 0)
                if overfit_gap and overfit_gap > 0.15:
                    status["rollback_recommended"] = True
                    status["reasons"].append(f"Overfitting detected: gap={overfit_gap:.3f}")

                # Check 2: Low holdout accuracy (< 60%)
                holdout_acc = holdout_data.get("holdout_accuracy", 1.0)
                if holdout_acc and holdout_acc < 0.6:
                    status["rollback_recommended"] = True
                    status["reasons"].append(f"Low holdout accuracy: {holdout_acc*100:.1f}%")

                # Check 3: Elo regression (dropped > 50 points recently)
                if config in elo_data and len(elo_data[config]) >= 2:
                    recent = elo_data[config][0]["rating"]
                    previous = max(e["rating"] for e in elo_data[config][:10])
                    if previous - recent > 50:
                        status["rollback_recommended"] = True
                        status["reasons"].append(f"Elo regression: {previous:.0f} -> {recent:.0f}")

                rollback_status["config_status"][config] = status
                if status["rollback_recommended"]:
                    rollback_status["candidates"].append(status)

            # Load recent rollback history if exists
            rollback_log = ai_root / "logs" / "rollbacks.json"
            if rollback_log.exists():
                with contextlib.suppress(json.JSONDecodeError, OSError, KeyError, IndexError):
                    rollback_status["recent_rollbacks"] = json.loads(rollback_log.read_text())[-10:]

        except (sqlite3.Error, OSError, ValueError, KeyError, TypeError):
            pass

        return rollback_status

    def _get_rollback_elo_data_sync(self, db_path: Path) -> dict[str, list]:
        """Get Elo data for rollback detection synchronously."""
        elo_data: dict[str, list] = {}
        if not db_path.exists():
            return elo_data

        try:
            with safe_db_connection(db_path) as conn:
                rows = conn.execute("""
                    SELECT board_type, num_players, participant_id, rating, timestamp
                    FROM rating_history
                    ORDER BY timestamp DESC
                    LIMIT 1000
                """).fetchall()

                for row in rows:
                    config = f"{row[0]}_{row[1]}p"
                    if config not in elo_data:
                        elo_data[config] = []
                    elo_data[config].append({"model": row[2], "rating": row[3], "timestamp": row[4]})
        except (sqlite3.Error, OSError):
            pass

        return elo_data

    async def execute_rollback(self, config: str, dry_run: bool = False) -> RollbackResult:
        """Execute a rollback for the given config by restoring previous model.

        Args:
            config: Config string like "square8_2p"
            dry_run: If True, only simulate the rollback without making changes

        Returns:
            RollbackResult with success status and details
        """
        result = RollbackResult(config=config, dry_run=dry_run)

        try:
            ai_root = Path(self._get_ai_service_path())
            models_dir = ai_root / "models"
            archive_dir = models_dir / "archive"
            archive_dir.mkdir(parents=True, exist_ok=True)

            parts = config.rsplit("_", 1)
            if len(parts) != 2 or not parts[1].endswith("p"):
                result.message = f"Invalid config format: {config}"
                return result

            board = parts[0]
            players = parts[1][:-1]

            board_abbrev = board.replace("square", "sq").replace("hexagonal", "hex")
            best_patterns = [
                f"ringrift_best_{board_abbrev}_{players}p.pth",
                f"ringrift_best_{board}_{players}p.pth",
            ]

            current_best = None
            for pattern in best_patterns:
                candidate = models_dir / pattern
                if candidate.exists():
                    current_best = candidate
                    break

            if not current_best:
                result.message = f"No best model found for {config}"
                return result

            checkpoint_dir = models_dir / "checkpoints"
            checkpoints = []
            if checkpoint_dir.exists():
                for ckpt in checkpoint_dir.glob(f"*{board_abbrev}*{players}p*.pth"):
                    try:
                        stat = ckpt.stat()
                        checkpoints.append({
                            "path": ckpt,
                            "mtime": stat.st_mtime,
                            "name": ckpt.name,
                        })
                    except AttributeError:
                        continue

            for archived in archive_dir.glob(f"*{board_abbrev}*{players}p*.pth"):
                try:
                    stat = archived.stat()
                    checkpoints.append({
                        "path": archived,
                        "mtime": stat.st_mtime,
                        "name": archived.name,
                    })
                except AttributeError:
                    continue

            checkpoints.sort(key=lambda x: x["mtime"], reverse=True)
            current_mtime = current_best.stat().st_mtime
            previous_checkpoints = [c for c in checkpoints if abs(c["mtime"] - current_mtime) > 60]

            if not previous_checkpoints:
                result.message = f"No previous checkpoints found for rollback of {config}"
                return result

            rollback_source = previous_checkpoints[0]
            result.details = {
                "current_model": current_best.name,
                "rollback_to": rollback_source["name"],
                "rollback_age_hours": round((time.time() - rollback_source["mtime"]) / 3600, 1),
                "available_checkpoints": len(previous_checkpoints),
            }

            if dry_run:
                result.success = True
                result.message = f"Dry run: Would rollback {current_best.name} to {rollback_source['name']}"
                return result

            timestamp = time.strftime("%Y%m%d_%H%M%S")
            archived_name = f"{current_best.stem}_archived_{timestamp}.pth"
            shutil.copy2(current_best, archive_dir / archived_name)
            shutil.copy2(rollback_source["path"], current_best)

            rollback_log = ai_root / "logs" / "rollbacks.json"
            rollback_log.parent.mkdir(parents=True, exist_ok=True)

            rollback_entry = {
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "config": config,
                "previous_model": current_best.name,
                "rolled_back_to": rollback_source["name"],
                "archived_as": archived_name,
            }

            try:
                existing = json.loads(rollback_log.read_text()) if rollback_log.exists() else []
            except (json.JSONDecodeError, OSError, KeyError, IndexError, AttributeError):
                existing = []

            existing.append(rollback_entry)
            rollback_log.write_text(json.dumps(existing[-100:], indent=2))

            result.success = True
            result.message = f"Successfully rolled back {config} from {current_best.name} to {rollback_source['name']}"

            if self._increment_rollback_counter:
                self._increment_rollback_counter()

            if self._send_notification:
                safe_create_task(self._send_notification(
                    title="Model Rollback Executed",
                    message=f"Rolled back {config} from {current_best.name} to {rollback_source['name']}",
                    level="warning",
                    fields={
                        "Config": config,
                        "Previous": current_best.name,
                        "Restored": rollback_source["name"],
                        "Age": f"{result.details['rollback_age_hours']:.1f}h",
                    },
                    node_id=self._node_id,
                ), name="analytics-notify-rollback")

        except Exception as e:  # noqa: BLE001
            result.message = f"Rollback failed: {e!s}"

        return result

    async def auto_rollback_check(self) -> list[RollbackResult]:
        """Automatically check and execute rollbacks for critical candidates.

        Returns list of executed rollbacks.
        """
        import os
        if os.environ.get("RINGRIFT_AUTO_ROLLBACK", "").lower() not in ("1", "true", "yes"):
            return []

        executed = []
        try:
            status = await self.check_rollback_conditions()
            for candidate in status.get("candidates", []):
                reasons = candidate.get("reasons", [])
                if len(reasons) >= 2 or any("Overfitting" in r for r in reasons):
                    config = candidate["config"]
                    result = await self.execute_rollback(config, dry_run=False)
                    executed.append(result)
                    if result.success:
                        logger.warning(f"[AUTO-ROLLBACK] Executed for {config}: {reasons}")
        except Exception as e:  # noqa: BLE001
            logger.error(f"[AUTO-ROLLBACK] Error: {e}")

        return executed

    # =========================================================================
    # Cache Management
    # =========================================================================

    def invalidate_cache(self, cache_name: str | None = None) -> None:
        """Invalidate one or all caches.

        Args:
            cache_name: Name of cache to invalidate, or None for all
        """
        with self._lock:
            if cache_name is None:
                self._victory_stats_cache = CacheEntry()
                self._game_analytics_cache = CacheEntry()
                self._training_metrics_cache = CacheEntry()
                self._holdout_metrics_cache = CacheEntry()
                self._mcts_stats_cache = CacheEntry()
                self._matchup_matrix_cache = CacheEntry()
                self._model_lineage_cache = CacheEntry()
                self._data_quality_cache = CacheEntry()
                self._training_efficiency_cache = CacheEntry()
            else:
                cache_attr = f"_{cache_name}_cache"
                if hasattr(self, cache_attr):
                    setattr(self, cache_attr, CacheEntry())

    def get_cache_stats(self) -> dict[str, Any]:
        """Get statistics about cache state."""
        now = time.time()
        with self._lock:
            return {
                "victory_stats": {
                    "valid": self._victory_stats_cache.is_valid(self.config.victory_stats_ttl),
                    "age_seconds": round(now - self._victory_stats_cache.timestamp, 1) if self._victory_stats_cache.timestamp else None,
                },
                "game_analytics": {
                    "valid": self._game_analytics_cache.is_valid(self.config.game_analytics_ttl),
                    "age_seconds": round(now - self._game_analytics_cache.timestamp, 1) if self._game_analytics_cache.timestamp else None,
                },
                "training_metrics": {
                    "valid": self._training_metrics_cache.is_valid(self.config.training_metrics_ttl),
                    "age_seconds": round(now - self._training_metrics_cache.timestamp, 1) if self._training_metrics_cache.timestamp else None,
                },
                "holdout_metrics": {
                    "valid": self._holdout_metrics_cache.is_valid(self.config.holdout_metrics_ttl),
                    "age_seconds": round(now - self._holdout_metrics_cache.timestamp, 1) if self._holdout_metrics_cache.timestamp else None,
                },
                "mcts_stats": {
                    "valid": self._mcts_stats_cache.is_valid(self.config.mcts_stats_ttl),
                    "age_seconds": round(now - self._mcts_stats_cache.timestamp, 1) if self._mcts_stats_cache.timestamp else None,
                },
                "matchup_matrix": {
                    "valid": self._matchup_matrix_cache.is_valid(self.config.matchup_matrix_ttl),
                    "age_seconds": round(now - self._matchup_matrix_cache.timestamp, 1) if self._matchup_matrix_cache.timestamp else None,
                },
                "model_lineage": {
                    "valid": self._model_lineage_cache.is_valid(self.config.model_lineage_ttl),
                    "age_seconds": round(now - self._model_lineage_cache.timestamp, 1) if self._model_lineage_cache.timestamp else None,
                },
                "data_quality": {
                    "valid": self._data_quality_cache.is_valid(self.config.data_quality_ttl),
                    "age_seconds": round(now - self._data_quality_cache.timestamp, 1) if self._data_quality_cache.timestamp else None,
                },
                "training_efficiency": {
                    "valid": self._training_efficiency_cache.is_valid(self.config.training_efficiency_ttl),
                    "age_seconds": round(now - self._training_efficiency_cache.timestamp, 1) if self._training_efficiency_cache.timestamp else None,
                },
            }


# =========================================================================
# Module-level Singleton Access
# =========================================================================


def get_analytics_cache_manager() -> AnalyticsCacheManager | None:
    """Get the global analytics cache manager instance."""
    return _analytics_cache_manager


def set_analytics_cache_manager(manager: AnalyticsCacheManager) -> None:
    """Set the global analytics cache manager instance."""
    global _analytics_cache_manager
    with _manager_lock:
        _analytics_cache_manager = manager


def reset_analytics_cache_manager() -> None:
    """Reset the global analytics cache manager instance."""
    global _analytics_cache_manager
    with _manager_lock:
        _analytics_cache_manager = None


def create_analytics_cache_manager(
    config: AnalyticsCacheConfig | None = None,
    get_ai_service_path: Callable[[], str] | None = None,
    is_in_startup_grace_period: Callable[[], bool] | None = None,
    increment_rollback_counter: Callable[[], None] | None = None,
    send_notification: Callable[..., Any] | None = None,
    node_id: str = "",
) -> AnalyticsCacheManager:
    """Create and register a new analytics cache manager.

    Args:
        config: Cache configuration
        get_ai_service_path: Callback to get AI service path
        is_in_startup_grace_period: Callback to check startup grace
        increment_rollback_counter: Callback to increment rollback counter
        send_notification: Callback to send notifications
        node_id: Node identifier for notifications

    Returns:
        The created manager instance
    """
    manager = AnalyticsCacheManager(
        config=config,
        get_ai_service_path=get_ai_service_path,
        is_in_startup_grace_period=is_in_startup_grace_period,
        increment_rollback_counter=increment_rollback_counter,
        send_notification=send_notification,
        node_id=node_id,
    )
    set_analytics_cache_manager(manager)
    return manager
