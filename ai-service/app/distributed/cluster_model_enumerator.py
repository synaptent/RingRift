"""Cluster Model Enumerator - Enumerate ALL models across the entire cluster.

This module provides unified model discovery across all cluster nodes,
tracking which models exist where, their types, and evaluation status.
Essential for comprehensive evaluation of all models under all harnesses.

Key features:
1. Query all nodes for model inventory via P2P endpoints
2. Merge results and deduplicate by model hash
3. Cross-reference with EloService to find unevaluated combinations
4. Generate all valid (model, harness, config) combinations

Usage:
    from app.distributed.cluster_model_enumerator import (
        ClusterModelEnumerator,
        get_cluster_model_enumerator,
        ModelInfo,
    )

    enumerator = get_cluster_model_enumerator()

    # Get all models across cluster
    models = await enumerator.enumerate_all_models()

    # Get unevaluated combinations
    unevaluated = enumerator.get_unevaluated_combinations()

    # Get stale combinations (not evaluated in 7 days)
    stale = enumerator.get_stale_combinations(max_age_days=7)
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import re
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import aiohttp

logger = logging.getLogger(__name__)

__all__ = [
    "ModelInfo",
    "EvaluationStatus",
    "ClusterModelEnumerator",
    "get_cluster_model_enumerator",
    "reset_cluster_model_enumerator",
]


# Harness compatibility matrix — lists harnesses used for Elo evaluation.
# Feb 28, 2026: Reduced from 4-5 harnesses per model to just gumbel_mcts.
# Other harnesses (policy_only, descent, minimax, maxn, brs) are BASELINE
# opponents used WITHIN the gauntlet, not separate evaluation harnesses.
# The old matrix created 4-5x more gauntlet work items than needed (5,241
# pending items on Feb 28 from 12 configs × 5 harnesses × many cycles).
HARNESS_COMPATIBILITY: dict[str, dict[str, list[str]]] = {
    "nn": {
        "2": ["gumbel_mcts"],
        "3": ["gumbel_mcts"],
        "4": ["gumbel_mcts"],
    },
    "nnue": {
        "2": ["minimax"],
        "3": [],
        "4": [],
    },
    "nnue_mp": {
        "2": [],
        "3": ["maxn"],
        "4": ["maxn"],
    },
}


@dataclass
class EvaluationStatus:
    """Evaluation status for a model under a specific harness."""

    harness: str
    last_evaluated: datetime | None
    elo_rating: float | None
    games_played: int
    config_key: str


@dataclass
class ModelInfo:
    """Information about a model discovered in the cluster."""

    model_path: str
    model_hash: str
    model_type: str  # "nn", "nnue", "nnue_mp"
    board_type: str
    num_players: int
    node_ids: list[str] = field(default_factory=list)
    size_bytes: int = 0
    modified_time: float = 0.0
    architecture: str = ""
    evaluations: list[EvaluationStatus] = field(default_factory=list)

    @property
    def config_key(self) -> str:
        """Return canonical config key like 'hex8_2p'."""
        return f"{self.board_type}_{self.num_players}p"

    @property
    def compatible_harnesses(self) -> list[str]:
        """Return list of compatible harnesses for this model."""
        player_key = str(self.num_players)
        return HARNESS_COMPATIBILITY.get(self.model_type, {}).get(player_key, [])

    @property
    def harnesses_evaluated(self) -> set[str]:
        """Return set of harnesses that have been evaluated."""
        return {e.harness for e in self.evaluations}

    @property
    def unevaluated_harnesses(self) -> list[str]:
        """Return harnesses that haven't been evaluated."""
        return [h for h in self.compatible_harnesses if h not in self.harnesses_evaluated]


@dataclass
class EvaluationCombination:
    """A (model, harness, config) combination for evaluation."""

    model_path: str
    model_hash: str
    harness: str
    config_key: str
    board_type: str
    num_players: int
    last_evaluated: datetime | None = None
    priority: float = 0.0

    @property
    def is_stale(self) -> bool:
        """Check if evaluation is stale (>7 days old or never evaluated)."""
        if self.last_evaluated is None:
            return True
        return (datetime.now() - self.last_evaluated).days > 7


class ClusterModelEnumerator:
    """Enumerate ALL models across the entire cluster.

    Queries all P2P nodes for their model inventory and cross-references
    with EloService to identify unevaluated (model, harness, config) combinations.
    """

    _instance: ClusterModelEnumerator | None = None
    _lock = threading.Lock()

    def __init__(
        self,
        p2p_port: int = 8770,
        cache_ttl_seconds: float = 300.0,
        request_timeout: float = 10.0,
    ):
        """Initialize the enumerator.

        Args:
            p2p_port: Port for P2P HTTP endpoints
            cache_ttl_seconds: How long to cache enumeration results
            request_timeout: Timeout for individual node requests
        """
        self.p2p_port = p2p_port
        self.cache_ttl_seconds = cache_ttl_seconds
        self.request_timeout = request_timeout

        # Cached state
        self._models_cache: dict[str, ModelInfo] = {}  # model_hash -> ModelInfo
        self._cache_timestamp: float = 0.0
        self._elo_ratings_cache: dict[str, dict] = {}  # composite_id -> rating info
        self._cache_lock = threading.RLock()

        # Local paths
        self._local_models_dir = Path(
            os.environ.get("RINGRIFT_MODELS_PATH", "models")
        )

    @classmethod
    def get_instance(cls) -> ClusterModelEnumerator:
        """Get singleton instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton instance (for testing)."""
        with cls._lock:
            cls._instance = None

    async def enumerate_all_models(
        self, force_refresh: bool = False
    ) -> list[ModelInfo]:
        """Query all nodes and return unified model list.

        Args:
            force_refresh: Force refresh even if cache is valid

        Returns:
            List of ModelInfo with all discovered models
        """
        # Check cache validity
        if not force_refresh and self._is_cache_valid():
            with self._cache_lock:
                return list(self._models_cache.values())

        # Get list of alive peers
        peers = await self._get_alive_peers()
        logger.info(f"[ClusterModelEnumerator] Querying {len(peers)} peers for models")

        # Query all peers in parallel
        tasks = [self._query_node_models(peer) for peer in peers]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Merge results
        models_by_hash: dict[str, ModelInfo] = {}

        for peer, result in zip(peers, results):
            if isinstance(result, Exception):
                logger.warning(f"[ClusterModelEnumerator] Failed to query {peer}: {result}")
                continue

            for model in result:
                if model.model_hash in models_by_hash:
                    # Merge node_ids
                    existing = models_by_hash[model.model_hash]
                    for node_id in model.node_ids:
                        if node_id not in existing.node_ids:
                            existing.node_ids.append(node_id)
                else:
                    models_by_hash[model.model_hash] = model

        # Also include local models
        local_models = await self._discover_local_models()
        for model in local_models:
            if model.model_hash in models_by_hash:
                existing = models_by_hash[model.model_hash]
                for node_id in model.node_ids:
                    if node_id not in existing.node_ids:
                        existing.node_ids.append(node_id)
            else:
                models_by_hash[model.model_hash] = model

        # Load evaluation status from EloService
        await self._load_evaluation_status(list(models_by_hash.values()))

        # Update cache
        with self._cache_lock:
            self._models_cache = models_by_hash
            self._cache_timestamp = time.time()

        logger.info(
            f"[ClusterModelEnumerator] Enumerated {len(models_by_hash)} unique models "
            f"across {len(peers)} peers"
        )

        return list(models_by_hash.values())

    def get_unevaluated_combinations(self) -> list[EvaluationCombination]:
        """Return (model, harness, config) tuples not yet evaluated.

        Returns:
            List of EvaluationCombination objects for unevaluated pairs
        """
        combinations = []

        with self._cache_lock:
            for model in self._models_cache.values():
                for harness in model.unevaluated_harnesses:
                    combinations.append(
                        EvaluationCombination(
                            model_path=model.model_path,
                            model_hash=model.model_hash,
                            harness=harness,
                            config_key=model.config_key,
                            board_type=model.board_type,
                            num_players=model.num_players,
                            last_evaluated=None,
                            priority=1.0,  # Highest priority for unevaluated
                        )
                    )

        # Sort by priority (highest first)
        combinations.sort(key=lambda x: -x.priority)
        return combinations

    def get_stale_combinations(
        self, max_age_days: int = 7
    ) -> list[EvaluationCombination]:
        """Return combinations not evaluated in max_age_days.

        Args:
            max_age_days: Maximum age in days before considering stale

        Returns:
            List of EvaluationCombination objects for stale pairs
        """
        combinations = []
        cutoff = datetime.now() - timedelta(days=max_age_days)

        with self._cache_lock:
            for model in self._models_cache.values():
                for eval_status in model.evaluations:
                    if eval_status.last_evaluated and eval_status.last_evaluated < cutoff:
                        # Calculate priority based on staleness
                        days_stale = (datetime.now() - eval_status.last_evaluated).days
                        priority = min(days_stale / max_age_days, 1.0)

                        combinations.append(
                            EvaluationCombination(
                                model_path=model.model_path,
                                model_hash=model.model_hash,
                                harness=eval_status.harness,
                                config_key=model.config_key,
                                board_type=model.board_type,
                                num_players=model.num_players,
                                last_evaluated=eval_status.last_evaluated,
                                priority=priority,
                            )
                        )

        # Sort by staleness (oldest first)
        combinations.sort(
            key=lambda x: x.last_evaluated or datetime.min
        )
        return combinations

    def get_all_combinations(self) -> list[EvaluationCombination]:
        """Return all valid (model, harness, config) combinations.

        Returns:
            List of all EvaluationCombination objects
        """
        combinations = []

        with self._cache_lock:
            for model in self._models_cache.values():
                for harness in model.compatible_harnesses:
                    # Find existing evaluation status
                    eval_status = next(
                        (e for e in model.evaluations if e.harness == harness),
                        None
                    )

                    combinations.append(
                        EvaluationCombination(
                            model_path=model.model_path,
                            model_hash=model.model_hash,
                            harness=harness,
                            config_key=model.config_key,
                            board_type=model.board_type,
                            num_players=model.num_players,
                            last_evaluated=eval_status.last_evaluated if eval_status else None,
                            priority=1.0 if eval_status is None else 0.5,
                        )
                    )

        return combinations

    def _is_cache_valid(self) -> bool:
        """Check if cache is still valid."""
        with self._cache_lock:
            if not self._models_cache:
                return False
            age = time.time() - self._cache_timestamp
            return age < self.cache_ttl_seconds

    async def _get_alive_peers(self) -> list[dict[str, Any]]:
        """Get list of alive peers from P2P status."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"http://localhost:{self.p2p_port}/status",
                    timeout=aiohttp.ClientTimeout(total=self.request_timeout),
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        peers = data.get("peers", {})
                        if not isinstance(peers, dict):
                            return []
                        return [
                            {"node_id": node_id, **info}
                            for node_id, info in peers.items()
                            if isinstance(info, dict) and info.get("alive")
                        ]
        except Exception as e:
            logger.warning(f"[ClusterModelEnumerator] Failed to get peers: {e}")

        return []

    async def _query_node_models(self, peer: dict[str, Any]) -> list[ModelInfo]:
        """Query a single node for its model inventory.

        Args:
            peer: Peer info dict with node_id, ip, port

        Returns:
            List of ModelInfo from this node
        """
        node_id = peer.get("node_id", "unknown")
        ip = peer.get("tailscale_ip") or peer.get("ip")
        port = peer.get("port", self.p2p_port)

        if not ip:
            return []

        try:
            url = f"http://{ip}:{port}/models/inventory"
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url,
                    timeout=aiohttp.ClientTimeout(total=self.request_timeout),
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        models = []
                        for model_data in data.get("models", []):
                            model = self._parse_model_data(model_data, node_id)
                            if model:
                                models.append(model)
                        return models
                    elif response.status == 404:
                        # Endpoint not implemented on this node
                        logger.debug(
                            f"[ClusterModelEnumerator] Node {node_id} doesn't have "
                            "/models/inventory endpoint"
                        )
                        return []
        except asyncio.TimeoutError:
            logger.debug(f"[ClusterModelEnumerator] Timeout querying {node_id}")
        except Exception as e:
            logger.debug(f"[ClusterModelEnumerator] Error querying {node_id}: {e}")

        return []

    def _parse_model_data(
        self, data: dict[str, Any], node_id: str
    ) -> ModelInfo | None:
        """Parse model data from node response.

        Args:
            data: Model data dict from node
            node_id: Node ID where model was found

        Returns:
            ModelInfo or None if parsing failed
        """
        try:
            path = data.get("path", "")
            model_type = data.get("type", "nn")
            board_type = data.get("board_type") or self._extract_board_type(path)
            num_players = data.get("num_players") or self._extract_num_players(path)

            if not board_type or not num_players:
                return None

            # Compute hash from path (ideally would use actual file hash)
            model_hash = data.get("hash") or self._compute_path_hash(path)

            return ModelInfo(
                model_path=path,
                model_hash=model_hash,
                model_type=model_type,
                board_type=board_type,
                num_players=num_players,
                node_ids=[node_id],
                size_bytes=data.get("size_bytes", 0),
                modified_time=data.get("modified", 0.0),
                architecture=data.get("architecture", ""),
            )
        except Exception as e:
            logger.debug(f"[ClusterModelEnumerator] Failed to parse model data: {e}")
            return None

    async def _discover_local_models(self) -> list[ModelInfo]:
        """Discover models in local models directory.

        Returns:
            List of ModelInfo from local filesystem
        """
        models = []
        node_id = os.environ.get("RINGRIFT_NODE_ID", "local")

        if not self._local_models_dir.exists():
            return models

        # Find all .pth files
        for pth_file in self._local_models_dir.rglob("*.pth"):
            try:
                board_type = self._extract_board_type(str(pth_file))
                num_players = self._extract_num_players(str(pth_file))
                model_type = self._detect_model_type(pth_file)

                if board_type and num_players:
                    model_hash = self._compute_path_hash(str(pth_file))
                    models.append(
                        ModelInfo(
                            model_path=str(pth_file),
                            model_hash=model_hash,
                            model_type=model_type,
                            board_type=board_type,
                            num_players=num_players,
                            node_ids=[node_id],
                            size_bytes=pth_file.stat().st_size,
                            modified_time=pth_file.stat().st_mtime,
                        )
                    )
            except Exception as e:
                logger.debug(f"[ClusterModelEnumerator] Error scanning {pth_file}: {e}")

        return models

    def _extract_board_type(self, path: str) -> str | None:
        """Extract board type from model path."""
        # Match patterns like hex8, square8, square19, hexagonal
        match = re.search(r"(hex8|square8|square19|hexagonal)", path.lower())
        return match.group(1) if match else None

    def _extract_num_players(self, path: str) -> int | None:
        """Extract number of players from model path."""
        # Match patterns like _2p, _3p, _4p
        match = re.search(r"_(\d)p", path.lower())
        return int(match.group(1)) if match else None

    def _detect_model_type(self, path: Path) -> str:
        """Detect model type from file.

        Args:
            path: Path to model file

        Returns:
            Model type: "nn", "nnue", or "nnue_mp"
        """
        name = path.name.lower()
        if "nnue_mp" in name:
            return "nnue_mp"
        elif "nnue" in name:
            return "nnue"
        return "nn"

    def _compute_path_hash(self, path: str) -> str:
        """Compute a hash from the model path.

        Note: This is a simple hash for identification. For true deduplication,
        we'd want to hash the actual file contents.
        """
        # Normalize path and hash
        normalized = Path(path).name  # Just the filename
        return hashlib.sha256(normalized.encode()).hexdigest()[:16]

    async def _load_evaluation_status(self, models: list[ModelInfo]) -> None:
        """Load evaluation status from EloService database.

        Args:
            models: List of models to load status for
        """
        try:
            # Try to load from unified Elo database
            elo_db_path = Path(
                os.environ.get("RINGRIFT_ELO_DB", "data/unified_elo.db")
            )

            if not elo_db_path.exists():
                logger.debug("[ClusterModelEnumerator] No Elo database found")
                return

            # Query for all ratings with composite IDs
            conn = sqlite3.connect(str(elo_db_path))
            try:
                cursor = conn.cursor()

                # Get all ratings - format is typically model_hash:harness:config
                cursor.execute("""
                    SELECT participant_id, elo, games_played, last_updated
                    FROM ratings
                """)

                ratings: dict[str, dict] = {}
                for row in cursor.fetchall():
                    participant_id, elo, games_played, last_updated = row
                    ratings[participant_id] = {
                        "elo": elo,
                        "games_played": games_played,
                        "last_updated": last_updated,
                    }

                # Match ratings to models
                for model in models:
                    for harness in model.compatible_harnesses:
                        # Try different composite ID formats
                        composite_ids = [
                            f"{model.model_hash}:{harness}:{model.config_key}",
                            f"{Path(model.model_path).stem}:{harness}",
                            f"{model.model_hash}:{harness}",
                        ]

                        for composite_id in composite_ids:
                            if composite_id in ratings:
                                rating = ratings[composite_id]
                                last_eval = None
                                if rating.get("last_updated"):
                                    try:
                                        last_eval = datetime.fromisoformat(
                                            rating["last_updated"]
                                        )
                                    except (ValueError, TypeError):
                                        pass

                                model.evaluations.append(
                                    EvaluationStatus(
                                        harness=harness,
                                        last_evaluated=last_eval,
                                        elo_rating=rating.get("elo"),
                                        games_played=rating.get("games_played", 0),
                                        config_key=model.config_key,
                                    )
                                )
                                break  # Found match, don't check other formats

            finally:
                conn.close()

        except Exception as e:
            logger.warning(
                f"[ClusterModelEnumerator] Failed to load evaluation status: {e}"
            )


# Singleton accessors
def get_cluster_model_enumerator() -> ClusterModelEnumerator:
    """Get singleton instance of ClusterModelEnumerator."""
    return ClusterModelEnumerator.get_instance()


def reset_cluster_model_enumerator() -> None:
    """Reset singleton instance (for testing)."""
    ClusterModelEnumerator.reset_instance()
