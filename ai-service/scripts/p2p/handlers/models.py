"""Model Inventory HTTP Handlers Mixin.

Provides HTTP endpoints for model inventory and evaluation status.
Used by ClusterModelEnumerator to discover models across the cluster.

January 2026: Created as part of Comprehensive Model Evaluation Pipeline.

Usage:
    class P2POrchestrator(ModelHandlersMixin, ...):
        pass

Endpoints:
    GET /models/inventory - Get all models on this node with metadata
    GET /models/evaluation_status - Get evaluation status for models on this node

Requires the implementing class to have:
    - node_id: str
    - _local_models_dir: Path (or use RINGRIFT_MODELS_PATH env var)
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

from aiohttp import web

from scripts.p2p.handlers.base import BaseP2PHandler
from scripts.p2p.handlers.timeout_decorator import (
    HANDLER_TIMEOUT_TOURNAMENT,
    handler_timeout,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Project root for default paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

# Default models directory
DEFAULT_MODELS_DIR = PROJECT_ROOT / "models"

# Model patterns to match
MODEL_PATTERNS = ["*.pth", "*.pt"]

# Board type regex pattern
BOARD_TYPE_PATTERN = re.compile(r"(hex8|square8|square19|hexagonal)", re.IGNORECASE)

# Player count regex pattern
PLAYER_COUNT_PATTERN = re.compile(r"_(\d)p", re.IGNORECASE)


class ModelHandlersMixin(BaseP2PHandler):
    """Mixin providing model inventory HTTP handlers.

    Inherits from BaseP2PHandler for consistent response formatting.

    Provides:
        - GET /models/inventory: Return all local models with metadata
        - GET /models/evaluation_status: Return evaluation status per model

    Required attributes in implementing class:
        - node_id: str
    """

    # Type hints for required attributes
    node_id: str

    @handler_timeout(HANDLER_TIMEOUT_TOURNAMENT)
    async def handle_model_inventory(self, request: web.Request) -> web.Response:
        """Return all models on this node with metadata.

        Response format:
        {
            "node_id": "node-1",
            "models": [
                {
                    "path": "models/canonical_hex8_2p.pth",
                    "type": "nn",
                    "board_type": "hex8",
                    "num_players": 2,
                    "size_bytes": 34567890,
                    "modified": 1704067200.0,
                    "hash": "abc123...",
                    "architecture": "v2"
                },
                ...
            ],
            "count": 15
        }
        """
        try:
            models = await asyncio.to_thread(self._discover_local_models_for_inventory)

            return web.json_response({
                "node_id": self.node_id,
                "models": models,
                "count": len(models),
            })
        except Exception as e:  # noqa: BLE001
            logger.error(f"[ModelHandler] Error getting inventory: {e}")
            return web.json_response({
                "error": str(e),
                "node_id": self.node_id,
            }, status=500)

    @handler_timeout(HANDLER_TIMEOUT_TOURNAMENT)
    async def handle_evaluation_status(self, request: web.Request) -> web.Response:
        """Return evaluation status for models on this node.

        Response format:
        {
            "node_id": "node-1",
            "statuses": [
                {
                    "path": "models/canonical_hex8_2p.pth",
                    "hash": "abc123...",
                    "harnesses_evaluated": ["gumbel_mcts", "minimax"],
                    "latest_eval": "2026-01-09T12:00:00Z",
                    "ratings": {
                        "gumbel_mcts": {"elo": 1450, "games": 100},
                        "minimax": {"elo": 1380, "games": 50}
                    }
                },
                ...
            ],
            "count": 15
        }
        """
        try:
            statuses = await asyncio.to_thread(self._get_model_evaluation_statuses)

            return web.json_response({
                "node_id": self.node_id,
                "statuses": statuses,
                "count": len(statuses),
            })
        except Exception as e:  # noqa: BLE001
            logger.error(f"[ModelHandler] Error getting evaluation status: {e}")
            return web.json_response({
                "error": str(e),
                "node_id": self.node_id,
            }, status=500)

    def _discover_local_models_for_inventory(self) -> list[dict[str, Any]]:
        """Discover all model files in local models directory.

        Returns:
            List of model metadata dicts
        """
        models = []
        models_dir = self._get_models_directory()

        if not models_dir.exists():
            logger.debug(f"[ModelHandler] Models directory does not exist: {models_dir}")
            return models

        # Find all model files
        for pattern in MODEL_PATTERNS:
            for model_path in models_dir.rglob(pattern):
                try:
                    model_info = self._extract_model_info(model_path)
                    if model_info:
                        models.append(model_info)
                except Exception as e:  # noqa: BLE001
                    logger.debug(f"[ModelHandler] Error processing {model_path}: {e}")

        logger.info(f"[ModelHandler] Discovered {len(models)} models in {models_dir}")
        return models

    def _extract_model_info(self, path: Path) -> dict[str, Any] | None:
        """Extract metadata from a model file.

        Args:
            path: Path to the model file

        Returns:
            Model metadata dict or None if extraction failed
        """
        try:
            # Extract board type from filename
            board_match = BOARD_TYPE_PATTERN.search(path.name)
            board_type = board_match.group(1).lower() if board_match else None

            # Extract player count from filename
            player_match = PLAYER_COUNT_PATTERN.search(path.name)
            num_players = int(player_match.group(1)) if player_match else None

            # Skip if we can't determine config
            if not board_type or not num_players:
                return None

            # Get file stats
            stat = path.stat()

            # Detect model type
            model_type = self._detect_model_type_from_path(path)

            # Compute hash for deduplication
            model_hash = self._compute_model_hash(path)

            # Detect architecture from filename
            architecture = self._detect_architecture(path)

            return {
                "path": str(path.relative_to(PROJECT_ROOT) if path.is_relative_to(PROJECT_ROOT) else path),
                "type": model_type,
                "board_type": board_type,
                "num_players": num_players,
                "size_bytes": stat.st_size,
                "modified": stat.st_mtime,
                "hash": model_hash,
                "architecture": architecture,
            }
        except Exception as e:  # noqa: BLE001
            logger.debug(f"[ModelHandler] Failed to extract info from {path}: {e}")
            return None

    def _get_models_directory(self) -> Path:
        """Get the models directory path.

        Returns:
            Path to models directory
        """
        # Check environment variable first
        env_path = os.environ.get("RINGRIFT_MODELS_PATH")
        if env_path:
            return Path(env_path)

        # Check if class has _local_models_dir attribute
        if hasattr(self, "_local_models_dir"):
            return Path(self._local_models_dir)  # type: ignore

        # Default to project models directory
        return DEFAULT_MODELS_DIR

    def _detect_model_type_from_path(self, path: Path) -> str:
        """Detect model type (nn, nnue, nnue_mp) from path.

        Args:
            path: Path to model file

        Returns:
            Model type string
        """
        name_lower = path.name.lower()

        if "nnue_mp" in name_lower or "nnue-mp" in name_lower:
            return "nnue_mp"
        if "nnue" in name_lower:
            return "nnue"

        # Default to neural network
        return "nn"

    def _detect_architecture(self, path: Path) -> str:
        """Detect neural network architecture from filename.

        Args:
            path: Path to model file

        Returns:
            Architecture string (e.g., "v2", "v4", "v5-heavy")
        """
        name_lower = path.name.lower()

        # Check for version patterns
        if "v5-heavy-xl" in name_lower or "v5_heavy_xl" in name_lower:
            return "v5-heavy-xl"
        if "v5-heavy-large" in name_lower or "v5_heavy_large" in name_lower:
            return "v5-heavy-large"
        if "v5-heavy" in name_lower or "v5_heavy" in name_lower or "v5heavy" in name_lower:
            return "v5-heavy"
        if "v5" in name_lower:
            return "v5"
        if "v4" in name_lower:
            return "v4"
        if "v3" in name_lower:
            return "v3"
        if "v2" in name_lower:
            return "v2"

        # Default
        return "v2"

    def _compute_model_hash(self, path: Path) -> str:
        """Compute hash for model file (using path + size + mtime for speed).

        Args:
            path: Path to model file

        Returns:
            Hash string for deduplication
        """
        try:
            stat = path.stat()
            # Use path + size + mtime for fast hashing without reading file
            content = f"{path}:{stat.st_size}:{stat.st_mtime}"
            return hashlib.sha256(content.encode()).hexdigest()[:16]
        except Exception:
            # Fallback to just path
            return hashlib.sha256(str(path).encode()).hexdigest()[:16]

    def _get_model_evaluation_statuses(self) -> list[dict[str, Any]]:
        """Get evaluation status for all local models.

        Returns:
            List of evaluation status dicts
        """
        statuses = []

        try:
            # Get Elo service for ratings
            elo_service = self._get_elo_service()
        except Exception as e:  # noqa: BLE001
            logger.debug(f"[ModelHandler] EloService not available: {e}")
            elo_service = None

        # Get model files
        models_dir = self._get_models_directory()
        if not models_dir.exists():
            return statuses

        for pattern in MODEL_PATTERNS:
            for model_path in models_dir.rglob(pattern):
                try:
                    status = self._get_single_model_status(model_path, elo_service)
                    if status:
                        statuses.append(status)
                except Exception as e:  # noqa: BLE001
                    logger.debug(f"[ModelHandler] Error getting status for {model_path}: {e}")

        return statuses

    def _get_single_model_status(
        self,
        path: Path,
        elo_service: Any | None,
    ) -> dict[str, Any] | None:
        """Get evaluation status for a single model.

        Args:
            path: Path to model file
            elo_service: EloService instance or None

        Returns:
            Status dict or None
        """
        # Extract basic info
        model_info = self._extract_model_info(path)
        if not model_info:
            return None

        model_hash = model_info["hash"]
        config_key = f"{model_info['board_type']}_{model_info['num_players']}p"

        # Get ratings from Elo service
        harnesses_evaluated = []
        ratings = {}
        latest_eval = None

        if elo_service:
            try:
                # Query ratings for this model
                all_ratings = self._query_model_ratings(
                    elo_service, model_hash, config_key
                )
                for harness, rating_info in all_ratings.items():
                    harnesses_evaluated.append(harness)
                    ratings[harness] = rating_info
                    # Track latest evaluation
                    if rating_info.get("timestamp"):
                        if latest_eval is None or rating_info["timestamp"] > latest_eval:
                            latest_eval = rating_info["timestamp"]
            except Exception as e:  # noqa: BLE001
                logger.debug(f"[ModelHandler] Error querying ratings: {e}")

        return {
            "path": model_info["path"],
            "hash": model_hash,
            "harnesses_evaluated": harnesses_evaluated,
            "latest_eval": latest_eval,
            "ratings": ratings,
        }

    def _get_elo_service(self) -> Any:
        """Get EloService instance.

        Returns:
            EloService instance

        Raises:
            ImportError: If EloService not available
        """
        from app.training.elo_service import get_elo_service
        return get_elo_service()

    def _query_model_ratings(
        self,
        elo_service: Any,
        model_hash: str,
        config_key: str,
    ) -> dict[str, dict[str, Any]]:
        """Query ratings for a model from EloService.

        Args:
            elo_service: EloService instance
            model_hash: Model hash for lookup
            config_key: Config key like 'hex8_2p'

        Returns:
            Dict mapping harness -> rating info
        """
        ratings = {}

        # Get all ratings for this model
        try:
            # EloService stores composite IDs as {model_hash}:{harness}:{config}
            all_participants = elo_service.get_all_participants()

            for participant_id in all_participants:
                # Parse composite ID
                parts = participant_id.split(":")
                if len(parts) >= 2 and parts[0] == model_hash:
                    harness = parts[1]
                    # Get rating info
                    rating = elo_service.get_rating(participant_id)
                    if rating:
                        ratings[harness] = {
                            "elo": rating.elo if hasattr(rating, "elo") else rating,
                            "games": getattr(rating, "games", 0),
                            "timestamp": getattr(rating, "timestamp", None),
                        }
        except Exception as e:  # noqa: BLE001
            logger.debug(f"[ModelHandler] Error querying ratings: {e}")

        return ratings


def setup_model_routes(app: web.Application, handler: ModelHandlersMixin) -> int:
    """Register model inventory routes.

    Args:
        app: aiohttp application
        handler: Handler mixin instance

    Returns:
        Number of routes registered
    """
    routes = [
        ("GET", "/models/inventory", handler.handle_model_inventory),
        ("GET", "/models/evaluation_status", handler.handle_evaluation_status),
    ]

    for method, path, handler_fn in routes:
        if method == "GET":
            app.router.add_get(path, handler_fn)
        elif method == "POST":
            app.router.add_post(path, handler_fn)

    logger.info(f"[ModelHandler] Registered {len(routes)} model routes")
    return len(routes)
