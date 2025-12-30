"""Model selector for self-play that automatically finds the best model.

This module provides automatic model selection for selfplay workers,
removing the need to manually specify --model-path. It searches for models
in order of preference:
1. Production model from registry (if promoted)
2. Latest checkpoint from training runs
3. Canonical baseline model
4. Cluster-wide model search (if enabled)
5. NNUE model (fallback)
6. None (use random policy)

Usage:
    selector = SelfplayModelSelector(board_type="square8", num_players=2)

    # Get the best available model path
    model_path = selector.get_current_model()

    # Enable cluster search for missing models
    selector = SelfplayModelSelector(
        board_type="square8", num_players=2,
        search_cluster=True,
    )

    # Subscribe to model updates (for hot reload)
    selector.subscribe(callback)

Integration with selfplay:
    # In SelfplayConfig initialization
    if model_path is None:
        selector = SelfplayModelSelector(board_type, num_players)
        model_path = selector.get_current_model()
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)

# Default search paths for models
_AI_SERVICE_ROOT = Path(__file__).resolve().parents[2]
_MODELS_DIR = _AI_SERVICE_ROOT / "models"
_CHECKPOINTS_DIR = _AI_SERVICE_ROOT / "data" / "checkpoints"
_NNUE_MODELS_DIR = _MODELS_DIR / "nnue"

# NFS path for cluster deployments
_NFS_MODELS_DIR = Path("/lambda/nfs/RingRift/ai-service/models")


class SelfplayModelSelector:
    """Automatic model selection for selfplay.

    Searches for the best available model in order:
    1. Production model from unified model store
    2. Latest checkpoint matching board/player config
    3. Canonical baseline model
    4. NNUE model (if available)
    """

    def __init__(
        self,
        board_type: str,
        num_players: int = 2,
        prefer_nnue: bool = False,
        cache_ttl: float = 60.0,
        search_cluster: bool = True,
    ):
        """Initialize model selector.

        Args:
            board_type: Board type (square8, hex8, hexagonal, etc.)
            num_players: Number of players (2, 3, or 4)
            prefer_nnue: If True, prefer NNUE models over policy/value nets
            cache_ttl: Time to live for cached model path (seconds)
            search_cluster: If True, search cluster for models when not found locally
        """
        self.board_type = board_type.lower()
        self.num_players = num_players
        self.prefer_nnue = prefer_nnue
        self.cache_ttl = cache_ttl
        self.search_cluster = search_cluster

        self._config_key = f"{self.board_type}_{self.num_players}p"
        self._cached_path: Path | None = None
        self._cache_time: float = 0
        self._subscribers: list[Callable[[Path], None]] = []

        # Lazy-load cluster discovery
        self._cluster_discovery = None

        logger.debug(
            f"SelfplayModelSelector initialized: config={self._config_key}, "
            f"prefer_nnue={prefer_nnue}, search_cluster={search_cluster}"
        )

    def get_current_model(self, force_refresh: bool = False) -> Path | None:
        """Get the current best model path.

        Args:
            force_refresh: If True, bypass cache and search again.

        Returns:
            Path to model file, or None if no model found.
        """
        import time

        # Check cache
        if not force_refresh and self._cached_path is not None:
            if time.time() - self._cache_time < self.cache_ttl:
                return self._cached_path

        # Search for model in order of preference
        model_path = None

        # 1. Try production model from registry
        model_path = self._get_production_model()
        if model_path and model_path.exists():
            logger.info(f"Using production model: {model_path}")
            self._update_cache(model_path)
            return model_path

        # 2. Try latest checkpoint
        model_path = self._get_latest_checkpoint()
        if model_path and model_path.exists():
            logger.info(f"Using latest checkpoint: {model_path}")
            self._update_cache(model_path)
            return model_path

        # 3. Try NNUE model (if preferred or no other option)
        if self.prefer_nnue:
            model_path = self._get_nnue_model()
            if model_path and model_path.exists():
                logger.info(f"Using NNUE model: {model_path}")
                self._update_cache(model_path)
                return model_path

        # 4. Try canonical baseline
        model_path = self._get_canonical_model()
        if model_path and model_path.exists():
            logger.info(f"Using canonical model: {model_path}")
            self._update_cache(model_path)
            return model_path

        # 5. Try cluster-wide search (will sync model if found)
        if self.search_cluster:
            model_path = self._get_cluster_model()
            if model_path and model_path.exists():
                logger.info(f"Using cluster model: {model_path}")
                self._update_cache(model_path)
                return model_path

        # 6. Try NNUE as fallback
        if not self.prefer_nnue:
            model_path = self._get_nnue_model()
            if model_path and model_path.exists():
                logger.info(f"Using NNUE model (fallback): {model_path}")
                self._update_cache(model_path)
                return model_path

        logger.warning(
            f"No model found for {self._config_key}. "
            "Selfplay will use random policy."
        )
        self._update_cache(None)
        return None

    def _get_cluster_model(self) -> Path | None:
        """Search cluster for model and sync if found.

        Uses ClusterModelDiscovery to:
        1. Check ClusterManifest for known model locations
        2. Query remote nodes via SSH if needed
        3. Sync the best model to local

        Returns:
            Path to synced local model, or None if not found
        """
        try:
            from app.models.cluster_discovery import get_cluster_model_discovery

            if self._cluster_discovery is None:
                self._cluster_discovery = get_cluster_model_discovery()

            # Use ensure_model_available which handles sync
            local_path = self._cluster_discovery.ensure_model_available(
                board_type=self.board_type,
                num_players=self.num_players,
                prefer_canonical=True,
            )

            if local_path and local_path.exists():
                logger.info(
                    f"Cluster model synced: {local_path.name}"
                )
                return local_path

        except ImportError:
            logger.debug("ClusterModelDiscovery not available")
        except Exception as e:
            logger.warning(f"Cluster model search failed: {e}")

        return None

    def _get_production_model(self) -> Path | None:
        """Try to get production model from unified model store."""
        try:
            from app.training.unified_model_store import UnifiedModelStore

            store = UnifiedModelStore()
            model_info = store.get_production(config_key=self._config_key)

            if model_info and model_info.model_path:
                return Path(model_info.model_path)

        except ImportError:
            logger.debug("UnifiedModelStore not available")
        except Exception as e:
            logger.debug(f"Error getting production model: {e}")

        return None

    def _get_latest_checkpoint(self) -> Path | None:
        """Find latest checkpoint matching this config."""
        # Check common checkpoint patterns - be strict about matching both board and players
        patterns = [
            f"*{self._config_key}*",  # e.g., *square8_2p*
            f"*{self.board_type}*_{self.num_players}p*",  # e.g., *square8*_2p*
        ]

        checkpoint_dirs = [
            _CHECKPOINTS_DIR,
            _AI_SERVICE_ROOT / "runs",
            Path.home() / "training_data",
        ]

        # Add NFS path if on cluster
        if _NFS_MODELS_DIR.exists():
            checkpoint_dirs.append(_NFS_MODELS_DIR.parent / "data" / "checkpoints")

        latest_path = None
        latest_time = 0.0

        for check_dir in checkpoint_dirs:
            if not check_dir.exists():
                continue

            for pattern in patterns:
                for path in check_dir.glob(f"**/{pattern}"):
                    if path.is_file() and path.suffix in (".pt", ".pth", ".ckpt"):
                        # Double-check the filename contains the right player count
                        name = path.name.lower()
                        if f"{self.num_players}p" not in name:
                            continue  # Skip if player count doesn't match

                        mtime = path.stat().st_mtime
                        if mtime > latest_time:
                            latest_time = mtime
                            latest_path = path

        return latest_path

    def _get_canonical_model(self) -> Path | None:
        """Get canonical baseline model."""
        # Standard naming convention
        canonical_names = [
            f"canonical_{self._config_key}.pth",
            f"canonical_{self._config_key}.pt",
            f"{self._config_key}_canonical.pth",
            f"{self._config_key}_best.pth",
        ]

        search_dirs = [_MODELS_DIR, _NFS_MODELS_DIR]

        for search_dir in search_dirs:
            if not search_dir.exists():
                continue
            for name in canonical_names:
                path = search_dir / name
                if path.exists():
                    return path

        return None

    def _get_nnue_model(self) -> Path | None:
        """Get NNUE model for this config."""
        # Standard NNUE naming
        nnue_names = [
            f"nnue_{self._config_key}.pt",
            f"nnue_{self.board_type}_{self.num_players}p.pt",
            f"nnue_{self.board_type}.pt",  # Legacy 2p naming
        ]

        search_dirs = [_NNUE_MODELS_DIR, _MODELS_DIR, _NFS_MODELS_DIR / "nnue", _NFS_MODELS_DIR]

        for search_dir in search_dirs:
            if not search_dir.exists():
                continue
            for name in nnue_names:
                path = search_dir / name
                if path.exists():
                    return path

        return None

    def _update_cache(self, path: Path | None) -> None:
        """Update the cached model path."""
        import time

        self._cached_path = path
        self._cache_time = time.time()

    def subscribe(self, callback: Callable[[Path | None], None]) -> None:
        """Subscribe to model updates.

        The callback will be called when a new model is promoted.

        Args:
            callback: Function to call with new model path.
        """
        self._subscribers.append(callback)

    def notify_update(self, new_path: Path | None) -> None:
        """Notify subscribers of a model update.

        Called by event system when MODEL_PROMOTED event is received.

        Args:
            new_path: Path to the new model.
        """
        self._update_cache(new_path)

        for callback in self._subscribers:
            try:
                callback(new_path)
            except Exception as e:
                logger.warning(f"Error in model update callback: {e}")

    def get_model_info(self) -> dict:
        """Get information about current model selection.

        Returns:
            Dictionary with model information.
        """
        model_path = self.get_current_model()

        return {
            "config_key": self._config_key,
            "model_path": str(model_path) if model_path else None,
            "model_exists": model_path.exists() if model_path else False,
            "model_type": self._classify_model(model_path) if model_path else "none",
            "cache_age": 0 if self._cache_time == 0 else (
                datetime.now().timestamp() - self._cache_time
            ),
        }

    def _classify_model(self, path: Path) -> str:
        """Classify the type of model."""
        name = path.name.lower()

        if "nnue" in name:
            return "nnue"
        elif "canonical" in name:
            return "canonical"
        elif "checkpoint" in str(path) or "runs" in str(path):
            return "checkpoint"
        else:
            return "unknown"


# Global selector cache
_selectors: dict[str, SelfplayModelSelector] = {}

# Event subscription flag
_event_subscription_initialized = False


def _on_model_promoted(event) -> None:
    """Handle MODEL_PROMOTED event and invalidate relevant caches.

    This provides hot-reload capability: when a model is promoted,
    all selectors for that config immediately pick up the new model
    instead of waiting for cache TTL expiry.

    Args:
        event: The MODEL_PROMOTED event with board_type, num_players, model_path
    """
    try:
        # Extract config from event
        board_type = getattr(event, "board_type", None) or event.get("board_type")
        num_players = getattr(event, "num_players", None) or event.get("num_players")
        model_path = getattr(event, "model_path", None) or event.get("model_path")

        if not board_type or not num_players:
            logger.warning("[SelfplayModelSelector] MODEL_PROMOTED event missing board_type or num_players")
            return

        config_key = f"{board_type.lower()}_{num_players}p"
        logger.info(f"[SelfplayModelSelector] MODEL_PROMOTED for {config_key}, invalidating cache")

        # Notify all selectors matching this config
        for key, selector in _selectors.items():
            if config_key in key:
                new_path = Path(model_path) if model_path else None
                selector.notify_update(new_path)
                logger.debug(f"[SelfplayModelSelector] Notified selector {key}")

    except Exception as e:
        logger.warning(f"[SelfplayModelSelector] Error handling MODEL_PROMOTED: {e}")


def _init_event_subscription() -> None:
    """Initialize subscription to MODEL_PROMOTED events.

    Called lazily on first selector creation to avoid import cycles.
    """
    global _event_subscription_initialized

    if _event_subscription_initialized:
        return

    try:
        from app.coordination.event_router import get_router
        # Dec 29, 2025: Fixed import path (was app.coordination.data_events)
        from app.distributed.data_events import DataEventType

        router = get_router()
        router.subscribe(DataEventType.MODEL_PROMOTED.value, _on_model_promoted)
        _event_subscription_initialized = True
        logger.info("[SelfplayModelSelector] Subscribed to MODEL_PROMOTED events for hot-reload")

    except ImportError as e:
        logger.debug(f"[SelfplayModelSelector] Event router not available: {e}")
    except Exception as e:
        logger.warning(f"[SelfplayModelSelector] Could not subscribe to MODEL_PROMOTED: {e}")


def get_model_for_config(
    board_type: str,
    num_players: int = 2,
    prefer_nnue: bool = False,
) -> Path | None:
    """Convenience function to get model path for a config.

    Caches selector instances for efficiency.

    Args:
        board_type: Board type (square8, hex8, etc.)
        num_players: Number of players (2, 3, or 4)
        prefer_nnue: If True, prefer NNUE models

    Returns:
        Path to model or None.
    """
    # Initialize event subscription on first use (for hot-reload)
    _init_event_subscription()

    cache_key = f"{board_type.lower()}_{num_players}p_{prefer_nnue}"

    if cache_key not in _selectors:
        _selectors[cache_key] = SelfplayModelSelector(
            board_type=board_type,
            num_players=num_players,
            prefer_nnue=prefer_nnue,
        )

    return _selectors[cache_key].get_current_model()


def list_available_models(board_type: str | None = None) -> list[dict]:
    """List all available models, optionally filtered by board type.

    Args:
        board_type: Optional filter for board type.

    Returns:
        List of model info dictionaries.
    """
    models = []

    search_dirs = [_MODELS_DIR, _NNUE_MODELS_DIR, _NFS_MODELS_DIR]

    for search_dir in search_dirs:
        if not search_dir.exists():
            continue

        for path in search_dir.glob("**/*.pt*"):
            name = path.stem.lower()

            # Extract config info from name
            config_key = None
            for pattern in ["square8", "square19", "hex8", "hexagonal", "hex"]:
                if pattern in name:
                    for players in ["2p", "3p", "4p"]:
                        if players in name:
                            config_key = f"{pattern}_{players}"
                            break
                    if config_key:
                        break

            # Filter by board type if specified
            if board_type and config_key:
                if board_type.lower() not in config_key:
                    continue

            models.append({
                "path": str(path),
                "name": path.name,
                "config_key": config_key,
                "size_mb": path.stat().st_size / (1024 * 1024),
                "modified": datetime.fromtimestamp(path.stat().st_mtime).isoformat(),
            })

    # Sort by modification time (newest first)
    models.sort(key=lambda m: m["modified"], reverse=True)

    return models
