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

    def get_model_with_architecture_weights(
        self,
        temperature: float = 0.5,
    ) -> tuple[Path | None, str | None]:
        """Get model using architecture weights for probabilistic selection.

        Session 17.11 (Jan 4, 2026): Cross-NN Architecture Curriculum Hierarchy.
        Uses architecture weights from ArchitectureTracker to probabilistically
        select which architecture's model to use for selfplay.

        Expected improvement: +25-35 Elo from better architecture allocation.

        Args:
            temperature: Softmax temperature for weight concentration (lower = more biased)

        Returns:
            Tuple of (model_path, selected_architecture) or (None, None) if no model found
        """
        import random

        try:
            from app.training.architecture_tracker import get_allocation_weights

            # Get architecture weights based on Elo performance
            weights = get_allocation_weights(
                board_type=self.board_type,
                num_players=self.num_players,
                temperature=temperature,
            )

            if not weights:
                # No architecture data - fall back to default selection
                model_path = self.get_current_model()
                return model_path, self._extract_architecture(model_path) if model_path else None

            # Weighted random selection of architecture
            architectures = list(weights.keys())
            probs = list(weights.values())

            # Normalize probabilities (should already sum to 1.0, but be safe)
            total = sum(probs)
            if total > 0:
                probs = [p / total for p in probs]

            selected_arch = random.choices(architectures, weights=probs, k=1)[0]

            # Find model for selected architecture
            model_path = self._find_model_for_architecture(selected_arch)

            if model_path:
                logger.info(
                    f"[SelfplayModelSelector] Selected {selected_arch} architecture "
                    f"(weight={weights.get(selected_arch, 0):.2f}): {model_path.name}"
                )
                return model_path, selected_arch

            # Fallback: try other architectures in order of weight
            for arch in sorted(architectures, key=lambda a: weights.get(a, 0), reverse=True):
                if arch != selected_arch:
                    model_path = self._find_model_for_architecture(arch)
                    if model_path:
                        logger.info(
                            f"[SelfplayModelSelector] Fallback to {arch} architecture: {model_path.name}"
                        )
                        return model_path, arch

            # Final fallback to default selection
            model_path = self.get_current_model()
            return model_path, self._extract_architecture(model_path) if model_path else None

        except ImportError:
            logger.debug("[SelfplayModelSelector] architecture_tracker not available")
            model_path = self.get_current_model()
            return model_path, self._extract_architecture(model_path) if model_path else None
        except Exception as e:
            logger.warning(f"[SelfplayModelSelector] Error in architecture selection: {e}")
            model_path = self.get_current_model()
            return model_path, self._extract_architecture(model_path) if model_path else None

    def _find_model_for_architecture(self, architecture: str) -> Path | None:
        """Find a model matching a specific architecture version.

        Session 17.11 (Jan 4, 2026): Helper for architecture-aware selection.

        Args:
            architecture: Architecture version (e.g., "v2", "v4", "v5-heavy")

        Returns:
            Path to model or None if not found
        """
        # Normalize architecture name for pattern matching
        arch_patterns = self._get_architecture_patterns(architecture)

        # Search in models directory
        search_dirs = [_MODELS_DIR, _NFS_MODELS_DIR]

        for search_dir in search_dirs:
            if not search_dir.exists():
                continue

            for path in search_dir.glob("*.pt*"):
                name = path.name.lower()

                # Must match config key
                if self._config_key not in name and not self._matches_config(name):
                    continue

                # Check if matches architecture pattern
                for pattern in arch_patterns:
                    if pattern in name:
                        return path

        return None

    def _get_architecture_patterns(self, architecture: str) -> list[str]:
        """Get filename patterns for architecture matching.

        Args:
            architecture: Architecture name (e.g., "v2", "v5-heavy")

        Returns:
            List of patterns to search for
        """
        arch_lower = architecture.lower().replace("-", "_")

        # Map architecture names to file patterns
        if arch_lower in ("v5_heavy_large", "v5-heavy-large"):
            return ["v5_heavy_large", "v5heavy_large", "v5heavylarge"]
        elif arch_lower in ("v5_heavy", "v5-heavy"):
            return ["v5_heavy", "v5heavy"]
        elif arch_lower == "v4":
            return ["_v4", "v4_"]
        elif arch_lower == "v3":
            return ["_v3", "v3_"]
        elif arch_lower == "v2":
            # v2 is default, might not have version in name
            return ["_v2", "v2_", "canonical_"]
        else:
            return [arch_lower]

    def _matches_config(self, name: str) -> bool:
        """Check if filename matches this config (board + players)."""
        return (
            self.board_type in name
            and f"{self.num_players}p" in name
        )

    def _extract_architecture(self, path: Path | None) -> str | None:
        """Extract architecture version from model path.

        Args:
            path: Model file path

        Returns:
            Architecture version string or None
        """
        if path is None:
            return None

        name = path.name.lower()

        # Check for architecture version in filename
        if "v5_heavy_large" in name or "v5heavylarge" in name:
            return "v5-heavy-large"
        elif "v5_heavy" in name or "v5heavy" in name:
            return "v5-heavy"
        elif "_v4" in name or "v4_" in name:
            return "v4"
        elif "_v3" in name or "v3_" in name:
            return "v3"
        elif "_v2" in name or "v2_" in name:
            return "v2"
        else:
            # Default is v2 for canonical models without version
            if "canonical" in name:
                return "v2"
            return None


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
        # Extract config from event. Mar 2026: Use get_event_payload() to handle
        # RouterEvent objects (which have .payload dict, not direct attributes).
        from app.coordination.event_router import get_event_payload
        payload = get_event_payload(event)
        board_type = payload.get("board_type")
        num_players = payload.get("num_players")
        model_path = payload.get("model_path")

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
    model_version: str = "v5",
) -> Path | None:
    """Convenience function to get model path for a config.

    Caches selector instances for efficiency.

    Jan 5, 2026 (Session 17.25): Added model_version parameter to support
    architecture selection feedback loop. When model_version is not "v5",
    looks for version-specific canonical models first (e.g., canonical_hex8_2p_v4.pth).

    Args:
        board_type: Board type (square8, hex8, etc.)
        num_players: Number of players (2, 3, or 4)
        prefer_nnue: If True, prefer NNUE models
        model_version: Architecture version (v2, v4, v5, v5_heavy, etc.)
            Default "v5" uses the default canonical model.

    Returns:
        Path to model or None.
    """
    # Initialize event subscription on first use (for hot-reload)
    _init_event_subscription()

    # Jan 5, 2026: Try version-specific model first if not default v5
    if model_version and model_version != "v5":
        # Normalize version name for file path (v5-heavy -> v5_heavy)
        version_normalized = model_version.replace("-", "_")
        versioned_name = f"canonical_{board_type.lower()}_{num_players}p_{version_normalized}.pth"
        versioned_path = _MODELS_DIR / versioned_name

        if versioned_path.exists():
            logger.info(
                f"[SelfplayModelSelector] Using architecture {model_version}: {versioned_name}"
            )
            return versioned_path
        else:
            # Also check NFS path for cluster deployments
            nfs_versioned = _NFS_MODELS_DIR / versioned_name
            if nfs_versioned.exists():
                logger.info(
                    f"[SelfplayModelSelector] Using NFS architecture {model_version}: {versioned_name}"
                )
                return nfs_versioned
            else:
                logger.debug(
                    f"[SelfplayModelSelector] No model for {model_version} at {versioned_path}, "
                    "falling back to default canonical"
                )

    # Default behavior: use selector for canonical/latest model
    cache_key = f"{board_type.lower()}_{num_players}p_{prefer_nnue}"

    if cache_key not in _selectors:
        _selectors[cache_key] = SelfplayModelSelector(
            board_type=board_type,
            num_players=num_players,
            prefer_nnue=prefer_nnue,
        )

    return _selectors[cache_key].get_current_model()


def get_model_with_architecture_weights(
    board_type: str,
    num_players: int = 2,
    temperature: float = 0.5,
) -> tuple[Path | None, str | None]:
    """Get model using architecture weights for probabilistic selection.

    Session 17.11 (Jan 4, 2026): Cross-NN Architecture Curriculum Hierarchy.
    Uses architecture weights from ArchitectureTracker to probabilistically
    select which architecture's model to use for selfplay.

    This function enables better allocation of training compute to architectures
    that are performing well, improving overall Elo gain rate.

    Expected improvement: +25-35 Elo from better architecture allocation.

    Args:
        board_type: Board type (square8, hex8, etc.)
        num_players: Number of players (2, 3, or 4)
        temperature: Softmax temperature (lower = more concentrated on best arch)

    Returns:
        Tuple of (model_path, architecture_name) or (None, None) if no model found

    Example:
        >>> model_path, arch = get_model_with_architecture_weights("hex8", 2)
        >>> print(f"Using {arch} architecture: {model_path}")
    """
    # Initialize event subscription on first use (for hot-reload)
    _init_event_subscription()

    cache_key = f"{board_type.lower()}_{num_players}p_arch"

    if cache_key not in _selectors:
        _selectors[cache_key] = SelfplayModelSelector(
            board_type=board_type,
            num_players=num_players,
        )

    return _selectors[cache_key].get_model_with_architecture_weights(temperature=temperature)


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
