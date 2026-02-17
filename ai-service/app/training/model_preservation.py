"""Model Preservation Service - Prevents deletion of high-Elo checkpoints.

January 2026: Closes the gap where checkpoint cleanup deletes models that have
earned high Elo ratings but haven't been promoted yet.

Problem: Training checkpoints with high Elo are being deleted by cleanup routines
that only consider validation loss, not Elo performance. This loses models that
should be promoted to "best" status.

Solution: Before deleting any checkpoint, check if it has significant Elo ratings
in the database. If so, preserve it to a dedicated directory.

Preservation Threshold Logic:
- Uses RELATIVE thresholds based on existing model distribution
- If >10 models in top decile (top 10%), preserves models in the top decile
- If <20 models exist, preserves models in top half (50%)
- Otherwise, preserves models in top quartile (top 25%)
- Falls back to 800 Elo if <3 models exist for comparison

This adaptive approach ensures:
- Early training: preserves relatively best models even at lower absolute Elo
- Mature training: threshold rises naturally with population quality
- Different configs: each board/player config has its own threshold

Usage:
    from app.training.model_preservation import (
        ModelPreservationService,
        get_preservation_service,
    )

    # Check before deleting
    service = get_preservation_service()
    if service.should_preserve(checkpoint_path, "hex8", 2):
        service.preserve_checkpoint(checkpoint_path, "hex8", 2)
    else:
        checkpoint_path.unlink()

    # Get current threshold for monitoring
    info = service.get_current_threshold("hex8", 2)
    print(f"Threshold: {info['elo_threshold']}, using {info['percentile_used']}")
"""

from __future__ import annotations

import hashlib
import logging
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Lazy imports to avoid circular dependencies
_elo_service = None


def _get_elo_service():
    """Lazy import of EloService to avoid circular dependency."""
    global _elo_service
    if _elo_service is None:
        try:
            from app.training.elo_service import get_elo_service
            _elo_service = get_elo_service()
        except ImportError:
            logger.warning("EloService not available, preservation disabled")
            return None
    return _elo_service


@dataclass
class PreservationConfig:
    """Configuration for model preservation.

    Attributes:
        use_relative_threshold: If True, use percentile-based threshold instead of fixed Elo.
        top_decile_min_count: Minimum models in top decile to use decile threshold.
            If fewer than this, falls back to quartile threshold.
        min_games_for_preservation: Minimum games played for reliable Elo estimate.
        preservation_dir: Directory to store preserved models.
        max_preserved_per_config: Maximum preserved models per board/player config.
            Oldest (by preservation time) are removed when limit exceeded.
        enabled: Whether preservation is active. Set to False to disable.
        fallback_min_elo: Fallback Elo threshold when no existing models to compare against.
    """
    use_relative_threshold: bool = True
    top_decile_min_count: int = 10  # If >10 in top decile, use decile; else use quartile
    min_games_for_preservation: int = 50
    preservation_dir: Path = field(default_factory=lambda: Path("models/preserved"))
    max_preserved_per_config: int = 25
    enabled: bool = True
    fallback_min_elo: float = 800.0  # Only used when <3 models exist for comparison


@dataclass
class PreservedModel:
    """Record of a preserved model."""
    original_path: str
    preserved_path: str
    board_type: str
    num_players: int
    content_hash: str
    elo_rating: float
    games_played: int
    preserved_at: float
    participant_id: str | None = None


class ModelPreservationService:
    """Service to prevent deletion of high-Elo models.

    Before checkpoint cleanup deletes a model, this service checks if the model
    has earned significant Elo ratings. If so, it preserves the model to a
    dedicated directory.

    The service uses content hash (SHA256) to identify models in the Elo database,
    matching the identity tracking in EloService.
    """

    _instance: "ModelPreservationService | None" = None

    def __init__(self, config: PreservationConfig | None = None):
        self.config = config or PreservationConfig()
        self._preserved_models: list[PreservedModel] = []
        self._preservation_dir = Path(self.config.preservation_dir)

        if self.config.enabled:
            self._preservation_dir.mkdir(parents=True, exist_ok=True)
            self._load_preserved_registry()

    @classmethod
    def get_instance(cls, config: PreservationConfig | None = None) -> "ModelPreservationService":
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = cls(config)
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton (for testing)."""
        cls._instance = None

    def _load_preserved_registry(self) -> None:
        """Load registry of preserved models from disk."""
        registry_path = self._preservation_dir / "registry.json"
        if registry_path.exists():
            import json
            try:
                with open(registry_path) as f:
                    data = json.load(f)
                self._preserved_models = [
                    PreservedModel(**m) for m in data.get("preserved", [])
                ]
            except (json.JSONDecodeError, OSError) as e:
                logger.warning(f"Could not load preservation registry: {e}")

    def _save_preserved_registry(self) -> None:
        """Save registry of preserved models to disk."""
        import json
        registry_path = self._preservation_dir / "registry.json"
        data = {
            "preserved": [
                {
                    "original_path": m.original_path,
                    "preserved_path": m.preserved_path,
                    "board_type": m.board_type,
                    "num_players": m.num_players,
                    "content_hash": m.content_hash,
                    "elo_rating": m.elo_rating,
                    "games_played": m.games_played,
                    "preserved_at": m.preserved_at,
                    "participant_id": m.participant_id,
                }
                for m in self._preserved_models
            ]
        }
        with open(registry_path, "w") as f:
            json.dump(data, f, indent=2)

    def _compute_model_hash(self, model_path: Path) -> str | None:
        """Compute SHA256 hash of model file content.

        Uses the same algorithm as EloService for identity matching.
        """
        if not model_path.exists():
            return None

        try:
            sha256 = hashlib.sha256()
            with open(model_path, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    sha256.update(chunk)
            return sha256.hexdigest()
        except (OSError, IOError) as e:
            logger.warning(f"Could not compute hash for {model_path}: {e}")
            return None

    def _get_elo_for_hash(
        self,
        content_hash: str,
        board_type: str,
        num_players: int,
    ) -> tuple[float | None, int, str | None]:
        """Look up Elo rating for a model by its content hash.

        Returns:
            Tuple of (elo_rating, games_played, participant_id) or (None, 0, None)
        """
        elo_service = _get_elo_service()
        if elo_service is None:
            return None, 0, None

        # Use EloService's hash-based identity lookup
        try:
            # Find participant ID by hash
            participant_id = elo_service._find_participant_by_hash(content_hash)
            if participant_id is None:
                return None, 0, None

            # Get rating for this participant
            rating = elo_service.get_rating(participant_id, board_type, num_players)
            if rating is None:
                return None, 0, participant_id

            return rating.rating, rating.games_played, participant_id
        except Exception as e:
            logger.warning(f"Error looking up Elo for hash {content_hash[:16]}...: {e}")
            return None, 0, None

    def _get_preservation_threshold(
        self,
        board_type: str,
        num_players: int,
    ) -> float:
        """Compute the Elo threshold for preservation based on existing model distribution.

        Feb 2026: Simplified to always use 50th percentile (top half) for
        consistent top-50% model protection across all population sizes.

        Returns:
            Elo threshold for preservation, or fallback_min_elo if no models exist
        """
        if not self.config.use_relative_threshold:
            return self.config.fallback_min_elo

        elo_service = _get_elo_service()
        if elo_service is None:
            return self.config.fallback_min_elo

        try:
            # Get all ratings for this config
            conn = elo_service._get_connection()
            cursor = conn.execute("""
                SELECT rating FROM elo_ratings
                WHERE board_type = ? AND num_players = ?
                AND games_played >= ?
                ORDER BY rating DESC
            """, (board_type, num_players, self.config.min_games_for_preservation))

            ratings = [row[0] for row in cursor.fetchall()]

            if len(ratings) < 3:
                logger.debug(
                    f"Only {len(ratings)} models with sufficient games for {board_type}_{num_players}p, "
                    f"using fallback threshold {self.config.fallback_min_elo}"
                )
                return self.config.fallback_min_elo

            # Always use 50th percentile (top half) for top-50% protection
            half_idx = max(0, len(ratings) // 2)
            threshold = ratings[half_idx] if half_idx < len(ratings) else ratings[0]

            logger.debug(
                f"Preservation threshold for {board_type}_{num_players}p: "
                f"{threshold:.0f} (50th percentile, {len(ratings)} models)"
            )
            return threshold

        except Exception as e:
            logger.warning(f"Error computing preservation threshold: {e}")
            return self.config.fallback_min_elo

    def should_preserve(
        self,
        checkpoint_path: Path,
        board_type: str,
        num_players: int,
    ) -> bool:
        """Check if checkpoint should be preserved based on Elo rating.

        Uses relative thresholds based on the distribution of existing models:
        - If >10 models in top decile, preserves models in top 10%
        - Otherwise, preserves models in top 25% (quartile)

        Args:
            checkpoint_path: Path to the checkpoint file
            board_type: Board type (e.g., "hex8", "square8")
            num_players: Number of players (2, 3, or 4)

        Returns:
            True if checkpoint has high enough Elo and should be preserved
        """
        if not self.config.enabled:
            return False

        if not checkpoint_path.exists():
            return False

        # Compute content hash
        content_hash = self._compute_model_hash(checkpoint_path)
        if content_hash is None:
            return False

        # Check if already preserved
        for m in self._preserved_models:
            if m.content_hash == content_hash:
                logger.debug(f"Model already preserved: {checkpoint_path.name}")
                return False

        # Look up Elo rating
        elo_rating, games_played, participant_id = self._get_elo_for_hash(
            content_hash, board_type, num_players
        )

        if elo_rating is None:
            return False

        # Check minimum games threshold first
        if games_played < self.config.min_games_for_preservation:
            logger.debug(
                f"Games {games_played} below threshold {self.config.min_games_for_preservation} "
                f"for {checkpoint_path.name}"
            )
            return False

        # Compute relative Elo threshold based on existing model distribution
        elo_threshold = self._get_preservation_threshold(board_type, num_players)

        if elo_rating < elo_threshold:
            logger.debug(
                f"Elo {elo_rating:.0f} below relative threshold {elo_threshold:.0f} "
                f"for {checkpoint_path.name}"
            )
            return False

        logger.info(
            f"Model qualifies for preservation: {checkpoint_path.name} "
            f"(Elo={elo_rating:.0f} >= threshold {elo_threshold:.0f}, games={games_played})"
        )
        return True

    def preserve_checkpoint(
        self,
        checkpoint_path: Path,
        board_type: str,
        num_players: int,
    ) -> Path | None:
        """Preserve a high-Elo checkpoint to the preservation directory.

        Args:
            checkpoint_path: Path to the checkpoint file
            board_type: Board type
            num_players: Number of players

        Returns:
            Path to preserved file, or None if preservation failed
        """
        if not self.config.enabled:
            return None

        if not checkpoint_path.exists():
            logger.warning(f"Cannot preserve non-existent file: {checkpoint_path}")
            return None

        # Compute hash and get Elo info
        content_hash = self._compute_model_hash(checkpoint_path)
        if content_hash is None:
            return None

        elo_rating, games_played, participant_id = self._get_elo_for_hash(
            content_hash, board_type, num_players
        )

        # Generate preserved filename
        config_key = f"{board_type}_{num_players}p"
        timestamp = int(time.time())
        elo_str = f"elo{int(elo_rating or 0)}"
        preserved_name = f"preserved_{config_key}_{elo_str}_{timestamp}.pth"
        preserved_path = self._preservation_dir / config_key / preserved_name

        # Create config subdirectory
        preserved_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            # Copy file to preservation directory
            shutil.copy2(checkpoint_path, preserved_path)

            # Record preservation
            record = PreservedModel(
                original_path=str(checkpoint_path),
                preserved_path=str(preserved_path),
                board_type=board_type,
                num_players=num_players,
                content_hash=content_hash,
                elo_rating=elo_rating or 0,
                games_played=games_played,
                preserved_at=time.time(),
                participant_id=participant_id,
            )
            self._preserved_models.append(record)
            self._save_preserved_registry()

            logger.info(
                f"Preserved high-Elo checkpoint: {checkpoint_path.name} -> {preserved_path} "
                f"(Elo={elo_rating:.0f}, games={games_played})"
            )

            # Cleanup old preserved models if over limit
            self._cleanup_preserved(board_type, num_players)

            return preserved_path

        except (OSError, shutil.Error) as e:
            logger.error(f"Failed to preserve checkpoint {checkpoint_path}: {e}")
            return None

    def _cleanup_preserved(self, board_type: str, num_players: int) -> None:
        """Remove oldest preserved models when over limit for a config.

        Keeps highest-Elo models, removes oldest by preservation time.
        """
        config_key = f"{board_type}_{num_players}p"

        # Get models for this config, sorted by Elo (highest first)
        config_models = [
            m for m in self._preserved_models
            if m.board_type == board_type and m.num_players == num_players
        ]

        if len(config_models) <= self.config.max_preserved_per_config:
            return

        # Sort by Elo descending, then by preserved_at descending
        config_models.sort(key=lambda m: (-m.elo_rating, -m.preserved_at))

        # Remove oldest/lowest-Elo models over the limit
        to_remove = config_models[self.config.max_preserved_per_config:]

        for model in to_remove:
            try:
                preserved_path = Path(model.preserved_path)
                if preserved_path.exists():
                    preserved_path.unlink()
                    logger.info(f"Removed old preserved model: {preserved_path.name}")
                self._preserved_models.remove(model)
            except (OSError, ValueError) as e:
                logger.warning(f"Failed to cleanup preserved model: {e}")

        self._save_preserved_registry()

    def get_preserved_models(
        self,
        board_type: str | None = None,
        num_players: int | None = None,
    ) -> list[PreservedModel]:
        """Get list of preserved models, optionally filtered by config.

        Args:
            board_type: Filter by board type (optional)
            num_players: Filter by player count (optional)

        Returns:
            List of PreservedModel records
        """
        models = self._preserved_models

        if board_type is not None:
            models = [m for m in models if m.board_type == board_type]

        if num_players is not None:
            models = [m for m in models if m.num_players == num_players]

        return models

    def get_best_preserved(
        self,
        board_type: str,
        num_players: int,
    ) -> PreservedModel | None:
        """Get the highest-Elo preserved model for a config.

        Args:
            board_type: Board type
            num_players: Number of players

        Returns:
            PreservedModel with highest Elo, or None if none preserved
        """
        models = self.get_preserved_models(board_type, num_players)
        if not models:
            return None
        return max(models, key=lambda m: m.elo_rating)

    def get_current_threshold(
        self,
        board_type: str,
        num_players: int,
    ) -> dict[str, Any]:
        """Get the current preservation threshold for a config.

        Useful for monitoring and debugging the relative threshold logic.

        Args:
            board_type: Board type
            num_players: Number of players

        Returns:
            Dict with threshold info: elo_threshold, percentile_used, model_count
        """
        threshold = self._get_preservation_threshold(board_type, num_players)

        elo_service = _get_elo_service()
        model_count = 0
        percentile_used = "fallback"

        if elo_service is not None:
            try:
                conn = elo_service._get_connection()
                cursor = conn.execute("""
                    SELECT COUNT(*) FROM elo_ratings
                    WHERE board_type = ? AND num_players = ?
                    AND games_played >= ?
                """, (board_type, num_players, self.config.min_games_for_preservation))
                model_count = cursor.fetchone()[0]

                if model_count >= 3:
                    decile_count = model_count // 10
                    if decile_count > self.config.top_decile_min_count:
                        percentile_used = "decile (90th)"
                    elif model_count < 20:
                        percentile_used = "half (50th)"
                    else:
                        percentile_used = "quartile (75th)"
            except Exception:
                pass

        return {
            "elo_threshold": threshold,
            "percentile_used": percentile_used,
            "model_count": model_count,
            "min_games_required": self.config.min_games_for_preservation,
            "use_relative_threshold": self.config.use_relative_threshold,
        }


# Module-level accessor
_service_instance: ModelPreservationService | None = None


def get_preservation_service(
    config: PreservationConfig | None = None
) -> ModelPreservationService:
    """Get the singleton preservation service instance.

    Args:
        config: Optional configuration (only used on first call)

    Returns:
        ModelPreservationService instance
    """
    return ModelPreservationService.get_instance(config)


def should_preserve_checkpoint(
    checkpoint_path: Path,
    board_type: str,
    num_players: int,
) -> bool:
    """Convenience function to check if checkpoint should be preserved.

    Args:
        checkpoint_path: Path to checkpoint file
        board_type: Board type
        num_players: Number of players

    Returns:
        True if checkpoint has high Elo and should be preserved
    """
    service = get_preservation_service()
    return service.should_preserve(checkpoint_path, board_type, num_players)


def preserve_checkpoint(
    checkpoint_path: Path,
    board_type: str,
    num_players: int,
) -> Path | None:
    """Convenience function to preserve a checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        board_type: Board type
        num_players: Number of players

    Returns:
        Path to preserved file, or None if not preserved
    """
    service = get_preservation_service()
    return service.preserve_checkpoint(checkpoint_path, board_type, num_players)
