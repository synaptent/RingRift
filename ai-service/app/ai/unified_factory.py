"""Unified Neural Network Factory for cross-model tournaments.

January 4, 2026 (Sprint 17.3): Created to enable cross-NN tournaments.
This factory provides a unified interface for loading neural network models
from checkpoint files and creating AI instances for tournament play.

Usage:
    from app.ai.neural_net import UnifiedNeuralNetFactory

    # Create AI from model path
    ai = UnifiedNeuralNetFactory.create(
        model_path="models/canonical_hex8_2p.pth",
        board_type="hex8",
        num_players=2,
    )
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

__all__ = ["UnifiedNeuralNetFactory"]


class UnifiedNeuralNetFactory:
    """Factory for creating neural network AI instances from model checkpoints.

    This factory wraps the game_gauntlet's get_ai_for_gauntlet function
    to provide a consistent interface for tournament_daemon's cross-NN matches.

    The factory supports:
    - CNN models (standard architecture)
    - GNN models (graph neural network)
    - Hybrid models (CNN+GNN ensemble)

    Example:
        # Load a trained model for tournament play
        ai = UnifiedNeuralNetFactory.create(
            model_path="models/canonical_hex8_2p.pth",
            board_type="hex8",
            num_players=2,
            temperature=0.1,  # Lower temperature for tournament play
        )

        # Use the AI in a game
        move = ai.choose_move(game_state)
    """

    @classmethod
    def create(
        cls,
        model_path: str | Path,
        board_type: str | Any,
        num_players: int = 2,
        player_number: int = 1,
        temperature: float = 0.1,
        model_type: str = "cnn",
    ) -> Any:
        """Create a neural network AI from a model checkpoint.

        Args:
            model_path: Path to the model checkpoint (.pth file)
            board_type: Board type string or BoardType enum
            num_players: Number of players (2, 3, or 4)
            player_number: Player number (1-indexed)
            temperature: Policy temperature (lower = more deterministic)
            model_type: Type of model - "cnn", "gnn", or "hybrid"

        Returns:
            AI instance (PolicyOnlyAI, UniversalAI, or GNNAI)

        Raises:
            FileNotFoundError: If model file doesn't exist
            ValueError: If model architecture is incompatible
            RuntimeError: If model loading fails

        Example:
            ai = UnifiedNeuralNetFactory.create(
                "models/canonical_hex8_2p.pth",
                board_type="hex8",
                num_players=2,
            )
        """
        # Convert board_type to string if it's an enum
        if hasattr(board_type, "value"):
            board_type_str = board_type.value
        else:
            board_type_str = str(board_type)

        # Resolve path
        path = Path(model_path)
        if not path.is_absolute():
            # Check common locations
            for prefix in [Path("."), Path("models"), Path("models/checkpoints")]:
                candidate = prefix / model_path
                if candidate.exists():
                    path = candidate
                    break

        if not path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        logger.info(
            f"UnifiedNeuralNetFactory: Loading {path} for "
            f"{board_type_str}_{num_players}p (player={player_number})"
        )

        # Use game_gauntlet's proven loading infrastructure
        try:
            from app.training.game_gauntlet import get_ai_for_gauntlet
            from app.models import BoardType

            # Convert string to BoardType enum
            try:
                board_type_enum = BoardType(board_type_str)
            except ValueError:
                # Try common aliases
                aliases = {
                    "sq8": "square8",
                    "sq19": "square19",
                    "hex": "hex8",
                }
                board_type_enum = BoardType(aliases.get(board_type_str, board_type_str))

            ai = get_ai_for_gauntlet(
                player=player_number,
                board_type=board_type_enum,
                model_path=str(path),
                temperature=temperature,
                num_players=num_players,
                model_type=model_type,
            )

            logger.info(
                f"UnifiedNeuralNetFactory: Successfully created AI "
                f"({type(ai).__name__}) from {path.name}"
            )
            return ai

        except ImportError as e:
            logger.error(f"Failed to import game_gauntlet: {e}")
            raise RuntimeError(
                f"Cannot create AI: game_gauntlet module not available: {e}"
            ) from e

    @classmethod
    def create_pair(
        cls,
        model_path_1: str | Path,
        model_path_2: str | Path,
        board_type: str | Any,
        num_players: int = 2,
        temperature: float = 0.1,
    ) -> tuple[Any, Any]:
        """Create a pair of AIs for head-to-head matches.

        Convenience method for creating two AIs with different player numbers.

        Args:
            model_path_1: Path to first model (will be player 1)
            model_path_2: Path to second model (will be player 2)
            board_type: Board type string or enum
            num_players: Number of players
            temperature: Policy temperature

        Returns:
            Tuple of (ai_1, ai_2)

        Example:
            ai_old, ai_new = UnifiedNeuralNetFactory.create_pair(
                "models/canonical_hex8_2p_v1.pth",
                "models/canonical_hex8_2p_v2.pth",
                board_type="hex8",
            )
        """
        ai_1 = cls.create(
            model_path=model_path_1,
            board_type=board_type,
            num_players=num_players,
            player_number=1,
            temperature=temperature,
        )
        ai_2 = cls.create(
            model_path=model_path_2,
            board_type=board_type,
            num_players=num_players,
            player_number=2,
            temperature=temperature,
        )
        return ai_1, ai_2

    @classmethod
    def verify_model(
        cls,
        model_path: str | Path,
        board_type: str | Any,
        num_players: int = 2,
    ) -> tuple[bool, str | None]:
        """Verify a model can be loaded for the given configuration.

        Args:
            model_path: Path to model checkpoint
            board_type: Expected board type
            num_players: Expected number of players

        Returns:
            Tuple of (success, error_message)
            If success is True, error_message is None.
            If success is False, error_message contains the reason.
        """
        try:
            ai = cls.create(
                model_path=model_path,
                board_type=board_type,
                num_players=num_players,
            )
            # Quick sanity check that it has the expected interface
            if not hasattr(ai, "choose_move"):
                return False, "AI instance missing choose_move method"
            return True, None
        except FileNotFoundError as e:
            return False, f"Model file not found: {e}"
        except ValueError as e:
            return False, f"Architecture mismatch: {e}"
        except RuntimeError as e:
            return False, f"Model loading failed: {e}"
        except Exception as e:
            return False, f"Unexpected error: {e}"
