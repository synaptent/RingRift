"""Mixed Opponent Selfplay - Diverse opponent training with random, heuristic, and MCTS mix.

This module enables selfplay with a configurable mix of opponent strengths:
- Random opponents (30% default) - Pure exploration, maximum diversity
- Heuristic opponents (40% default) - Fast tactical play
- MCTS opponents (30% default) - Strong strategic play

This creates a robust training curriculum where the model learns to handle
opponents of varying skill levels, improving generalization.

Usage:
    from app.training.mixed_opponent_selfplay import MixedOpponentSelfplayRunner

    runner = MixedOpponentSelfplayRunner.from_config(
        board_type="hex8",
        num_players=2,
        num_games=1000,
        opponent_mix={"random": 0.3, "heuristic": 0.4, "mcts": 0.3}
    )
    stats = runner.run()

Or via scripts/selfplay.py:
    python scripts/selfplay.py --board hex8 --num-players 2 --mixed-opponents --num-games 1000
"""

from __future__ import annotations

import logging
import random
import time
import uuid
from typing import TYPE_CHECKING

from .selfplay_config import EngineMode, SelfplayConfig
from .selfplay_runner import GameResult, SelfplayRunner

if TYPE_CHECKING:
    from ..models import BoardType, GameState, Move

logger = logging.getLogger(__name__)


class MixedOpponentSelfplayRunner(SelfplayRunner):
    """Selfplay runner with mixed opponent diversity.

    Generates games against a mix of opponent types:
    - Random: Pure random moves for maximum exploration
    - Heuristic: Fast heuristic AI for tactical diversity
    - MCTS: Monte Carlo Tree Search for strategic depth

    The mix percentages are configurable, defaulting to 30% random, 40% heuristic, 30% MCTS.
    This creates training data with diverse opponent behaviors, improving model robustness.
    """

    def __init__(self, config: SelfplayConfig, opponent_mix: dict[str, float] | None = None):
        """Initialize mixed opponent selfplay.

        Args:
            config: Selfplay configuration
            opponent_mix: Dict mapping opponent type to probability
                         Default: {"random": 0.3, "heuristic": 0.4, "mcts": 0.3}
        """
        # Set engine mode to MIXED for metadata tracking
        config.engine_mode = EngineMode.MIXED
        super().__init__(config)

        # Default opponent mix: 30% random, 40% heuristic, 30% MCTS
        self.opponent_mix = opponent_mix or {
            "random": 0.3,
            "heuristic": 0.4,
            "mcts": 0.3,
        }

        # Validate mix sums to ~1.0
        total = sum(self.opponent_mix.values())
        if abs(total - 1.0) > 0.01:
            logger.warning(
                f"Opponent mix probabilities sum to {total:.3f}, not 1.0. "
                "Normalizing to ensure valid distribution."
            )
            # Normalize to ensure valid probability distribution
            self.opponent_mix = {k: v / total for k, v in self.opponent_mix.items()}

        # Track opponent usage for reporting
        self._opponent_counts = {k: 0 for k in self.opponent_mix.keys()}

        # AI engines - initialized in setup()
        self._random_ai = None
        self._heuristic_ais = {}
        self._mcts_ai = None
        self._engine = None

    def setup(self) -> None:
        """Initialize AI engines for each opponent type."""
        super().setup()

        from ..game_engine import GameEngine
        from ..ai.factory import AIFactory, create_mcts
        from ..models import AIConfig, AIType, BoardType
        from ..ai.gumbel_common import GUMBEL_BUDGET_QUALITY

        self._engine = GameEngine
        board_type = BoardType(self.config.board_type)

        logger.info(f"  Opponent mix: {self.opponent_mix}")

        # Initialize random AI (if needed)
        if self.opponent_mix.get("random", 0) > 0:
            ai_config = AIConfig(
                board_type=board_type,
                num_players=self.config.num_players,
                difficulty=1,  # Minimum difficulty for random
            )
            # Random AI is just player 0 (will be used for any player)
            self._random_ai = AIFactory.create(
                AIType.RANDOM,
                player_number=1,
                config=ai_config,
            )

        # Initialize heuristic AIs (if needed)
        if self.opponent_mix.get("heuristic", 0) > 0:
            ai_config = AIConfig(
                board_type=board_type,
                num_players=self.config.num_players,
                difficulty=6,  # Mid-level heuristic
            )
            for p in range(1, self.config.num_players + 1):
                self._heuristic_ais[p] = AIFactory.create(
                    AIType.HEURISTIC,
                    player_number=p,
                    config=ai_config,
                )

        # Initialize MCTS AI (if needed)
        if self.opponent_mix.get("mcts", 0) > 0:
            # Use throughput budget for MCTS (faster, still reasonable quality)
            self._mcts_ai = create_mcts(
                board_type=board_type.value,
                num_players=self.config.num_players,
                mode="standard",
                simulation_budget=GUMBEL_BUDGET_QUALITY,  # 800 simulations (increased from 64 for 2000+ Elo quality)
                device=self.config.device or "cuda",
            )

    def select_opponent_for_game(self) -> str:
        """Select opponent type for next game based on configured mix.

        Returns:
            Opponent type: "random", "heuristic", or "mcts"
        """
        # Sample from weighted distribution
        opponent_types = list(self.opponent_mix.keys())
        weights = [self.opponent_mix[t] for t in opponent_types]

        selected = random.choices(opponent_types, weights=weights, k=1)[0]
        self._opponent_counts[selected] += 1

        return selected

    def run_game(self, game_idx: int) -> GameResult:
        """Run a single game with a randomly selected opponent type.

        Args:
            game_idx: Game index

        Returns:
            GameResult with game metadata including opponent type
        """
        from ..training.initial_state import create_initial_state
        from ..models import BoardType, GameStatus

        start_time = time.time()
        game_id = str(uuid.uuid4())

        # Select opponent type for this game
        opponent_type = self.select_opponent_for_game()

        # PFSP opponent selection (Phase 7 - December 2025)
        current_model, pfsp_opponent = self._get_pfsp_context()

        board_type = BoardType(self.config.board_type)
        initial_state = create_initial_state(board_type, self.config.num_players)
        state = initial_state
        moves = []
        move_objects = []

        max_moves = getattr(self.config, 'max_moves', 500)

        while state.game_status != GameStatus.COMPLETED and len(moves) < max_moves:
            current_player = state.current_player

            # Select AI based on opponent type
            if opponent_type == "random":
                ai = self._random_ai
            elif opponent_type == "heuristic":
                ai = self._heuristic_ais[current_player]
            elif opponent_type == "mcts":
                ai = self._mcts_ai
            else:
                # Fallback to heuristic
                logger.warning(f"Unknown opponent type {opponent_type}, using heuristic")
                ai = self._heuristic_ais[current_player]

            move = ai.select_move(state)
            if not move:
                break

            state = self._engine.apply_move(state, move)
            moves.append({"player": current_player, "move": str(move)})
            move_objects.append(move)

        duration_ms = (time.time() - start_time) * 1000
        winner = getattr(state, "winner", None)

        # Record PFSP result (Phase 7 - December 2025)
        if pfsp_opponent is not None:
            current_model_won = winner == 0
            is_draw = winner is None or winner < 0
            self.record_pfsp_result(current_model, pfsp_opponent, current_model_won, is_draw)

        return GameResult(
            game_id=game_id,
            winner=winner,
            num_moves=len(moves),
            duration_ms=duration_ms,
            moves=moves,
            metadata={
                "engine": "mixed_opponent",
                "engine_mode": self.config.engine_mode.value,
                "opponent_type": opponent_type,
                "board_type": self.config.board_type,
                "num_players": self.config.num_players,
                "source": self.config.source,
                "pfsp_opponent": pfsp_opponent,
            },
            initial_state=initial_state,
            final_state=state,
            move_objects=move_objects,
        )

    def teardown(self) -> None:
        """Log opponent usage statistics before teardown."""
        super().teardown()

        # Report opponent distribution
        total_games = sum(self._opponent_counts.values())
        if total_games > 0:
            logger.info("Opponent usage distribution:")
            for opponent_type, count in sorted(self._opponent_counts.items()):
                pct = 100 * count / total_games
                target_pct = 100 * self.opponent_mix.get(opponent_type, 0)
                logger.info(
                    f"  {opponent_type}: {count}/{total_games} ({pct:.1f}%, "
                    f"target: {target_pct:.1f}%)"
                )
