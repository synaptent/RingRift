"""Mixed Opponent Selfplay - Diverse opponent training with full harness variety.

This module enables selfplay with a configurable mix of opponent types:
- Random opponents - Pure exploration, maximum diversity
- Heuristic opponents - Fast tactical play
- MCTS opponents - Strong strategic play via Gumbel MCTS
- Minimax opponents (2-player only) - Alpha-beta search with NN/NNUE evaluation
- MaxN opponents (3-4 player only) - Multi-player search
- BRS opponents (3-4 player only) - Best Reply Search (fast multiplayer)
- PolicyOnly opponents - Direct neural network policy sampling
- Descent opponents - Gradient-based move selection

Default mix for 2-player: random(15%), heuristic(25%), mcts(25%), minimax(20%), policy_only(15%)
Default mix for 3-4 player: random(15%), heuristic(20%), mcts(25%), maxn(15%), brs(15%), policy_only(10%)

This creates a robust training curriculum where the model learns to handle
opponents of varying skill levels and search styles, improving generalization.

Usage:
    from app.training.mixed_opponent_selfplay import MixedOpponentSelfplayRunner

    runner = MixedOpponentSelfplayRunner.from_config(
        board_type="hex8",
        num_players=2,
        num_games=1000,
        opponent_mix={"random": 0.15, "heuristic": 0.25, "mcts": 0.25, "minimax": 0.20, "policy_only": 0.15}
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
    - MCTS: Monte Carlo Tree Search for strategic depth (Gumbel MCTS)
    - Minimax: Alpha-beta search (2-player only)
    - MaxN: Multi-player search (3-4 player only)
    - BRS: Best Reply Search - fast multiplayer (3-4 player only)
    - PolicyOnly: Direct neural network policy sampling
    - Descent: Gradient-based move selection

    The mix percentages are configurable and vary by player count to respect
    search algorithm constraints (e.g., minimax is 2-player only).
    """

    # Default mixes for different player counts
    DEFAULT_MIX_2P = {
        "random": 0.15,
        "heuristic": 0.25,
        "mcts": 0.25,
        "minimax": 0.20,
        "policy_only": 0.15,
    }

    DEFAULT_MIX_MULTIPLAYER = {
        "random": 0.15,
        "heuristic": 0.20,
        "mcts": 0.25,
        "maxn": 0.15,
        "brs": 0.15,
        "policy_only": 0.10,
    }

    # Opponent types and their player count restrictions
    OPPONENT_PLAYER_RESTRICTIONS = {
        "random": (2, 4),      # All player counts
        "heuristic": (2, 4),   # All player counts
        "mcts": (2, 4),        # All player counts
        "minimax": (2, 2),     # 2-player only (alpha-beta)
        "maxn": (3, 4),        # 3-4 player only
        "brs": (3, 4),         # 3-4 player only
        "policy_only": (2, 4), # All player counts
        "descent": (2, 4),     # All player counts
    }

    def __init__(self, config: SelfplayConfig, opponent_mix: dict[str, float] | None = None):
        """Initialize mixed opponent selfplay.

        Args:
            config: Selfplay configuration
            opponent_mix: Dict mapping opponent type to probability.
                         If None, uses player-count-appropriate defaults.
        """
        # Set engine mode to MIXED for metadata tracking
        config.engine_mode = EngineMode.MIXED
        super().__init__(config)

        # Select default mix based on player count
        if opponent_mix is None:
            if config.num_players == 2:
                opponent_mix = self.DEFAULT_MIX_2P.copy()
            else:
                opponent_mix = self.DEFAULT_MIX_MULTIPLAYER.copy()

        # Filter out opponents incompatible with this player count
        self.opponent_mix = self._filter_compatible_opponents(opponent_mix, config.num_players)

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
        self._minimax_ai = None
        self._maxn_ai = None
        self._brs_ai = None
        self._policy_only_ai = None
        self._descent_ai = None
        self._engine = None

    def _filter_compatible_opponents(
        self, mix: dict[str, float], num_players: int
    ) -> dict[str, float]:
        """Filter opponent mix to only include types compatible with player count.

        Args:
            mix: Original opponent mix
            num_players: Number of players in game

        Returns:
            Filtered mix with only compatible opponent types
        """
        filtered = {}
        for opponent_type, prob in mix.items():
            if prob <= 0:
                continue
            restrictions = self.OPPONENT_PLAYER_RESTRICTIONS.get(opponent_type, (2, 4))
            min_players, max_players = restrictions
            if min_players <= num_players <= max_players:
                filtered[opponent_type] = prob
            else:
                logger.debug(
                    f"Excluding {opponent_type} (requires {min_players}-{max_players}p) "
                    f"for {num_players}-player game"
                )
        return filtered

    def setup(self) -> None:
        """Initialize AI engines for each opponent type."""
        super().setup()

        from ..game_engine import GameEngine
        from ..ai.factory import AIFactory, create_mcts
        from ..models import AIConfig, AIType, BoardType
        from ..ai.gumbel_common import GUMBEL_BUDGET_QUALITY, GUMBEL_BUDGET_STANDARD
        from ..ai.harness import create_harness, HarnessType

        self._engine = GameEngine
        board_type = BoardType(self.config.board_type)

        logger.info(f"  Opponent mix: {self.opponent_mix}")

        # Get model path for neural network-based opponents
        model_path = self._get_model_path()

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
            # Use quality budget for MCTS (800 simulations for 2000+ Elo quality)
            self._mcts_ai = create_mcts(
                board_type=board_type.value,
                num_players=self.config.num_players,
                mode="standard",
                simulation_budget=GUMBEL_BUDGET_QUALITY,
                device=self.config.device or "cuda",
            )

        # Initialize Minimax AI (2-player only, if needed)
        if self.opponent_mix.get("minimax", 0) > 0 and self.config.num_players == 2:
            try:
                self._minimax_ai = create_harness(
                    HarnessType.MINIMAX,
                    model_path=model_path,
                    board_type=board_type,
                    num_players=2,
                    depth=4,  # Moderate depth for variety
                    difficulty=6,
                )
                logger.info("  Initialized Minimax opponent")
            except Exception as e:
                logger.warning(f"  Failed to initialize Minimax: {e}")
                self._minimax_ai = None

        # Initialize MaxN AI (3-4 player only, if needed)
        if self.opponent_mix.get("maxn", 0) > 0 and self.config.num_players >= 3:
            try:
                self._maxn_ai = create_harness(
                    HarnessType.MAXN,
                    model_path=model_path,
                    board_type=board_type,
                    num_players=self.config.num_players,
                    depth=3,  # Shallower for multiplayer (exponential branching)
                    difficulty=5,
                )
                logger.info("  Initialized MaxN opponent")
            except Exception as e:
                logger.warning(f"  Failed to initialize MaxN: {e}")
                self._maxn_ai = None

        # Initialize BRS AI (3-4 player only, if needed)
        if self.opponent_mix.get("brs", 0) > 0 and self.config.num_players >= 3:
            try:
                self._brs_ai = create_harness(
                    HarnessType.BRS,
                    model_path=model_path,
                    board_type=board_type,
                    num_players=self.config.num_players,
                    depth=4,  # Faster than MaxN, can go deeper
                    difficulty=5,
                )
                logger.info("  Initialized BRS opponent")
            except Exception as e:
                logger.warning(f"  Failed to initialize BRS: {e}")
                self._brs_ai = None

        # Initialize PolicyOnly AI (if needed)
        if self.opponent_mix.get("policy_only", 0) > 0:
            try:
                self._policy_only_ai = create_harness(
                    HarnessType.POLICY_ONLY,
                    model_path=model_path,
                    board_type=board_type,
                    num_players=self.config.num_players,
                    difficulty=5,
                )
                logger.info("  Initialized PolicyOnly opponent")
            except Exception as e:
                logger.warning(f"  Failed to initialize PolicyOnly: {e}")
                self._policy_only_ai = None

        # Initialize Descent AI (if needed)
        if self.opponent_mix.get("descent", 0) > 0:
            try:
                self._descent_ai = create_harness(
                    HarnessType.DESCENT,
                    model_path=model_path,
                    board_type=board_type,
                    num_players=self.config.num_players,
                    simulations=GUMBEL_BUDGET_STANDARD,  # 150 for balanced exploration
                    difficulty=5,
                )
                logger.info("  Initialized Descent opponent")
            except Exception as e:
                logger.warning(f"  Failed to initialize Descent: {e}")
                self._descent_ai = None

    def _get_model_path(self) -> str | None:
        """Get the model path for neural network-based opponents.

        Returns:
            Path to model checkpoint, or None if not available.
        """
        from pathlib import Path

        # Check config for explicit model path
        if hasattr(self.config, 'model_path') and self.config.model_path:
            return self.config.model_path

        # Try to find canonical model for this config
        config_key = f"{self.config.board_type}_{self.config.num_players}p"
        model_dir = Path(__file__).parent.parent.parent / "models"

        # Check for canonical model
        canonical_path = model_dir / f"canonical_{config_key}.pth"
        if canonical_path.exists():
            return str(canonical_path)

        # Check for ringrift_best symlink
        best_path = model_dir / f"ringrift_best_{config_key}.pth"
        if best_path.exists():
            return str(best_path)

        logger.debug(f"No model found for {config_key}, neural opponents may fail")
        return None

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

            # Select move based on opponent type
            move = self._get_move_for_opponent(opponent_type, state, current_player)
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

    def _get_move_for_opponent(
        self, opponent_type: str, state: "GameState", current_player: int
    ) -> "Move | None":
        """Get move from the appropriate AI based on opponent type.

        Args:
            opponent_type: Type of opponent ("random", "heuristic", "mcts", etc.)
            state: Current game state
            current_player: Current player number

        Returns:
            Selected move, or None if no move available
        """
        # Standard AI types using direct select_move
        if opponent_type == "random":
            if self._random_ai:
                return self._random_ai.select_move(state)
        elif opponent_type == "heuristic":
            ai = self._heuristic_ais.get(current_player)
            if ai:
                return ai.select_move(state)
        elif opponent_type == "mcts":
            if self._mcts_ai:
                return self._mcts_ai.select_move(state)

        # Harness-based AI types using evaluate() interface
        elif opponent_type == "minimax":
            if self._minimax_ai:
                try:
                    move, _ = self._minimax_ai.evaluate(state, current_player)
                    return move
                except Exception as e:
                    logger.debug(f"Minimax error: {e}")
        elif opponent_type == "maxn":
            if self._maxn_ai:
                try:
                    move, _ = self._maxn_ai.evaluate(state, current_player)
                    return move
                except Exception as e:
                    logger.debug(f"MaxN error: {e}")
        elif opponent_type == "brs":
            if self._brs_ai:
                try:
                    move, _ = self._brs_ai.evaluate(state, current_player)
                    return move
                except Exception as e:
                    logger.debug(f"BRS error: {e}")
        elif opponent_type == "policy_only":
            if self._policy_only_ai:
                try:
                    move, _ = self._policy_only_ai.evaluate(state, current_player)
                    return move
                except Exception as e:
                    logger.debug(f"PolicyOnly error: {e}")
        elif opponent_type == "descent":
            if self._descent_ai:
                try:
                    move, _ = self._descent_ai.evaluate(state, current_player)
                    return move
                except Exception as e:
                    logger.debug(f"Descent error: {e}")

        # Fallback to heuristic if the selected opponent is unavailable
        fallback_ai = self._heuristic_ais.get(current_player)
        if fallback_ai:
            logger.debug(f"Falling back to heuristic for {opponent_type}")
            return fallback_ai.select_move(state)

        # Last resort: random AI
        if self._random_ai:
            logger.debug(f"Falling back to random for {opponent_type}")
            return self._random_ai.select_move(state)

        return None

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
