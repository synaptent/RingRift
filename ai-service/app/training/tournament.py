"""
Tournament system for evaluating AI models
"""

import logging
import os
import sys
import time
from datetime import datetime

from app.ai.descent_ai import DescentAI
from app.game_engine import GameEngine
from app.models import AIConfig, BoardState, BoardType, GamePhase, GameState, GameStatus, Player, TimeControl
from app.utils.progress_reporter import SoakProgressReporter

# Model cache for memory-efficient model sharing (2025-12)
try:
    from app.ai.model_cache import clear_model_cache
    HAS_MODEL_CACHE = True
except ImportError:
    HAS_MODEL_CACHE = False
    clear_model_cache = None  # type: ignore

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Victory reason categories for statistics tracking.
# These align with the canonical game engine victory conditions (R172, etc.):
# - "elimination": Player reached victory_threshold for eliminated rings.
# - "territory": Player reached territory_victory_threshold.
# - "last_player_standing": R172 LPS victory - sole player with real
#   actions for three consecutive rounds.
# - "structural": Global stalemate resolved by tie-breakers.
# - "unknown": Catch-all for edge cases.
VICTORY_REASONS = [
    "elimination",
    "territory",
    "last_player_standing",
    "structural",
    "unknown",
]


def infer_victory_reason(game_state: GameState) -> str:
    """Infer the victory reason from the final game state.

    This mirrors the victory condition ladder in GameEngine._check_victory:
    1. Ring elimination victory (victory_threshold reached).
    2. Territory victory (territory_victory_threshold reached).
    3. LPS victory (lps_exclusive_player_for_completed_round matched winner).
    4. Structural termination (no stacks, tie-breaker resolution).

    Args:
        game_state: The final GameState after the game has finished.

    Returns:
        One of the VICTORY_REASONS strings.
    """
    if game_state.game_status != GameStatus.COMPLETED:
        return "unknown"

    winner = game_state.winner
    if winner is None:
        return "unknown"

    # Check ring elimination victory.
    elim_rings = game_state.board.eliminated_rings
    eliminated_for_winner = elim_rings.get(str(winner), 0)
    if eliminated_for_winner >= game_state.victory_threshold:
        return "elimination"

    # Check territory victory.
    territory_counts: dict[int, int] = {}
    for p_id in game_state.board.collapsed_spaces.values():
        territory_counts[p_id] = territory_counts.get(p_id, 0) + 1
    threshold = game_state.territory_victory_threshold
    if territory_counts.get(winner, 0) >= threshold:
        return "territory"

    # Check LPS victory (R172).
    if game_state.lps_exclusive_player_for_completed_round == winner:
        return "last_player_standing"

    # Structural termination (no stacks remaining or tie-breaker resolution).
    if not game_state.board.stacks:
        return "structural"

    return "unknown"


class Tournament:
    def __init__(
        self,
        model_path_a: str,
        model_path_b: str,
        num_games: int = 20,
        k_elo: int = 32,
        board_type: BoardType = BoardType.SQUARE8,
        num_players: int = 2,
        max_moves: int = 10000,
    ):
        self.model_path_a = model_path_a
        self.model_path_b = model_path_b
        self.num_games = num_games
        self.board_type = board_type
        self.num_players = num_players
        self.max_moves = max_moves
        self.results = {"A": 0, "B": 0, "Draw": 0}
        # Simple Elo-like rating system for candidate (A) vs best (B).
        self.k_elo = k_elo
        self.ratings = {"A": 1500.0, "B": 1500.0}
        # Victory reason statistics: counts by reason for analysis.
        # Tracks R172 LPS wins separately from elimination/territory.
        self.victory_reasons: dict[str, int] = dict.fromkeys(VICTORY_REASONS, 0)

    def _create_ai(self, player_number: int, model_path: str) -> DescentAI:
        """Create an AI instance with specific model weights.

        Uses the full model path as nn_model_id so NeuralNetAI loads and caches
        the model through the standard model_cache infrastructure. This ensures
        models are shared efficiently across games and properly released.

        Note: NeuralNetAI supports .pth paths directly as nn_model_id (see
        neural_net.py line 3653-3673).
        """
        # Pass full path - NeuralNetAI detects .pth suffix and loads directly
        config = AIConfig(
            difficulty=10,
            randomness=0.1,
            think_time=500,
            rngSeed=None,
            nn_model_id=model_path,  # Full path supported since 2025-12
        )
        return DescentAI(player_number, config)

    def run(self) -> dict[str, int]:
        """Run the tournament.

        For 2-player games, alternates colors between models A and B.
        For 3-4 player games, rotates the candidate (A) through each seat
        position while other seats are filled with model B.
        """
        logger.info(
            "Starting tournament: %s vs %s (%s %dp)",
            self.model_path_a,
            self.model_path_b,
            self.board_type.value,
            self.num_players,
        )

        # Initialize progress reporter for time-based progress output (~10s intervals)
        model_a_name = os.path.basename(self.model_path_a)
        model_b_name = os.path.basename(self.model_path_b)
        progress_reporter = SoakProgressReporter(
            total_games=self.num_games,
            report_interval_sec=10.0,
            context_label=f"{model_a_name}_vs_{model_b_name}_{self.board_type.value}_{self.num_players}p",
        )

        for i in range(self.num_games):
            game_start_time = time.time()

            # Create AIs for all players
            # For multiplayer: rotate candidate (A) through seats
            candidate_seat = (i % self.num_players) + 1
            ais = {}
            seat_labels = {}

            for player_num in range(1, self.num_players + 1):
                if self.num_players == 2:
                    # 2-player: alternate colors
                    if i % 2 == 0:
                        model = self.model_path_a if player_num == 1 else self.model_path_b
                        label = "A" if player_num == 1 else "B"
                    else:
                        model = self.model_path_b if player_num == 1 else self.model_path_a
                        label = "B" if player_num == 1 else "A"
                else:
                    # Multiplayer: candidate sits in rotating seat
                    if player_num == candidate_seat:
                        model = self.model_path_a
                        label = "A"
                    else:
                        model = self.model_path_b
                        label = "B"

                ais[player_num] = self._create_ai(player_num, model)
                seat_labels[player_num] = label

            winner, final_state = self._play_game_multiplayer(ais)

            # Track victory reason for LPS and other victory types.
            victory_reason = infer_victory_reason(final_state)
            self.victory_reasons[victory_reason] += 1

            if winner and winner in seat_labels:
                winner_label = seat_labels[winner]
                self.results[winner_label] += 1
                self._update_elo(winner_label)
                winner_label_str = winner_label
            else:
                self.results["Draw"] += 1
                self._update_elo(None)
                winner_label_str = "Draw"

            logger.info(
                "Game %d/%d: Winner P%s (%s) via %s",
                i + 1,
                self.num_games,
                winner,
                winner_label_str,
                victory_reason,
            )

            # Record game completion for progress reporting
            game_duration = time.time() - game_start_time
            progress_reporter.record_game(
                moves=0,  # Move count tracked in _play_game but not returned
                duration_sec=game_duration,
            )

        # Emit final progress summary
        progress_reporter.finish()

        logger.info("Tournament finished. Results: %s", self.results)
        logger.info("Victory reasons: %s", self.victory_reasons)
        logger.info(
            "Final Elo ratings: A=%.1f, B=%.1f",
            self.ratings["A"],
            self.ratings["B"],
        )

        # Clear model cache after tournament to release GPU/MPS memory
        if HAS_MODEL_CACHE and clear_model_cache is not None:
            clear_model_cache()
            logger.debug("Cleared model cache after tournament")

        return self.results

    def _play_game(
        self, ai1: DescentAI, ai2: DescentAI
    ) -> tuple[int | None, GameState]:
        """Play a single 2-player game and return (winner, final_state).

        Returns:
            A tuple of (winner player number or None, final GameState).
        """
        return self._play_game_multiplayer({1: ai1, 2: ai2})

    def _play_game_multiplayer(
        self, ais: dict[int, DescentAI]
    ) -> tuple[int | None, GameState]:
        """Play a single game with any number of players.

        Args:
            ais: Dictionary mapping player number to AI instance.

        Returns:
            A tuple of (winner player number or None, final GameState).
        """
        # Initialize game state
        state = self._create_initial_state()
        move_count = 0
        termination_reason = None

        while state.game_status == GameStatus.ACTIVE and move_count < self.max_moves:
            current_player = state.current_player
            ai = ais.get(current_player)

            if ai is None:
                logger.error(f"No AI for player {current_player}")
                termination_reason = "no_ai"
                break

            move = ai.select_move(state)

            if not move:
                # No moves available - game engine should handle this
                # via forced elimination or LPS checks
                logger.warning(f"No move from P{current_player}, breaking")
                termination_reason = "no_moves"
                break

            state = GameEngine.apply_move(state, move)
            move_count += 1

        # Determine winner if game ended without one (2025-12-16 fix)
        winner = state.winner
        if winner is None and state.game_status == GameStatus.ACTIVE:
            # Game exited without natural completion - determine winner by tiebreaker
            if move_count >= self.max_moves:
                termination_reason = "max_moves"

            # Use ring count + territory as tiebreaker
            winner = self._determine_winner_by_tiebreaker(state)
            if winner is not None:
                logger.info(f"Determined winner P{winner} via tiebreaker ({termination_reason})")

        return winner, state

    def _determine_winner_by_tiebreaker(self, state: GameState) -> int | None:
        """Determine winner using tiebreaker rules when no natural victory.

        Tiebreaker priority (2025-12-16):
        1. Most eliminated rings
        2. Most territory spaces
        3. Fewest rings in hand (committed more to board)
        """
        best_player = None
        best_score = (-1, -1, float('inf'))  # (elim_rings, territory, -rings_in_hand)

        for player in state.players:
            elim_rings = state.board.eliminated_rings.get(str(player.player_number), 0)
            territory = 0
            for p_id in state.board.collapsed_spaces.values():
                if p_id == player.player_number:
                    territory += 1
            rings_in_hand = player.rings_in_hand

            score = (elim_rings, territory, -rings_in_hand)
            if score > best_score:
                best_score = score
                best_player = player.player_number

        return best_player

    def _update_elo(self, winner_label: str | None) -> None:
        """Update Elo-like ratings for candidate (A) and best (B)."""
        ra = self.ratings["A"]
        rb = self.ratings["B"]
        # Expected scores
        ea = 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))
        eb = 1.0 - ea

        if winner_label == "A":
            sa, sb = 1.0, 0.0
        elif winner_label == "B":
            sa, sb = 0.0, 1.0
        else:
            # Draw
            sa = sb = 0.5

        self.ratings["A"] = ra + self.k_elo * (sa - ea)
        self.ratings["B"] = rb + self.k_elo * (sb - eb)

    def _create_initial_state(self) -> GameState:
        """Create initial game state for any board type and player count.

        Uses centralized BOARD_CONFIGS and victory threshold calculations
        from app.rules.core to ensure consistency with the game engine.
        """
        from app.rules.core import (
            BOARD_CONFIGS,
            get_territory_victory_threshold,
            get_victory_threshold,
        )

        # Get board configuration
        if self.board_type in BOARD_CONFIGS:
            config = BOARD_CONFIGS[self.board_type]
            size = config.size
            rings_per_player = config.rings_per_player
        else:
            # Fallback to square8-style defaults
            size = 8
            rings_per_player = 18

        # Create players
        players = [
            Player(
                id=f"p{idx}",
                username=f"AI {idx}",
                type="ai",
                playerNumber=idx,
                isReady=True,
                timeRemaining=600,
                ringsInHand=rings_per_player,
                eliminatedRings=0,
                territorySpaces=0,
                aiDifficulty=10,
            )
            for idx in range(1, self.num_players + 1)
        ]

        total_rings = rings_per_player * self.num_players
        victory_threshold = get_victory_threshold(self.board_type, self.num_players)
        territory_threshold = get_territory_victory_threshold(self.board_type)

        return GameState(
            id="tournament",
            boardType=self.board_type,
            rngSeed=None,
            board=BoardState(
                type=self.board_type,
                size=size,
                stacks={},
                markers={},
                collapsedSpaces={},
                eliminatedRings={},
            ),
            players=players,
            currentPhase=GamePhase.RING_PLACEMENT,
            currentPlayer=1,
            moveHistory=[],
            timeControl=TimeControl(
                initialTime=600,
                increment=0,
                type="blitz",
            ),
            gameStatus=GameStatus.ACTIVE,
            createdAt=datetime.now(),
            lastMoveAt=datetime.now(),
            isRated=False,
            maxPlayers=self.num_players,
            totalRingsInPlay=total_rings,
            totalRingsEliminated=0,
            victoryThreshold=victory_threshold,
            territoryVictoryThreshold=territory_threshold,
            chainCaptureState=None,
            mustMoveFromStackKey=None,
            zobristHash=None,
            lpsRoundIndex=0,
            lpsExclusivePlayerForCompletedRound=None,
        )


def run_tournament(
    model_a_path: str,
    model_b_path: str,
    num_games: int = 20,
    board_type: BoardType = BoardType.SQUARE8,
    num_players: int = 2,
    max_moves: int = 10000,
    seed: int | None = None,
) -> dict[str, any]:
    """
    Convenience function to run a tournament between two models.

    Parameters
    ----------
    model_a_path : str
        Path to the candidate model checkpoint.
    model_b_path : str
        Path to the baseline model checkpoint.
    num_games : int
        Number of games to play (alternating colors for 2p, rotating seats for 3-4p).
    board_type : BoardType
        Board type to use (square8, square19, hexagonal all supported).
    num_players : int
        Number of players (2, 3, or 4 supported).
    max_moves : int
        Maximum moves per game.
    seed : Optional[int]
        Random seed for reproducibility.

    Returns
    -------
    Dict[str, any]
        Results dictionary with keys:
        - model_a_wins: Number of wins for model A
        - model_b_wins: Number of wins for model B
        - draws: Number of draws
        - total_games: Total games played
        - avg_game_length: Average game length in moves
        - victory_reasons: Dict of victory reason counts
        - board_type: Board type used
        - num_players: Number of players
    """
    # Validate player count
    if num_players < 2:
        num_players = 2
    if num_players > 4:
        num_players = 4

    tournament = Tournament(
        model_path_a=model_a_path,
        model_path_b=model_b_path,
        num_games=num_games,
        board_type=board_type,
        num_players=num_players,
        max_moves=max_moves,
    )

    results = tournament.run()

    return {
        "model_a_wins": results.get("A", 0),
        "model_b_wins": results.get("B", 0),
        "draws": results.get("Draw", 0),
        "total_games": num_games,
        "avg_game_length": 0,  # Not tracked by Tournament class currently
        "victory_reasons": tournament.victory_reasons,
        "board_type": board_type.value,
        "num_players": num_players,
        "elo_ratings": tournament.ratings,
    }


def run_tournament_adaptive(
    model_a_path: str,
    model_b_path: str,
    promotion_threshold: float = 0.55,
    confidence: float = 0.95,
    min_games: int = 30,
    max_games: int = 300,
    batch_size: int = 20,
    ci_width_target: float = 0.04,
    board_type: BoardType = BoardType.SQUARE8,
    num_players: int = 2,
    max_moves: int = 10000,
    seed: int | None = None,
) -> dict[str, any]:
    """
    Run an adaptive tournament that stops early when statistically decisive.

    This function runs games in batches and computes Wilson confidence intervals
    after each batch. It stops early when:
    1. Lower bound > threshold (obvious winner - promote)
    2. Upper bound < threshold (obvious loser - reject)
    3. CI width < ci_width_target (sufficient precision achieved)

    This approach:
    - Promotes obvious winners quickly (30-50 games)
    - Rejects obvious losers quickly (30-50 games)
    - Runs more games for marginal cases (200-300)

    Parameters
    ----------
    model_a_path : str
        Path to the candidate model checkpoint.
    model_b_path : str
        Path to the baseline model checkpoint.
    promotion_threshold : float
        Win rate threshold for promotion (default 0.55).
    confidence : float
        Confidence level for Wilson interval (default 0.95).
    min_games : int
        Minimum games to play before allowing early stopping (default 30).
    max_games : int
        Maximum games to play (default 300).
    batch_size : int
        Number of games per batch (default 20).
    ci_width_target : float
        Target confidence interval width for stopping (default 0.04 = ±2%).
    board_type : BoardType
        Board type to use.
    num_players : int
        Number of players (2, 3, or 4).
    max_moves : int
        Maximum moves per game.
    seed : Optional[int]
        Random seed for reproducibility.

    Returns
    -------
    Dict[str, any]
        Results dictionary with keys:
        - model_a_wins: Number of wins for model A
        - model_b_wins: Number of wins for model B
        - draws: Number of draws
        - total_games: Total games played
        - win_rate: Final win rate
        - ci_lower: Wilson lower bound
        - ci_upper: Wilson upper bound
        - early_stopped: Whether evaluation stopped early
        - stop_reason: Reason for stopping ('obvious_winner', 'obvious_loser',
                       'ci_converged', 'max_games')
    """
    from app.training.significance import wilson_score_interval

    # Validate player count
    if num_players < 2:
        num_players = 2
    if num_players > 4:
        num_players = 4

    total_wins = 0
    total_losses = 0
    total_draws = 0
    games_played = 0
    victory_reasons: dict[str, int] = dict.fromkeys(VICTORY_REASONS, 0)

    current_seed = seed if seed is not None else int(time.time())
    stop_reason = "max_games"
    early_stopped = False

    while games_played < max_games:
        # Determine batch size for this iteration
        remaining = max_games - games_played
        current_batch = min(batch_size, remaining)

        # Run a batch of games
        tournament = Tournament(
            model_path_a=model_a_path,
            model_path_b=model_b_path,
            num_games=current_batch,
            board_type=board_type,
            num_players=num_players,
            max_moves=max_moves,
        )

        # Set seed for reproducibility
        if seed is not None:
            tournament.seed = current_seed
            current_seed += current_batch

        results = tournament.run()

        # Accumulate results
        total_wins += results.get("A", 0)
        total_losses += results.get("B", 0)
        total_draws += results.get("Draw", 0)
        games_played += current_batch

        # Merge victory reasons
        for reason, count in tournament.victory_reasons.items():
            victory_reasons[reason] = victory_reasons.get(reason, 0) + count

        # Check for early stopping after minimum games
        if games_played >= min_games:
            total = total_wins + total_losses + total_draws
            if total > 0:
                win_rate = total_wins / total
                ci_lower, ci_upper = wilson_score_interval(
                    total_wins, total, confidence=confidence
                )
                ci_width = ci_upper - ci_lower

                # Check stopping conditions
                if ci_lower > promotion_threshold:
                    # Obvious winner - lower bound exceeds threshold
                    stop_reason = "obvious_winner"
                    early_stopped = True
                    logger.info(
                        f"Early stop: obvious winner after {games_played} games "
                        f"(win_rate={win_rate:.1%}, CI_lower={ci_lower:.1%} > {promotion_threshold:.1%})"
                    )
                    break
                elif ci_upper < promotion_threshold:
                    # Obvious loser - upper bound below threshold
                    stop_reason = "obvious_loser"
                    early_stopped = True
                    logger.info(
                        f"Early stop: obvious loser after {games_played} games "
                        f"(win_rate={win_rate:.1%}, CI_upper={ci_upper:.1%} < {promotion_threshold:.1%})"
                    )
                    break
                elif ci_width <= ci_width_target:
                    # Sufficient precision achieved
                    stop_reason = "ci_converged"
                    early_stopped = True
                    logger.info(
                        f"Early stop: CI converged after {games_played} games "
                        f"(win_rate={win_rate:.1%}, CI_width={ci_width:.1%} <= {ci_width_target:.1%})"
                    )
                    break

    # Compute final statistics
    total = total_wins + total_losses + total_draws
    win_rate = total_wins / total if total > 0 else 0.0
    ci_lower, ci_upper = wilson_score_interval(
        total_wins, total, confidence=confidence
    ) if total > 0 else (0.0, 0.0)

    return {
        "model_a_wins": total_wins,
        "model_b_wins": total_losses,
        "draws": total_draws,
        "total_games": games_played,
        "win_rate": win_rate,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "early_stopped": early_stopped,
        "stop_reason": stop_reason,
        "victory_reasons": victory_reasons,
        "board_type": board_type.value,
        "num_players": num_players,
    }


if __name__ == "__main__":
    if len(sys.argv) >= 3:
        t = Tournament(sys.argv[1], sys.argv[2])
        t.run()
    else:
        print("Usage: python tournament.py <model_a_path> <model_b_path>")
