"""
Tournament system for evaluating AI models
"""

import sys
import os
import logging
import time
import torch
from typing import Dict, Optional
from datetime import datetime

from app.ai.descent_ai import DescentAI
from app.game_engine import GameEngine
from app.utils.progress_reporter import SoakProgressReporter
from app.models import (
    GameState, BoardType, BoardState, GamePhase, GameStatus, TimeControl,
    Player, AIConfig
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Victory reason categories for statistics tracking.
# These align with the canonical game engine victory conditions (R172, etc.):
# - "elimination": Player reached victory_threshold for eliminated rings.
# - "territory": Player reached territory_victory_threshold.
# - "last_player_standing": R172 LPS victory - sole player with real
#   actions for two consecutive rounds.
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
    territory_counts: Dict[int, int] = {}
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
    ):
        self.model_path_a = model_path_a
        self.model_path_b = model_path_b
        self.num_games = num_games
        self.results = {"A": 0, "B": 0, "Draw": 0}
        # Simple Elo-like rating system for candidate (A) vs best (B).
        self.k_elo = k_elo
        self.ratings = {"A": 1500.0, "B": 1500.0}
        # Victory reason statistics: counts by reason for analysis.
        # Tracks R172 LPS wins separately from elimination/territory.
        self.victory_reasons: Dict[str, int] = {
            reason: 0 for reason in VICTORY_REASONS
        }

    def _create_ai(self, player_number: int, model_path: str) -> DescentAI:
        """Create an AI instance with specific model weights.

        The checkpoint basename (without .pth) is treated as the nn_model_id so
        that NeuralNetAI can load it via AIConfig.nn_model_id. We keep the
        manual load as a fallback in case older DescentAI/NeuralNetAI versions
        ignore nn_model_id.
        """
        model_id = os.path.splitext(os.path.basename(model_path))[0]
        config = AIConfig(
            difficulty=10,
            randomness=0.1,
            think_time=500,
            rngSeed=None,
            nn_model_id=model_id,
        )
        ai = DescentAI(player_number, config)

        # Fallback/manual load for robustness with legacy implementations.
        if ai.neural_net and os.path.exists(model_path):
            try:
                ai.neural_net.model.load_state_dict(
                    torch.load(model_path, weights_only=True)
                )
                ai.neural_net.model.eval()
            except Exception as e:
                logger.error(f"Failed to load model {model_path}: {e}")

        return ai

    def run(self) -> Dict[str, int]:
        """Run the tournament"""
        logger.info(
            "Starting tournament: %s vs %s",
            self.model_path_a,
            self.model_path_b,
        )

        # Initialize progress reporter for time-based progress output (~10s intervals)
        model_a_name = os.path.basename(self.model_path_a)
        model_b_name = os.path.basename(self.model_path_b)
        progress_reporter = SoakProgressReporter(
            total_games=self.num_games,
            report_interval_sec=10.0,
            context_label=f"{model_a_name}_vs_{model_b_name}",
        )

        for i in range(self.num_games):
            game_start_time = time.time()
            # Alternate colors
            if i % 2 == 0:
                p1_model = self.model_path_a
                p2_model = self.model_path_b
                p1_label = "A"
                p2_label = "B"
            else:
                p1_model = self.model_path_b
                p2_model = self.model_path_a
                p1_label = "B"
                p2_label = "A"

            ai1 = self._create_ai(1, p1_model)
            ai2 = self._create_ai(2, p2_model)

            winner, final_state = self._play_game(ai1, ai2)

            # Track victory reason for LPS and other victory types.
            victory_reason = infer_victory_reason(final_state)
            self.victory_reasons[victory_reason] += 1

            if winner == 1:
                self.results[p1_label] += 1
                self._update_elo(p1_label)
            elif winner == 2:
                self.results[p2_label] += 1
                self._update_elo(p2_label)
            else:
                self.results["Draw"] += 1
                self._update_elo(None)

            if winner == 1:
                winner_label_str = p1_label
            elif winner == 2:
                winner_label_str = p2_label
            else:
                winner_label_str = "Draw"
            logger.info(
                "Game %d/%d: Winner %s (%s) via %s",
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
        return self.results

    def _play_game(
        self, ai1: DescentAI, ai2: DescentAI
    ) -> tuple[Optional[int], GameState]:
        """Play a single game and return (winner, final_state).

        Returns:
            A tuple of (winner player number or None, final GameState).
        """
        # Initialize game state
        state = self._create_initial_state()
        move_count = 0

        while state.game_status == GameStatus.ACTIVE and move_count < 200:
            current_player = state.current_player
            ai = ai1 if current_player == 1 else ai2

            move = ai.select_move(state)

            if not move:
                # No moves available, current player loses
                state.winner = 2 if current_player == 1 else 1
                state.game_status = GameStatus.COMPLETED
                break

            state = GameEngine.apply_move(state, move)
            move_count += 1

        return state.winner, state

    def _update_elo(self, winner_label: Optional[str]) -> None:
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
        """Create initial game state"""
        # Simplified version of generate_data.create_initial_state
        size = 8
        rings = 18
        return GameState(
            id="tournament",
            boardType=BoardType.SQUARE8,
            rngSeed=None,
            board=BoardState(
                type=BoardType.SQUARE8,
                size=size,
                stacks={},
                markers={},
                collapsedSpaces={},
                eliminatedRings={},
            ),
            players=[
                Player(
                    id="p1",
                    username="AI 1",
                    type="ai",
                    playerNumber=1,
                    isReady=True,
                    timeRemaining=600,
                    ringsInHand=rings,
                    eliminatedRings=0,
                    territorySpaces=0,
                    aiDifficulty=10,
                ),
                Player(
                    id="p2",
                    username="AI 2",
                    type="ai",
                    playerNumber=2,
                    isReady=True,
                    timeRemaining=600,
                    ringsInHand=rings,
                    eliminatedRings=0,
                    territorySpaces=0,
                    aiDifficulty=10,
                ),
            ],
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
            maxPlayers=2,
            totalRingsInPlay=rings * 2,
            totalRingsEliminated=0,
            # Per RR-CANON-R061: victoryThreshold = round(ringsPerPlayer × (2/3 + 1/3 × (numPlayers - 1)))
            # For 2p: round(18 × (2/3 + 1/3 × 1)) = round(18 × 1) = 18
            victoryThreshold=round(rings * (2 / 3 + (1 / 3) * (2 - 1))),
            # Per RR-CANON-R062: territoryVictoryThreshold = floor(totalSpaces / 2) + 1
            territoryVictoryThreshold=(size * size) // 2 + 1,  # 33 for 8x8
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
    max_moves: int = 200,
    seed: Optional[int] = None,
) -> Dict[str, any]:
    """
    Convenience function to run a tournament between two models.

    Parameters
    ----------
    model_a_path : str
        Path to the candidate model checkpoint.
    model_b_path : str
        Path to the baseline model checkpoint.
    num_games : int
        Number of games to play (alternating colors).
    board_type : BoardType
        Board type to use (currently only SQUARE8 fully supported).
    num_players : int
        Number of players (currently only 2 supported).
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
    """
    # Currently Tournament class only supports SQUARE8 2-player
    # TODO: Extend Tournament class for other board types and player counts
    if board_type != BoardType.SQUARE8:
        logger.warning(
            "Tournament currently only supports SQUARE8, using SQUARE8 instead of %s",
            board_type.value,
        )

    if num_players != 2:
        logger.warning(
            "Tournament currently only supports 2 players, using 2 instead of %d",
            num_players,
        )

    tournament = Tournament(
        model_path_a=model_a_path,
        model_path_b=model_b_path,
        num_games=num_games,
    )

    results = tournament.run()

    return {
        "model_a_wins": results.get("A", 0),
        "model_b_wins": results.get("B", 0),
        "draws": results.get("Draw", 0),
        "total_games": num_games,
        "avg_game_length": 0,  # Not tracked by Tournament class currently
        "victory_reasons": tournament.victory_reasons,
    }


if __name__ == "__main__":
    if len(sys.argv) >= 3:
        t = Tournament(sys.argv[1], sys.argv[2])
        t.run()
    else:
        print("Usage: python tournament.py <model_a_path> <model_b_path>")
