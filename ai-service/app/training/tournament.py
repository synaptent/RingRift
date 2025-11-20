"""
Tournament system for evaluating AI models
"""

import sys
import os
import logging
import torch
from typing import Dict, Optional
from datetime import datetime

# Add app to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

from app.ai.mcts_ai import MCTSAI
from app.game_engine import GameEngine
from app.models import (
    GameState, BoardType, BoardState, GamePhase, GameStatus, TimeControl,
    Player, AIConfig
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Tournament:
    def __init__(self, model_path_a: str, model_path_b: str, num_games: int = 20):
        self.model_path_a = model_path_a
        self.model_path_b = model_path_b
        self.num_games = num_games
        self.results = {"A": 0, "B": 0, "Draw": 0}
        
    def _create_ai(self, player_number: int, model_path: str) -> MCTSAI:
        """Create an AI instance with specific model weights"""
        config = AIConfig(difficulty=10, randomness=0.1, think_time=500)
        ai = MCTSAI(player_number, config)
        
        # Manually load weights if neural net exists
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
        logger.info(f"Starting tournament: {self.model_path_a} vs {self.model_path_b}")
        
        for i in range(self.num_games):
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
            
            winner = self._play_game(ai1, ai2)
            
            if winner == 1:
                self.results[p1_label] += 1
            elif winner == 2:
                self.results[p2_label] += 1
            else:
                self.results["Draw"] += 1
                
            logger.info(f"Game {i+1}/{self.num_games}: Winner {winner} ({p1_label if winner==1 else p2_label if winner==2 else 'Draw'})")
            
        logger.info(f"Tournament finished. Results: {self.results}")
        return self.results

    def _play_game(self, ai1: MCTSAI, ai2: MCTSAI) -> Optional[int]:
        """Play a single game"""
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
                state.game_status = GameStatus.FINISHED
                break
                
            state = GameEngine.apply_move(state, move)
            move_count += 1
            
        return state.winner

    def _create_initial_state(self) -> GameState:
        """Create initial game state"""
        # Simplified version of generate_data.create_initial_state
        size = 8
        rings = 18
        return GameState(
            id="tournament",
            boardType=BoardType.SQUARE8,
            board=BoardState(
                type=BoardType.SQUARE8,
                size=size,
                stacks={},
                markers={},
                collapsedSpaces={},
                eliminatedRings={}
            ),
            players=[
                Player(
                    id="p1", username="AI 1", type="ai", playerNumber=1,
                    isReady=True, timeRemaining=600, ringsInHand=rings,
                    eliminatedRings=0, territorySpaces=0, aiDifficulty=10
                ),
                Player(
                    id="p2", username="AI 2", type="ai", playerNumber=2,
                    isReady=True, timeRemaining=600, ringsInHand=rings,
                    eliminatedRings=0, territorySpaces=0, aiDifficulty=10
                )
            ],
            currentPhase=GamePhase.RING_PLACEMENT,
            currentPlayer=1,
            moveHistory=[],
            timeControl=TimeControl(initialTime=600, increment=0, type="blitz"),
            gameStatus=GameStatus.ACTIVE,
            createdAt=datetime.now(),
            lastMoveAt=datetime.now(),
            isRated=False,
            maxPlayers=2,
            totalRingsInPlay=rings * 2,
            totalRingsEliminated=0,
            victoryThreshold=19,
            territoryVictoryThreshold=33
        )


if __name__ == "__main__":
    if len(sys.argv) >= 3:
        t = Tournament(sys.argv[1], sys.argv[2])
        t.run()
    else:
        print("Usage: python tournament.py <model_a_path> <model_b_path>")