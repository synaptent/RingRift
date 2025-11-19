"""
Self-play data generation script for RingRift AI
"""

import os
import sys
import random
import numpy as np
from datetime import datetime
import time
import uuid

# Add parent directory to path to import app modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from app.models import GameState, BoardType, GamePhase, GameStatus, Player, TimeControl, BoardState, AIConfig
from app.ai.heuristic_ai import HeuristicAI
from app.ai.minimax_ai import MinimaxAI
from app.ai.mcts_ai import MCTSAI
from app.game_engine import GameEngine
from app.ai.neural_net import NeuralNetAI

def create_initial_game_state():
    """Create a fresh game state for self-play"""
    board = BoardState(
        type=BoardType.SQUARE8,
        size=8,
        stacks={},
        markers={},
        collapsedSpaces={},
        eliminatedRings={}
    )
    
    players = [
        Player(
            id="player1",
            username="AI_1",
            type="ai",
            playerNumber=1,
            isReady=True,
            timeRemaining=600000,
            aiDifficulty=5,
            ringsInHand=20, # Standard for 2 players
            eliminatedRings=0,
            territorySpaces=0
        ),
        Player(
            id="player2",
            username="AI_2",
            type="ai",
            playerNumber=2,
            isReady=True,
            timeRemaining=600000,
            aiDifficulty=5,
            ringsInHand=20,
            eliminatedRings=0,
            territorySpaces=0
        )
    ]
    
    return GameState(
        id=str(uuid.uuid4()),
        boardType=BoardType.SQUARE8,
        board=board,
        players=players,
        currentPhase=GamePhase.RING_PLACEMENT,
        currentPlayer=1,
        moveHistory=[],
        timeControl=TimeControl(initialTime=600, increment=5, type="standard"),
        gameStatus=GameStatus.ACTIVE,
        createdAt=datetime.now(),
        lastMoveAt=datetime.now(),
        isRated=False,
        maxPlayers=2,
        totalRingsInPlay=0,
        totalRingsEliminated=0,
        victoryThreshold=3, # Standard victory condition
        territoryVictoryThreshold=10 # Standard territory condition
    )

def run_self_play_game(ai1, ai2):
    """Run a single game of self-play"""
    game_state = create_initial_game_state()
    game_history = [] # List of (state_features, current_player)
    
    move_count = 0
    max_moves = 200 # Prevent infinite loops
    
    while game_state.game_status == GameStatus.ACTIVE and move_count < max_moves:
        current_player_num = game_state.current_player
        current_ai = ai1 if current_player_num == 1 else ai2
        
        # Record state features before move
        # We use the NeuralNetAI's feature extraction logic
        # Create a dummy NN AI just to access the extraction method
        # Ideally this should be a static utility
        dummy_nn = NeuralNetAI(1, AIConfig(difficulty=1))
        features = dummy_nn._extract_features(game_state)
        game_history.append((features, current_player_num))
        
        # Select move
        move = current_ai.select_move(game_state)
        
        if not move:
            # No valid moves - pass or game over?
            # In RingRift, if you can't move, you might lose or pass.
            # For simplicity, let's assume game over if no moves in placement/movement
            print(f"No valid moves for Player {current_player_num}. Game Over.")
            game_state.game_status = GameStatus.FINISHED
            game_state.winner = 2 if current_player_num == 1 else 1
            break
            
        # Apply move
        game_state = GameEngine.apply_move(game_state, move)
        
        # Update game state meta-data (turn switching, etc.)
        # GameEngine.apply_move is currently a simulation and doesn't update everything
        # We need to manually switch turns for this loop
        game_state.current_player = 2 if current_player_num == 1 else 1
        
        # Check victory conditions (simplified)
        # Real victory check is complex, we'll use a simple heuristic check or move limit
        # For data generation, we might rely on the heuristic evaluation to determine a winner
        # if the game goes too long, or implement proper victory checks.
        
        # Check ring elimination victory
        p1 = next(p for p in game_state.players if p.player_number == 1)
        p2 = next(p for p in game_state.players if p.player_number == 2)
        
        if p1.eliminated_rings >= game_state.victory_threshold:
            game_state.game_status = GameStatus.FINISHED
            game_state.winner = 1
        elif p2.eliminated_rings >= game_state.victory_threshold:
            game_state.game_status = GameStatus.FINISHED
            game_state.winner = 2
            
        move_count += 1
        
    # If max moves reached, evaluate winner by score
    if game_state.game_status == GameStatus.ACTIVE:
        score1 = ai1.evaluate_position(game_state)
        # ai2 evaluates from its perspective, so positive score is good for ai2
        score2 = ai2.evaluate_position(game_state) 
        
        if score1 > score2:
            game_state.winner = 1
        elif score2 > score1:
            game_state.winner = 2
        else:
            game_state.winner = 0 # Draw
            
    return game_history, game_state.winner

def generate_dataset(num_games=100, output_file="training_data.npy"):
    """Generate dataset from self-play games"""
    all_data = []
    
    print(f"Generating {num_games} games of self-play data...")
    
    # Initialize AIs
    # We can mix and match AI types
    ai1 = HeuristicAI(1, AIConfig(difficulty=5))
    ai2 = MinimaxAI(2, AIConfig(difficulty=3)) # Slightly weaker opponent for variety
    
    for i in range(num_games):
        if i % 10 == 0:
            print(f"Simulating game {i+1}/{num_games}...")
            
        history, winner = run_self_play_game(ai1, ai2)
        
        # Process history into training samples
        # Outcome: 1 if player won, -1 if lost, 0 if draw
        for features, player_num in history:
            outcome = 0.0
            if winner == player_num:
                outcome = 1.0
            elif winner == 0:
                outcome = 0.0
            else:
                outcome = -1.0
                
            all_data.append((features, outcome))
            
    # Save to file
    print(f"Saving {len(all_data)} samples to {output_file}...")
    np.save(output_file, np.array(all_data, dtype=object))
    print("Done!")

if __name__ == "__main__":
    # Ensure directory exists
    os.makedirs("ai-service/app/training/data", exist_ok=True)
    generate_dataset(num_games=10, output_file="ai-service/app/training/data/self_play_data.npy")