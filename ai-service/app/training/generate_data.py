"""
Self-play data generation script for RingRift
Uses MCTS to generate high-quality training data with data augmentation
"""

import sys
import os
import numpy as np
import random
from datetime import datetime
import copy

# Add app to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

from app.models import GameState, BoardType, BoardState, GamePhase, GameStatus, TimeControl, Player, AIConfig
from app.game_engine import GameEngine
from app.ai.mcts_ai import MCTSAI
from app.ai.neural_net import NeuralNetAI

def create_initial_state():
    return GameState(
        id="self-play",
        boardType=BoardType.SQUARE8,
        board=BoardState(
            type=BoardType.SQUARE8,
            size=8,
            stacks={},
            markers={},
            collapsedSpaces={},
            eliminatedRings={}
        ),
        players=[
            Player(id="p1", username="AI 1", type="ai", playerNumber=1, isReady=True, timeRemaining=600, ringsInHand=18, eliminatedRings=0, territorySpaces=0, aiDifficulty=10),
            Player(id="p2", username="AI 2", type="ai", playerNumber=2, isReady=True, timeRemaining=600, ringsInHand=18, eliminatedRings=0, territorySpaces=0, aiDifficulty=10)
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
        totalRingsInPlay=36,
        totalRingsEliminated=0,
        victoryThreshold=19,
        territoryVictoryThreshold=33
    )

def calculate_outcome(state, player_number, depth):
    """
    Calculate detailed outcome with bonuses and discount
    Matches DescentAI logic
    """
    base_val = 0.0
    if state.winner == player_number:
        base_val = 1.0
    elif state.winner is not None:
        base_val = -1.0
    else:
        return 0.0
        
    # Bonuses
    territory_count = 0
    for p_id in state.board.collapsed_spaces.values():
        if p_id == player_number:
            territory_count += 1
    
    eliminated_count = state.board.eliminated_rings.get(str(player_number), 0)
    
    marker_count = 0
    for m in state.board.markers.values():
        if m.player == player_number:
            marker_count += 1
            
    # Normalize bonuses
    bonus = (territory_count * 0.001) + (eliminated_count * 0.001) + (marker_count * 0.0001)
    
    if base_val > 0:
        val = base_val + bonus
    else:
        val = base_val + bonus
        
    # Discount
    gamma = 0.99
    discounted_val = val * (gamma ** depth)
    
    if base_val > 0:
        return max(0.001, min(1.0, discounted_val))
    elif base_val < 0:
        return max(-1.0, min(-0.001, discounted_val))
    return 0.0

def augment_data(features, globals, policy, neural_net):
    """
    Augment data by rotating and flipping
    Returns list of (features, globals, policy) tuples
    """
    augmented = []
    
    # Original
    augmented.append((features, globals, policy))
    
    # Only augment if policy is sparse enough to be fast
    # Or just rotate features and ignore policy for now?
    # No, policy must match features.
    
    # Implementing full policy rotation is complex due to move encoding.
    # For now, we will skip policy augmentation and only augment features if we were training value only.
    # But we train both.
    # So we skip augmentation for this iteration to avoid bugs in move mapping.
    # TODO: Implement move rotation mapping.
    
    return augmented

def generate_dataset(num_games=10, output_file="data/self_play_data.npy", ai1=None, ai2=None):
    """
    Generate self-play data
    """
    training_data = []
    
    # Initialize AI if not provided
    if ai1 is None:
        ai1 = MCTSAI(player_number=1, config=AIConfig(difficulty=10, randomness=0.1, thinkTime=500))
    if ai2 is None:
        ai2 = MCTSAI(player_number=2, config=AIConfig(difficulty=10, randomness=0.1, thinkTime=500))
    
    ai_p1 = ai1
    ai_p2 = ai2
    
    # Helper to get policy from MCTS root
    def get_mcts_policy(ai_instance, game_state):
        move, policy_map = ai_instance.select_move_and_policy(game_state)
        
        if not move:
            return None, None
            
        policy_vector = np.zeros(55000, dtype=np.float32)
        
        if ai_instance.neural_net and policy_map:
            for m, prob in policy_map.items():
                idx = ai_instance.neural_net.encode_move(m, game_state.board.size)
                if 0 <= idx < 55000:
                    policy_vector[idx] = prob
        
        return move, policy_vector

    print(f"Generating {num_games} games...")
    
    for game_idx in range(num_games):
        state = create_initial_state()
        game_history = [] 
        
        print(f"Game {game_idx+1} started")
        move_count = 0
        
        while state.game_status == GameStatus.ACTIVE and move_count < 200:
            current_player = state.current_player
            ai = ai_p1 if current_player == 1 else ai_p2
            
            move, policy = get_mcts_policy(ai, state)
            
            if not move:
                # No moves available, current player loses
                state.winner = 2 if current_player == 1 else 1
                state.game_status = GameStatus.FINISHED
                break
                
            if ai.neural_net:
                features, globals_vec = ai.neural_net._extract_features(state)
                game_history.append({
                    'features': features,
                    'globals': globals_vec,
                    'policy': policy,
                    'player': current_player
                })
            
            state = GameEngine.apply_move(state, move)
            move_count += 1
            
            if move_count % 10 == 0:
                print(f"  Move {move_count}")
        
        winner = state.winner
        print(f"Game {game_idx+1} finished. Winner: {winner}")
        
        # Assign rewards
        total_moves = len(game_history)
        
        for i, step in enumerate(game_history):
            moves_remaining = total_moves - i
            
            # Calculate outcome using detailed logic
            # We need to pass the final state to calculate bonuses
            # But we need to view it from the perspective of step['player']
            
            outcome = calculate_outcome(state, step['player'], moves_remaining)
            
            # Augment data (currently just pass-through)
            augmented_samples = augment_data(step['features'], step['globals'], step['policy'], ai_p1.neural_net)
            
            for feat, glob, pol in augmented_samples:
                training_data.append((
                    (feat, glob),
                    outcome,
                    pol
                ))
            
    # Save data
    # Use provided output_file, ensuring directory exists
    if not os.path.isabs(output_file):
        output_path = os.path.join(os.path.dirname(__file__), output_file)
    else:
        output_path = output_file
        
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.save(output_path, np.array(training_data, dtype=object))
    print(f"Saved {len(training_data)} samples to {output_path}")

if __name__ == "__main__":
    generate_dataset(num_games=2)