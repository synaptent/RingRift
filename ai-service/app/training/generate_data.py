"""
Self-play data generation script for RingRift
Uses MCTS to generate high-quality training data with data augmentation
"""

import sys
import os
import numpy as np
from datetime import datetime
from collections import deque

# Add app to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

from app.ai.mcts_ai import MCTSAI  # noqa: E402
from app.game_engine import GameEngine  # noqa: E402
from app.models import (  # noqa: E402
    GameState, BoardType, BoardState, GamePhase, GameStatus, TimeControl,
    Player, AIConfig
)


def create_initial_state(board_type=BoardType.SQUARE8):
    """Create initial state for self-play, supporting different board types"""
    if board_type == BoardType.SQUARE8:
        size = 8
        rings = 18
        victory_threshold = 19
        territory_threshold = 33
    elif board_type == BoardType.HEXAGONAL:
        size = 11  # Side length 11? Or radius? Rules say 11 spaces per side.
        # Assuming size parameter represents radius or side length for
        # BoardManager. BoardManager usually expects radius for hex.
        # If 11 per side, radius is likely 10 or 11.
        # Let's assume standard hex size for now.
        rings = 36
        victory_threshold = 37  # >36
        territory_threshold = 166  # >165
    else:
        # Default/Fallback
        size = 8
        rings = 18
        victory_threshold = 19
        territory_threshold = 33
        
    return GameState(
        id="self-play",
        boardType=board_type,
        board=BoardState(
            type=board_type,
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
        victoryThreshold=victory_threshold,
        territoryVictoryThreshold=territory_threshold
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
    bonus = (
        (territory_count * 0.001) +
        (eliminated_count * 0.001) +
        (marker_count * 0.0001)
    )
    
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
    
    # Rotate 90, 180, 270
    # Policy rotation requires re-mapping the policy indices.
    # We use the neural_net instance to decode, transform, and re-encode moves.
    
    # Helper to transform a policy vector
    def transform_policy(policy_vec, k_rot, flip_h):
        if np.all(policy_vec == 0):
            return policy_vec
            
        new_policy = np.zeros_like(policy_vec)
        board_size = neural_net.board_size
        
        # Iterate over non-zero policy indices
        indices = np.nonzero(policy_vec)[0]
        
        # Create a dummy game state for decoding context if needed
        # (decode_move mostly needs board size which is in neural_net)
        # We pass a dummy state with current_player=1 just to satisfy signature
        from app.models import (
            GameState, BoardState, BoardType, TimeControl, GameStatus,
            GamePhase
        )
        dummy_state = GameState(
            id="dummy", boardType=BoardType.SQUARE8,
            board=BoardState(type=BoardType.SQUARE8, size=board_size),
            players=[], currentPhase=GamePhase.MOVEMENT, currentPlayer=1,
            moveHistory=[],
            timeControl=TimeControl(
                initialTime=0, increment=0, type="blitz"
            ),
            gameStatus=GameStatus.ACTIVE,
            createdAt=datetime.now(),
            lastMoveAt=datetime.now(),
            isRated=False, maxPlayers=2, totalRingsInPlay=0,
            totalRingsEliminated=0,
            victoryThreshold=0, territoryVictoryThreshold=0
        )

        for idx in indices:
            prob = policy_vec[idx]
            move = neural_net.decode_move(idx, dummy_state)
            
            if not move:
                continue
                
            # Transform move coordinates
            # 1. Rotate k times (90 deg counter-clockwise)
            # 2. Flip horizontal if needed
            
            # Helper to rotate point (x, y) in N*N grid k times
            def rotate_point(x, y, n, k):
                for _ in range(k):
                    x, y = y, n - 1 - x
                return x, y
                
            # Helper to flip point (x, y) horizontally
            def flip_point(x, y, n):
                return n - 1 - x, y
            
            # Transform 'to'
            tx, ty = move.to.x, move.to.y
            tx, ty = rotate_point(tx, ty, board_size, k_rot)
            if flip_h:
                tx, ty = flip_point(tx, ty, board_size)
            move.to.x, move.to.y = tx, ty
            
            # Transform 'from' if exists
            if move.from_pos:
                fx, fy = move.from_pos.x, move.from_pos.y
                fx, fy = rotate_point(fx, fy, board_size, k_rot)
                if flip_h:
                    fx, fy = flip_point(fx, fy, board_size)
                move.from_pos.x, move.from_pos.y = fx, fy
                
            # Transform 'capture_target' if exists
            if move.capture_target:
                cx, cy = move.capture_target.x, move.capture_target.y
                cx, cy = rotate_point(cx, cy, board_size, k_rot)
                if flip_h:
                    cx, cy = flip_point(cx, cy, board_size)
                move.capture_target.x, move.capture_target.y = cx, cy
                
            # Re-encode
            new_idx = neural_net.encode_move(move, board_size)
            if 0 <= new_idx < len(new_policy):
                new_policy[new_idx] = prob
                
        return new_policy

    # Generate augmentations
    # Rotations: 1, 2, 3 (90, 180, 270)
    for k in range(1, 4):
        # Rotate features (C, H, W) - axes=(1, 2) means rotate along H and W
        rotated_features = np.rot90(features, k=k, axes=(1, 2))
        rotated_policy = transform_policy(policy, k, False)
        augmented.append((rotated_features, globals, rotated_policy))
        
    # Flip (Horizontal)
    flipped_features = np.flip(features, axis=2)
    flipped_policy = transform_policy(policy, 0, True)
    augmented.append((flipped_features, globals, flipped_policy))
    
    # Flip + Rotations
    for k in range(1, 4):
        # Match the transform_policy logic:
        # 1. Rotate k times
        # 2. Flip horizontal

        r_feat = np.rot90(features, k=k, axes=(1, 2))
        rf_features = np.flip(r_feat, axis=2)

        # Policy transform with k_rot=k, flip_h=True does exactly that order.
        rf_policy = transform_policy(policy, k, True)

        augmented.append((rf_features, globals, rf_policy))
        
    return augmented


def generate_dataset(
    num_games=10, output_file="data/self_play_data.npy",
    ai1=None, ai2=None, board_type=BoardType.SQUARE8
):
    """
    Generate self-play data
    """
    training_data = []
    
    # Initialize AI if not provided
    if ai1 is None:
        ai1 = MCTSAI(
            player_number=1,
            config=AIConfig(difficulty=10, randomness=0.1, thinkTime=500)
        )
    if ai2 is None:
        ai2 = MCTSAI(
            player_number=2,
            config=AIConfig(difficulty=10, randomness=0.1, thinkTime=500)
        )
    
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
                idx = ai_instance.neural_net.encode_move(
                    m, game_state.board.size
                )
                if 0 <= idx < 55000:
                    policy_vector[idx] = prob
        
        return move, policy_vector

    print(f"Generating {num_games} games on {board_type}...")
    
    for game_idx in range(num_games):
        # Alternate board types if desired, or stick to one
        # For now, use the passed board_type
        state = create_initial_state(board_type)
        game_history = []
        
        # History buffer for feature stacking (length 3 + current = 4)
        # We store the raw 10-channel features here
        feature_history = deque(maxlen=4)
        
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
                
                # Update history
                feature_history.append(features)
                
                # Stack features
                # If history is not full, pad with the first frame (or zeros?)
                # NeuralNetAI logic usually repeats the current frame if
                # history is empty, but here we are building history over time.
                # Let's pad with the oldest available frame to match typical RL
                # behavior or just repeat the first frame if we have < 4.
                
                history_list = list(feature_history)
                while len(history_list) < 4:
                    # Prepend the first frame (padding)
                    history_list.insert(0, history_list[0])
                    
                # Stack along channel dimension (axis 0)
                # Each feature is (10, H, W)
                # Result should be (40, H, W)
                stacked_features = np.concatenate(history_list, axis=0)
                
                game_history.append({
                    'features': stacked_features,
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
            augmented_samples = augment_data(
                step['features'], step['globals'], step['policy'],
                ai_p1.neural_net
            )
            
            for feat, glob, pol in augmented_samples:
                training_data.append((
                    (feat, glob),
                    outcome,
                    pol
                ))
            
    # Save data with Experience Replay (Append mode)
    # Use provided output_file, ensuring directory exists
    if not os.path.isabs(output_file):
        output_path = os.path.join(os.path.dirname(__file__), output_file)
    else:
        output_path = output_file
        
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Load existing data if available
    existing_data = []
    if os.path.exists(output_path):
        try:
            existing_data = np.load(output_path, allow_pickle=True).tolist()
            print(f"Loaded {len(existing_data)} existing samples")
        except Exception as e:
            print(f"Could not load existing data: {e}")
            
    # Append new data
    combined_data = existing_data + training_data
    
    # Limit buffer size (Experience Replay Buffer)
    MAX_BUFFER_SIZE = 50000
    if len(combined_data) > MAX_BUFFER_SIZE:
        combined_data = combined_data[-MAX_BUFFER_SIZE:]
        print(f"Buffer full, keeping last {MAX_BUFFER_SIZE} samples")
        
    # TODO: Refactor to save as separate structured arrays (HDF5 or separate .npy)
    # for better scalability and mmap support.
    np.save(output_path, np.array(combined_data, dtype=object))
    print(
        f"Saved {len(training_data)} new samples. "
        f"Total buffer size: {len(combined_data)}"
    )


if __name__ == "__main__":
    generate_dataset(num_games=2)