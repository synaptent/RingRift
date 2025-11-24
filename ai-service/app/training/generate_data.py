"""
Self-play data generation script for RingRift
Uses MCTS to generate high-quality training data with data augmentation
"""

import sys
import os
import numpy as np
from datetime import datetime

from app.ai.descent_ai import DescentAI  # noqa: E402
from app.ai.neural_net import INVALID_MOVE_INDEX  # noqa: E402
from app.game_engine import GameEngine  # noqa: E402
from app.models import (  # noqa: E402
    GameState, BoardType, BoardState, GamePhase, GameStatus, TimeControl,
    Player, AIConfig
)
from app.training.env import RingRiftEnv  # noqa: E402


def create_initial_state(
    board_type: BoardType = BoardType.SQUARE8,
    num_players: int = 2,
) -> GameState:
    """Create an initial GameState for self-play.

    This mirrors the core TypeScript `createInitialGameState` logic for
    per-player ring counts and victory thresholds while defaulting to an
    ACTIVE game suitable for self-play training.

    Parameters
    ----------
    board_type:
        Board geometry to use (square8, square19, hexagonal).
    num_players:
        Number of active players in the game (2–4 supported).
    """
    # Clamp to a sensible range to avoid constructing degenerate states.
    if num_players < 2:
        num_players = 2
    if num_players > 4:
        num_players = 4

    # Board configuration aligned with src/shared/types/game.ts BOARD_CONFIGS.
    if board_type == BoardType.SQUARE8:
        size = 8
        rings_per_player = 18
        total_spaces = 64
    elif board_type == BoardType.SQUARE19:
        size = 19
        rings_per_player = 36
        total_spaces = 361
    elif board_type == BoardType.HEXAGONAL:
        # Hex boards use radius 11 and 36 rings per player in TS.
        size = 11
        rings_per_player = 36
        total_spaces = 331
    else:
        # Fallback to square8-style defaults if an unknown board is passed.
        size = 8
        rings_per_player = 18
        total_spaces = 64

    # Victory thresholds: strictly more than half of total rings / spaces.
    total_rings = rings_per_player * num_players
    victory_threshold = (total_rings // 2) + 1
    territory_threshold = (total_spaces // 2) + 1

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
        for idx in range(1, num_players + 1)
    ]

    return GameState(
        id="self-play",
        boardType=board_type,
        board=BoardState(
            type=board_type,
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
        timeControl=TimeControl(initialTime=600, increment=0, type="blitz"),
        # For training we start in an ACTIVE state so env loops run.
        gameStatus=GameStatus.ACTIVE,
        createdAt=datetime.now(),
        lastMoveAt=datetime.now(),
        isRated=False,
        maxPlayers=num_players,
        # Training use-cases historically treated this as total rings available
        # in the game rather than "placed on board". Preserve that behaviour
        # but generalise to N players.
        totalRingsInPlay=total_rings,
        totalRingsEliminated=0,
        victoryThreshold=victory_threshold,
        territoryVictoryThreshold=territory_threshold,
        chainCaptureState=None,
        mustMoveFromStackKey=None,
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


def augment_data(
    features,
    globals,
    policy_indices,
    policy_values,
    neural_net,
    board_type: BoardType,
):
    """
    Augment data by rotating and flipping.

    Returns a list of (features, globals, policy_indices, policy_values)
    tuples. For hexagonal boards we currently skip augmentation because
    simple rotations/flips of the rectangular embedding do not correspond
    cleanly to legal hex symmetries.
    """
    # Hex boards: no geometric augmentation for now.
    if board_type == BoardType.HEXAGONAL:
        return [(features, globals, policy_indices, policy_values)]

    augmented = []

    # Original sample
    augmented.append((features, globals, policy_indices, policy_values))

    # Helper to transform a sparse policy
    def transform_policy(indices, values, k_rot, flip_h):
        if len(indices) == 0:
            return indices, values

        new_indices = []
        new_values = []
        board_size = neural_net.board_size

        # Create a dummy game state for decoding/encoding context.
        from app.models import (
            GameState,
            BoardState,
            TimeControl,
            GameStatus,
            GamePhase,
            Position,
        )
        dummy_state = GameState(
            id="dummy",
            boardType=board_type,
            board=BoardState(type=board_type, size=board_size),
            players=[],
            currentPhase=GamePhase.MOVEMENT,
            currentPlayer=1,
            moveHistory=[],
            timeControl=TimeControl(
                initialTime=0,
                increment=0,
                type="blitz",
            ),
            gameStatus=GameStatus.ACTIVE,
            createdAt=datetime.now(),
            lastMoveAt=datetime.now(),
            isRated=False,
            maxPlayers=2,
            totalRingsInPlay=0,
            totalRingsEliminated=0,
            victoryThreshold=0,
            territoryVictoryThreshold=0,
            chainCaptureState=None,
            mustMoveFromStackKey=None,
        )

        for idx, prob in zip(indices, values):
            move = neural_net.decode_move(idx, dummy_state)
            if not move:
                continue

            # Transform move coordinates in the CNN's 2D embedding.
            def rotate_point(x, y, n, k):
                """Rotate (x, y) in an n×n grid k times by 90°."""
                for _ in range(k):
                    x, y = y, n - 1 - x
                return x, y

            def flip_point(x, y, n):
                """Horizontal flip of (x, y) in an n×n grid."""
                return n - 1 - x, y

            # Transform 'to'
            tx, ty = move.to.x, move.to.y
            tx, ty = rotate_point(tx, ty, board_size, k_rot)
            if flip_h:
                tx, ty = flip_point(tx, ty, board_size)
            new_to = Position(x=tx, y=ty)

            new_from = None
            new_capture_target = None

            # Transform 'from' if exists
            if move.from_pos:
                fx, fy = move.from_pos.x, move.from_pos.y
                fx, fy = rotate_point(fx, fy, board_size, k_rot)
                if flip_h:
                    fx, fy = flip_point(fx, fy, board_size)
                new_from = Position(x=fx, y=fy)

            # Transform 'capture_target' if exists
            if move.capture_target:
                cx, cy = move.capture_target.x, move.capture_target.y
                cx, cy = rotate_point(cx, cy, board_size, k_rot)
                if flip_h:
                    cx, cy = flip_point(cx, cy, board_size)
                new_capture_target = Position(x=cx, y=cy)

            # Create new Move object with transformed coordinates
            move = move.model_copy(
                update={
                    "to": new_to,
                    "from_pos": new_from,
                    "capture_target": new_capture_target,
                }
            )

            # Re-encode using canonical coordinates derived from the dummy
            # board geometry. Moves that fall outside the fixed 19×19 policy
            # grid return INVALID_MOVE_INDEX and are skipped.
            new_idx = neural_net.encode_move(move, dummy_state.board)
            if new_idx != INVALID_MOVE_INDEX:
                new_indices.append(new_idx)
                new_values.append(prob)

        return (
            np.array(new_indices, dtype=np.int32),
            np.array(new_values, dtype=np.float32),
        )

    for k in range(1, 4):
        # Rotate features (C, H, W)
        rotated_features = np.rot90(features, k=k, axes=(1, 2))
        r_indices, r_values = transform_policy(
            policy_indices,
            policy_values,
            k,
            False,
        )
        augmented.append((rotated_features, globals, r_indices, r_values))

    # Flip (horizontal)
    flipped_features = np.flip(features, axis=2)
    f_indices, f_values = transform_policy(
        policy_indices,
        policy_values,
        0,
        True,
    )
    augmented.append((flipped_features, globals, f_indices, f_values))

    # Flip + rotations
    for k in range(1, 4):
        r_feat = np.rot90(features, k=k, axes=(1, 2))
        rf_features = np.flip(r_feat, axis=2)
        rf_indices, rf_values = transform_policy(
            policy_indices,
            policy_values,
            k,
            True,
        )
        augmented.append((rf_features, globals, rf_indices, rf_values))

    return augmented


def generate_dataset(
    num_games=10, output_file="data/dataset.npz",
    ai1=None, ai2=None, board_type=BoardType.SQUARE8,
    seed=None
):
    """
    Generate self-play data using DescentAI and RingRiftEnv.
    Logs (state, best_move, root_value) for training.
    """
    if seed is not None:
        import random
        import torch
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    # Accumulate data in separate lists
    new_features = []
    new_globals = []
    new_values = []
    new_policy_indices = []
    new_policy_values = []

    # Initialize AI if not provided
    if ai1 is None:
        ai1 = DescentAI(
            player_number=1,
            config=AIConfig(difficulty=10, randomness=0.1, think_time=500)
        )
    if ai2 is None:
        ai2 = DescentAI(
            player_number=2,
            config=AIConfig(difficulty=10, randomness=0.1, think_time=500)
        )

    ai_p1 = ai1
    ai_p2 = ai2

    print(f"Generating {num_games} games on {board_type}...")

    env = RingRiftEnv(
        board_type=board_type, max_moves=200, reward_on="terminal"
    )

    for game_idx in range(num_games):
        # Set seed for each game if provided, incrementing to ensure variety
        game_seed = seed + game_idx if seed is not None else None
        state = env.reset(seed=game_seed)
        game_history = []

        # History buffer for this game
        # List of feature planes (10, 8, 8)
        state_history = []
        history_length = 3

        print(f"Game {game_idx+1} started")
        move_count = 0

        while (
            state.game_status == GameStatus.ACTIVE
            and move_count < env.max_moves
        ):
            current_player = state.current_player
            ai = ai_p1 if current_player == 1 else ai_p2

            # Use DescentAI to select move
            # DescentAI doesn't return a policy distribution, but we can
            # construct a one-hot target for the best move.
            # Or better, we can use the root value from the search as a target.
            move = ai.select_move(state)

            if not move:
                # No moves available, current player loses
                state.winner = 2 if current_player == 1 else 1
                state.game_status = GameStatus.FINISHED
                break

            # Encode state and action
            if ai.neural_net:
                # Collect Tree Learning data (search log)
                search_data = ai.get_search_data()
                
                # Process search data (features, value)
                # We don't have policy for these intermediate nodes, so we use zero policy
                # Or we can skip policy training for these samples
                for feat, val in search_data:
                    # We need to reconstruct the full input (stacked history)
                    # This is tricky because search nodes might be deep in the tree
                    # and we don't have their history easily available.
                    # However, for "Simple AlphaZero", they just use the current state features
                    # without history for the auxiliary value targets, OR they assume history is negligible.
                    # Given our architecture requires history, we might need to approximate.
                    # For now, let's use the current game history as the context,
                    # even though it's slightly incorrect for deep nodes (history should shift).
                    # But since Descent doesn't simulate opponent moves in history during search (it's just state),
                    # the history remains "what happened before search started".
                    
                    # Construct stacked features using current game history
                    hist_list = state_history[::-1]
                    while len(hist_list) < history_length:
                        hist_list.append(np.zeros_like(feat))
                    hist_list = hist_list[:history_length]
                    stack_list = [feat] + hist_list
                    stacked_features = np.concatenate(stack_list, axis=0)
                    
                    # Globals need to be re-calculated for the search node state?
                    # The search log only stored features.
                    # We should probably store globals too in DescentAI.
                    # For now, let's use the root globals as approximation or skip globals update.
                    # Actually, let's just use the root globals.
                    _, root_globals = ai.neural_net._extract_features(state)
                    
                    # Policy is unknown/irrelevant for these value-only samples
                    # We can use a zero vector or a uniform vector.
                    # Policy is unknown/irrelevant for these value-only samples
                    # Use empty sparse arrays
                    p_indices = np.array([], dtype=np.int32)
                    p_values = np.array([], dtype=np.float32)
                    
                    # Augment and add immediately
                    augmented_samples = augment_data(
                        stacked_features,
                        root_globals,
                        p_indices,
                        p_values,
                        ai.neural_net,
                        state.board.type,
                    )
                    
                    for f, g, pi, pv in augmented_samples:
                        new_features.append(f)
                        new_globals.append(g)
                        new_values.append(val) # Use the search value directly
                        new_policy_indices.append(pi)
                        new_policy_values.append(pv)

                # Now handle the actual root move (standard AlphaZero training)
                features, globals_vec = ai.neural_net._extract_features(state)

                # Construct stacked features
                hist_list = state_history[::-1]
                while len(hist_list) < history_length:
                    hist_list.append(np.zeros_like(features))
                hist_list = hist_list[:history_length]
                stack_list = [features] + hist_list
                stacked_features = np.concatenate(stack_list, axis=0)

                # Update history
                state_history.append(features)
                if len(state_history) > history_length + 1:
                    state_history.pop(0)

                # Encode policy using ordinal distribution from search values
                # We need to access the search tree to get children values
                # DescentAI stores this in transposition_table
                state_key = ai._get_state_key(state)
                p_indices = []
                p_values = []
                
                if state_key in ai.transposition_table:
                    entry = ai.transposition_table[state_key]
                    if len(entry) == 3:
                        _, children_values, _ = entry
                    else:
                        _, children_values = entry
                        
                    if children_values:
                        # Extract values and compute soft targets
                        # children_values: {move_key: (move, val, prob)}
                        moves_data = []
                        for m_key, data in children_values.items():
                            m = data[0]
                            v = data[1]
                            moves_data.append((m, v))
                            
                        # Sort by value (descending for current player)
                        # Since we want probability distribution, we want higher prob for better moves
                        # If current_player == ai.player_number, higher value is better
                        # If current_player != ai.player_number, lower value is better (but we are training from perspective of current player?)
                        # Actually, DescentAI always stores values from perspective of player 1 (or consistent perspective)?
                        # No, DescentAI evaluates:
                        # if state.current_player == self.player_number: best_val = max(...)
                        # else: best_val = min(...)
                        # So values are relative to self.player_number.
                        
                        # We need to normalize values to be "goodness for current player"
                        is_maximizing = (current_player == ai.player_number)
                        
                        # Sort moves by "goodness"
                        if is_maximizing:
                            moves_data.sort(key=lambda x: x[1], reverse=True)
                        else:
                            moves_data.sort(key=lambda x: x[1])
                            
                        # Rank-based distribution (Cohen-Solal)
                        # p(rank) ~ 1 / (rank + k)
                        # rank 0 is best
                        k_rank = 1.0
                        probs = []
                        for rank in range(len(moves_data)):
                            probs.append(1.0 / (rank + k_rank))
                            
                        # Normalize
                        probs = np.array(probs)
                        probs = probs / probs.sum()
                        
                        # Encode
                        # Encode
                        for i, (m, _) in enumerate(moves_data):
                            idx = ai.neural_net.encode_move(m, state.board)
                            if idx != INVALID_MOVE_INDEX:
                                p_indices.append(idx)
                                p_values.append(probs[i])

                # Fallback if no search data (shouldn't happen if search ran)
                if not p_indices:
                    idx = ai.neural_net.encode_move(move, state.board)
                    if idx != INVALID_MOVE_INDEX:
                        p_indices.append(idx)
                        p_values.append(1.0)
                game_history.append({
                    'features': stacked_features,
                    'globals': globals_vec,
                    'policy_indices': np.array(p_indices, dtype=np.int32),
                    'policy_values': np.array(p_values, dtype=np.float32),
                    'player': current_player
                })

            state, _, done, _ = env.step(move)
            move_count += 1

            if move_count % 10 == 0:
                print(f"  Move {move_count}")

            if done:
                break

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
            
            # Augment data for training; board_type is fixed per dataset.
            augmented_samples = augment_data(
                step["features"],
                step["globals"],
                step["policy_indices"],
                step["policy_values"],
                ai_p1.neural_net,
                board_type,
            )
            
            for feat, glob, pi, pv in augmented_samples:
                new_features.append(feat)
                new_globals.append(glob)
                new_values.append(outcome)
                new_policy_indices.append(pi)
                new_policy_values.append(pv)
            
    # Save data with Experience Replay (Append mode)
    # Use provided output_file, ensuring directory exists
    if not os.path.isabs(output_file):
        output_path = os.path.join(os.path.dirname(__file__), output_file)
    else:
        output_path = output_file
        
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Convert new data to numpy arrays (object array for sparse policies)
    new_features = np.array(new_features, dtype=np.float32)
    new_globals = np.array(new_globals, dtype=np.float32)
    new_values = np.array(new_values, dtype=np.float32)
    # Sparse policies stored as object arrays of numpy arrays
    new_policy_indices = np.array(new_policy_indices, dtype=object)
    new_policy_values = np.array(new_policy_values, dtype=object)
    
    print(f"Generated {len(new_values)} new samples")

    # Load existing data if available
    if os.path.exists(output_path):
        try:
            with np.load(output_path, allow_pickle=True) as data:
                # Check if keys exist (handling potential format changes)
                if 'features' in data:
                    existing_features = data['features']
                    existing_globals = data['globals']
                    existing_values = data['values']
                    
                    # Handle migration from dense to sparse
                    if 'policy_indices' in data:
                        existing_policy_indices = data['policy_indices']
                        existing_policy_values = data['policy_values']
                    else:
                        print("Migrating dense policies to sparse...")
                        # Convert dense to sparse
                        dense_policies = data['policies']
                        existing_policy_indices = []
                        existing_policy_values = []
                        for p in dense_policies:
                            indices = np.nonzero(p)[0]
                            values = p[indices]
                            existing_policy_indices.append(indices.astype(np.int32))
                            existing_policy_values.append(values.astype(np.float32))
                        existing_policy_indices = np.array(existing_policy_indices, dtype=object)
                        existing_policy_values = np.array(existing_policy_values, dtype=object)
                    
                    print(f"Loaded {len(existing_values)} existing samples")
                    
                    # Concatenate
                    new_features = np.concatenate(
                        [existing_features, new_features]
                    )
                    new_globals = np.concatenate(
                        [existing_globals, new_globals]
                    )
                    new_values = np.concatenate(
                        [existing_values, new_values]
                    )
                    new_policy_indices = np.concatenate(
                        [existing_policy_indices, new_policy_indices]
                    )
                    new_policy_values = np.concatenate(
                        [existing_policy_values, new_policy_values]
                    )
        except Exception as e:
            print(f"Could not load existing data (starting fresh): {e}")
            
    # Limit buffer size (Experience Replay Buffer)
    MAX_BUFFER_SIZE = 50000
    if len(new_values) > MAX_BUFFER_SIZE:
        print(
            f"Buffer full ({len(new_values)}), "
            f"keeping last {MAX_BUFFER_SIZE} samples"
        )
        new_features = new_features[-MAX_BUFFER_SIZE:]
        new_globals = new_globals[-MAX_BUFFER_SIZE:]
        new_values = new_values[-MAX_BUFFER_SIZE:]
        new_policy_indices = new_policy_indices[-MAX_BUFFER_SIZE:]
        new_policy_values = new_policy_values[-MAX_BUFFER_SIZE:]
        
    # Save as compressed npz
    np.savez_compressed(
        output_path,
        features=new_features,
        globals=new_globals,
        values=new_values,
        policy_indices=new_policy_indices,
        policy_values=new_policy_values
    )


if __name__ == "__main__":
    generate_dataset(num_games=2)
