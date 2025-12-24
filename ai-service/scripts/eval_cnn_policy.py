#!/usr/bin/env python3
"""Evaluate CNN policy network against Random baseline."""

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.training.train_cnn_policy import CNNPolicyNet
from app.game_engine import GameEngine
from app.models import BoardType, GameStatus, Position
from app.training.train_gmo_selfplay import create_initial_state
from app.ai._neural_net_legacy import encode_move_for_board
from app.rules.geometry import BoardGeometry


def _pos_from_key(key: str) -> Position:
    """Parse position from string key like '3,4'."""
    parts = key.split(",")
    return Position(x=int(parts[0]), y=int(parts[1]))


def extract_features(state) -> tuple[np.ndarray, np.ndarray]:
    """Extract features from game state for square8 boards."""
    board = state.board
    board_size = 8
    features = np.zeros((14, board_size, board_size), dtype=np.float32)
    current_player = state.current_player

    # Channel 0/1: Stacks (height normalized)
    for pos_key, stack in board.stacks.items():
        try:
            pos = _pos_from_key(pos_key)
        except (ValueError, IndexError):
            continue
        cx, cy = pos.x, pos.y
        if not (0 <= cx < board_size and 0 <= cy < board_size):
            continue
        val = min(stack.stack_height / 5.0, 1.0)
        if stack.controlling_player == current_player:
            features[0, cx, cy] = val
        else:
            features[1, cx, cy] = val

    # Channel 2/3: Markers
    for pos_key, marker in board.markers.items():
        try:
            pos = _pos_from_key(pos_key)
        except (ValueError, IndexError):
            continue
        cx, cy = pos.x, pos.y
        if not (0 <= cx < board_size and 0 <= cy < board_size):
            continue
        if marker.player == current_player:
            features[2, cx, cy] = 1.0
        else:
            features[3, cx, cy] = 1.0

    # Channel 4/5: Collapsed spaces
    for pos_key, owner in board.collapsed_spaces.items():
        try:
            pos = _pos_from_key(pos_key)
        except (ValueError, IndexError):
            continue
        cx, cy = pos.x, pos.y
        if not (0 <= cx < board_size and 0 <= cy < board_size):
            continue
        if owner == current_player:
            features[4, cx, cy] = 1.0
        else:
            features[5, cx, cy] = 1.0

    # Channel 6/7: Liberties
    for pos_key, stack in board.stacks.items():
        try:
            pos = _pos_from_key(pos_key)
        except (ValueError, IndexError):
            continue
        cx, cy = pos.x, pos.y
        if not (0 <= cx < board_size and 0 <= cy < board_size):
            continue
        neighbors = BoardGeometry.get_adjacent_positions(pos, board.type, board.size)
        liberties = 0
        for npos in neighbors:
            if not BoardGeometry.is_within_bounds(npos, board.type, board.size):
                continue
            n_key = npos.to_key()
            if n_key in board.stacks or n_key in board.collapsed_spaces:
                continue
            liberties += 1
        val = min(liberties / 8.0, 1.0)
        if stack.controlling_player == current_player:
            features[6, cx, cy] = val
        else:
            features[7, cx, cy] = val

    # Channel 8/9: Line potential
    for pos_key, marker in board.markers.items():
        try:
            pos = _pos_from_key(pos_key)
        except (ValueError, IndexError):
            continue
        cx, cy = pos.x, pos.y
        if not (0 <= cx < board_size and 0 <= cy < board_size):
            continue
        neighbors = BoardGeometry.get_adjacent_positions(pos, board.type, board.size)
        friendly = 0
        for npos in neighbors:
            nm = board.markers.get(npos.to_key())
            if nm and nm.player == marker.player:
                friendly += 1
        val = min(friendly / 8.0, 1.0)
        if marker.player == current_player:
            features[8, cx, cy] = val
        else:
            features[9, cx, cy] = val

    # Channel 10/11: Cap presence
    for pos_key, stack in board.stacks.items():
        try:
            pos = _pos_from_key(pos_key)
        except (ValueError, IndexError):
            continue
        cx, cy = pos.x, pos.y
        if not (0 <= cx < board_size and 0 <= cy < board_size):
            continue
        if getattr(stack, 'has_cap', False):
            if stack.controlling_player == current_player:
                features[10, cx, cy] = 1.0
            else:
                features[11, cx, cy] = 1.0

    # Channel 12: Valid board position mask
    features[12, :, :] = 1.0

    # Global features (20 dims)
    # Phase (5), Rings info (4), Turn (1), Special phases (2), Reserved (8)
    globals_ = np.zeros(20, dtype=np.float32)

    # Phase one-hot (indices 0-4)
    from app.models import GamePhase
    phase_map = {
        GamePhase.RING_PLACEMENT: 0,
        GamePhase.MOVEMENT: 1,
        GamePhase.CAPTURE: 2,
        GamePhase.LINE_PROCESSING: 3,
        GamePhase.TERRITORY_PROCESSING: 4,
    }
    phase_idx = phase_map.get(state.current_phase, 1)  # Default to movement
    globals_[phase_idx] = 1.0

    # Rings info
    ring_norm = 15.0  # Standard rings per player for 2p
    my_player = next((p for p in state.players if p.player_number == current_player), None)
    opp_player = next((p for p in state.players if p.player_number != current_player), None)

    if my_player:
        globals_[5] = my_player.rings_in_hand / ring_norm
        globals_[7] = my_player.eliminated_rings / ring_norm
    if opp_player:
        globals_[6] = opp_player.rings_in_hand / ring_norm
        globals_[8] = opp_player.eliminated_rings / ring_norm

    # Is my turn (always 1.0 from current player's perspective)
    globals_[9] = 1.0

    # Special phase flags
    globals_[10] = 1.0 if state.current_phase == GamePhase.CHAIN_CAPTURE else 0.0
    globals_[11] = 1.0 if state.current_phase == GamePhase.FORCED_ELIMINATION else 0.0

    return features, globals_


def load_model(checkpoint_path: str, device: str = "cpu"):
    """Load CNN policy model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model = CNNPolicyNet(
        input_channels=56,  # Default
        global_features=20,
        hidden_channels=checkpoint.get("hidden_channels", 128),
        num_blocks=checkpoint.get("num_blocks", 6),
        board_size=checkpoint.get("board_size", 8),
        action_space_size=checkpoint.get("action_space_size", 8192),
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, checkpoint.get("action_space_size", 8192)


def get_move_by_policy(model, state, action_space_size: int, device: str, debug: bool = False, invert: bool = False):
    """Select move using policy network."""
    legal_moves = GameEngine.get_valid_moves(state, state.current_player)
    if not legal_moves:
        return None

    # Encode state (14 channels)
    features, globals_ = extract_features(state)

    # Stack with 3 history frames of zeros to get 56 channels
    history_frames = [np.zeros_like(features) for _ in range(3)]
    stacked = np.concatenate([features, *history_frames], axis=0)

    features_t = torch.from_numpy(stacked).float().unsqueeze(0).to(device)
    globals_t = torch.from_numpy(globals_).float().unsqueeze(0).to(device)

    with torch.no_grad():
        policy_logits, _ = model(features_t, globals_t)

    # Convert logits to probabilities
    probs = F.softmax(policy_logits, dim=-1).squeeze(0).cpu().numpy()

    # Score legal moves
    best_move = None
    best_prob = -1
    worst_move = None
    worst_prob = float('inf')
    move_probs = []

    for move in legal_moves:
        # Get action index for this move
        action_idx = encode_move_for_board(move, state.board)
        if action_idx >= 0 and action_idx < action_space_size:
            prob = probs[action_idx]
            move_probs.append((move, action_idx, prob))
            if prob > best_prob:
                best_prob = prob
                best_move = move
            if prob < worst_prob:
                worst_prob = prob
                worst_move = move

    if debug and move_probs:
        # Sort by probability and show top 3
        move_probs.sort(key=lambda x: x[2], reverse=True)
        print(f"  [Debug] Top 3 moves (of {len(move_probs)} legal):")
        for m, idx, p in move_probs[:3]:
            print(f"    idx={idx}, prob={p:.4f}, type={m.type}")

    # Select best or worst move based on invert flag
    selected_move = worst_move if invert else best_move

    # Fallback to random if no valid move found
    if selected_move is None:
        selected_move = random.choice(legal_moves)

    return selected_move


def play_game(model, action_space_size: int, ai_player: int, game_id: str, device: str, debug: bool = False, invert: bool = False) -> tuple[int | None, int]:
    """Play one game and return (winner, move_count)."""
    state = create_initial_state(
        game_id=game_id,
        board_type=BoardType.SQUARE8,
        rng_seed=hash(game_id) % (2**31),
    )

    move_count = 0
    max_moves = 400

    while state.game_status == GameStatus.ACTIVE and move_count < max_moves:
        current = state.current_player
        legal_moves = GameEngine.get_valid_moves(state, current)

        if not legal_moves:
            # Check for bookkeeping
            req = GameEngine.get_phase_requirement(state, current)
            if req:
                move = GameEngine.synthesize_bookkeeping_move(req, state)
                if move:
                    state = GameEngine.apply_move(state, move)
                    move_count += 1
                    continue
            break

        if current == ai_player:
            # Debug only first 3 moves
            should_debug = debug and move_count < 3
            move = get_move_by_policy(model, state, action_space_size, device, debug=should_debug, invert=invert)
        else:
            # Random opponent
            move = random.choice(legal_moves)

        if move is None:
            break

        state = GameEngine.apply_move(state, move)
        move_count += 1

    return state.winner, move_count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="models/cnn_policy/cnn_policy_best.pt")
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--invert", action="store_true", help="Invert policy (pick lowest prob move)")
    args = parser.parse_args()

    print(f"Loading model from {args.checkpoint}...")
    model, action_space_size = load_model(args.checkpoint, args.device)
    print(f"Action space size: {action_space_size}")
    if args.invert:
        print("INVERTED MODE: Picking lowest probability moves")

    wins_as_p1 = 0
    wins_as_p2 = 0
    games_per_side = args.games // 2

    print(f"\nPlaying as P1 ({games_per_side} games)...")
    for i in range(games_per_side):
        winner, moves = play_game(model, action_space_size, 1, f"p1_game_{i}", args.device, invert=args.invert)
        if winner == 1:
            wins_as_p1 += 1
        result = "Win" if winner == 1 else ("Loss" if winner == 2 else "Draw")
        print(f"  Game {i+1}: {result} ({moves} moves)")

    print(f"\nPlaying as P2 ({games_per_side} games)...")
    for i in range(games_per_side):
        winner, moves = play_game(model, action_space_size, 2, f"p2_game_{i}", args.device, invert=args.invert)
        if winner == 2:
            wins_as_p2 += 1
        result = "Win" if winner == 2 else ("Loss" if winner == 1 else "Draw")
        print(f"  Game {i+1}: {result} ({moves} moves)")

    total_wins = wins_as_p1 + wins_as_p2
    total_games = args.games

    print(f"\n{'='*60}")
    print(f"RESULTS: CNN Policy vs Random")
    print(f"{'='*60}")
    print(f"Wins as P1: {wins_as_p1}/{games_per_side} ({100*wins_as_p1/games_per_side:.1f}%)")
    print(f"Wins as P2: {wins_as_p2}/{games_per_side} ({100*wins_as_p2/games_per_side:.1f}%)")
    print(f"Total: {total_wins}/{total_games} ({100*total_wins/total_games:.1f}%)")


if __name__ == "__main__":
    main()
