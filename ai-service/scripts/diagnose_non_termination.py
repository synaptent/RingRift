#!/usr/bin/env python3
"""Diagnose non-termination issues in Gumbel MCTS selfplay."""

import os
import sys
import logging

# Suppress info logs for cleaner output
logging.basicConfig(level=logging.WARNING)

# Ensure app imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from collections import Counter
from app.models import BoardType, GameStatus, AIConfig, MoveType
from app.training.env import TrainingEnvConfig, make_env, get_theoretical_max_moves
from app.rules.core import compute_progress_snapshot
from app.rules.phase_machine import compute_had_any_action_this_turn, player_has_stacks_on_board

def count_rings_per_player(state):
    """Count rings for each player (on board + in hand)."""
    counts = {}
    for player in state.players:
        rings_on_board = sum(
            1 for stack in state.board.stacks.values()
            for ring_owner in stack.rings
            if ring_owner == player.player_number
        )
        counts[player.player_number] = rings_on_board + player.rings_in_hand
    return counts


def run_diagnostic_game(seed: int, max_script_moves: int = 250):
    """Run a game with detailed diagnostics."""
    from app.ai.gumbel_mcts_ai import GumbelMCTSAI

    print(f"\n{'='*60}")
    print(f"Diagnostic game with seed={seed}")
    print(f"{'='*60}")

    # Create env
    config = TrainingEnvConfig(board_type=BoardType.SQUARE8, num_players=2)
    env = make_env(config)
    state = env.reset(seed=seed)

    theoretical_max = get_theoretical_max_moves(BoardType.SQUARE8, 2)
    print(f"Theoretical max moves: {theoretical_max}")

    # Create AI with fresh weights for reproducibility
    ai_config = AIConfig(
        difficulty=9,
        randomness=0.0,
        use_neural_net=True,
        gumbel_simulation_budget=32,
        allow_fresh_weights=True,
    )

    # Use random fallback for CPU testing
    try:
        ai1 = GumbelMCTSAI(1, ai_config, BoardType.SQUARE8)
        ai2 = GumbelMCTSAI(2, ai_config, BoardType.SQUARE8)
        ais = {1: ai1, 2: ai2}
        print("Using GumbelMCTSAI")
    except Exception as e:
        print(f"Gumbel failed: {e}")
        print("Using random moves instead")
        ais = None

    # Tracking
    move_types = Counter()
    s_history = []
    ring_history = []
    forced_elim_count = 0

    script_move_count = 0
    env_move_count = 0

    prev_s = 0
    stall_moves = 0
    max_stall = 0

    while state.game_status == GameStatus.ACTIVE and script_move_count < max_script_moves:
        current_player = state.current_player

        # Get S-invariant before move
        snapshot = compute_progress_snapshot(state)
        s_before = snapshot['S']

        # Check forced elimination conditions
        # Note: This checks AFTER the previous move, so may not reflect current state exactly
        had_action = compute_had_any_action_this_turn(state)
        has_stacks = player_has_stacks_on_board(state, current_player)

        # Select move
        if ais:
            ai = ais[current_player]
            ai.player_number = current_player
            move = ai.select_move(state)
        else:
            import random
            moves = env.legal_moves()
            move = random.choice(moves) if moves else None

        if move is None:
            print(f"No move at script_move={script_move_count}")
            break

        # Track move type
        move_types[move.type.value] += 1
        if move.type == MoveType.FORCED_ELIMINATION:
            forced_elim_count += 1

        # Apply move
        state, _, done, info = env.step(move)
        script_move_count += 1
        env_move_count = info.get('move_count', env_move_count)

        # Get S-invariant after move
        snapshot = compute_progress_snapshot(state)
        s_after = snapshot['S']

        # Track S progress
        if s_after == prev_s:
            stall_moves += 1
            max_stall = max(max_stall, stall_moves)
        else:
            stall_moves = 0
        prev_s = s_after

        # Periodic logging
        if script_move_count % 25 == 0 or done:
            rings = count_rings_per_player(state)
            s_history.append(s_after)
            ring_history.append(rings.copy())
            print(f"  Move {script_move_count:3d} (env: {env_move_count:4d}): S={s_after:3d} "
                  f"(m={snapshot['markers']:2d}, c={snapshot['collapsed']:2d}, e={snapshot['eliminated']:2d}) "
                  f"rings={rings} phase={state.current_phase.value}")

        if done:
            break

    # Final summary
    print(f"\n{'='*60}")
    print(f"FINAL STATUS: {state.game_status.value}")
    print(f"Winner: {state.winner}")
    print(f"Script moves: {script_move_count}, Env moves: {env_move_count}")
    print(f"Max consecutive moves without S progress: {max_stall}")
    print(f"Forced eliminations: {forced_elim_count}")
    print()

    final_snapshot = compute_progress_snapshot(state)
    print(f"Final S-invariant: {final_snapshot['S']}")
    print(f"  Markers: {final_snapshot['markers']}")
    print(f"  Collapsed: {final_snapshot['collapsed']}")
    print(f"  Eliminated: {final_snapshot['eliminated']}")
    print()

    final_rings = count_rings_per_player(state)
    print(f"Final rings per player: {final_rings}")
    print()

    print("Move type distribution:")
    for mt, count in move_types.most_common():
        print(f"  {mt}: {count}")

    return state.game_status == GameStatus.COMPLETED


if __name__ == "__main__":
    # Test with different seeds
    import random

    completed = 0
    total = 5

    for i in range(total):
        seed = random.randint(0, 10000)
        if run_diagnostic_game(seed):
            completed += 1

    print(f"\n{'='*60}")
    print(f"SUMMARY: {completed}/{total} games completed")
    print(f"{'='*60}")
