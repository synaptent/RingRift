#!/usr/bin/env python3
"""Generate soft target data using simple Gumbel MCTS (not batch).

This is simpler and faster than the batch MultiTreeMCTS for moderate sample sizes.
"""

import sys
sys.stdout.reconfigure(line_buffering=True)

import numpy as np
import time
import torch

print("Initializing...")

from app.ai.gumbel_mcts_ai import GumbelMCTSAI
from app.models import AIConfig, AIType, BoardType, GameStatus
from app.rules.default_engine import DefaultRulesEngine
from app.training.encoding import get_encoder_for_board_type
from app.training.initial_state import create_initial_state

# Setup
engine = DefaultRulesEngine()
encoder = get_encoder_for_board_type(BoardType.HEX8, version="v3")

config = AIConfig(
    ai_type=AIType.GUMBEL_MCTS,
    difficulty=5,
    gumbel_simulation_budget=16,  # Low for speed
    gumbel_num_sampled_actions=8,
)

ai = GumbelMCTSAI(
    player_number=0,  # Will be updated per move
    config=config,
    board_type=BoardType.HEX8,
)

print("Starting game generation...")
all_samples = []
start = time.time()

NUM_GAMES = 100
for game_num in range(NUM_GAMES):
    state = create_initial_state(board_type=BoardType.HEX8, num_players=2)
    game_samples = []
    move_count = 0

    while state.game_status == GameStatus.ACTIVE and move_count < 200:
        current_player = state.current_player

        # Update AI player number to match current player
        ai.player_number = current_player

        # Get move
        try:
            move = ai.select_move(state)
            if move is None:
                break
            # Get policy distribution from last search
            policy_moves, policy_probs = ai.get_visit_distribution()
        except Exception as e:
            print(f"  Error getting move: {e}")
            break

        if policy_moves and len(policy_moves) > 1:
            # Encode state
            features, globals_feat = encoder.encode_state(state)

            # Convert policy moves/probs to indices/values
            policy_indices = []
            policy_values = []

            for m, prob in zip(policy_moves, policy_probs):
                if prob > 0:
                    idx = encoder.encode_move(m, state.board)
                    policy_indices.append(idx)
                    policy_values.append(prob)

            if policy_indices:
                game_samples.append({
                    "features": features,
                    "globals": globals_feat,
                    "policy_indices": policy_indices,
                    "policy_values": policy_values,
                    "player": current_player,
                })

        try:
            state = engine.apply_move(state, move)
            move_count += 1
        except Exception as e:
            print(f"  Error applying move: {e}")
            break

    # Assign values based on winner
    winner = state.winner
    for s in game_samples:
        if winner is None:
            s["value"] = 0.0
        elif s["player"] == winner:
            s["value"] = 1.0
        else:
            s["value"] = -1.0
        all_samples.append(s)

    if (game_num + 1) % 10 == 0:
        elapsed = time.time() - start
        rate = len(all_samples) / elapsed if elapsed > 0 else 0
        print(f"Game {game_num+1}/{NUM_GAMES}: {len(all_samples)} samples ({rate:.1f}/s)")

total_time = time.time() - start
print(f"\nDone! {len(all_samples)} samples from {NUM_GAMES} games in {total_time:.1f}s")

# Save to NPZ
if all_samples:
    features = np.stack([s["features"] for s in all_samples])
    globals_arr = np.stack([s["globals"] for s in all_samples])
    values = np.array([s["value"] for s in all_samples])

    max_actions = max(len(s["policy_indices"]) for s in all_samples)
    policy_indices = np.zeros((len(all_samples), max_actions), dtype=np.int32)
    policy_values = np.zeros((len(all_samples), max_actions), dtype=np.float32)

    for i, s in enumerate(all_samples):
        n = len(s["policy_indices"])
        policy_indices[i, :n] = s["policy_indices"]
        policy_values[i, :n] = s["policy_values"]

    np.savez_compressed("data/training/soft_targets_hex8_2p.npz",
        features=features,
        globals=globals_arr,
        values=values,
        policy_indices=policy_indices,
        policy_values=policy_values,
        board_type="hex8",
        encoder_version="v3",
        source="gumbel_mcts_soft"
    )

    # Stats
    avg_actions = np.mean([np.sum(pv > 0) for pv in policy_values])
    entropies = []
    for pv in policy_values:
        pv_valid = pv[pv > 0]
        if len(pv_valid) > 0:
            pv_norm = pv_valid / pv_valid.sum()
            entropy = -np.sum(pv_norm * np.log(pv_norm + 1e-10))
            entropies.append(entropy)

    print(f"\nSaved: {features.shape}")
    print(f"Avg actions/sample: {avg_actions:.1f}")
    print(f"Avg policy entropy: {np.mean(entropies):.3f}")
else:
    print("ERROR: No samples collected!")
