#!/usr/bin/env python3
"""Diagnose EBMO model behavior to understand poor performance."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path

import numpy as np
import torch

from app.ai.ebmo_ai import EBMO_AI
from app.ai.ebmo_network import EBMOConfig, load_ebmo_model
from app.game.game import Game
from app.models import AIConfig


def diagnose_model(model_path: str):
    """Run diagnostic checks on EBMO model."""
    print(f"\n{'='*60}")
    print("EBMO Model Diagnostics")
    print(f"Model: {model_path}")
    print(f"{'='*60}\n")

    # Check if model exists
    if not Path(model_path).exists():
        print(f"ERROR: Model not found at {model_path}")
        return

    # Load model
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    config = EBMOConfig()
    network, info = load_ebmo_model(model_path, device, config)
    network.eval()

    print("Model loaded successfully")
    print(f"  Device: {device}")
    print(f"  Info: {info}")

    # Create AI instance
    ai_config = AIConfig(difficulty=5)
    ai = EBMO_AI(player_number=1, config=ai_config, model_path=model_path)

    # Create a fresh game
    game = Game()
    state = game.get_state()

    print("\n--- Game State ---")
    print(f"  Phase: {state.phase}")
    print(f"  Current player: {state.current_player}")

    # Get valid moves
    valid_moves = ai.get_valid_moves(state)
    print(f"  Valid moves: {len(valid_moves)}")

    if not valid_moves:
        print("  No valid moves!")
        return

    # Show first few moves
    print("\n  Sample moves:")
    for i, move in enumerate(valid_moves[:5]):
        print(f"    {i}: {move.type.value} to=({move.to.x}, {move.to.y})")

    # Get move energies
    print("\n--- Move Energies ---")
    energies = ai.get_move_energies(state)

    # Sort by energy
    sorted_energies = sorted(energies.items(), key=lambda x: x[1])

    print("  Top 5 lowest energy (best) moves:")
    for i, (move_key, energy) in enumerate(sorted_energies[:5]):
        print(f"    {i+1}. {move_key}: {energy:.4f}")

    print("\n  Top 5 highest energy (worst) moves:")
    for i, (move_key, energy) in enumerate(sorted_energies[-5:]):
        print(f"    {i+1}. {move_key}: {energy:.4f}")

    # Check energy distribution
    energy_values = list(energies.values())
    print("\n--- Energy Statistics ---")
    print(f"  Min energy: {min(energy_values):.4f}")
    print(f"  Max energy: {max(energy_values):.4f}")
    print(f"  Mean energy: {np.mean(energy_values):.4f}")
    print(f"  Std energy: {np.std(energy_values):.4f}")
    print(f"  Range: {max(energy_values) - min(energy_values):.4f}")

    # Check if energies are discriminative
    if np.std(energy_values) < 0.01:
        print("\n  WARNING: Energy values have very low variance!")
        print("  Model may not be discriminating between moves.")

    # Test what move the model selects
    print("\n--- Move Selection Test ---")
    selected_move = ai.select_move(state)
    print(f"  Selected move: {selected_move.type.value}")
    if selected_move.to:
        print(f"    to: ({selected_move.to.x}, {selected_move.to.y})")

    # Check if it's selecting skip moves
    skip_types = {'skip_placement', 'skip_capture', 'no_placement_action'}
    if selected_move.type.value in skip_types:
        print("\n  WARNING: Model is selecting a SKIP move!")
        print("  This could explain 0% win rate.")

    # Play a few moves and see what happens
    print("\n--- Simulated Game (first 10 moves) ---")
    test_game = Game()
    ai1 = EBMO_AI(player_number=1, config=ai_config, model_path=model_path)
    ai2 = EBMO_AI(player_number=2, config=ai_config, model_path=model_path)

    for i in range(10):
        state = test_game.get_state()
        current_ai = ai1 if state.current_player == 1 else ai2

        move = current_ai.select_move(state)
        if move is None:
            print(f"  Turn {i+1}: No move available")
            break

        print(f"  Turn {i+1} (P{state.current_player}): {move.type.value}", end="")
        if move.to:
            print(f" to ({move.to.x}, {move.to.y})")
        else:
            print()

        # Apply move
        result = test_game.apply_move(move)
        if not result.success:
            print(f"    ERROR: Move failed - {result.message}")
            break

    # Check model parameter statistics
    print("\n--- Model Parameter Statistics ---")
    total_params = 0
    zero_params = 0
    nan_params = 0

    for _name, param in network.named_parameters():
        total_params += param.numel()
        zero_params += (param == 0).sum().item()
        nan_params += torch.isnan(param).sum().item()

    print(f"  Total parameters: {total_params:,}")
    print(f"  Zero parameters: {zero_params:,} ({100*zero_params/total_params:.2f}%)")
    print(f"  NaN parameters: {nan_params:,}")

    if nan_params > 0:
        print("\n  CRITICAL: Model has NaN parameters!")

    if zero_params / total_params > 0.5:
        print("\n  WARNING: More than 50% of parameters are zero!")


def compare_models(model1_path: str, model2_path: str):
    """Compare energy distributions between two models."""
    print(f"\n{'='*60}")
    print("Model Comparison")
    print(f"{'='*60}\n")

    torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    ai_config = AIConfig(difficulty=5)

    ai1 = EBMO_AI(player_number=1, config=ai_config, model_path=model1_path)
    ai2 = EBMO_AI(player_number=1, config=ai_config, model_path=model2_path)

    # Create a game state
    game = Game()
    state = game.get_state()

    # Get energies from both models
    energies1 = ai1.get_move_energies(state)
    energies2 = ai2.get_move_energies(state)

    print(f"Model 1: {model1_path}")
    vals1 = list(energies1.values())
    print(f"  Energy range: [{min(vals1):.4f}, {max(vals1):.4f}]")
    print(f"  Energy std: {np.std(vals1):.4f}")

    print(f"\nModel 2: {model2_path}")
    vals2 = list(energies2.values())
    print(f"  Energy range: [{min(vals2):.4f}, {max(vals2):.4f}]")
    print(f"  Energy std: {np.std(vals2):.4f}")

    # Show selected moves
    move1 = ai1.select_move(state)
    move2 = ai2.select_move(state)

    print("\nSelected moves:")
    print(f"  Model 1: {move1.type.value}")
    print(f"  Model 2: {move2.type.value}")


if __name__ == "__main__":
    # Diagnose the improved model
    improved_model = "models/ebmo/ebmo_improved_best.pt"
    diagnose_model(improved_model)

    # Check if original self-play model exists for comparison
    original_model = "models/ebmo/ebmo_selfplay_best.pt"
    if Path(original_model).exists():
        print("\n" + "="*60)
        print("Comparing with original self-play model")
        print("="*60)
        diagnose_model(original_model)
        compare_models(improved_model, original_model)
