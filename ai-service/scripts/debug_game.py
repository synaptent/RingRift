#!/usr/bin/env python
"""Debug a single game to see what moves EBMO selects."""

import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.ai.ebmo_ai import EBMO_AI
from app.ai.ebmo_network import EBMOConfig
from app.ai.factory import AIFactory
from app.models.core import AIType, AIConfig, BoardType
from app.game_engine import GameEngine
from app.training.generate_data import create_initial_state


def debug_game(model_path: str):
    """Play a single game with verbose output."""
    print(f"\n{'='*60}")
    print(f"Debug game with: {model_path}")
    print(f"{'='*60}\n")

    state = create_initial_state(board_type=BoardType.SQUARE8, num_players=2)
    engine = GameEngine()

    # Create EBMO
    config = EBMOConfig(use_direct_eval=True)
    ebmo = EBMO_AI(
        player_number=1,
        config=AIConfig(difficulty=5),
        model_path=model_path,
        ebmo_config=config,
    )

    # Random opponent
    random_ai = AIFactory.create(AIType.RANDOM, 2, AIConfig(difficulty=1))

    for move_num in range(50):
        if state.winner is not None:
            print(f"\n*** Game over! Winner: Player {state.winner} ***")
            break

        current = state.current_player
        ai = ebmo if current == 1 else random_ai
        ai_name = "EBMO" if current == 1 else "Random"

        # Get valid moves
        valid_moves = ai.get_valid_moves(state)
        if not valid_moves:
            break

        # Show energy distribution for EBMO by move type
        if current == 1:
            energies = ebmo.get_move_energies(state, valid_moves)
            
            # Group by move type
            by_type = {}
            for move, energy_val in zip(valid_moves, energies.values()):
                mtype = move.type.value
                if mtype not in by_type:
                    by_type[mtype] = []
                by_type[mtype].append(energy_val)
            
            print(f"\nMove {move_num+1} (Player {current} - {ai_name})")
            print(f"  Valid moves by type (min energy):")
            for mtype, vals in sorted(by_type.items(), key=lambda x: min(x[1])):
                print(f"    {mtype}: {min(vals):.4f} (count={len(vals)})")

        move = ai.select_move(state)
        if move is None:
            break

        # Show selected move
        move_desc = f"{move.type.value}"
        if move.from_pos:
            move_desc += f" from ({move.from_pos.x},{move.from_pos.y})"
        if move.to:
            move_desc += f" to ({move.to.x},{move.to.y})"

        if current == 1:
            print(f"  Selected: {move_desc}")
        else:
            print(f"Move {move_num+1} (P{current} {ai_name}): {move_desc}")

        state = engine.apply_move(state, move)

    print(f"\nGame ended. Winner: {state.winner}")


if __name__ == "__main__":
    debug_game("models/ebmo/ebmo_square8_epoch69.pt")
