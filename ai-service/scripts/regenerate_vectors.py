#!/usr/bin/env python3
"""
Script to regenerate stale contract vector fixtures.

This script:
1. Loads vectors from JSON files
2. Deserializes input state and move
3. Applies the move using GameEngine
4. Runs complete_turn_phases() to advance through automatic phases
5. Captures the actual currentPlayer, currentPhase, and other assertions
6. Updates the vector's expectedOutput.assertions in the JSON file
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.game_engine import GameEngine
from app.rules.serialization import (
    compute_collapsed_count,
    compute_marker_count,
    compute_s_invariant,
    compute_stack_count,
)

# Path to v2 vectors
VECTORS_DIR = (
    Path(__file__).parent.parent.parent
    / "tests"
    / "fixtures"
    / "contract-vectors"
    / "v2"
)

# Import functions we need - replicate them here to avoid import issues
def complete_turn_phases(state):
    """Complete automatic turn phases after applying a move."""
    from app.game_engine import GameEngine, PhaseRequirementType

    MAX_ITERATIONS = 20
    iterations = 0

    while iterations < MAX_ITERATIONS:
        iterations += 1

        # Exit if game is not active
        status_value = (
            state.game_status.value
            if hasattr(state.game_status, "value")
            else str(state.game_status)
        )
        if status_value != "active":
            break

        # Check if there's a phase requirement for the current player
        requirement = GameEngine.get_phase_requirement(
            state, state.current_player
        )

        if requirement is None:
            # Interactive moves exist, stop auto-advancing
            break

        # Only auto-inject for line and territory no-action phases
        if requirement.type == PhaseRequirementType.NO_LINE_ACTION_REQUIRED:
            bookkeeping = GameEngine.synthesize_bookkeeping_move(
                requirement, state
            )
            state = GameEngine.apply_move(state, bookkeeping, trace_mode=True)
        elif requirement.type == PhaseRequirementType.NO_TERRITORY_ACTION_REQUIRED:
            bookkeeping = GameEngine.synthesize_bookkeeping_move(
                requirement, state
            )
            state = GameEngine.apply_move(state, bookkeeping, trace_mode=True)
        else:
            # Other requirements represent actual turn-end or player decisions
            break

    return state


def load_all_vectors():
    """Load all test vectors from all v2 bundles."""
    from app.rules.serialization import TestVectorBundle

    all_vectors = []
    if not VECTORS_DIR.exists():
        return all_vectors

    for bundle_path in sorted(VECTORS_DIR.glob("*.vectors.json")):
        with bundle_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        bundle = TestVectorBundle(data)
        all_vectors.extend(bundle.vectors)

    return all_vectors


def get_vector_by_id(vector_id: str):
    """Get a specific vector by ID."""
    vectors = load_all_vectors()
    for v in vectors:
        if v.id == vector_id:
            return v
    return None


# Vectors to regenerate
VECTORS_TO_FIX = [
    "capture.overtaking.adjacent",
    "movement.simple.one_space",
    "movement.diagonal",
    "movement.tall_stack.exact_distance",
    "line.movement.creates_marker",
    "line.diagonal.no_line",
    "territory.movement.no_disconnection",
    "territory.initial_state.empty",
    "chain_capture.depth2.segment2.square19",
]

# Map vector IDs to their bundle files
VECTOR_TO_BUNDLE = {
    "capture.overtaking.adjacent": "capture.vectors.json",
    "movement.simple.one_space": "movement.vectors.json",
    "movement.diagonal": "movement.vectors.json",
    "movement.tall_stack.exact_distance": "movement.vectors.json",
    "line.movement.creates_marker": "line_detection.vectors.json",
    "line.diagonal.no_line": "line_detection.vectors.json",
    "territory.movement.no_disconnection": "territory.vectors.json",
    "territory.initial_state.empty": "territory.vectors.json",
    "chain_capture.depth2.segment2.square19": "chain_capture.vectors.json",
}


def get_original_assertions(vector_id: str) -> dict[str, Any]:
    """Get the original assertions from the JSON file for reference.

    Args:
        vector_id: The ID of the vector

    Returns:
        The original assertions dictionary
    """
    bundle_file = VECTOR_TO_BUNDLE.get(vector_id)
    if not bundle_file:
        return {}

    bundle_path = VECTORS_DIR / bundle_file
    if not bundle_path.exists():
        return {}

    with open(bundle_path, 'r', encoding='utf-8') as f:
        bundle = json.load(f)

    for vector in bundle['vectors']:
        if vector['id'] == vector_id:
            return vector.get('expectedOutput', {}).get('assertions', {})

    return {}


def regenerate_vector(vector_id: str) -> dict[str, Any]:
    """Regenerate a single vector by executing it and capturing the output.

    Args:
        vector_id: The ID of the vector to regenerate

    Returns:
        Dictionary with the new assertions
    """
    print(f"\n{'='*60}")
    print(f"Regenerating: {vector_id}")
    print('='*60)

    # Load the vector
    vector = get_vector_by_id(vector_id)
    if vector is None:
        print(f"ERROR: Vector {vector_id} not found")
        return None

    # Get original assertions to see what fields we should preserve
    original_assertions = get_original_assertions(vector_id)

    # Compute initial S-invariant
    initial_s = compute_s_invariant(vector.input_state)
    print(f"Initial S-invariant: {initial_s}")

    # Apply the move
    try:
        result_state = GameEngine.apply_move(
            vector.input_state,
            vector.input_move,
            trace_mode=True,
        )
        print(f"After apply_move:")
        print(f"  currentPlayer: {result_state.current_player}")
        print(f"  currentPhase: {result_state.current_phase.value}")
        print(f"  gameStatus: {result_state.game_status.value}")
    except Exception as e:
        print(f"ERROR applying move: {e}")
        import traceback
        traceback.print_exc()
        return None

    # Complete automatic turn phases
    result_state = complete_turn_phases(result_state)
    print(f"\nAfter complete_turn_phases:")
    print(f"  currentPlayer: {result_state.current_player}")
    print(f"  currentPhase: {result_state.current_phase.value}")
    print(f"  gameStatus: {result_state.game_status.value}")

    # Compute final metrics
    final_s = compute_s_invariant(result_state)
    s_delta = final_s - initial_s
    stack_count = compute_stack_count(result_state)
    marker_count = compute_marker_count(result_state)
    collapsed_count = compute_collapsed_count(result_state)

    print(f"\nComputed metrics:")
    print(f"  stackCount: {stack_count}")
    print(f"  markerCount: {marker_count}")
    print(f"  collapsedCount: {collapsed_count}")
    print(f"  sInvariantDelta: {s_delta} (S: {initial_s} -> {final_s})")

    # Build new assertions
    new_assertions = {
        "currentPlayer": result_state.current_player,
        "currentPhase": result_state.current_phase.value,
        "gameStatus": result_state.game_status.value,
        "stackCount": stack_count,
        "markerCount": marker_count,
        "collapsedCount": collapsed_count,
        "sInvariantDelta": s_delta,
    }

    # Add player-specific assertions if they were in the original
    if "player1RingsInHand" in original_assertions:
        player1 = result_state.players[0]
        new_assertions["player1RingsInHand"] = player1.rings_in_hand
        print(f"  player1RingsInHand: {player1.rings_in_hand}")

    if "player1EliminatedRings" in original_assertions:
        player1 = result_state.players[0]
        new_assertions["player1EliminatedRings"] = player1.eliminated_rings
        print(f"  player1EliminatedRings: {player1.eliminated_rings}")

    if "player2RingsInHand" in original_assertions:
        player2 = result_state.players[1]
        new_assertions["player2RingsInHand"] = player2.rings_in_hand
        print(f"  player2RingsInHand: {player2.rings_in_hand}")

    return new_assertions


def update_vector_in_bundle(bundle_path: Path, vector_id: str, new_assertions: dict[str, Any]):
    """Update a vector's assertions in its bundle file.

    Args:
        bundle_path: Path to the bundle JSON file
        vector_id: The ID of the vector to update
        new_assertions: The new assertions dictionary
    """
    print(f"\nUpdating {bundle_path.name}...")

    # Load the bundle
    with open(bundle_path, 'r', encoding='utf-8') as f:
        bundle = json.load(f)

    # Find and update the vector
    updated = False
    for vector in bundle['vectors']:
        if vector['id'] == vector_id:
            vector['expectedOutput']['assertions'] = new_assertions
            updated = True
            break

    if not updated:
        print(f"ERROR: Vector {vector_id} not found in {bundle_path.name}")
        return False

    # Write the bundle back
    with open(bundle_path, 'w', encoding='utf-8') as f:
        json.dump(bundle, f, indent=2, ensure_ascii=False)
        f.write('\n')  # Add trailing newline

    print(f"✓ Updated {vector_id} in {bundle_path.name}")
    return True


def main():
    """Main script entry point."""
    print("Contract Vector Regeneration Script")
    print("=" * 60)

    success_count = 0
    failure_count = 0

    for vector_id in VECTORS_TO_FIX:
        # Clear cache before each vector
        GameEngine.clear_cache()

        # Regenerate the vector
        new_assertions = regenerate_vector(vector_id)
        if new_assertions is None:
            print(f"✗ FAILED: {vector_id}")
            failure_count += 1
            continue

        # Update the bundle file
        bundle_file = VECTOR_TO_BUNDLE.get(vector_id)
        if bundle_file is None:
            print(f"ERROR: No bundle file mapping for {vector_id}")
            failure_count += 1
            continue

        bundle_path = VECTORS_DIR / bundle_file
        if not bundle_path.exists():
            print(f"ERROR: Bundle file not found: {bundle_path}")
            failure_count += 1
            continue

        if update_vector_in_bundle(bundle_path, vector_id, new_assertions):
            success_count += 1
        else:
            failure_count += 1

    # Print summary
    print("\n" + "=" * 60)
    print(f"Summary: {success_count} succeeded, {failure_count} failed")
    print("=" * 60)

    if failure_count > 0:
        print("\nNext steps:")
        print("1. Review the failures above")
        print("2. Run the contract tests to verify:")
        print("   pytest tests/contracts/test_contract_vectors.py -v")
        sys.exit(1)
    else:
        print("\n✓ All vectors regenerated successfully!")
        print("\nNext steps:")
        print("1. Run the contract tests to verify:")
        print("   pytest tests/contracts/test_contract_vectors.py -v")
        print("2. Review and commit the changes to the JSON files")


if __name__ == "__main__":
    main()
