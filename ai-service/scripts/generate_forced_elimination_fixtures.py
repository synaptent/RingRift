#!/usr/bin/env python3
"""Generate parity fixtures for forced elimination scenarios.

This script creates forced elimination parity fixtures by:
1. Converting contract vectors from tests/fixtures/contract-vectors/v2/forced_elimination.vectors.json
2. Creating additional synthetic FE scenarios

Usage (from ai-service/):
    python scripts/generate_forced_elimination_fixtures.py --max-fixtures 20
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add ai-service to path for imports
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.models import (
    BoardState,
    BoardType,
    GamePhase,
    GameState,
    GameStatus,
    Move,
    MoveType,
    Player,
    Position,
    RingStack,
    TimeControl,
)
from app.game_engine import GameEngine
from app.rules import global_actions as ga


def repo_root() -> Path:
    """Return the monorepo root."""
    return Path(__file__).resolve().parents[2]


def compute_simple_hash(state: GameState) -> str:
    """Compute a simple hash of the game state."""
    parts = [
        str(state.current_player),
        state.current_phase.value,
        state.game_status.value,
        str(len(state.board.stacks)),
        str(state.total_rings_eliminated),
    ]
    return hashlib.md5("".join(parts).encode()).hexdigest()[:16]


def _make_game_state(
    board_type: BoardType = BoardType.SQUARE8,
    num_players: int = 2,
    current_player: int = 1,
    current_phase: GamePhase = GamePhase.MOVEMENT,
) -> GameState:
    """Create a game state for FE testing."""
    board = BoardState(type=board_type, size=8 if board_type == BoardType.SQUARE8 else 11)

    rings_per_player = 18 if num_players == 2 else 12
    players = [
        Player(
            id=f"p{i}",
            username=f"player{i}",
            type="human",
            playerNumber=i,
            isReady=True,
            timeRemaining=60,
            aiDifficulty=None,
            ringsInHand=rings_per_player,
            eliminatedRings=0,
            territorySpaces=0,
        )
        for i in range(1, num_players + 1)
    ]

    now = datetime.now()

    return GameState(
        id="test-fe-fixture",
        boardType=board_type,
        board=board,
        players=players,
        currentPhase=current_phase,
        currentPlayer=current_player,
        timeControl=TimeControl(initialTime=60, increment=0, type="blitz"),
        gameStatus=GameStatus.ACTIVE,
        createdAt=now,
        lastMoveAt=now,
        isRated=False,
        maxPlayers=num_players,
        totalRingsInPlay=0,
        totalRingsEliminated=0,
        victoryThreshold=19,
        territoryVictoryThreshold=33,
        chainCaptureState=None,
        mustMoveFromStackKey=None,
        zobristHash=None,
    )


def _block_all_positions(state: GameState, except_positions: List[Position]) -> None:
    """Collapse all board positions except specified ones."""
    except_keys = {p.to_key() for p in except_positions}
    for x in range(state.board.size):
        for y in range(state.board.size):
            pos = Position(x=x, y=y)
            key = pos.to_key()
            if key not in except_keys and key not in state.board.stacks:
                state.board.collapsed_spaces[key] = 2  # Enemy territory


def generate_fixture_from_vector(vector: Dict[str, Any], fixture_id: str) -> Dict[str, Any]:
    """Convert a contract vector to a parity fixture."""
    input_data = vector["input"]
    move_data = input_data["move"]
    state_data = input_data["state"]

    # Build canonical move
    canonical_move = {
        "id": move_data.get("id", fixture_id),
        "type": move_data["type"],
        "player": move_data["player"],
        "from": None,
        "to": move_data.get("to"),
        "captureTarget": None,
        "capturedStacks": None,
        "captureChain": None,
        "overtakenRings": None,
        "placedOnStack": None,
        "placementCount": None,
        "stackMoved": None,
        "minimumDistance": None,
        "actualDistance": None,
        "markerLeft": None,
        "formedLines": None,
        "collapsedMarkers": None,
        "claimedTerritory": None,
        "disconnectedRegions": None,
        "eliminatedRings": None,
        "timestamp": datetime.now().isoformat(),
        "thinkTime": 0,
        "moveNumber": move_data.get("moveNumber", 1),
    }

    # Build python summary from input state
    python_summary = {
        "move_index": move_data.get("moveNumber", 1) - 1,
        "current_player": state_data["currentPlayer"],
        "current_phase": state_data["currentPhase"],
        "game_status": "active" if state_data["gameStatus"] == "active" else state_data["gameStatus"],
        "state_hash": hashlib.md5(json.dumps(state_data, sort_keys=True).encode()).hexdigest()[:16],
    }

    return {
        "canonical_move": canonical_move,
        "canonical_move_index": move_data.get("moveNumber", 1) - 1,
        "fixture_id": fixture_id,
        "source": "contract_vector",
        "vector_id": vector["id"],
        "move_type": move_data["type"],
        "python_summary": python_summary,
        "generated_at": datetime.now().isoformat(),
        "purpose": "forced_elimination_coverage",
        "description": vector.get("description", ""),
        "tags": vector.get("tags", []),
    }


def generate_synthetic_fixture(
    scenario_name: str,
    description: str,
    board_type: BoardType,
    num_players: int,
    stack_configs: List[Dict[str, Any]],
    player_rings_in_hand: Dict[int, int],
    player_eliminated: Dict[int, int] = None,
    current_phase: GamePhase = GamePhase.MOVEMENT,
) -> Dict[str, Any]:
    """Generate a synthetic FE fixture."""
    state = _make_game_state(
        board_type=board_type,
        num_players=num_players,
        current_player=1,
        current_phase=current_phase,
    )

    # Set player rings
    for player_num, rings in player_rings_in_hand.items():
        if player_num <= len(state.players):
            state.players[player_num - 1].rings_in_hand = rings

    # Set eliminated rings
    if player_eliminated:
        for player_num, elim in player_eliminated.items():
            if player_num <= len(state.players):
                state.players[player_num - 1].eliminated_rings = elim
                state.board.eliminated_rings[str(player_num)] = elim

    # Add stacks
    stack_positions = []
    for config in stack_configs:
        pos = Position(x=config["x"], y=config["y"])
        stack_positions.append(pos)
        rings = [config["player"]] * config["height"]
        state.board.stacks[pos.to_key()] = RingStack(
            position=pos,
            rings=rings,
            stackHeight=config["height"],
            capHeight=config["height"],
            controllingPlayer=config["player"],
        )

    # Block all other positions to force FE
    _block_all_positions(state, stack_positions)

    # Generate the FE move
    fe_move = Move(
        id=f"fe-{scenario_name}",
        type=MoveType.FORCED_ELIMINATION,
        player=1,
        to=stack_positions[0] if stack_positions else None,
        timestamp=datetime.now(),
        thinkTime=0,
        moveNumber=1,
    )

    canonical_move = {
        "id": fe_move.id,
        "type": "forced_elimination",
        "player": 1,
        "from": None,
        "to": {"x": stack_positions[0].x, "y": stack_positions[0].y, "z": None} if stack_positions else None,
        "captureTarget": None,
        "capturedStacks": None,
        "captureChain": None,
        "overtakenRings": None,
        "placedOnStack": None,
        "placementCount": None,
        "stackMoved": None,
        "minimumDistance": None,
        "actualDistance": None,
        "markerLeft": None,
        "formedLines": None,
        "collapsedMarkers": None,
        "claimedTerritory": None,
        "disconnectedRegions": None,
        "eliminatedRings": None,
        "timestamp": datetime.now().isoformat(),
        "thinkTime": 0,
        "moveNumber": 1,
    }

    python_summary = {
        "move_index": 0,
        "current_player": state.current_player,
        "current_phase": state.current_phase.value,
        "game_status": state.game_status.value,
        "state_hash": compute_simple_hash(state),
    }

    return {
        "canonical_move": canonical_move,
        "canonical_move_index": 0,
        "fixture_id": f"synthetic_fe_{scenario_name}",
        "source": "synthetic",
        "move_type": "forced_elimination",
        "python_summary": python_summary,
        "generated_at": datetime.now().isoformat(),
        "purpose": "forced_elimination_coverage",
        "description": description,
        "board_type": board_type.value,
        "num_players": num_players,
    }


def get_synthetic_scenarios() -> List[Dict[str, Any]]:
    """Get list of synthetic FE scenarios to generate."""
    scenarios = [
        # Single trapped stack scenarios
        {
            "name": "single_trapped_corner",
            "description": "Single stack trapped in corner (0,0)",
            "board_type": BoardType.SQUARE8,
            "num_players": 2,
            "stacks": [{"x": 0, "y": 0, "height": 2, "player": 1}],
            "rings_in_hand": {1: 0, 2: 18},
        },
        {
            "name": "single_trapped_center",
            "description": "Single stack trapped in center",
            "board_type": BoardType.SQUARE8,
            "num_players": 2,
            "stacks": [{"x": 3, "y": 3, "height": 3, "player": 1}],
            "rings_in_hand": {1: 0, 2: 18},
        },
        {
            "name": "single_tall_stack",
            "description": "Single tall stack (height 5) trapped",
            "board_type": BoardType.SQUARE8,
            "num_players": 2,
            "stacks": [{"x": 4, "y": 4, "height": 5, "player": 1}],
            "rings_in_hand": {1: 0, 2: 18},
        },
        # Multiple stack scenarios (smallest cap height selection)
        {
            "name": "multi_stack_cap_selection",
            "description": "Multiple stacks - should select smallest capHeight",
            "board_type": BoardType.SQUARE8,
            "num_players": 2,
            "stacks": [
                {"x": 0, "y": 0, "height": 4, "player": 1},
                {"x": 7, "y": 7, "height": 1, "player": 1},  # Smallest - selected
            ],
            "rings_in_hand": {1: 0, 2: 18},
        },
        {
            "name": "multi_stack_equal_height",
            "description": "Multiple stacks with equal capHeight",
            "board_type": BoardType.SQUARE8,
            "num_players": 2,
            "stacks": [
                {"x": 0, "y": 0, "height": 2, "player": 1},
                {"x": 7, "y": 7, "height": 2, "player": 1},
            ],
            "rings_in_hand": {1: 0, 2: 18},
        },
        {
            "name": "three_stacks_varied",
            "description": "Three stacks with varied heights",
            "board_type": BoardType.SQUARE8,
            "num_players": 2,
            "stacks": [
                {"x": 0, "y": 0, "height": 3, "player": 1},
                {"x": 4, "y": 4, "height": 1, "player": 1},  # Smallest
                {"x": 7, "y": 7, "height": 5, "player": 1},
            ],
            "rings_in_hand": {1: 0, 2: 18},
        },
        # Multi-player scenarios
        {
            "name": "three_player_fe",
            "description": "Forced elimination in 3-player game",
            "board_type": BoardType.SQUARE8,
            "num_players": 3,
            "stacks": [{"x": 3, "y": 3, "height": 2, "player": 1}],
            "rings_in_hand": {1: 0, 2: 12, 3: 12},
        },
        {
            "name": "four_player_fe",
            "description": "Forced elimination in 4-player game",
            "board_type": BoardType.SQUARE8,
            "num_players": 4,
            "stacks": [{"x": 3, "y": 3, "height": 2, "player": 1}],
            "rings_in_hand": {1: 0, 2: 9, 3: 9, 4: 9},
        },
        # Near-victory threshold scenarios
        {
            "name": "near_victory_threshold",
            "description": "FE that pushes player near victory threshold",
            "board_type": BoardType.SQUARE8,
            "num_players": 2,
            "stacks": [{"x": 3, "y": 3, "height": 3, "player": 1}],
            "rings_in_hand": {1: 0, 2: 18},
            "eliminated": {1: 16},  # 16 + 3 = 19 = victory
        },
        {
            "name": "at_victory_threshold",
            "description": "FE that exactly reaches victory threshold",
            "board_type": BoardType.SQUARE8,
            "num_players": 2,
            "stacks": [{"x": 3, "y": 3, "height": 2, "player": 1}],
            "rings_in_hand": {1: 0, 2: 18},
            "eliminated": {1: 17},  # 17 + 2 = 19 = victory
        },
        # Edge position scenarios
        {
            "name": "edge_top",
            "description": "Stack trapped on top edge",
            "board_type": BoardType.SQUARE8,
            "num_players": 2,
            "stacks": [{"x": 4, "y": 0, "height": 2, "player": 1}],
            "rings_in_hand": {1: 0, 2: 18},
        },
        {
            "name": "edge_right",
            "description": "Stack trapped on right edge",
            "board_type": BoardType.SQUARE8,
            "num_players": 2,
            "stacks": [{"x": 7, "y": 4, "height": 2, "player": 1}],
            "rings_in_hand": {1: 0, 2: 18},
        },
    ]
    return scenarios


def main():
    parser = argparse.ArgumentParser(description="Generate forced elimination parity fixtures")
    parser.add_argument("--max-fixtures", type=int, default=20, help="Maximum fixtures to generate")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for fixtures")
    args = parser.parse_args()

    root = repo_root()
    output_dir = Path(args.output_dir) if args.output_dir else root / "ai-service" / "parity_fixtures"
    output_dir.mkdir(parents=True, exist_ok=True)

    generated = 0

    # 1. Convert contract vectors
    vectors_path = root / "tests" / "fixtures" / "contract-vectors" / "v2" / "forced_elimination.vectors.json"
    if vectors_path.exists():
        print(f"Loading contract vectors from {vectors_path}...")
        with open(vectors_path) as f:
            vectors_data = json.load(f)

        vectors = vectors_data.get("vectors", [])
        print(f"Found {len(vectors)} contract vectors")

        for vector in vectors:
            if generated >= args.max_fixtures:
                break

            vector_id = vector["id"]
            fixture_id = f"contract_fe_{vector_id.replace('.', '_')}"
            filename = f"{fixture_id}.json"
            output_path = output_dir / filename

            if output_path.exists():
                print(f"  Skipping existing: {filename}")
                continue

            fixture = generate_fixture_from_vector(vector, fixture_id)

            with open(output_path, "w") as f:
                json.dump(fixture, f, indent=2, default=str)

            generated += 1
            print(f"  [{generated}/{args.max_fixtures}] Generated: {filename}")
    else:
        print(f"Contract vectors not found at {vectors_path}")

    # 2. Generate synthetic scenarios
    print("\nGenerating synthetic FE scenarios...")
    scenarios = get_synthetic_scenarios()

    for scenario in scenarios:
        if generated >= args.max_fixtures:
            break

        filename = f"synthetic_fe_{scenario['name']}.json"
        output_path = output_dir / filename

        if output_path.exists():
            print(f"  Skipping existing: {filename}")
            continue

        fixture = generate_synthetic_fixture(
            scenario_name=scenario["name"],
            description=scenario["description"],
            board_type=scenario["board_type"],
            num_players=scenario["num_players"],
            stack_configs=scenario["stacks"],
            player_rings_in_hand=scenario["rings_in_hand"],
            player_eliminated=scenario.get("eliminated"),
        )

        with open(output_path, "w") as f:
            json.dump(fixture, f, indent=2, default=str)

        generated += 1
        print(f"  [{generated}/{args.max_fixtures}] Generated: {filename}")

    print(f"\n{'='*60}")
    print(f"Generated {generated} forced elimination fixtures")
    print(f"Output directory: {output_dir}")

    # Count total FE fixtures
    fe_count = 0
    for p in output_dir.glob("*.json"):
        try:
            with open(p) as f:
                data = json.load(f)
                if (
                    data.get("move_type") == "forced_elimination"
                    or data.get("canonical_move", {}).get("type") == "forced_elimination"
                    or "forced_elimination" in str(data.get("purpose", ""))
                ):
                    fe_count += 1
        except (json.JSONDecodeError, OSError):
            pass

    print(f"Total FE fixtures in directory: {fe_count}")


if __name__ == "__main__":
    main()
