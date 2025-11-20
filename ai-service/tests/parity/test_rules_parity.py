import sys
import os
import json
import subprocess
import random
import pytest
from datetime import datetime

# Add app directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

from app.models import (  # noqa: E402
    GameState, BoardState, BoardType, GamePhase, Player, TimeControl,
    GameStatus, Move, RingStack, Position, LineInfo, Territory, MoveType
)
from app.game_engine import GameEngine  # noqa: E402


def create_initial_state() -> GameState:
    return GameState(
        id="test-game",
        boardType=BoardType.SQUARE8,
        board=BoardState(type=BoardType.SQUARE8, size=8),
        players=[
            Player(
                id="p1",
                username="Player 1",
                type="human",
                playerNumber=1,
                isReady=True,
                timeRemaining=600,
                ringsInHand=20,
                eliminatedRings=0,
                territorySpaces=0,
                aiDifficulty=None,
            ),
            Player(
                id="p2",
                username="Player 2",
                type="human",
                playerNumber=2,
                isReady=True,
                timeRemaining=600,
                ringsInHand=20,
                eliminatedRings=0,
                territorySpaces=0,
                aiDifficulty=None,
            ),
        ],
        currentPhase=GamePhase.RING_PLACEMENT,
        currentPlayer=1,
        moveHistory=[],
        timeControl=TimeControl(
            initialTime=600, increment=5, type="standard"
        ),
        gameStatus=GameStatus.ACTIVE,
        createdAt=datetime.now(),
        lastMoveAt=datetime.now(),
        isRated=False,
        maxPlayers=2,
        totalRingsInPlay=0,
        totalRingsEliminated=0,
        victoryThreshold=3,
        territoryVictoryThreshold=10,
        chainCaptureState=None,
    )


def validate_move_parity(state: GameState, move: Move, test_id: str) -> None:
    """Helper to validate a move against both TS engines.

    Canonical order per project:
    1. Rules documents (ringrift_complete_rules / compact rules)
    2. TS engine (RuleEngine + GameEngine + rules modules)
    3. Python AI engine mirroring TS and rules.

    This helper treats Python as the *driver* of state+move scenarios,
    but TS as the canonical validator. For primitive moves owned by the
    TS RuleEngine (placement / movement / capture), we require both the
    RuleEngine CLI and the ClientSandboxEngine CLI to accept the move.

    For post-move processing phases (line_formation, territory_claim),
    the TS architecture routes legality through lineProcessing and
    territoryProcessing rather than RuleEngine.validateMove. In those
    cases, we only assert parity against the sandbox CLI.
    """

    # Prepare input JSON
    input_data = {
        "gameState": state.model_dump(by_alias=True),
        "move": move.model_dump(by_alias=True),
    }

    # Convert datetime objects to strings
    input_data["gameState"]["createdAt"] = (
        input_data["gameState"]["createdAt"].isoformat()
    )
    input_data["gameState"]["lastMoveAt"] = (
        input_data["gameState"]["lastMoveAt"].isoformat()
    )
    input_data["move"]["timestamp"] = (
        input_data["move"]["timestamp"].isoformat()
    )

    input_file = f"parity_test_input_{test_id}.json"
    with open(input_file, "w") as f:
        json.dump(input_data, f)

    try:
        # Decide whether this move type is within the RuleEngine's remit.
        # Line formation and territory processing are handled by dedicated
        # TS modules (lineProcessing, territoryProcessing) invoked from
        # GameEngine/TurnEngine, not by RuleEngine.validateMove.
        # Forced elimination is also a high-level engine move.
        check_rule_engine = move.type not in (
            MoveType.LINE_FORMATION,
            MoveType.TERRITORY_CLAIM,
            MoveType.FORCED_ELIMINATION
        )

        if check_rule_engine:
            # Run TypeScript CLI (RuleEngine)
            ts_script = os.path.abspath(
                os.path.join(
                    os.path.dirname(__file__),
                    "../../../src/server/game/test-parity-cli.ts",
                )
            )
            cmd = [
                "npx", "ts-node", "--transpile-only", ts_script, input_file
            ]

            result = subprocess.run(
                cmd, capture_output=True, text=True, check=True
            )
            output = json.loads(result.stdout)

            assert output["status"] == "success", "TypeScript script failed"

            # Since we only send moves we consider valid under the canonical
            # rules, we expect TS to accept them as well.
            if not output["isValid"]:
                print(f"RuleEngine validation failed for move: {move}")
                print(f"State: {state}")

            assert output["isValid"], (
                f"Move {move} is valid in Python but invalid in TypeScript "
                "(RuleEngine)"
            )

            # Check state transition parity if available
            if "stateHash" in output:
                # Note: Python does not yet implement a hashGameState that is
                # guaranteed to match TS. For now we just assert that TS
                # returns some non-empty hash when validation succeeds.
                assert output["stateHash"], \
                    "RuleEngine did not return a state hash"

        # Run TypeScript CLI (ClientSandboxEngine) for *all* move types.
        sandbox_script = os.path.abspath(
            os.path.join(
                os.path.dirname(__file__),
                "../../../src/client/sandbox/test-sandbox-parity-cli.ts",
            )
        )
        cmd_sandbox = [
            "npx", "ts-node", "--transpile-only", sandbox_script, input_file
        ]

        result_sandbox = subprocess.run(
            cmd_sandbox, capture_output=True, text=True, check=True
        )

        # Parse output - handle potential TS errors in stdout
        try:
            output_sandbox = json.loads(result_sandbox.stdout)
        except json.JSONDecodeError:
            pytest.fail(
                f"Failed to parse Sandbox CLI output: {result_sandbox.stdout}"
            )

        assert output_sandbox["status"] == "success", \
            "TypeScript script failed (Sandbox)"

        if not output_sandbox["isValid"]:
            print(f"ClientSandboxEngine validation failed for move: {move}")

        assert output_sandbox["isValid"], (
            f"Move {move} is valid in Python but invalid in TypeScript "
            "(ClientSandboxEngine)"
        )

    except subprocess.CalledProcessError as e:
        pytest.fail(
            f"Error running TypeScript script: {e}\n"
            f"Stdout: {e.stdout}\nStderr: {e.stderr}"
        )
    finally:
        if os.path.exists(input_file):
            os.remove(input_file)


@pytest.mark.parametrize("iteration", range(10))
def test_random_move_parity(iteration: int) -> None:
    """Test parity for random moves from initial state.

    This still uses the Python GameEngine for move generation, but now the
    TS CLIs are robust to type-only diagnostics via --transpile-only.
    """

    # 1. Create initial state
    state = create_initial_state()

    # 2. Generate valid moves using Python engine
    moves = GameEngine.get_valid_moves(state, state.current_player)
    assert len(moves) > 0, "No moves generated!"

    # 3. Pick a random move
    move = random.choice(moves)
    print(f"Selected move: {move.type} to {move.to}")

    # 4. Validate move using TypeScript engines
    validate_move_parity(state, move, f"random_{iteration}")


def test_chain_capture_parity() -> None:
    """Test parity for a canonical overtaking capture segment.

    The Python GameEngine still uses a simplified capture model, but for
    TS parity we construct an overtaking_capture segment that matches the
    canonical TS semantics: from -> captureTarget -> landing beyond.
    """

    state = create_initial_state()
    state.current_phase = GamePhase.MOVEMENT

    # Place stacks to enable capture
    # Player 1 at (2,2) height 2
    # Player 2 at (2,3) height 1
    # Empty at (2,4)
    # Player 2 at (2,5) height 1
    # Empty at (2,6)

    p1_stack = RingStack(
        position=Position(x=2, y=2),
        rings=[1, 1],
        stackHeight=2,
        capHeight=2,
        controllingPlayer=1,
    )
    state.board.stacks["2,2"] = p1_stack

    p2_stack1 = RingStack(
        position=Position(x=2, y=3),
        rings=[2],
        stackHeight=1,
        capHeight=1,
        controllingPlayer=2,
    )
    state.board.stacks["2,3"] = p2_stack1

    p2_stack2 = RingStack(
        position=Position(x=2, y=5),
        rings=[2],
        stackHeight=1,
        capHeight=1,
        controllingPlayer=2,
    )
    state.board.stacks["2,5"] = p2_stack2

    # Sanity-check that Python thinks there are some moves from this state.
    python_moves = GameEngine.get_valid_moves(state, 1)
    assert python_moves, \
        "Python engine produced no moves for chain capture scenario"

    # Construct a canonical TS-style overtaking capture segment over (2,3)
    # landing on empty (2,4).
    canonical_capture_data = {
        "id": "simulated",
        "type": MoveType.OVERTAKING_CAPTURE,
        "player": 1,
        "from": Position(x=2, y=2),
        "to": Position(x=2, y=4),
        "captureTarget": Position(x=2, y=3),
        "placedOnStack": None,
        "placementCount": None,
        "timestamp": state.last_move_at,
        "thinkTime": 0,
        "moveNumber": 1,
    }
    canonical_capture = Move(**canonical_capture_data)

    validate_move_parity(state, canonical_capture, "chain_capture_segment")


def test_line_formation_parity() -> None:
    """Test parity for line formation scenarios.

    Python AI engine does not yet expose full line-processing moves; for
    TS parity we construct a canonical line_formation move referencing a
    pre-formed line for Player 1.
    """

    state = create_initial_state()
    state.current_phase = GamePhase.LINE_PROCESSING

    # Create a line for Player 1: (0,0) to (0,4)
    positions = [Position(x=0, y=i) for i in range(5)]
    line = LineInfo(
        positions=positions,
        player=1,
        length=5,
        direction=Position(x=0, y=1),
    )
    state.board.formed_lines = [line]

    canonical_line_move_data = {
        "id": "simulated",
        "type": MoveType.LINE_FORMATION,
        "player": 1,
        "from": None,
        "to": positions[0],
        "captureTarget": None,
        "placedOnStack": None,
        "placementCount": None,
        "timestamp": state.last_move_at,
        "thinkTime": 0,
        "moveNumber": 1,
    }
    canonical_line_move = Move(**canonical_line_move_data)

    validate_move_parity(state, canonical_line_move, "line_formation_0")


def test_territory_collapse_parity() -> None:
    """Test parity for territory collapse scenarios.

    As with line formation, Python’s AI engine does not yet synthesize
    explicit territory_claim moves. For parity we construct a canonical
    territory_claim move for an eligible disconnected region.
    """

    state = create_initial_state()
    state.current_phase = GamePhase.TERRITORY_PROCESSING

    # Create a disconnected territory for Player 1: region at (7,7)
    territory = Territory(
        spaces=[Position(x=7, y=7)],
        controllingPlayer=1,
        isDisconnected=True,
    )
    state.board.territories["region_1"] = territory

    # Need a stack outside to allow processing
    p1_stack = RingStack(
        position=Position(x=0, y=0),
        rings=[1],
        stackHeight=1,
        capHeight=1,
        controllingPlayer=1,
    )
    state.board.stacks["0,0"] = p1_stack

    canonical_territory_move_data = {
        "id": "simulated",
        "type": MoveType.TERRITORY_CLAIM,
        "player": 1,
        "from": None,
        "to": Position(x=7, y=7),
        "captureTarget": None,
        "placedOnStack": None,
        "placementCount": None,
        "timestamp": state.last_move_at,
        "thinkTime": 0,
        "moveNumber": 1,
    }
    canonical_territory_move = Move(**canonical_territory_move_data)

    validate_move_parity(state, canonical_territory_move, "territory_0")


def test_cyclic_capture_parity() -> None:
    """Test parity for cyclic capture scenarios.

    Constructs a scenario where a cyclic capture is possible and validates
    that the TS engine accepts the capture segments.
    """
    state = create_initial_state()
    state.current_phase = GamePhase.CAPTURE
    state.current_player = 1

    # Setup cyclic scenario (simplified triangle)
    # P1 at (2,2) height 2
    # P2 at (2,3) height 1
    # P2 at (3,2) height 1
    # Empty at (2,4) and (4,2)

    p1_stack = RingStack(
        position=Position(x=2, y=2),
        rings=[1, 1],
        stackHeight=2,
        capHeight=2,
        controllingPlayer=1,
    )
    state.board.stacks["2,2"] = p1_stack

    p2_stack1 = RingStack(
        position=Position(x=2, y=3),
        rings=[2],
        stackHeight=1,
        capHeight=1,
        controllingPlayer=2,
    )
    state.board.stacks["2,3"] = p2_stack1

    p2_stack2 = RingStack(
        position=Position(x=3, y=2),
        rings=[2],
        stackHeight=1,
        capHeight=1,
        controllingPlayer=2,
    )
    state.board.stacks["3,2"] = p2_stack2

    # Move 1: (2,2) -> over (2,3) -> (2,4)
    move1_data = {
        "id": "simulated_cyclic_1",
        "type": MoveType.OVERTAKING_CAPTURE,
        "player": 1,
        "from": Position(x=2, y=2),
        "to": Position(x=2, y=4),
        "captureTarget": Position(x=2, y=3),
        "placedOnStack": None,
        "placementCount": None,
        "timestamp": state.last_move_at,
        "thinkTime": 0,
        "moveNumber": 1,
    }
    move1 = Move(**move1_data)

    validate_move_parity(state, move1, "cyclic_capture_1")


def test_forced_elimination_parity() -> None:
    """Test parity for forced elimination scenarios.

    Constructs a scenario where a player is blocked and has no rings in hand,
    forcing an elimination move.
    """
    state = create_initial_state()
    state.current_phase = GamePhase.RING_PLACEMENT
    state.current_player = 1

    # Player 1 has no rings in hand
    state.players[0].rings_in_hand = 0

    # Player 1 has a stack at (0,0)
    p1_stack = RingStack(
        position=Position(x=0, y=0),
        rings=[1, 1],
        stackHeight=2,
        capHeight=2,
        controllingPlayer=1,
    )
    state.board.stacks["0,0"] = p1_stack

    # Block (0,0) with collapsed spaces
    state.board.collapsed_spaces["1,0"] = 0
    state.board.collapsed_spaces["0,1"] = 0
    state.board.collapsed_spaces["1,1"] = 0

    # Forced elimination move
    forced_elim_move_data = {
        "id": "simulated_forced_elim",
        "type": MoveType.FORCED_ELIMINATION,
        "player": 1,
        "from": None,
        "to": Position(x=0, y=0),  # Target stack
        "captureTarget": None,
        "placedOnStack": None,
        "placementCount": None,
        "timestamp": state.last_move_at,
        "thinkTime": 0,
        "moveNumber": 1,
    }
    move = Move(**forced_elim_move_data)

    validate_move_parity(state, move, "forced_elimination_1")


if __name__ == "__main__":
    pytest.main([__file__])
