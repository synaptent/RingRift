import sys
import os
import json
import subprocess
import random
import pytest
from datetime import datetime

# Archived Python↔TS rules parity harness using subprocess-based TS CLIs.
#
# Original location: ai-service/tests/parity/test_rules_parity.py
# Status:
# - Marked skipped by default due to TS CLI timeouts in CI.
# - Superseded in practice by fixture-based parity tests in
#   ai-service/tests/parity/test_rules_parity_fixtures.py and related suites.
# - Retained here for historical reference; file name is prefixed with
#   "archived_" so it is not collected by pytest (python_files = test_*.py).

# TODO-RULES-PARITY-SUBPROCESS: These tests use subprocess calls to TS CLI
# (test-parity-cli.ts, test-sandbox-parity-cli.ts) which frequently timeout
# in CI environments due to ts-node startup overhead and inter-process
# communication delays. Skip pending optimization of the TS CLI subprocess
# architecture or conversion to a single-process test harness.
pytestmark = pytest.mark.skip(
    reason="TODO-RULES-PARITY-SUBPROCESS: TS CLI subprocess timeouts"
)

# Test timeout guards to prevent hanging in CI
SUBPROCESS_TIMEOUT_SECONDS = 30
TEST_TIMEOUT_SECONDS = 30

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
                cmd, capture_output=True, text=True, check=True,
                timeout=SUBPROCESS_TIMEOUT_SECONDS
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
            cmd_sandbox, capture_output=True, text=True, check=True,
            timeout=SUBPROCESS_TIMEOUT_SECONDS
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
            print(
                f"Sandbox validation failed for move: {move}\n"
                f"Output: {output_sandbox}"
            )

        assert output_sandbox["isValid"], (
            f"Move {move} is valid in Python but invalid in Sandbox "
            "(ClientSandboxEngine)"
        )

    finally:
        # Clean up temporary file
        if os.path.exists(input_file):
            os.remove(input_file)

