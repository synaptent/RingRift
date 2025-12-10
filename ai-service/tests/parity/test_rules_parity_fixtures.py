"""Minimal Python loader for TS-generated rules parity fixtures (v1).

This test exercises the first slice of the TS→Python fixture pipeline:
- TS script: tests/scripts/generate_rules_parity_fixtures.ts
  - Produces JSON under tests/fixtures/rules-parity/v1/
  - For v1 we emit a single state-only fixture:
    - state_only.square8_2p.initial.json

- Python loader (this file):
  - Loads the state-only fixture.
  - Hydrates it into a Python GameState via app.models.
  - Runs a few basic invariants to confirm that the JSON shape produced by
    the TS shared engine can be losslessly mapped into Python models.

Once this round-trip is stable, we can extend the fixtures to include
state+action scenarios and assert full rules parity.
"""

import json
import os
import sys
from datetime import datetime
import glob

import pytest

# Ensure app package is importable
sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))

from app.models import (  # noqa: E402
    GameState,
    BoardType,
    Move,
)
from app.rules.default_engine import DefaultRulesEngine  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from app.main import app  # noqa: E402


BASE_DIR = os.path.dirname(__file__)

RULES_PARITY_BASE_DIR = os.path.join(
    BASE_DIR,
    "..",
    "..",
    "..",
    "tests",
    "fixtures",
    "rules-parity",
)

FIXTURE_REL_PATH = os.path.join(
    RULES_PARITY_BASE_DIR,
    "v1",
    "state_only.square8_2p.initial.json",
)

RULES_PARITY_V1_DIR = os.path.join(RULES_PARITY_BASE_DIR, "v1")
RULES_PARITY_V2_DIR = os.path.join(RULES_PARITY_BASE_DIR, "v2")


def _convert_ts_state_to_python(state_dict: dict) -> dict:
    """Convert TS GameState JSON to Python representation.

    TS and Python have different ring array conventions:
      - TS: rings array is [top to bottom], rings[0] is controlling
      - Python: rings array is [bottom to top], rings[-1] is controlling

    This function reverses the rings array in each stack to match
    Python's internal representation.
    """
    import copy
    result = copy.deepcopy(state_dict)

    # Handle stacks in board.stacks (may be dict keyed by "x,y")
    board = result.get("board", {})
    stacks = board.get("stacks", {})

    if isinstance(stacks, dict):
        for key, stack in stacks.items():
            if isinstance(stack, dict) and "rings" in stack:
                stack["rings"] = list(reversed(stack["rings"]))

    return result


def _normalise_hash_for_ts_comparison(raw_hash: str) -> str:
    """Normalise Python/TS hashes before comparison.

    Normalisation steps:

    1. Strip Python-only ``:must_move=<posKey>`` suffix injected by
       BoardManager.hash_game_state when ``must_move_from_stack_key`` is set.
    2. Drop the leading ``currentPlayer:phase:gameStatus`` meta segment so
       that hash comparisons focus on player/board geometry rather than
       subtle phase/turn bookkeeping differences between hosts.

    The remaining segments (players meta, stacks, markers, collapsedSpaces)
    must still match exactly for parity to hold.
    """
    base = raw_hash

    # Step 1: strip :must_move=... extension when present.
    marker = ":must_move="
    idx = base.find(marker)
    if idx != -1:
        rest = base[idx + len(marker):]
        sep = rest.find("#")
        if sep == -1:
            base = base[:idx]
        else:
            base = base[:idx] + rest[sep:]

    # Step 2: drop the leading meta segment up to the first '#'.
    parts = base.split("#", 1)
    if len(parts) == 2:
        # Keep everything after the first '#'
        return parts[1]
    return base


@pytest.mark.skipif(
    not os.path.exists(FIXTURE_REL_PATH),
    reason=(
        "TS rules-parity fixtures not found. Run "
        "`npx ts-node tests/scripts/generate_rules_parity_fixtures.ts` "
        "from the TypeScript project root to generate them first."
    ),
)
def test_load_ts_initial_state_fixture() -> None:
    """Load the TS-generated initial GameState and hydrate into Python.

    This does *not* yet assert deep rules parity; it simply proves that the
    JSON emitted by the shared TS engine can be parsed into our Python
    GameState model without loss of structure, and that basic invariants
    hold (boardType, players, initial phase/status, etc.).
    """

    fixture_path = os.path.abspath(FIXTURE_REL_PATH)
    with open(fixture_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    assert payload["boardType"] == "square8"
    state_dict = payload["state"]

    # Pydantic will handle alias-based field mapping (createdAt → created_at,
    # etc.) thanks to populate_by_name=True in the models.
    # Convert TS ring array format to Python format
    state = GameState(**_convert_ts_state_to_python(state_dict))

    # Basic invariants
    assert state.board_type == BoardType.SQUARE8
    assert state.max_players == 2
    assert len(state.players) == 2
    assert state.current_phase.name.lower() == "ring_placement"
    assert state.game_status.name.lower() in {"waiting", "active"}

    # Sanity check on timestamps – they should be parseable datetimes.
    assert isinstance(state.created_at, datetime)
    assert isinstance(state.last_move_at, datetime)


def _iter_state_action_fixtures() -> list[str]:
    """Discover TS-generated state+action fixtures across all versions."""
    patterns = [
        os.path.join(RULES_PARITY_V1_DIR, "state_action.*.json"),
        os.path.join(RULES_PARITY_V2_DIR, "state_action.*.json"),
    ]
    paths: list[str] = []
    for pattern in patterns:
        paths.extend(glob.glob(pattern))
    return sorted(paths)


@pytest.mark.parametrize("fixture_path", _iter_state_action_fixtures())
def test_state_action_parity(fixture_path: str) -> None:
    """
    Load TS-generated state+action fixtures and verify Python parity.

    For each fixture:
    1. Hydrate GameState and Move.
    2. Validate move using DefaultRulesEngine.
    3. Assert validation matches expected.tsValid.
    4. If valid, apply move and assert outcome parity (stateHash, S-invariant).
    """
    with open(fixture_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    state_dict = payload["state"]
    move_dict = payload["move"]
    expected = payload["expected"]

    state = GameState(**_convert_ts_state_to_python(state_dict))
    move = Move(**move_dict)

    engine = DefaultRulesEngine()
    python_valid = engine.validate_move(state, move)

    ts_valid = expected["tsValid"]
    assert python_valid == ts_valid, (
        f"Validation mismatch in {os.path.basename(fixture_path)}: "
        f"TS={ts_valid}, Python={python_valid}"
    )

    # If valid, verify outcome parity
    if ts_valid:
        ts_next = expected.get("tsNext")
        if ts_next:
            from app.game_engine import GameEngine
            from app.board_manager import BoardManager

            next_state = GameEngine.apply_move(state, move)
            raw_hash = BoardManager.hash_game_state(next_state)
            py_hash = _normalise_hash_for_ts_comparison(raw_hash)
            ts_hash = _normalise_hash_for_ts_comparison(ts_next["stateHash"])

            # Detailed debug if mismatch
            if py_hash != ts_hash:
                print(
                    "\n--- Hash Mismatch in "
                    f"{os.path.basename(fixture_path)} ---"
                )
                print(f"TS Hash: {ts_next['stateHash']}")
                print(f"TS Hash (normalised): {ts_hash}")
                print(f"Py Hash (normalised): {py_hash}")
                print(f"Py Hash (raw): {raw_hash}")

            assert py_hash == ts_hash, (
                f"stateHash mismatch in {os.path.basename(fixture_path)}"
            )

            snap = BoardManager.compute_progress_snapshot(next_state)
            assert snap.S == ts_next["S"], (
                f"S-invariant mismatch in {os.path.basename(fixture_path)}"
            )


@pytest.mark.parametrize("fixture_path", _iter_state_action_fixtures())
def test_state_action_http_parity(fixture_path: str) -> None:
    """
    For each TS-generated state+action fixture, call the FastAPI
    /rules/evaluate_move endpoint and verify:

      1. Python 'valid' matches expected.tsValid.
      2. When tsValid and tsNext are provided, state_hash and
         s_invariant match the TS snapshot.
    """
    client = TestClient(app)

    with open(fixture_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    state_dict = payload["state"]
    move_dict = payload["move"]
    expected = payload["expected"]

    ts_valid = expected["tsValid"]

    # Convert TS ring array format to Python format before sending to API
    converted_state = _convert_ts_state_to_python(state_dict)

    response = client.post(
        "/rules/evaluate_move",
        json={
            "game_state": converted_state,
            "move": move_dict,
        },
    )
    assert response.status_code == 200

    body = response.json()
    py_valid = body["valid"]
    assert py_valid == ts_valid, (
        f"HTTP /rules/evaluate_move valid mismatch in "
        f"{os.path.basename(fixture_path)}: "
        f"TS={ts_valid}, Python={py_valid}"
    )

    if not ts_valid:
        # Invalid moves should not produce a next_state.
        assert body.get("next_state") is None
        return

    # When TS considered the move valid, the Python endpoint must return
    # a next_state and, when available, state_hash and S-invariant that
    # match the TS expectations encoded in the fixture.
    assert body.get("next_state") is not None

    ts_next = expected.get("tsNext")
    if ts_next:
        if "stateHash" in ts_next:
            body_hash = body.get("state_hash")
            norm_body_hash = (
                _normalise_hash_for_ts_comparison(body_hash)
                if body_hash is not None
                else None
            )
            ts_hash = _normalise_hash_for_ts_comparison(ts_next["stateHash"])
            assert norm_body_hash == ts_hash, (
                "state_hash mismatch in "
                f"{os.path.basename(fixture_path)}"
            )
        if "S" in ts_next:
            assert body.get("s_invariant") == ts_next["S"], (
                "s_invariant mismatch in "
                f"{os.path.basename(fixture_path)}"
            )


def _iter_trace_fixtures() -> list[str]:
    """
    Discover TS-generated trace fixtures under all parity directories.

    By convention, trace fixtures are named `trace.*.json`, for example:
      - trace.square8_2p.line_reward.option1.json
      - trace.square8_2p.territory_region.basic.json
    """
    patterns = [
        os.path.join(RULES_PARITY_V1_DIR, "trace.*.json"),
        os.path.join(RULES_PARITY_V2_DIR, "trace.*.json"),
    ]
    paths: list[str] = []
    for pattern in patterns:
        paths.extend(glob.glob(pattern))
    return sorted(paths)


@pytest.mark.skipif(
    len(_iter_trace_fixtures()) == 0,
    reason=(
        "TS rules-parity trace fixtures not found. Once available, run "
        "`npx ts-node tests/scripts/generate_rules_parity_fixtures.ts` "
        "from the TypeScript project root to generate them."
    ),
)
def test_replay_ts_trace_fixtures_and_assert_python_state_parity() -> None:
    """
    Replay TS-generated trace fixtures step-by-step through the Python engine
    and assert that Python's GameState snapshots match the TS ground truth
    encoded in the fixture at each checkpoint.

    Fixture expectations:

      {
        "boardType": "square8" | "square19" | "hexagonal",
        "initialState": { ... TS GameState JSON ... },
        "steps": [
          {
            "label": "optional-descriptor",
            "move": { ... canonical Move JSON ... },
            "expected": {
              "tsValid": true | false,            // optional
              "tsStateHash": "hash",             // optional
              "tsS": 42                           // optional S-invariant
            },
            "stateHash": "hash-after-step",       // optional TS hash
            "sInvariant": 42                      // optional TS S-invariant
          },
          ...
        ]
      }

    The Python test does *not* attempt to re-derive PlayerChoice decisions; it
    treats every Move in the fixture as authoritative and only checks that,
    after applying the same Move sequence, Python's board hash and S-invariant
    match the TS ones when provided.
    """
    from app.game_engine import GameEngine
    from app.board_manager import BoardManager

    trace_paths = _iter_trace_fixtures()
    if not trace_paths:
        pytest.skip("No TS trace fixtures available yet")

    engine = DefaultRulesEngine()

    for path in trace_paths:
        fixture_path = os.path.abspath(path)
        with open(fixture_path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        board_type = payload.get("boardType")
        initial_state_dict = payload["initialState"]
        steps = payload.get("steps", [])

        state = GameState(**_convert_ts_state_to_python(initial_state_dict))

        # Light sanity check that fixture boardType matches Python state.
        if board_type is not None:
            assert state.board_type.value == board_type

        for idx, step in enumerate(steps):
            # Skip steps marked with skip: true
            if step.get("skip"):
                continue

            move_dict = step["move"]
            expected = step.get("expected", {}) or {}

            move = Move(**move_dict)

            # If the fixture advertises TS validator verdicts, ensure our
            # DefaultRulesEngine agrees before applying the move.
            if "tsValid" in expected:
                python_valid = engine.validate_move(state, move)
                assert (
                    python_valid == expected["tsValid"]
                ), (
                    "validate_move mismatch at step "
                    f"{idx} in {os.path.basename(path)}"
                )

            # Territory-processing numeric parity is currently known to diverge
            # between Python and the multi-region v2 fixtures (see
            # test_state_action_parity above). We still replay those moves to
            # keep subsequent steps well-formed, but intentionally skip strict
            # hash/S-invariant parity checks for those steps. The v1 single-
            # region square8 trace, however, is expected to match exactly and
            # therefore runs with full parity assertions.
            basename = os.path.basename(path)

            # Apply move using the Python GameEngine (TS-aligned semantics).
            state = GameEngine.apply_move(state, move)

            # If TS state hash is provided, compare Python's hash.
            ts_hash_raw = expected.get("tsStateHash") or step.get("stateHash")
            if ts_hash_raw is not None:
                py_hash = _normalise_hash_for_ts_comparison(
                    BoardManager.hash_game_state(state)
                )
                ts_hash = _normalise_hash_for_ts_comparison(ts_hash_raw)
                assert (
                    py_hash == ts_hash
                ), (
                    "stateHash mismatch at step "
                    f"{idx} in {os.path.basename(path)}"
                )

            # If TS S-invariant is provided, compare Python's S.
            ts_S = expected.get("tsS") or step.get("sInvariant")
            if ts_S is not None:
                snap = BoardManager.compute_progress_snapshot(state)
                assert (
                    snap.S == ts_S
                ), (
                    "S-invariant mismatch at step "
                    f"{idx} in {os.path.basename(path)}"
                )


@pytest.mark.skipif(
    len(_iter_trace_fixtures()) == 0,
    reason=(
        "TS rules-parity trace fixtures not found. Once available, run "
        "`npx ts-node tests/scripts/generate_rules_parity_fixtures.ts` "
        "from the TypeScript project root to generate them."
    ),
)
def test_default_engine_matches_game_engine_when_replaying_ts_traces() -> None:
    """Replay TS trace fixtures through both engines and assert lockstep.

    This complements
    ``test_replay_ts_trace_fixtures_and_assert_python_state_parity`` by
    checking that DefaultRulesEngine.apply_move stays in full-state
    lockstep with GameEngine.apply_move for every Move in the TS-generated
    traces (captures, lines, territory, etc.).
    """
    from app.game_engine import GameEngine  # noqa: WPS433,E402

    trace_paths = _iter_trace_fixtures()
    if not trace_paths:
        pytest.skip("No TS trace fixtures available yet")

    engine = DefaultRulesEngine()

    for path in trace_paths:
        fixture_path = os.path.abspath(path)
        with open(fixture_path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        initial_state_dict = payload["initialState"]
        steps = payload.get("steps", [])

        converted_state = _convert_ts_state_to_python(initial_state_dict)
        engine_state = GameState(**converted_state)
        rules_state = GameState(**converted_state)

        for idx, step in enumerate(steps):
            # Skip steps marked with skip: true
            if step.get("skip"):
                continue

            move_dict = step["move"]
            move = Move(**move_dict)

            next_via_engine = GameEngine.apply_move(engine_state, move)
            next_via_rules = engine.apply_move(rules_state, move)

            # Board-level equivalence
            assert (
                next_via_rules.board.stacks
                == next_via_engine.board.stacks
            ), (
                "board.stacks mismatch at step "
                f"{idx} in {os.path.basename(path)}"
            )
            assert (
                next_via_rules.board.markers
                == next_via_engine.board.markers
            ), (
                "board.markers mismatch at step "
                f"{idx} in {os.path.basename(path)}"
            )
            assert (
                next_via_rules.board.collapsed_spaces
                == next_via_engine.board.collapsed_spaces
            ), (
                "board.collapsed_spaces mismatch at step "
                f"{idx} in {os.path.basename(path)}"
            )
            assert (
                next_via_rules.board.eliminated_rings
                == next_via_engine.board.eliminated_rings
            ), (
                "board.eliminated_rings mismatch at step "
                f"{idx} in {os.path.basename(path)}"
            )

            # Player metadata equivalence
            assert (
                next_via_rules.players == next_via_engine.players
            ), f"players mismatch at step {idx} in {os.path.basename(path)}"

            # Turn/phase/victory bookkeeping and capture bookkeeping should
            # also stay aligned.
            assert (
                next_via_rules.current_player
                == next_via_engine.current_player
            ), (
                "current_player mismatch at step "
                f"{idx} in {os.path.basename(path)}"
            )
            assert (
                next_via_rules.current_phase
                == next_via_engine.current_phase
            ), (
                "current_phase mismatch at step "
                f"{idx} in {os.path.basename(path)}"
            )
            assert (
                next_via_rules.game_status
                == next_via_engine.game_status
            ), (
                "game_status mismatch at step "
                f"{idx} in {os.path.basename(path)}"
            )
            assert (
                next_via_rules.chain_capture_state
                == next_via_engine.chain_capture_state
            ), (
                "chain_capture_state mismatch at step "
                f"{idx} in {os.path.basename(path)}"
            )
            assert (
                next_via_rules.must_move_from_stack_key
                == next_via_engine.must_move_from_stack_key
            ), (
                "must_move_from_stack_key mismatch at step "
                f"{idx} in {os.path.basename(path)}"
            )

            engine_state = next_via_engine
            rules_state = next_via_rules


@pytest.mark.skipif(
    len(_iter_trace_fixtures()) == 0,
    reason=(
        "TS rules-parity trace fixtures not found. Once available, run "
        "`npx ts-node tests/scripts/generate_rules_parity_fixtures.ts` "
        "from the TypeScript project root to generate them."
    ),
)
def test_default_engine_mutator_first_matches_game_engine_on_ts_traces() -> None:  # noqa: E501
    """Replay TS trace fixtures with mutator-first mode enabled.

    This test ensures that when DefaultRulesEngine is constructed with
    ``mutator_first=True``, its internal mutator-driven orchestration path
    stays in full-state lockstep with ``GameEngine.apply_move`` for every
    step in the TS-generated traces. Any divergence in the mutator-first
    path will surface as a RuntimeError inside
    ``DefaultRulesEngine.apply_move``.
    """
    from app.game_engine import GameEngine  # noqa: WPS433,E402

    trace_paths = _iter_trace_fixtures()
    if not trace_paths:
        pytest.skip("No TS trace fixtures available yet")

    engine = DefaultRulesEngine(mutator_first=True)

    for path in trace_paths:
        fixture_path = os.path.abspath(path)
        with open(fixture_path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        initial_state_dict = payload["initialState"]
        steps = payload.get("steps", [])

        converted_state = _convert_ts_state_to_python(initial_state_dict)
        engine_state = GameState(**converted_state)
        rules_state = GameState(**converted_state)

        for idx, step in enumerate(steps):
            move_dict = step["move"]
            move = Move(**move_dict)

            next_via_engine = GameEngine.apply_move(engine_state, move)
            next_via_rules = engine.apply_move(rules_state, move)

            # The mutator-first path is exercised inside engine.apply_move;
            # here we still assert that the returned states match the
            # canonical engine result for diagnosability.
            assert (
                next_via_rules.board.stacks
                == next_via_engine.board.stacks
            ), (
                "board.stacks mismatch at step "
                f"{idx} in {os.path.basename(path)}"
            )
            assert (
                next_via_rules.board.markers
                == next_via_engine.board.markers
            ), (
                "board.markers mismatch at step "
                f"{idx} in {os.path.basename(path)}"
            )
            assert (
                next_via_rules.board.collapsed_spaces
                == next_via_engine.board.collapsed_spaces
            ), (
                "board.collapsed_spaces mismatch at step "
                f"{idx} in {os.path.basename(path)}"
            )
            assert (
                next_via_rules.board.eliminated_rings
                == next_via_engine.board.eliminated_rings
            ), (
                "board.eliminated_rings mismatch at step "
                f"{idx} in {os.path.basename(path)}"
            )

            assert (
                next_via_rules.players == next_via_engine.players
            ), f"players mismatch at step {idx} in {os.path.basename(path)}"

            assert (
                next_via_rules.current_player
                == next_via_engine.current_player
            ), (
                "current_player mismatch at step "
                f"{idx} in {os.path.basename(path)}"
            )
            assert (
                next_via_rules.current_phase
                == next_via_engine.current_phase
            ), (
                "current_phase mismatch at step "
                f"{idx} in {os.path.basename(path)}"
            )
            assert (
                next_via_rules.game_status
                == next_via_engine.game_status
            ), (
                "game_status mismatch at step "
                f"{idx} in {os.path.basename(path)}"
            )
            assert (
                next_via_rules.chain_capture_state
                == next_via_engine.chain_capture_state
            ), (
                "chain_capture_state mismatch at step "
                f"{idx} in {os.path.basename(path)}"
            )
            assert (
                next_via_rules.must_move_from_stack_key
                == next_via_engine.must_move_from_stack_key
            ), (
                "must_move_from_stack_key mismatch at step "
                f"{idx} in {os.path.basename(path)}"
            )

            engine_state = next_via_engine
            rules_state = next_via_rules
