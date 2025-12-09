import sys
import os
import json
import random
from datetime import datetime
from typing import Any, Dict, List

# Add app directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

from app.models import (  # type: ignore[import]
    GameState,
    BoardState,
    BoardType,
    GamePhase,
    Player,
    TimeControl,
    GameStatus,
    RingStack,
    Position,
)
from app.game_engine import GameEngine  # type: ignore[import]
from app.board_manager import BoardManager  # type: ignore[import]
from app.db.game_replay import GameReplayDB  # type: ignore[import]


def create_initial_state():
    return GameState(
        id="test-game",
        boardType=BoardType.SQUARE8,
        board=BoardState(type=BoardType.SQUARE8, size=8),
        players=[
            Player(
                id="p1", username="Player 1", type="human", playerNumber=1,
                isReady=True, timeRemaining=600, ringsInHand=20,
                eliminatedRings=0, territorySpaces=0, aiDifficulty=None
            ),
            Player(
                id="p2", username="Player 2", type="human", playerNumber=2,
                isReady=True, timeRemaining=600, ringsInHand=20,
                eliminatedRings=0, territorySpaces=0, aiDifficulty=None
            )
        ],
        currentPhase=GamePhase.RING_PLACEMENT,
        currentPlayer=1,
        moveHistory=[],
        timeControl=TimeControl(initialTime=600, increment=5, type="standard"),
        gameStatus=GameStatus.ACTIVE,
        createdAt=datetime.now(),
        lastMoveAt=datetime.now(),
        isRated=False,
        maxPlayers=2,
        totalRingsInPlay=0,
        totalRingsEliminated=0,
        victoryThreshold=3,
        territoryVictoryThreshold=10
    )


def generate_random_game_trace(seed: int, max_moves: int = 50, scenario: str | None = None):
    random.seed(seed)
    state = create_initial_state()
    
    if scenario == "chain_capture":
        # Setup chain capture scenario
        state.current_phase = GamePhase.MOVEMENT
        # Player 1 at (2,2) height 2
        p1_stack = RingStack(
            position=Position(x=2, y=2),
            rings=[1, 1],
            stackHeight=2,
            capHeight=2,
            controllingPlayer=1
        )
        state.board.stacks["2,2"] = p1_stack
        
        # Player 2 at (2,3) height 1
        p2_stack1 = RingStack(
            position=Position(x=2, y=3),
            rings=[2],
            stackHeight=1,
            capHeight=1,
            controllingPlayer=2
        )
        state.board.stacks["2,3"] = p2_stack1
        
        # Player 2 at (2,5) height 1
        p2_stack2 = RingStack(
            position=Position(x=2, y=5),
            rings=[2],
            stackHeight=1,
            capHeight=1,
            controllingPlayer=2
        )
        state.board.stacks["2,5"] = p2_stack2
        
    elif scenario == "forced_elimination":
        # Setup forced elimination scenario
        # Player 1 has no rings in hand and no moves
        # This is hard to setup perfectly randomly, but we can try to create a blocked state
        pass

    trace: List[Dict[str, Any]] = []
    
    for _ in range(max_moves):
        if state.game_status != GameStatus.ACTIVE:
            break
            
        moves = GameEngine.get_valid_moves(state, state.current_player)
        if not moves:
            break
            
        move = random.choice(moves)
        
        # Capture state before move
        state_before = json.loads(state.model_dump_json(by_alias=True))
        
        # Apply move
        state = GameEngine.apply_move(state, move)
        
        # Capture state after move
        state_after = json.loads(state.model_dump_json(by_alias=True))
        
        move_dict = json.loads(move.model_dump_json(by_alias=True))

        # Normalize capture move types to the segmented canonical model used
        # by the TypeScript engines and shared Move type.
        # The Python GameEngine currently uses "chain_capture" for
        # follow-up segments in a capture chain.
        # In the TS model, these are represented as
        # "continue_capture_segment" moves with from/captureTarget/to
        # populated per segment.
        # We rewrite the type here so that the exported parity vectors
        # speak the same segmented dialect as the TS backend and sandbox.
        if move_dict.get("type") == "chain_capture":
            move_dict["type"] = "continue_capture_segment"
        
        trace.append({
            "stateBefore": state_before,
            "move": move_dict,
            "stateAfter": state_after,
            "sInvariant": BoardManager.compute_progress_snapshot(state).S,
            "stateHash": BoardManager.hash_game_state(state)
        })
        
    return trace


def generate_trace_from_replay_db(
    db_path: str,
    game_id: str,
    max_moves: int | None = None,
) -> List[Dict[str, Any]]:
    """
    Generate a Python→TS parity trace from a recorded self-play GameReplayDB game.

    This uses GameReplayDB history entries (state_before/state_after) plus the
    canonical moves from game_moves to build the same vector shape consumed by
    tests/unit/Python_vs_TS.traceParity.test.ts:

      { stateBefore, move, stateAfter, sInvariant, stateHash }
    """
    db = GameReplayDB(db_path)
    meta = db.get_game_metadata(game_id)
    if not meta:
        raise RuntimeError(f"Game {game_id} not found in {db_path}")

    total_moves = int(meta["total_moves"])
    limit = total_moves if max_moves is None else min(total_moves, max_moves)

    moves = db.get_moves(game_id, start=0, end=limit)
    history = db.get_all_history_entries(game_id, include_full_states=True)

    if not history:
        raise RuntimeError(f"No history entries for game {game_id} in {db_path}")

    trace: List[Dict[str, Any]] = []

    for idx in range(min(limit, len(moves), len(history))):
        entry = history[idx]
        move = moves[idx]

        state_before: GameState
        state_after: GameState

        if entry.get("state_before") is not None and entry.get("state_after") is not None:
            state_before = entry["state_before"]
            state_after = entry["state_after"]
        else:
            # Fallback reconstruction path (should be rare for self-play DBs)
            if idx == 0:
                initial = db.get_initial_state(game_id)
                if initial is None:
                    raise RuntimeError(f"Missing initial_state for {game_id}")
                state_before = initial
            else:
                prev_state = db.get_state_at_move(game_id, idx - 1)
                if prev_state is None:
                    raise RuntimeError(f"get_state_at_move({game_id}, {idx-1}) returned None")
                state_before = prev_state

            after_state = db.get_state_at_move(game_id, idx)
            if after_state is None:
                raise RuntimeError(f"get_state_at_move({game_id}, {idx}) returned None")
            state_after = after_state

        state_before_json = json.loads(state_before.model_dump_json(by_alias=True))
        state_after_json = json.loads(state_after.model_dump_json(by_alias=True))

        move_dict = json.loads(move.model_dump_json(by_alias=True))
        # Normalise legacy 'chain_capture' moves to the segmented canonical
        # 'continue_capture_segment' type used by TS engines.
        if move_dict.get("type") == "chain_capture":
            move_dict["type"] = "continue_capture_segment"

        s_invariant = BoardManager.compute_progress_snapshot(state_after).S
        state_hash = BoardManager.hash_game_state(state_after)

        trace.append(
            {
                "stateBefore": state_before_json,
                "move": move_dict,
                "stateAfter": state_after_json,
                "sInvariant": s_invariant,
                "stateHash": state_hash,
            }
        )

    return trace


def _maybe_generate_parity_vectors(output_dir: str) -> None:
    """
    Medium-term helper: use parity_summary.latest.json to pick a few
    representative replay-parity divergences and export full Python
    traces for them as TS traceParity vectors.
    """
    summary_path = os.path.join(os.path.dirname(__file__), "../../parity_summary.latest.json")
    if not os.path.exists(summary_path):
        print("No parity_summary.latest.json found; skipping parity-based vector generation.")
        return

    try:
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)
    except Exception as exc:  # pragma: no cover - defensive
        print(f"Failed to load parity_summary.latest.json: {exc}")
        return

    semantic = summary.get("semantic_divergences") or []
    if not semantic:
        print("No semantic_divergences in parity_summary.latest.json; nothing to generate.")
        return

    representatives: Dict[str, Dict[str, Any]] = {}

    for entry in semantic:
        ps = entry.get("python_summary") or {}
        ts = entry.get("ts_summary") or {}
        if not ps or not ts:
            continue

        # Capture vs movement (including chain_capture vs movement)
        if (
            "current_phase" in entry.get("mismatch_kinds", [])
            and ps.get("current_phase") in ("capture", "chain_capture")
            and ts.get("current_phase") == "movement"
            and "capture_movement" not in representatives
        ):
            representatives["capture_movement"] = entry

        # Movement vs territory_processing
        if (
            "current_phase" in entry.get("mismatch_kinds", [])
            and ps.get("current_phase") == "movement"
            and ts.get("current_phase") == "territory_processing"
            and "movement_territory" not in representatives
        ):
            representatives["movement_territory"] = entry

        # Active vs completed status
        if (
            "game_status" in entry.get("mismatch_kinds", [])
            and ps.get("game_status") == "active"
            and ts.get("game_status") == "completed"
            and "active_completed" not in representatives
        ):
            representatives["active_completed"] = entry

        if len(representatives) == 3:
            break

    if not representatives:
        print("No suitable semantic divergences found for parity-based vectors.")
        return

    for label, entry in representatives.items():
        db_path = entry["db_path"]
        game_id = entry["game_id"]
        short_id = game_id.split("-")[0]
        print(f"Generating parity trace vector '{label}' for game {game_id}...")
        try:
            trace = generate_trace_from_replay_db(db_path, game_id)
        except Exception as exc:  # pragma: no cover - defensive
            print(f"  Skipping {game_id} ({label}) due to error: {exc}")
            continue

        out_name = f"trace_parity_{label}_{short_id}.json"
        out_path = os.path.join(output_dir, out_name)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(trace, f, indent=2)
        print(f"  Wrote {out_path}")


if __name__ == "__main__":
    output_dir = os.path.join(os.path.dirname(__file__), "vectors")
    os.makedirs(output_dir, exist_ok=True)

    # Copy recovery contract vectors into the parity vectors folder to ensure
    # parity harnesses exercise recovery Option 1/2 semantics.
    try:
        recovery_src = os.path.join(
            os.path.dirname(__file__),
            "../../tests/fixtures/contract-vectors/v2/recovery_action.vectors.json",
        )
        if os.path.exists(recovery_src):
            recovery_dst = os.path.join(output_dir, "recovery_action.vectors.json")
            with open(recovery_src, "r", encoding="utf-8") as f_in:
                with open(recovery_dst, "w", encoding="utf-8") as f_out:
                    f_out.write(f_in.read())
            print(f"Copied recovery_action.vectors.json to {recovery_dst}")
    except Exception as exc:  # pragma: no cover - defensive
        print(f"Failed to copy recovery vectors: {exc}")

    seeds = [42, 123, 999]

    for seed in seeds:
        print(f"Generating trace for seed {seed}...")
        trace = generate_random_game_trace(seed)

        output_file = os.path.join(output_dir, f"trace_seed_{seed}.json")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(trace, f, indent=2)

    # Generate scenario traces
    print("Generating chain capture scenario trace...")
    trace_chain = generate_random_game_trace(1001, scenario="chain_capture")
    scenario_path = os.path.join(
        output_dir,
        "trace_scenario_chain_capture.json",
    )
    with open(scenario_path, "w", encoding="utf-8") as f:
        json.dump(trace_chain, f, indent=2)

    # Generate parity-based traces from existing replay divergences (if any)
    _maybe_generate_parity_vectors(output_dir)

    print("Done generating test vectors.")
