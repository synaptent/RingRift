import sys
import os
import json
import random
from datetime import datetime

# Add app directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

from app.models import (
    GameState, BoardState, BoardType, GamePhase, Player, TimeControl,
    GameStatus, RingStack, Position
)
from app.game_engine import GameEngine
from app.board_manager import BoardManager


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

def generate_random_game_trace(seed, max_moves=50, scenario=None):
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

    trace = []
    
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


if __name__ == "__main__":
    output_dir = os.path.join(os.path.dirname(__file__), "vectors")
    os.makedirs(output_dir, exist_ok=True)
    
    seeds = [42, 123, 999]
    
    for seed in seeds:
        print(f"Generating trace for seed {seed}...")
        trace = generate_random_game_trace(seed)
        
        output_file = os.path.join(output_dir, f"trace_seed_{seed}.json")
        with open(output_file, "w") as f:
            json.dump(trace, f, indent=2)

    # Generate scenario traces
    print("Generating chain capture scenario trace...")
    trace_chain = generate_random_game_trace(1001, scenario="chain_capture")
    scenario_path = os.path.join(
        output_dir,
        "trace_scenario_chain_capture.json",
    )
    with open(scenario_path, "w") as f:
        json.dump(trace_chain, f, indent=2)
            
    print("Done generating test vectors.")
