"""
Debug script to investigate 'No move found' scenarios
Runs a single game and dumps state when AI fails to find a move
"""
import sys
import os
import time
import random
from datetime import datetime
import json

# Add app to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

from app.models import GameState, BoardType, BoardState, GamePhase, GameStatus, TimeControl, Player, AIConfig
from app.game_engine import GameEngine
from app.ai.mcts_ai import MCTSAI
from app.ai.descent_ai import DescentAI
from app.board_manager import BoardManager

def create_initial_state():
    return GameState(
        id="debug_game",
        boardType=BoardType.SQUARE8,
        board=BoardState(
            type=BoardType.SQUARE8,
            size=8,
            stacks={},
            markers={},
            collapsedSpaces={},
            eliminatedRings={}
        ),
        players=[
            Player(id="p1", username="Descent", type="ai", playerNumber=1, isReady=True, timeRemaining=600, ringsInHand=18, eliminatedRings=0, territorySpaces=0, aiDifficulty=5),
            Player(id="p2", username="MCTS", type="ai", playerNumber=2, isReady=True, timeRemaining=600, ringsInHand=18, eliminatedRings=0, territorySpaces=0, aiDifficulty=5)
        ],
        currentPhase=GamePhase.RING_PLACEMENT,
        currentPlayer=1,
        moveHistory=[],
        timeControl=TimeControl(initialTime=600, increment=0, type="blitz"),
        gameStatus=GameStatus.ACTIVE,
        createdAt=datetime.now(),
        lastMoveAt=datetime.now(),
        isRated=False,
        maxPlayers=2,
        totalRingsInPlay=36,
        totalRingsEliminated=0,
        victoryThreshold=19,
        territoryVictoryThreshold=33
    )

def print_board_state(state: GameState):
    print("\n=== Board State Dump ===")
    print(f"Phase: {state.current_phase}")
    print(f"Current Player: {state.current_player}")
    
    p1 = next(p for p in state.players if p.player_number == 1)
    p2 = next(p for p in state.players if p.player_number == 2)
    print(f"P1 Rings in Hand: {p1.rings_in_hand}")
    print(f"P2 Rings in Hand: {p2.rings_in_hand}")
    
    print("\nStacks:")
    for k, v in state.board.stacks.items():
        print(f"  {k}: Player {v.controlling_player}, Height {v.stack_height}, Cap {v.cap_height}, Rings {v.rings}")
        
    print("\nMarkers:")
    for k, v in state.board.markers.items():
        print(f"  {k}: Player {v.player}")
        
    print("\nCollapsed:")
    print(state.board.collapsed_spaces)
    print("========================\n")

def run_debug_game():
    print("Running debug game: Descent (P1) vs MCTS (P2)")
    
    # Use moderate think time
    config = AIConfig(difficulty=5, randomness=0.1, thinkTime=500)
    
    p1_ai = DescentAI(1, config)
    p2_ai = MCTSAI(2, config)
    
    state = create_initial_state()
    move_count = 0
    
    while state.game_status == GameStatus.ACTIVE and move_count < 200:
        current_player = state.current_player
        ai = p1_ai if current_player == 1 else p2_ai
        ai_name = "Descent" if current_player == 1 else "MCTS"
        
        print(f"Move {move_count + 1}: {ai_name} (P{current_player}) thinking...", end="", flush=True)
        
        start_time = time.time()
        try:
            move = ai.select_move(state)
        except Exception as e:
            print(f"\nERROR in select_move: {e}")
            import traceback
            traceback.print_exc()
            print_board_state(state)
            return

        duration = time.time() - start_time
        print(f" Done ({duration:.2f}s)")
        
        if not move:
            print(f"\n!!! No move found for {ai_name} (P{current_player}) !!!")
            print("Checking if GameEngine generates any moves...")
            valid_moves = GameEngine.get_valid_moves(state, current_player)
            print(f"GameEngine.get_valid_moves returned {len(valid_moves)} moves.")
            if len(valid_moves) > 0:
                print("First 3 valid moves:")
                for m in valid_moves[:3]:
                    print(f"  {m.type} to {m.to}")
            
            print_board_state(state)
            break
            
        try:
            state = GameEngine.apply_move(state, move)
            move_count += 1
            print(f"  Applied: {move.type} to {move.to}")
        except Exception as e:
            print(f"\nERROR applying move: {e}")
            import traceback
            traceback.print_exc()
            print_board_state(state)
            break
            
    print(f"Game ended. Winner: {state.winner}")

if __name__ == "__main__":
    run_debug_game()