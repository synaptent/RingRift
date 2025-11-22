import sys
import os
import time
import argparse

from app.models import GameState, BoardType, GamePhase, GameStatus, Player, TimeControl, BoardState, AIConfig
from app.ai.heuristic_ai import HeuristicAI
from app.ai.minimax_ai import MinimaxAI
from app.ai.mcts_ai import MCTSAI
from app.ai.random_ai import RandomAI
from app.game_engine import GameEngine
from app.board_manager import BoardManager
import uuid
from datetime import datetime

def create_game_state():
    board = BoardState(
        type=BoardType.SQUARE8,
        size=8,
        stacks={},
        markers={},
        collapsedSpaces={},
        eliminatedRings={}
    )
    
    players = [
        Player(
            id="player1",
            username="AI_1",
            type="ai",
            playerNumber=1,
            isReady=True,
            timeRemaining=600000,
            aiDifficulty=5,
            ringsInHand=20,
            eliminatedRings=0,
            territorySpaces=0
        ),
        Player(
            id="player2",
            username="AI_2",
            type="ai",
            playerNumber=2,
            isReady=True,
            timeRemaining=600000,
            aiDifficulty=5,
            ringsInHand=20,
            eliminatedRings=0,
            territorySpaces=0
        )
    ]
    
    return GameState(
        id=str(uuid.uuid4()),
        boardType=BoardType.SQUARE8,
        board=board,
        players=players,
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

def run_game(ai1, ai2):
    game_state = create_game_state()
    move_count = 0
    max_moves = 200
    
    print(f"Starting game: {ai1.__class__.__name__} (P1) vs {ai2.__class__.__name__} (P2)")
    
    while game_state.game_status == GameStatus.ACTIVE and move_count < max_moves:
        current_player_num = game_state.current_player
        current_ai = ai1 if current_player_num == 1 else ai2
        
        start_time = time.time()
        move = current_ai.select_move(game_state)
        duration = time.time() - start_time
        
        if not move:
            print(f"No valid moves for Player {current_player_num}. Game Over.")
            game_state.game_status = GameStatus.FINISHED
            game_state.winner = 2 if current_player_num == 1 else 1
            break
            
        print(f"Move {move_count}: Player {current_player_num} {move.type} to {move.to.to_key()} ({duration:.2f}s)")
        game_state = GameEngine.apply_move(game_state, move)
        
        # Log game stats
        snapshot = BoardManager.compute_progress_snapshot(game_state)
        p1 = next(p for p in game_state.players if p.player_number == 1)
        p2 = next(p for p in game_state.players if p.player_number == 2)
        
        print(
            f"  Stats: S={snapshot.S}, Markers={snapshot.markers}, "
            f"Collapsed={snapshot.collapsed}, Eliminated={snapshot.eliminated}"
        )
        print(
            f"  P1: Hand={p1.rings_in_hand}, Elim={p1.eliminated_rings}, "
            f"Terr={p1.territory_spaces}"
        )
        print(
            f"  P2: Hand={p2.rings_in_hand}, Elim={p2.eliminated_rings}, "
            f"Terr={p2.territory_spaces}"
        )
        
        # Check victory
        p1 = next(p for p in game_state.players if p.player_number == 1)
        p2 = next(p for p in game_state.players if p.player_number == 2)
        
        if p1.eliminated_rings >= game_state.victory_threshold:
            game_state.game_status = GameStatus.FINISHED
            game_state.winner = 1
        elif p2.eliminated_rings >= game_state.victory_threshold:
            game_state.game_status = GameStatus.FINISHED
            game_state.winner = 2
            
        move_count += 1
        
    if game_state.game_status == GameStatus.ACTIVE:
        print("Max moves reached. Draw.")
        return 0
        
    print(f"Game Over. Winner: Player {game_state.winner}")
    return game_state.winner

def run_tournament(rounds=2):
    ais = {
        "Random": RandomAI(1, AIConfig(difficulty=1, thinkTime=0)),
        "Heuristic": HeuristicAI(1, AIConfig(difficulty=5, thinkTime=0)),
        "Minimax": MinimaxAI(1, AIConfig(difficulty=3, thinkTime=0)), # Lower difficulty for speed
        "MCTS": MCTSAI(1, AIConfig(difficulty=3, thinkTime=0))
    }
    
    results = {name: 0 for name in ais}
    
    matchups = [
        ("Heuristic", "Random"),
        ("Minimax", "Random"),
        ("MCTS", "Random"),
        ("Minimax", "Heuristic"),
        ("MCTS", "Heuristic"),
        ("MCTS", "Minimax")
    ]
    
    for p1_name, p2_name in matchups:
        print(f"\nMatchup: {p1_name} vs {p2_name}")
        p1_wins = 0
        p2_wins = 0
        draws = 0
        
        for i in range(rounds):
            # Game 1: P1 vs P2
            ai1 = ais[p1_name]
            ai2 = ais[p2_name]
            ai1.player_number = 1
            ai2.player_number = 2
            
            winner = run_game(ai1, ai2)
            if winner == 1: p1_wins += 1
            elif winner == 2: p2_wins += 1
            else: draws += 1
            
            # Game 2: P2 vs P1 (swap)
            ai1 = ais[p2_name]
            ai2 = ais[p1_name]
            ai1.player_number = 1
            ai2.player_number = 2
            
            winner = run_game(ai1, ai2)
            if winner == 1: p2_wins += 1
            elif winner == 2: p1_wins += 1
            else: draws += 1
            
        print(f"Result: {p1_name} {p1_wins} - {p2_wins} {p2_name} ({draws} draws)")
        results[p1_name] += p1_wins
        results[p2_name] += p2_wins

    print("\nTournament Results:")
    for name, score in sorted(results.items(), key=lambda x: x[1], reverse=True):
        print(f"{name}: {score}")

if __name__ == "__main__":
    run_tournament(rounds=1)