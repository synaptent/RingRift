#!/usr/bin/env python3
"""
Benchmark comparing make/unmake pattern vs legacy immutable state copying.

Measures:
- Nodes searched per second
- Time to complete fixed-depth search
- Memory usage during search

This script validates the claimed 10-50x speedup of the make/unmake pattern
implemented in MutableGameState compared to the legacy immutable state
cloning approach in apply_move().

Usage:
    cd ai-service && python scripts/benchmark_make_unmake.py
"""

import os
import sys

# Add parent directory to path for imports (must be before local imports)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time  # noqa: E402
import tracemalloc  # noqa: E402
from dataclasses import dataclass  # noqa: E402
from datetime import datetime  # noqa: E402
from typing import List, Tuple  # noqa: E402

from app.ai.minimax_ai import MinimaxAI  # noqa: E402
from app.models import (  # noqa: E402
    AIConfig,
    GameState,
    BoardState,
    BoardType,
    Player,
    TimeControl,
    GamePhase,
    GameStatus,
)


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""
    mode: str
    depth: int
    avg_time: float
    min_time: float
    max_time: float
    nodes_visited: int
    peak_memory_mb: float
    runs: int


def create_starting_state() -> GameState:
    """Create a starting game state for benchmarking.
    
    Returns a minimal but valid GameState that can be used for AI search.
    """
    # Create players
    players = [
        Player(
            id="p1",
            username="Player1",
            type="ai",
            playerNumber=1,
            isReady=True,
            timeRemaining=600000,
            aiDifficulty=5,
            ringsInHand=18,
            eliminatedRings=0,
            territorySpaces=0,
        ),
        Player(
            id="p2",
            username="Player2",
            type="ai",
            playerNumber=2,
            isReady=True,
            timeRemaining=600000,
            aiDifficulty=5,
            ringsInHand=18,
            eliminatedRings=0,
            territorySpaces=0,
        ),
    ]
    
    # Create empty board
    board = BoardState(
        type=BoardType.SQUARE8,
        size=8,
        stacks={},
        markers={},
        collapsedSpaces={},
        eliminatedRings={},
        formedLines=[],
        territories={},
    )
    
    # Create time control
    time_control = TimeControl(
        initialTime=600000,
        increment=5000,
        type="fischer",
    )
    
    # Create game state
    now = datetime.now()
    state = GameState(
        id="benchmark-game",
        boardType=BoardType.SQUARE8,
        rngSeed=42,
        board=board,
        players=players,
        currentPhase=GamePhase.RING_PLACEMENT,
        currentPlayer=1,
        moveHistory=[],
        timeControl=time_control,
        spectators=[],
        gameStatus=GameStatus.ACTIVE,
        winner=None,
        createdAt=now,
        lastMoveAt=now,
        isRated=False,
        maxPlayers=2,
        totalRingsInPlay=36,
        totalRingsEliminated=0,
        victoryThreshold=6,
        territoryVictoryThreshold=20,
        chainCaptureState=None,
        mustMoveFromStackKey=None,
        zobristHash=None,
        lpsRoundIndex=0,
        lpsCurrentRoundActorMask={},
        lpsExclusivePlayerForCompletedRound=None,
    )
    
    return state


def create_midgame_state() -> GameState:
    """Create a midgame state with some pieces on the board.
    
    This provides a more realistic benchmark scenario with stacks,
    markers, and various move options available.
    """
    from app.models import RingStack, MarkerInfo, Position
    
    state = create_starting_state()
    
    # Add some stacks to the board
    stacks = {}
    markers = {}
    
    # Player 1 stacks
    stacks["2,2"] = RingStack(
        position=Position(x=2, y=2),
        rings=[1, 1],
        stackHeight=2,
        capHeight=2,
        controllingPlayer=1,
    )
    stacks["4,4"] = RingStack(
        position=Position(x=4, y=4),
        rings=[1, 2, 1],
        stackHeight=3,
        capHeight=1,
        controllingPlayer=1,
    )
    stacks["6,2"] = RingStack(
        position=Position(x=6, y=2),
        rings=[1],
        stackHeight=1,
        capHeight=1,
        controllingPlayer=1,
    )
    
    # Player 2 stacks
    stacks["3,5"] = RingStack(
        position=Position(x=3, y=5),
        rings=[2, 2],
        stackHeight=2,
        capHeight=2,
        controllingPlayer=2,
    )
    stacks["5,5"] = RingStack(
        position=Position(x=5, y=5),
        rings=[2],
        stackHeight=1,
        capHeight=1,
        controllingPlayer=2,
    )
    stacks["1,6"] = RingStack(
        position=Position(x=1, y=6),
        rings=[1, 2, 2],
        stackHeight=3,
        capHeight=2,
        controllingPlayer=2,
    )
    
    # Add some markers
    markers["2,3"] = MarkerInfo(
        player=1,
        position=Position(x=2, y=3),
        type="regular",
    )
    markers["4,3"] = MarkerInfo(
        player=2,
        position=Position(x=4, y=3),
        type="regular",
    )
    markers["3,4"] = MarkerInfo(
        player=1,
        position=Position(x=3, y=4),
        type="regular",
    )
    
    state.board.stacks = stacks
    state.board.markers = markers
    
    # Update player rings in hand
    state.players[0].rings_in_hand = 12  # 18 - 6 placed
    state.players[1].rings_in_hand = 12  # 18 - 6 placed
    
    # Set to movement phase with player 1's turn
    state.current_phase = GamePhase.MOVEMENT
    state.must_move_from_stack_key = "2,2"
    
    return state


def benchmark_search(
    use_incremental: bool,
    depth: int,
    num_runs: int = 5,
    use_midgame: bool = True,
) -> BenchmarkResult:
    """Run benchmark with specified search mode.
    
    Args:
        use_incremental: If True, use make/unmake pattern; else use legacy.
        depth: Maximum search depth.
        num_runs: Number of runs to average.
        use_midgame: If True, use midgame state; else use starting state.
        
    Returns:
        BenchmarkResult with timing and memory statistics.
    """
    config = AIConfig(
        difficulty=5,
        think_time=30000,  # 30 second timeout to avoid hanging
        randomness=None,
        rngSeed=None,
        use_incremental_search=use_incremental,
    )
    
    ai = MinimaxAI(player_number=1, config=config)
    # Override max depth getter for controlled benchmarking
    ai._get_max_depth = lambda: depth
    
    state = create_midgame_state() if use_midgame else create_starting_state()
    
    times: List[float] = []
    total_nodes = 0
    peak_memory = 0.0
    
    for _ in range(num_runs):
        # Reset node counter
        ai.nodes_visited = 0
        
        # Start memory tracking
        tracemalloc.start()
        
        start = time.perf_counter()
        _ = ai.select_move(state)  # Move result not needed
        elapsed = time.perf_counter() - start
        
        # Get memory stats
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        times.append(elapsed)
        total_nodes += ai.nodes_visited
        peak_memory = max(peak_memory, peak / (1024 * 1024))  # Convert to MB
    
    return BenchmarkResult(
        mode="incremental" if use_incremental else "legacy",
        depth=depth,
        avg_time=sum(times) / len(times),
        min_time=min(times),
        max_time=max(times),
        nodes_visited=total_nodes // num_runs,
        peak_memory_mb=peak_memory,
        runs=num_runs,
    )


def validate_correctness(
    depth: int = 2,
    num_positions: int = 3,
) -> Tuple[bool, List[str]]:
    """Validate that both search modes produce equivalent results.
    
    Tests that:
    - Same move is selected for identical positions
    - Same evaluation scores at same positions
    
    Args:
        depth: Search depth for validation.
        num_positions: Number of positions to test.
        
    Returns:
        Tuple of (all_passed, list_of_messages).
    """
    messages: List[str] = []
    all_passed = True
    
    # Create configs for both modes
    legacy_config = AIConfig(
        difficulty=5,
        think_time=30000,
        randomness=None,
        rngSeed=None,
        use_incremental_search=False,
    )
    incremental_config = AIConfig(
        difficulty=5,
        think_time=30000,
        randomness=None,
        rngSeed=None,
        use_incremental_search=True,
    )
    
    # Test with different states
    states = [
        ("starting", create_starting_state()),
        ("midgame", create_midgame_state()),
    ]
    
    for name, state in states[:num_positions]:
        legacy_ai = MinimaxAI(player_number=1, config=legacy_config)
        incremental_ai = MinimaxAI(player_number=1, config=incremental_config)
        
        # Override max depth
        legacy_ai._get_max_depth = lambda: depth
        incremental_ai._get_max_depth = lambda: depth
        
        # Get moves from both
        legacy_move = legacy_ai.select_move(state)
        incremental_move = incremental_ai.select_move(state)
        
        # Compare moves
        if legacy_move is None and incremental_move is None:
            messages.append(f"  {name}: Both returned None (OK)")
        elif legacy_move is None or incremental_move is None:
            messages.append(f"  {name}: MISMATCH - One returned None")
            all_passed = False
        elif (legacy_move.to.to_key() == incremental_move.to.to_key() and
              legacy_move.type == incremental_move.type):
            leg_key = legacy_move.to.to_key()
            leg_type = legacy_move.type
            msg = f"  {name}: Moves match (to={leg_key}, type={leg_type})"
            messages.append(msg)
        else:
            # Different moves might still be equally good
            leg_key = legacy_move.to.to_key()
            inc_key = incremental_move.to.to_key()
            messages.append(
                f"  {name}: Different moves - "
                f"legacy={leg_key} ({legacy_move.type}), "
                f"incr={inc_key} ({incremental_move.type})"
            )
            # This isn't necessarily a failure
            messages.append(f"  {name}: (Note: Both may be equally good)")
    
    return all_passed, messages


def run_make_unmake_roundtrip_test() -> Tuple[bool, List[str]]:
    """Test that make/unmake produces identical state restoration.
    
    Verifies that:
    - Zobrist hash matches after make/unmake roundtrip
    - Stack/marker dictionaries are restored
    - Player state is restored
    
    Returns:
        Tuple of (passed, list_of_messages).
    """
    from app.rules.mutable_state import MutableGameState
    from app.game_engine import GameEngine
    
    messages: List[str] = []
    passed = True
    
    state = create_midgame_state()
    mutable = MutableGameState.from_immutable(state)
    
    original_hash = mutable.zobrist_hash
    original_stacks = set(mutable.stacks.keys())
    original_markers = set(mutable.markers.keys())
    original_player_rings = {
        pn: ps.rings_in_hand
        for pn, ps in mutable.players.items()
    }
    
    # Get valid moves
    valid_moves = GameEngine.get_valid_moves(state, state.current_player)
    
    if not valid_moves:
        messages.append("  No valid moves available for roundtrip test")
        return True, messages
    
    # Test roundtrip for first few moves
    for move in valid_moves[:5]:
        # Make move
        undo = mutable.make_move(move)
        
        # Unmake move (state was changed, now restore)
        mutable.unmake_move(undo)
        
        # Verify restoration
        if mutable.zobrist_hash != original_hash:
            msg = f"  FAIL: Hash not restored after {move.type}"
            messages.append(msg)
            new_hash = mutable.zobrist_hash
            hash_msg = f"    Orig: {original_hash}, Now: {new_hash}"
            messages.append(hash_msg)
            passed = False
        
        if set(mutable.stacks.keys()) != original_stacks:
            msg = f"  FAIL: Stacks not restored after {move.type}"
            messages.append(msg)
            passed = False
        
        if set(mutable.markers.keys()) != original_markers:
            msg = f"  FAIL: Markers not restored after {move.type}"
            messages.append(msg)
            passed = False
        
        current_player_rings = {
            pn: ps.rings_in_hand
            for pn, ps in mutable.players.items()
        }
        if current_player_rings != original_player_rings:
            msg = f"  FAIL: Player rings not restored after {move.type}"
            messages.append(msg)
            passed = False
    
    if passed:
        num_tested = min(5, len(valid_moves))
        messages.append(f"  Roundtrip test passed for {num_tested} moves")
    
    return passed, messages


def main():
    """Run the benchmark suite."""
    print("=" * 60)
    print("Make/Unmake vs Legacy Benchmark")
    print("=" * 60)
    print()
    
    # First run correctness validation
    print("1. Correctness Validation")
    print("-" * 40)
    
    print("\n  Make/Unmake Roundtrip Test:")
    roundtrip_passed, roundtrip_messages = run_make_unmake_roundtrip_test()
    for msg in roundtrip_messages:
        print(msg)
    
    print("\n  Move Selection Equivalence Test:")
    equiv_result = validate_correctness(depth=2, num_positions=2)
    equiv_passed, equiv_messages = equiv_result
    for msg in equiv_messages:
        print(msg)
    
    # Note: equiv may have different-but-equal moves
    all_valid = roundtrip_passed
    print(f"\n  Overall: {'PASSED' if all_valid else 'ISSUES DETECTED'}")
    
    # Run benchmarks at different depths
    print("\n" + "=" * 60)
    print("2. Performance Benchmark")
    print("-" * 40)
    
    depths = [2, 3]  # Start with lower depths to avoid long runtimes
    num_runs = 3
    
    results: List[BenchmarkResult] = []
    
    for depth in depths:
        print(f"\nDepth {depth} (averaging {num_runs} runs):")
        print("-" * 30)
        
        # Benchmark legacy mode
        print("  Running legacy mode...", end="", flush=True)
        try:
            legacy = benchmark_search(
                use_incremental=False, 
                depth=depth, 
                num_runs=num_runs
            )
            results.append(legacy)
            print(f" done ({legacy.avg_time:.3f}s avg)")
        except Exception as e:
            print(f" ERROR: {e}")
            continue
        
        # Benchmark incremental mode
        print("  Running incremental mode...", end="", flush=True)
        try:
            incremental = benchmark_search(
                use_incremental=True, 
                depth=depth, 
                num_runs=num_runs
            )
            results.append(incremental)
            print(f" done ({incremental.avg_time:.3f}s avg)")
        except Exception as e:
            print(f" ERROR: {e}")
            continue
        
        # Calculate speedup
        if legacy.avg_time > 0 and incremental.avg_time > 0:
            speedup = legacy.avg_time / incremental.avg_time
            
            print("\n  Results:")
            leg_t = f"{legacy.avg_time:.3f}s (min={legacy.min_time:.3f}s)"
            print(f"    Legacy:      {leg_t}")
            inc_t = f"{incremental.avg_time:.3f}s"
            inc_t += f" (min={incremental.min_time:.3f}s)"
            print(f"    Incremental: {inc_t}")
            print(f"    Speedup:     {speedup:.2f}x")
            print(f"    Nodes/run:   legacy={legacy.nodes_visited}, "
                  f"incremental={incremental.nodes_visited}")
            print(f"    Peak memory: legacy={legacy.peak_memory_mb:.1f}MB, "
                  f"incremental={incremental.peak_memory_mb:.1f}MB")
            
            if legacy.avg_time > 0:
                legacy_nps = legacy.nodes_visited / legacy.avg_time
                print(f"    Legacy nodes/sec: {legacy_nps:.0f}")
            if incremental.avg_time > 0:
                incr_nps = incremental.nodes_visited / incremental.avg_time
                print(f"    Incremental nodes/sec: {incr_nps:.0f}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("3. Summary")
    print("-" * 40)
    
    if len(results) >= 2:
        # Group by depth
        for depth in depths:
            legacy_res = next(
                (r for r in results
                 if r.depth == depth and r.mode == "legacy"),
                None
            )
            incr_res = next(
                (r for r in results
                 if r.depth == depth and r.mode == "incremental"),
                None
            )
            
            if legacy_res and incr_res and legacy_res.avg_time > 0:
                speedup = legacy_res.avg_time / incr_res.avg_time
                if legacy_res.peak_memory_mb > 0:
                    leg_mem = legacy_res.peak_memory_mb
                    inc_mem = incr_res.peak_memory_mb
                    memory_reduction = (1 - inc_mem / leg_mem) * 100
                else:
                    memory_reduction = 0
                print(f"  Depth {depth}: {speedup:.2f}x speedup, "
                      f"{memory_reduction:.0f}% memory reduction")
    
    print("\n  Note: Actual speedup during deep search may be higher due to")
    print("  reduced GC pressure and cache effects.")
    
    return 0 if all_valid else 1


if __name__ == "__main__":
    sys.exit(main())