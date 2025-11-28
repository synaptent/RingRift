"""
Tests for MutableGameState and MoveUndo classes.

This module tests the make/unmake move pattern implementation:
- from_immutable / to_immutable round-trip
- Simple make/unmake for placement moves
- Make/unmake preserves state equality
"""

import pytest
from datetime import datetime

from app.models import (
    GameState,
    BoardState,
    BoardType,
    GamePhase,
    GameStatus,
    Move,
    MoveType,
    Position,
    RingStack,
    MarkerInfo,
    Player,
    TimeControl,
    LineInfo,
    Territory,
)
from app.rules.mutable_state import (
    MutableGameState,
    MutableStack,
    MutablePlayerState,
    MoveUndo,
)


# =============================================================================
# Test Fixtures
# =============================================================================


def create_minimal_game_state() -> GameState:
    """Create a minimal valid GameState for testing."""
    now = datetime.now()
    
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
    
    players = [
        Player(
            id="p1",
            username="Player1",
            type="human",
            playerNumber=1,
            isReady=True,
            timeRemaining=600000,
            aiDifficulty=None,
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
    
    time_control = TimeControl(
        initialTime=600000,
        increment=5000,
        type="fischer",
    )
    
    return GameState(
        id="test-game-1",
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


def create_game_state_with_stacks() -> GameState:
    """Create a GameState with some stacks for movement testing."""
    state = create_minimal_game_state()
    
    # Add a stack at position 2,2 controlled by player 1
    pos = Position(x=2, y=2)
    stack = RingStack(
        position=pos,
        rings=[1, 1, 1],
        stackHeight=3,
        capHeight=3,
        controllingPlayer=1,
    )
    state.board.stacks[pos.to_key()] = stack
    
    # Add a marker at position 3,3 for player 2
    marker_pos = Position(x=3, y=3)
    marker = MarkerInfo(
        player=2,
        position=marker_pos,
        type="regular",
    )
    state.board.markers[marker_pos.to_key()] = marker
    
    # Reduce player 1 rings in hand
    state.players[0].rings_in_hand = 15
    
    return state


def create_place_ring_move(
    player: int,
    to_pos: Position,
    count: int = 1
) -> Move:
    """Create a PLACE_RING move."""
    # Use model_construct to bypass validation for test moves
    return Move.model_construct(
        id="test-move",
        type=MoveType.PLACE_RING,
        player=player,
        from_pos=None,
        to=to_pos,
        capture_target=None,
        captured_stacks=None,
        capture_chain=None,
        overtaken_rings=None,
        placed_on_stack=False,
        placement_count=count,
        stack_moved=None,
        minimum_distance=None,
        actual_distance=None,
        marker_left=None,
        formed_lines=None,
        collapsed_markers=None,
        claimed_territory=None,
        disconnected_regions=None,
        eliminated_rings=None,
        timestamp=datetime.now(),
        think_time=0,
        move_number=1,
    )


def create_move_stack_move(
    player: int,
    from_pos: Position,
    to_pos: Position,
) -> Move:
    """Create a MOVE_STACK move."""
    # Use model_construct to bypass validation for test moves
    return Move.model_construct(
        id="test-move",
        type=MoveType.MOVE_STACK,
        player=player,
        from_pos=from_pos,
        to=to_pos,
        capture_target=None,
        captured_stacks=None,
        capture_chain=None,
        overtaken_rings=None,
        placed_on_stack=None,
        placement_count=None,
        stack_moved=None,
        minimum_distance=None,
        actual_distance=None,
        marker_left=None,
        formed_lines=None,
        collapsed_markers=None,
        claimed_territory=None,
        disconnected_regions=None,
        eliminated_rings=None,
        timestamp=datetime.now(),
        think_time=0,
        move_number=1,
    )


def create_capture_move(
    player: int,
    from_pos: Position,
    target_pos: Position,
    to_pos: Position,
) -> Move:
    """Create a CHAIN_CAPTURE move."""
    return Move.model_construct(
        id="test-move",
        type=MoveType.CHAIN_CAPTURE,
        player=player,
        from_pos=from_pos,
        to=to_pos,
        capture_target=target_pos,
        captured_stacks=None,
        capture_chain=None,
        overtaken_rings=None,
        placed_on_stack=None,
        placement_count=None,
        stack_moved=None,
        minimum_distance=None,
        actual_distance=None,
        marker_left=None,
        formed_lines=None,
        collapsed_markers=None,
        claimed_territory=None,
        disconnected_regions=None,
        eliminated_rings=None,
        timestamp=datetime.now(),
        think_time=0,
        move_number=1,
    )


def create_eliminate_move(
    player: int,
    to_pos: Position,
) -> Move:
    """Create an ELIMINATE_RINGS_FROM_STACK move."""
    return Move.model_construct(
        id="test-move",
        type=MoveType.ELIMINATE_RINGS_FROM_STACK,
        player=player,
        from_pos=None,
        to=to_pos,
        capture_target=None,
        captured_stacks=None,
        capture_chain=None,
        overtaken_rings=None,
        placed_on_stack=None,
        placement_count=None,
        stack_moved=None,
        minimum_distance=None,
        actual_distance=None,
        marker_left=None,
        formed_lines=None,
        collapsed_markers=None,
        claimed_territory=None,
        disconnected_regions=None,
        eliminated_rings=None,
        timestamp=datetime.now(),
        think_time=0,
        move_number=1,
    )


def create_process_line_move(
    player: int,
    to_pos: Position,
    line_positions: list,
) -> Move:
    """Create a PROCESS_LINE move."""
    line_info = LineInfo(
        positions=line_positions,
        player=player,
        length=len(line_positions),
        direction=Position(x=1, y=0),
    )
    return Move.model_construct(
        id="test-move",
        type=MoveType.PROCESS_LINE,
        player=player,
        from_pos=None,
        to=to_pos,
        capture_target=None,
        captured_stacks=None,
        capture_chain=None,
        overtaken_rings=None,
        placed_on_stack=None,
        placement_count=None,
        stack_moved=None,
        minimum_distance=None,
        actual_distance=None,
        marker_left=None,
        formed_lines=(line_info,),
        collapsed_markers=None,
        claimed_territory=None,
        disconnected_regions=None,
        eliminated_rings=None,
        timestamp=datetime.now(),
        think_time=0,
        move_number=1,
    )


def create_process_territory_move(
    player: int,
    to_pos: Position,
    region_spaces: list,
) -> Move:
    """Create a PROCESS_TERRITORY_REGION move."""
    territory = Territory(
        spaces=region_spaces,
        controllingPlayer=0,
        isDisconnected=True,
    )
    return Move.model_construct(
        id="test-move",
        type=MoveType.PROCESS_TERRITORY_REGION,
        player=player,
        from_pos=None,
        to=to_pos,
        capture_target=None,
        captured_stacks=None,
        capture_chain=None,
        overtaken_rings=None,
        placed_on_stack=None,
        placement_count=None,
        stack_moved=None,
        minimum_distance=None,
        actual_distance=None,
        marker_left=None,
        formed_lines=None,
        collapsed_markers=None,
        claimed_territory=None,
        disconnected_regions=(territory,),
        eliminated_rings=None,
        timestamp=datetime.now(),
        think_time=0,
        move_number=1,
    )


def create_game_state_with_capture_setup() -> GameState:
    """Create a GameState with stacks set up for capture testing."""
    state = create_minimal_game_state()
    
    # Add attacker stack at (2,2) controlled by player 1
    attacker_pos = Position(x=2, y=2)
    attacker = RingStack(
        position=attacker_pos,
        rings=[1, 1],
        stackHeight=2,
        capHeight=2,
        controllingPlayer=1,
    )
    state.board.stacks[attacker_pos.to_key()] = attacker
    
    # Add target stack at (3,3) controlled by player 2
    target_pos = Position(x=3, y=3)
    target = RingStack(
        position=target_pos,
        rings=[2],
        stackHeight=1,
        capHeight=1,
        controllingPlayer=2,
    )
    state.board.stacks[target_pos.to_key()] = target
    
    state.players[0].rings_in_hand = 16
    
    return state


def create_game_state_with_markers_for_line() -> GameState:
    """Create a GameState with markers forming a line."""
    state = create_minimal_game_state()
    
    # Add markers in a horizontal line at y=3
    for x in range(3, 6):
        pos = Position(x=x, y=3)
        marker = MarkerInfo(
            player=1,
            position=pos,
            type="regular",
        )
        state.board.markers[pos.to_key()] = marker
    
    return state


# =============================================================================
# MutableStack Tests
# =============================================================================


class TestMutableStack:
    """Tests for MutableStack helper class."""
    
    def test_from_ring_stack(self):
        """Test creating MutableStack from RingStack."""
        pos = Position(x=1, y=2)
        ring_stack = RingStack(
            position=pos,
            rings=[1, 2, 1],
            stackHeight=3,
            capHeight=1,
            controllingPlayer=1,
        )
        
        mutable = MutableStack.from_ring_stack(ring_stack)
        
        assert mutable.position == pos
        assert mutable.rings == [1, 2, 1]
        assert mutable.stack_height == 3
        assert mutable.cap_height == 1
        assert mutable.controlling_player == 1
    
    def test_to_ring_stack(self):
        """Test converting MutableStack back to RingStack."""
        pos = Position(x=3, y=4)
        mutable = MutableStack(
            position=pos,
            rings=[2, 2],
            stack_height=2,
            cap_height=2,
            controlling_player=2,
        )
        
        ring_stack = mutable.to_ring_stack()
        
        assert ring_stack.position == pos
        assert ring_stack.rings == [2, 2]
        assert ring_stack.stack_height == 2
        assert ring_stack.cap_height == 2
        assert ring_stack.controlling_player == 2
    
    def test_copy(self):
        """Test copying MutableStack."""
        pos = Position(x=0, y=0)
        original = MutableStack(
            position=pos,
            rings=[1, 1],
            stack_height=2,
            cap_height=2,
            controlling_player=1,
        )
        
        copy = original.copy()
        
        # Modify original
        original.rings.append(2)
        original.stack_height = 3
        
        # Copy should be unaffected
        assert copy.rings == [1, 1]
        assert copy.stack_height == 2
    
    def test_recompute_properties(self):
        """Test recomputing stack properties."""
        pos = Position(x=0, y=0)
        stack = MutableStack(
            position=pos,
            rings=[1, 2, 2, 2],
            stack_height=0,  # Incorrect
            cap_height=0,  # Incorrect
            controlling_player=0,  # Incorrect
        )
        
        stack.recompute_properties()
        
        assert stack.stack_height == 4
        assert stack.controlling_player == 2
        assert stack.cap_height == 3  # Three 2's on top


class TestMutablePlayerState:
    """Tests for MutablePlayerState helper class."""
    
    def test_from_player(self):
        """Test creating MutablePlayerState from Player."""
        player = Player(
            id="test-player",
            username="Tester",
            type="human",
            playerNumber=1,
            isReady=True,
            timeRemaining=500000,
            aiDifficulty=None,
            ringsInHand=15,
            eliminatedRings=3,
            territorySpaces=5,
        )
        
        mutable = MutablePlayerState.from_player(player)
        
        assert mutable.player_number == 1
        assert mutable.rings_in_hand == 15
        assert mutable.eliminated_rings == 3
        assert mutable.territory_spaces == 5
        assert mutable.id == "test-player"
    
    def test_to_player(self):
        """Test converting back to Player."""
        mutable = MutablePlayerState(
            player_number=2,
            rings_in_hand=10,
            eliminated_rings=8,
            territory_spaces=0,
            id="p2",
            username="Bot",
            player_type="ai",
        )
        
        player = mutable.to_player()
        
        assert player.player_number == 2
        assert player.rings_in_hand == 10
        assert player.eliminated_rings == 8
        assert player.id == "p2"


# =============================================================================
# MutableGameState Tests
# =============================================================================


class TestMutableGameState:
    """Tests for MutableGameState class."""
    
    def test_from_immutable_empty_board(self):
        """Test creating MutableGameState from empty GameState."""
        state = create_minimal_game_state()
        
        mutable = MutableGameState.from_immutable(state)
        
        assert mutable.current_player == 1
        assert mutable.current_phase == GamePhase.RING_PLACEMENT
        assert mutable.board_type == BoardType.SQUARE8
        assert len(mutable.stacks) == 0
        assert len(mutable.markers) == 0
        assert len(mutable.players) == 2
        assert mutable.players[1].rings_in_hand == 18
        assert mutable.players[2].rings_in_hand == 18
    
    def test_from_immutable_with_stacks(self):
        """Test creating MutableGameState with existing stacks."""
        state = create_game_state_with_stacks()
        
        mutable = MutableGameState.from_immutable(state)
        
        assert len(mutable.stacks) == 1
        assert len(mutable.markers) == 1
        
        stack = mutable.stacks["2,2"]
        assert stack.controlling_player == 1
        assert stack.stack_height == 3
        
        marker = mutable.markers["3,3"]
        assert marker.player == 2
    
    def test_to_immutable_round_trip(self):
        """Test from_immutable -> to_immutable preserves state."""
        original = create_game_state_with_stacks()
        
        mutable = MutableGameState.from_immutable(original)
        restored = mutable.to_immutable()
        
        # Compare key fields
        assert restored.current_player == original.current_player
        assert restored.current_phase == original.current_phase
        assert restored.board_type == original.board_type
        assert len(restored.board.stacks) == len(original.board.stacks)
        assert len(restored.board.markers) == len(original.board.markers)
        assert len(restored.players) == len(original.players)
        
        # Compare stacks
        for key, orig_stack in original.board.stacks.items():
            rest_stack = restored.board.stacks[key]
            assert rest_stack.rings == orig_stack.rings
            assert rest_stack.controlling_player == (
                orig_stack.controlling_player
            )
        
        # Compare players
        for i, orig_player in enumerate(original.players):
            rest_player = restored.players[i]
            assert rest_player.rings_in_hand == orig_player.rings_in_hand
            assert rest_player.player_number == orig_player.player_number


class TestMakeUnmakePlacement:
    """Tests for make/unmake of placement moves."""
    
    def test_make_place_ring_new_stack(self):
        """Test placing a ring on empty space."""
        state = create_minimal_game_state()
        mutable = MutableGameState.from_immutable(state)
        
        pos = Position(x=3, y=3)
        move = create_place_ring_move(player=1, to_pos=pos, count=1)
        
        _ = mutable.make_move(move)
        
        # Verify stack was created
        assert "3,3" in mutable.stacks
        stack = mutable.stacks["3,3"]
        assert stack.rings == [1]
        assert stack.controlling_player == 1
        assert stack.stack_height == 1
        
        # Verify player rings decreased
        assert mutable.players[1].rings_in_hand == 17
        
        # Verify phase changed
        assert mutable.current_phase == GamePhase.MOVEMENT
    
    def test_make_place_ring_multi_ring(self):
        """Test placing multiple rings at once."""
        state = create_minimal_game_state()
        mutable = MutableGameState.from_immutable(state)
        
        pos = Position(x=4, y=4)
        move = create_place_ring_move(player=1, to_pos=pos, count=3)
        
        _ = mutable.make_move(move)
        
        stack = mutable.stacks["4,4"]
        assert stack.rings == [1, 1, 1]
        assert stack.stack_height == 3
        assert stack.cap_height == 3
        assert mutable.players[1].rings_in_hand == 15
    
    def test_make_place_ring_on_existing_stack(self):
        """Test placing ring on existing stack."""
        state = create_game_state_with_stacks()
        mutable = MutableGameState.from_immutable(state)
        
        pos = Position(x=2, y=2)  # Existing stack position
        move = create_place_ring_move(player=1, to_pos=pos, count=1)
        
        original_height = mutable.stacks["2,2"].stack_height
        _ = mutable.make_move(move)
        
        stack = mutable.stacks["2,2"]
        assert stack.stack_height == original_height + 1
        assert stack.rings[-1] == 1  # New ring on top
    
    def test_unmake_place_ring_new_stack(self):
        """Test unmake removes newly created stack."""
        state = create_minimal_game_state()
        mutable = MutableGameState.from_immutable(state)
        
        pos = Position(x=5, y=5)
        move = create_place_ring_move(player=1, to_pos=pos, count=2)
        
        original_hash = mutable.zobrist_hash
        original_rings = mutable.players[1].rings_in_hand
        
        undo = mutable.make_move(move)
        
        # Verify changes
        assert "5,5" in mutable.stacks
        assert mutable.players[1].rings_in_hand == original_rings - 2
        
        # Unmake
        mutable.unmake_move(undo)
        
        # Verify restored
        assert "5,5" not in mutable.stacks
        assert mutable.players[1].rings_in_hand == original_rings
        assert mutable.zobrist_hash == original_hash
        assert mutable.current_phase == GamePhase.RING_PLACEMENT
    
    def test_unmake_place_ring_on_existing(self):
        """Test unmake restores original stack state."""
        state = create_game_state_with_stacks()
        mutable = MutableGameState.from_immutable(state)
        
        pos = Position(x=2, y=2)
        original_stack = mutable.stacks["2,2"].copy()
        original_hash = mutable.zobrist_hash
        
        move = create_place_ring_move(player=1, to_pos=pos, count=1)
        undo = mutable.make_move(move)
        
        # Verify change
        assert mutable.stacks["2,2"].stack_height == 4
        
        # Unmake
        mutable.unmake_move(undo)
        
        # Verify restored
        stack = mutable.stacks["2,2"]
        assert stack.stack_height == original_stack.stack_height
        assert stack.rings == original_stack.rings
        assert mutable.zobrist_hash == original_hash


class TestMakeUnmakeMovement:
    """Tests for make/unmake of movement moves."""
    
    def test_make_move_stack_simple(self):
        """Test simple stack movement."""
        state = create_game_state_with_stacks()
        mutable = MutableGameState.from_immutable(state)
        
        from_pos = Position(x=2, y=2)
        to_pos = Position(x=5, y=2)  # Move 3 spaces right
        
        original_stack_rings = list(mutable.stacks["2,2"].rings)
        
        move = create_move_stack_move(
            player=1, from_pos=from_pos, to_pos=to_pos
        )
        _ = mutable.make_move(move)
        
        # Stack moved
        assert "2,2" not in mutable.stacks
        assert "5,2" in mutable.stacks
        
        # Stack preserved
        assert mutable.stacks["5,2"].rings == original_stack_rings
        
        # Marker left behind
        assert "2,2" in mutable.markers
        assert mutable.markers["2,2"].player == 1
    
    def test_unmake_move_stack_simple(self):
        """Test unmake restores stack to original position."""
        state = create_game_state_with_stacks()
        mutable = MutableGameState.from_immutable(state)
        
        from_pos = Position(x=2, y=2)
        to_pos = Position(x=5, y=2)
        
        original_hash = mutable.zobrist_hash
        
        move = create_move_stack_move(
            player=1, from_pos=from_pos, to_pos=to_pos
        )
        undo = mutable.make_move(move)
        
        # Unmake
        mutable.unmake_move(undo)
        
        # Stack restored
        assert "2,2" in mutable.stacks
        assert "5,2" not in mutable.stacks
        
        # Marker removed
        assert "2,2" not in mutable.markers
        
        # Hash restored
        assert mutable.zobrist_hash == original_hash


class TestZobristHashConsistency:
    """Tests for Zobrist hash consistency during make/unmake."""
    
    def test_hash_round_trip_placement(self):
        """Verify hash is consistent after make/unmake placement."""
        state = create_minimal_game_state()
        mutable = MutableGameState.from_immutable(state)
        
        original_hash = mutable.zobrist_hash
        
        # Make multiple moves
        moves_and_undos = []
        for i in range(5):
            pos = Position(x=i, y=i)
            move = create_place_ring_move(player=1, to_pos=pos, count=1)
            undo = mutable.make_move(move)
            moves_and_undos.append(undo)
        
        # Unmake all in reverse order
        for undo in reversed(moves_and_undos):
            mutable.unmake_move(undo)
        
        assert mutable.zobrist_hash == original_hash
    
    def test_hash_changes_on_placement(self):
        """Verify hash changes when state changes."""
        state = create_minimal_game_state()
        mutable = MutableGameState.from_immutable(state)
        
        hash1 = mutable.zobrist_hash
        
        move = create_place_ring_move(
            player=1, to_pos=Position(x=1, y=1), count=1
        )
        mutable.make_move(move)
        
        hash2 = mutable.zobrist_hash
        
        # Hash should be different after move
        assert hash1 != hash2


class TestMoveUndoCapture:
    """Tests for MoveUndo capturing state correctly."""
    
    def test_undo_captures_player_state(self):
        """Test that undo captures player ring counts."""
        state = create_minimal_game_state()
        mutable = MutableGameState.from_immutable(state)
        
        move = create_place_ring_move(
            player=1, to_pos=Position(x=0, y=0), count=2
        )
        undo = mutable.make_move(move)
        
        assert 1 in undo.prev_rings_in_hand
        assert undo.prev_rings_in_hand[1] == 18  # Original value
    
    def test_undo_captures_phase(self):
        """Test that undo captures phase state."""
        state = create_minimal_game_state()
        mutable = MutableGameState.from_immutable(state)
        
        move = create_place_ring_move(
            player=1, to_pos=Position(x=0, y=0), count=1
        )
        undo = mutable.make_move(move)
        
        assert undo.prev_phase == GamePhase.RING_PLACEMENT
        assert mutable.current_phase == GamePhase.MOVEMENT


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for complex scenarios."""
    
    def test_multiple_placements_and_unmakes(self):
        """Test sequence of placements followed by unmakes."""
        state = create_minimal_game_state()
        mutable = MutableGameState.from_immutable(state)
        
        undos = []
        
        # Player 1 places
        pos1 = Position(x=0, y=0)
        move1 = create_place_ring_move(player=1, to_pos=pos1, count=1)
        undos.append(mutable.make_move(move1))
        
        # Manually switch player for testing
        mutable._active_player = 2
        mutable._phase = GamePhase.RING_PLACEMENT
        
        # Player 2 places
        pos2 = Position(x=7, y=7)
        move2 = create_place_ring_move(player=2, to_pos=pos2, count=1)
        undos.append(mutable.make_move(move2))
        
        # Verify both stacks exist
        assert "0,0" in mutable.stacks
        assert "7,7" in mutable.stacks
        
        # Unmake in reverse
        mutable.unmake_move(undos.pop())
        assert "7,7" not in mutable.stacks
        assert "0,0" in mutable.stacks
        
        mutable.unmake_move(undos.pop())
        assert "0,0" not in mutable.stacks
        assert len(mutable.stacks) == 0
    
    def test_state_equality_after_round_trip(self):
        """Verify state is identical after make/unmake round trip."""
        state = create_game_state_with_stacks()
        mutable = MutableGameState.from_immutable(state)
        
        # Capture full state snapshot
        orig_stacks = {k: v.copy() for k, v in mutable.stacks.items()}
        orig_markers = {k: v.copy() for k, v in mutable.markers.items()}
        orig_collapsed = dict(mutable.collapsed_spaces)
        orig_player_rings = {
            pn: p.rings_in_hand for pn, p in mutable.players.items()
        }
        orig_hash = mutable.zobrist_hash
        orig_phase = mutable.current_phase
        orig_player = mutable.current_player
        
        # Make move
        move = create_place_ring_move(
            player=1, to_pos=Position(x=4, y=4), count=2
        )
        undo = mutable.make_move(move)
        
        # Unmake
        mutable.unmake_move(undo)
        
        # Verify all state restored
        assert len(mutable.stacks) == len(orig_stacks)
        for k, v in orig_stacks.items():
            assert k in mutable.stacks
            assert mutable.stacks[k].rings == v.rings
        
        assert len(mutable.markers) == len(orig_markers)
        assert mutable.collapsed_spaces == orig_collapsed
        
        for pn, rings in orig_player_rings.items():
            assert mutable.players[pn].rings_in_hand == rings
        
        assert mutable.zobrist_hash == orig_hash
        assert mutable.current_phase == orig_phase
        assert mutable.current_player == orig_player


class TestMakeUnmakeCapture:
    """Tests for make/unmake of capture moves."""
    
    def test_make_capture_basic(self):
        """Test basic capture operation."""
        state = create_game_state_with_capture_setup()
        mutable = MutableGameState.from_immutable(state)
        
        # Set up positions
        from_pos = Position(x=2, y=2)
        target_pos = Position(x=3, y=3)
        landing_pos = Position(x=4, y=4)
        
        move = create_capture_move(
            player=1,
            from_pos=from_pos,
            target_pos=target_pos,
            to_pos=landing_pos,
        )
        
        _ = mutable.make_move(move)
        
        # Attacker should be removed from source
        assert "2,2" not in mutable.stacks
        
        # Attacker should be at landing
        assert "4,4" in mutable.stacks
        landing_stack = mutable.stacks["4,4"]
        # Original attacker had [1,1], captured 2, so now [2,1,1]
        assert landing_stack.stack_height == 3
        assert landing_stack.rings[0] == 2  # Captured ring at bottom
        assert landing_stack.rings[-1] == 1  # Controlling player on top
        
        # Marker left at departure
        assert "2,2" in mutable.markers
        assert mutable.markers["2,2"].player == 1
        
        # Target eliminated (was height 1)
        assert "3,3" not in mutable.stacks
    
    def test_unmake_capture_restores_state(self):
        """Test that unmake restores all capture changes."""
        state = create_game_state_with_capture_setup()
        mutable = MutableGameState.from_immutable(state)
        
        original_hash = mutable.zobrist_hash
        original_stacks = {k: v.copy() for k, v in mutable.stacks.items()}
        
        from_pos = Position(x=2, y=2)
        target_pos = Position(x=3, y=3)
        landing_pos = Position(x=4, y=4)
        
        move = create_capture_move(
            player=1,
            from_pos=from_pos,
            target_pos=target_pos,
            to_pos=landing_pos,
        )
        
        undo = mutable.make_move(move)
        mutable.unmake_move(undo)
        
        # Stacks should be restored
        assert "2,2" in mutable.stacks
        assert "3,3" in mutable.stacks
        assert "4,4" not in mutable.stacks
        
        # Original stack states restored
        assert mutable.stacks["2,2"].rings == original_stacks["2,2"].rings
        assert mutable.stacks["3,3"].rings == original_stacks["3,3"].rings
        
        # Marker removed
        assert "2,2" not in mutable.markers
        
        # Hash restored
        assert mutable.zobrist_hash == original_hash


class TestMakeUnmakeEliminate:
    """Tests for make/unmake of elimination moves."""
    
    def test_make_eliminate_rings(self):
        """Test eliminating rings from a stack."""
        state = create_game_state_with_stacks()
        mutable = MutableGameState.from_immutable(state)
        
        pos = Position(x=2, y=2)
        original_height = mutable.stacks["2,2"].stack_height
        original_eliminated = mutable.players[1].eliminated_rings
        
        move = create_eliminate_move(player=1, to_pos=pos)
        _ = mutable.make_move(move)
        
        # Stack should have rings eliminated (cap height was 3)
        # After eliminating 3 rings, stack is empty
        assert "2,2" not in mutable.stacks
        
        # Player eliminated rings should increase
        assert mutable.players[1].eliminated_rings == original_eliminated + 3
        
        # Total eliminated should increase
        assert mutable.total_rings_eliminated == 3
    
    def test_unmake_eliminate_restores_state(self):
        """Test that unmake restores eliminated rings."""
        state = create_game_state_with_stacks()
        mutable = MutableGameState.from_immutable(state)
        
        pos = Position(x=2, y=2)
        original_hash = mutable.zobrist_hash
        original_stack = mutable.stacks["2,2"].copy()
        
        move = create_eliminate_move(player=1, to_pos=pos)
        undo = mutable.make_move(move)
        mutable.unmake_move(undo)
        
        # Stack should be restored
        assert "2,2" in mutable.stacks
        assert mutable.stacks["2,2"].rings == original_stack.rings
        
        # Hash restored
        assert mutable.zobrist_hash == original_hash


class TestMakeUnmakeProcessLine:
    """Tests for make/unmake of line processing moves."""
    
    def test_make_process_line_collapses_markers(self):
        """Test that process line collapses markers."""
        state = create_game_state_with_markers_for_line()
        mutable = MutableGameState.from_immutable(state)
        
        # Line positions
        line_positions = [Position(x=x, y=3) for x in range(3, 6)]
        move = create_process_line_move(
            player=1,
            to_pos=line_positions[0],
            line_positions=line_positions,
        )
        
        _ = mutable.make_move(move)
        
        # Markers should be replaced with collapsed spaces
        for x in range(3, 6):
            key = f"{x},3"
            assert key not in mutable.markers
            assert key in mutable.collapsed_spaces
            assert mutable.collapsed_spaces[key] == 1
    
    def test_unmake_process_line_restores_markers(self):
        """Test that unmake restores markers."""
        state = create_game_state_with_markers_for_line()
        mutable = MutableGameState.from_immutable(state)
        
        original_hash = mutable.zobrist_hash
        original_markers = {k: v.copy() for k, v in mutable.markers.items()}
        
        line_positions = [Position(x=x, y=3) for x in range(3, 6)]
        move = create_process_line_move(
            player=1,
            to_pos=line_positions[0],
            line_positions=line_positions,
        )
        
        undo = mutable.make_move(move)
        mutable.unmake_move(undo)
        
        # Markers should be restored
        for x in range(3, 6):
            key = f"{x},3"
            assert key in mutable.markers
            assert key not in mutable.collapsed_spaces
        
        # Hash restored
        assert mutable.zobrist_hash == original_hash


class TestMakeUnmakeProcessTerritory:
    """Tests for make/unmake of territory processing moves."""
    
    def test_make_process_territory_simple(self):
        """Test basic territory processing."""
        state = create_minimal_game_state()
        mutable = MutableGameState.from_immutable(state)
        
        # Create a simple territory with a single space
        territory_pos = Position(x=4, y=4)
        region_spaces = [territory_pos]
        
        move = create_process_territory_move(
            player=1,
            to_pos=territory_pos,
            region_spaces=region_spaces,
        )
        
        _ = mutable.make_move(move)
        
        # Space should be collapsed
        assert "4,4" in mutable.collapsed_spaces
        assert mutable.collapsed_spaces["4,4"] == 1
    
    def test_unmake_process_territory_restores_state(self):
        """Test that unmake restores territory processing."""
        state = create_minimal_game_state()
        
        # Add a stack so self-elimination has something to eliminate
        stack_pos = Position(x=0, y=0)
        stack = RingStack(
            position=stack_pos,
            rings=[1, 1],
            stackHeight=2,
            capHeight=2,
            controllingPlayer=1,
        )
        state.board.stacks[stack_pos.to_key()] = stack
        
        mutable = MutableGameState.from_immutable(state)
        
        original_hash = mutable.zobrist_hash
        territory_pos = Position(x=4, y=4)
        region_spaces = [territory_pos]
        
        move = create_process_territory_move(
            player=1,
            to_pos=territory_pos,
            region_spaces=region_spaces,
        )
        
        undo = mutable.make_move(move)
        mutable.unmake_move(undo)
        
        # Collapsed space should be removed
        assert "4,4" not in mutable.collapsed_spaces
        
        # Hash restored
        assert mutable.zobrist_hash == original_hash


class TestCaptureWithSelfElimination:
    """Tests for capture moves that land on own marker."""
    
    def test_capture_landing_on_own_marker_triggers_self_elimination(self):
        """Test that landing on own marker triggers self-elimination."""
        state = create_game_state_with_capture_setup()
        
        # Add own marker at landing position
        landing_pos = Position(x=4, y=4)
        marker = MarkerInfo(
            player=1,
            position=landing_pos,
            type="regular",
        )
        state.board.markers[landing_pos.to_key()] = marker
        
        mutable = MutableGameState.from_immutable(state)
        
        from_pos = Position(x=2, y=2)
        target_pos = Position(x=3, y=3)
        
        move = create_capture_move(
            player=1,
            from_pos=from_pos,
            target_pos=target_pos,
            to_pos=landing_pos,
        )
        
        _ = mutable.make_move(move)
        
        # Marker at landing should be removed
        assert "4,4" not in mutable.markers
        
        # Stack at landing should exist but one ring eliminated
        # Original: [2,1,1] after capture, then eliminate 1, becomes [2,1]
        assert "4,4" in mutable.stacks
        stack = mutable.stacks["4,4"]
        assert stack.stack_height == 2


class TestHashConsistencyNewMoves:
    """Tests for Zobrist hash consistency with new move types."""
    
    def test_hash_round_trip_capture(self):
        """Verify hash is consistent after make/unmake capture."""
        state = create_game_state_with_capture_setup()
        mutable = MutableGameState.from_immutable(state)
        
        original_hash = mutable.zobrist_hash
        
        from_pos = Position(x=2, y=2)
        target_pos = Position(x=3, y=3)
        landing_pos = Position(x=4, y=4)
        
        move = create_capture_move(
            player=1,
            from_pos=from_pos,
            target_pos=target_pos,
            to_pos=landing_pos,
        )
        
        undo = mutable.make_move(move)
        mutable.unmake_move(undo)
        
        assert mutable.zobrist_hash == original_hash
    
    def test_hash_round_trip_eliminate(self):
        """Verify hash is consistent after make/unmake eliminate."""
        state = create_game_state_with_stacks()
        mutable = MutableGameState.from_immutable(state)
        
        original_hash = mutable.zobrist_hash
        
        move = create_eliminate_move(player=1, to_pos=Position(x=2, y=2))
        undo = mutable.make_move(move)
        mutable.unmake_move(undo)
        
        assert mutable.zobrist_hash == original_hash
    
    def test_hash_round_trip_line(self):
        """Verify hash is consistent after make/unmake line."""
        state = create_game_state_with_markers_for_line()
        mutable = MutableGameState.from_immutable(state)
        
        original_hash = mutable.zobrist_hash
        
        line_positions = [Position(x=x, y=3) for x in range(3, 6)]
        move = create_process_line_move(
            player=1,
            to_pos=line_positions[0],
            line_positions=line_positions,
        )
        
        undo = mutable.make_move(move)
        mutable.unmake_move(undo)
        
        assert mutable.zobrist_hash == original_hash


class TestPhaseTransitions:
    """Tests for phase transitions during gameplay."""
    
    def test_placement_to_movement_transition(self):
        """Test that placement moves transition to movement phase."""
        state = create_minimal_game_state()
        mutable = MutableGameState.from_immutable(state)
        
        assert mutable.current_phase == GamePhase.RING_PLACEMENT
        
        move = create_place_ring_move(
            player=1, to_pos=Position(x=3, y=3), count=1
        )
        _ = mutable.make_move(move)
        
        assert mutable.current_phase == GamePhase.MOVEMENT
    
    def test_movement_to_placement_transition(self):
        """Test that movement ends turn and returns to placement."""
        state = create_game_state_with_stacks()
        mutable = MutableGameState.from_immutable(state)
        
        # Start in placement phase
        mutable._phase = GamePhase.RING_PLACEMENT
        
        # Do placement
        move = create_place_ring_move(
            player=1, to_pos=Position(x=4, y=4), count=1
        )
        _ = mutable.make_move(move)
        
        assert mutable.current_phase == GamePhase.MOVEMENT
        
        # Now do movement from the new stack
        from_pos = Position(x=4, y=4)
        to_pos = Position(x=4, y=6)
        move2 = create_move_stack_move(
            player=1, from_pos=from_pos, to_pos=to_pos
        )
        _ = mutable.make_move(move2)
        
        # Should end turn and go back to placement for next player
        assert mutable.current_phase == GamePhase.RING_PLACEMENT
        assert mutable.current_player == 2
    
    def test_phase_restored_after_unmake(self):
        """Test that phase is properly restored after unmake."""
        state = create_minimal_game_state()
        mutable = MutableGameState.from_immutable(state)
        
        original_phase = mutable.current_phase
        
        move = create_place_ring_move(
            player=1, to_pos=Position(x=3, y=3), count=1
        )
        undo = mutable.make_move(move)
        
        assert mutable.current_phase == GamePhase.MOVEMENT
        
        mutable.unmake_move(undo)
        
        assert mutable.current_phase == original_phase


class TestVictoryDetection:
    """Tests for victory detection functionality."""
    
    def test_count_rings_for_player(self):
        """Test ring counting for a player."""
        state = create_game_state_with_stacks()
        mutable = MutableGameState.from_immutable(state)
        
        # Player 1 has 15 in hand + 3 on board = 18
        count = mutable._count_rings_for_player(1)
        assert count == 18  # 15 in hand + 3 rings in stack at 2,2
    
    def test_is_player_eliminated_false(self):
        """Test that player with rings is not eliminated."""
        state = create_minimal_game_state()
        mutable = MutableGameState.from_immutable(state)
        
        assert not mutable._is_player_eliminated(1)
        assert not mutable._is_player_eliminated(2)
    
    def test_is_player_eliminated_true(self):
        """Test that player with no rings is eliminated."""
        state = create_minimal_game_state()
        mutable = MutableGameState.from_immutable(state)
        
        # Manually eliminate player 2's rings
        mutable._players[2].rings_in_hand = 0
        
        assert mutable._is_player_eliminated(2)
    
    def test_get_eliminated_players_empty(self):
        """Test no eliminated players at game start."""
        state = create_minimal_game_state()
        mutable = MutableGameState.from_immutable(state)
        
        eliminated = mutable.get_eliminated_players()
        assert len(eliminated) == 0
    
    def test_get_eliminated_players_with_eliminated(self):
        """Test detecting eliminated players."""
        state = create_minimal_game_state()
        mutable = MutableGameState.from_immutable(state)
        
        # Eliminate player 2
        mutable._players[2].rings_in_hand = 0
        
        eliminated = mutable.get_eliminated_players()
        assert 2 in eliminated
        assert 1 not in eliminated
    
    def test_is_game_over_false(self):
        """Test game not over at start."""
        state = create_minimal_game_state()
        mutable = MutableGameState.from_immutable(state)
        
        assert not mutable.is_game_over()
    
    def test_is_game_over_one_player_remaining(self):
        """Test game over when only one player has rings."""
        state = create_minimal_game_state()
        mutable = MutableGameState.from_immutable(state)
        
        # Eliminate player 2
        mutable._players[2].rings_in_hand = 0
        
        assert mutable.is_game_over()
    
    def test_is_game_over_status_completed(self):
        """Test game over when status is COMPLETED."""
        state = create_minimal_game_state()
        mutable = MutableGameState.from_immutable(state)
        
        mutable._game_status = GameStatus.COMPLETED
        
        assert mutable.is_game_over()
    
    def test_get_winner_none_when_not_over(self):
        """Test no winner when game is not over."""
        state = create_minimal_game_state()
        mutable = MutableGameState.from_immutable(state)
        
        assert mutable.get_winner() is None
    
    def test_get_winner_last_player_standing(self):
        """Test winner is the last player with rings."""
        state = create_minimal_game_state()
        mutable = MutableGameState.from_immutable(state)
        
        # Eliminate player 2
        mutable._players[2].rings_in_hand = 0
        
        winner = mutable.get_winner()
        assert winner == 1
    
    def test_get_winner_from_stored_value(self):
        """Test winner returned from stored value."""
        state = create_minimal_game_state()
        mutable = MutableGameState.from_immutable(state)
        
        mutable._winner = 1
        mutable._game_status = GameStatus.COMPLETED
        
        assert mutable.get_winner() == 1
    
    def test_victory_triggers_on_elimination(self):
        """Test that eliminating all opponent rings triggers victory."""
        state = create_game_state_with_stacks()
        mutable = MutableGameState.from_immutable(state)
        
        # Set player 2 to have only 1 ring (the stack we're about to capture)
        mutable._players[2].rings_in_hand = 0
        # Add a p2 stack at a position
        p2_pos = Position(x=5, y=5)
        mutable._stacks[p2_pos.to_key()] = MutableStack(
            position=p2_pos,
            rings=[2],
            stack_height=1,
            cap_height=1,
            controlling_player=2,
        )
        
        # Eliminate that stack
        move = create_eliminate_move(player=1, to_pos=p2_pos)
        _ = mutable.make_move(move)
        
        # Check victory
        assert mutable.is_game_over()
        assert mutable.get_winner() == 1
        assert mutable._game_status == GameStatus.COMPLETED
    
    def test_victory_restored_after_unmake(self):
        """Test that victory state is properly restored after unmake."""
        state = create_game_state_with_stacks()
        mutable = MutableGameState.from_immutable(state)
        
        # Set player 2 to have only 1 ring
        mutable._players[2].rings_in_hand = 0
        p2_pos = Position(x=5, y=5)
        mutable._stacks[p2_pos.to_key()] = MutableStack(
            position=p2_pos,
            rings=[2],
            stack_height=1,
            cap_height=1,
            controlling_player=2,
        )
        
        original_status = mutable._game_status
        original_winner = mutable._winner
        
        # Eliminate the stack (triggers victory)
        move = create_eliminate_move(player=1, to_pos=p2_pos)
        undo = mutable.make_move(move)
        
        assert mutable.is_game_over()
        assert mutable.get_winner() == 1
        
        # Unmake should restore
        mutable.unmake_move(undo)
        
        assert mutable._game_status == original_status
        assert mutable._winner == original_winner
        assert not mutable.is_game_over()


class TestPlayerSkipping:
    """Tests for skipping eliminated players during turn rotation."""
    
    def test_skip_eliminated_player_on_turn_end(self):
        """Test that eliminated players are skipped."""
        state = create_game_state_with_stacks()
        mutable = MutableGameState.from_immutable(state)
        
        # Eliminate player 2
        mutable._players[2].rings_in_hand = 0
        
        # Do a move that ends turn
        from_pos = Position(x=2, y=2)
        to_pos = Position(x=2, y=5)
        move = create_move_stack_move(
            player=1, from_pos=from_pos, to_pos=to_pos
        )
        _ = mutable.make_move(move)
        
        # Since player 2 is eliminated and only player 1 remains,
        # the game should be over
        assert mutable.is_game_over()
        assert mutable.get_winner() == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])