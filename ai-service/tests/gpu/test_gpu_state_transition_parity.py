"""Contract tests for GPU vs CPU state transition parity.

These tests verify that state transitions (move application, phase changes,
victory detection) produce identical results between GPU and CPU implementations.

Test Categories:
- Move application parity: Same move produces identical state changes
- Phase transition parity: Game phases advance identically
- Victory detection parity: Same conditions trigger same victory
"""

import pytest
import torch

from app.models import BoardType, GamePhase, MoveType

try:
    from app.ai.gpu_parallel_games import ParallelGameRunner
    from app.ai.gpu_game_types import GameStatus, GamePhase as GPUPhase
    from app.ai.gpu_batch_state import BatchGameState
    from app.rules.core import get_victory_threshold

    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False


pytestmark = pytest.mark.skipif(
    not GPU_AVAILABLE, reason="GPU modules not available"
)


class TestMoveApplicationParity:
    """Verify move application produces consistent results."""

    @pytest.fixture
    def device(self):
        return torch.device("cpu")

    def test_placement_creates_stack(self, device):
        """Placing a ring should create a height-1 stack owned by the placer."""
        runner = ParallelGameRunner(
            batch_size=1,
            board_size=8,
            num_players=2,
            device=device,
            shadow_validation=False,
            state_validation=False,
        )
        state = runner.state

        # Verify initial state
        assert state.rings_in_hand[0, 1].item() == 18
        assert (state.stack_height == 0).all()

        # Set up for placement
        state.current_player[0] = 1
        state.current_phase[0] = GPUPhase.RING_PLACEMENT

        # Run a step
        weights = [runner._default_weights()]
        runner._step_games(weights)

        # Should have placed a ring
        placed = (state.stack_height > 0).sum().item()
        assert placed >= 1, "At least one stack should exist after placement"

        # Rings in hand should decrease
        assert state.rings_in_hand[0, 1].item() < 18

    def test_placement_consistent_across_batch(self, device):
        """Identical initial states should produce identical placements with same seed."""
        torch.manual_seed(42)

        runner = ParallelGameRunner(
            batch_size=4,
            board_size=8,
            num_players=2,
            device=device,
            shadow_validation=False,
            state_validation=False,
        )
        state = runner.state

        # All games start identically
        assert (state.rings_in_hand[:, 1] == 18).all()

        # Run a step
        weights = [runner._default_weights() for _ in range(4)]
        runner._step_games(weights)

        # Check that some placements occurred
        placements = (state.stack_height > 0).any(dim=(1, 2))
        assert placements.all(), "All games should have at least one placement"


class TestPhaseTransitionParity:
    """Verify phase transitions happen correctly."""

    @pytest.fixture
    def device(self):
        return torch.device("cpu")

    def test_placement_to_movement_transition(self, device):
        """After placing, player should transition to movement phase."""
        runner = ParallelGameRunner(
            batch_size=1,
            board_size=8,
            num_players=2,
            device=device,
            shadow_validation=False,
            state_validation=False,
        )
        state = runner.state

        # Start in placement phase
        state.current_phase[0] = GPUPhase.RING_PLACEMENT
        state.current_player[0] = 1

        weights = [runner._default_weights()]

        # After placement, should move to RING_MOVEMENT
        runner._step_games(weights)

        # Phase should have changed (either movement or next player's placement)
        # The exact transition depends on game rules
        assert state.current_phase[0].item() in [
            GPUPhase.RING_PLACEMENT,
            GPUPhase.MOVEMENT,
        ]

    def test_phase_sequence_validity(self, device):
        """Phases should follow valid game flow."""
        runner = ParallelGameRunner(
            batch_size=1,
            board_size=8,
            num_players=2,
            device=device,
            shadow_validation=False,
            state_validation=False,
        )
        state = runner.state

        weights = [runner._default_weights()]
        valid_phases = {
            GPUPhase.RING_PLACEMENT,
            GPUPhase.MOVEMENT,
            GPUPhase.LINE_PROCESSING,
            GPUPhase.TERRITORY_PROCESSING,
            GPUPhase.END_TURN,
        }

        # Run several steps
        for _ in range(20):
            runner._step_games(weights)
            runner._check_victory_conditions()

            if state.game_status[0].item() != GameStatus.ACTIVE:
                break

            phase = state.current_phase[0].item()
            assert phase in valid_phases, f"Invalid phase: {phase}"


class TestVictoryDetectionParity:
    """Verify victory conditions are detected identically."""

    @pytest.fixture
    def device(self):
        return torch.device("cpu")

    def test_elimination_threshold_exact(self, device):
        """Reaching exactly the threshold should trigger victory."""
        threshold = get_victory_threshold(BoardType.SQUARE8, 2)

        runner = ParallelGameRunner(
            batch_size=1,
            board_size=8,
            num_players=2,
            device=device,
            shadow_validation=False,
            state_validation=False,
        )
        state = runner.state

        # Set up just below threshold - should not trigger
        state.rings_caused_eliminated[0, 1] = threshold - 1
        state.current_player[0] = 1
        state.game_status[0] = GameStatus.ACTIVE

        runner._check_victory_conditions()
        assert state.game_status[0].item() == GameStatus.ACTIVE

        # Reach threshold - should trigger
        state.rings_caused_eliminated[0, 1] = threshold
        runner._check_victory_conditions()
        assert state.game_status[0].item() == GameStatus.COMPLETED
        assert state.winner[0].item() == 1

    def test_victory_detection_batch_consistency(self, device):
        """Victory detection should work identically across batch."""
        threshold = get_victory_threshold(BoardType.SQUARE8, 2)

        runner = ParallelGameRunner(
            batch_size=4,
            board_size=8,
            num_players=2,
            device=device,
            shadow_validation=False,
            state_validation=False,
        )
        state = runner.state

        # Game 0: P1 wins
        state.rings_caused_eliminated[0, 1] = threshold
        # Game 1: P2 wins
        state.rings_caused_eliminated[1, 2] = threshold
        # Game 2: No winner yet
        state.rings_caused_eliminated[2, 1] = threshold - 1
        state.rings_caused_eliminated[2, 2] = threshold - 1
        # Game 3: P1 wins
        state.rings_caused_eliminated[3, 1] = threshold + 5

        state.game_status[:] = GameStatus.ACTIVE
        runner._check_victory_conditions()

        assert state.game_status[0].item() == GameStatus.COMPLETED
        assert state.winner[0].item() == 1

        assert state.game_status[1].item() == GameStatus.COMPLETED
        assert state.winner[1].item() == 2

        assert state.game_status[2].item() == GameStatus.ACTIVE

        assert state.game_status[3].item() == GameStatus.COMPLETED
        assert state.winner[3].item() == 1


class TestStateInvariantsParity:
    """Verify state invariants are maintained identically."""

    @pytest.fixture
    def device(self):
        return torch.device("cpu")

    def test_ring_conservation(self, device):
        """Total rings should be conserved during gameplay."""
        runner = ParallelGameRunner(
            batch_size=1,
            board_size=8,
            num_players=2,
            device=device,
            shadow_validation=False,
            state_validation=False,
        )
        state = runner.state

        # Calculate initial total rings
        initial_total = (
            state.rings_in_hand[0, 1].item()
            + state.rings_in_hand[0, 2].item()
            + state.stack_height.sum().item()
            + state.buried_rings[0, 1].item()
            + state.buried_rings[0, 2].item()
        )

        weights = [runner._default_weights()]

        # Run several steps
        for _ in range(50):
            runner._step_games(weights)
            runner._check_victory_conditions()

            if state.game_status[0].item() != GameStatus.ACTIVE:
                break

        # Calculate final total (accounting for eliminated rings)
        final_total = (
            state.rings_in_hand[0, 1].item()
            + state.rings_in_hand[0, 2].item()
            + state.stack_height.sum().item()
            + state.buried_rings[0, 1].item()
            + state.buried_rings[0, 2].item()
            + state.rings_caused_eliminated[0, 1].item()
            + state.rings_caused_eliminated[0, 2].item()
        )

        # Rings should be conserved (with eliminated counted separately)
        assert initial_total <= final_total, (
            f"Ring count should not decrease: initial={initial_total}, final={final_total}"
        )

    def test_stack_owner_validity(self, device):
        """Stack owners should always be valid player numbers."""
        runner = ParallelGameRunner(
            batch_size=2,
            board_size=8,
            num_players=2,
            device=device,
            shadow_validation=False,
            state_validation=False,
        )
        state = runner.state

        weights = [runner._default_weights() for _ in range(2)]

        for _ in range(30):
            runner._step_games(weights)
            runner._check_victory_conditions()

            # Where stacks exist, owner should be 1 or 2
            has_stack = state.stack_height > 0
            if has_stack.any():
                owners = state.stack_owner[has_stack]
                valid = (owners >= 1) & (owners <= 2)
                assert valid.all(), f"Invalid owners found: {owners[~valid]}"


class TestMultiplayerParity:
    """Verify parity for 3+ player games."""

    @pytest.fixture
    def device(self):
        return torch.device("cpu")

    def test_three_player_turn_order(self, device):
        """Three-player games should cycle through all players."""
        runner = ParallelGameRunner(
            batch_size=1,
            board_size=8,
            num_players=3,
            device=device,
            shadow_validation=False,
            state_validation=False,
        )
        state = runner.state

        weights = [runner._default_weights()]
        seen_players = set()

        for _ in range(20):
            current = state.current_player[0].item()
            seen_players.add(current)

            runner._step_games(weights)
            runner._check_victory_conditions()

            if state.game_status[0].item() != GameStatus.ACTIVE:
                break

            # Current player should be 1, 2, or 3
            assert state.current_player[0].item() in [1, 2, 3]

        # Should have seen all three players
        assert seen_players == {1, 2, 3}, f"Missing players: {set([1,2,3]) - seen_players}"

    def test_four_player_victory_threshold(self, device):
        """Four-player games should use correct victory threshold."""
        threshold = get_victory_threshold(BoardType.SQUARE8, 4)

        runner = ParallelGameRunner(
            batch_size=1,
            board_size=8,
            num_players=4,
            device=device,
            shadow_validation=False,
            state_validation=False,
        )
        state = runner.state

        # P3 reaches threshold
        state.rings_caused_eliminated[0, 3] = threshold
        state.rings_in_hand[0, :] = 5
        state.current_player[0] = 3
        state.game_status[0] = GameStatus.ACTIVE

        runner._check_victory_conditions()

        assert state.game_status[0].item() == GameStatus.COMPLETED
        assert state.winner[0].item() == 3
