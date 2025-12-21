"""Testing utilities and shared fixtures for RingRift AI service.

This package provides reusable test helpers, fixtures, and factories
that can be imported into conftest.py files or test modules directly.

Modules:
    fixtures: Core game state, player, and board fixtures
    coordination_fixtures: Database, event bus, and cluster fixtures

Usage in conftest.py:
    from app.testing.fixtures import (
        create_player,
        create_game_state,
        create_board_state,
    )

    @pytest.fixture
    def game_state():
        return create_game_state()

Usage in tests:
    from app.testing import create_player, create_move

    def test_player_moves():
        player = create_player(player_number=1)
        move = create_move(player=player)
        ...
"""

from app.testing.coordination_fixtures import (
    MockClusterState,
    MockEventBus,
    MockHostSyncState,
    MockNodeResources,
    MockTrainingState,
    create_cluster_state,
    create_coordination_db,
    create_host_sync_state,
    create_node_resources,
    create_temp_db,
    create_training_state,
)
from app.testing.fixtures import (
    create_board_state,
    create_game_state,
    create_move,
    create_player,
    create_position,
    create_ring_stack,
)

__all__ = [
    # Core fixtures
    "create_board_state",
    "create_game_state",
    "create_move",
    "create_player",
    "create_position",
    "create_ring_stack",
    # Coordination fixtures
    "MockClusterState",
    "MockEventBus",
    "MockHostSyncState",
    "MockNodeResources",
    "MockTrainingState",
    "create_cluster_state",
    "create_coordination_db",
    "create_host_sync_state",
    "create_node_resources",
    "create_temp_db",
    "create_training_state",
]
