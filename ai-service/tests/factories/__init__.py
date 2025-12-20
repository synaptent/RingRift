"""Test Factories and Fixtures for RingRift AI Service.

This package provides reusable factory functions and fixtures for tests:
- Game state factories
- Config factories
- Mock objects
- Common test data

Usage:
    from tests.factories import (
        create_game_state,
        create_training_config,
        create_mock_model,
    )

    def test_something():
        state = create_game_state(board_type="square8", num_players=2)
        config = create_training_config(learning_rate=0.001)
        ...

Integration with pytest:
    Import fixtures in conftest.py:

        from tests.factories.fixtures import (
            game_state,
            training_config,
            temp_db,
        )

    Then use in tests:
        def test_training(training_config, temp_db):
            ...

Factory Guidelines:
    - Factories return real objects with sensible defaults
    - All parameters should be optional with defaults
    - Factories should be composable
    - Use snake_case for factory names
"""

from tests.factories.game import (
    create_game_state,
    create_board_config,
    create_move,
    create_game_record,
)
from tests.factories.config import (
    create_training_config,
    create_unified_config,
    create_evaluation_config,
)
from tests.factories.mocks import (
    create_mock_model,
    create_mock_coordinator,
    MockAsyncContext,
)

__all__ = [
    "MockAsyncContext",
    "create_board_config",
    "create_evaluation_config",
    "create_game_record",
    # Game factories
    "create_game_state",
    "create_mock_coordinator",
    # Mocks
    "create_mock_model",
    "create_move",
    # Config factories
    "create_training_config",
    "create_unified_config",
]
