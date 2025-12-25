# Testing Module

Reusable test helpers, fixtures, and factories for RingRift AI service tests.

## Overview

This module provides shared test utilities that can be imported into `conftest.py` files or test modules directly:

- Core game state fixtures
- Coordination/cluster fixtures
- Mock objects

## Key Components

### Core Fixtures

```python
from app.testing import (
    create_player,
    create_game_state,
    create_board_state,
    create_move,
    create_position,
    create_ring_stack,
)

# Create a player
player = create_player(player_number=1, elo=1200)

# Create a game state
state = create_game_state(
    board_type="hex8",
    num_players=2,
    current_player=1,
)

# Create a board state
board = create_board_state(board_type="square8")

# Create a move
move = create_move(
    player=player,
    from_pos=(2, 3),
    to_pos=(4, 5),
)

# Create a position
pos = create_position(row=3, col=4)

# Create a ring stack
stack = create_ring_stack(owner=1, height=3)
```

### Coordination Fixtures

```python
from app.testing import (
    create_temp_db,
    create_coordination_db,
    create_cluster_state,
    create_training_state,
    create_host_sync_state,
    create_node_resources,
)

# Create temporary database
db_path = create_temp_db()

# Create coordination database with schema
db = create_coordination_db()

# Create cluster state
cluster = create_cluster_state(
    leader_id="node-1",
    alive_peers=["node-1", "node-2", "node-3"],
)

# Create training state
training = create_training_state(
    board_type="hex8",
    num_players=2,
    epoch=15,
    loss=0.025,
)

# Create host sync state
sync_state = create_host_sync_state(
    host="worker-1",
    last_sync_time=datetime.now(),
    pending_files=5,
)

# Create node resources
resources = create_node_resources(
    gpu_memory_used=40.0,
    gpu_memory_total=96.0,
    cpu_percent=25.0,
)
```

### Mock Objects

```python
from app.testing import (
    MockEventBus,
    MockClusterState,
    MockTrainingState,
    MockHostSyncState,
    MockNodeResources,
)

# Mock event bus for testing event emission
event_bus = MockEventBus()
event_bus.emit("TRAINING_COMPLETE", {"model": "v3"})
assert event_bus.events[-1].type == "TRAINING_COMPLETE"

# Mock cluster state
cluster = MockClusterState()
cluster.set_leader("node-1")
cluster.add_peer("node-2")
```

## Usage in conftest.py

```python
# tests/conftest.py
import pytest
from app.testing import create_game_state, create_player

@pytest.fixture
def game_state():
    return create_game_state()

@pytest.fixture
def player():
    return create_player(player_number=1)

@pytest.fixture
def two_players():
    return [
        create_player(player_number=1),
        create_player(player_number=2),
    ]
```

## Usage in Tests

```python
from app.testing import create_player, create_move

def test_player_moves():
    player = create_player(player_number=1)
    move = create_move(player=player, from_pos=(0, 0), to_pos=(1, 1))

    assert move.player.player_number == 1
    assert move.from_pos == (0, 0)
    assert move.to_pos == (1, 1)

def test_with_fixtures(game_state, player):
    # Use pytest fixtures
    game_state.apply_move(player, some_move)
    ...
```

## Factory Customization

All factories accept keyword arguments for customization:

```python
# Default game state
state = create_game_state()

# Customized game state
state = create_game_state(
    board_type="square19",
    num_players=4,
    current_player=3,
    moves_made=50,
)
```

## See Also

- `tests/conftest.py` - Main test configuration
- `pytest` - Python testing framework
