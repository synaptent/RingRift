# Interfaces Module

Shared protocols and abstract base classes for breaking circular dependencies.

## Overview

This module defines Protocol classes and Abstract Base Classes that help break circular dependencies between modules. Implementations live in their respective packages but depend on these interfaces.

## Architecture

```
app.interfaces (this package) - no dependencies except typing/abc
└── Used by: app.ai, app.game_engine, app.training, app.models
```

## Protocols Defined

### HashProvider

Interface for Zobrist-style position hashing:

```python
from app.interfaces import HashProvider

class ZobristHasher(HashProvider):
    def hash_position(self, board_state: list[list[int]]) -> int:
        """Compute hash for board position."""
        ...

    def incremental_hash(self, prev_hash: int, move: dict) -> int:
        """Update hash incrementally after a move."""
        ...
```

### MoveCacheProvider

Interface for caching move computations:

```python
from app.interfaces import MoveCacheProvider

class TranspositionTable(MoveCacheProvider):
    def get_cached_moves(self, position_hash: int) -> list[dict] | None:
        """Retrieve cached moves for position."""
        ...

    def cache_moves(self, position_hash: int, moves: list[dict]) -> None:
        """Cache moves for position."""
        ...
```

### ModelProvider

Interface for loading AI models:

```python
from app.interfaces import ModelProvider

class TorchModelLoader(ModelProvider):
    def load_model(self, path: str) -> Any:
        """Load model from path."""
        ...

    def get_model_info(self, path: str) -> dict:
        """Get model metadata."""
        ...
```

### EncodingProvider

Interface for move encoding/decoding:

```python
from app.interfaces import EncodingProvider

class MoveEncoder(EncodingProvider):
    def encode_move(self, move: dict, board_type: str) -> int:
        """Encode move to integer index."""
        ...

    def decode_move(self, index: int, board_type: str) -> dict:
        """Decode integer index to move dict."""
        ...
```

## Usage Pattern

```python
# In app.ai (depends on interface, not implementation)
from app.interfaces import HashProvider

class MCTSNode:
    def __init__(self, hasher: HashProvider):
        self.hasher = hasher

    def get_position_hash(self, state):
        return self.hasher.hash_position(state)

# In app.zobrist (provides implementation)
from app.interfaces import HashProvider

class ZobristHasher(HashProvider):
    ...

# In main code (wires things together)
from app.ai import MCTSNode
from app.zobrist import ZobristHasher

node = MCTSNode(hasher=ZobristHasher())
```

## Benefits

1. **Breaks circular imports**: Modules depend on interfaces, not concrete implementations
2. **Enables testing**: Easy to mock interfaces in tests
3. **Supports pluggability**: Swap implementations without changing consumers
4. **Documents contracts**: Clear API boundaries between modules
