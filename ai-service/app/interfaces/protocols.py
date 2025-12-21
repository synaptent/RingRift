"""Protocol definitions for breaking circular dependencies.

These protocols define the interfaces that allow modules to depend on
abstractions rather than concrete implementations.

Migration Plan:
1. Modules import protocols from here instead of concrete classes
2. Concrete implementations remain in original locations
3. Dependency injection wires up implementations at runtime

Current Circular Dependencies to Break:
- game_engine -> ai.zobrist: Use HashProvider
- game_engine -> ai.move_cache: Use MoveCacheProvider
- models.loader -> ai.nnue: Use ModelProvider
- ai -> training.encoding: Use EncodingProvider
"""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from app.models import GameState, Move


@runtime_checkable
class HashProvider(Protocol):
    """Protocol for Zobrist-style position hashing.

    Implementations: app.core.zobrist.ZobristHash

    Used by: GameEngine for incremental hash updates
    """

    @abstractmethod
    def compute_initial_hash(self, state: GameState) -> int:
        """Compute full hash for a game state."""
        ...

    @abstractmethod
    def get_player_hash(self, player: int) -> int:
        """Get hash component for player."""
        ...

    @abstractmethod
    def get_phase_hash(self, phase: Any) -> int:
        """Get hash component for game phase."""
        ...

    @abstractmethod
    def get_marker_hash(self, position: str, player: int) -> int:
        """Get hash component for marker at position."""
        ...

    @abstractmethod
    def get_stack_hash(self, position: str, heights: tuple[int, ...]) -> int:
        """Get hash component for stack at position."""
        ...

    @abstractmethod
    def get_collapsed_hash(self, position: str) -> int:
        """Get hash component for collapsed position."""
        ...


@runtime_checkable
class MoveCacheProvider(Protocol):
    """Protocol for move caching.

    Implementations: app.ai.move_cache.MoveCache

    Used by: GameEngine for caching valid moves by position hash
    """

    @abstractmethod
    def get(self, state: GameState, player: int) -> list[Move] | None:
        """Get cached moves for state/player, or None if not cached."""
        ...

    @abstractmethod
    def set(self, state: GameState, player: int, moves: list[Move]) -> None:
        """Cache moves for state/player."""
        ...

    @abstractmethod
    def clear(self) -> None:
        """Clear all cached moves."""
        ...


@runtime_checkable
class ModelProvider(Protocol):
    """Protocol for AI model loading.

    Implementations:
        - app.ai.nnue.NNUENetwork
        - app.ai.nnue_policy.NNUEPolicyNetwork

    Used by: app.models.loader for dynamic model loading
    """

    @abstractmethod
    def load(self, path: str, device: str = "cpu") -> Any:
        """Load model from path."""
        ...

    @abstractmethod
    def get_model_type(self) -> str:
        """Return model type identifier."""
        ...


@runtime_checkable
class EncodingProvider(Protocol):
    """Protocol for move encoding/decoding.

    Implementations: app.training.encoding

    Used by: Neural network models for policy head indexing
    """

    @abstractmethod
    def encode_move(self, move: Move, board_type: str) -> int:
        """Encode move to policy index."""
        ...

    @abstractmethod
    def decode_move(self, index: int, board_type: str, state: GameState) -> Move | None:
        """Decode policy index to move."""
        ...


# =============================================================================
# Serialization Protocol
# =============================================================================


from typing import TypeVar

T = TypeVar("T")


@runtime_checkable
class SerializableProtocol(Protocol):
    """Protocol for serializable objects.

    Implementations: app.core.marshalling.Serializable mixin

    Used by: All dataclasses that need to_dict/from_dict serialization

    Note: Prefer using the Serializable mixin from app.core.marshalling
    for implementation. This protocol is for type checking.

    Example:
        from app.core import Serializable

        @dataclass
        class MyConfig(Serializable):
            name: str
            value: int

        config = MyConfig(name="test", value=42)
        data = config.to_dict()
        restored = MyConfig.from_dict(data)
    """

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        ...

    @classmethod
    def from_dict(cls: type[T], data: dict[str, Any]) -> T:
        """Deserialize from dictionary."""
        ...


# Type aliases for common patterns
HashValue = int
PolicyIndex = int
