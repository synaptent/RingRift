from __future__ import annotations

from typing import Protocol, List

from app.models import GameState, Move


class Validator(Protocol):
    """
    Interface for move validators.
    Mirrors the TS Validator interface.
    """
    def validate(self, state: GameState, move: Move) -> bool:
        """Return True if the move is valid in the given state."""
        ...


class Mutator(Protocol):
    """
    Interface for state mutators.
    Mirrors the TS Mutator interface.
    """
    def apply(self, state: GameState, move: Move) -> None:
        """
        Apply the move to the state in-place.
        Note: The engine wrapper handles state copying; mutators work on
        the copy.
        """
        ...


class RulesEngine(Protocol):
    """Abstract rules engine interface.

    Mirrors the TS rules engine responsibilities.
    """

    def get_valid_moves(self, state: GameState, player: int) -> List[Move]:
        """Return all legal moves for `player` given `state`."""
        ...

    def apply_move(self, state: GameState, move: Move) -> GameState:
        """
        Apply `move` to `state` and return the resulting GameState.

        Implementations must treat `state` as immutable and return a
        new instance.
        """
        ...

    def validate_move(self, state: GameState, move: Move) -> bool:
        """
        Validate if a specific move is legal in the current state.
        """
        ...