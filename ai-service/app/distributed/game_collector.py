"""In-memory game collector for distributed workers.

This module provides an InMemoryGameCollector that implements the same
store_game interface as GameReplayDB but keeps games in memory for
serialization and transfer back to the coordinator.

Usage:
    collector = InMemoryGameCollector()

    # Use with record_completed_game (same interface as GameReplayDB)
    record_completed_game(
        db=collector,  # Works because it has the same store_game method
        initial_state=initial,
        final_state=final,
        moves=moves,
        metadata=metadata,
    )

    # Get serialized game data for transfer
    game_data = collector.get_serialized_games()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.models import GameState, Move


@dataclass
class HistoryEntry:
    """A history entry with before/after state snapshots."""

    move_number: int
    player: int
    phase_before: str
    phase_after: str
    status_before: str
    status_after: str
    state_before: GameState
    state_after: GameState


@dataclass
class CollectedGame:
    """A game collected in memory."""

    game_id: str
    initial_state: GameState
    final_state: GameState
    moves: list[Move]
    metadata: dict[str, Any] = field(default_factory=dict)
    history_entries: list[HistoryEntry] = field(default_factory=list)


class InMemoryGameCollector:
    """In-memory game collector that mimics GameReplayDB interface.

    This class implements the store_game method so it can be used as a drop-in
    replacement for GameReplayDB when you want to capture games in memory
    rather than write them to SQLite.
    """

    def __init__(self):
        self._games: list[CollectedGame] = []

    def store_game(
        self,
        game_id: str,
        initial_state: GameState,
        final_state: GameState,
        moves: list[Move],
        choices: list[dict] | None = None,
        metadata: dict[str, Any] | None = None,
        store_history_entries: bool = True,
        compress_states: bool = False,  # Not used in memory, but kept for API compat
        snapshot_interval: int = 20,  # Ignored for in-memory storage
    ) -> str:
        """Store a game in memory.

        This method has the same signature as GameReplayDB.store_game
        so it can be used interchangeably.

        Args:
            game_id: Unique game identifier
            initial_state: Initial game state
            final_state: Final game state
            moves: List of all moves in order
            choices: Optional list of player choices (ignored for in-memory storage)
            metadata: Optional game metadata
            store_history_entries: If True (default), compute and store history
                                   entries with before/after state snapshots
            compress_states: Ignored for in-memory storage
            snapshot_interval: Ignored for in-memory storage
        """
        # Import here to avoid circular imports
        from app.game_engine import GameEngine

        history_entries: list[HistoryEntry] = []

        if store_history_entries and moves:
            prev_state = initial_state
            for i, move in enumerate(moves):
                state_after = GameEngine.apply_move(prev_state, move)
                entry = HistoryEntry(
                    move_number=i,
                    player=move.player,
                    phase_before=prev_state.current_phase.value,
                    phase_after=state_after.current_phase.value,
                    status_before=prev_state.game_status.value,
                    status_after=state_after.game_status.value,
                    state_before=prev_state,
                    state_after=state_after,
                )
                history_entries.append(entry)
                prev_state = state_after

        game = CollectedGame(
            game_id=game_id,
            initial_state=initial_state,
            final_state=final_state,
            moves=moves,
            metadata=metadata or {},
            history_entries=history_entries,
        )
        self._games.append(game)
        return game_id

    def get_games(self) -> list[CollectedGame]:
        """Get all collected games."""
        return self._games

    def clear(self) -> None:
        """Clear all collected games."""
        self._games = []

    def get_serialized_games(
        self,
        include_history_entries: bool = True,
    ) -> list[dict[str, Any]]:
        """Get all games as JSON-serializable dictionaries.

        Returns a list of game dictionaries suitable for network transfer.
        Each game contains:
        - game_id: str
        - initial_state: serialized GameState
        - final_state: serialized GameState
        - moves: list of serialized Move objects
        - metadata: dict
        - history_entries: list of history entry dicts (if include_history_entries)

        Args:
            include_history_entries: If True (default), include history entries
                                     with before/after states. Set to False for
                                     smaller payloads (states will be recomputed
                                     when writing to DB).
        """
        serialized = []
        for game in self._games:
            game_dict: dict[str, Any] = {
                "game_id": game.game_id,
                "initial_state": game.initial_state.model_dump(),
                "final_state": game.final_state.model_dump(),
                "moves": [m.model_dump() for m in game.moves],
                "metadata": game.metadata,
            }

            if include_history_entries and game.history_entries:
                game_dict["history_entries"] = [
                    {
                        "move_number": e.move_number,
                        "player": e.player,
                        "phase_before": e.phase_before,
                        "phase_after": e.phase_after,
                        "status_before": e.status_before,
                        "status_after": e.status_after,
                        "state_before": e.state_before.model_dump(),
                        "state_after": e.state_after.model_dump(),
                    }
                    for e in game.history_entries
                ]

            serialized.append(game_dict)
        return serialized


def deserialize_game_data(
    game_data: dict[str, Any],
) -> CollectedGame:
    """Deserialize a game dictionary back to a CollectedGame.

    Args:
        game_data: Serialized game data from get_serialized_games()

    Returns:
        CollectedGame with reconstructed GameState and Move objects
    """
    history_entries: list[HistoryEntry] = []
    if "history_entries" in game_data:
        for e in game_data["history_entries"]:
            history_entries.append(HistoryEntry(
                move_number=e["move_number"],
                player=e["player"],
                phase_before=e["phase_before"],
                phase_after=e["phase_after"],
                status_before=e["status_before"],
                status_after=e["status_after"],
                state_before=GameState.model_validate(e["state_before"]),
                state_after=GameState.model_validate(e["state_after"]),
            ))

    return CollectedGame(
        game_id=game_data["game_id"],
        initial_state=GameState.model_validate(game_data["initial_state"]),
        final_state=GameState.model_validate(game_data["final_state"]),
        moves=[Move.model_validate(m) for m in game_data["moves"]],
        metadata=game_data.get("metadata", {}),
        history_entries=history_entries,
    )


def write_games_to_db(
    db,  # GameReplayDB
    game_data_list: list[dict[str, Any]],
    extra_metadata: dict[str, Any] | None = None,
) -> int:
    """Write serialized game data to a GameReplayDB.

    This is used by the coordinator to write games received from
    distributed workers to the central database.

    Args:
        db: GameReplayDB instance
        game_data_list: List of serialized game dictionaries
        extra_metadata: Additional metadata to merge into each game

    Returns:
        Number of games written
    """
    count = 0
    for game_data in game_data_list:
        try:
            game = deserialize_game_data(game_data)
            metadata = game.metadata.copy()
            if extra_metadata:
                metadata.update(extra_metadata)

            db.store_game(
                game_id=game.game_id,
                initial_state=game.initial_state,
                final_state=game.final_state,
                moves=game.moves,
                metadata=metadata,
            )
            count += 1
        except Exception as e:
            # Log but don't fail - we want to write as many games as possible
            print(f"WARNING: Failed to write game {game_data.get('game_id', 'unknown')}: {e}")

    return count
