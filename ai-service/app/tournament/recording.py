"""Tournament recording helpers for canonical GameReplayDB output."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.db.unified_recording import RecordSource, RecordingConfig


@dataclass(frozen=True)
class TournamentRecordingOptions:
    """Configuration for tournament game recording."""

    enabled: bool = True
    source: str = RecordSource.TOURNAMENT
    engine_mode: str = "tournament"
    db_prefix: str = "tournament"
    db_dir: str = "data/games"
    fsm_validation: bool = True
    parity_mode: str | None = None
    snapshot_interval: int | None = None
    store_history_entries: bool = True
    tags: list[str] = field(default_factory=list)
    extra_metadata: dict[str, Any] = field(default_factory=dict)

    def build_recording_config(
        self,
        board_type: str,
        num_players: int,
        *,
        model_id: str | None = None,
        difficulty: int | None = None,
    ) -> RecordingConfig:
        """Build a RecordingConfig for the given match settings."""
        return RecordingConfig(
            board_type=board_type,
            num_players=num_players,
            source=self.source,
            engine_mode=self.engine_mode,
            model_id=model_id,
            difficulty=difficulty,
            tags=list(self.tags),
            db_prefix=self.db_prefix,
            db_dir=self.db_dir,
            store_history_entries=self.store_history_entries,
            snapshot_interval=self.snapshot_interval,
            parity_mode=self.parity_mode,
            fsm_validation=self.fsm_validation,
        )
