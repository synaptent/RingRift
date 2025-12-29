"""Tournament Configuration Module.

Provides shared configuration dataclasses for tournament CLI entrypoints,
including TournamentConfig for cross-configuration tournament settings.

Usage:
    from app.tournament.config import TournamentConfig

    config = TournamentConfig.from_args(args, yaml_config, config_path)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.utils.canonical_naming import normalize_board_type


@dataclass(frozen=True)
class TournamentConfig:
    """Shared tournament configuration for CLI entrypoints."""

    mode: str
    board_type: str | None
    num_players: int | None
    games_per_matchup: int | None
    max_moves: int | None
    seed: int
    output_dir: str
    config_path: str
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_args(
        cls,
        args: Any,
        config: Any,
        config_path: str,
    ) -> TournamentConfig:
        mode = getattr(args, "mode", "unknown")
        board_raw = getattr(args, "board", None)
        board_type = normalize_board_type(board_raw) if board_raw else None
        num_players = getattr(args, "players", None)
        if num_players is None:
            num_players = getattr(args, "num_players", None)

        games = getattr(args, "games", None)
        if games is None and hasattr(config, "tournament"):
            games = getattr(config.tournament, "default_games_per_matchup", None)

        max_moves = getattr(args, "max_moves", None)
        seed = int(getattr(args, "seed", 42))
        output_dir = getattr(args, "output_dir", "tournament_results")

        extra: dict[str, Any] = {}
        for key in ("tiers", "profiles_dir", "boards"):
            value = getattr(args, key, None)
            if value is not None:
                extra[key] = value

        return cls(
            mode=mode,
            board_type=board_type,
            num_players=num_players,
            games_per_matchup=games,
            max_moves=max_moves,
            seed=seed,
            output_dir=output_dir,
            config_path=config_path,
            extra=extra,
        )

    def summary(self) -> dict[str, Any]:
        payload = {
            "mode": self.mode,
            "board_type": self.board_type,
            "num_players": self.num_players,
            "games_per_matchup": self.games_per_matchup,
            "max_moves": self.max_moves,
            "seed": self.seed,
            "output_dir": self.output_dir,
            "config_path": self.config_path,
        }
        if self.extra:
            payload.update(self.extra)
        return payload
