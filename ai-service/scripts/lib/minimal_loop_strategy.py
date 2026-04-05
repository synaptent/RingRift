"""Board-aware strategy helpers for the minimal AlphaZero loop.

The standalone minimal loop was originally tuned for fast small-board configs.
Large boards inherit those same defaults, which makes a single iteration take
multiple days on a single node. This module centralizes cheaper large-board
profiles and transfer-learning bootstrap hints so the CLI can stay small.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class LoopProfile:
    """Resolved per-iteration settings for the minimal loop."""

    games_per_iter: int
    eval_games: int
    budget: int
    epochs: int
    batch_size: int


STANDARD_PROFILE = LoopProfile(
    games_per_iter=300,
    eval_games=100,
    budget=128,
    epochs=15,
    batch_size=512,
)


# Large-board iterations need an order-of-magnitude less selfplay/eval work to
# stay within a single-node workday. These settings are intentionally conservative
# bootstrapping profiles, not final high-Elo throughput profiles.
LARGE_BOARD_PROFILES: dict[str, LoopProfile] = {
    "square19_2p": LoopProfile(games_per_iter=48, eval_games=20, budget=64, epochs=10, batch_size=256),
    "square19_3p": LoopProfile(games_per_iter=36, eval_games=16, budget=56, epochs=10, batch_size=256),
    "square19_4p": LoopProfile(games_per_iter=28, eval_games=14, budget=48, epochs=8, batch_size=256),
    "hexagonal_2p": LoopProfile(games_per_iter=36, eval_games=16, budget=48, epochs=10, batch_size=256),
    "hexagonal_3p": LoopProfile(games_per_iter=24, eval_games=12, budget=40, epochs=8, batch_size=256),
    "hexagonal_4p": LoopProfile(games_per_iter=18, eval_games=10, budget=32, epochs=6, batch_size=256),
}


# Stronger large-board checkpoints are more useful as bootstrap sources than
# restarting weak 2p/3p configs from scratch.
TRANSFER_HINTS: dict[str, str] = {
    "square19_2p": "ringrift_best_square19_4p.pth",
    "square19_3p": "ringrift_best_square19_4p.pth",
    "hexagonal_2p": "ringrift_best_hexagonal_4p.pth",
    "hexagonal_3p": "ringrift_best_hexagonal_4p.pth",
}


def get_config_key(board_type: str, num_players: int) -> str:
    """Return canonical board/player config key."""
    return f"{board_type}_{num_players}p"


def get_profile_name(board_type: str, num_players: int, profile: str) -> str:
    """Resolve requested profile name to the effective profile."""
    if profile not in {"auto", "standard", "large-board"}:
        raise ValueError(f"Unsupported profile: {profile}")
    if profile != "auto":
        return profile
    return "large-board" if get_config_key(board_type, num_players) in LARGE_BOARD_PROFILES else "standard"


def resolve_loop_profile(
    board_type: str,
    num_players: int,
    profile: str = "auto",
    *,
    games_per_iter: int | None = None,
    eval_games: int | None = None,
    budget: int | None = None,
    epochs: int | None = None,
    batch_size: int | None = None,
) -> dict[str, object]:
    """Resolve board-aware defaults with explicit CLI overrides."""
    config_key = get_config_key(board_type, num_players)
    profile_name = get_profile_name(board_type, num_players, profile)
    base = LARGE_BOARD_PROFILES.get(config_key, STANDARD_PROFILE) if profile_name == "large-board" else STANDARD_PROFILE
    resolved = LoopProfile(
        games_per_iter=games_per_iter if games_per_iter is not None else base.games_per_iter,
        eval_games=eval_games if eval_games is not None else base.eval_games,
        budget=budget if budget is not None else base.budget,
        epochs=epochs if epochs is not None else base.epochs,
        batch_size=batch_size if batch_size is not None else base.batch_size,
    )
    return {
        "config_key": config_key,
        "profile": profile_name,
        "settings": asdict(resolved),
    }


def recommend_transfer_source(board_type: str, num_players: int) -> str | None:
    """Return a same-board checkpoint name that is a good bootstrap candidate."""
    return TRANSFER_HINTS.get(get_config_key(board_type, num_players))
