"""Tests for minimal-loop board-aware strategy helpers."""

from __future__ import annotations

from scripts.lib.minimal_loop_strategy import (
    LARGE_BOARD_PROFILES,
    STANDARD_PROFILE,
    recommend_transfer_source,
    resolve_loop_profile,
)


def test_auto_profile_uses_standard_defaults_for_small_boards():
    resolved = resolve_loop_profile("hex8", 2, "auto")

    assert resolved["profile"] == "standard"
    assert resolved["settings"]["games_per_iter"] == STANDARD_PROFILE.games_per_iter
    assert resolved["settings"]["budget"] == STANDARD_PROFILE.budget


def test_auto_profile_uses_large_board_preset_for_square19():
    resolved = resolve_loop_profile("square19", 3, "auto")
    expected = LARGE_BOARD_PROFILES["square19_3p"]

    assert resolved["profile"] == "large-board"
    assert resolved["settings"]["games_per_iter"] == expected.games_per_iter
    assert resolved["settings"]["eval_games"] == expected.eval_games
    assert resolved["settings"]["budget"] == expected.budget
    assert resolved["settings"]["epochs"] == expected.epochs


def test_explicit_overrides_win_over_auto_profile_defaults():
    resolved = resolve_loop_profile(
        "hexagonal",
        2,
        "auto",
        games_per_iter=12,
        eval_games=6,
        budget=24,
    )

    assert resolved["profile"] == "large-board"
    assert resolved["settings"]["games_per_iter"] == 12
    assert resolved["settings"]["eval_games"] == 6
    assert resolved["settings"]["budget"] == 24


def test_transfer_hint_prefers_same_board_strong_4p_checkpoint():
    assert recommend_transfer_source("square19", 2) == "ringrift_best_square19_4p.pth"
    assert recommend_transfer_source("hexagonal", 3) == "ringrift_best_hexagonal_4p.pth"
    assert recommend_transfer_source("hex8", 2) is None
