"""Tests for eval-progress derivation in training_status."""

from __future__ import annotations

import pytest

from scripts.training_status import EVAL_STAGE_TARGETS, _derive_eval_progress


class TestDeriveEvalProgressBasic:
    def test_none_checkpoint_returns_none(self) -> None:
        assert _derive_eval_progress(None) is None

    def test_non_dict_returns_none(self) -> None:
        assert _derive_eval_progress("not a dict") is None  # type: ignore[arg-type]
        assert _derive_eval_progress([]) is None  # type: ignore[arg-type]

    def test_malformed_numeric_fields_returns_none(self) -> None:
        ckpt = {"candidate_wins": "not-a-number"}
        assert _derive_eval_progress(ckpt) is None

    def test_empty_checkpoint_yields_stage_1_with_zero_games(self) -> None:
        progress = _derive_eval_progress({})
        assert progress is not None
        assert progress["games_played"] == 0
        assert progress["current_stage"] == 1
        assert progress["next_stage_target"] == EVAL_STAGE_TARGETS[0]
        assert progress["running_wr"] is None
        assert progress["seat_games"] is None


class TestStageDeduction:
    def test_mid_stage_1(self) -> None:
        ckpt = {"candidate_wins": 10, "best_wins": 15, "games_played": 25}
        progress = _derive_eval_progress(ckpt)
        assert progress is not None
        assert progress["current_stage"] == 1
        assert progress["next_stage_target"] == 50

    def test_exactly_at_stage_1_boundary(self) -> None:
        ckpt = {"candidate_wins": 25, "best_wins": 25, "games_played": 50}
        progress = _derive_eval_progress(ckpt)
        assert progress is not None
        assert progress["current_stage"] == 2
        assert progress["next_stage_target"] == 100

    def test_mid_stage_2(self) -> None:
        ckpt = {"candidate_wins": 40, "best_wins": 35, "games_played": 75}
        progress = _derive_eval_progress(ckpt)
        assert progress is not None
        assert progress["current_stage"] == 2
        assert progress["next_stage_target"] == 100

    def test_final_stage(self) -> None:
        ckpt = {"candidate_wins": 180, "best_wins": 160, "games_played": 340}
        progress = _derive_eval_progress(ckpt)
        assert progress is not None
        assert progress["current_stage"] == 4
        # Past stage 3 boundary, in final stage
        assert progress["next_stage_target"] == 400

    def test_past_final_stage(self) -> None:
        ckpt = {"candidate_wins": 210, "best_wins": 190, "games_played": 400}
        progress = _derive_eval_progress(ckpt)
        assert progress is not None
        assert progress["current_stage"] == 4


class TestRunningWinRate:
    def test_no_decided_games_yields_none(self) -> None:
        ckpt = {"candidate_wins": 0, "best_wins": 0, "draws": 3, "games_played": 3}
        progress = _derive_eval_progress(ckpt)
        assert progress is not None
        assert progress["running_wr"] is None

    def test_draws_excluded_from_wr_denominator(self) -> None:
        ckpt = {"candidate_wins": 6, "best_wins": 4, "draws": 2, "games_played": 12}
        progress = _derive_eval_progress(ckpt)
        assert progress is not None
        # 6 / (6 + 4) = 0.6, draws not in denominator
        assert progress["running_wr"] == 0.6

    def test_rounded_to_three_decimals(self) -> None:
        ckpt = {"candidate_wins": 7, "best_wins": 8, "games_played": 15}
        progress = _derive_eval_progress(ckpt)
        assert progress is not None
        assert progress["running_wr"] == round(7 / 15, 3)


class TestSeatAggregation:
    def test_no_seat_outcomes_yields_none_seat_fields(self) -> None:
        ckpt = {"candidate_wins": 5, "best_wins": 5, "games_played": 10}
        progress = _derive_eval_progress(ckpt)
        assert progress is not None
        assert progress["seat_games"] is None
        assert progress["seat_wr_partial"] is None
        assert progress["seat_imbalance_ratio"] is None

    def test_balanced_seat_play(self) -> None:
        # 10 games, candidate plays seats 1/2/3 evenly, wins some
        seat_outcomes = [
            {"i": i, "candidate_player": (i % 3) + 1, "won": (i % 2 == 0)}
            for i in range(9)
        ]
        ckpt = {
            "candidate_wins": 5,
            "best_wins": 4,
            "games_played": 9,
            "seat_outcomes": seat_outcomes,
        }
        progress = _derive_eval_progress(ckpt)
        assert progress is not None
        assert progress["seat_games"] == {1: 3, 2: 3, 3: 3}
        # seats 1,2,3 each played 3 games; wins depend on (i%2==0) pattern
        assert set(progress["seat_wr_partial"].keys()) == {1, 2, 3}
        # Imbalance ratio should be computed when all seats have some wins
        assert progress["seat_imbalance_ratio"] is not None

    def test_heavy_seat_imbalance_detected(self) -> None:
        # Candidate wins 90% in seat 1, 10% in seat 2 — 9x imbalance
        seat_outcomes = [{"i": i, "candidate_player": 1, "won": True} for i in range(9)]
        seat_outcomes.append({"i": 9, "candidate_player": 1, "won": False})
        seat_outcomes.extend(
            [{"i": 10 + i, "candidate_player": 2, "won": False} for i in range(9)]
        )
        seat_outcomes.append({"i": 19, "candidate_player": 2, "won": True})
        ckpt = {
            "candidate_wins": 10,
            "best_wins": 10,
            "games_played": 20,
            "seat_outcomes": seat_outcomes,
        }
        progress = _derive_eval_progress(ckpt)
        assert progress is not None
        assert progress["seat_wr_partial"][1] == 0.9
        assert progress["seat_wr_partial"][2] == 0.1
        assert progress["seat_imbalance_ratio"] == 9.0

    def test_zero_wr_seat_does_not_divide_by_zero(self) -> None:
        seat_outcomes = [
            {"i": 0, "candidate_player": 1, "won": True},
            {"i": 1, "candidate_player": 2, "won": False},
        ]
        ckpt = {
            "candidate_wins": 1,
            "best_wins": 1,
            "games_played": 2,
            "seat_outcomes": seat_outcomes,
        }
        progress = _derive_eval_progress(ckpt)
        assert progress is not None
        # seat 2 has 0 wins, min wr = 0 → ratio not computable, should be None
        assert progress["seat_imbalance_ratio"] is None

    def test_malformed_seat_outcome_entries_ignored(self) -> None:
        seat_outcomes = [
            {"i": 0, "candidate_player": 1, "won": True},
            "not a dict",
            {"missing_candidate_player": True},
            {"i": 1, "candidate_player": "not-int", "won": False},
            {"i": 2, "candidate_player": 2, "won": False},
        ]
        ckpt = {
            "candidate_wins": 1,
            "best_wins": 1,
            "games_played": 2,
            "seat_outcomes": seat_outcomes,
        }
        progress = _derive_eval_progress(ckpt)
        assert progress is not None
        assert progress["seat_games"] == {1: 1, 2: 1}
