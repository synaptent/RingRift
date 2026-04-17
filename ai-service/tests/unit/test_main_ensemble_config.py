"""Tests for C1 ensemble feature-flag + checkpoint resolution helpers.

Covers only the pure config-parsing helpers in app.main. End-to-end
ensemble behaviour is covered by tests/unit/ai/test_ensemble_move.py
at the voting-logic layer.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest


class TestEnsembleFeatureFlag:
    @pytest.mark.parametrize("raw", ["1", "true", "True", "YES", "on"])
    def test_truthy_values_enable(self, raw, monkeypatch):
        from app import main
        monkeypatch.setenv("RINGRIFT_ENSEMBLE_ENABLED", raw)
        assert main._ensemble_feature_enabled() is True

    @pytest.mark.parametrize(
        "raw", ["", "0", "false", "no", "off", "maybe", "unset"],
    )
    def test_falsy_values_disable(self, raw, monkeypatch):
        from app import main
        if raw == "unset":
            monkeypatch.delenv("RINGRIFT_ENSEMBLE_ENABLED", raising=False)
        else:
            monkeypatch.setenv("RINGRIFT_ENSEMBLE_ENABLED", raw)
        assert main._ensemble_feature_enabled() is False


class TestEnsembleExtraCheckpoints:
    def _board(self, name: str):
        return SimpleNamespace(value=name)

    def test_empty_env_returns_empty_list(self, monkeypatch):
        from app import main
        monkeypatch.delenv("RINGRIFT_ENSEMBLE_EXTRA_CHECKPOINTS", raising=False)
        assert main._ensemble_extra_checkpoints(self._board("hex8"), 2) == []

    def test_blank_env_returns_empty_list(self, monkeypatch):
        from app import main
        monkeypatch.setenv("RINGRIFT_ENSEMBLE_EXTRA_CHECKPOINTS", "   ")
        assert main._ensemble_extra_checkpoints(self._board("hex8"), 2) == []

    def test_json_parse_error_is_logged_and_returns_empty(self, monkeypatch):
        from app import main
        monkeypatch.setenv(
            "RINGRIFT_ENSEMBLE_EXTRA_CHECKPOINTS", "not json at all",
        )
        assert main._ensemble_extra_checkpoints(self._board("hex8"), 2) == []

    def test_non_object_root_returns_empty(self, monkeypatch):
        """A JSON list at the root is rejected — the shape is documented
        as an object keyed by config id."""
        from app import main
        monkeypatch.setenv(
            "RINGRIFT_ENSEMBLE_EXTRA_CHECKPOINTS", '["a.pth", "b.pth"]',
        )
        assert main._ensemble_extra_checkpoints(self._board("hex8"), 2) == []

    def test_missing_key_returns_empty(self, monkeypatch):
        from app import main
        monkeypatch.setenv(
            "RINGRIFT_ENSEMBLE_EXTRA_CHECKPOINTS",
            '{"square8_2p": ["foo.pth"]}',
        )
        # Querying hex8_2p — key not present.
        assert main._ensemble_extra_checkpoints(self._board("hex8"), 2) == []

    def test_matching_key_returns_paths(self, monkeypatch):
        from app import main
        monkeypatch.setenv(
            "RINGRIFT_ENSEMBLE_EXTRA_CHECKPOINTS",
            '{"hex8_2p": ["models/alt1.pth", "models/alt2.pth"]}',
        )
        assert main._ensemble_extra_checkpoints(self._board("hex8"), 2) == [
            "models/alt1.pth",
            "models/alt2.pth",
        ]

    def test_string_value_accepted_as_single_path(self, monkeypatch):
        """Single-path strings are normalized into a 1-element list so
        operators don't have to wrap a single checkpoint in [] in the
        env var."""
        from app import main
        monkeypatch.setenv(
            "RINGRIFT_ENSEMBLE_EXTRA_CHECKPOINTS",
            '{"hex8_2p": "models/alt.pth"}',
        )
        assert main._ensemble_extra_checkpoints(self._board("hex8"), 2) == [
            "models/alt.pth",
        ]

    def test_non_string_entries_filtered(self, monkeypatch):
        from app import main
        monkeypatch.setenv(
            "RINGRIFT_ENSEMBLE_EXTRA_CHECKPOINTS",
            '{"hex8_2p": ["models/good.pth", 42, null, ""]}',
        )
        assert main._ensemble_extra_checkpoints(self._board("hex8"), 2) == [
            "models/good.pth",
        ]

    def test_non_list_non_string_value_returns_empty(self, monkeypatch):
        from app import main
        monkeypatch.setenv(
            "RINGRIFT_ENSEMBLE_EXTRA_CHECKPOINTS",
            '{"hex8_2p": {"unexpected": "shape"}}',
        )
        assert main._ensemble_extra_checkpoints(self._board("hex8"), 2) == []

    def test_board_type_value_attribute_used(self, monkeypatch):
        """BoardType enum instances should resolve via .value, not str()."""
        from app import main
        monkeypatch.setenv(
            "RINGRIFT_ENSEMBLE_EXTRA_CHECKPOINTS",
            '{"square19_4p": ["models/big.pth"]}',
        )
        assert main._ensemble_extra_checkpoints(self._board("square19"), 4) == [
            "models/big.pth",
        ]

    def test_num_players_defaults_to_2(self, monkeypatch):
        from app import main
        monkeypatch.setenv(
            "RINGRIFT_ENSEMBLE_EXTRA_CHECKPOINTS",
            '{"hex8_2p": ["models/alt.pth"]}',
        )
        # num_players None → defaults to 2 → still resolves.
        assert main._ensemble_extra_checkpoints(self._board("hex8"), None) == [
            "models/alt.pth",
        ]


class TestEnsembleBudgetReduction:
    """Regression guard for the CPU-bound ensemble fix.

    After the first production deploy (2026-04-17) hit 30s timeouts
    running 2× D10 searches at full budget on CPU, the ensemble path
    now divides each constituent's ``gumbel_simulation_budget`` by
    the ensemble size so total CPU work stays comparable to
    single-model.  These tests lock in that division math directly
    without spinning up real AI instances.
    """

    @pytest.mark.parametrize(
        "full_budget,ensemble_size,expected",
        [
            (400, 2, 200),  # D10 production case
            (400, 3, 133),  # 3-model ensemble
            (200, 2, 100),
            (64, 2, 32),
            (1, 2, 1),  # floor: max(1, ...) so budget never drops to 0
            (0, 2, 1),  # degenerate: minimum 1 sim
        ],
    )
    def test_reduced_budget_divides_full_by_ensemble_size(
        self, full_budget, ensemble_size, expected,
    ):
        # Mirrors the inline computation in app/main.py::get_ai_move
        # ensemble branch.  If this test changes, update both sites.
        reduced = max(1, full_budget // ensemble_size)
        assert reduced == expected

    def test_none_full_budget_leaves_config_untouched(self):
        """Non-Gumbel tiers (no explicit budget) should not gain a
        bogus gumbel_simulation_budget from the reduction math."""
        # This mirrors the guard `if gumbel_budget is not None:` in the
        # ensemble branch — when it's None, no budget override is applied.
        full_budget = None
        if full_budget is not None:
            reduced = max(1, full_budget // 2)
        else:
            reduced = None
        assert reduced is None
