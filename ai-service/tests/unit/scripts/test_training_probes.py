"""Tests for training effectiveness probes (TEP)."""
from __future__ import annotations

import math
import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from scripts.lib.training_probes import (
    ProbeResult,
    _check_probe_value_head_health,
    _inference_probe,
    _loss_convergence_check,
    _weight_delta_check,
    run_training_probes,
)

pytestmark = pytest.mark.timeout(30)


# ---------------------------------------------------------------------------
# ProbeResult tests
# ---------------------------------------------------------------------------

class TestProbeResult:
    def test_default_is_passing(self):
        r = ProbeResult()
        assert not r.critical
        assert r.warnings == []
        assert r.summary == "all probes passed"

    def test_critical_summary(self):
        r = ProbeResult(critical=True, warnings=["Zero gradient effect"])
        assert "CRITICAL" in r.summary
        assert "Zero gradient" in r.summary

    def test_warnings_only_summary(self):
        r = ProbeResult(warnings=["Low entropy"])
        assert not r.critical
        assert "Low entropy" in r.summary


# ---------------------------------------------------------------------------
# Loss convergence tests
# ---------------------------------------------------------------------------

class TestLossConvergenceCheck:
    def test_no_loss_data_is_not_critical(self):
        """When training output has no parseable loss lines, it's not critical."""
        crit, warns, details = _loss_convergence_check({"elapsed_s": 42.0})
        assert not crit
        assert "note" in details

    def test_detects_nan_loss(self):
        """NaN in final loss should be critical."""
        crit, warns, details = _loss_convergence_check({
            "last_epoch_line": "Epoch [15/15], Train Loss: nan, Val Loss: nan, Policy Acc: 0.0%"
        })
        assert crit
        assert any("NaN" in w for w in warns)
        assert details.get("has_nan") is True

    def test_parses_epoch_line_losses(self):
        """Should extract Train and Val loss from a standard epoch log line."""
        crit, warns, details = _loss_convergence_check({
            "last_epoch_line": "Epoch [15/15], Train Loss: 0.3456, Val Loss: 0.4567, Policy Acc: 62.3%"
        })
        assert not crit
        assert "parsed_losses" in details
        assert len(details["parsed_losses"]) == 2
        assert abs(details["parsed_losses"][0] - 0.3456) < 1e-4
        assert abs(details["parsed_losses"][1] - 0.4567) < 1e-4


class TestProbeValueHeadHealth:
    def test_nonfinite_values_are_critical(self):
        crit, warns, details = _check_probe_value_head_health([0.1, float("nan"), 0.2])
        assert crit
        assert any("NONFINITE_VALUE_HEAD" in w for w in warns)
        assert details["nonfinite_value_samples"] == 1

    def test_dead_value_head_is_critical(self):
        crit, warns, details = _check_probe_value_head_health([0.5] * 6)
        assert crit
        assert any("DEAD_VALUE_HEAD" in w for w in warns)
        assert details["value_std"] == 0.0

    def test_saturated_probe_values_are_critical(self):
        crit, warns, details = _check_probe_value_head_health(
            [-0.999, -0.998, -1.0, 0.999, 1.0, 0.998]
        )
        assert crit
        assert any("SATURATED_VALUE_HEAD" in w for w in warns)
        assert details["saturated_value_ratio"] >= 0.9

    def test_healthy_probe_values_pass(self):
        crit, warns, details = _check_probe_value_head_health(
            [-0.6, -0.2, 0.0, 0.15, 0.3, 0.7]
        )
        assert not crit
        assert warns == []
        assert details["value_samples"] == 6


# ---------------------------------------------------------------------------
# Weight delta tests
# ---------------------------------------------------------------------------

class TestWeightDeltaCheck:
    def _save_state_dict(self, path: str, sd: dict):
        import torch
        torch.save(sd, path)

    def test_identical_weights_is_critical(self):
        """If candidate == best (zero delta), should be critical failure."""
        import torch
        with tempfile.TemporaryDirectory() as td:
            sd = {"layer.weight": torch.randn(10, 10), "layer.bias": torch.randn(10)}
            cand_path = os.path.join(td, "candidate.pth")
            best_path = os.path.join(td, "best.pth")
            self._save_state_dict(cand_path, sd)
            self._save_state_dict(best_path, sd)

            crit, warns, details = _weight_delta_check(cand_path, best_path)
            assert crit
            assert details["weight_delta_l2"] < 1e-8
            assert any("Zero gradient" in w for w in warns)

    def test_normal_delta_passes(self):
        """Moderate weight changes should pass without issues."""
        import torch
        with tempfile.TemporaryDirectory() as td:
            sd_best = {"layer.weight": torch.randn(10, 10), "layer.bias": torch.randn(10)}
            sd_cand = {
                "layer.weight": sd_best["layer.weight"] + torch.randn(10, 10) * 0.01,
                "layer.bias": sd_best["layer.bias"] + torch.randn(10) * 0.01,
            }
            cand_path = os.path.join(td, "candidate.pth")
            best_path = os.path.join(td, "best.pth")
            self._save_state_dict(cand_path, sd_cand)
            self._save_state_dict(best_path, sd_best)

            crit, warns, details = _weight_delta_check(cand_path, best_path)
            assert not crit
            assert details["weight_delta_l2"] > 1e-8

    def test_large_delta_warns(self):
        """Very large weight delta should produce a warning."""
        import torch
        with tempfile.TemporaryDirectory() as td:
            sd_best = {"layer.weight": torch.zeros(10, 10)}
            sd_cand = {"layer.weight": torch.ones(10, 10) * 5.0}
            cand_path = os.path.join(td, "candidate.pth")
            best_path = os.path.join(td, "best.pth")
            self._save_state_dict(cand_path, sd_cand)
            self._save_state_dict(best_path, sd_best)

            crit, warns, details = _weight_delta_check(cand_path, best_path)
            assert not crit
            assert details["weight_delta_l2"] > 10.0
            assert any("divergence" in w for w in warns)

    def test_handles_model_state_dict_wrapper(self):
        """Models saved with model_state_dict wrapper should be unpacked."""
        import torch
        with tempfile.TemporaryDirectory() as td:
            inner = {"layer.weight": torch.randn(10, 10)}
            wrapped = {"model_state_dict": inner, "optimizer": {}}
            cand_path = os.path.join(td, "candidate.pth")
            best_path = os.path.join(td, "best.pth")
            self._save_state_dict(cand_path, wrapped)
            self._save_state_dict(best_path, {"model_state_dict": inner.copy()})

            crit, warns, details = _weight_delta_check(cand_path, best_path)
            # Identical inner dicts → zero delta → critical
            assert crit

    def test_missing_file_is_critical(self):
        """Missing model file should be critical."""
        crit, warns, details = _weight_delta_check("/nonexistent/a.pth", "/nonexistent/b.pth")
        assert crit
        assert "error" in details


# ---------------------------------------------------------------------------
# Integration: run_training_probes
# ---------------------------------------------------------------------------

class TestRunTrainingProbes:
    def test_all_probes_pass_with_mocked_inference(self):
        """With normal weight delta and no inference issues, result should pass."""
        import torch
        with tempfile.TemporaryDirectory() as td:
            sd_best = {"layer.weight": torch.randn(10, 10)}
            sd_cand = {
                "layer.weight": sd_best["layer.weight"] + torch.randn(10, 10) * 0.01,
            }
            cand_path = os.path.join(td, "candidate.pth")
            best_path = os.path.join(td, "best.pth")
            torch.save(sd_cand, cand_path)
            torch.save(sd_best, best_path)

            train_info = {
                "elapsed_s": 120.0,
                "last_epoch_line": "Epoch [15/15], Train Loss: 0.34, Val Loss: 0.45, Policy Acc: 60.0%",
            }

            # Mock the inference probe to avoid needing a real model
            with patch("scripts.lib.training_probes._inference_probe") as mock_inf:
                mock_inf.return_value = (False, [], {"moves_played": 10, "inference_ok": True})
                from app.models import BoardType
                result = run_training_probes(
                    cand_path, best_path, train_info,
                    BoardType.HEX8, 2, 128,
                )
            assert not result.critical
            assert result.elapsed_s >= 0

    def test_critical_weight_delta_skips_inference(self):
        """If weight delta is critical, inference probe should be skipped."""
        import torch
        with tempfile.TemporaryDirectory() as td:
            sd = {"layer.weight": torch.randn(10, 10)}
            cand_path = os.path.join(td, "candidate.pth")
            best_path = os.path.join(td, "best.pth")
            torch.save(sd, cand_path)
            torch.save(sd, best_path)

            train_info = {"elapsed_s": 10.0}

            with patch("scripts.lib.training_probes._inference_probe") as mock_inf:
                from app.models import BoardType
                result = run_training_probes(
                    cand_path, best_path, train_info,
                    BoardType.HEX8, 2, 128,
                )
            # Inference should NOT have been called
            mock_inf.assert_not_called()
            assert result.critical
            assert "inference" not in result.details


class TestInferenceProbeRootValueHealth:
    def test_inference_probe_uses_root_value_stats(self):
        import app.ai.gumbel_mcts_ai as gumbel_module
        import app.training.env as env_module
        from app.models import BoardType, GameStatus

        root_values = iter([-0.999, -0.998, -1.0, 0.999, 1.0, 0.998])

        class FakeAI:
            def __init__(self, _player, _cfg, _board_type):
                self.player_number = 1
                self._last_search_actions = []
                self._last_search_stats = {}

            def select_move(self, _state):
                self._last_search_actions = [
                    MagicMock(visit_count=8),
                    MagicMock(visit_count=4),
                ]
                self._last_search_stats = {
                    "root_value": next(root_values),
                    "heuristic_fallback": False,
                }
                return "move"

        class FakeState:
            def __init__(self, done=False):
                self.game_status = GameStatus.COMPLETED if done else GameStatus.ACTIVE
                self.current_player = 1

        class FakeEnv:
            def __init__(self):
                self._steps = 0

            def reset(self, seed=None):
                return FakeState(done=False)

            def legal_moves(self):
                return ["move"]

            def step(self, _move):
                self._steps += 1
                done = self._steps >= 6
                return FakeState(done=done), 0.0, done, {}

        def _fake_make_env(_cfg):
            return FakeEnv()

        def _fake_tmax(_board_type, _num_players):
            return 100

        class _FakeEnvCfg:
            def __init__(self, **_kwargs):
                pass

        with patch.object(gumbel_module, "GumbelMCTSAI", FakeAI), \
             patch.object(env_module, "make_env", _fake_make_env), \
             patch.object(env_module, "get_theoretical_max_moves", _fake_tmax), \
             patch.object(env_module, "TrainingEnvConfig", _FakeEnvCfg):
            critical, warns, details = _inference_probe(
                "/tmp/candidate.pth",
                BoardType.HEX8,
                2,
                128,
                model_version="v5-heavy",
            )

        assert critical
        assert any("SATURATED_VALUE_HEAD" in w for w in warns)
        assert details["value_samples"] == 6
        assert details["saturated_value_ratio"] >= 0.9


# ---------------------------------------------------------------------------
# Edge case tests
# ---------------------------------------------------------------------------

class TestLossConvergenceEdgeCases:
    """Edge cases for loss convergence parsing."""

    def test_scientific_notation_loss(self):
        """Should parse losses in scientific notation (e.g., 1.23e-04)."""
        crit, warns, details = _loss_convergence_check({
            "last_epoch_line": "Epoch [5/5], Train Loss: 1.23e-04, Val Loss: 2.45e-03"
        })
        assert not crit
        assert len(details["parsed_losses"]) == 2
        assert abs(details["parsed_losses"][0] - 1.23e-4) < 1e-8
        assert abs(details["parsed_losses"][1] - 2.45e-3) < 1e-8

    def test_nan_in_train_but_not_val(self):
        """Should detect NaN even if only Train Loss is NaN."""
        crit, warns, details = _loss_convergence_check({
            "last_epoch_line": "Epoch [5/5], Train Loss: nan, Val Loss: 0.45"
        })
        assert crit
        assert details.get("has_nan") is True

    def test_empty_dict_input(self):
        """Empty training info dict should not crash."""
        crit, warns, details = _loss_convergence_check({})
        assert not crit
        assert "note" in details

    def test_multiple_epoch_lines(self):
        """Handles both last_epoch_line and log_line simultaneously."""
        crit, warns, details = _loss_convergence_check({
            "last_epoch_line": "Epoch [15/15], Train Loss: 0.34, Val Loss: 0.45",
            "log_line": "Epoch [10/15], Train Loss: 0.50, Val Loss: 0.60",
        })
        assert not crit
        assert len(details["parsed_losses"]) == 4

    def test_non_string_values_in_info(self):
        """Non-string values for epoch line keys should be handled."""
        crit, warns, details = _loss_convergence_check({
            "last_epoch_line": 12345,  # not a string
            "log_line": None,
        })
        assert not crit
        assert "note" in details


class TestWeightDeltaEdgeCases:
    """Edge cases for weight delta checks."""

    def test_no_common_keys(self):
        """Models with completely different key names should be critical."""
        import torch
        with tempfile.TemporaryDirectory() as td:
            sd_cand = {"encoder.weight": torch.randn(10, 10)}
            sd_best = {"decoder.weight": torch.randn(10, 10)}
            cand_path = os.path.join(td, "candidate.pth")
            best_path = os.path.join(td, "best.pth")
            torch.save(sd_cand, cand_path)
            torch.save(sd_best, best_path)

            crit, warns, details = _weight_delta_check(cand_path, best_path)
            assert crit
            assert "No common parameters" in details.get("error", "")

    def test_shape_mismatch_keys_skipped(self):
        """Keys with different shapes should be skipped, not crash."""
        import torch
        with tempfile.TemporaryDirectory() as td:
            sd_cand = {
                "layer1.weight": torch.randn(10, 10),
                "layer2.weight": torch.randn(20, 20),  # different shape
            }
            sd_best = {
                "layer1.weight": torch.randn(10, 10),
                "layer2.weight": torch.randn(15, 15),  # different shape from cand
            }
            cand_path = os.path.join(td, "candidate.pth")
            best_path = os.path.join(td, "best.pth")
            torch.save(sd_cand, cand_path)
            torch.save(sd_best, best_path)

            crit, warns, details = _weight_delta_check(cand_path, best_path)
            # Should compare only layer1.weight (common shape)
            assert details["params_compared"] == 1


# ---------------------------------------------------------------------------
# A3: model_version propagation regression test
# ---------------------------------------------------------------------------
#
# The v4 experiment on gh200-8 was silently stalled for ~20 hours because
# the training probe did not pass --model-version through to the inference
# probe's AIConfig. That bug was fixed in commit beafb4a07 by threading
# `model_version` through run_training_probes -> _inference_probe -> AIConfig.
# This test locks in the propagation contract so a future refactor cannot
# reintroduce the same silent failure for v4 / v5-heavy / future non-v2
# architectures.


def _run_probes_with_stub_ai(model_version, feature_version=None):
    """Invoke run_training_probes with stubbed AI/env and return the
    AIConfig that _inference_probe constructed.

    _inference_probe's imports are resolved lazily from the real modules,
    so we patch the `GumbelMCTSAI` / `make_env` / `get_theoretical_max_moves`
    attributes on those modules in-place.  Everything else in the probe
    chain runs for real against our fakes.
    """
    import torch

    import app.ai.gumbel_mcts_ai as gumbel_module
    import app.training.env as env_module
    import scripts.lib.training_probes as tp
    from app.models import GameStatus

    captured: dict = {}

    class FakeAI:
        def __init__(self, _player, cfg, _board_type):
            captured["config"] = cfg
            self._last_search_actions = []
            self._last_search_stats = {}
            self.player_number = 1

        def select_move(self, _state):
            return None

        def reset_for_new_game(self, **_kwargs):
            pass

    class _FakeState:
        game_status = GameStatus.ACTIVE
        current_player = 1

    class FakeEnv:
        num_players = 2

        def reset(self, seed=None):
            return _FakeState()

        def legal_moves(self):
            return []

        def step(self, _move):
            s = _FakeState()
            s.game_status = GameStatus.COMPLETED
            return s, 0.0, True, {}

    def _fake_make_env(_cfg):
        return FakeEnv()

    def _fake_tmax(_board_type, _num_players):
        return 100

    class _FakeEnvCfg:
        def __init__(self, **_kwargs):
            pass

    with patch.object(gumbel_module, "GumbelMCTSAI", FakeAI), \
         patch.object(env_module, "make_env", _fake_make_env), \
         patch.object(env_module, "get_theoretical_max_moves", _fake_tmax), \
         patch.object(env_module, "TrainingEnvConfig", _FakeEnvCfg):
        with tempfile.TemporaryDirectory() as td:
            c = os.path.join(td, "candidate.pth")
            b = os.path.join(td, "best.pth")
            # Use distinct random weights so _weight_delta_check produces a
            # non-zero L2.  Identical state dicts would collapse to L2=0,
            # trigger the "zero gradient effect" critical path, and skip
            # the inference probe — silently hiding the very behaviour
            # this test is supposed to exercise.
            torch.save({"layer.weight": torch.randn(4, 4)}, c)
            torch.save({"layer.weight": torch.randn(4, 4)}, b)
            kwargs = {}
            if model_version is not _UNSET:
                kwargs["model_version"] = model_version
            if feature_version is not None:
                kwargs["feature_version"] = feature_version
            tp.run_training_probes(
                c, b,
                {"last_epoch_line": "Epoch 1, Train Loss: 0.5"},
                MagicMock(),
                2, 16,
                **kwargs,
            )

    return captured.get("config")


class _UnsetSentinel:
    pass


_UNSET = _UnsetSentinel()


class TestModelVersionPropagation:
    """run_training_probes must forward model_version to AIConfig."""

    def test_v4_threads_version_to_aiconfig(self):
        cfg = _run_probes_with_stub_ai("v4")
        assert cfg is not None, "FakeAI was never constructed — probe chain broke"
        assert getattr(cfg, "nn_model_version", None) == "v4"
        assert cfg.nn_model_id.endswith("candidate.pth")
        assert cfg.allow_fresh_weights is False
        assert cfg.use_neural_net is True

    def test_v5_heavy_threads_version_to_aiconfig(self):
        cfg = _run_probes_with_stub_ai("v5-heavy")
        assert cfg is not None
        assert getattr(cfg, "nn_model_version", None) == "v5-heavy"

    def test_v5_heavy_large_threads_version_to_aiconfig(self):
        cfg = _run_probes_with_stub_ai("v5-heavy-large")
        assert cfg is not None
        assert getattr(cfg, "nn_model_version", None) == "v5-heavy-large"

    def test_v2_default_is_none(self):
        """nn_model_version should be None for v2 so legacy AIConfig call
        sites keep working. This was the "pass through when not v2"
        convention used in _make_ai throughout minimal_alphazero_loop.py."""
        cfg = _run_probes_with_stub_ai("v2")
        assert cfg is not None
        assert getattr(cfg, "nn_model_version", None) is None

    def test_missing_model_version_is_none(self):
        """Omitting model_version entirely must behave the same as v2."""
        cfg = _run_probes_with_stub_ai(_UNSET)
        assert cfg is not None
        assert getattr(cfg, "nn_model_version", None) is None

    def test_feature_version_threads_to_aiconfig(self):
        cfg = _run_probes_with_stub_ai("v5-heavy", feature_version=3)
        assert cfg is not None
        assert getattr(cfg, "nn_model_version", None) == "v5-heavy"
        assert getattr(cfg, "feature_version", None) == 3
