"""Tests for training effectiveness probes (TEP)."""
from __future__ import annotations

import math
import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from scripts.lib.training_probes import (
    ProbeResult,
    _loss_convergence_check,
    _weight_delta_check,
    run_training_probes,
)


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
