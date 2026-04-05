"""Tests for the loop self-healing module.

Verifies failure pattern classification, recovery actions, budget enforcement,
and the attempt_recovery coordinator.
"""
from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from scripts.lib.loop_self_healing import (
    FailureContext,
    FailurePattern,
    RecoveryResult,
    attempt_recovery,
    classify_failure,
    reset_recovery_counts,
    _recover_oom,
    _recover_identical_data,
    _recover_dead_model,
    _recover_arch_mismatch,
    _no_auto_fix,
    MAX_RECOVERIES_PER_PATTERN,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ctx(
    error_message: str = "",
    stage: str = "training",
    config_key: str = "hex8_2p",
    work_dir: str = "/tmp/test_loop",
    model_path: str = "/tmp/test_loop/models/best.pth",
    batch_size: int = 512,
    selfplay_randomness: float = 0.25,
) -> FailureContext:
    return FailureContext(
        error_message=error_message,
        stage=stage,
        config_key=config_key,
        work_dir=work_dir,
        model_path=model_path,
        batch_size=batch_size,
        selfplay_randomness=selfplay_randomness,
    )


@pytest.fixture(autouse=True)
def _reset_counts():
    """Reset recovery counts before each test."""
    reset_recovery_counts()
    yield
    reset_recovery_counts()


# ---------------------------------------------------------------------------
# 1. Failure Pattern Classification
# ---------------------------------------------------------------------------

class TestClassifyFailure:
    def test_oom_cuda_out_of_memory(self):
        ctx = _make_ctx(error_message="RuntimeError: CUDA out of memory. Tried to allocate 2GB")
        assert classify_failure(ctx) == FailurePattern.OOM

    def test_oom_generic(self):
        ctx = _make_ctx(error_message="torch.cuda.OutOfMemoryError: out of memory")
        assert classify_failure(ctx) == FailurePattern.OOM

    def test_identical_data(self):
        ctx = _make_ctx(
            error_message="CRITICAL -- Training data is near-identical to a recent iteration"
        )
        assert classify_failure(ctx) == FailurePattern.IDENTICAL_DATA

    def test_dead_model_zero_gradient(self):
        ctx = _make_ctx(error_message="CRITICAL; Zero gradient effect: weight delta L2=0.00e+00")
        assert classify_failure(ctx) == FailurePattern.DEAD_MODEL

    def test_dead_model_heuristic_fallback(self):
        ctx = _make_ctx(error_message="Heuristic fallback triggered 10/10 times")
        assert classify_failure(ctx) == FailurePattern.DEAD_MODEL

    def test_arch_mismatch_encoding(self):
        ctx = _make_ctx(
            error_message="ENCODING MISMATCH: NPZ has 14ch but contract expects 56ch"
        )
        assert classify_failure(ctx) == FailurePattern.ARCH_MISMATCH

    def test_arch_mismatch_encoder(self):
        ctx = _make_ctx(error_message="encoder mismatch detected for model")
        assert classify_failure(ctx) == FailurePattern.ARCH_MISMATCH

    def test_arch_mismatch_channel(self):
        ctx = _make_ctx(error_message="channel mismatch: model has 40 but data has 56")
        assert classify_failure(ctx) == FailurePattern.ARCH_MISMATCH

    def test_unknown_error(self):
        ctx = _make_ctx(error_message="Something completely unexpected happened")
        assert classify_failure(ctx) == FailurePattern.UNKNOWN

    def test_empty_error(self):
        ctx = _make_ctx(error_message="")
        assert classify_failure(ctx) == FailurePattern.UNKNOWN


# ---------------------------------------------------------------------------
# 2. Recovery Actions
# ---------------------------------------------------------------------------

class TestRecoverOOM:
    def test_halves_batch_size(self):
        ctx = _make_ctx(batch_size=512)
        result = _recover_oom(ctx)
        assert result.recovered
        assert result.action == "retry_smaller_batch"
        assert result.adjustments["batch_size"] == 256

    def test_cannot_reduce_below_minimum(self):
        ctx = _make_ctx(batch_size=32)
        result = _recover_oom(ctx)
        assert not result.recovered
        assert "minimum" in result.message.lower() or "cannot reduce" in result.message.lower()


class TestRecoverIdenticalData:
    def test_increases_randomness(self):
        with tempfile.TemporaryDirectory() as td:
            # Create a fake NPZ file so it can be deleted
            npz_path = Path(td) / "iter_001.npz"
            npz_path.write_bytes(b"fake")

            ctx = _make_ctx(
                work_dir=td,
                selfplay_randomness=0.25,
            )
            result = _recover_identical_data(ctx)
            assert result.recovered
            assert result.action == "retry_higher_randomness"
            assert result.adjustments["selfplay_randomness"] == pytest.approx(0.35)
            # NPZ should have been deleted
            assert not npz_path.exists()

    def test_caps_randomness_at_max(self):
        ctx = _make_ctx(selfplay_randomness=0.5)
        result = _recover_identical_data(ctx)
        assert not result.recovered
        assert "max" in result.message.lower() or "cannot increase" in result.message.lower()

    def test_handles_no_npz_files(self):
        with tempfile.TemporaryDirectory() as td:
            ctx = _make_ctx(work_dir=td, selfplay_randomness=0.2)
            result = _recover_identical_data(ctx)
            assert result.recovered
            assert result.adjustments["selfplay_randomness"] == pytest.approx(0.3)


class TestRecoverDeadModel:
    def test_rolls_back_from_canonical(self):
        with tempfile.TemporaryDirectory() as td:
            models_dir = Path(td) / "models"
            models_dir.mkdir()
            best = models_dir / "best.pth"
            best.write_bytes(b"broken_weights")

            canonical = Path(td) / "models" / "canonical_hex8_2p.pth"
            canonical.write_bytes(b"good_weights")

            ctx = _make_ctx(
                config_key="hex8_2p",
                model_path=str(best),
            )
            result = _recover_dead_model(ctx)
            assert result.recovered
            assert result.action == "rollback_model"
            # best.pth should now contain canonical weights
            assert best.read_bytes() == b"good_weights"

    @patch("scripts.lib.loop_self_healing._download_canonical_from_s3")
    def test_falls_back_to_s3_when_no_local_canonical(self, mock_s3):
        mock_s3.return_value = True
        with tempfile.TemporaryDirectory() as td:
            models_dir = Path(td) / "models"
            models_dir.mkdir()
            best = models_dir / "best.pth"
            best.write_bytes(b"broken")

            ctx = _make_ctx(
                config_key="hex8_2p",
                model_path=str(best),
            )
            result = _recover_dead_model(ctx)
            assert result.recovered
            mock_s3.assert_called_once()

    @patch("scripts.lib.loop_self_healing._download_canonical_from_s3")
    def test_fails_when_no_canonical_anywhere(self, mock_s3):
        mock_s3.return_value = False
        with tempfile.TemporaryDirectory() as td:
            models_dir = Path(td) / "models"
            models_dir.mkdir()
            best = models_dir / "best.pth"
            best.write_bytes(b"broken")

            ctx = _make_ctx(
                config_key="hex8_2p",
                model_path=str(best),
            )
            result = _recover_dead_model(ctx)
            assert not result.recovered


class TestRecoverArchMismatch:
    @patch("scripts.lib.loop_self_healing._download_canonical_from_s3")
    def test_downloads_from_s3(self, mock_s3):
        mock_s3.return_value = True
        ctx = _make_ctx(config_key="square8_2p")
        result = _recover_arch_mismatch(ctx)
        assert result.recovered
        assert result.action == "redownload_canonical"
        mock_s3.assert_called_once_with("square8_2p", ctx.model_path)

    @patch("scripts.lib.loop_self_healing._download_canonical_from_s3")
    def test_fails_when_s3_unavailable(self, mock_s3):
        mock_s3.return_value = False
        ctx = _make_ctx(config_key="square8_2p")
        result = _recover_arch_mismatch(ctx)
        assert not result.recovered


class TestNoAutoFix:
    def test_returns_not_recovered(self):
        ctx = _make_ctx(error_message="strange error")
        result = _no_auto_fix(ctx)
        assert not result.recovered
        assert result.action == "none"


# ---------------------------------------------------------------------------
# 3. Recovery Coordinator
# ---------------------------------------------------------------------------

class TestAttemptRecovery:
    def test_unknown_pattern_does_not_recover(self):
        ctx = _make_ctx(error_message="something weird happened")
        result = attempt_recovery(ctx)
        assert not result.recovered
        assert result.action == "none"

    def test_oom_recovery_succeeds(self):
        ctx = _make_ctx(
            error_message="CUDA out of memory",
            batch_size=256,
        )
        result = attempt_recovery(ctx)
        assert result.recovered
        assert result.adjustments["batch_size"] == 128

    def test_budget_exhaustion(self):
        """After MAX_RECOVERIES_PER_PATTERN attempts, recovery is refused."""
        for i in range(MAX_RECOVERIES_PER_PATTERN):
            ctx = _make_ctx(
                error_message="CUDA out of memory",
                batch_size=512 // (2 ** i),
            )
            result = attempt_recovery(ctx)
            # May or may not recover depending on batch size, but attempts are consumed

        # Next attempt should be budget-exhausted
        ctx = _make_ctx(
            error_message="CUDA out of memory",
            batch_size=256,
        )
        result = attempt_recovery(ctx)
        assert not result.recovered
        assert "budget_exhausted" in result.action

    def test_dead_model_limited_to_one_recovery(self):
        """DEAD_MODEL pattern should only allow 1 recovery attempt."""
        with tempfile.TemporaryDirectory() as td:
            models_dir = Path(td) / "models"
            models_dir.mkdir()
            best = models_dir / "best.pth"
            best.write_bytes(b"broken")
            canonical = models_dir / "canonical_hex8_2p.pth"
            canonical.write_bytes(b"good")

            # First attempt should succeed
            ctx = _make_ctx(
                error_message="Zero gradient effect: weight delta L2=0.0",
                model_path=str(best),
                config_key="hex8_2p",
            )
            result = attempt_recovery(ctx)
            assert result.recovered

            # Second attempt should be budget-exhausted
            best.write_bytes(b"broken_again")
            result2 = attempt_recovery(ctx)
            assert not result2.recovered
            assert "budget_exhausted" in result2.action

    def test_reset_recovery_counts(self):
        """reset_recovery_counts should allow fresh attempts."""
        ctx = _make_ctx(
            error_message="CUDA out of memory",
            batch_size=256,
        )
        # Exhaust budget
        for _ in range(MAX_RECOVERIES_PER_PATTERN):
            attempt_recovery(ctx)

        result = attempt_recovery(ctx)
        assert not result.recovered

        # Reset and try again
        reset_recovery_counts()
        result = attempt_recovery(ctx)
        assert result.recovered

    @patch("scripts.lib.loop_self_healing._download_canonical_from_s3")
    def test_arch_mismatch_recovery_path(self, mock_s3):
        """Full path: classify -> recover -> result for arch mismatch."""
        mock_s3.return_value = True
        ctx = _make_ctx(
            error_message="ENCODING MISMATCH: NPZ has 14ch but contract expects 56ch",
            stage="export",
            config_key="square8_2p",
        )
        result = attempt_recovery(ctx)
        assert result.recovered
        assert result.action == "redownload_canonical"


class TestRecoveryResult:
    def test_defaults(self):
        r = RecoveryResult(recovered=True, action="test", message="ok")
        assert r.adjustments == {}

    def test_with_adjustments(self):
        r = RecoveryResult(
            recovered=True,
            action="retry_smaller_batch",
            message="halved",
            adjustments={"batch_size": 128},
        )
        assert r.adjustments["batch_size"] == 128
