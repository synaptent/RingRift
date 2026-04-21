"""Regression test for arch-mismatch self-heal model-version safety.

Gh200-11 v5-heavy on 2026-04-21: after a clean reset and a successful
iter-1 promotion at 0.590 WR, iter-2 training hit ``Encoder mismatch:
init_weights=v2, data=v3``. The self-heal classified that as
``arch_mismatch`` and downloaded ``canonical_hex8_2p.pth`` from S3,
which is the v2-family canonical. That 40-channel file overwrote the
64-channel v5-heavy ``best.pth``, poisoning the lane into a persistent
crash-restart loop (NRestarts=116 before intervention).

Fix: ``_recover_arch_mismatch`` now refuses the S3 redownload when
``model_version`` is set and not v2, returning ``recovered=False`` so
the circuit breaker trips cleanly. Lanes that require a fresh arch
recovery need manual intervention.

This test locks the guard.
"""
from __future__ import annotations

from unittest.mock import patch

from scripts.lib.loop_self_healing import (
    FailureContext,
    RecoveryResult,
    _recover_arch_mismatch,
)


def _make_ctx(model_version: str | None) -> FailureContext:
    return FailureContext(
        error_message="ValueError: Encoder mismatch: init_weights=v2, data=v3",
        stage="training",
        config_key="hex8_2p",
        work_dir="/tmp/work",
        model_path="/tmp/work/models/best.pth",
        model_version=model_version,
    )


class TestArchMismatchRecoveryRefusesNonV2:
    def test_v5_heavy_refuses_s3_redownload(self) -> None:
        ctx = _make_ctx("v5-heavy")
        with patch(
            "scripts.lib.loop_self_healing._download_canonical_from_s3"
        ) as mock_dl:
            result = _recover_arch_mismatch(ctx)
        assert mock_dl.call_count == 0, "S3 download must not be attempted for v5-heavy"
        assert result.recovered is False
        assert "v5-heavy" in result.message.lower() or "model_version" in result.message.lower()
        assert result.action == "redownload_canonical_skipped"

    def test_v4_refuses_s3_redownload(self) -> None:
        ctx = _make_ctx("v4")
        with patch(
            "scripts.lib.loop_self_healing._download_canonical_from_s3"
        ) as mock_dl:
            result = _recover_arch_mismatch(ctx)
        assert mock_dl.call_count == 0
        assert result.recovered is False
        assert result.action == "redownload_canonical_skipped"

    def test_v3_refuses_s3_redownload(self) -> None:
        ctx = _make_ctx("v3")
        with patch(
            "scripts.lib.loop_self_healing._download_canonical_from_s3"
        ) as mock_dl:
            result = _recover_arch_mismatch(ctx)
        assert mock_dl.call_count == 0
        assert result.recovered is False


class TestArchMismatchRecoveryProceedsForV2:
    def test_v2_still_attempts_s3_redownload(self) -> None:
        ctx = _make_ctx("v2")
        with patch(
            "scripts.lib.loop_self_healing._download_canonical_from_s3",
            return_value=True,
        ) as mock_dl:
            result = _recover_arch_mismatch(ctx)
        assert mock_dl.call_count == 1
        assert result.recovered is True
        assert result.action == "redownload_canonical"

    def test_none_model_version_defaults_to_v2_behavior(self) -> None:
        """Backward compatibility: callers that don't pass model_version
        (i.e. legacy FailureContext construction) continue to behave as
        before — the S3 redownload is attempted."""
        ctx = _make_ctx(None)
        with patch(
            "scripts.lib.loop_self_healing._download_canonical_from_s3",
            return_value=True,
        ) as mock_dl:
            result = _recover_arch_mismatch(ctx)
        assert mock_dl.call_count == 1
        assert result.recovered is True


class TestFailureContextHasModelVersion:
    def test_model_version_field_exists_and_defaults_none(self) -> None:
        ctx = FailureContext(
            error_message="x",
            stage="training",
            config_key="hex8_2p",
            work_dir="/tmp",
            model_path="/tmp/best.pth",
        )
        assert hasattr(ctx, "model_version")
        assert ctx.model_version is None

    def test_s3_url_not_constructed_with_model_version_suffix(self) -> None:
        """The guard is at the caller level: we don't synthesize a
        model-version-aware S3 URL because those paths don't exist on
        S3. Instead we refuse the download entirely. This test documents
        that intentional choice."""
        from scripts.lib import loop_self_healing
        import inspect

        src = inspect.getsource(loop_self_healing._download_canonical_from_s3)
        # The S3 URL must not append a version suffix; it remains v2-family
        assert "canonical_{config_key}_v" not in src
        assert "model_version" not in src
