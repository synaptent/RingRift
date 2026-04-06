"""Integration test: run Data Quality Sentinel on real NPZ files from the data/ directory.

This test only runs if real NPZ files exist locally (skipped in CI where no
training data is available).  It validates that the DQS pipeline handles the
actual file format produced by the export pipeline — including sparse policy
format with object arrays, metadata keys, and varied channel counts.
"""
from __future__ import annotations

import glob
import os

import pytest

from scripts.lib.data_quality_sentinel import (
    QualityVerdict,
    check_data_quality,
    compute_fingerprint,
)

NPZ_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "training")
REAL_NPZ_FILES = sorted(glob.glob(os.path.join(NPZ_DIR, "*.npz")))


@pytest.mark.skipif(
    not REAL_NPZ_FILES,
    reason="No real NPZ files found in data/training/ — skipping integration test",
)
class TestDQSRealNPZ:
    """Run DQS against every real NPZ file in data/training/."""

    @pytest.mark.parametrize("npz_path", REAL_NPZ_FILES, ids=lambda p: os.path.basename(p))
    def test_fingerprint_does_not_crash(self, npz_path: str) -> None:
        """compute_fingerprint should succeed without exceptions on every real NPZ."""
        fp = compute_fingerprint(npz_path, mmap=False)
        assert fp.n_samples > 0
        assert fp.n_channels > 0
        assert fp.policy_entropy_median >= 0.0, (
            f"Entropy must be >= 0, got {fp.policy_entropy_median}"
        )
        assert fp.policy_entropy_p10 >= 0.0
        assert fp.policy_entropy_p90 >= fp.policy_entropy_p10

    @pytest.mark.parametrize("npz_path", REAL_NPZ_FILES, ids=lambda p: os.path.basename(p))
    def test_check_data_quality_end_to_end(self, npz_path: str, tmp_path) -> None:
        """check_data_quality should return a valid QualityVerdict on real data."""
        verdict = check_data_quality(npz_path, work_dir=str(tmp_path), save_history=False)
        assert isinstance(verdict, QualityVerdict)
        assert verdict.fingerprint.n_samples > 0
        assert isinstance(verdict.summary, str)
        assert len(verdict.summary) > 0

    def test_first_real_npz_has_reasonable_stats(self) -> None:
        """The first available real NPZ should have plausible feature statistics."""
        npz_path = REAL_NPZ_FILES[0]
        fp = compute_fingerprint(npz_path, mmap=False)

        # Basic sanity: features should not be all zeros
        assert any(s > 0.001 for s in fp.feature_stds), (
            f"All feature channels have near-zero std: {fp.feature_stds[:5]}..."
        )
        # Values should have some variance in real training data
        assert fp.value_std > 0.01, f"Value std suspiciously low: {fp.value_std}"
