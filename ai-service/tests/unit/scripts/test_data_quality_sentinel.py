"""Tests for the Data Quality Sentinel (DQS).

Verifies fingerprint computation, cross-iteration comparison, quality verdicts,
and the end-to-end check_data_quality pipeline using synthetic NPZ data.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from scripts.lib.data_quality_sentinel import (
    ComparisonResult,
    DataFingerprint,
    QualityVerdict,
    check_data_quality,
    compare_with_history,
    compute_fingerprint,
    compute_verdict,
    _save_fingerprint,
    HISTORY_FILENAME,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_npz(
    tmp_dir: Path,
    *,
    n_samples: int = 100,
    n_channels: int = 14,
    board_h: int = 9,
    board_w: int = 9,
    n_actions: int = 61,
    feature_scale: float = 1.0,
    policy_uniform: bool = False,
    policy_entropy_bits: float | None = None,
    value_std: float = 0.5,
    value_mean: float = 0.0,
    seed: int = 42,
) -> str:
    """Create a synthetic NPZ file for testing."""
    rng = np.random.RandomState(seed)
    features = rng.randn(n_samples, n_channels, board_h, board_w).astype(np.float32) * feature_scale

    if policy_uniform:
        policy = np.ones((n_samples, n_actions), dtype=np.float32) / n_actions
    elif policy_entropy_bits is not None:
        # Create policy with approximate target entropy by using a temperature
        # Higher temperature -> more uniform -> higher entropy
        logits = rng.randn(n_samples, n_actions).astype(np.float32)
        # Rough mapping: temp ~0.1 gives low entropy, temp ~5 gives high entropy
        temp = policy_entropy_bits / 2.0 + 0.1
        policy = _softmax(logits / max(temp, 0.01))
    else:
        logits = rng.randn(n_samples, n_actions).astype(np.float32)
        policy = _softmax(logits)

    values = rng.randn(n_samples).astype(np.float32) * value_std + value_mean

    path = str(tmp_dir / "test_data.npz")
    np.savez_compressed(path, features=features, policy=policy, value=values)
    return path


def _softmax(x: np.ndarray) -> np.ndarray:
    """Row-wise softmax."""
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestComputeFingerprint:
    """Test NPZ fingerprint computation."""

    def test_basic_fingerprint(self, tmp_path: Path) -> None:
        """Fingerprint correctly captures NPZ dimensions and statistics."""
        npz = _make_npz(tmp_path, n_samples=200, n_channels=14)
        fp = compute_fingerprint(npz, mmap=False)

        assert fp.n_samples == 200
        assert fp.n_channels == 14
        assert len(fp.feature_means) == 14
        assert len(fp.feature_stds) == 14
        assert fp.policy_entropy_median > 0
        assert fp.value_std > 0
        assert fp.checksum != 0.0

    def test_feature_stats_vary_by_channel(self, tmp_path: Path) -> None:
        """Per-channel means and stds are distinct (not all identical)."""
        npz = _make_npz(tmp_path, n_samples=500, n_channels=10, seed=123)
        fp = compute_fingerprint(npz, mmap=False)

        # With random data, not all channel means should be identical
        means = fp.feature_means
        assert len(set(round(m, 3) for m in means)) > 1, "All channel means are identical"

    def test_zero_variance_features_detected(self, tmp_path: Path) -> None:
        """Constant features produce near-zero stds."""
        npz = _make_npz(tmp_path, n_samples=50, feature_scale=0.0)
        fp = compute_fingerprint(npz, mmap=False)

        assert all(s < 0.001 for s in fp.feature_stds)

    def test_policy_entropy_range(self, tmp_path: Path) -> None:
        """Policy entropy is positive and reasonable for non-degenerate policies."""
        npz = _make_npz(tmp_path, n_samples=100, n_actions=61)
        fp = compute_fingerprint(npz, mmap=False)

        # Random logits through softmax should give moderate entropy
        assert fp.policy_entropy_median > 0.5
        assert fp.policy_entropy_p10 >= 0.0
        assert fp.policy_entropy_p90 >= fp.policy_entropy_p10

    def test_uniform_policy_high_entropy(self, tmp_path: Path) -> None:
        """Uniform policy should have maximum entropy ~log2(61) = 5.93."""
        npz = _make_npz(tmp_path, n_samples=50, n_actions=61, policy_uniform=True)
        fp = compute_fingerprint(npz, mmap=False)

        max_entropy = np.log2(61)
        assert fp.policy_entropy_median > max_entropy - 0.1

    def test_mmap_mode(self, tmp_path: Path) -> None:
        """Fingerprinting works with mmap_mode='r'."""
        npz = _make_npz(tmp_path, n_samples=50)
        fp_mmap = compute_fingerprint(npz, mmap=True)
        fp_nommap = compute_fingerprint(npz, mmap=False)

        assert fp_mmap.n_samples == fp_nommap.n_samples
        assert abs(fp_mmap.checksum - fp_nommap.checksum) < 1e-3


class TestCompareWithHistory:
    """Test cross-iteration comparison logic."""

    def test_no_history_no_issues(self) -> None:
        """First iteration (no history) should not flag identical data."""
        fp = DataFingerprint(
            n_samples=100, n_channels=14,
            feature_means=[0.5] * 14, feature_stds=[1.0] * 14,
            policy_entropy_median=3.0, policy_entropy_p10=2.0, policy_entropy_p90=4.0,
            value_mean=0.1, value_std=0.5, checksum=12345.0,
        )
        result = compare_with_history(fp, [])
        assert not result.identical_data
        assert not result.low_diversity

    def test_identical_data_detected(self) -> None:
        """Near-identical feature means across iterations triggers flag."""
        fp_old = DataFingerprint(
            n_samples=100, n_channels=4,
            feature_means=[0.5, 0.3, 0.7, 0.1], feature_stds=[1.0] * 4,
            policy_entropy_median=3.0, policy_entropy_p10=2.0, policy_entropy_p90=4.0,
            value_mean=0.1, value_std=0.5, checksum=100.0,
        )
        fp_new = DataFingerprint(
            n_samples=100, n_channels=4,
            feature_means=[0.5001, 0.3001, 0.7001, 0.1001], feature_stds=[1.0] * 4,
            policy_entropy_median=3.0, policy_entropy_p10=2.0, policy_entropy_p90=4.0,
            value_mean=0.1, value_std=0.5, checksum=100.1,
        )
        result = compare_with_history(fp_new, [fp_old])
        assert result.identical_data

    def test_different_data_not_flagged(self) -> None:
        """Sufficiently different feature means should not flag identical."""
        fp_old = DataFingerprint(
            n_samples=100, n_channels=4,
            feature_means=[0.5, 0.3, 0.7, 0.1], feature_stds=[1.0] * 4,
            policy_entropy_median=3.0, policy_entropy_p10=2.0, policy_entropy_p90=4.0,
            value_mean=0.1, value_std=0.5, checksum=100.0,
        )
        fp_new = DataFingerprint(
            n_samples=100, n_channels=4,
            feature_means=[0.6, 0.4, 0.8, 0.2], feature_stds=[1.0] * 4,
            policy_entropy_median=3.0, policy_entropy_p10=2.0, policy_entropy_p90=4.0,
            value_mean=0.1, value_std=0.5, checksum=200.0,
        )
        result = compare_with_history(fp_new, [fp_old])
        assert not result.identical_data

    def test_low_entropy_flagged(self) -> None:
        """Policy entropy below 1.0 should trigger low_diversity."""
        fp = DataFingerprint(
            n_samples=100, n_channels=4,
            feature_means=[0.5] * 4, feature_stds=[1.0] * 4,
            policy_entropy_median=0.3, policy_entropy_p10=0.1, policy_entropy_p90=0.5,
            value_mean=0.1, value_std=0.5, checksum=100.0,
        )
        result = compare_with_history(fp, [])
        assert result.low_diversity


class TestComputeVerdict:
    """Test quality verdict logic."""

    def test_healthy_data_passes(self) -> None:
        """Healthy data should produce a PASS verdict."""
        fp = DataFingerprint(
            n_samples=1000, n_channels=14,
            feature_means=[0.5] * 14, feature_stds=[1.0] * 14,
            policy_entropy_median=3.5, policy_entropy_p10=2.0, policy_entropy_p90=5.0,
            value_mean=0.1, value_std=0.5, checksum=50000.0,
        )
        comp = ComparisonResult(
            identical_data=False, low_diversity=False,
            no_value_variance=False, high_draw_rate=False,
        )
        verdict = compute_verdict(fp, comp)
        assert verdict.passed
        assert not verdict.critical
        assert len(verdict.warnings) == 0
        assert "PASS" in verdict.summary

    def test_critical_low_entropy(self) -> None:
        """Extremely low policy entropy should be CRITICAL."""
        fp = DataFingerprint(
            n_samples=100, n_channels=14,
            feature_means=[0.5] * 14, feature_stds=[1.0] * 14,
            policy_entropy_median=0.2, policy_entropy_p10=0.1, policy_entropy_p90=0.3,
            value_mean=0.1, value_std=0.5, checksum=100.0,
        )
        comp = ComparisonResult(
            identical_data=False, low_diversity=True,
            no_value_variance=False, high_draw_rate=False,
        )
        verdict = compute_verdict(fp, comp)
        assert not verdict.passed
        assert verdict.critical
        assert any("entropy" in w.lower() for w in verdict.warnings)

    def test_warn_moderate_low_entropy(self) -> None:
        """Moderately low entropy (0.5 < median < 1.5) should be WARN, not CRITICAL."""
        fp = DataFingerprint(
            n_samples=100, n_channels=14,
            feature_means=[0.5] * 14, feature_stds=[1.0] * 14,
            policy_entropy_median=1.0, policy_entropy_p10=0.5, policy_entropy_p90=2.0,
            value_mean=0.1, value_std=0.5, checksum=100.0,
        )
        comp = ComparisonResult(
            identical_data=False, low_diversity=False,
            no_value_variance=False, high_draw_rate=False,
        )
        verdict = compute_verdict(fp, comp)
        assert verdict.passed  # WARN, not CRITICAL
        assert not verdict.critical
        assert len(verdict.warnings) == 1
        assert "WARN" in verdict.summary

    def test_critical_zero_variance_features(self) -> None:
        """All-zero feature variance should be CRITICAL."""
        fp = DataFingerprint(
            n_samples=100, n_channels=14,
            feature_means=[0.0] * 14, feature_stds=[0.0005] * 14,
            policy_entropy_median=3.0, policy_entropy_p10=2.0, policy_entropy_p90=4.0,
            value_mean=0.1, value_std=0.5, checksum=0.0,
        )
        comp = ComparisonResult(
            identical_data=False, low_diversity=False,
            no_value_variance=False, high_draw_rate=False,
        )
        verdict = compute_verdict(fp, comp)
        assert verdict.critical
        assert any("variance" in w.lower() for w in verdict.warnings)

    def test_critical_identical_data(self) -> None:
        """Identical data across iterations should be CRITICAL."""
        fp = DataFingerprint(
            n_samples=100, n_channels=14,
            feature_means=[0.5] * 14, feature_stds=[1.0] * 14,
            policy_entropy_median=3.0, policy_entropy_p10=2.0, policy_entropy_p90=4.0,
            value_mean=0.1, value_std=0.5, checksum=100.0,
        )
        comp = ComparisonResult(
            identical_data=True, low_diversity=False,
            no_value_variance=False, high_draw_rate=False,
        )
        verdict = compute_verdict(fp, comp)
        assert verdict.critical
        assert any("identical" in w.lower() for w in verdict.warnings)

    def test_warn_low_value_std(self) -> None:
        """Low value target std (but not critical otherwise) should be WARN."""
        fp = DataFingerprint(
            n_samples=100, n_channels=14,
            feature_means=[0.5] * 14, feature_stds=[1.0] * 14,
            policy_entropy_median=3.0, policy_entropy_p10=2.0, policy_entropy_p90=4.0,
            value_mean=0.0, value_std=0.005, checksum=100.0,
        )
        comp = ComparisonResult(
            identical_data=False, low_diversity=False,
            no_value_variance=True, high_draw_rate=False,
        )
        verdict = compute_verdict(fp, comp)
        assert verdict.passed  # WARN only
        assert len(verdict.warnings) == 1
        assert "value" in verdict.warnings[0].lower()


class TestEndToEnd:
    """Test the full check_data_quality pipeline."""

    def test_pass_with_good_data(self, tmp_path: Path) -> None:
        """Good synthetic data produces a PASS verdict."""
        npz = _make_npz(tmp_path, n_samples=200, n_channels=14, value_std=0.5)
        verdict = check_data_quality(npz, work_dir=str(tmp_path), save_history=True)

        assert verdict.passed
        assert not verdict.critical
        assert verdict.fingerprint.n_samples == 200

        # History file should be created
        history_path = tmp_path / HISTORY_FILENAME
        assert history_path.exists()
        lines = history_path.read_text().strip().split("\n")
        assert len(lines) == 1

    def test_critical_with_constant_features(self, tmp_path: Path) -> None:
        """Constant features should produce a CRITICAL verdict."""
        npz = _make_npz(tmp_path, n_samples=100, feature_scale=0.0)
        verdict = check_data_quality(npz, work_dir=str(tmp_path))

        assert verdict.critical
        assert not verdict.passed

    def test_identical_data_across_iterations(self, tmp_path: Path) -> None:
        """Running DQS twice on the exact same data flags CRITICAL on second run."""
        npz = _make_npz(tmp_path, n_samples=100, seed=42)

        # First check: should pass (no history)
        v1 = check_data_quality(npz, work_dir=str(tmp_path))
        assert v1.passed

        # Second check with same NPZ: should flag identical data
        v2 = check_data_quality(npz, work_dir=str(tmp_path))
        assert v2.critical
        assert any("identical" in w.lower() for w in v2.warnings)

    def test_history_accumulates(self, tmp_path: Path) -> None:
        """Multiple calls accumulate history entries."""
        for seed in range(5):
            npz = _make_npz(tmp_path, n_samples=50, seed=seed * 1000)
            check_data_quality(npz, work_dir=str(tmp_path))

        history_path = tmp_path / HISTORY_FILENAME
        lines = [l for l in history_path.read_text().strip().split("\n") if l.strip()]
        assert len(lines) == 5

    def test_skip_history_save(self, tmp_path: Path) -> None:
        """save_history=False should not create a history file."""
        npz = _make_npz(tmp_path, n_samples=50)
        check_data_quality(npz, work_dir=str(tmp_path), save_history=False)

        history_path = tmp_path / HISTORY_FILENAME
        assert not history_path.exists()

    def test_missing_history_graceful(self, tmp_path: Path) -> None:
        """First iteration with no history file works without error."""
        npz = _make_npz(tmp_path, n_samples=100)
        verdict = check_data_quality(npz, work_dir=str(tmp_path))
        # Should not raise and should produce a valid verdict
        assert isinstance(verdict, QualityVerdict)
        assert verdict.fingerprint.n_samples == 100


class TestHistoryPersistence:
    """Test JSONL history file read/write."""

    def test_save_and_load_roundtrip(self, tmp_path: Path) -> None:
        """Fingerprint survives a save -> load roundtrip."""
        from scripts.lib.data_quality_sentinel import _load_history

        fp = DataFingerprint(
            n_samples=500, n_channels=14,
            feature_means=[0.1 * i for i in range(14)],
            feature_stds=[0.5 + 0.01 * i for i in range(14)],
            policy_entropy_median=3.2, policy_entropy_p10=1.5, policy_entropy_p90=4.8,
            value_mean=-0.05, value_std=0.42, checksum=98765.4,
        )
        _save_fingerprint(str(tmp_path), fp)
        loaded = _load_history(str(tmp_path), max_entries=10)

        assert len(loaded) == 1
        lfp = loaded[0]
        assert lfp.n_samples == 500
        assert lfp.n_channels == 14
        assert abs(lfp.policy_entropy_median - 3.2) < 1e-6
        assert abs(lfp.checksum - 98765.4) < 1e-2

    def test_history_max_entries(self, tmp_path: Path) -> None:
        """_load_history respects max_entries limit."""
        from scripts.lib.data_quality_sentinel import _load_history

        for i in range(10):
            fp = DataFingerprint(
                n_samples=i * 10, n_channels=4,
                feature_means=[float(i)] * 4, feature_stds=[1.0] * 4,
                policy_entropy_median=2.0, policy_entropy_p10=1.0, policy_entropy_p90=3.0,
                value_mean=0.0, value_std=0.5, checksum=float(i),
            )
            _save_fingerprint(str(tmp_path), fp)

        loaded = _load_history(str(tmp_path), max_entries=3)
        assert len(loaded) == 3
        # Should be the last 3 entries
        assert loaded[0].n_samples == 70
        assert loaded[1].n_samples == 80
        assert loaded[2].n_samples == 90


# ---------------------------------------------------------------------------
# Edge case tests
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Edge cases for DQS hardening."""

    def test_sparse_policy_single_move_entropy_is_zero(self, tmp_path: Path) -> None:
        """Sparse policy with single move per sample should yield entropy=0.0, not negative."""
        n_samples = 50
        # Create sparse policy: each sample has exactly 1 move with probability 1.0
        policy_indices = np.empty((n_samples, 1), dtype=object)
        policy_values = np.empty((n_samples, 1), dtype=object)
        for i in range(n_samples):
            policy_indices[i, 0] = np.array([i % 61])
            policy_values[i, 0] = np.array([1.0])

        features = np.random.randn(n_samples, 14, 9, 9).astype(np.float32)
        values = np.random.randn(n_samples).astype(np.float32) * 0.5

        path = str(tmp_path / "sparse_single.npz")
        np.savez_compressed(path, features=features, policy_indices=policy_indices,
                            policy_values=policy_values, values=values)

        fp = compute_fingerprint(path, mmap=False)
        assert fp.policy_entropy_median >= 0.0, f"Entropy should be >= 0, got {fp.policy_entropy_median}"
        assert fp.policy_entropy_median < 0.01, "Single-move should have near-zero entropy"

    def test_sparse_policy_multi_move_has_positive_entropy(self, tmp_path: Path) -> None:
        """Sparse policy with multiple moves per sample should have positive entropy."""
        n_samples = 50
        rng = np.random.RandomState(42)
        policy_indices = np.empty((n_samples,), dtype=object)
        policy_values = np.empty((n_samples,), dtype=object)
        for i in range(n_samples):
            n_moves = 5 + (i % 10)
            vals = rng.dirichlet(np.ones(n_moves))
            policy_indices[i] = np.arange(n_moves)
            policy_values[i] = vals

        features = rng.randn(n_samples, 14, 9, 9).astype(np.float32)
        values = rng.randn(n_samples).astype(np.float32) * 0.5

        path = str(tmp_path / "sparse_multi.npz")
        np.savez_compressed(path, features=features, policy_indices=policy_indices,
                            policy_values=policy_values, values=values)

        fp = compute_fingerprint(path, mmap=False)
        assert fp.policy_entropy_median > 0.5, f"Multi-move policy should have entropy > 0.5, got {fp.policy_entropy_median}"

    def test_npz_with_only_features_no_policy_no_value(self, tmp_path: Path) -> None:
        """NPZ with only features (no policy or value arrays) should still produce a fingerprint."""
        features = np.random.randn(50, 14, 9, 9).astype(np.float32)
        path = str(tmp_path / "features_only.npz")
        np.savez_compressed(path, features=features)

        fp = compute_fingerprint(path, mmap=False)
        assert fp.n_samples == 50
        assert fp.n_channels == 14
        assert fp.policy_entropy_median == 0.0
        assert fp.value_mean == 0.0
        assert fp.value_std == 0.0

    def test_history_missing_field_forward_compat(self, tmp_path: Path) -> None:
        """History entries missing a field (e.g., old format) should load with defaults."""
        from scripts.lib.data_quality_sentinel import _load_history

        # Write a history entry that's missing the 'checksum' field
        incomplete = {
            "timestamp": "2026-01-01T00:00:00Z",
            "fingerprint": {
                "n_samples": 100,
                "n_channels": 14,
                "feature_means": [0.5] * 14,
                "feature_stds": [1.0] * 14,
                "policy_entropy_median": 3.0,
                "policy_entropy_p10": 2.0,
                "policy_entropy_p90": 4.0,
                "value_mean": 0.1,
                "value_std": 0.5,
                # missing 'checksum'
            },
        }
        history_path = tmp_path / HISTORY_FILENAME
        history_path.write_text(json.dumps(incomplete) + "\n")

        loaded = _load_history(str(tmp_path))
        assert len(loaded) == 1
        assert loaded[0].n_samples == 100
        assert loaded[0].checksum == 0.0  # default

    def test_history_corrupt_line_skips_gracefully(self, tmp_path: Path) -> None:
        """A corrupt JSON line in history should not crash."""
        from scripts.lib.data_quality_sentinel import _load_history

        history_path = tmp_path / HISTORY_FILENAME
        history_path.write_text("not valid json\n")

        # Should return empty list, not crash
        loaded = _load_history(str(tmp_path))
        assert loaded == []

    def test_dense_policy_entropy_never_negative(self, tmp_path: Path) -> None:
        """Dense policy entropy should never be negative, even for deterministic policies."""
        n_samples = 50
        n_actions = 61
        # Create a one-hot policy (deterministic)
        policy = np.zeros((n_samples, n_actions), dtype=np.float32)
        for i in range(n_samples):
            policy[i, i % n_actions] = 1.0

        features = np.random.randn(n_samples, 14, 9, 9).astype(np.float32)
        values = np.random.randn(n_samples).astype(np.float32)

        path = str(tmp_path / "onehot.npz")
        np.savez_compressed(path, features=features, policy=policy, value=values)

        fp = compute_fingerprint(path, mmap=False)
        assert fp.policy_entropy_median >= 0.0, f"Entropy should be >= 0, got {fp.policy_entropy_median}"
        assert fp.policy_entropy_p10 >= 0.0
        assert fp.policy_entropy_p90 >= 0.0

    def test_empty_npz_samples(self, tmp_path: Path) -> None:
        """NPZ with 0 samples should not crash."""
        features = np.zeros((0, 14, 9, 9), dtype=np.float32)
        policy = np.zeros((0, 61), dtype=np.float32)
        values = np.zeros((0,), dtype=np.float32)

        path = str(tmp_path / "empty.npz")
        np.savez_compressed(path, features=features, policy=policy, value=values)

        # This may raise or return a degenerate fingerprint -- should not crash
        try:
            fp = compute_fingerprint(path, mmap=False)
            assert fp.n_samples == 0
        except (ValueError, IndexError):
            pass  # Acceptable: empty data can't produce meaningful stats

    def test_check_data_quality_end_to_end_sparse(self, tmp_path: Path) -> None:
        """End-to-end check_data_quality with sparse policy format."""
        n_samples = 100
        rng = np.random.RandomState(123)
        policy_indices = np.empty((n_samples,), dtype=object)
        policy_values = np.empty((n_samples,), dtype=object)
        for i in range(n_samples):
            n_moves = 3 + (i % 8)
            vals = rng.dirichlet(np.ones(n_moves))
            policy_indices[i] = np.arange(n_moves)
            policy_values[i] = vals

        features = rng.randn(n_samples, 14, 9, 9).astype(np.float32)
        values = rng.randn(n_samples).astype(np.float32) * 0.5

        path = str(tmp_path / "sparse_e2e.npz")
        np.savez_compressed(path, features=features, policy_indices=policy_indices,
                            policy_values=policy_values, values=values)

        verdict = check_data_quality(path, work_dir=str(tmp_path))
        assert isinstance(verdict, QualityVerdict)
        assert verdict.fingerprint.n_samples == 100
        assert verdict.fingerprint.policy_entropy_median > 0
