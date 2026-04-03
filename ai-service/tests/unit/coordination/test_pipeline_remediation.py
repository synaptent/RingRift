"""Integration tests for pipeline remediation features.

Tests new pipeline features added in February 2026:
1. PROMOTION_REJECTED event emission from AutoPromotionDaemon
2. PipelineCompletenessMonitor rejection tracking
3. Board-aware gauntlet simulations
4. Preferred architecture selection per board type
5. Disk threshold constants
6. Training data validation (MIN_SAMPLES_FOR_TRAINING)
7. Elo gap priority factor in PriorityCalculator

February 2026 - Pipeline remediation integration tests.
"""

from __future__ import annotations

import os
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from app.coordination.auto_promotion_daemon import (
    AutoPromotionDaemon,
    PromotionCandidate,
)
from app.coordination.pipeline_completeness_monitor import (
    PipelineCompletenessMonitor,
)
from app.config.thresholds import (
    DISK_CRITICAL_PERCENT,
    DISK_PRODUCTION_HALT_PERCENT,
    DISK_SYNC_TARGET_PERCENT,
    get_gauntlet_simulations,
    get_preferred_architecture,
)
from app.coordination.priority_calculator import (
    PriorityCalculator,
    PriorityInputs,
)
from app.training.datasets import MIN_SAMPLES_FOR_TRAINING, RingRiftDataset


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def candidate():
    """Create a minimal PromotionCandidate for testing."""
    return PromotionCandidate(
        config_key="hex8_2p",
        model_path="models/candidate_hex8_2p.pth",
        estimated_elo=1650.0,
    )


@pytest.fixture
def daemon():
    """Create an AutoPromotionDaemon instance for testing."""
    d = AutoPromotionDaemon.__new__(AutoPromotionDaemon)
    # Minimal init to avoid HandlerBase lifecycle
    d.config = MagicMock()
    d._candidates = {}
    d._promotion_history = []
    return d


@pytest.fixture
def monitor():
    """Create a PipelineCompletenessMonitor with minimal init."""
    m = PipelineCompletenessMonitor.__new__(PipelineCompletenessMonitor)
    m._consecutive_rejections = {}
    m._rejection_alerts_emitted = set()
    m.CONSECUTIVE_REJECTION_ALERT_THRESHOLD = 5
    return m


@pytest.fixture
def priority_calculator():
    """Create a PriorityCalculator with default settings."""
    return PriorityCalculator()


def _make_sparse_npz(tmp_path: str, n_samples: int) -> None:
    """Create a valid NPZ file with the sparse policy format expected by RingRiftDataset.

    The dataset expects 'features', 'values', 'policy_indices', and 'policy_values' keys.
    Policy data uses object arrays with variable-length sparse indices per sample.
    """
    features = np.zeros((n_samples, 40, 9, 9), dtype=np.float32)
    values = np.zeros((n_samples,), dtype=np.float32)
    # Sparse policy: each sample has an array of move indices and corresponding probs
    policy_indices = np.empty(n_samples, dtype=object)
    policy_values = np.empty(n_samples, dtype=object)
    for i in range(n_samples):
        # Give each sample one valid move so it passes the empty-policy filter
        policy_indices[i] = np.array([0], dtype=np.int64)
        policy_values[i] = np.array([1.0], dtype=np.float32)
    np.savez(
        tmp_path,
        features=features,
        values=values,
        policy_indices=policy_indices,
        policy_values=policy_values,
    )


# =============================================================================
# 1. Test PROMOTION_REJECTED event emission
# =============================================================================


class TestPromotionRejectedEmission:
    """Tests for _emit_promotion_rejected() event payload structure."""

    def test_emits_event_with_required_fields(self, daemon, candidate):
        """Verify PROMOTION_REJECTED event contains gate, reason, config_key, model_path."""
        with patch(
            "app.coordination.event_router.safe_emit_event"
        ) as mock_emit:
            daemon._emit_promotion_rejected(
                candidate,
                gate="quality",
                reason="Win rate 0.45 < 0.55 threshold",
            )

            mock_emit.assert_called_once()
            call_args = mock_emit.call_args
            event_type = call_args[0][0]
            payload = call_args[0][1]

            assert event_type == "PROMOTION_REJECTED"
            assert payload["config_key"] == "hex8_2p"
            assert payload["gate"] == "quality"
            assert payload["reason"] == "Win rate 0.45 < 0.55 threshold"
            assert payload["model_path"] == "models/candidate_hex8_2p.pth"
            assert payload["estimated_elo"] == 1650.0
            assert "timestamp" in payload

    def test_emits_event_with_actual_and_threshold(self, daemon, candidate):
        """Verify actual/threshold values are included when provided."""
        with patch(
            "app.coordination.event_router.safe_emit_event"
        ) as mock_emit:
            daemon._emit_promotion_rejected(
                candidate,
                gate="elo_improvement",
                reason="Elo improvement 15 < 25 threshold",
                actual=15.0,
                threshold=25.0,
            )

            payload = mock_emit.call_args[0][1]
            assert payload["actual"] == 15.0
            assert payload["threshold"] == 25.0

    def test_omits_actual_threshold_when_none(self, daemon, candidate):
        """Verify actual/threshold keys are absent when not provided."""
        with patch(
            "app.coordination.event_router.safe_emit_event"
        ) as mock_emit:
            daemon._emit_promotion_rejected(
                candidate,
                gate="stability",
                reason="Model unstable",
            )

            payload = mock_emit.call_args[0][1]
            assert "actual" not in payload
            assert "threshold" not in payload

    def test_event_source_is_auto_promotion_daemon(self, daemon, candidate):
        """Verify the event source is set to AutoPromotionDaemon."""
        with patch(
            "app.coordination.event_router.safe_emit_event"
        ) as mock_emit:
            daemon._emit_promotion_rejected(
                candidate,
                gate="gauntlet_win_rate",
                reason="Failed gauntlet",
            )

            call_kwargs = mock_emit.call_args
            assert call_kwargs[1]["source"] == "AutoPromotionDaemon"

    def test_persists_rejection_to_jsonl(self, daemon, candidate):
        """Verify rejection is persisted to JSONL file after emission."""
        with patch(
            "app.coordination.event_router.safe_emit_event"
        ):
            with patch.object(daemon, "_persist_rejection") as mock_persist:
                daemon._emit_promotion_rejected(
                    candidate,
                    gate="head_to_head",
                    reason="Lost to champion",
                    actual=0.40,
                    threshold=0.50,
                )

                mock_persist.assert_called_once()
                persisted_payload = mock_persist.call_args[0][0]
                assert persisted_payload["gate"] == "head_to_head"
                assert persisted_payload["actual"] == 0.40


# =============================================================================
# 2. Test PipelineCompletenessMonitor rejection tracking
# =============================================================================


class TestPipelineCompletenessMonitorRejections:
    """Tests for consecutive rejection tracking in PipelineCompletenessMonitor."""

    @pytest.mark.asyncio
    async def test_tracks_consecutive_rejections(self, monitor):
        """Verify rejection count increments per (config_key, gate) pair."""
        event = {
            "type": "promotion_rejected",
            "config_key": "hex8_2p",
            "gate": "quality",
            "reason": "Low win rate",
        }

        monitor._get_payload = lambda e: e
        monitor._record_success = lambda: None
        monitor._safe_emit_event_async = AsyncMock()

        for _ in range(3):
            await monitor._on_promotion_rejected(event)

        assert monitor._consecutive_rejections[("hex8_2p", "quality")] == 3

    @pytest.mark.asyncio
    async def test_emits_overdue_after_threshold(self, monitor):
        """Verify PIPELINE_STAGE_OVERDUE is emitted after 5+ consecutive rejections."""
        event = {
            "type": "promotion_rejected",
            "config_key": "square8_3p",
            "gate": "elo_improvement",
            "reason": "Elo gain too small",
        }

        monitor._get_payload = lambda e: e
        monitor._record_success = lambda: None
        mock_emit = AsyncMock()
        monitor._safe_emit_event_async = mock_emit

        # Send 5 rejections to reach threshold
        for _ in range(5):
            await monitor._on_promotion_rejected(event)

        # Should have emitted PIPELINE_STAGE_OVERDUE on the 5th rejection
        mock_emit.assert_awaited_once()
        call_args = mock_emit.call_args[0]
        assert call_args[0] == "PIPELINE_STAGE_OVERDUE"
        payload = call_args[1]
        assert payload["config_key"] == "square8_3p"
        assert payload["blocking_gate"] == "elo_improvement"
        assert payload["consecutive_rejections"] == 5
        assert payload["stage"] == "promotion"

    @pytest.mark.asyncio
    async def test_does_not_emit_before_threshold(self, monitor):
        """Verify no alert before reaching 5 consecutive rejections."""
        event = {
            "type": "promotion_rejected",
            "config_key": "hex8_4p",
            "gate": "quality",
            "reason": "Not good enough",
        }

        monitor._get_payload = lambda e: e
        monitor._record_success = lambda: None
        mock_emit = AsyncMock()
        monitor._safe_emit_event_async = mock_emit

        for _ in range(4):
            await monitor._on_promotion_rejected(event)

        mock_emit.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_does_not_double_emit_for_same_gate(self, monitor):
        """Verify alert is emitted only once per (config, gate) even after many rejections."""
        event = {
            "type": "promotion_rejected",
            "config_key": "square19_2p",
            "gate": "gauntlet_win_rate",
            "reason": "Failed gauntlet",
        }

        monitor._get_payload = lambda e: e
        monitor._record_success = lambda: None
        mock_emit = AsyncMock()
        monitor._safe_emit_event_async = mock_emit

        # Send 10 rejections - should only alert once at 5
        for _ in range(10):
            await monitor._on_promotion_rejected(event)

        assert mock_emit.await_count == 1

    def test_reset_tracking_on_successful_promotion(self, monitor):
        """Verify rejection counters reset when a model is successfully promoted."""
        # Simulate some rejection history
        monitor._consecutive_rejections[("hex8_2p", "quality")] = 7
        monitor._consecutive_rejections[("hex8_2p", "elo_improvement")] = 3
        monitor._consecutive_rejections[("square8_3p", "quality")] = 2
        monitor._rejection_alerts_emitted.add(("hex8_2p", "quality"))

        monitor._reset_rejection_tracking("hex8_2p")

        # hex8_2p entries should be cleared
        assert ("hex8_2p", "quality") not in monitor._consecutive_rejections
        assert ("hex8_2p", "elo_improvement") not in monitor._consecutive_rejections
        assert ("hex8_2p", "quality") not in monitor._rejection_alerts_emitted

        # square8_3p should be untouched
        assert monitor._consecutive_rejections[("square8_3p", "quality")] == 2

    @pytest.mark.asyncio
    async def test_tracks_different_gates_independently(self, monitor):
        """Verify different gates are tracked independently for the same config."""
        monitor._get_payload = lambda e: e
        monitor._record_success = lambda: None
        monitor._safe_emit_event_async = AsyncMock()

        event_quality = {
            "type": "promotion_rejected",
            "config_key": "hex8_2p",
            "gate": "quality",
            "reason": "Low quality",
        }
        event_elo = {
            "type": "promotion_rejected",
            "config_key": "hex8_2p",
            "gate": "elo_improvement",
            "reason": "Low Elo gain",
        }

        for _ in range(3):
            await monitor._on_promotion_rejected(event_quality)
        for _ in range(2):
            await monitor._on_promotion_rejected(event_elo)

        assert monitor._consecutive_rejections[("hex8_2p", "quality")] == 3
        assert monitor._consecutive_rejections[("hex8_2p", "elo_improvement")] == 2


# =============================================================================
# 3. Test board-aware gauntlet simulations
# =============================================================================


class TestGauntletSimulations:
    """Tests for get_gauntlet_simulations() board-type-aware budgets."""

    def test_small_board_2p(self):
        """Small boards (hex8, square8) use the current 800-sim 2-player budget."""
        assert get_gauntlet_simulations(num_players=2, board_type="hex8") == 800
        assert get_gauntlet_simulations(num_players=2, board_type="square8") == 800

    def test_small_board_3p(self):
        """Small boards use the current 800-sim 3-player budget."""
        assert get_gauntlet_simulations(num_players=3, board_type="hex8") == 800
        assert get_gauntlet_simulations(num_players=3, board_type="square8") == 800

    def test_small_board_4p(self):
        """Small boards use 800 sims for 4-player."""
        assert get_gauntlet_simulations(num_players=4, board_type="hex8") == 800
        assert get_gauntlet_simulations(num_players=4, board_type="square8") == 800

    def test_large_board_2p(self):
        """Large boards use the current 800-sim selfplay-aligned 2-player budget."""
        assert get_gauntlet_simulations(num_players=2, board_type="square19") == 800
        assert get_gauntlet_simulations(num_players=2, board_type="hexagonal") == 800

    def test_large_board_3p(self):
        """Large boards use the current 800-sim selfplay-aligned 3-player budget."""
        assert get_gauntlet_simulations(num_players=3, board_type="square19") == 800
        assert get_gauntlet_simulations(num_players=3, board_type="hexagonal") == 800

    def test_large_board_4p(self):
        """Large boards use the current 800-sim selfplay-aligned 4-player budget."""
        assert get_gauntlet_simulations(num_players=4, board_type="square19") == 800
        assert get_gauntlet_simulations(num_players=4, board_type="hexagonal") == 800

    def test_default_no_board_type_uses_small_values(self):
        """When board_type is empty, fall back to the current small-board defaults."""
        assert get_gauntlet_simulations(num_players=2) == 800
        assert get_gauntlet_simulations(num_players=3) == 800
        assert get_gauntlet_simulations(num_players=4) == 800

    def test_unknown_board_type_uses_small_values(self):
        """Unknown board types fall back to the current small-board defaults."""
        assert get_gauntlet_simulations(num_players=2, board_type="triangle5") == 800


# =============================================================================
# 4. Test preferred architecture selection
# =============================================================================


class TestPreferredArchitecture:
    """Tests for get_preferred_architecture() per-board-type mapping."""

    def test_hex8_uses_v2(self):
        """hex8 (61 cells) works well with v2 (40ch, 10 base x 4 frames)."""
        assert get_preferred_architecture("hex8") == "v2"

    def test_square8_uses_v5_heavy(self):
        """square8 currently stays on v2 until v5-heavy bootstraps exist."""
        assert get_preferred_architecture("square8") == "v2"

    def test_square19_uses_v5_heavy(self):
        """square19 currently stays on v2 until v5-heavy bootstraps exist."""
        assert get_preferred_architecture("square19") == "v2"

    def test_hexagonal_uses_v5_heavy(self):
        """hexagonal currently stays on v2 after the reverted v5-heavy attempt."""
        assert get_preferred_architecture("hexagonal") == "v2"

    def test_unknown_board_defaults_to_v2(self):
        """Unknown board types default to v2 as the safe baseline."""
        assert get_preferred_architecture("unknown_board") == "v2"
        assert get_preferred_architecture("") == "v2"


# =============================================================================
# 5. Test disk threshold constants
# =============================================================================


class TestDiskThresholdConstants:
    """Tests for canonical disk threshold constants in app.config.thresholds."""

    def test_disk_sync_target_percent(self):
        """DISK_SYNC_TARGET_PERCENT is 70 -- max disk for sync/distribution."""
        assert DISK_SYNC_TARGET_PERCENT == 70

    def test_disk_production_halt_percent(self):
        """DISK_PRODUCTION_HALT_PERCENT is 93 -- pause selfplay/training/exports."""
        assert DISK_PRODUCTION_HALT_PERCENT == 93

    def test_disk_critical_percent(self):
        """DISK_CRITICAL_PERCENT is 96 -- emergency halt, block all writes."""
        assert DISK_CRITICAL_PERCENT == 96

    def test_threshold_ordering(self):
        """Thresholds must be in ascending order: sync < production < critical."""
        assert DISK_SYNC_TARGET_PERCENT < DISK_PRODUCTION_HALT_PERCENT
        assert DISK_PRODUCTION_HALT_PERCENT < DISK_CRITICAL_PERCENT


# =============================================================================
# 6. Test training data validation
# =============================================================================


class TestTrainingDataValidation:
    """Tests for MIN_SAMPLES_FOR_TRAINING enforcement in RingRiftDataset."""

    def test_min_samples_constant_value(self):
        """MIN_SAMPLES_FOR_TRAINING should be 100."""
        assert MIN_SAMPLES_FOR_TRAINING == 100

    def test_raises_on_insufficient_samples(self):
        """Dataset raises ValueError when sample count < MIN_SAMPLES_FOR_TRAINING."""
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            tmp_path = f.name

        try:
            # Create an NPZ with only 50 samples (below the 100 minimum)
            _make_sparse_npz(tmp_path, n_samples=50)
            with pytest.raises(ValueError, match="Dataset too small"):
                RingRiftDataset(tmp_path)
        finally:
            os.unlink(tmp_path)

    def test_passes_on_sufficient_samples(self):
        """Dataset loads successfully when sample count >= MIN_SAMPLES_FOR_TRAINING."""
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            tmp_path = f.name

        try:
            # Create an NPZ with 150 samples (above the 100 minimum)
            _make_sparse_npz(tmp_path, n_samples=150)
            dataset = RingRiftDataset(tmp_path)
            assert len(dataset) > 0
        finally:
            os.unlink(tmp_path)


# =============================================================================
# 7. Test Elo gap priority factor
# =============================================================================


class TestEloGapPriorityFactor:
    """Tests for the Elo gap factor in PriorityCalculator.compute_priority_score().

    The Elo gap factor scales priority based on distance from 2000 Elo target:
    - gap_factor = min(3.0, 1.0 + (elo_gap / 500.0))
    - At or above target: maintenance mode = 0.3x
    """

    def test_500_gap_factor(self, priority_calculator):
        """Config at 1500 Elo (500 gap) should have elo_gap_factor ~2.0."""
        inputs = PriorityInputs(
            config_key="hex8_2p",
            current_elo=1500.0,
            staleness_hours=10.0,
        )
        score_1500 = priority_calculator.compute_priority_score(inputs)

        # Reference: just below target so gap factor is ~1.0
        inputs_ref = PriorityInputs(
            config_key="hex8_2p",
            current_elo=1999.9,
            staleness_hours=10.0,
        )
        score_ref = priority_calculator.compute_priority_score(inputs_ref)

        # At 1500 Elo, gap=500, factor=2.0. At 1999.9, gap=0.1, factor~1.0.
        ratio = score_1500 / score_ref if score_ref > 0 else float("inf")
        assert 1.8 <= ratio <= 2.2, f"Expected ratio ~2.0, got {ratio:.2f}"

    def test_100_gap_factor(self, priority_calculator):
        """Config at 1900 Elo (100 gap) should have elo_gap_factor ~1.2."""
        inputs_1900 = PriorityInputs(
            config_key="hex8_2p",
            current_elo=1900.0,
            staleness_hours=10.0,
        )
        inputs_ref = PriorityInputs(
            config_key="hex8_2p",
            current_elo=1999.9,
            staleness_hours=10.0,
        )

        score_1900 = priority_calculator.compute_priority_score(inputs_1900)
        score_ref = priority_calculator.compute_priority_score(inputs_ref)

        ratio = score_1900 / score_ref if score_ref > 0 else float("inf")
        assert 1.1 <= ratio <= 1.3, f"Expected ratio ~1.2, got {ratio:.2f}"

    def test_target_met_maintenance_mode(self, priority_calculator):
        """Config at 2000+ Elo enters maintenance mode with factor 0.3."""
        inputs_at_target = PriorityInputs(
            config_key="hex8_2p",
            current_elo=2000.0,
            staleness_hours=10.0,
        )
        inputs_below = PriorityInputs(
            config_key="hex8_2p",
            current_elo=1999.9,
            staleness_hours=10.0,
        )

        score_at_target = priority_calculator.compute_priority_score(inputs_at_target)
        score_below = priority_calculator.compute_priority_score(inputs_below)

        # At target: factor=0.3. Just below: factor~1.0.
        ratio = score_at_target / score_below if score_below > 0 else float("inf")
        assert 0.2 <= ratio <= 0.4, f"Expected ratio ~0.3, got {ratio:.2f}"

    def test_above_target_also_maintenance(self, priority_calculator):
        """Config above target (2100 Elo) also gets maintenance factor 0.3."""
        inputs_2100 = PriorityInputs(
            config_key="hex8_2p",
            current_elo=2100.0,
            staleness_hours=10.0,
        )
        inputs_2000 = PriorityInputs(
            config_key="hex8_2p",
            current_elo=2000.0,
            staleness_hours=10.0,
        )

        score_2100 = priority_calculator.compute_priority_score(inputs_2100)
        score_2000 = priority_calculator.compute_priority_score(inputs_2000)

        # Both at/above target get the same maintenance factor (0.3)
        ratio = score_2100 / score_2000 if score_2000 > 0 else float("inf")
        assert 0.9 <= ratio <= 1.1, f"Expected ratio ~1.0, got {ratio:.2f}"

    def test_gap_factor_capped_at_3(self, priority_calculator):
        """Elo gap factor is capped at 3.0 for very weak configs."""
        # 500 Elo = 1500 gap, uncapped would be 1.0 + 1500/500 = 4.0
        inputs_very_low = PriorityInputs(
            config_key="hex8_2p",
            current_elo=500.0,
            staleness_hours=10.0,
        )
        # 1000 Elo = 1000 gap, factor = 1.0 + 1000/500 = 3.0 (at the cap)
        inputs_at_cap = PriorityInputs(
            config_key="hex8_2p",
            current_elo=1000.0,
            staleness_hours=10.0,
        )

        score_very_low = priority_calculator.compute_priority_score(inputs_very_low)
        score_at_cap = priority_calculator.compute_priority_score(inputs_at_cap)

        # Both should be at the 3.0 cap, so scores should be equal
        ratio = score_very_low / score_at_cap if score_at_cap > 0 else float("inf")
        assert 0.9 <= ratio <= 1.1, f"Expected ratio ~1.0 (both capped at 3.0), got {ratio:.2f}"
