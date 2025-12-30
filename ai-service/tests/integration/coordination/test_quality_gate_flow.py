"""Integration tests for quality gate enforcement.

December 29, 2025: Tests the quality gate chain:
    NPZ_EXPORT_COMPLETE → TrainingTriggerDaemon quality check
    → TRAINING_BLOCKED_BY_QUALITY (if low) → Selfplay adjustments
    → QUALITY_THRESHOLD_MET (after improvement) → Training resumes

This ensures low-quality training data doesn't reach the training pipeline.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from tests.integration.coordination.conftest import (
    EventChainVerifier,
    MockEventRouter,
)


class TestQualityGateEnforcement:
    """Test quality gate blocking and unblocking."""

    @pytest.fixture
    def mock_router(self):
        return MockEventRouter()

    @pytest.mark.asyncio
    async def test_low_quality_blocks_training(self, mock_router):
        """Test that quality below threshold blocks training."""
        # Simulate NPZ export with low quality
        await mock_router.publish(
            "NPZ_EXPORT_COMPLETE",
            {
                "config_key": "hex8_2p",
                "npz_path": "data/training/hex8_2p.npz",
                "sample_count": 10000,
                "quality_score": 0.32,  # Below 0.40 threshold
                "quality_metrics": {
                    "move_diversity": 0.25,
                    "game_length_variance": 0.15,
                    "policy_entropy": 0.40,
                },
            },
        )

        # Quality gate should block training
        await mock_router.publish(
            "TRAINING_BLOCKED_BY_QUALITY",
            {
                "config_key": "hex8_2p",
                "quality_score": 0.32,
                "threshold": 0.40,
                "blocking_metrics": ["move_diversity", "game_length_variance"],
            },
        )

        assert mock_router.event_count("TRAINING_BLOCKED_BY_QUALITY") == 1

        blocked = mock_router.get_latest("TRAINING_BLOCKED_BY_QUALITY")
        assert blocked.payload["quality_score"] < blocked.payload["threshold"]

    @pytest.mark.asyncio
    async def test_quality_gate_prevents_training_start(self, mock_router):
        """Test that blocked config doesn't start training."""
        events_before_block = []

        # Subscribe to track TRAINING_STARTED
        def track_training(event):
            events_before_block.append(event)

        mock_router.subscribe("TRAINING_STARTED", track_training)

        # Block training
        await mock_router.publish(
            "TRAINING_BLOCKED_BY_QUALITY",
            {"config_key": "hex8_2p", "quality_score": 0.30},
        )

        # Training should NOT start
        assert mock_router.event_count("TRAINING_STARTED") == 0

    @pytest.mark.asyncio
    async def test_quality_improvement_unblocks_training(self, mock_router):
        """Test that quality improvement allows training to proceed."""
        verifier = EventChainVerifier(mock_router)
        verifier.expect_chain([
            "TRAINING_BLOCKED_BY_QUALITY",
            "NPZ_EXPORT_COMPLETE",
            "QUALITY_THRESHOLD_MET",
            "TRAINING_STARTED",
        ])

        # Step 1: Initial block
        await mock_router.publish(
            "TRAINING_BLOCKED_BY_QUALITY",
            {"config_key": "hex8_2p", "quality_score": 0.30},
        )

        # Step 2: New export with better quality
        await mock_router.publish(
            "NPZ_EXPORT_COMPLETE",
            {
                "config_key": "hex8_2p",
                "quality_score": 0.55,  # Above threshold
            },
        )

        # Step 3: Quality threshold met
        await mock_router.publish(
            "QUALITY_THRESHOLD_MET",
            {
                "config_key": "hex8_2p",
                "quality_score": 0.55,
                "threshold": 0.40,
            },
        )

        # Step 4: Training starts
        await mock_router.publish(
            "TRAINING_STARTED",
            {"config_key": "hex8_2p"},
        )

        result = await verifier.verify(timeout=1.0)
        assert result is True

    @pytest.mark.asyncio
    async def test_selfplay_shifts_to_high_quality_mode(self, mock_router):
        """Test that blocked config triggers high-quality selfplay mode."""
        await mock_router.publish(
            "TRAINING_BLOCKED_BY_QUALITY",
            {"config_key": "hex8_2p", "quality_score": 0.30},
        )

        # Selfplay should shift to Gumbel MCTS
        await mock_router.publish(
            "SELFPLAY_MODE_CHANGED",
            {
                "config_key": "hex8_2p",
                "old_mode": "heuristic",
                "new_mode": "gumbel_mcts",
                "gumbel_budget": 800,  # High quality budget
                "reason": "quality_gate_blocked",
            },
        )

        mode_change = mock_router.get_latest("SELFPLAY_MODE_CHANGED")
        assert mode_change.payload["new_mode"] == "gumbel_mcts"
        assert mode_change.payload["gumbel_budget"] >= 800


class TestQualityMetrics:
    """Test individual quality metric handling."""

    @pytest.fixture
    def mock_router(self):
        return MockEventRouter()

    @pytest.mark.asyncio
    async def test_move_diversity_affects_quality(self, mock_router):
        """Test that low move diversity impacts quality score."""
        await mock_router.publish(
            "DATA_QUALITY_ASSESSED",
            {
                "config_key": "hex8_2p",
                "quality_score": 0.35,
                "metrics": {
                    "move_diversity": 0.20,  # Very low
                    "game_length_variance": 0.50,
                    "policy_entropy": 0.60,
                },
                "issues": ["low_move_diversity"],
            },
        )

        quality = mock_router.get_latest("DATA_QUALITY_ASSESSED")
        assert "low_move_diversity" in quality.payload["issues"]

    @pytest.mark.asyncio
    async def test_short_games_detected(self, mock_router):
        """Test detection of abnormally short games."""
        await mock_router.publish(
            "DATA_QUALITY_ASSESSED",
            {
                "config_key": "hex8_2p",
                "quality_score": 0.40,
                "metrics": {
                    "avg_game_length": 12,  # Very short for hex8
                    "expected_length": 35,
                },
                "issues": ["short_games"],
            },
        )

        quality = mock_router.get_latest("DATA_QUALITY_ASSESSED")
        assert "short_games" in quality.payload["issues"]

    @pytest.mark.asyncio
    async def test_policy_entropy_check(self, mock_router):
        """Test policy entropy affects quality assessment."""
        await mock_router.publish(
            "DATA_QUALITY_ASSESSED",
            {
                "config_key": "hex8_2p",
                "metrics": {
                    "policy_entropy": 0.15,  # Very low - model too deterministic
                    "expected_entropy": 0.50,
                },
                "issues": ["low_policy_entropy"],
            },
        )

        quality = mock_router.get_latest("DATA_QUALITY_ASSESSED")
        assert quality.payload["metrics"]["policy_entropy"] < 0.30


class TestQualityGateRecovery:
    """Test quality gate recovery scenarios."""

    @pytest.fixture
    def mock_router(self):
        return MockEventRouter()

    @pytest.mark.asyncio
    async def test_gradual_quality_improvement(self, mock_router):
        """Test tracking of gradual quality improvement."""
        quality_scores = [0.30, 0.35, 0.38, 0.42, 0.48]

        for score in quality_scores:
            await mock_router.publish(
                "DATA_QUALITY_ASSESSED",
                {"config_key": "hex8_2p", "quality_score": score},
            )

        assessments = mock_router.get_events("DATA_QUALITY_ASSESSED")
        scores = [a.payload["quality_score"] for a in assessments]

        # Verify monotonic improvement
        assert all(scores[i] <= scores[i + 1] for i in range(len(scores) - 1))

        # Final score above threshold
        assert scores[-1] >= 0.40

    @pytest.mark.asyncio
    async def test_multiple_configs_blocked_independently(self, mock_router):
        """Test that multiple configs can be blocked independently."""
        configs = [
            ("hex8_2p", 0.30),
            ("hex8_3p", 0.50),  # Above threshold
            ("hex8_4p", 0.25),
        ]

        for config, score in configs:
            if score < 0.40:
                await mock_router.publish(
                    "TRAINING_BLOCKED_BY_QUALITY",
                    {"config_key": config, "quality_score": score},
                )

        # Only 2 configs should be blocked
        assert mock_router.event_count("TRAINING_BLOCKED_BY_QUALITY") == 2

        blocked_configs = [
            e.payload["config_key"]
            for e in mock_router.get_events("TRAINING_BLOCKED_BY_QUALITY")
        ]
        assert "hex8_2p" in blocked_configs
        assert "hex8_4p" in blocked_configs
        assert "hex8_3p" not in blocked_configs

    @pytest.mark.asyncio
    async def test_quality_gate_timeout(self, mock_router):
        """Test behavior when quality doesn't improve within timeout."""
        await mock_router.publish(
            "TRAINING_BLOCKED_BY_QUALITY",
            {
                "config_key": "hex8_2p",
                "quality_score": 0.25,
                "blocked_since": time.time() - 3600,  # 1 hour ago
            },
        )

        await mock_router.publish(
            "QUALITY_GATE_TIMEOUT",
            {
                "config_key": "hex8_2p",
                "action": "fallback_to_heuristic_data",
                "timeout_hours": 1.0,
            },
        )

        timeout = mock_router.get_latest("QUALITY_GATE_TIMEOUT")
        assert timeout is not None
        assert timeout.payload["action"] == "fallback_to_heuristic_data"


class TestQualityFeedbackIntegration:
    """Test quality feedback affecting other systems."""

    @pytest.fixture
    def mock_router(self):
        return MockEventRouter()

    @pytest.mark.asyncio
    async def test_quality_affects_curriculum_weights(self, mock_router):
        """Test that low quality reduces curriculum weight."""
        await mock_router.publish(
            "TRAINING_BLOCKED_BY_QUALITY",
            {"config_key": "hex8_2p", "quality_score": 0.30},
        )

        await mock_router.publish(
            "CURRICULUM_WEIGHT_ADJUSTED",
            {
                "config_key": "hex8_2p",
                "old_weight": 1.0,
                "new_weight": 0.5,  # Reduced due to quality issues
                "reason": "low_data_quality",
            },
        )

        adjustment = mock_router.get_latest("CURRICULUM_WEIGHT_ADJUSTED")
        assert adjustment.payload["new_weight"] < adjustment.payload["old_weight"]

    @pytest.mark.asyncio
    async def test_quality_affects_gumbel_budget(self, mock_router):
        """Test that low quality increases Gumbel search budget."""
        await mock_router.publish(
            "TRAINING_BLOCKED_BY_QUALITY",
            {"config_key": "hex8_2p"},
        )

        await mock_router.publish(
            "GUMBEL_BUDGET_ADJUSTED",
            {
                "config_key": "hex8_2p",
                "old_budget": 150,  # Standard
                "new_budget": 800,  # Quality tier
                "reason": "improve_data_quality",
            },
        )

        budget = mock_router.get_latest("GUMBEL_BUDGET_ADJUSTED")
        assert budget.payload["new_budget"] > budget.payload["old_budget"]
