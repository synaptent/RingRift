"""Tests for metrics/orchestrator (Prometheus metrics for pipeline).

Tests cover:
- Metric definitions exist
- Helper functions for recording metrics
- Pipeline state constants
"""

from unittest.mock import MagicMock, patch

import pytest

# =============================================================================
# Test Metric Definitions
# =============================================================================

class TestSelfplayMetrics:
    """Tests for selfplay metric definitions."""

    def test_selfplay_games_total_exists(self):
        """Test SELFPLAY_GAMES_TOTAL metric exists."""
        from app.metrics.orchestrator import SELFPLAY_GAMES_TOTAL
        assert SELFPLAY_GAMES_TOTAL is not None

    def test_selfplay_games_per_second_exists(self):
        """Test SELFPLAY_GAMES_PER_SECOND metric exists."""
        from app.metrics.orchestrator import SELFPLAY_GAMES_PER_SECOND
        assert SELFPLAY_GAMES_PER_SECOND is not None

    def test_selfplay_batch_duration_exists(self):
        """Test SELFPLAY_BATCH_DURATION metric exists."""
        from app.metrics.orchestrator import SELFPLAY_BATCH_DURATION
        assert SELFPLAY_BATCH_DURATION is not None

    def test_selfplay_errors_total_exists(self):
        """Test SELFPLAY_ERRORS_TOTAL metric exists."""
        from app.metrics.orchestrator import SELFPLAY_ERRORS_TOTAL
        assert SELFPLAY_ERRORS_TOTAL is not None

    def test_selfplay_queue_size_exists(self):
        """Test SELFPLAY_QUEUE_SIZE metric exists."""
        from app.metrics.orchestrator import SELFPLAY_QUEUE_SIZE
        assert SELFPLAY_QUEUE_SIZE is not None


class TestTrainingMetrics:
    """Tests for training metric definitions."""

    def test_training_runs_total_exists(self):
        """Test TRAINING_RUNS_TOTAL metric exists."""
        from app.metrics.orchestrator import TRAINING_RUNS_TOTAL
        assert TRAINING_RUNS_TOTAL is not None

    def test_training_run_duration_exists(self):
        """Test TRAINING_RUN_DURATION metric exists."""
        from app.metrics.orchestrator import TRAINING_RUN_DURATION
        assert TRAINING_RUN_DURATION is not None

    def test_training_loss_exists(self):
        """Test TRAINING_LOSS metric exists."""
        from app.metrics.orchestrator import TRAINING_LOSS
        assert TRAINING_LOSS is not None

    def test_training_accuracy_exists(self):
        """Test TRAINING_ACCURACY metric exists."""
        from app.metrics.orchestrator import TRAINING_ACCURACY
        assert TRAINING_ACCURACY is not None

    def test_training_samples_processed_exists(self):
        """Test TRAINING_SAMPLES_PROCESSED metric exists."""
        from app.metrics.orchestrator import TRAINING_SAMPLES_PROCESSED
        assert TRAINING_SAMPLES_PROCESSED is not None


class TestEvaluationMetrics:
    """Tests for evaluation metric definitions."""

    def test_evaluation_games_total_exists(self):
        """Test EVALUATION_GAMES_TOTAL metric exists."""
        from app.metrics.orchestrator import EVALUATION_GAMES_TOTAL
        assert EVALUATION_GAMES_TOTAL is not None

    def test_evaluation_elo_delta_exists(self):
        """Test EVALUATION_ELO_DELTA metric exists."""
        from app.metrics.orchestrator import EVALUATION_ELO_DELTA
        assert EVALUATION_ELO_DELTA is not None

    def test_evaluation_win_rate_exists(self):
        """Test EVALUATION_WIN_RATE metric exists."""
        from app.metrics.orchestrator import EVALUATION_WIN_RATE
        assert EVALUATION_WIN_RATE is not None


class TestModelPromotionMetrics:
    """Tests for model promotion metric definitions."""

    def test_model_promotions_total_exists(self):
        """Test MODEL_PROMOTIONS_TOTAL metric exists."""
        from app.metrics.orchestrator import MODEL_PROMOTIONS_TOTAL
        assert MODEL_PROMOTIONS_TOTAL is not None

    def test_model_promotion_elo_gain_exists(self):
        """Test MODEL_PROMOTION_ELO_GAIN metric exists."""
        from app.metrics.orchestrator import MODEL_PROMOTION_ELO_GAIN
        assert MODEL_PROMOTION_ELO_GAIN is not None

    def test_current_model_elo_exists(self):
        """Test CURRENT_MODEL_ELO metric exists."""
        from app.metrics.orchestrator import CURRENT_MODEL_ELO
        assert CURRENT_MODEL_ELO is not None


class TestPipelineMetrics:
    """Tests for pipeline metric definitions."""

    def test_pipeline_stage_duration_exists(self):
        """Test PIPELINE_STAGE_DURATION metric exists."""
        from app.metrics.orchestrator import PIPELINE_STAGE_DURATION
        assert PIPELINE_STAGE_DURATION is not None

    def test_pipeline_iterations_total_exists(self):
        """Test PIPELINE_ITERATIONS_TOTAL metric exists."""
        from app.metrics.orchestrator import PIPELINE_ITERATIONS_TOTAL
        assert PIPELINE_ITERATIONS_TOTAL is not None

    def test_pipeline_state_exists(self):
        """Test PIPELINE_STATE metric exists."""
        from app.metrics.orchestrator import PIPELINE_STATE
        assert PIPELINE_STATE is not None


class TestSyncMetrics:
    """Tests for sync metric definitions."""

    def test_data_sync_duration_exists(self):
        """Test DATA_SYNC_DURATION metric exists."""
        from app.metrics.orchestrator import DATA_SYNC_DURATION
        assert DATA_SYNC_DURATION is not None

    def test_data_sync_games_exists(self):
        """Test DATA_SYNC_GAMES metric exists."""
        from app.metrics.orchestrator import DATA_SYNC_GAMES
        assert DATA_SYNC_GAMES is not None

    def test_model_sync_duration_exists(self):
        """Test MODEL_SYNC_DURATION metric exists."""
        from app.metrics.orchestrator import MODEL_SYNC_DURATION
        assert MODEL_SYNC_DURATION is not None

    def test_sync_coordinator_ops_exists(self):
        """Test SYNC_COORDINATOR_OPS metric exists."""
        from app.metrics.orchestrator import SYNC_COORDINATOR_OPS
        assert SYNC_COORDINATOR_OPS is not None


class TestQualityMetrics:
    """Tests for training data quality metrics."""

    def test_training_data_quality_score_exists(self):
        """Test TRAINING_DATA_QUALITY_SCORE metric exists."""
        from app.metrics.orchestrator import TRAINING_DATA_QUALITY_SCORE
        assert TRAINING_DATA_QUALITY_SCORE is not None

    def test_training_data_high_quality_count_exists(self):
        """Test TRAINING_DATA_HIGH_QUALITY_COUNT metric exists."""
        from app.metrics.orchestrator import TRAINING_DATA_HIGH_QUALITY_COUNT
        assert TRAINING_DATA_HIGH_QUALITY_COUNT is not None


# =============================================================================
# Test Helper Functions
# =============================================================================

class TestRecordSelfplayBatch:
    """Tests for record_selfplay_batch function."""

    def test_record_selfplay_batch(self):
        """Test recording selfplay batch metrics."""
        from app.metrics.orchestrator import record_selfplay_batch

        # Should not raise
        record_selfplay_batch(
            board_type="square8",
            num_players=2,
            games=100,
            duration_seconds=60.5,
            errors=0,
        )

    def test_record_selfplay_batch_with_errors(self):
        """Test recording selfplay batch with errors."""
        from app.metrics.orchestrator import record_selfplay_batch

        record_selfplay_batch(
            board_type="hex6",
            num_players=3,
            games=50,
            duration_seconds=30.0,
            errors=5,
        )

    def test_record_selfplay_batch_zero_duration(self):
        """Test recording batch with zero duration."""
        from app.metrics.orchestrator import record_selfplay_batch

        # Should not raise (handles division by zero)
        record_selfplay_batch(
            board_type="square8",
            num_players=2,
            games=10,
            duration_seconds=0,
        )


class TestRecordTrainingRun:
    """Tests for record_training_run function."""

    def test_record_training_run(self):
        """Test recording training run metrics."""
        from app.metrics.orchestrator import record_training_run

        record_training_run(
            board_type="square8",
            num_players=2,
            duration_seconds=3600,
            final_loss=0.25,
            final_accuracy=0.85,
            samples=10000,
        )

    def test_record_training_run_minimal(self):
        """Test recording training run with minimal fields."""
        from app.metrics.orchestrator import record_training_run

        record_training_run(
            board_type="square8",
            num_players=2,
            duration_seconds=1800,
            final_loss=0.30,
        )


class TestRecordEvaluation:
    """Tests for record_evaluation function."""

    def test_record_evaluation(self):
        """Test recording evaluation metrics."""
        from app.metrics.orchestrator import record_evaluation

        record_evaluation(
            board_type="square8",
            num_players=2,
            games=100,
            elo_delta=30.0,
            win_rate=0.58,
            duration_seconds=300,
        )

    def test_record_evaluation_minimal(self):
        """Test recording evaluation with minimal fields."""
        from app.metrics.orchestrator import record_evaluation

        record_evaluation(
            board_type="hex6",
            num_players=3,
            games=50,
            elo_delta=15.0,
        )


class TestRecordModelPromotion:
    """Tests for record_model_promotion function."""

    def test_record_model_promotion(self):
        """Test recording model promotion metrics."""
        from app.metrics.orchestrator import record_model_promotion

        record_model_promotion(
            board_type="square8",
            num_players=2,
            elo_gain=25.0,
            new_elo=1625.0,
        )

    def test_record_promotion_rejection(self):
        """Test recording promotion rejection."""
        from app.metrics.orchestrator import record_promotion_rejection

        record_promotion_rejection(
            board_type="square8",
            num_players=2,
            reason="insufficient_elo_gain",
        )


class TestRecordPipelineStage:
    """Tests for record_pipeline_stage function."""

    def test_record_pipeline_stage_success(self):
        """Test recording successful pipeline stage."""
        from app.metrics.orchestrator import record_pipeline_stage

        record_pipeline_stage(
            stage="selfplay",
            duration_seconds=300,
            success=True,
        )

    def test_record_pipeline_stage_failure(self):
        """Test recording failed pipeline stage."""
        from app.metrics.orchestrator import record_pipeline_stage

        record_pipeline_stage(
            stage="training",
            duration_seconds=100,
            success=False,
            error_type="OutOfMemoryError",
        )


class TestRecordDataSync:
    """Tests for record_data_sync function."""

    def test_record_data_sync_success(self):
        """Test recording successful data sync."""
        from app.metrics.orchestrator import record_data_sync

        record_data_sync(
            source="gh200-a",
            destination="h100-1",
            games=500,
            duration_seconds=30,
            success=True,
        )

    def test_record_data_sync_failure(self):
        """Test recording failed data sync."""
        from app.metrics.orchestrator import record_data_sync

        record_data_sync(
            source="gh200-a",
            destination="h100-1",
            games=0,
            duration_seconds=5,
            success=False,
            error_type="ConnectionError",
        )


class TestRecordModelSync:
    """Tests for record_model_sync function."""

    def test_record_model_sync(self):
        """Test recording model sync metrics."""
        from app.metrics.orchestrator import record_model_sync

        record_model_sync(
            model_type="nnue",
            duration_seconds=15.5,
            direction="push",
        )


class TestRecordSyncCoordinatorOp:
    """Tests for record_sync_coordinator_op function."""

    def test_record_sync_coordinator_op(self):
        """Test recording sync coordinator operation."""
        from app.metrics.orchestrator import record_sync_coordinator_op

        record_sync_coordinator_op(
            category="games",
            transport="aria2",
            files_synced=10,
            bytes_transferred=1024 * 1024,
            duration_seconds=5.0,
            success=True,
        )


# =============================================================================
# Test Pipeline State
# =============================================================================

class TestPipelineState:
    """Tests for pipeline state functions and constants."""

    def test_pipeline_state_constants(self):
        """Test pipeline state constants exist."""
        from app.metrics.orchestrator import (
            PIPELINE_EVALUATION,
            PIPELINE_IDLE,
            PIPELINE_PROMOTION,
            PIPELINE_SELFPLAY,
            PIPELINE_TRAINING,
        )

        assert PIPELINE_IDLE == 0
        assert PIPELINE_SELFPLAY == 1
        assert PIPELINE_TRAINING == 2
        assert PIPELINE_EVALUATION == 3
        assert PIPELINE_PROMOTION == 4

    def test_set_pipeline_state(self):
        """Test set_pipeline_state function."""
        from app.metrics.orchestrator import (
            PIPELINE_TRAINING,
            set_pipeline_state,
        )

        set_pipeline_state("unified_ai_loop", PIPELINE_TRAINING)


class TestTimePipelineStage:
    """Tests for time_pipeline_stage context manager."""

    def test_time_pipeline_stage_success(self):
        """Test timing successful pipeline stage."""
        from app.metrics.orchestrator import time_pipeline_stage

        with time_pipeline_stage("test_stage"):
            pass  # Simulate work

    def test_time_pipeline_stage_failure(self):
        """Test timing failed pipeline stage."""
        from app.metrics.orchestrator import time_pipeline_stage

        with pytest.raises(ValueError), time_pipeline_stage("test_stage"):
            raise ValueError("Test error")


# =============================================================================
# Test Queue/Iteration Tracking
# =============================================================================

class TestQueueTracking:
    """Tests for queue and iteration tracking."""

    def test_update_selfplay_queue_size(self):
        """Test updating selfplay queue size."""
        from app.metrics.orchestrator import update_selfplay_queue_size

        update_selfplay_queue_size(queue_size=10)

    def test_record_pipeline_iteration(self):
        """Test recording pipeline iteration."""
        from app.metrics.orchestrator import record_pipeline_iteration

        record_pipeline_iteration()

    def test_get_selfplay_queue_size(self):
        """Test getting selfplay queue size."""
        from app.metrics.orchestrator import (
            get_selfplay_queue_size,
            update_selfplay_queue_size,
        )

        update_selfplay_queue_size(5)
        size = get_selfplay_queue_size()
        # May return 0 or 5 depending on metric state
        assert isinstance(size, (int, float))


# =============================================================================
# Test Quality Metrics Functions
# =============================================================================

class TestQualityMetricsFunctions:
    """Tests for training data quality metric functions."""

    def test_record_training_data_quality(self):
        """Test recording training data quality metrics."""
        from app.metrics.orchestrator import record_training_data_quality

        record_training_data_quality(
            board_type="square8",
            num_players=2,
            avg_quality=0.75,
            min_quality=0.3,
            max_quality=0.95,
            high_quality_count=500,
            total_games=1000,
            avg_elo=1550.0,
            decisive_ratio=0.65,
        )

    def test_update_quality_bridge_status(self):
        """Test updating quality bridge status."""
        from app.metrics.orchestrator import update_quality_bridge_status

        update_quality_bridge_status(
            quality_lookup_size=1000,
            elo_lookup_size=800,
            refresh_age_seconds=60.0,
            avg_quality=0.72,
        )

    def test_record_high_quality_sync(self):
        """Test recording high quality sync."""
        from app.metrics.orchestrator import record_high_quality_sync

        record_high_quality_sync(games_synced=50, transport="aria2")

    def test_update_data_server_status(self):
        """Test updating data server status."""
        from app.metrics.orchestrator import update_data_server_status

        update_data_server_status(port=6800, running=True)
        update_data_server_status(port=6800, running=False)

    def test_update_sync_sources_count(self):
        """Test updating sync sources count."""
        from app.metrics.orchestrator import update_sync_sources_count

        update_sync_sources_count(count=3)

    def test_record_nfs_skip(self):
        """Test recording NFS skip."""
        from app.metrics.orchestrator import record_nfs_skip

        record_nfs_skip(category="games")


# =============================================================================
# Integration Tests
# =============================================================================

class TestMetricsIntegration:
    """Integration tests for orchestrator metrics."""

    def test_all_record_functions_callable(self):
        """Test all record functions are callable."""
        from app.metrics import orchestrator

        assert callable(orchestrator.record_selfplay_batch)
        assert callable(orchestrator.record_training_run)
        assert callable(orchestrator.record_evaluation)
        assert callable(orchestrator.record_model_promotion)
        assert callable(orchestrator.record_pipeline_stage)
        assert callable(orchestrator.record_data_sync)
        assert callable(orchestrator.record_model_sync)

    def test_full_pipeline_metrics_flow(self):
        """Test recording metrics for full pipeline."""
        from app.metrics.orchestrator import (
            PIPELINE_EVALUATION,
            PIPELINE_IDLE,
            PIPELINE_PROMOTION,
            PIPELINE_SELFPLAY,
            PIPELINE_TRAINING,
            record_evaluation,
            record_model_promotion,
            record_selfplay_batch,
            record_training_run,
            set_pipeline_state,
        )

        # Selfplay
        set_pipeline_state("test", PIPELINE_SELFPLAY)
        record_selfplay_batch("square8", 2, 100, 60.0)

        # Training
        set_pipeline_state("test", PIPELINE_TRAINING)
        record_training_run("square8", 2, 3600, 0.25)

        # Evaluation
        set_pipeline_state("test", PIPELINE_EVALUATION)
        record_evaluation("square8", 2, 100, 30.0)

        # Promotion
        set_pipeline_state("test", PIPELINE_PROMOTION)
        record_model_promotion("square8", 2, 30.0, 1630.0)

        # Idle
        set_pipeline_state("test", PIPELINE_IDLE)
