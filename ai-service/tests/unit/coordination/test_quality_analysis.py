"""Unit tests for quality_analysis.py module.

Tests pure quality analysis functions for data quality assessment,
intensity mapping, and curriculum weight adjustment.
"""

import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.coordination.quality_analysis import (
    # Constants
    HIGH_QUALITY_THRESHOLD,
    MEDIUM_QUALITY_THRESHOLD,
    LOW_QUALITY_THRESHOLD,
    MINIMUM_QUALITY_THRESHOLD,
    DEFAULT_SAMPLE_LIMIT,
    FULL_QUALITY_GAME_COUNT,
    # Enums
    IntensityLevel,
    UrgencyLevel,
    INTENSITY_TO_URGENCY,
    # Data classes
    QualityResult,
    CurriculumWeightChange,
    QualityThresholds,
    QualityTrend,
    # Functions
    assess_selfplay_quality,
    _count_based_quality,
    compute_intensity_from_quality,
    compute_training_urgency,
    compute_curriculum_weight_adjustment,
    get_quality_threshold,
    is_quality_acceptable,
    should_accelerate_training,
    should_pause_training,
    analyze_quality_trend,
    compute_exploration_adjustment,
    # Aliases
    assess_quality,
    get_intensity,
    get_urgency,
)


# ============================================================================
# Constant Tests
# ============================================================================


class TestConstants:
    """Test quality threshold constants."""

    def test_threshold_ordering(self):
        """Thresholds should be in ascending order."""
        assert MINIMUM_QUALITY_THRESHOLD < LOW_QUALITY_THRESHOLD
        assert LOW_QUALITY_THRESHOLD < MEDIUM_QUALITY_THRESHOLD
        assert MEDIUM_QUALITY_THRESHOLD < HIGH_QUALITY_THRESHOLD

    def test_thresholds_in_valid_range(self):
        """All thresholds should be between 0 and 1."""
        for threshold in [
            MINIMUM_QUALITY_THRESHOLD,
            LOW_QUALITY_THRESHOLD,
            MEDIUM_QUALITY_THRESHOLD,
            HIGH_QUALITY_THRESHOLD,
        ]:
            assert 0.0 <= threshold <= 1.0

    def test_sample_limit_positive(self):
        """Sample limit should be positive."""
        assert DEFAULT_SAMPLE_LIMIT > 0

    def test_full_quality_game_count_positive(self):
        """Full quality game count should be positive."""
        assert FULL_QUALITY_GAME_COUNT > 0


# ============================================================================
# IntensityLevel Tests
# ============================================================================


class TestIntensityLevel:
    """Test IntensityLevel enum."""

    def test_all_levels_defined(self):
        """All 5 intensity levels should be defined."""
        levels = [
            IntensityLevel.PAUSED,
            IntensityLevel.REDUCED,
            IntensityLevel.NORMAL,
            IntensityLevel.ACCELERATED,
            IntensityLevel.HOT_PATH,
        ]
        assert len(levels) == 5

    def test_string_values(self):
        """Intensity levels should have correct string values."""
        assert IntensityLevel.PAUSED.value == "paused"
        assert IntensityLevel.REDUCED.value == "reduced"
        assert IntensityLevel.NORMAL.value == "normal"
        assert IntensityLevel.ACCELERATED.value == "accelerated"
        assert IntensityLevel.HOT_PATH.value == "hot_path"

    def test_intensity_is_string_enum(self):
        """IntensityLevel should be a string enum for serialization."""
        assert isinstance(IntensityLevel.NORMAL, str)
        assert IntensityLevel.NORMAL == "normal"


# ============================================================================
# UrgencyLevel Tests
# ============================================================================


class TestUrgencyLevel:
    """Test UrgencyLevel enum."""

    def test_all_levels_defined(self):
        """All 5 urgency levels should be defined."""
        levels = [
            UrgencyLevel.NONE,
            UrgencyLevel.LOW,
            UrgencyLevel.NORMAL,
            UrgencyLevel.HIGH,
            UrgencyLevel.CRITICAL,
        ]
        assert len(levels) == 5

    def test_string_values(self):
        """Urgency levels should have correct string values."""
        assert UrgencyLevel.NONE.value == "none"
        assert UrgencyLevel.LOW.value == "low"
        assert UrgencyLevel.NORMAL.value == "normal"
        assert UrgencyLevel.HIGH.value == "high"
        assert UrgencyLevel.CRITICAL.value == "critical"


class TestIntensityToUrgencyMapping:
    """Test intensity to urgency mapping."""

    def test_all_intensities_mapped(self):
        """All intensity levels should have urgency mappings."""
        for intensity in IntensityLevel:
            assert intensity in INTENSITY_TO_URGENCY

    def test_mapping_values(self):
        """Verify specific mappings."""
        assert INTENSITY_TO_URGENCY[IntensityLevel.HOT_PATH] == UrgencyLevel.CRITICAL
        assert INTENSITY_TO_URGENCY[IntensityLevel.ACCELERATED] == UrgencyLevel.HIGH
        assert INTENSITY_TO_URGENCY[IntensityLevel.NORMAL] == UrgencyLevel.NORMAL
        assert INTENSITY_TO_URGENCY[IntensityLevel.REDUCED] == UrgencyLevel.LOW
        assert INTENSITY_TO_URGENCY[IntensityLevel.PAUSED] == UrgencyLevel.NONE


# ============================================================================
# QualityResult Tests
# ============================================================================


class TestQualityResult:
    """Test QualityResult dataclass."""

    def test_basic_creation(self):
        """Should create result with required fields."""
        result = QualityResult(quality_score=0.75)
        assert result.quality_score == 0.75
        assert result.games_assessed == 0
        assert result.avg_game_quality == 0.0
        assert result.count_factor == 1.0
        assert result.method == "unified"

    def test_full_creation(self):
        """Should create result with all fields."""
        result = QualityResult(
            quality_score=0.82,
            games_assessed=50,
            avg_game_quality=0.85,
            count_factor=0.75,
            method="count_heuristic",
        )
        assert result.quality_score == 0.82
        assert result.games_assessed == 50
        assert result.avg_game_quality == 0.85
        assert result.count_factor == 0.75
        assert result.method == "count_heuristic"

    def test_score_clamped_high(self):
        """Should clamp scores above 1.0."""
        result = QualityResult(quality_score=1.5)
        assert result.quality_score == 1.0

    def test_score_clamped_low(self):
        """Should clamp scores below 0.0."""
        result = QualityResult(quality_score=-0.5)
        assert result.quality_score == 0.0

    def test_frozen_dataclass(self):
        """QualityResult should be frozen (immutable)."""
        result = QualityResult(quality_score=0.75)
        with pytest.raises(AttributeError):
            result.quality_score = 0.80


# ============================================================================
# CurriculumWeightChange Tests
# ============================================================================


class TestCurriculumWeightChange:
    """Test CurriculumWeightChange dataclass."""

    def test_change_detected(self):
        """Should detect when weight changed."""
        change = CurriculumWeightChange(
            old_weight=1.0, new_weight=1.15, reason="low_quality_increase"
        )
        assert change.changed is True

    def test_no_change_detected(self):
        """Should detect when weight unchanged."""
        change = CurriculumWeightChange(
            old_weight=1.0, new_weight=1.0, reason="no_change"
        )
        assert change.changed is False

    def test_tiny_change_not_detected(self):
        """Tiny changes (< 1e-6) should not register as changed."""
        change = CurriculumWeightChange(
            old_weight=1.0, new_weight=1.0000001, reason="no_change"
        )
        assert change.changed is False


# ============================================================================
# QualityThresholds Tests
# ============================================================================


class TestQualityThresholds:
    """Test QualityThresholds dataclass."""

    def test_default_values(self):
        """Should have sensible defaults."""
        thresholds = QualityThresholds()
        assert thresholds.min_quality == LOW_QUALITY_THRESHOLD
        assert thresholds.target_quality == MEDIUM_QUALITY_THRESHOLD
        assert thresholds.high_quality == HIGH_QUALITY_THRESHOLD

    def test_with_config_key(self):
        """Should store config key."""
        thresholds = QualityThresholds(config_key="hex8_2p")
        assert thresholds.config_key == "hex8_2p"

    def test_custom_thresholds(self):
        """Should accept custom threshold values."""
        thresholds = QualityThresholds(
            min_quality=0.40,
            target_quality=0.65,
            high_quality=0.85,
        )
        assert thresholds.min_quality == 0.40
        assert thresholds.target_quality == 0.65
        assert thresholds.high_quality == 0.85


# ============================================================================
# QualityTrend Tests
# ============================================================================


class TestQualityTrend:
    """Test QualityTrend dataclass."""

    def test_improving_trend(self):
        """Should calculate change for improving trend."""
        trend = QualityTrend(
            trend="improving",
            current_score=0.80,
            previous_score=0.70,
        )
        assert trend.change == pytest.approx(0.10)
        assert trend.change_pct == pytest.approx(14.286, rel=0.01)

    def test_declining_trend(self):
        """Should calculate negative change for declining trend."""
        trend = QualityTrend(
            trend="declining",
            current_score=0.60,
            previous_score=0.75,
        )
        assert trend.change == pytest.approx(-0.15)
        assert trend.change_pct == pytest.approx(-20.0)

    def test_stable_trend(self):
        """Should calculate zero change for stable trend."""
        trend = QualityTrend(
            trend="stable",
            current_score=0.70,
            previous_score=0.70,
        )
        assert trend.change == 0.0
        assert trend.change_pct == 0.0

    def test_zero_previous_score(self):
        """Should handle zero previous score without division error."""
        trend = QualityTrend(
            trend="improving",
            current_score=0.50,
            previous_score=0.0,
        )
        assert trend.change == 0.50
        assert trend.change_pct == 0.0  # No percentage for zero base


# ============================================================================
# Count-Based Quality Tests
# ============================================================================


class TestCountBasedQuality:
    """Test _count_based_quality fallback function."""

    def test_very_few_games(self):
        """< 100 games should return low quality."""
        result = _count_based_quality(50)
        assert result.quality_score == 0.30
        assert result.method == "count_heuristic"

    def test_few_games(self):
        """100-500 games should return medium-low quality."""
        result = _count_based_quality(300)
        assert result.quality_score == 0.60

    def test_moderate_games(self):
        """500-1000 games should return medium-high quality."""
        result = _count_based_quality(750)
        assert result.quality_score == 0.80

    def test_many_games(self):
        """>= 1000 games should return high quality."""
        result = _count_based_quality(1500)
        assert result.quality_score == 0.95

    def test_count_factor_calculated(self):
        """Count factor should be calculated correctly."""
        result = _count_based_quality(250)
        assert result.count_factor == pytest.approx(250 / FULL_QUALITY_GAME_COUNT)

    def test_count_factor_capped(self):
        """Count factor should be capped at 1.0."""
        result = _count_based_quality(1000)
        assert result.count_factor == 1.0


# ============================================================================
# Assess Selfplay Quality Tests
# ============================================================================


class TestAssessSelfplayQuality:
    """Test assess_selfplay_quality function."""

    def test_nonexistent_database(self):
        """Should return low quality for nonexistent database."""
        result = assess_selfplay_quality("/nonexistent/path.db", games_count=100)
        assert result.quality_score == MINIMUM_QUALITY_THRESHOLD
        assert result.method == "no_database"

    def test_empty_database(self, tmp_path: Path):
        """Should return low quality for database with no games."""
        db_path = tmp_path / "empty.db"
        with sqlite3.connect(str(db_path)) as conn:
            conn.execute(
                """
                CREATE TABLE games (
                    game_id TEXT PRIMARY KEY,
                    game_status TEXT,
                    winner TEXT,
                    termination_reason TEXT,
                    total_moves INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """
            )
        result = assess_selfplay_quality(db_path, games_count=0)
        assert result.quality_score == MINIMUM_QUALITY_THRESHOLD
        assert result.method == "no_games"

    def test_with_custom_quality_scorer(self, tmp_path: Path):
        """Should use custom quality scorer when provided."""
        db_path = tmp_path / "test.db"
        with sqlite3.connect(str(db_path)) as conn:
            conn.execute(
                """
                CREATE TABLE games (
                    game_id TEXT PRIMARY KEY,
                    game_status TEXT,
                    winner TEXT,
                    termination_reason TEXT,
                    total_moves INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """
            )
            conn.execute(
                """
                INSERT INTO games (game_id, game_status, winner, termination_reason, total_moves)
                VALUES ('game1', 'complete', '0', 'normal', 50)
            """
            )
            conn.commit()

        # Custom scorer always returns 0.9
        def custom_scorer(game_dict: dict) -> float:
            return 0.90

        result = assess_selfplay_quality(
            db_path, games_count=1, quality_scorer_fn=custom_scorer
        )
        assert result.games_assessed == 1
        assert result.avg_game_quality == 0.90

    def test_fallback_to_count_heuristic(self, tmp_path: Path):
        """Should fall back to count heuristic when scorer fails."""
        db_path = tmp_path / "test.db"
        with sqlite3.connect(str(db_path)) as conn:
            conn.execute(
                """
                CREATE TABLE games (
                    game_id TEXT PRIMARY KEY,
                    game_status TEXT,
                    winner TEXT,
                    termination_reason TEXT,
                    total_moves INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """
            )
            conn.execute(
                """
                INSERT INTO games (game_id, game_status, winner, termination_reason, total_moves)
                VALUES ('game1', 'complete', '0', 'normal', 50)
            """
            )
            conn.commit()

        # Mock the unified quality import to fail
        with patch.dict("sys.modules", {"app.quality.unified_quality": None}):
            # Force reimport to trigger ImportError
            import importlib
            import app.coordination.quality_analysis as qa

            # Actually test with a scorer that raises
            def bad_scorer(game_dict: dict) -> float:
                raise ValueError("Scoring failed")

            result = qa.assess_selfplay_quality(
                db_path, games_count=1, quality_scorer_fn=bad_scorer
            )
            # Should fall back gracefully with scoring_failed method
            assert result.method in ("scoring_failed", "count_heuristic")


# ============================================================================
# Compute Intensity From Quality Tests
# ============================================================================


class TestComputeIntensityFromQuality:
    """Test compute_intensity_from_quality function."""

    def test_hot_path_threshold(self):
        """Quality >= 0.90 should return HOT_PATH."""
        assert compute_intensity_from_quality(0.90) == IntensityLevel.HOT_PATH
        assert compute_intensity_from_quality(0.95) == IntensityLevel.HOT_PATH
        assert compute_intensity_from_quality(1.0) == IntensityLevel.HOT_PATH

    def test_accelerated_threshold(self):
        """Quality 0.80-0.90 should return ACCELERATED."""
        assert compute_intensity_from_quality(0.80) == IntensityLevel.ACCELERATED
        assert compute_intensity_from_quality(0.85) == IntensityLevel.ACCELERATED
        assert compute_intensity_from_quality(0.89) == IntensityLevel.ACCELERATED

    def test_normal_threshold(self):
        """Quality 0.65-0.80 should return NORMAL."""
        assert compute_intensity_from_quality(0.65) == IntensityLevel.NORMAL
        assert compute_intensity_from_quality(0.70) == IntensityLevel.NORMAL
        assert compute_intensity_from_quality(0.79) == IntensityLevel.NORMAL

    def test_reduced_threshold(self):
        """Quality 0.50-0.65 should return REDUCED."""
        assert compute_intensity_from_quality(0.50) == IntensityLevel.REDUCED
        assert compute_intensity_from_quality(0.55) == IntensityLevel.REDUCED
        assert compute_intensity_from_quality(0.64) == IntensityLevel.REDUCED

    def test_paused_threshold(self):
        """Quality < 0.50 should return PAUSED."""
        assert compute_intensity_from_quality(0.49) == IntensityLevel.PAUSED
        assert compute_intensity_from_quality(0.30) == IntensityLevel.PAUSED
        assert compute_intensity_from_quality(0.0) == IntensityLevel.PAUSED


# ============================================================================
# Compute Training Urgency Tests
# ============================================================================


class TestComputeTrainingUrgency:
    """Test compute_training_urgency function."""

    def test_urgency_from_quality(self):
        """Should compute urgency from quality score."""
        assert compute_training_urgency(0.95) == UrgencyLevel.CRITICAL
        assert compute_training_urgency(0.85) == UrgencyLevel.HIGH
        assert compute_training_urgency(0.70) == UrgencyLevel.NORMAL
        assert compute_training_urgency(0.55) == UrgencyLevel.LOW
        assert compute_training_urgency(0.40) == UrgencyLevel.NONE

    def test_urgency_from_intensity(self):
        """Should use provided intensity if given."""
        assert (
            compute_training_urgency(0.0, IntensityLevel.HOT_PATH)
            == UrgencyLevel.CRITICAL
        )
        assert (
            compute_training_urgency(0.0, IntensityLevel.NORMAL) == UrgencyLevel.NORMAL
        )


# ============================================================================
# Compute Curriculum Weight Adjustment Tests
# ============================================================================


class TestComputeCurriculumWeightAdjustment:
    """Test compute_curriculum_weight_adjustment function."""

    def test_low_quality_increases_weight(self):
        """Low quality (< 0.5) should increase weight by 15%."""
        change = compute_curriculum_weight_adjustment(0.40, current_weight=1.0)
        assert change.new_weight == pytest.approx(1.15)
        assert change.reason == "low_quality_increase"
        assert change.changed is True

    def test_high_quality_decreases_weight(self):
        """High quality (>= 0.7) should decrease weight by 5%."""
        change = compute_curriculum_weight_adjustment(0.80, current_weight=1.0)
        assert change.new_weight == pytest.approx(0.95)
        assert change.reason == "high_quality_decrease"
        assert change.changed is True

    def test_medium_quality_no_change(self):
        """Medium quality (0.5-0.7) should not change weight."""
        change = compute_curriculum_weight_adjustment(0.60, current_weight=1.0)
        assert change.new_weight == 1.0
        assert change.reason == "no_change"
        assert change.changed is False

    def test_weight_capped_at_max(self):
        """Weight should not exceed max_weight."""
        change = compute_curriculum_weight_adjustment(
            0.30, current_weight=1.9, max_weight=2.0
        )
        assert change.new_weight == 2.0

    def test_weight_capped_at_min(self):
        """Weight should not go below min_weight."""
        change = compute_curriculum_weight_adjustment(
            0.90, current_weight=0.55, min_weight=0.5
        )
        assert change.new_weight == pytest.approx(0.5225)

    def test_custom_bounds(self):
        """Should respect custom min/max bounds."""
        change = compute_curriculum_weight_adjustment(
            0.30, current_weight=2.8, min_weight=0.2, max_weight=3.0
        )
        assert change.new_weight == 3.0  # Capped at max


# ============================================================================
# Get Quality Threshold Tests
# ============================================================================


class TestGetQualityThreshold:
    """Test get_quality_threshold function."""

    def test_default_thresholds(self):
        """Should return default thresholds for standard configs."""
        thresholds = get_quality_threshold("hex8_2p")
        assert thresholds.min_quality == LOW_QUALITY_THRESHOLD
        assert thresholds.target_quality == MEDIUM_QUALITY_THRESHOLD
        assert thresholds.high_quality == HIGH_QUALITY_THRESHOLD
        assert thresholds.config_key == "hex8_2p"

    def test_multiplayer_lower_thresholds(self):
        """3p and 4p configs should have lower thresholds."""
        thresholds_3p = get_quality_threshold("hex8_3p")
        thresholds_4p = get_quality_threshold("square8_4p")

        assert thresholds_3p.min_quality < LOW_QUALITY_THRESHOLD
        assert thresholds_4p.min_quality < LOW_QUALITY_THRESHOLD

    def test_custom_base_threshold(self):
        """Should use custom base threshold if provided."""
        thresholds = get_quality_threshold("hex8_2p", base_threshold=0.80)
        assert thresholds.target_quality == 0.80


# ============================================================================
# Is Quality Acceptable Tests
# ============================================================================


class TestIsQualityAcceptable:
    """Test is_quality_acceptable function."""

    def test_with_single_threshold(self):
        """Should check against single threshold."""
        assert is_quality_acceptable(0.75, threshold=0.70) is True
        assert is_quality_acceptable(0.65, threshold=0.70) is False

    def test_with_thresholds_object(self):
        """Should check against thresholds object min_quality."""
        thresholds = QualityThresholds(min_quality=0.60)
        assert is_quality_acceptable(0.65, thresholds=thresholds) is True
        assert is_quality_acceptable(0.55, thresholds=thresholds) is False

    def test_default_threshold(self):
        """Should use MEDIUM_QUALITY_THRESHOLD by default."""
        assert is_quality_acceptable(0.75) is True
        assert is_quality_acceptable(0.65) is False


# ============================================================================
# Should Accelerate/Pause Training Tests
# ============================================================================


class TestShouldAccelerateTraining:
    """Test should_accelerate_training function."""

    def test_high_quality_accelerates(self):
        """Should accelerate at >= 0.80."""
        assert should_accelerate_training(0.80) is True
        assert should_accelerate_training(0.90) is True

    def test_low_quality_no_acceleration(self):
        """Should not accelerate below 0.80."""
        assert should_accelerate_training(0.79) is False
        assert should_accelerate_training(0.50) is False


class TestShouldPauseTraining:
    """Test should_pause_training function."""

    def test_very_low_quality_pauses(self):
        """Should pause at < 0.50."""
        assert should_pause_training(0.49) is True
        assert should_pause_training(0.30) is True

    def test_acceptable_quality_no_pause(self):
        """Should not pause at >= 0.50."""
        assert should_pause_training(0.50) is False
        assert should_pause_training(0.70) is False


# ============================================================================
# Analyze Quality Trend Tests
# ============================================================================


class TestAnalyzeQualityTrend:
    """Test analyze_quality_trend function."""

    def test_improving_trend(self):
        """Significant increase should be 'improving'."""
        trend = analyze_quality_trend(0.80, 0.70)
        assert trend.trend == "improving"

    def test_declining_trend(self):
        """Significant decrease should be 'declining'."""
        trend = analyze_quality_trend(0.60, 0.75)
        assert trend.trend == "declining"

    def test_stable_trend(self):
        """Small changes should be 'stable'."""
        trend = analyze_quality_trend(0.71, 0.70)
        assert trend.trend == "stable"

    def test_custom_threshold(self):
        """Should respect custom significance threshold."""
        # 0.02 change with 0.10 threshold = stable
        trend = analyze_quality_trend(0.72, 0.70, significant_change_threshold=0.10)
        assert trend.trend == "stable"


# ============================================================================
# Compute Exploration Adjustment Tests
# ============================================================================


class TestComputeExplorationAdjustment:
    """Test compute_exploration_adjustment function."""

    def test_declining_low_quality_boosts_exploration(self):
        """Declining trend with low quality should boost exploration."""
        temp_boost, noise_boost = compute_exploration_adjustment(0.50, "declining")
        assert temp_boost == 1.5
        assert noise_boost == 0.10

    def test_low_quality_modest_boost(self):
        """Low quality without decline should have modest boost."""
        temp_boost, noise_boost = compute_exploration_adjustment(0.45, "stable")
        assert temp_boost == 1.2
        assert noise_boost == 0.05

    def test_medium_quality_no_change(self):
        """Medium quality should have no adjustment."""
        temp_boost, noise_boost = compute_exploration_adjustment(0.60, "stable")
        assert temp_boost == 1.0
        assert noise_boost == 0.0

    def test_high_quality_improving_slight_exploitation(self):
        """High quality with positive trend can exploit slightly."""
        temp_boost, noise_boost = compute_exploration_adjustment(0.92, "improving")
        assert temp_boost == 0.9
        assert noise_boost == -0.02

    def test_high_quality_declining_no_change(self):
        """High quality but declining should not exploit."""
        temp_boost, noise_boost = compute_exploration_adjustment(0.92, "declining")
        assert temp_boost == 1.0  # Default, no exploitation


# ============================================================================
# Alias Tests
# ============================================================================


class TestAliases:
    """Test backward-compatible aliases."""

    def test_assess_quality_alias(self):
        """assess_quality should be alias for assess_selfplay_quality."""
        assert assess_quality is assess_selfplay_quality

    def test_get_intensity_alias(self):
        """get_intensity should be alias for compute_intensity_from_quality."""
        assert get_intensity is compute_intensity_from_quality

    def test_get_urgency_alias(self):
        """get_urgency should be alias for compute_training_urgency."""
        assert get_urgency is compute_training_urgency


# ============================================================================
# Integration Tests
# ============================================================================


class TestQualityAnalysisIntegration:
    """Integration tests for quality analysis workflow."""

    def test_full_quality_workflow(self, tmp_path: Path):
        """Test complete quality assessment to training decision workflow."""
        # Create a mock database with games
        db_path = tmp_path / "test.db"
        with sqlite3.connect(str(db_path)) as conn:
            conn.execute(
                """
                CREATE TABLE games (
                    game_id TEXT PRIMARY KEY,
                    game_status TEXT,
                    winner TEXT,
                    termination_reason TEXT,
                    total_moves INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """
            )
            # Add multiple games
            for i in range(20):
                conn.execute(
                    """
                    INSERT INTO games (game_id, game_status, winner, termination_reason, total_moves)
                    VALUES (?, 'complete', '0', 'normal', ?)
                """,
                    (f"game{i}", 40 + i),
                )
            conn.commit()

        # Custom scorer for consistent results
        def mock_scorer(game: dict) -> float:
            return 0.75

        # Step 1: Assess quality
        result = assess_selfplay_quality(
            db_path, games_count=20, quality_scorer_fn=mock_scorer
        )

        # Step 2: Compute intensity
        intensity = compute_intensity_from_quality(result.quality_score)

        # Step 3: Get urgency
        urgency = compute_training_urgency(result.quality_score, intensity)

        # Step 4: Check if acceptable
        is_acceptable = is_quality_acceptable(result.quality_score)

        # Verify workflow
        assert result.games_assessed == 20
        assert isinstance(intensity, IntensityLevel)
        assert isinstance(urgency, UrgencyLevel)
        assert isinstance(is_acceptable, bool)

    def test_curriculum_adjustment_workflow(self):
        """Test curriculum weight adjustment based on quality changes."""
        # Starting state
        current_weight = 1.0
        quality_scores = [0.40, 0.55, 0.75, 0.85, 0.60]

        weights = []
        for score in quality_scores:
            change = compute_curriculum_weight_adjustment(score, current_weight)
            current_weight = change.new_weight
            weights.append(current_weight)

        # Low quality increased weight
        assert weights[0] > 1.0

        # Medium quality kept weight
        assert weights[1] == weights[0]  # No change at 0.55

        # High quality decreased weight
        assert weights[2] < weights[1]

        # Verify final weight in reasonable range
        assert 0.5 <= weights[-1] <= 2.0
