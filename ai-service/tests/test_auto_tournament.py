"""
Tests for Automated Tournament Pipeline

Tests cover:
- Elo rating calculation
- Statistical significance (binomial p-value)
- Champion promotion logic
- Tournament execution with mock AI
- Report generation
- Model registration and management
"""

import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from app.training.auto_tournament import (
    AutoTournamentPipeline,
    calculate_binomial_p_value,
    calculate_elo_change,
    expected_score,
    RegisteredModel,
    ChallengerResult,
    should_promote,
    _binomial_coefficient,
)
from app.training.model_versioning import (
    ModelMetadata,
    save_model_checkpoint,
)


# =============================================================================
# Test Fixtures
# =============================================================================


class SimpleTestModel(nn.Module):
    """Simple test model for mocking neural network checkpoints."""

    ARCHITECTURE_VERSION = "v1.0.0"

    def __init__(self, input_size: int = 10, hidden_size: int = 20):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


@pytest.fixture
def temp_dir():
    """Create temporary directory for test artifacts."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def models_dir(temp_dir):
    """Create models subdirectory."""
    path = os.path.join(temp_dir, "models")
    os.makedirs(path)
    return path


@pytest.fixture
def results_dir(temp_dir):
    """Create results subdirectory."""
    path = os.path.join(temp_dir, "results")
    os.makedirs(path)
    return path


@pytest.fixture
def pipeline(models_dir, results_dir):
    """Create a fresh pipeline for testing."""
    return AutoTournamentPipeline(
        models_dir=models_dir,
        results_dir=results_dir,
    )


@pytest.fixture
def sample_model_checkpoint(models_dir):
    """Create a sample versioned checkpoint."""
    model = SimpleTestModel()
    path = os.path.join(models_dir, "model_v1.pth")
    save_model_checkpoint(
        model,
        path,
        training_info={"epochs": 100, "loss": 0.05},
    )
    return path


@pytest.fixture
def sample_metadata():
    """Create sample model metadata."""
    return ModelMetadata(
        architecture_version="v1.0.0",
        model_class="SimpleTestModel",
        config={"input_size": 10, "hidden_size": 20},
        training_info={"epochs": 100},
        checksum="abc123" * 10,
    )


# =============================================================================
# Elo Calculation Tests
# =============================================================================


class TestEloCalculation:
    """Tests for Elo rating calculation."""

    def test_equal_rating_expected_score(self):
        """Equal ratings should give 0.5 expected score."""
        assert expected_score(1500, 1500) == pytest.approx(0.5)

    def test_higher_rating_advantage(self):
        """Higher rating should have better expected score."""
        score = expected_score(1600, 1500)
        assert score > 0.5
        assert score < 1.0

    def test_lower_rating_disadvantage(self):
        """Lower rating should have worse expected score."""
        score = expected_score(1400, 1500)
        assert score < 0.5
        assert score > 0.0

    def test_elo_symmetry(self):
        """Expected scores should be symmetric."""
        score_a = expected_score(1500, 1600)
        score_b = expected_score(1600, 1500)
        assert score_a + score_b == pytest.approx(1.0)

    def test_elo_change_win(self):
        """Winner should gain rating, loser should lose rating."""
        new_a, new_b = calculate_elo_change(1500, 1500, 1.0, k_factor=32)
        assert new_a > 1500  # Winner gained
        assert new_b < 1500  # Loser lost
        assert new_a + new_b == pytest.approx(3000)  # Total preserved

    def test_elo_change_draw(self):
        """Equal-rated draw should not change ratings much."""
        new_a, new_b = calculate_elo_change(1500, 1500, 0.5, k_factor=32)
        assert new_a == pytest.approx(1500)
        assert new_b == pytest.approx(1500)

    def test_elo_change_upset(self):
        """Upset win should give bigger rating change."""
        # Low-rated beats high-rated
        low_win_a, _ = calculate_elo_change(1400, 1600, 1.0, k_factor=32)

        # High-rated beats low-rated
        high_win_a, _ = calculate_elo_change(1600, 1400, 1.0, k_factor=32)

        # Upset should give bigger gain
        upset_gain = low_win_a - 1400
        normal_gain = high_win_a - 1600
        assert upset_gain > normal_gain

    def test_k_factor_effect(self):
        """Higher K-factor should give bigger rating changes."""
        small_k_a, _ = calculate_elo_change(1500, 1500, 1.0, k_factor=16)
        large_k_a, _ = calculate_elo_change(1500, 1500, 1.0, k_factor=32)

        assert large_k_a - 1500 == pytest.approx(2 * (small_k_a - 1500))


# =============================================================================
# Statistical Significance Tests
# =============================================================================


class TestBinomialPValue:
    """Tests for binomial p-value calculation."""

    def test_binomial_coefficient_simple(self):
        """Test basic binomial coefficient calculation."""
        assert _binomial_coefficient(5, 0) == 1
        assert _binomial_coefficient(5, 1) == 5
        assert _binomial_coefficient(5, 2) == 10
        assert _binomial_coefficient(5, 5) == 1
        assert _binomial_coefficient(10, 5) == 252

    def test_binomial_coefficient_edge_cases(self):
        """Test edge cases for binomial coefficient."""
        assert _binomial_coefficient(0, 0) == 1
        assert _binomial_coefficient(5, 6) == 0  # k > n
        assert _binomial_coefficient(5, -1) == 0  # k < 0

    def test_perfect_win_p_value(self):
        """100% win rate should have very low p-value."""
        p = calculate_binomial_p_value(10, 10, 0.5)
        assert p < 0.01  # Very significant

    def test_half_win_p_value(self):
        """50% win rate should not be significant."""
        p = calculate_binomial_p_value(5, 10, 0.5)
        assert p > 0.3  # Not significant

    def test_zero_trials_p_value(self):
        """Zero trials should return p=1."""
        p = calculate_binomial_p_value(0, 0, 0.5)
        assert p == 1.0

    def test_marginal_significance(self):
        """Test around the significance boundary."""
        # 55% on 100 games (55 wins, 45 losses = 55 wins in 100)
        p = calculate_binomial_p_value(55, 100, 0.5)
        # Should be around 0.18 (not quite significant)
        assert 0.1 < p < 0.3

        # 60 wins in 100 games
        p = calculate_binomial_p_value(60, 100, 0.5)
        # Should be significant
        assert p < 0.05

    def test_p_value_decreases_with_more_wins(self):
        """P-value should decrease as win rate increases."""
        p_50 = calculate_binomial_p_value(50, 100, 0.5)
        p_55 = calculate_binomial_p_value(55, 100, 0.5)
        p_60 = calculate_binomial_p_value(60, 100, 0.5)

        assert p_50 > p_55 > p_60


# =============================================================================
# Champion Promotion Logic Tests
# =============================================================================


class TestPromotionLogic:
    """Tests for champion promotion decision logic."""

    def test_promotion_all_criteria_met(self):
        """Should promote when all criteria are met."""
        result = ChallengerResult(
            challenger_id="new",
            champion_id="old",
            challenger_wins=35,
            champion_wins=15,
            draws=0,
            total_games=50,
            challenger_win_rate=0.70,
            champion_win_rate=0.30,
            statistical_p_value=0.001,
            is_statistically_significant=True,
            challenger_final_elo=1550,
            champion_final_elo=1450,
            should_promote=True,
        )
        assert should_promote(result) is True

    def test_no_promotion_low_win_rate(self):
        """Should not promote with win rate < 55%."""
        result = ChallengerResult(
            challenger_id="new",
            champion_id="old",
            challenger_wins=27,
            champion_wins=23,
            draws=0,
            total_games=50,
            challenger_win_rate=0.54,  # Below threshold
            champion_win_rate=0.46,
            statistical_p_value=0.30,
            is_statistically_significant=False,
            challenger_final_elo=1510,
            champion_final_elo=1490,
            should_promote=False,
        )
        assert should_promote(result) is False

    def test_no_promotion_not_significant(self):
        """Should not promote without statistical significance."""
        result = ChallengerResult(
            challenger_id="new",
            champion_id="old",
            challenger_wins=30,
            champion_wins=20,
            draws=0,
            total_games=50,
            challenger_win_rate=0.60,
            champion_win_rate=0.40,
            statistical_p_value=0.10,  # Not significant
            is_statistically_significant=False,
            challenger_final_elo=1540,
            champion_final_elo=1460,
            should_promote=False,
        )
        assert should_promote(result) is False

    def test_no_promotion_lower_elo(self):
        """Should not promote if challenger has lower Elo."""
        result = ChallengerResult(
            challenger_id="new",
            champion_id="old",
            challenger_wins=28,
            champion_wins=22,
            draws=0,
            total_games=50,
            challenger_win_rate=0.56,
            champion_win_rate=0.44,
            statistical_p_value=0.04,
            is_statistically_significant=True,
            challenger_final_elo=1480,  # Lower!
            champion_final_elo=1520,
            should_promote=False,
        )
        assert should_promote(result) is False


# =============================================================================
# RegisteredModel Tests
# =============================================================================


class TestRegisteredModel:
    """Tests for RegisteredModel dataclass."""

    def test_win_rate_calculation(self, sample_metadata):
        """Test win rate property calculation."""
        model = RegisteredModel(
            model_id="test",
            model_path="/path/to/model",
            metadata=sample_metadata,
            games_played=100,
            wins=60,
            losses=35,
            draws=5,
        )
        assert model.win_rate == pytest.approx(60.0)

    def test_win_rate_zero_games(self, sample_metadata):
        """Win rate should be 0 with no games played."""
        model = RegisteredModel(
            model_id="test",
            model_path="/path/to/model",
            metadata=sample_metadata,
        )
        assert model.win_rate == 0.0

    def test_serialization_roundtrip(self, sample_metadata):
        """Test model can be serialized and deserialized."""
        model = RegisteredModel(
            model_id="test",
            model_path="/path/to/model",
            metadata=sample_metadata,
            elo_rating=1600,
            is_champion=True,
            games_played=50,
            wins=30,
            losses=15,
            draws=5,
        )

        data = model.to_dict()
        restored = RegisteredModel.from_dict(data)

        assert restored.model_id == model.model_id
        assert restored.elo_rating == model.elo_rating
        assert restored.is_champion == model.is_champion
        assert restored.metadata.architecture_version == \
            model.metadata.architecture_version


# =============================================================================
# AutoTournamentPipeline Tests
# =============================================================================


class TestAutoTournamentPipeline:
    """Tests for AutoTournamentPipeline class."""

    def test_initialization(self, pipeline, models_dir, results_dir):
        """Test pipeline initializes correctly."""
        assert pipeline.models_dir == models_dir
        assert pipeline.results_dir == results_dir
        assert len(pipeline._models) == 0

    def test_register_model(self, pipeline, sample_model_checkpoint):
        """Test model registration."""
        model_id = pipeline.register_model(sample_model_checkpoint)

        assert model_id is not None
        assert model_id in pipeline._models

        model = pipeline.get_model(model_id)
        assert model is not None
        assert model.model_path == sample_model_checkpoint
        assert model.elo_rating == 1500.0
        assert model.is_champion is True  # First model is champion

    def test_register_multiple_models(self, pipeline, models_dir):
        """Test registering multiple models."""
        # Create two model checkpoints
        for i in range(2):
            model = SimpleTestModel()
            path = os.path.join(models_dir, f"model_{i}.pth")
            save_model_checkpoint(model, path)

        id1 = pipeline.register_model(
            os.path.join(models_dir, "model_0.pth")
        )
        id2 = pipeline.register_model(
            os.path.join(models_dir, "model_1.pth")
        )

        assert len(pipeline._models) == 2
        assert pipeline.get_model(id1).is_champion is True
        assert pipeline.get_model(id2).is_champion is False

    def test_duplicate_registration(self, pipeline, sample_model_checkpoint):
        """Same checkpoint should return same ID."""
        id1 = pipeline.register_model(sample_model_checkpoint)
        id2 = pipeline.register_model(sample_model_checkpoint)

        assert id1 == id2
        assert len(pipeline._models) == 1

    def test_get_champion(self, pipeline, sample_model_checkpoint):
        """Test getting current champion."""
        pipeline.register_model(sample_model_checkpoint)
        champion = pipeline.get_champion()

        assert champion is not None
        assert champion.is_champion is True

    def test_promote_champion(self, pipeline, models_dir):
        """Test champion promotion."""
        # Create two models
        for i in range(2):
            model = SimpleTestModel()
            path = os.path.join(models_dir, f"model_{i}.pth")
            save_model_checkpoint(model, path)

        id1 = pipeline.register_model(
            os.path.join(models_dir, "model_0.pth")
        )
        id2 = pipeline.register_model(
            os.path.join(models_dir, "model_1.pth")
        )

        # Initially model_0 is champion
        assert pipeline.get_model(id1).is_champion is True
        assert pipeline.get_model(id2).is_champion is False

        # Promote model_1
        pipeline.promote_champion(id2)

        assert pipeline.get_model(id1).is_champion is False
        assert pipeline.get_model(id2).is_champion is True

    def test_get_elo_rankings(self, pipeline, models_dir):
        """Test Elo rankings are sorted correctly."""
        # Create models with different ratings
        for i in range(3):
            model = SimpleTestModel()
            path = os.path.join(models_dir, f"model_{i}.pth")
            save_model_checkpoint(model, path)

        ids = []
        for i in range(3):
            model_id = pipeline.register_model(
                os.path.join(models_dir, f"model_{i}.pth"),
                initial_elo=1400 + i * 100,  # 1400, 1500, 1600
            )
            ids.append(model_id)

        rankings = pipeline.get_elo_rankings()

        # Should be sorted by Elo descending
        assert len(rankings) == 3
        assert rankings[0][1] == 1600
        assert rankings[1][1] == 1500
        assert rankings[2][1] == 1400

    def test_list_models(self, pipeline, sample_model_checkpoint):
        """Test listing all models."""
        pipeline.register_model(sample_model_checkpoint)
        models = pipeline.list_models()

        assert len(models) == 1
        assert models[0].model_path == sample_model_checkpoint

    def test_registry_persistence(self, models_dir, results_dir):
        """Test registry is persisted and loaded."""
        # Create pipeline and register model
        pipeline1 = AutoTournamentPipeline(
            models_dir=models_dir,
            results_dir=results_dir,
        )

        model = SimpleTestModel()
        path = os.path.join(models_dir, "model.pth")
        save_model_checkpoint(model, path)

        model_id = pipeline1.register_model(path)

        # Create new pipeline instance - should load registry
        pipeline2 = AutoTournamentPipeline(
            models_dir=models_dir,
            results_dir=results_dir,
        )

        assert len(pipeline2._models) == 1
        assert model_id in pipeline2._models


# =============================================================================
# Report Generation Tests
# =============================================================================


class TestReportGeneration:
    """Tests for report generation."""

    def test_generate_report_empty(self, pipeline):
        """Test report generation with no models."""
        report = pipeline.generate_report()

        assert "# AI Tournament Performance Report" in report
        assert "No champion registered" in report

    def test_generate_report_with_models(
        self, pipeline, sample_model_checkpoint
    ):
        """Test report generation with registered models."""
        pipeline.register_model(sample_model_checkpoint)
        report = pipeline.generate_report()

        assert "# AI Tournament Performance Report" in report
        assert "## Current Champion" in report
        assert "## Elo Rankings" in report
        assert "## Model Details" in report
        assert "1500.0" in report  # Default Elo

    def test_save_report(self, pipeline, sample_model_checkpoint):
        """Test saving report to file."""
        pipeline.register_model(sample_model_checkpoint)
        report_path = pipeline.save_report()

        assert os.path.exists(report_path)

        with open(report_path) as f:
            content = f.read()
        assert "# AI Tournament Performance Report" in content


# =============================================================================
# Tournament Execution Tests (with mocking)
# =============================================================================


class TestTournamentExecution:
    """Tests for tournament execution with mocked AI."""

    @patch('app.training.auto_tournament.Tournament')
    def test_run_tournament_mocked(
        self, mock_tournament_cls, pipeline, models_dir
    ):
        """Test tournament execution with mocked Tournament class."""
        # Create two model checkpoints
        for i in range(2):
            model = SimpleTestModel()
            path = os.path.join(models_dir, f"model_{i}.pth")
            save_model_checkpoint(model, path)

        pipeline.register_model(
            os.path.join(models_dir, "model_0.pth")
        )
        pipeline.register_model(
            os.path.join(models_dir, "model_1.pth")
        )

        # Mock Tournament instance
        mock_tournament = MagicMock()
        mock_tournament.run.return_value = {"A": 6, "B": 3, "Draw": 1}
        mock_tournament.ratings = {"A": 1520, "B": 1480}
        mock_tournament.victory_reasons = {"elimination": 9, "unknown": 1}
        mock_tournament_cls.return_value = mock_tournament

        # Run tournament
        result = pipeline.run_tournament(games_per_match=10)

        assert result is not None
        assert len(result.participants) == 2
        assert len(result.matches) == 10
        assert mock_tournament.run.called

    @patch('app.training.auto_tournament.Tournament')
    def test_evaluate_challenger_mocked(
        self, mock_tournament_cls, pipeline, models_dir
    ):
        """Test challenger evaluation with mocked Tournament class."""
        # Create champion model
        champ_model = SimpleTestModel()
        champ_path = os.path.join(models_dir, "champion.pth")
        save_model_checkpoint(champ_model, champ_path)
        pipeline.register_model(champ_path)

        # Create challenger model (different weights for different checksum)
        chall_model = SimpleTestModel()
        chall_model.fc1.weight.data.fill_(0.5)  # Different weights
        chall_path = os.path.join(models_dir, "challenger.pth")
        save_model_checkpoint(chall_model, chall_path)

        # Mock Tournament - challenger wins convincingly
        # Create a real dict for ratings that gets modified
        ratings_dict = {"A": 1500.0, "B": 1500.0}

        def run_side_effect():
            # Simulate Tournament.run() updating ratings
            ratings_dict["A"] = 1580.0
            ratings_dict["B"] = 1420.0
            return {"A": 35, "B": 15, "Draw": 0}

        mock_tournament = MagicMock()
        mock_tournament.run.side_effect = run_side_effect
        mock_tournament.ratings = ratings_dict
        mock_tournament.victory_reasons = {"elimination": 50}
        mock_tournament_cls.return_value = mock_tournament

        result = pipeline.evaluate_challenger(chall_path, games=50)

        assert result is not None
        assert result.challenger_wins == 35
        assert result.champion_wins == 15
        assert result.challenger_win_rate == 0.7
        assert result.challenger_final_elo == 1580.0
        assert result.champion_final_elo == 1420.0
        assert result.should_promote is True

    @patch('app.training.auto_tournament.Tournament')
    def test_evaluate_challenger_no_promotion(
        self, mock_tournament_cls, pipeline, models_dir
    ):
        """Test challenger not promoted if criteria not met."""
        # Create champion model
        champ_model = SimpleTestModel()
        champ_path = os.path.join(models_dir, "champion.pth")
        save_model_checkpoint(champ_model, champ_path)
        pipeline.register_model(champ_path)

        # Create challenger model (different weights for different checksum)
        chall_model = SimpleTestModel()
        chall_model.fc1.weight.data.fill_(0.3)  # Different weights
        chall_path = os.path.join(models_dir, "challenger.pth")
        save_model_checkpoint(chall_model, chall_path)

        # Mock Tournament - challenger wins slightly
        mock_tournament = MagicMock()
        mock_tournament.run.return_value = {"A": 27, "B": 23, "Draw": 0}
        mock_tournament.ratings = {"A": 1510, "B": 1490}
        mock_tournament.victory_reasons = {"elimination": 50}
        mock_tournament_cls.return_value = mock_tournament

        result = pipeline.evaluate_challenger(chall_path, games=50)

        assert result.challenger_win_rate < 0.55
        assert result.should_promote is False


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestErrorHandling:
    """Tests for error handling."""

    def test_register_nonexistent_model(self, pipeline):
        """Should raise FileNotFoundError for missing model."""
        with pytest.raises(FileNotFoundError):
            pipeline.register_model("/nonexistent/model.pth")

    def test_tournament_insufficient_participants(self, pipeline):
        """Should raise ValueError with fewer than 2 participants."""
        with pytest.raises(ValueError) as exc_info:
            pipeline.run_tournament()

        assert "at least 2 participants" in str(exc_info.value)

    def test_evaluate_no_champion(self, pipeline, models_dir):
        """Should raise ValueError if no champion registered."""
        model = SimpleTestModel()
        path = os.path.join(models_dir, "model.pth")
        save_model_checkpoint(model, path)

        with pytest.raises(ValueError) as exc_info:
            pipeline.evaluate_challenger(path)

        assert "No champion registered" in str(exc_info.value)

    def test_promote_unknown_model(self, pipeline):
        """Should raise ValueError for unknown model ID."""
        with pytest.raises(ValueError) as exc_info:
            pipeline.promote_champion("nonexistent_id")

        assert "Model not found" in str(exc_info.value)


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for the full pipeline."""

    def test_full_workflow(self, pipeline, models_dir):
        """Test complete workflow: register, tournament, report."""
        # Create two models
        for i in range(2):
            model = SimpleTestModel()
            path = os.path.join(models_dir, f"model_{i}.pth")
            save_model_checkpoint(
                model,
                path,
                training_info={"epochs": 100 * (i + 1)},
            )

        # Register models
        first_id = pipeline.register_model(
            os.path.join(models_dir, "model_0.pth")
        )
        pipeline.register_model(
            os.path.join(models_dir, "model_1.pth")
        )

        # Verify initial state
        assert pipeline.get_champion().model_id == first_id
        assert len(pipeline.get_elo_rankings()) == 2

        # Generate and save report
        report = pipeline.generate_report()
        assert "model_0" in report or first_id in report

        report_path = pipeline.save_report("test_report.md")
        assert os.path.exists(report_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])