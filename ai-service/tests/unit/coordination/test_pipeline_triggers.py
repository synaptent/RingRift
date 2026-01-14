import subprocess
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from app.coordination.pipeline_triggers import (
    PipelineTrigger,
    TriggerConfig,
    PrerequisiteResult,
    get_pipeline_trigger,
    validate_pipeline_prerequisites,
)


# =============================================================================
# PrerequisiteResult Tests
# =============================================================================


class TestPrerequisiteResult:
    """Tests for PrerequisiteResult dataclass."""

    def test_passed_result_is_truthy(self):
        result = PrerequisiteResult(passed=True, message="OK")
        assert result
        assert bool(result) is True

    def test_failed_result_is_falsy(self):
        result = PrerequisiteResult(passed=False, message="Failed")
        assert not result
        assert bool(result) is False

    def test_default_details_is_empty_dict(self):
        result = PrerequisiteResult(passed=True, message="OK")
        assert result.details == {}

    def test_details_preserved(self):
        details = {"key": "value", "count": 42}
        result = PrerequisiteResult(passed=True, message="OK", details=details)
        assert result.details == details

    def test_message_preserved(self):
        result = PrerequisiteResult(passed=False, message="Custom error message")
        assert result.message == "Custom error message"


# =============================================================================
# TriggerConfig Tests
# =============================================================================


class TestTriggerConfig:
    """Tests for TriggerConfig dataclass."""

    def test_default_values(self):
        config = TriggerConfig()
        assert config.min_games_for_sync == 100
        assert config.min_games_for_export == 50
        assert config.min_samples_for_training == 10000
        assert config.min_npz_size_bytes == 100_000
        assert config.min_win_rate_vs_random == 0.85
        assert config.validate_npz_integrity is True
        assert config.max_nan_ratio == 0.001
        assert config.validate_policy_bounds is True

    def test_custom_values(self):
        config = TriggerConfig(
            min_games_for_sync=500,
            min_samples_for_training=5000,
            min_win_rate_vs_random=0.90,
        )
        assert config.min_games_for_sync == 500
        assert config.min_samples_for_training == 5000
        assert config.min_win_rate_vs_random == 0.90

    def test_ai_service_root_is_path(self):
        config = TriggerConfig()
        assert isinstance(config.ai_service_root, Path)


# =============================================================================
# PipelineTrigger Iteration Counter Tests
# =============================================================================


class TestIterationCounter:
    """Tests for PipelineTrigger iteration counter."""

    def test_first_iteration_is_one(self):
        trigger = PipelineTrigger()
        iteration = trigger._get_iteration("hex8_2p")
        assert iteration == 1

    def test_iterations_increment(self):
        trigger = PipelineTrigger()
        assert trigger._get_iteration("hex8_2p") == 1
        assert trigger._get_iteration("hex8_2p") == 2
        assert trigger._get_iteration("hex8_2p") == 3

    def test_different_configs_have_separate_counters(self):
        trigger = PipelineTrigger()
        assert trigger._get_iteration("hex8_2p") == 1
        assert trigger._get_iteration("square8_4p") == 1
        assert trigger._get_iteration("hex8_2p") == 2
        assert trigger._get_iteration("square8_4p") == 2


# =============================================================================
# Board Size Tests
# =============================================================================


class TestGetMaxPolicyIndex:
    """Tests for _get_max_policy_index method."""

    def test_known_board_types(self):
        trigger = PipelineTrigger()
        assert trigger._get_max_policy_index("hex8") == 61
        assert trigger._get_max_policy_index("square8") == 64
        assert trigger._get_max_policy_index("square19") == 361
        assert trigger._get_max_policy_index("hexagonal") == 469

    def test_unknown_board_type_returns_none(self):
        trigger = PipelineTrigger()
        assert trigger._get_max_policy_index("unknown_board") is None
        assert trigger._get_max_policy_index("") is None


# =============================================================================
# Evaluation Check Tests
# =============================================================================


class TestCheckEvaluationPassed:
    """Tests for check_evaluation_passed method."""

    @pytest.mark.asyncio
    async def test_passes_when_both_thresholds_met(self):
        config = TriggerConfig(
            min_win_rate_vs_random=0.80,
            min_win_rate_vs_heuristic=0.60,
        )
        trigger = PipelineTrigger(config)
        result = await trigger.check_evaluation_passed(0.90, 0.70)
        assert result.passed
        assert result.details["win_rate_vs_random"] == 0.90
        assert result.details["win_rate_vs_heuristic"] == 0.70

    @pytest.mark.asyncio
    async def test_fails_when_random_threshold_not_met(self):
        config = TriggerConfig(
            min_win_rate_vs_random=0.85,
            min_win_rate_vs_heuristic=0.60,
        )
        trigger = PipelineTrigger(config)
        result = await trigger.check_evaluation_passed(0.80, 0.70)
        assert not result.passed
        assert "threshold" in result.message.lower()
        assert len(result.details["issues"]) >= 1
        assert "random" in result.details["issues"][0].lower()

    @pytest.mark.asyncio
    async def test_fails_when_heuristic_threshold_not_met(self):
        config = TriggerConfig(
            min_win_rate_vs_random=0.80,
            min_win_rate_vs_heuristic=0.65,
        )
        trigger = PipelineTrigger(config)
        result = await trigger.check_evaluation_passed(0.90, 0.50)
        assert not result.passed
        assert "heuristic" in result.message.lower() or len(result.details["issues"]) >= 1

    @pytest.mark.asyncio
    async def test_fails_when_both_thresholds_not_met(self):
        config = TriggerConfig(
            min_win_rate_vs_random=0.85,
            min_win_rate_vs_heuristic=0.65,
        )
        trigger = PipelineTrigger(config)
        result = await trigger.check_evaluation_passed(0.50, 0.40)
        assert not result.passed
        assert len(result.details["issues"]) == 2


# =============================================================================
# NPZ Validation Tests
# =============================================================================


class TestNpzValidation:
    """Tests for NPZ validation in check_npz_exists."""

    @pytest.mark.asyncio
    async def test_detects_nan_values(self, tmp_path):
        training_dir = tmp_path / "data" / "training"
        training_dir.mkdir(parents=True)

        # Create NPZ with NaN values
        features = np.array([[1.0, np.nan, 3.0], [4.0, 5.0, np.nan]])
        np.savez(training_dir / "hex8_2p.npz", features=features)

        config = TriggerConfig(
            ai_service_root=tmp_path,
            min_samples_for_training=1,
            min_npz_size_bytes=0,
            validate_npz_integrity=True,
            max_nan_ratio=0.0,  # No NaN allowed
        )
        trigger = PipelineTrigger(config)
        result = await trigger.check_npz_exists("hex8", 2)

        assert not result.passed
        assert "nan" in result.message.lower() or "integrity" in result.message.lower()

    @pytest.mark.asyncio
    async def test_allows_nan_below_threshold(self, tmp_path):
        training_dir = tmp_path / "data" / "training"
        training_dir.mkdir(parents=True)

        # Create NPZ with small percentage of NaN
        features = np.ones((100, 10))
        features[0, 0] = np.nan  # 0.1% NaN

        np.savez(training_dir / "hex8_2p.npz", features=features)

        config = TriggerConfig(
            ai_service_root=tmp_path,
            min_samples_for_training=1,
            min_npz_size_bytes=0,
            validate_npz_integrity=True,
            max_nan_ratio=0.01,  # 1% NaN allowed
        )
        trigger = PipelineTrigger(config)
        result = await trigger.check_npz_exists("hex8", 2)

        assert result.passed

    @pytest.mark.asyncio
    async def test_detects_inf_values(self, tmp_path):
        training_dir = tmp_path / "data" / "training"
        training_dir.mkdir(parents=True)

        features = np.array([[1.0, np.inf, 3.0]])
        np.savez(training_dir / "hex8_2p.npz", features=features)

        config = TriggerConfig(
            ai_service_root=tmp_path,
            min_samples_for_training=1,
            min_npz_size_bytes=0,
            validate_npz_integrity=True,
        )
        trigger = PipelineTrigger(config)
        result = await trigger.check_npz_exists("hex8", 2)

        assert not result.passed
        assert "inf" in result.message.lower() or "integrity" in result.message.lower()

    @pytest.mark.asyncio
    async def test_skips_validation_when_disabled(self, tmp_path):
        training_dir = tmp_path / "data" / "training"
        training_dir.mkdir(parents=True)

        features = np.array([[np.nan, np.inf]])
        np.savez(training_dir / "hex8_2p.npz", features=features)

        config = TriggerConfig(
            ai_service_root=tmp_path,
            min_samples_for_training=1,
            min_npz_size_bytes=0,
            validate_npz_integrity=False,
            validate_policy_bounds=False,
        )
        trigger = PipelineTrigger(config)
        result = await trigger.check_npz_exists("hex8", 2)

        assert result.passed  # Validation disabled, should pass


# =============================================================================
# Database Check Tests (Additional)
# =============================================================================


class TestDatabaseChecks:
    """Additional tests for database checking."""

    @pytest.mark.asyncio
    async def test_check_databases_insufficient_games(self, monkeypatch, tmp_path):
        class FakeDiscovery:
            def find_databases_for_config(self, _board_type, _num_players):
                return [SimpleNamespace(game_count=10, path=Path("/fake/db.db"))]

        monkeypatch.setattr("app.utils.game_discovery.GameDiscovery", FakeDiscovery)

        # Jan 2026: Mock get_min_games_for_export to return 100 (test threshold)
        # NOTE: Must patch at source location since import happens inside method
        monkeypatch.setattr(
            "app.config.thresholds.get_min_games_for_export",
            lambda num_players, bootstrap_mode=False: 100,
        )

        # Create a fake model so bootstrap_mode=False
        model_dir = tmp_path / "models"
        model_dir.mkdir(parents=True)
        (model_dir / "canonical_square8_2p.pth").touch()

        config = TriggerConfig(ai_service_root=tmp_path)
        trigger = PipelineTrigger(config)
        result = await trigger.check_databases_exist("square8", 2)

        assert not result.passed
        assert "insufficient" in result.message.lower()
        assert result.details["total_games"] == 10
        assert result.details["min_required"] == 100

    @pytest.mark.asyncio
    async def test_check_databases_handles_exception(self, monkeypatch):
        def raise_error():
            raise RuntimeError("Discovery failed")

        monkeypatch.setattr(
            "app.utils.game_discovery.GameDiscovery",
            lambda: type("Fake", (), {"find_databases_for_config": lambda *_: raise_error()})(),
        )

        trigger = PipelineTrigger()
        result = await trigger.check_databases_exist("square8", 2)

        assert not result.passed
        assert "error" in result.message.lower()


# =============================================================================
# Model Check Tests (Additional)
# =============================================================================


class TestModelChecks:
    """Additional tests for model checking."""

    @pytest.mark.asyncio
    async def test_check_model_not_found(self, tmp_path):
        trigger = PipelineTrigger(TriggerConfig(ai_service_root=tmp_path))
        result = await trigger.check_model_exists("square8", 2, model_path="/nonexistent/path.pth")

        assert not result.passed
        assert "not found" in result.message.lower()

    @pytest.mark.asyncio
    async def test_check_model_default_path(self, tmp_path):
        # Create default model path
        model_dir = tmp_path / "models" / "square8_2p"
        model_dir.mkdir(parents=True)
        model_path = model_dir / "latest.pth"
        model_path.write_bytes(b"x" * 1000)

        trigger = PipelineTrigger(TriggerConfig(ai_service_root=tmp_path))
        result = await trigger.check_model_exists("square8", 2)

        assert result.passed
        assert "latest.pth" in result.details["model_path"]


# =============================================================================
# Singleton Tests
# =============================================================================


class TestSingleton:
    """Tests for singleton pattern."""

    def test_get_pipeline_trigger_returns_instance(self):
        # Reset the global singleton for testing
        import app.coordination.pipeline_triggers as module

        module._pipeline_trigger = None

        trigger = get_pipeline_trigger()
        assert isinstance(trigger, PipelineTrigger)

    def test_get_pipeline_trigger_returns_same_instance(self):
        import app.coordination.pipeline_triggers as module

        module._pipeline_trigger = None

        trigger1 = get_pipeline_trigger()
        trigger2 = get_pipeline_trigger()
        assert trigger1 is trigger2


# =============================================================================
# Validate Pipeline Prerequisites Tests
# =============================================================================


class TestValidatePipelinePrerequisites:
    """Tests for validate_pipeline_prerequisites function."""

    @pytest.mark.asyncio
    async def test_sync_has_no_prerequisites(self):
        result = await validate_pipeline_prerequisites("hex8", 2, "sync")
        assert result.passed
        assert "no prerequisites" in result.message.lower()

    @pytest.mark.asyncio
    async def test_unknown_stage_fails(self):
        result = await validate_pipeline_prerequisites("hex8", 2, "unknown_stage")
        assert not result.passed
        assert "unknown stage" in result.message.lower()

    @pytest.mark.asyncio
    async def test_export_stage_checks_databases(self, monkeypatch):
        class FakeDiscovery:
            def find_databases_for_config(self, _board_type, _num_players):
                return []

        monkeypatch.setattr("app.utils.game_discovery.GameDiscovery", FakeDiscovery)

        # Reset singleton
        import app.coordination.pipeline_triggers as module

        module._pipeline_trigger = None

        result = await validate_pipeline_prerequisites("hex8", 2, "export")
        assert not result.passed


# =============================================================================
# Trigger Function Tests
# =============================================================================


class TestTriggerFunctions:
    """Tests for trigger functions."""

    @pytest.mark.asyncio
    async def test_trigger_sync_returns_result(self, monkeypatch):
        mock_result = MagicMock()
        mock_result.success = True
        mock_trigger = AsyncMock(return_value=mock_result)
        monkeypatch.setattr(
            "app.coordination.pipeline_triggers.trigger_data_sync",
            mock_trigger,
        )

        trigger = PipelineTrigger()
        result = await trigger.trigger_sync_after_selfplay("hex8", 2)

        assert result.success
        mock_trigger.assert_called_once()

    @pytest.mark.asyncio
    async def test_trigger_export_fails_on_missing_databases(self, monkeypatch):
        class FakeDiscovery:
            def find_databases_for_config(self, _board_type, _num_players):
                return []

        monkeypatch.setattr("app.utils.game_discovery.GameDiscovery", FakeDiscovery)

        trigger = PipelineTrigger()
        result = await trigger.trigger_export_after_sync("hex8", 2)

        assert not result.success
        assert "prerequisite" in result.error.lower()

    @pytest.mark.asyncio
    async def test_trigger_export_skips_validation_when_requested(self, monkeypatch):
        mock_result = MagicMock()
        mock_result.success = True
        mock_trigger = AsyncMock(return_value=mock_result)
        monkeypatch.setattr(
            "app.coordination.pipeline_triggers.trigger_npz_export",
            mock_trigger,
        )

        trigger = PipelineTrigger()
        result = await trigger.trigger_export_after_sync("hex8", 2, skip_validation=True)

        assert result.success
        mock_trigger.assert_called_once()

    @pytest.mark.asyncio
    async def test_trigger_promotion_fails_on_low_win_rate(self):
        config = TriggerConfig(
            min_win_rate_vs_random=0.85,
            min_win_rate_vs_heuristic=0.60,
        )
        trigger = PipelineTrigger(config)
        result = await trigger.trigger_promotion_after_evaluation(
            board_type="hex8",
            num_players=2,
            model_path="/fake/model.pth",
            win_rate_vs_random=0.50,
            win_rate_vs_heuristic=0.30,
        )

        assert not result.success
        assert "prerequisite" in result.error.lower()


# =============================================================================
# Original Tests (Preserved)
# =============================================================================


@pytest.mark.asyncio
async def test_check_databases_exist_empty(monkeypatch) -> None:
    class FakeDiscovery:
        def find_databases_for_config(self, _board_type: str, _num_players: int):
            return []

    monkeypatch.setattr("app.utils.game_discovery.GameDiscovery", FakeDiscovery)

    trigger = PipelineTrigger()
    result = await trigger.check_databases_exist("square8", 2)

    assert not result.passed
    assert result.details["databases_found"] == 0


@pytest.mark.asyncio
async def test_check_databases_exist_counts_games(monkeypatch, tmp_path) -> None:
    db_path = tmp_path / "games.db"
    db_path.write_text("placeholder")

    class FakeDiscovery:
        def find_databases_for_config(self, _board_type: str, _num_players: int):
            return [SimpleNamespace(game_count=7, path=db_path)]

    monkeypatch.setattr("app.utils.game_discovery.GameDiscovery", FakeDiscovery)

    # Jan 2026: Mock get_min_games_for_export to return 1 (low threshold for test)
    # NOTE: Must patch at source location since import happens inside method
    monkeypatch.setattr(
        "app.config.thresholds.get_min_games_for_export",
        lambda num_players, bootstrap_mode=False: 1,
    )

    config = TriggerConfig(ai_service_root=tmp_path)
    trigger = PipelineTrigger(config)
    result = await trigger.check_databases_exist("square8", 2)

    assert result.passed
    assert result.details["total_games"] == 7


@pytest.mark.asyncio
async def test_check_npz_exists_prefers_largest(monkeypatch, tmp_path) -> None:
    training_dir = tmp_path / "data" / "training"
    training_dir.mkdir(parents=True)

    np.savez(training_dir / "square8_2p_v1.npz", features=np.zeros((2, 3)))
    np.savez(training_dir / "square8_2p_iter2.npz", features=np.zeros((5, 3)))

    config = TriggerConfig(
        ai_service_root=tmp_path,
        min_samples_for_training=1,
        min_npz_size_bytes=0,
    )
    trigger = PipelineTrigger(config)
    result = await trigger.check_npz_exists("square8", 2)

    assert result.passed
    assert result.details["samples"] == 5


@pytest.mark.asyncio
async def test_check_model_exists_with_explicit_path(tmp_path) -> None:
    model_path = tmp_path / "model.pth"
    model_path.write_bytes(b"model")

    trigger = PipelineTrigger(TriggerConfig(ai_service_root=tmp_path))
    result = await trigger.check_model_exists("square8", 2, model_path=str(model_path))

    assert result.passed
    assert result.details["model_path"] == str(model_path)


@pytest.mark.asyncio
async def test_check_no_training_running(monkeypatch) -> None:
    def fake_run_running(*_args, **_kwargs):
        return subprocess.CompletedProcess(args=["pgrep"], returncode=0, stdout="123\n", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run_running)

    trigger = PipelineTrigger()
    result = await trigger.check_no_training_running("square8", 2)

    assert not result.passed
    assert "123" in result.message


@pytest.mark.asyncio
async def test_check_no_training_running_when_clear(monkeypatch) -> None:
    def fake_run_clear(*_args, **_kwargs):
        return subprocess.CompletedProcess(args=["pgrep"], returncode=1, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run_clear)

    trigger = PipelineTrigger()
    result = await trigger.check_no_training_running("square8", 2)

    assert result.passed
