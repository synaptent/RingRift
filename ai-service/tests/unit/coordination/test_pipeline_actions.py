"""Tests for app.coordination.pipeline_actions - Pipeline Action Triggers.

This module tests the pipeline action triggers that invoke actual work
for each pipeline stage (sync, export, training, evaluation, promotion).
"""

from __future__ import annotations

import asyncio
import sys
import types
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.coordination.pipeline_actions import (
    ActionConfig,
    ActionPriority,
    StageCompletionResult,
    QUALITY_GATE_THRESHOLD,
    _check_training_data_quality,
    _get_ai_service_root,
    trigger_data_sync,
    trigger_evaluation,
    trigger_npz_export,
    trigger_promotion,
    trigger_training,
)


@pytest.fixture
def allow_training_quality_gate():
    """Let training-action tests focus on subprocess behavior, not gate sourcing."""
    with patch(
        "app.coordination.pipeline_actions._check_training_data_quality",
        new=AsyncMock(return_value=(True, QUALITY_GATE_THRESHOLD)),
    ):
        yield


# =============================================================================
# ActionPriority Tests
# =============================================================================


class TestActionPriority:
    """Tests for ActionPriority enum."""

    def test_all_priorities_defined(self):
        """All expected priority levels should exist."""
        assert ActionPriority.LOW.value == "low"
        assert ActionPriority.NORMAL.value == "normal"
        assert ActionPriority.HIGH.value == "high"
        assert ActionPriority.CRITICAL.value == "critical"

    def test_priority_count(self):
        """Should have exactly 4 priority levels."""
        assert len(ActionPriority) == 4


# =============================================================================
# StageCompletionResult Tests
# =============================================================================


class TestStageCompletionResult:
    """Tests for StageCompletionResult dataclass."""

    def test_minimal_result(self):
        """Should create result with minimal required fields."""
        result = StageCompletionResult(
            success=True,
            stage="test_stage",
            iteration=1,
        )

        assert result.success is True
        assert result.stage == "test_stage"
        assert result.iteration == 1
        assert result.duration_seconds == 0.0
        assert result.output_path is None
        assert result.error is None
        assert result.metadata == {}
        assert result.exit_code == 0
        assert result.stdout == ""
        assert result.stderr == ""

    def test_full_result(self):
        """Should create result with all fields."""
        result = StageCompletionResult(
            success=True,
            stage="training",
            iteration=5,
            duration_seconds=123.45,
            output_path="/path/to/model.pth",
            error=None,
            metadata={"batch_size": 512},
            exit_code=0,
            stdout="Training complete",
            stderr="",
        )

        assert result.success is True
        assert result.stage == "training"
        assert result.iteration == 5
        assert result.duration_seconds == 123.45
        assert result.output_path == "/path/to/model.pth"
        assert result.metadata["batch_size"] == 512
        assert result.stdout == "Training complete"

    def test_failure_result(self):
        """Should capture failure details."""
        result = StageCompletionResult(
            success=False,
            stage="export",
            iteration=2,
            error="Database not found",
            exit_code=1,
            stderr="Error: No database",
        )

        assert result.success is False
        assert result.error == "Database not found"
        assert result.exit_code == 1

    def test_to_dict(self):
        """to_dict should return serializable dictionary."""
        result = StageCompletionResult(
            success=True,
            stage="sync",
            iteration=3,
            duration_seconds=10.5,
            output_path="/path",
            error=None,
            metadata={"hosts": ["a", "b"]},
        )

        d = result.to_dict()

        assert d["success"] is True
        assert d["stage"] == "sync"
        assert d["iteration"] == 3
        assert d["duration_seconds"] == 10.5
        assert d["output_path"] == "/path"
        assert d["error"] is None
        assert d["metadata"]["hosts"] == ["a", "b"]

    def test_to_dict_excludes_internal_fields(self):
        """to_dict should not include stdout/stderr/exit_code."""
        result = StageCompletionResult(
            success=True,
            stage="test",
            iteration=1,
            stdout="output",
            stderr="errors",
            exit_code=0,
        )

        d = result.to_dict()

        assert "stdout" not in d
        assert "stderr" not in d
        assert "exit_code" not in d


# =============================================================================
# ActionConfig Tests
# =============================================================================


class TestActionConfig:
    """Tests for ActionConfig dataclass."""

    def test_default_config(self):
        """Should have sensible defaults."""
        config = ActionConfig()

        assert config.sync_script == "scripts/unified_data_sync.py"
        assert config.export_script == "scripts/export_replay_dataset.py"
        assert config.train_module == "app.training.train"
        assert config.evaluate_script == "scripts/quick_gauntlet.py"
        assert config.promote_script == "scripts/auto_promote.py"

    def test_timeout_defaults(self):
        """Timeouts should have reasonable defaults."""
        config = ActionConfig()

        assert config.sync_timeout == 1800.0  # 30 min
        assert config.export_timeout == 3600.0  # 1 hour
        assert config.training_timeout == 86400.0  # 24 hours
        assert config.evaluation_timeout == 7200.0  # 2 hours
        assert config.promotion_timeout == 600.0  # 10 min

    def test_path_defaults(self):
        """Path defaults should be set."""
        config = ActionConfig()

        assert config.data_dir == "data"
        assert config.models_dir == "models"
        assert config.training_data_dir == "data/training"
        assert config.games_dir == "data/games"

    def test_custom_config(self):
        """Should accept custom values."""
        config = ActionConfig(
            python_executable="/usr/bin/python3.11",
            sync_timeout=3600.0,
            models_dir="custom/models",
        )

        assert config.python_executable == "/usr/bin/python3.11"
        assert config.sync_timeout == 3600.0
        assert config.models_dir == "custom/models"


class TestCheckTrainingDataQuality:
    """Tests for the training data quality gate helper."""

    @pytest.mark.asyncio
    async def test_blocks_training_when_no_quality_source_returns_signal(self):
        quality_module = types.SimpleNamespace(get_quality_daemon=lambda: None)
        orchestrator_module = types.SimpleNamespace(
            get_data_pipeline_orchestrator=lambda: None
        )

        with patch.dict(
            sys.modules,
            {
                "app.coordination.quality_monitor_daemon": quality_module,
                "app.coordination.data_pipeline_orchestrator": orchestrator_module,
            },
        ):
            quality_ok, quality_score = await _check_training_data_quality(
                "hex8_2p",
                "/tmp/hex8_2p.npz",
            )

        assert quality_ok is False
        assert quality_score == 0.0

    @pytest.mark.asyncio
    async def test_uses_orchestrator_quality_check_with_current_signature(self):
        orchestrator = AsyncMock()
        orchestrator._check_training_data_quality = AsyncMock(return_value=True)
        quality_module = types.SimpleNamespace(get_quality_daemon=lambda: None)
        orchestrator_module = types.SimpleNamespace(
            get_data_pipeline_orchestrator=lambda: orchestrator
        )

        with patch.dict(
            sys.modules,
            {
                "app.coordination.quality_monitor_daemon": quality_module,
                "app.coordination.data_pipeline_orchestrator": orchestrator_module,
            },
        ):
            quality_ok, quality_score = await _check_training_data_quality(
                "hex8_2p",
                "/tmp/hex8_2p.npz",
            )

        assert quality_ok is True
        assert quality_score == QUALITY_GATE_THRESHOLD
        orchestrator._check_training_data_quality.assert_awaited_once_with(
            "/tmp/hex8_2p.npz",
            0,
        )


# =============================================================================
# Helper Function Tests
# =============================================================================


class TestGetAiServiceRoot:
    """Tests for _get_ai_service_root helper."""

    def test_finds_root_from_file_location(self):
        """Should find ai-service root relative to module."""
        root = _get_ai_service_root()

        # Should be a Path object
        assert isinstance(root, Path)
        # Should exist or be the fallback path
        # The function uses heuristics so we just verify it returns a path

    def test_env_var_override(self):
        """Environment variable should override detection."""
        with patch.dict("os.environ", {"RINGRIFT_AI_SERVICE_PATH": "/custom/path"}):
            root = _get_ai_service_root()
            assert root == Path("/custom/path")


# =============================================================================
# Subprocess Helper Tests
# =============================================================================


class TestRunSubprocess:
    """Tests for _run_subprocess helper via action triggers."""

    @pytest.mark.asyncio
    async def test_subprocess_success(self):
        """Should capture successful subprocess output."""
        mock_process = MagicMock()
        mock_process.pid = 1234
        mock_process.returncode = 0
        mock_process.communicate = AsyncMock(return_value=(b"success output", b""))
        mock_process.wait = AsyncMock()
        mock_process.terminate = MagicMock()
        mock_process.kill = MagicMock()

        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            with patch("app.coordination.pipeline_actions._emit_sync_complete"):
                result = await trigger_data_sync(
                    board_type="hex8",
                    num_players=2,
                    iteration=1,
                )

        assert result.exit_code == 0
        assert "success output" in result.stdout

    @pytest.mark.asyncio
    async def test_subprocess_timeout(self):
        """Should handle subprocess timeout."""
        mock_process = MagicMock()
        mock_process.pid = 1234
        mock_process.returncode = None
        mock_process.terminate = MagicMock()
        mock_process.kill = MagicMock()
        mock_process.wait = AsyncMock()
        mock_process.communicate = AsyncMock(side_effect=asyncio.TimeoutError())

        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            result = await trigger_data_sync(
                board_type="hex8",
                num_players=2,
                iteration=1,
                config=ActionConfig(sync_timeout=0.001),
            )

        assert result.success is False
        assert "timed out" in result.stderr.lower()
        mock_process.terminate.assert_called_once()


# =============================================================================
# trigger_data_sync Tests
# =============================================================================


class TestTriggerDataSync:
    """Tests for trigger_data_sync action."""

    @pytest.fixture
    def mock_subprocess(self):
        """Create mock for subprocess execution."""
        mock = AsyncMock()
        mock.returncode = 0
        mock.communicate = AsyncMock(return_value=(b"Synced 10 files", b""))
        return mock

    @pytest.mark.asyncio
    async def test_basic_sync(self, mock_subprocess):
        """Should run one-shot unified sync with the expected command."""
        with patch("asyncio.create_subprocess_exec", return_value=mock_subprocess) as exec_mock:
            with patch("app.coordination.pipeline_actions._emit_sync_complete"):
                result = await trigger_data_sync(
                    board_type="hex8",
                    num_players=2,
                    iteration=1,
                )

        assert result.success is True
        assert result.stage == "data_sync"
        assert result.iteration == 1
        assert result.metadata["board_type"] == "hex8"
        assert result.metadata["num_players"] == 2

        # Verify command was constructed correctly
        call_args = exec_mock.call_args
        cmd_str = " ".join(call_args[0])
        assert "unified_data_sync.py" in cmd_str
        assert "--once" in cmd_str

    @pytest.mark.asyncio
    async def test_sync_with_hosts(self, mock_subprocess):
        """Should pass hosts filter to sync script."""
        with patch("asyncio.create_subprocess_exec", return_value=mock_subprocess) as exec_mock:
            with patch("app.coordination.pipeline_actions._emit_sync_complete"):
                result = await trigger_data_sync(
                    board_type="hex8",
                    num_players=2,
                    iteration=1,
                    hosts=["node1", "node2"],
                )

        assert result.success is True
        assert result.metadata["hosts"] == ["node1", "node2"]

        call_args = exec_mock.call_args
        cmd_str = " ".join(call_args[0])
        assert "--hosts" in cmd_str
        assert "node1,node2" in cmd_str

    @pytest.mark.asyncio
    async def test_sync_failure(self):
        """Should handle sync failure gracefully."""
        mock_process = AsyncMock()
        mock_process.returncode = 1
        mock_process.communicate = AsyncMock(return_value=(b"", b"Connection refused"))

        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            result = await trigger_data_sync(
                board_type="hex8",
                num_players=2,
                iteration=1,
            )

        assert result.success is False
        assert result.exit_code == 1
        assert "Connection refused" in result.stderr

    @pytest.mark.asyncio
    async def test_sync_exception(self):
        """Should handle exceptions during sync."""
        with patch("asyncio.create_subprocess_exec", side_effect=OSError("Command not found")):
            result = await trigger_data_sync(
                board_type="hex8",
                num_players=2,
                iteration=1,
            )

        assert result.success is False
        assert "Command not found" in result.error


# =============================================================================
# trigger_npz_export Tests
# =============================================================================


class TestTriggerNpzExport:
    """Tests for trigger_npz_export action."""

    @pytest.fixture
    def mock_export_subprocess(self):
        """Create mock for export subprocess."""
        mock = AsyncMock()
        mock.returncode = 0
        mock.communicate = AsyncMock(
            return_value=(b"Exported 12345 samples to output.npz", b"")
        )
        return mock

    @pytest.mark.asyncio
    async def test_basic_export(self, mock_export_subprocess):
        """Should run export with correct arguments."""
        with patch("asyncio.create_subprocess_exec", return_value=mock_export_subprocess):
            with patch("pathlib.Path.exists", return_value=True):
                with patch("app.coordination.pipeline_actions._emit_npz_export_complete"):
                    result = await trigger_npz_export(
                        board_type="square8",
                        num_players=4,
                        iteration=3,
                    )

        assert result.success is True
        assert result.stage == "npz_export"
        assert result.iteration == 3
        assert result.metadata["board_type"] == "square8"
        assert result.metadata["num_players"] == 4
        assert result.metadata["samples_exported"] == 12345

    @pytest.mark.asyncio
    async def test_export_parses_samples(self):
        """Should parse sample count from various output formats."""
        mock_process = AsyncMock()
        mock_process.returncode = 0
        mock_process.communicate = AsyncMock(
            return_value=(b"Processing...\nExported 5,432 samples\nDone", b"")
        )

        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            with patch("pathlib.Path.exists", return_value=True):
                with patch("app.coordination.pipeline_actions._emit_npz_export_complete"):
                    result = await trigger_npz_export(
                        board_type="hex8",
                        num_players=2,
                        iteration=1,
                    )

        assert result.metadata["samples_exported"] == 5432

    @pytest.mark.asyncio
    async def test_export_with_discovery(self, mock_export_subprocess):
        """Should pass use_discovery flag."""
        with patch("asyncio.create_subprocess_exec", return_value=mock_export_subprocess) as exec_mock:
            with patch("pathlib.Path.exists", return_value=True):
                with patch("app.coordination.pipeline_actions._emit_npz_export_complete"):
                    await trigger_npz_export(
                        board_type="hex8",
                        num_players=2,
                        iteration=1,
                        use_discovery=True,
                    )

        cmd_str = " ".join(exec_mock.call_args[0])
        assert "--use-discovery" in cmd_str

    @pytest.mark.asyncio
    async def test_export_output_path_missing(self):
        """Should fail if output file not created."""
        mock_process = AsyncMock()
        mock_process.returncode = 0
        mock_process.communicate = AsyncMock(return_value=(b"Done", b""))

        with patch("app.coordination.pipeline_actions._get_ai_service_root", return_value=Path("/tmp/mock_ai_service")):
            with patch("asyncio.create_subprocess_exec", return_value=mock_process):
                with patch("pathlib.Path.exists", return_value=False):
                    result = await trigger_npz_export(
                        board_type="hex8",
                        num_players=2,
                        iteration=1,
                    )

        assert result.success is False


# =============================================================================
# trigger_training Tests
# =============================================================================


class TestTriggerTraining:
    """Tests for trigger_training action."""

    @pytest.fixture
    def mock_training_subprocess(self):
        """Create mock for training subprocess."""
        mock = AsyncMock()
        mock.returncode = 0
        mock.communicate = AsyncMock(
            return_value=(
                b"Epoch 50/50\ntrain_loss: 0.234\nval_loss: 0.198\npolicy_accuracy: 76.5%",
                b"",
            )
        )
        return mock

    @pytest.mark.asyncio
    async def test_basic_training(self, mock_training_subprocess, allow_training_quality_gate):
        """Should run training with correct arguments."""
        with patch("asyncio.create_subprocess_exec", return_value=mock_training_subprocess) as exec_mock:
            with patch("pathlib.Path.exists", return_value=True):
                with patch("app.coordination.pipeline_actions._emit_training_complete"):
                    result = await trigger_training(
                        board_type="hex8",
                        num_players=2,
                        npz_path="/data/training.npz",
                        iteration=5,
                    )

        assert result.success is True
        assert result.stage == "training"
        assert result.metadata["board_type"] == "hex8"
        assert result.metadata["policy_accuracy"] == 76.5

        cmd_str = " ".join(exec_mock.call_args[0])
        assert "-m" in cmd_str
        assert "app.training.train" in cmd_str

    @pytest.mark.asyncio
    async def test_training_parses_metrics(self, allow_training_quality_gate):
        """Should parse training metrics from output."""
        mock_process = AsyncMock()
        mock_process.returncode = 0
        mock_process.communicate = AsyncMock(
            return_value=(
                b"train_loss: 0.123\nval_loss: 0.156\npolicy acc: 82.3%",
                b"",
            )
        )

        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            with patch("pathlib.Path.exists", return_value=True):
                with patch("app.coordination.pipeline_actions._emit_training_complete"):
                    result = await trigger_training(
                        board_type="hex8",
                        num_players=2,
                        npz_path="/data/training.npz",
                        iteration=1,
                    )

        assert result.metadata["train_loss"] == 0.123
        assert result.metadata["val_loss"] == 0.156
        assert result.metadata["policy_accuracy"] == 82.3

    @pytest.mark.asyncio
    async def test_training_with_transfer_weights(self, mock_training_subprocess, allow_training_quality_gate):
        """Should pass init_weights for transfer learning."""
        with patch("asyncio.create_subprocess_exec", return_value=mock_training_subprocess) as exec_mock:
            with patch("pathlib.Path.exists", return_value=True):
                with patch("app.coordination.pipeline_actions._emit_training_complete"):
                    await trigger_training(
                        board_type="hex8",
                        num_players=4,
                        npz_path="/data/training.npz",
                        iteration=1,
                        init_weights="/models/pretrained.pth",
                    )

        cmd_str = " ".join(exec_mock.call_args[0])
        assert "--init-weights" in cmd_str
        assert "/models/pretrained.pth" in cmd_str

    @pytest.mark.asyncio
    async def test_training_failure_emits_event(self, allow_training_quality_gate):
        """Should emit training_failed event on failure."""
        mock_process = AsyncMock()
        mock_process.returncode = 1
        mock_process.communicate = AsyncMock(return_value=(b"", b"CUDA out of memory"))

        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            with patch("app.coordination.pipeline_actions._emit_training_failed") as emit_mock:
                result = await trigger_training(
                    board_type="hex8",
                    num_players=2,
                    npz_path="/data/training.npz",
                    iteration=1,
                )

        assert result.success is False
        emit_mock.assert_called_once()


# =============================================================================
# trigger_evaluation Tests
# =============================================================================


class TestTriggerEvaluation:
    """Tests for trigger_evaluation action."""

    @pytest.fixture
    def mock_eval_subprocess(self):
        """Create mock for evaluation subprocess."""
        mock = AsyncMock()
        mock.returncode = 0
        mock.communicate = AsyncMock(
            return_value=(
                b"vs random: 92.0%\nvs heuristic: 68.5%\nelo delta: +45",
                b"",
            )
        )
        return mock

    @pytest.mark.asyncio
    async def test_basic_evaluation(self, mock_eval_subprocess):
        """Should run evaluation with correct arguments."""
        with patch("asyncio.create_subprocess_exec", return_value=mock_eval_subprocess) as exec_mock:
            with patch("app.coordination.pipeline_actions._emit_evaluation_complete"):
                result = await trigger_evaluation(
                    model_path="/models/model.pth",
                    board_type="hex8",
                    num_players=2,
                    iteration=3,
                )

        assert result.success is True
        assert result.stage == "evaluation"
        assert result.metadata["model_path"] == "/models/model.pth"
        assert result.metadata["win_rates"]["random"] == 92.0
        assert result.metadata["win_rates"]["heuristic"] == 68.5
        assert result.metadata["elo_delta"] == 45.0

        cmd_str = " ".join(exec_mock.call_args[0])
        assert "--model" in cmd_str
        assert "/models/model.pth" in cmd_str

    @pytest.mark.asyncio
    async def test_evaluation_promotion_eligibility(self):
        """Should calculate promotion eligibility correctly."""
        # Eligible case
        mock_process = AsyncMock()
        mock_process.returncode = 0
        mock_process.communicate = AsyncMock(
            return_value=(b"vs random: 90.0%\nvs heuristic: 65.0%", b"")
        )

        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            with patch("app.coordination.pipeline_actions._emit_evaluation_complete"):
                result = await trigger_evaluation(
                    model_path="/models/model.pth",
                    board_type="hex8",
                    num_players=2,
                )

        assert result.metadata["promotion_eligible"] is True

    @pytest.mark.asyncio
    async def test_evaluation_not_eligible(self):
        """Should detect non-eligible results."""
        mock_process = AsyncMock()
        mock_process.returncode = 0
        mock_process.communicate = AsyncMock(
            return_value=(b"vs random: 80.0%\nvs heuristic: 55.0%", b"")
        )

        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            with patch("app.coordination.pipeline_actions._emit_evaluation_complete"):
                result = await trigger_evaluation(
                    model_path="/models/model.pth",
                    board_type="hex8",
                    num_players=2,
                )

        # 80% vs random < 85% threshold
        assert result.metadata["promotion_eligible"] is False

    @pytest.mark.asyncio
    async def test_evaluation_with_custom_games(self, mock_eval_subprocess):
        """Should pass num_games parameter."""
        with patch("asyncio.create_subprocess_exec", return_value=mock_eval_subprocess) as exec_mock:
            with patch("app.coordination.pipeline_actions._emit_evaluation_complete"):
                await trigger_evaluation(
                    model_path="/models/model.pth",
                    board_type="hex8",
                    num_players=2,
                    num_games=200,
                )

        cmd_str = " ".join(exec_mock.call_args[0])
        assert "--games" in cmd_str
        assert "200" in cmd_str


# =============================================================================
# trigger_promotion Tests
# =============================================================================


class TestTriggerPromotion:
    """Tests for trigger_promotion action."""

    @pytest.fixture
    def eligible_results(self):
        """Gauntlet results that meet promotion thresholds."""
        return {
            "win_rates": {
                "random": 92.0,
                "heuristic": 68.0,
            },
            "elo_delta": 50,
        }

    @pytest.fixture
    def ineligible_results(self):
        """Gauntlet results that don't meet thresholds."""
        return {
            "win_rates": {
                "random": 80.0,
                "heuristic": 55.0,
            },
        }

    @pytest.mark.asyncio
    async def test_promotion_eligible(self, eligible_results):
        """Should run promotion script for eligible models."""
        mock_process = AsyncMock()
        mock_process.returncode = 0
        mock_process.communicate = AsyncMock(return_value=(b"Promoted successfully", b""))

        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            with patch("app.coordination.pipeline_actions._emit_promotion_complete"):
                result = await trigger_promotion(
                    model_path="/models/model.pth",
                    gauntlet_results=eligible_results,
                    board_type="hex8",
                    num_players=2,
                    iteration=5,
                )

        assert result.success is True
        assert result.stage == "promotion"
        assert result.metadata["promoted"] is True

    @pytest.mark.asyncio
    async def test_promotion_ineligible_skips(self, ineligible_results):
        """Should skip promotion for ineligible models."""
        result = await trigger_promotion(
            model_path="/models/model.pth",
            gauntlet_results=ineligible_results,
            board_type="hex8",
            num_players=2,
            iteration=5,
        )

        # Success = True but promoted = False
        assert result.success is True
        assert result.metadata["promoted"] is False
        assert "thresholds" in result.metadata["reason"].lower()

    @pytest.mark.asyncio
    async def test_promotion_with_cluster_sync(self, eligible_results):
        """Should pass sync_to_cluster flag."""
        mock_process = AsyncMock()
        mock_process.returncode = 0
        mock_process.communicate = AsyncMock(return_value=(b"Promoted and synced", b""))

        with patch("asyncio.create_subprocess_exec", return_value=mock_process) as exec_mock:
            with patch("app.coordination.pipeline_actions._emit_promotion_complete"):
                result = await trigger_promotion(
                    model_path="/models/model.pth",
                    gauntlet_results=eligible_results,
                    board_type="hex8",
                    num_players=2,
                    sync_to_cluster=True,
                )

        assert result.metadata["synced_to_cluster"] is True
        cmd_str = " ".join(exec_mock.call_args[0])
        assert "--sync-to-cluster" in cmd_str

    @pytest.mark.asyncio
    async def test_promotion_failure(self, eligible_results):
        """Should handle promotion script failure."""
        mock_process = AsyncMock()
        mock_process.returncode = 1
        mock_process.communicate = AsyncMock(
            return_value=(b"", b"Permission denied: /models")
        )

        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            result = await trigger_promotion(
                model_path="/models/model.pth",
                gauntlet_results=eligible_results,
                board_type="hex8",
                num_players=2,
            )

        assert result.success is False
        assert "Permission denied" in result.stderr


# =============================================================================
# Event Emission Tests
# =============================================================================


class TestEventEmission:
    """Tests for event emission helpers."""

    @pytest.mark.asyncio
    async def test_sync_emits_event_on_success(self):
        """trigger_data_sync should emit event on success."""
        mock_process = AsyncMock()
        mock_process.returncode = 0
        mock_process.communicate = AsyncMock(return_value=(b"Done", b""))

        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            with patch("app.coordination.pipeline_actions._emit_sync_complete") as emit_mock:
                await trigger_data_sync("hex8", 2, 1)

        emit_mock.assert_called_once()

    @pytest.mark.asyncio
    async def test_sync_no_event_on_failure(self):
        """trigger_data_sync should not emit success event on failure."""
        mock_process = AsyncMock()
        mock_process.returncode = 1
        mock_process.communicate = AsyncMock(return_value=(b"", b"Error"))

        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            with patch("app.coordination.pipeline_actions._emit_sync_complete") as emit_mock:
                await trigger_data_sync("hex8", 2, 1)

        emit_mock.assert_not_called()

    @pytest.mark.asyncio
    async def test_training_emits_complete_on_success(self, allow_training_quality_gate):
        """trigger_training should emit training_complete on success."""
        mock_process = AsyncMock()
        mock_process.returncode = 0
        mock_process.communicate = AsyncMock(return_value=(b"Done", b""))

        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            with patch("pathlib.Path.exists", return_value=True):
                with patch("app.coordination.pipeline_actions._emit_training_complete") as emit_mock:
                    await trigger_training("hex8", 2, "/data.npz", 1)

        emit_mock.assert_called_once()

    @pytest.mark.asyncio
    async def test_training_emits_failed_on_failure(self, allow_training_quality_gate):
        """trigger_training should emit training_failed on failure."""
        mock_process = AsyncMock()
        mock_process.returncode = 1
        mock_process.communicate = AsyncMock(return_value=(b"", b"Error"))

        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            with patch("app.coordination.pipeline_actions._emit_training_failed") as emit_mock:
                await trigger_training("hex8", 2, "/data.npz", 1)

        emit_mock.assert_called_once()

    @pytest.mark.asyncio
    async def test_event_emission_failure_logged(self):
        """Event helpers should catch and log failures internally."""
        mock_process = AsyncMock()
        mock_process.returncode = 0
        mock_process.communicate = AsyncMock(return_value=(b"Done", b""))

        # Mock the inner import to fail - the helper has its own try/except
        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            with patch.dict("sys.modules", {"app.coordination.event_emitters": None}):
                # The _emit_sync_complete helper catches ImportError internally
                # So the sync should still succeed
                result = await trigger_data_sync("hex8", 2, 1)

        # Sync still succeeds even if event emission fails
        assert result.success is True


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for action chaining."""

    @pytest.mark.asyncio
    async def test_export_to_training_chain(self, allow_training_quality_gate):
        """Should chain export output to training input."""
        # Mock export
        export_mock = AsyncMock()
        export_mock.returncode = 0
        export_mock.communicate = AsyncMock(
            return_value=(b"Exported 1000 samples", b"")
        )

        with patch("asyncio.create_subprocess_exec", return_value=export_mock):
            with patch("pathlib.Path.exists", return_value=True):
                with patch("app.coordination.pipeline_actions._emit_npz_export_complete"):
                    export_result = await trigger_npz_export(
                        board_type="hex8",
                        num_players=2,
                        iteration=1,
                    )

        assert export_result.success is True
        assert export_result.output_path is not None

        # Use export output for training
        train_mock = AsyncMock()
        train_mock.returncode = 0
        train_mock.communicate = AsyncMock(
            return_value=(b"val_loss: 0.15\npolicy_accuracy: 75%", b"")
        )

        with patch("asyncio.create_subprocess_exec", return_value=train_mock):
            with patch("pathlib.Path.exists", return_value=True):
                with patch("app.coordination.pipeline_actions._emit_training_complete"):
                    train_result = await trigger_training(
                        board_type="hex8",
                        num_players=2,
                        npz_path=export_result.output_path,
                        iteration=1,
                    )

        assert train_result.success is True
        assert train_result.metadata["policy_accuracy"] == 75.0

    @pytest.mark.asyncio
    async def test_evaluation_to_promotion_chain(self):
        """Should use evaluation results to decide promotion."""
        # Mock evaluation with eligible results
        eval_mock = AsyncMock()
        eval_mock.returncode = 0
        eval_mock.communicate = AsyncMock(
            return_value=(b"vs random: 90%\nvs heuristic: 65%", b"")
        )

        with patch("asyncio.create_subprocess_exec", return_value=eval_mock):
            with patch("app.coordination.pipeline_actions._emit_evaluation_complete"):
                eval_result = await trigger_evaluation(
                    model_path="/models/model.pth",
                    board_type="hex8",
                    num_players=2,
                )

        assert eval_result.success is True
        assert eval_result.metadata["promotion_eligible"] is True

        # Use win_rates for promotion decision
        promo_mock = AsyncMock()
        promo_mock.returncode = 0
        promo_mock.communicate = AsyncMock(return_value=(b"Promoted", b""))

        with patch("asyncio.create_subprocess_exec", return_value=promo_mock):
            with patch("app.coordination.pipeline_actions._emit_promotion_complete"):
                promo_result = await trigger_promotion(
                    model_path="/models/model.pth",
                    gauntlet_results={"win_rates": eval_result.metadata["win_rates"]},
                    board_type="hex8",
                    num_players=2,
                )

        assert promo_result.success is True
        assert promo_result.metadata["promoted"] is True


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Edge case and boundary condition tests."""

    @pytest.mark.asyncio
    async def test_empty_stdout_parsing(self, allow_training_quality_gate):
        """Should handle empty stdout gracefully."""
        mock_process = AsyncMock()
        mock_process.returncode = 0
        mock_process.communicate = AsyncMock(return_value=(b"", b""))

        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            with patch("pathlib.Path.exists", return_value=True):
                with patch("app.coordination.pipeline_actions._emit_training_complete"):
                    result = await trigger_training(
                        board_type="hex8",
                        num_players=2,
                        npz_path="/data.npz",
                        iteration=1,
                    )

        assert result.success is True
        assert result.metadata["train_loss"] == 0.0
        assert result.metadata["policy_accuracy"] == 0.0

    @pytest.mark.asyncio
    async def test_malformed_stdout_parsing(self, allow_training_quality_gate):
        """Should handle malformed output without crashing."""
        mock_process = AsyncMock()
        mock_process.returncode = 0
        mock_process.communicate = AsyncMock(
            return_value=(b"train_loss: not_a_number\npolicy_accuracy: ???", b"")
        )

        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            with patch("pathlib.Path.exists", return_value=True):
                with patch("app.coordination.pipeline_actions._emit_training_complete"):
                    result = await trigger_training(
                        board_type="hex8",
                        num_players=2,
                        npz_path="/data.npz",
                        iteration=1,
                    )

        # Should not crash, should use default values
        assert result.success is True

    @pytest.mark.asyncio
    async def test_unicode_in_stderr(self):
        """Should handle unicode characters in output."""
        mock_process = AsyncMock()
        mock_process.returncode = 1
        mock_process.communicate = AsyncMock(
            return_value=(b"", "Error: файл не найден".encode("utf-8"))
        )

        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            result = await trigger_data_sync("hex8", 2, 1)

        assert result.success is False
        assert "файл" in result.stderr

    @pytest.mark.asyncio
    async def test_empty_gauntlet_results(self):
        """Should handle empty gauntlet results."""
        result = await trigger_promotion(
            model_path="/models/model.pth",
            gauntlet_results={},
            board_type="hex8",
            num_players=2,
        )

        # Empty = not eligible
        assert result.success is True
        assert result.metadata["promoted"] is False

    @pytest.mark.asyncio
    async def test_partial_gauntlet_results(self):
        """Should handle partial gauntlet results."""
        result = await trigger_promotion(
            model_path="/models/model.pth",
            gauntlet_results={"win_rates": {"random": 90.0}},  # Missing heuristic
            board_type="hex8",
            num_players=2,
        )

        # Missing heuristic = not eligible
        assert result.metadata["promoted"] is False

    @pytest.mark.asyncio
    async def test_duration_calculated(self):
        """Should calculate duration correctly."""
        mock_process = AsyncMock()
        mock_process.returncode = 0

        async def slow_communicate():
            await asyncio.sleep(0.05)  # 50ms
            return (b"Done", b"")

        mock_process.communicate = slow_communicate

        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            with patch("app.coordination.pipeline_actions._emit_sync_complete"):
                result = await trigger_data_sync("hex8", 2, 1)

        assert result.duration_seconds >= 0.05
