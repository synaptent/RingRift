"""Integration tests for the feedback loop system.

Tests the complete feedback cycle:
- Evaluation → Curriculum bridge
- Training triggers → Selfplay integration
- Health → Recovery integration
- Event-driven coordination
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

# Import the integration modules
from app.integration.evaluation_curriculum_bridge import (
    EvaluationCurriculumBridge,
    CurriculumState,
    create_evaluation_bridge,
)


class TestEvaluationCurriculumBridge:
    """Tests for the Evaluation → Curriculum feedback bridge."""

    def test_bridge_creation(self):
        """Test basic bridge creation."""
        bridge = EvaluationCurriculumBridge()
        assert bridge is not None
        assert len(bridge.state.weights) == len(bridge.ALL_CONFIGS)

    def test_all_configs_initialized(self):
        """Test all configs start with default weight."""
        bridge = EvaluationCurriculumBridge()
        for config in bridge.ALL_CONFIGS:
            assert bridge.state.weights[config] == 1.0

    def test_add_evaluation_result_updates_history(self):
        """Test that evaluation results are tracked."""
        bridge = EvaluationCurriculumBridge()

        bridge.add_evaluation_result(
            config_key="square8_2p",
            elo=1600,
            win_rate=0.55,
            games_played=100,
        )

        assert "square8_2p" in bridge.state.elo_history
        assert len(bridge.state.elo_history["square8_2p"]) == 1
        assert bridge.state.elo_history["square8_2p"][0] == 1600

    def test_weak_config_gets_higher_weight(self):
        """Test that weak configs get increased curriculum weight."""
        bridge = EvaluationCurriculumBridge(
            weak_threshold=0.45,
            max_weight=2.0,
        )

        # Add a weak config (below 0.45 win rate)
        for _ in range(5):
            bridge.add_evaluation_result(
                config_key="square8_2p",
                win_rate=0.35,  # Below threshold
            )

        weight = bridge.state.weights["square8_2p"]
        assert weight > 1.0, f"Expected weight > 1.0 for weak config, got {weight}"

    def test_strong_config_gets_lower_weight(self):
        """Test that strong configs get reduced curriculum weight."""
        bridge = EvaluationCurriculumBridge(
            strong_threshold=0.55,
            min_weight=0.5,
        )

        # Add a strong config (above 0.55 win rate)
        for _ in range(5):
            bridge.add_evaluation_result(
                config_key="hexagonal_2p",
                win_rate=0.70,  # Above threshold
            )

        weight = bridge.state.weights["hexagonal_2p"]
        assert weight < 1.0, f"Expected weight < 1.0 for strong config, got {weight}"

    def test_get_weak_configs(self):
        """Test detection of weak configurations."""
        bridge = EvaluationCurriculumBridge(weak_threshold=0.45)

        # Add weak config data
        for _ in range(5):
            bridge.add_evaluation_result(
                config_key="square19_4p",
                win_rate=0.30,
            )

        # Add strong config data
        for _ in range(5):
            bridge.add_evaluation_result(
                config_key="square8_2p",
                win_rate=0.60,
            )

        weak_configs = bridge.get_weak_configs()
        assert "square19_4p" in weak_configs
        assert "square8_2p" not in weak_configs

    def test_selfplay_coordinator_integration(self):
        """Test that bridge updates selfplay coordinator."""
        mock_coordinator = MagicMock()
        mock_coordinator.update_curriculum_weights = MagicMock()

        bridge = EvaluationCurriculumBridge(selfplay_coordinator=mock_coordinator)

        bridge.add_evaluation_result(
            config_key="square8_2p",
            win_rate=0.40,
        )

        mock_coordinator.update_curriculum_weights.assert_called_once()

    def test_callback_registration(self):
        """Test callback registration and invocation."""
        bridge = EvaluationCurriculumBridge()
        callback_data = []

        def callback(config_key, weights):
            callback_data.append((config_key, weights.copy()))

        bridge.register_callback(callback)
        bridge.add_evaluation_result(
            config_key="hexagonal_3p",
            elo=1550,
        )

        assert len(callback_data) == 1
        assert callback_data[0][0] == "hexagonal_3p"

    def test_get_status(self):
        """Test status reporting."""
        bridge = EvaluationCurriculumBridge()

        bridge.add_evaluation_result(config_key="square8_2p", elo=1600)

        status = bridge.get_status()
        assert "weights" in status
        assert "update_count" in status
        assert status["update_count"] == 1
        assert status["configs_tracked"] >= 1


class TestTrainingSelfplayIntegration:
    """Tests for training trigger → selfplay integration."""

    @pytest.mark.asyncio
    async def test_integrate_selfplay_with_training_creates_callbacks(self):
        """Test that integration creates proper callbacks."""
        from app.integration.p2p_integration import integrate_selfplay_with_training

        # Mock selfplay coordinator
        mock_coordinator = MagicMock()
        mock_coordinator.register_callback = MagicMock()

        result = integrate_selfplay_with_training(
            selfplay_coordinator=mock_coordinator,
            auto_trigger=False,
        )

        assert "triggers" in result
        assert "game_counts" in result
        assert "on_games_completed" in result

    @pytest.mark.asyncio
    async def test_games_completed_updates_triggers(self):
        """Test that game completion updates training triggers."""
        from app.integration.p2p_integration import integrate_selfplay_with_training
        from app.training.training_triggers import TrainingTriggers

        mock_coordinator = MagicMock()
        triggers = TrainingTriggers()

        result = integrate_selfplay_with_training(
            selfplay_coordinator=mock_coordinator,
            training_triggers=triggers,
            auto_trigger=False,
        )

        # Simulate game completion
        on_games_completed = result["on_games_completed"]
        await on_games_completed(
            config_key="square8_2p",
            new_games=100,
            total_games=600,
        )

        # Check game count was tracked
        assert result["game_counts"]["square8_2p"] == 600


class TestHealthRecoveryIntegration:
    """Tests for health check → recovery integration."""

    def test_health_recovery_integration_creation(self):
        """Test creation of health-recovery integration."""
        from app.distributed.health_checks import HealthRecoveryIntegration

        integration = HealthRecoveryIntegration(
            auto_recover=True,
            check_interval=30,
        )

        assert integration.auto_recover is True
        assert integration.check_interval == 30

    def test_health_recovery_status(self):
        """Test health-recovery status reporting."""
        from app.distributed.health_checks import HealthRecoveryIntegration

        integration = HealthRecoveryIntegration()
        status = integration.get_status()

        assert "running" in status
        assert "auto_recover" in status
        assert "consecutive_failures" in status
        assert "recovery_cooldown" in status


class TestFullFeedbackLoop:
    """Tests for the complete feedback loop integration."""

    def test_create_evaluation_bridge_with_all_components(self):
        """Test creating evaluation bridge with all optional components."""
        mock_controller = MagicMock()
        mock_router = MagicMock()
        mock_coordinator = MagicMock()
        mock_coordinator.update_curriculum_weights = MagicMock()

        bridge = create_evaluation_bridge(
            feedback_controller=mock_controller,
            feedback_router=mock_router,
            selfplay_coordinator=mock_coordinator,
        )

        assert bridge is not None
        assert bridge.selfplay_coordinator == mock_coordinator

    @pytest.mark.asyncio
    async def test_evaluation_to_curriculum_to_selfplay_flow(self):
        """Test complete evaluation → curriculum → selfplay flow."""
        # Setup mock selfplay coordinator
        mock_coordinator = MagicMock()
        weight_updates = []

        def capture_weights(weights):
            weight_updates.append(weights.copy())

        mock_coordinator.update_curriculum_weights = capture_weights

        # Create bridge
        bridge = EvaluationCurriculumBridge(selfplay_coordinator=mock_coordinator)

        # Simulate evaluation results for multiple configs
        configs_results = [
            ("square8_2p", 0.60),   # Strong
            ("square8_3p", 0.35),   # Weak
            ("hexagonal_2p", 0.50), # Normal
        ]

        for config, win_rate in configs_results:
            for _ in range(3):  # Add multiple results for reliable detection
                bridge.add_evaluation_result(
                    config_key=config,
                    win_rate=win_rate,
                )

        # Verify weights were updated
        assert len(weight_updates) > 0

        # Verify weak config has higher weight
        final_weights = weight_updates[-1]
        assert final_weights.get("square8_3p", 1.0) > final_weights.get("square8_2p", 1.0)
