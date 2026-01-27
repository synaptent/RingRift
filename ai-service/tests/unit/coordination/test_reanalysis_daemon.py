"""Unit tests for ReanalysisDaemon.

January 27, 2026 - Phase 2.1: Reanalysis Pipeline Integration
"""

from __future__ import annotations

import asyncio
import json
import tempfile
import time
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.coordination.reanalysis_daemon import (
    ConfigReanalysisState,
    ReanalysisConfig,
    ReanalysisDaemon,
    create_reanalysis_daemon,
    get_reanalysis_daemon,
    reset_reanalysis_daemon,
)


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def config(temp_dir: Path) -> ReanalysisConfig:
    """Create test configuration."""
    return ReanalysisConfig(
        min_elo_delta=50,
        min_interval_hours=6.0,
        enabled=True,
        dry_run=True,  # Don't actually run reanalysis in tests
        state_path=temp_dir / "reanalysis_state.json",
        training_dir=temp_dir / "training",
        output_dir=temp_dir / "reanalysis",
    )


@pytest.fixture
def daemon(config: ReanalysisConfig) -> ReanalysisDaemon:
    """Create a ReanalysisDaemon instance for testing."""
    reset_reanalysis_daemon()
    d = create_reanalysis_daemon(config)
    yield d
    reset_reanalysis_daemon()


class TestReanalysisConfig:
    """Tests for ReanalysisConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = ReanalysisConfig()
        assert config.min_elo_delta == 50
        assert config.min_interval_hours == 6.0
        assert config.max_games_per_run == 1000
        assert config.value_blend_ratio == 0.7
        assert config.policy_blend_ratio == 0.8
        assert config.enabled is True
        assert config.dry_run is False
        assert config.use_mcts is False

    def test_path_conversion(self):
        """Test that string paths are converted to Path objects."""
        config = ReanalysisConfig(
            state_path="data/state.json",
            training_dir="data/training",
            output_dir="data/output",
        )
        assert isinstance(config.state_path, Path)
        assert isinstance(config.training_dir, Path)
        assert isinstance(config.output_dir, Path)


class TestConfigReanalysisState:
    """Tests for ConfigReanalysisState dataclass."""

    def test_default_values(self):
        """Test default state values."""
        state = ConfigReanalysisState(config_key="hex8_2p")
        assert state.config_key == "hex8_2p"
        assert state.last_reanalysis_elo == 0.0
        assert state.last_reanalysis_time == 0.0
        assert state.current_elo == 0.0
        assert state.reanalysis_count == 0
        assert state.last_npz_path == ""
        assert state.last_output_path == ""


class TestReanalysisDaemon:
    """Tests for ReanalysisDaemon."""

    def test_initialization(self, daemon: ReanalysisDaemon):
        """Test daemon initialization."""
        assert daemon.config.enabled is True
        assert daemon.config.dry_run is True
        assert len(daemon._config_states) == 0
        assert len(daemon._reanalysis_in_progress) == 0

    def test_get_event_subscriptions(self, daemon: ReanalysisDaemon):
        """Test event subscription registration."""
        subscriptions = daemon._get_event_subscriptions()
        assert "model_promoted" in subscriptions
        assert "evaluation_completed" in subscriptions

    def test_should_reanalyze_elo_delta(self, daemon: ReanalysisDaemon):
        """Test reanalysis trigger based on Elo delta."""
        state = ConfigReanalysisState(
            config_key="hex8_2p",
            current_elo=1100,
            last_reanalysis_elo=1000,
            last_reanalysis_time=0,  # Long ago
        )

        # Elo delta of 100 > threshold of 50
        assert daemon._should_reanalyze(state) is True

    def test_should_not_reanalyze_small_delta(self, daemon: ReanalysisDaemon):
        """Test reanalysis not triggered for small Elo delta."""
        state = ConfigReanalysisState(
            config_key="hex8_2p",
            current_elo=1030,
            last_reanalysis_elo=1000,
            last_reanalysis_time=0,
        )

        # Elo delta of 30 < threshold of 50
        assert daemon._should_reanalyze(state) is False

    def test_should_not_reanalyze_too_recent(self, daemon: ReanalysisDaemon):
        """Test reanalysis not triggered if ran too recently."""
        state = ConfigReanalysisState(
            config_key="hex8_2p",
            current_elo=1100,
            last_reanalysis_elo=1000,
            last_reanalysis_time=time.time() - 3600,  # 1 hour ago
        )

        # 1 hour < 6 hour minimum interval
        assert daemon._should_reanalyze(state) is False

    def test_should_reanalyze_after_interval(self, daemon: ReanalysisDaemon):
        """Test reanalysis triggered after minimum interval."""
        state = ConfigReanalysisState(
            config_key="hex8_2p",
            current_elo=1100,
            last_reanalysis_elo=1000,
            last_reanalysis_time=time.time() - (7 * 3600),  # 7 hours ago
        )

        # 7 hours > 6 hour minimum interval AND delta > threshold
        assert daemon._should_reanalyze(state) is True

    def test_get_or_create_state(self, daemon: ReanalysisDaemon):
        """Test state creation for new configs."""
        assert "hex8_2p" not in daemon._config_states

        state = daemon._get_or_create_state("hex8_2p")

        assert state.config_key == "hex8_2p"
        assert "hex8_2p" in daemon._config_states

        # Getting again returns same state
        state2 = daemon._get_or_create_state("hex8_2p")
        assert state is state2

    def test_health_check_healthy(self, daemon: ReanalysisDaemon):
        """Test health check returns healthy status."""
        daemon._successful_reanalyses = 5
        daemon._failed_reanalyses = 0

        result = daemon.health_check()

        assert result.healthy is True
        assert result.status == "healthy"
        assert result.details["enabled"] is True
        assert result.details["success_rate"] == 1.0

    def test_health_check_degraded(self, daemon: ReanalysisDaemon):
        """Test health check returns degraded on high failure rate."""
        daemon._successful_reanalyses = 2
        daemon._failed_reanalyses = 8

        result = daemon.health_check()

        assert result.healthy is False
        assert result.status == "degraded"
        assert result.details["success_rate"] == 0.2

    def test_health_check_disabled(self, daemon: ReanalysisDaemon):
        """Test health check reports disabled status."""
        daemon.config.enabled = False

        result = daemon.health_check()

        assert result.status == "disabled"
        assert result.details["enabled"] is False

    def test_get_stats(self, daemon: ReanalysisDaemon):
        """Test stats retrieval."""
        daemon._total_reanalyses = 10
        daemon._successful_reanalyses = 8
        daemon._failed_reanalyses = 2
        daemon._config_states["hex8_2p"] = ConfigReanalysisState(
            config_key="hex8_2p",
            current_elo=1100,
            last_reanalysis_elo=1000,
            reanalysis_count=3,
        )

        stats = daemon.get_stats()

        assert stats["total_reanalyses"] == 10
        assert stats["successful_reanalyses"] == 8
        assert stats["failed_reanalyses"] == 2
        assert "hex8_2p" in stats["config_states"]
        assert stats["config_states"]["hex8_2p"]["elo_delta"] == 100


class TestReanalysisDaemonState:
    """Tests for state persistence."""

    def test_save_and_load_state(self, config: ReanalysisConfig):
        """Test state persistence to disk."""
        reset_reanalysis_daemon()
        daemon = create_reanalysis_daemon(config)

        # Add some state
        daemon._config_states["hex8_2p"] = ConfigReanalysisState(
            config_key="hex8_2p",
            current_elo=1100,
            last_reanalysis_elo=1000,
            last_reanalysis_time=12345.0,
            reanalysis_count=3,
        )
        daemon._total_reanalyses = 10
        daemon._successful_reanalyses = 8
        daemon._failed_reanalyses = 2

        # Save state
        daemon._save_state()

        assert config.state_path.exists()

        # Create new daemon and verify state loaded
        reset_reanalysis_daemon()
        daemon2 = create_reanalysis_daemon(config)

        assert "hex8_2p" in daemon2._config_states
        assert daemon2._config_states["hex8_2p"].current_elo == 1100
        assert daemon2._config_states["hex8_2p"].last_reanalysis_elo == 1000
        assert daemon2._config_states["hex8_2p"].reanalysis_count == 3
        assert daemon2._total_reanalyses == 10

        reset_reanalysis_daemon()


class TestReanalysisDaemonEvents:
    """Tests for event handling."""

    @pytest.mark.asyncio
    async def test_on_model_promoted_triggers_reanalysis(
        self, daemon: ReanalysisDaemon
    ):
        """Test MODEL_PROMOTED event triggers reanalysis check."""
        # Set up state with old reanalysis
        daemon._config_states["hex8_2p"] = ConfigReanalysisState(
            config_key="hex8_2p",
            last_reanalysis_elo=1000,
            last_reanalysis_time=0,  # Long ago
        )

        event = {
            "config_key": "hex8_2p",
            "elo": 1100,  # Delta of 100 > threshold of 50
            "model_path": "models/canonical_hex8_2p.pth",
        }

        # Should trigger reanalysis (in dry run mode)
        await daemon._on_model_promoted(event)

        # State should be updated
        assert daemon._config_states["hex8_2p"].current_elo == 1100

    @pytest.mark.asyncio
    async def test_on_model_promoted_no_config_key(self, daemon: ReanalysisDaemon):
        """Test MODEL_PROMOTED event with missing config_key is ignored."""
        event = {
            "elo": 1100,
            "model_path": "models/test.pth",
        }

        # Should not raise, just log warning
        await daemon._on_model_promoted(event)

    @pytest.mark.asyncio
    async def test_on_evaluation_completed_updates_elo(
        self, daemon: ReanalysisDaemon
    ):
        """Test EVALUATION_COMPLETED event updates Elo tracking."""
        event = {
            "config_key": "hex8_2p",
            "elo": 1150,
        }

        await daemon._on_evaluation_completed(event)

        assert "hex8_2p" in daemon._config_states
        assert daemon._config_states["hex8_2p"].current_elo == 1150

    @pytest.mark.asyncio
    async def test_on_evaluation_completed_only_updates_higher_elo(
        self, daemon: ReanalysisDaemon
    ):
        """Test EVALUATION_COMPLETED only updates if new Elo is higher."""
        daemon._config_states["hex8_2p"] = ConfigReanalysisState(
            config_key="hex8_2p",
            current_elo=1200,
        )

        event = {
            "config_key": "hex8_2p",
            "elo": 1100,  # Lower than current
        }

        await daemon._on_evaluation_completed(event)

        # Elo should not be updated
        assert daemon._config_states["hex8_2p"].current_elo == 1200


class TestReanalysisDaemonNPZFinding:
    """Tests for NPZ file discovery."""

    def test_find_npz_for_config_basic(self, daemon: ReanalysisDaemon, temp_dir: Path):
        """Test finding NPZ file with basic naming."""
        daemon.config.training_dir = temp_dir

        # Create test NPZ file
        npz_path = temp_dir / "hex8_2p.npz"
        npz_path.touch()

        result = daemon._find_npz_for_config("hex8_2p")

        assert result == npz_path

    def test_find_npz_for_config_canonical(
        self, daemon: ReanalysisDaemon, temp_dir: Path
    ):
        """Test finding NPZ file with canonical naming."""
        daemon.config.training_dir = temp_dir

        npz_path = temp_dir / "canonical_hex8_2p.npz"
        npz_path.touch()

        result = daemon._find_npz_for_config("hex8_2p")

        assert result == npz_path

    def test_find_npz_for_config_not_found(
        self, daemon: ReanalysisDaemon, temp_dir: Path
    ):
        """Test NPZ file not found returns None."""
        daemon.config.training_dir = temp_dir

        result = daemon._find_npz_for_config("hex8_2p")

        assert result is None


class TestReanalysisDaemonFactory:
    """Tests for factory functions."""

    def test_create_reanalysis_daemon(self, config: ReanalysisConfig):
        """Test factory function creates daemon."""
        reset_reanalysis_daemon()
        daemon = create_reanalysis_daemon(config)

        assert isinstance(daemon, ReanalysisDaemon)
        assert daemon.config.dry_run is True

        reset_reanalysis_daemon()

    @pytest.mark.asyncio
    async def test_get_reanalysis_daemon_singleton(self, config: ReanalysisConfig):
        """Test singleton getter returns same instance."""
        reset_reanalysis_daemon()

        # First call creates instance
        daemon1 = await get_reanalysis_daemon()
        daemon2 = await get_reanalysis_daemon()

        assert daemon1 is daemon2

        reset_reanalysis_daemon()


class TestReanalysisDaemonDisabled:
    """Tests for disabled daemon behavior."""

    @pytest.mark.asyncio
    async def test_run_cycle_when_disabled(self, daemon: ReanalysisDaemon):
        """Test run cycle does nothing when disabled."""
        daemon.config.enabled = False

        await daemon._run_cycle()

        # Should return immediately without doing anything
        assert daemon._last_check_time == 0.0

    @pytest.mark.asyncio
    async def test_on_model_promoted_when_disabled(self, daemon: ReanalysisDaemon):
        """Test event handling when disabled."""
        daemon.config.enabled = False

        event = {
            "config_key": "hex8_2p",
            "elo": 1100,
        }

        await daemon._on_model_promoted(event)

        # Should not create state when disabled
        assert "hex8_2p" not in daemon._config_states
