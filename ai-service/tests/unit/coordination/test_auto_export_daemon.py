"""Tests for AutoExportDaemon.

Tests the automatic NPZ export triggering when selfplay completes.
"""

import asyncio
import pytest
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from app.coordination.auto_export_daemon import (
    AutoExportDaemon,
    AutoExportConfig,
    ConfigExportState,
)


@pytest.fixture
def mock_config():
    """Create a test configuration."""
    return AutoExportConfig(
        enabled=True,
        min_games_threshold=100,
        export_cooldown_seconds=10,
        max_concurrent_exports=2,
        export_timeout_seconds=60,
        use_incremental_export=True,
        require_completed_games=True,
        min_moves=10,
        output_dir=Path("/tmp/test_exports"),
        persist_state=False,  # Disable for tests
    )


@pytest.fixture
def daemon(mock_config):
    """Create a test daemon instance."""
    return AutoExportDaemon(config=mock_config)


class TestAutoExportDaemon:
    """Test AutoExportDaemon functionality."""

    @pytest.mark.asyncio
    async def test_daemon_initialization(self, daemon):
        """Test daemon initializes correctly."""
        assert daemon.config.min_games_threshold == 100
        assert daemon.config.export_cooldown_seconds == 10
        assert daemon._running is False
        assert len(daemon._export_states) == 0

    @pytest.mark.asyncio
    async def test_record_games_creates_state(self, daemon):
        """Test that recording games creates config state."""
        await daemon._record_games("hex8_2p", "hex8", 2, 50)

        assert "hex8_2p" in daemon._export_states
        state = daemon._export_states["hex8_2p"]
        assert state.board_type == "hex8"
        assert state.num_players == 2
        assert state.games_since_last_export == 50

    @pytest.mark.asyncio
    async def test_record_games_accumulates(self, daemon):
        """Test that recording games accumulates count."""
        await daemon._record_games("hex8_2p", "hex8", 2, 50)
        await daemon._record_games("hex8_2p", "hex8", 2, 30)

        state = daemon._export_states["hex8_2p"]
        assert state.games_since_last_export == 80

    @pytest.mark.asyncio
    async def test_threshold_triggers_export(self, daemon):
        """Test that reaching threshold triggers export."""
        with patch.object(daemon, '_run_export', new_callable=AsyncMock) as mock_export:
            # Record games below threshold
            await daemon._record_games("hex8_2p", "hex8", 2, 50)
            mock_export.assert_not_called()

            # Record more games to exceed threshold
            await daemon._record_games("hex8_2p", "hex8", 2, 60)

            # Give async task time to start
            await asyncio.sleep(0.1)

            # Export should have been triggered
            mock_export.assert_called_once_with("hex8_2p")

    @pytest.mark.asyncio
    async def test_cooldown_prevents_immediate_reexport(self, daemon):
        """Test that cooldown prevents rapid re-exports."""
        state = ConfigExportState(
            config_key="hex8_2p",
            board_type="hex8",
            num_players=2,
            games_since_last_export=150,
            last_export_time=time.time(),  # Just exported
        )
        daemon._export_states["hex8_2p"] = state

        with patch.object(daemon, '_run_export', new_callable=AsyncMock) as mock_export:
            await daemon._maybe_trigger_export("hex8_2p")

            # Should not trigger due to cooldown
            mock_export.assert_not_called()

    @pytest.mark.asyncio
    async def test_export_in_progress_prevents_duplicate(self, daemon):
        """Test that export_in_progress flag prevents duplicate exports."""
        state = ConfigExportState(
            config_key="hex8_2p",
            board_type="hex8",
            num_players=2,
            games_since_last_export=150,
            export_in_progress=True,
        )
        daemon._export_states["hex8_2p"] = state

        with patch.object(daemon, '_run_export', new_callable=AsyncMock) as mock_export:
            await daemon._maybe_trigger_export("hex8_2p")

            # Should not trigger due to in-progress flag
            mock_export.assert_not_called()

    @pytest.mark.asyncio
    async def test_selfplay_complete_handler(self, daemon):
        """Test SELFPLAY_COMPLETE event handler."""
        # Create mock result object
        result = MagicMock()
        result.board_type = "square8"
        result.num_players = 2
        result.games_generated = 75
        result.metadata = {}

        await daemon._on_selfplay_complete(result)

        # Verify state was updated
        assert "square8_2p" in daemon._export_states
        state = daemon._export_states["square8_2p"]
        assert state.games_since_last_export == 75

    @pytest.mark.asyncio
    async def test_selfplay_complete_with_metadata_fallback(self, daemon):
        """Test SELFPLAY_COMPLETE handler falls back to metadata."""
        # Create mock result with missing direct attributes
        result = MagicMock()
        result.board_type = None
        result.num_players = None
        result.games_generated = 0
        result.metadata = {
            "board_type": "hex8",
            "num_players": 4,
        }

        await daemon._on_selfplay_complete(result)

        # Verify state was created from metadata
        assert "hex8_4p" in daemon._export_states

    @pytest.mark.asyncio
    async def test_sync_complete_handler(self, daemon):
        """Test SYNC_COMPLETE event handler."""
        # Create mock result object
        result = MagicMock()
        result.metadata = {
            "config_key": "square8_2p",
            "games_synced": 125,
        }

        await daemon._on_sync_complete(result)

        # Verify state was updated
        assert "square8_2p" in daemon._export_states
        state = daemon._export_states["square8_2p"]
        assert state.games_since_last_export == 125

    @pytest.mark.asyncio
    async def test_parse_sample_count(self, daemon):
        """Test sample count parsing from export output."""
        outputs = [
            ("Exported 12345 samples to file.npz", 12345),
            ("Processing complete\nsamples: 5000\nDone", 5000),
            ("Total samples: 999", 999),
            ("Created file with 42 samples", 42),
            ("No match here", None),
        ]

        for output, expected in outputs:
            result = daemon._parse_sample_count(output)
            assert result == expected, f"Failed for: {output}"

    @pytest.mark.asyncio
    async def test_emit_export_started_uses_unified_event_helper(self, daemon):
        """Export-start notifications should go through the unified event helper."""
        with patch(
            "app.coordination.event_emission_helpers.safe_emit_event_async",
            new_callable=AsyncMock,
            return_value=True,
        ) as mock_emit:
            await daemon._emit_export_started("hex8_2p", games_pending=125)

        mock_emit.assert_awaited_once()
        event_type, payload = mock_emit.await_args.args[:2]
        assert event_type == "NPZ_EXPORT_STARTED"
        assert payload["config"] == "hex8_2p"
        assert payload["config_key"] == "hex8_2p"
        assert payload["games_pending"] == 125
        assert payload["board_type"] == "hex8"
        assert payload["num_players"] == 2
        assert payload["success"] is True
        assert payload["iteration"] == 0

    @pytest.mark.asyncio
    async def test_emit_export_complete_uses_unified_event_helper(self, daemon, tmp_path):
        """Export-complete notifications should go through the unified event helper."""
        daemon._export_states["hex8_2p"] = ConfigExportState(
            config_key="hex8_2p",
            board_type="hex8",
            num_players=2,
            games_since_last_export=0,
            last_export_games=77,
        )
        output_path = tmp_path / "hex8_2p.npz"
        output_path.write_bytes(b"npz")

        with patch(
            "app.coordination.event_emission_helpers.safe_emit_event_async",
            new_callable=AsyncMock,
            return_value=True,
        ) as mock_emit:
            await daemon._emit_export_complete("hex8_2p", output_path, samples=456)

        mock_emit.assert_awaited_once()
        event_type, payload = mock_emit.await_args.args[:2]
        assert event_type == "NPZ_EXPORT_COMPLETE"
        assert payload["config"] == "hex8_2p"
        assert payload["config_key"] == "hex8_2p"
        assert payload["output_path"] == str(output_path)
        assert payload["samples"] == 456
        assert payload["samples_exported"] == 456
        assert payload["games_exported"] == 77
        assert payload["board_type"] == "hex8"
        assert payload["num_players"] == 2
        assert payload["success"] is True
        assert payload["iteration"] == 0

    def test_config_defaults(self):
        """Test AutoExportConfig default values.

        December 2025: Thresholds lowered for faster training iteration:
        - min_games_threshold: 500 → 100 → 50
        - export_cooldown_seconds: 1800 → 300 → 60
        """
        config = AutoExportConfig()

        assert config.enabled is True
        # Lowered from 100 → 50 (Dec 2025) for faster training iteration
        assert config.min_games_threshold == 50
        # Reduced from 300s → 60s (Dec 2025) to minimize training data lag
        assert config.export_cooldown_seconds == 60
        # Increased from 2 → 4 (Jan 2026) for better parallelism
        assert config.max_concurrent_exports == 4
        assert config.use_incremental_export is True
        assert config.require_completed_games is True
        assert config.min_moves == 10
        assert config.persist_state is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
