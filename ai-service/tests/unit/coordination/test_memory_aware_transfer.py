"""Tests for memory-aware transfer functions in rsync_command_builder.py.

Tests should_use_rsync(), aria2_pull_file(), and trigger_remote_pull()
which implement memory-aware transfer fallback from rsync to aria2/HTTP.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.coordination.rsync_command_builder import (
    aria2_pull_file,
    should_use_rsync,
    trigger_remote_pull,
)


# =============================================================================
# should_use_rsync() tests
# =============================================================================


@patch("app.coordination.rsync_command_builder.logger")
def test_should_use_rsync_memory_ok(mock_logger):
    """50% memory used, coordinator node -> should allow rsync."""
    with patch(
        "app.utils.resource_guard.get_memory_usage",
        return_value=(50.0, 48.0, 96.0),
    ):
        with patch("app.config.env.env") as mock_env:
            mock_env.is_coordinator = True
            result = should_use_rsync()
            assert result is True


@patch("app.coordination.rsync_command_builder.logger")
def test_should_not_use_rsync_memory_high(mock_logger):
    """75% memory used on coordinator -> should NOT allow rsync."""
    with patch(
        "app.utils.resource_guard.get_memory_usage",
        return_value=(75.0, 24.0, 96.0),
    ):
        with patch("app.config.env.env") as mock_env:
            mock_env.is_coordinator = True
            result = should_use_rsync()
            assert result is False
            # Verify the fallback was logged
            mock_logger.info.assert_called_once()
            log_msg = mock_logger.info.call_args[0][0]
            assert "aria2" in log_msg.lower() or "fallback" in log_msg.lower()


@patch("app.coordination.rsync_command_builder.logger")
def test_should_use_rsync_not_coordinator(mock_logger):
    """High memory but NOT coordinator -> should still allow rsync (COORDINATOR_ONLY=True)."""
    with patch(
        "app.utils.resource_guard.get_memory_usage",
        return_value=(85.0, 14.0, 96.0),
    ):
        with patch("app.config.env.env") as mock_env:
            mock_env.is_coordinator = False
            result = should_use_rsync()
            assert result is True


@patch("app.coordination.rsync_command_builder.logger")
def test_should_use_rsync_feature_disabled(mock_logger):
    """ENABLED=False -> should always allow rsync regardless of memory."""
    # TransferMemoryDefaults reads env vars at class definition time, so we
    # must mock the class itself rather than patching os.environ after import.
    mock_defaults = MagicMock()
    mock_defaults.ENABLED = False
    mock_defaults.COORDINATOR_ONLY = True
    mock_defaults.RSYNC_MEMORY_THRESHOLD = 70.0

    with patch(
        "app.config.coordination_defaults.TransferMemoryDefaults",
        return_value=mock_defaults,
    ):
        with patch(
            "app.utils.resource_guard.get_memory_usage",
            return_value=(95.0, 5.0, 96.0),
        ):
            with patch("app.config.env.env") as mock_env:
                mock_env.is_coordinator = True
                result = should_use_rsync()
                assert result is True


# =============================================================================
# aria2_pull_file() tests
# =============================================================================


@pytest.mark.asyncio
async def test_aria2_pull_file_success():
    """Mock Aria2Transport.download_file() returning success."""
    mock_transport = MagicMock()
    mock_transport.is_available.return_value = True
    mock_transport.download_file = AsyncMock(
        return_value=(True, 1048576, "")
    )

    with patch(
        "app.distributed.aria2_transport.Aria2Transport",
        return_value=mock_transport,
    ):
        local_path = Path("/tmp/test_downloads")
        with patch.object(local_path, "mkdir"):
            success, bytes_dl, error = await aria2_pull_file(
                http_url="http://node-1:8770/data/file.db",
                local_path=local_path,
                filename="file.db",
                expected_checksum="abc123",
            )

        assert success is True
        assert bytes_dl == 1048576
        assert error == ""
        mock_transport.download_file.assert_awaited_once_with(
            sources=["http://node-1:8770/data/file.db"],
            output_dir=local_path,
            filename="file.db",
            expected_checksum="abc123",
        )


@pytest.mark.asyncio
async def test_aria2_pull_file_unavailable():
    """aria2c not installed -> returns (False, 0, error message)."""
    mock_transport = MagicMock()
    mock_transport.is_available.return_value = False

    with patch(
        "app.distributed.aria2_transport.Aria2Transport",
        return_value=mock_transport,
    ):
        success, bytes_dl, error = await aria2_pull_file(
            http_url="http://node-1:8770/data/file.db",
            local_path=Path("/tmp/test_downloads"),
        )

        assert success is False
        assert bytes_dl == 0
        assert "aria2c not installed" in error


# =============================================================================
# trigger_remote_pull() tests
# =============================================================================


@pytest.mark.asyncio
async def test_trigger_remote_pull_success():
    """Mock aiohttp POST returning 200 -> True."""
    mock_resp = AsyncMock()
    mock_resp.status = 200

    mock_session = AsyncMock()
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)
    mock_session.post = MagicMock(
        return_value=AsyncMock(
            __aenter__=AsyncMock(return_value=mock_resp),
            __aexit__=AsyncMock(return_value=False),
        )
    )

    with patch("aiohttp.ClientSession", return_value=mock_session):
        result = await trigger_remote_pull(
            target_host="192.168.1.1",
            target_port=8770,
            source_node_id="node-1",
            files=["file.db"],
        )
        assert result is True


@pytest.mark.asyncio
async def test_trigger_remote_pull_timeout():
    """Connection failure -> False."""
    import asyncio

    mock_session = AsyncMock()
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)
    mock_session.post = MagicMock(
        side_effect=asyncio.TimeoutError("connection timed out"),
    )

    with patch("aiohttp.ClientSession", return_value=mock_session):
        result = await trigger_remote_pull(
            target_host="192.168.1.1",
            target_port=8770,
            source_node_id="node-1",
            files=["file.db"],
            timeout=5.0,
        )
        assert result is False
