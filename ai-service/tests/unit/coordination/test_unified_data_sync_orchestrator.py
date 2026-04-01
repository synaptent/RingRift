"""Focused tests for UnifiedDataSyncOrchestrator event emission helpers."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.coordination.unified_data_sync_orchestrator import (
    OrchestratorConfig,
    UnifiedDataSyncOrchestrator,
)


@pytest.fixture
def orchestrator() -> UnifiedDataSyncOrchestrator:
    """Create a sync orchestrator with a mocked cluster manifest."""
    with patch(
        "app.coordination.unified_data_sync_orchestrator.get_cluster_manifest",
        return_value=MagicMock(),
    ):
        return UnifiedDataSyncOrchestrator(
            OrchestratorConfig(
                s3_bucket="ringrift-test-bucket",
                owc_host="owc-test-host",
                owc_base_path="/tmp/owc",
            )
        )


class TestBackupTriggerEmission:
    """Test backup trigger events use the unified publish helper."""

    @pytest.mark.asyncio
    async def test_trigger_s3_backup_uses_publish_helper(
        self,
        orchestrator: UnifiedDataSyncOrchestrator,
    ) -> None:
        request = {
            "db_path": "/tmp/canonical_hex8_2p.db",
            "config_key": "hex8_2p",
        }

        with patch(
            "app.coordination.unified_data_sync_orchestrator.publish",
            new_callable=AsyncMock,
        ) as mock_publish:
            await orchestrator._trigger_s3_backup(request)

        mock_publish.assert_awaited_once()
        _, kwargs = mock_publish.call_args
        assert kwargs["event_type"] == "BACKUP_REQUESTED"
        assert kwargs["payload"]["destination"] == "s3"
        assert kwargs["payload"]["config_key"] == "hex8_2p"
        assert kwargs["payload"]["s3_bucket"] == "ringrift-test-bucket"

    @pytest.mark.asyncio
    async def test_trigger_owc_backup_uses_publish_helper(
        self,
        orchestrator: UnifiedDataSyncOrchestrator,
    ) -> None:
        request = {
            "db_path": "/tmp/canonical_hex8_2p.db",
            "config_key": "hex8_2p",
        }

        with patch(
            "app.coordination.unified_data_sync_orchestrator.publish",
            new_callable=AsyncMock,
        ) as mock_publish:
            await orchestrator._trigger_owc_backup(request)

        mock_publish.assert_awaited_once()
        _, kwargs = mock_publish.call_args
        assert kwargs["event_type"] == "BACKUP_REQUESTED"
        assert kwargs["payload"]["destination"] == "owc"
        assert kwargs["payload"]["config_key"] == "hex8_2p"
        assert kwargs["payload"]["owc_host"] == "owc-test-host"
        assert kwargs["payload"]["owc_base_path"] == "/tmp/owc"
