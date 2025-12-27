"""Tests for delivery verification handlers.

Tests the DeliveryHandlersMixin which provides:
- POST /delivery/verify - Verify file received and matches checksum
- GET /delivery/status/{node_id} - Get delivery status for a node

December 2025: Created as part of Phase 3 infrastructure improvements.
"""

import hashlib
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aiohttp import web
from aiohttp.test_utils import AioHTTPTestCase, unittest_run_loop

from scripts.p2p.handlers.delivery import (
    DeliveryHandlersMixin,
    compute_file_checksum,
)


class TestComputeFileChecksum:
    """Test the compute_file_checksum utility function."""

    def test_compute_sha256_checksum(self, tmp_path: Path):
        """Test computing SHA256 checksum of a file."""
        test_file = tmp_path / "test.txt"
        test_file.write_bytes(b"Hello, World!")

        checksum = compute_file_checksum(test_file)

        # Verify checksum is correct
        expected = hashlib.sha256(b"Hello, World!").hexdigest()
        assert checksum == expected

    def test_compute_checksum_large_file(self, tmp_path: Path):
        """Test checksum computation on a larger file."""
        test_file = tmp_path / "large.bin"
        # Create a 1MB file
        data = b"x" * (1024 * 1024)
        test_file.write_bytes(data)

        checksum = compute_file_checksum(test_file)

        expected = hashlib.sha256(data).hexdigest()
        assert checksum == expected

    def test_compute_checksum_file_not_found(self, tmp_path: Path):
        """Test that FileNotFoundError is raised for missing files."""
        missing_file = tmp_path / "missing.txt"

        with pytest.raises(FileNotFoundError):
            compute_file_checksum(missing_file)

    def test_compute_checksum_with_string_path(self, tmp_path: Path):
        """Test that string paths work."""
        test_file = tmp_path / "test.txt"
        test_file.write_bytes(b"test content")

        checksum = compute_file_checksum(str(test_file))

        expected = hashlib.sha256(b"test content").hexdigest()
        assert checksum == expected


class MockOrchestrator(DeliveryHandlersMixin):
    """Mock orchestrator for testing the mixin."""

    def __init__(self, node_id: str = "test-node"):
        self.node_id = node_id


class TestDeliveryVerifyHandler:
    """Test the handle_delivery_verify endpoint."""

    @pytest.fixture
    def orchestrator(self):
        """Create a mock orchestrator."""
        return MockOrchestrator(node_id="test-node-1")

    @pytest.fixture
    def test_file(self, tmp_path: Path):
        """Create a test file."""
        test_file = tmp_path / "model.pth"
        test_file.write_bytes(b"fake model content")
        return test_file

    @pytest.mark.asyncio
    async def test_verify_existing_file(self, orchestrator, test_file):
        """Test verifying an existing file."""
        request = MagicMock()
        request.json = AsyncMock(return_value={
            "file_path": str(test_file),
        })

        response = await orchestrator.handle_delivery_verify(request)

        assert response.status == 200
        data = await self._get_json_response(response)
        assert data["verified"] is True
        assert data["exists"] is True
        assert data["file_path"] == str(test_file)
        assert "actual_checksum" in data
        assert data["actual_checksum"].startswith("sha256:")
        assert data["node_id"] == "test-node-1"

    @pytest.mark.asyncio
    async def test_verify_missing_file(self, orchestrator, tmp_path: Path):
        """Test verifying a missing file."""
        missing = tmp_path / "missing.pth"
        request = MagicMock()
        request.json = AsyncMock(return_value={
            "file_path": str(missing),
        })

        response = await orchestrator.handle_delivery_verify(request)

        assert response.status == 200
        data = await self._get_json_response(response)
        assert data["verified"] is False
        assert data["exists"] is False
        assert "error" in data

    @pytest.mark.asyncio
    async def test_verify_with_expected_checksum_match(self, orchestrator, test_file):
        """Test verifying with matching expected checksum."""
        expected = hashlib.sha256(b"fake model content").hexdigest()

        request = MagicMock()
        request.json = AsyncMock(return_value={
            "file_path": str(test_file),
            "expected_checksum": f"sha256:{expected}",
        })

        response = await orchestrator.handle_delivery_verify(request)

        assert response.status == 200
        data = await self._get_json_response(response)
        assert data["verified"] is True
        assert data["checksum_match"] is True

    @pytest.mark.asyncio
    async def test_verify_with_expected_checksum_mismatch(self, orchestrator, test_file):
        """Test verifying with mismatching expected checksum."""
        request = MagicMock()
        request.json = AsyncMock(return_value={
            "file_path": str(test_file),
            "expected_checksum": "sha256:wrongchecksum",
        })

        response = await orchestrator.handle_delivery_verify(request)

        assert response.status == 200
        data = await self._get_json_response(response)
        assert data["verified"] is False
        assert data["checksum_match"] is False

    @pytest.mark.asyncio
    async def test_verify_missing_file_path(self, orchestrator):
        """Test error when file_path is missing."""
        request = MagicMock()
        request.json = AsyncMock(return_value={})

        response = await orchestrator.handle_delivery_verify(request)

        assert response.status == 400
        data = await self._get_json_response(response)
        assert "error" in data

    @pytest.mark.asyncio
    async def test_verify_with_file_type_model(self, orchestrator, tmp_path: Path):
        """Test type-specific validation for model files."""
        model_file = tmp_path / "model.pth"
        model_file.write_bytes(b"not a real pytorch file")

        request = MagicMock()
        request.json = AsyncMock(return_value={
            "file_path": str(model_file),
            "file_type": "model",
        })

        response = await orchestrator.handle_delivery_verify(request)

        assert response.status == 200
        data = await self._get_json_response(response)
        # Should have type_validation (will fail since not real model)
        assert "type_validation" in data
        assert data["type_validation"]["type"] == "model"

    @pytest.mark.asyncio
    async def test_verify_normalizes_checksum_format(self, orchestrator, test_file):
        """Test that checksum without sha256: prefix is normalized."""
        expected = hashlib.sha256(b"fake model content").hexdigest()

        request = MagicMock()
        request.json = AsyncMock(return_value={
            "file_path": str(test_file),
            "expected_checksum": expected,  # No sha256: prefix
        })

        response = await orchestrator.handle_delivery_verify(request)

        assert response.status == 200
        data = await self._get_json_response(response)
        assert data["checksum_match"] is True

    async def _get_json_response(self, response: web.Response) -> dict:
        """Helper to extract JSON from web.Response."""
        import json
        return json.loads(response.body)


class TestDeliveryStatusHandler:
    """Test the handle_delivery_status endpoint."""

    @pytest.fixture
    def orchestrator(self):
        """Create a mock orchestrator."""
        return MockOrchestrator(node_id="test-node-1")

    @pytest.mark.asyncio
    async def test_status_missing_node_id(self, orchestrator):
        """Test error when node_id is missing."""
        request = MagicMock()
        request.match_info = {}

        response = await orchestrator.handle_delivery_status(request)

        assert response.status == 400
        data = await self._get_json_response(response)
        assert "error" in data

    @pytest.mark.asyncio
    async def test_status_ledger_not_available(self, orchestrator):
        """Test response when delivery ledger is not available."""
        request = MagicMock()
        request.match_info = {"node_id": "runpod-h100"}

        # Patch the import to raise ImportError
        with patch.dict('sys.modules', {'app.coordination.delivery_ledger': None}):
            response = await orchestrator.handle_delivery_status(request)

        # Should return 501 Not Implemented
        assert response.status == 501
        data = await self._get_json_response(response)
        assert data["available"] is False

    @pytest.mark.asyncio
    async def test_status_with_ledger(self, orchestrator):
        """Test status when ledger is available."""
        request = MagicMock()
        request.match_info = {"node_id": "runpod-h100"}

        # Mock the delivery ledger
        mock_ledger = MagicMock()
        mock_ledger.get_node_delivery_status.return_value = {
            "total_verified": 15,
            "failure_rate_24h": 0.05,
            "status_counts": {"verified": 12, "failed": 3},
        }

        mock_delivery = MagicMock()
        mock_delivery.to_dict.return_value = {"delivery_id": "test-123"}
        mock_ledger.get_deliveries_for_node.return_value = [mock_delivery]

        with patch(
            "scripts.p2p.handlers.delivery.get_delivery_ledger",
            return_value=mock_ledger,
        ):
            # Need to make the import work
            import sys
            mock_module = MagicMock()
            mock_module.get_delivery_ledger = MagicMock(return_value=mock_ledger)
            sys.modules["app.coordination.delivery_ledger"] = mock_module

            response = await orchestrator.handle_delivery_status(request)

            # Clean up
            del sys.modules["app.coordination.delivery_ledger"]

        assert response.status == 200
        data = await self._get_json_response(response)
        assert data["node_id"] == "runpod-h100"
        assert data["total_verified"] == 15
        assert data["failure_rate_24h"] == 0.05

    async def _get_json_response(self, response: web.Response) -> dict:
        """Helper to extract JSON from web.Response."""
        import json
        return json.loads(response.body)


class TestValidateFileType:
    """Test the _validate_file_type method."""

    @pytest.fixture
    def orchestrator(self):
        """Create a mock orchestrator."""
        return MockOrchestrator(node_id="test-node-1")

    @pytest.mark.asyncio
    async def test_validate_unknown_type(self, orchestrator, tmp_path: Path):
        """Test validation with unknown file type."""
        test_file = tmp_path / "test.xyz"
        test_file.write_bytes(b"content")

        result = await orchestrator._validate_file_type(test_file, "unknown")

        assert result["type"] == "unknown"
        assert result["valid"] is True  # Unknown types pass by default

    @pytest.mark.asyncio
    async def test_validate_npz_valid(self, orchestrator, tmp_path: Path):
        """Test validation of valid NPZ file."""
        pytest.importorskip("numpy")
        import numpy as np

        npz_file = tmp_path / "data.npz"
        np.savez(npz_file, states=np.zeros((100, 10)))

        result = await orchestrator._validate_file_type(npz_file, "npz")

        assert result["type"] == "npz"
        assert result["valid"] is True
        assert "arrays" in result
        assert result["total_samples"] == 100

    @pytest.mark.asyncio
    async def test_validate_npz_invalid(self, orchestrator, tmp_path: Path):
        """Test validation of invalid NPZ file."""
        npz_file = tmp_path / "data.npz"
        npz_file.write_bytes(b"not a valid npz file")

        result = await orchestrator._validate_file_type(npz_file, "npz")

        assert result["type"] == "npz"
        assert result["valid"] is False
        assert "error" in result

    @pytest.mark.asyncio
    async def test_validate_games_valid(self, orchestrator, tmp_path: Path):
        """Test validation of valid SQLite games database."""
        import sqlite3

        db_file = tmp_path / "games.db"
        conn = sqlite3.connect(str(db_file))
        conn.execute("CREATE TABLE games (game_id TEXT)")
        conn.execute("INSERT INTO games VALUES ('game-1')")
        conn.execute("INSERT INTO games VALUES ('game-2')")
        conn.commit()
        conn.close()

        result = await orchestrator._validate_file_type(db_file, "games")

        assert result["type"] == "games"
        assert result["valid"] is True
        assert result["has_games_table"] is True
        assert result["game_count"] == 2

    @pytest.mark.asyncio
    async def test_validate_games_missing_table(self, orchestrator, tmp_path: Path):
        """Test validation of SQLite DB without games table."""
        import sqlite3

        db_file = tmp_path / "empty.db"
        conn = sqlite3.connect(str(db_file))
        conn.execute("CREATE TABLE other_data (id INTEGER)")
        conn.commit()
        conn.close()

        result = await orchestrator._validate_file_type(db_file, "games")

        assert result["type"] == "games"
        assert result["valid"] is True  # DB is valid, just missing games
        assert result["has_games_table"] is False


class TestIntegration:
    """Integration tests for delivery handlers."""

    @pytest.mark.asyncio
    async def test_full_verification_workflow(self, tmp_path: Path):
        """Test complete verification workflow."""
        # Create test files
        model_file = tmp_path / "model.pth"
        model_file.write_bytes(b"model data")

        orchestrator = MockOrchestrator(node_id="integration-test")

        # 1. Verify file exists and get checksum
        request1 = MagicMock()
        request1.json = AsyncMock(return_value={
            "file_path": str(model_file),
        })

        response1 = await orchestrator.handle_delivery_verify(request1)
        assert response1.status == 200

        import json
        data1 = json.loads(response1.body)
        actual_checksum = data1["actual_checksum"]

        # 2. Verify with expected checksum
        request2 = MagicMock()
        request2.json = AsyncMock(return_value={
            "file_path": str(model_file),
            "expected_checksum": actual_checksum,
        })

        response2 = await orchestrator.handle_delivery_verify(request2)
        assert response2.status == 200

        data2 = json.loads(response2.body)
        assert data2["verified"] is True
        assert data2["checksum_match"] is True
