"""Tests for scripts.p2p.client module.

Tests the P2P orchestrator client library.
"""

import json
import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime

from scripts.p2p.client import (
    P2PClient,
    P2PClientError,
    JobType,
    JobStatus,
    JobRequest,
    JobResult,
    NodeInfo,
    ClusterStatus,
    get_client,
    get_cluster_status,
    submit_selfplay_job,
    submit_training_job,
)


class TestJobRequest:
    """Tests for JobRequest dataclass."""

    def test_basic_creation(self):
        """Test creating a basic job request."""
        job = JobRequest(
            job_type="selfplay",
            config_key="square8_2p",
            num_games=1000,
        )
        assert job.job_type == "selfplay"
        assert job.config_key == "square8_2p"
        assert job.num_games == 1000
        assert job.priority == 1
        assert job.node_id is None

    def test_with_node_assignment(self):
        """Test job request with specific node."""
        job = JobRequest(
            job_type="training",
            config_key="hex_2p",
            node_id="lambda-gh200-a",
            priority=2,
        )
        assert job.node_id == "lambda-gh200-a"
        assert job.priority == 2

    def test_with_metadata(self):
        """Test job request with metadata."""
        job = JobRequest(
            job_type="training",
            config_key="square8_2p",
            metadata={"epochs": 50, "batch_size": 256},
        )
        assert job.metadata["epochs"] == 50
        assert job.metadata["batch_size"] == 256


class TestJobResult:
    """Tests for JobResult dataclass."""

    def test_basic_result(self):
        """Test creating a job result."""
        result = JobResult(
            job_id="job-123",
            status=JobStatus.COMPLETED,
            node_id="lambda-gh200-a",
            output_path="/data/output",
        )
        assert result.job_id == "job-123"
        assert result.status == JobStatus.COMPLETED
        assert result.node_id == "lambda-gh200-a"

    def test_failed_result(self):
        """Test failed job result."""
        result = JobResult(
            job_id="job-456",
            status=JobStatus.FAILED,
            node_id="lambda-gh200-b",
            error="Out of memory",
        )
        assert result.status == JobStatus.FAILED
        assert result.error == "Out of memory"


class TestNodeInfo:
    """Tests for NodeInfo dataclass."""

    def test_node_info(self):
        """Test creating node info."""
        node = NodeInfo(
            node_id="lambda-gh200-a",
            host="100.1.2.3",
            online=True,
            gpu_percent=85.5,
            gpu_memory_percent=60.0,
            selfplay_jobs=3,
            training_jobs=1,
            role="leader",
            gpu_name="GH200",
        )
        assert node.node_id == "lambda-gh200-a"
        assert node.online is True
        assert node.gpu_percent == 85.5
        assert node.selfplay_jobs == 3


class TestClusterStatus:
    """Tests for ClusterStatus dataclass."""

    def test_cluster_status(self):
        """Test creating cluster status."""
        nodes = [
            NodeInfo(node_id="node-1", host="100.1.1.1", online=True),
            NodeInfo(node_id="node-2", host="100.1.1.2", online=True),
            NodeInfo(node_id="node-3", host="100.1.1.3", online=False),
        ]
        status = ClusterStatus(
            leader_id="node-1",
            role="follower",
            total_nodes=3,
            online_nodes=[n for n in nodes if n.online],
            active_selfplay_count=5,
            active_training_count=2,
        )
        assert status.leader_id == "node-1"
        assert len(status.online_nodes) == 2
        assert status.active_selfplay_count == 5


class TestP2PClient:
    """Tests for P2PClient class."""

    @pytest.fixture
    def client(self):
        """Create a client with auto-discovery disabled."""
        return P2PClient(host="localhost", port=8770, auto_discover=False)

    def test_client_creation(self, client):
        """Test client initialization."""
        assert client.host == "localhost"
        assert client.port == 8770
        assert client.base_url == "http://localhost:8770"

    def test_health_check_success(self, client):
        """Test successful health check."""
        mock_response = {"status": "healthy", "running": True}

        with patch.object(client, '_request', return_value=mock_response):
            assert client.health_check() is True

    def test_health_check_failure(self, client):
        """Test failed health check."""
        with patch.object(client, '_request', side_effect=P2PClientError("Connection refused")):
            assert client.health_check() is False

    def test_get_status(self, client):
        """Test getting cluster status."""
        mock_response = {
            "role": "leader",
            "leader_id": "lambda-gh200-a",
            "self": {
                "node_id": "lambda-gh200-a",
                "host": "100.1.2.3",
                "gpu_percent": 85.0,
                "gpu_memory_percent": 60.0,
                "cpu_percent": 45.0,
                "memory_percent": 30.0,
                "disk_percent": 50.0,
                "selfplay_jobs": 3,
                "training_jobs": 1,
                "gpu_name": "GH200",
            },
            "peers": {
                "node-2": {
                    "host": "100.1.2.4",
                    "last_heartbeat": 9999999999,  # Future timestamp
                    "gpu_percent": 70.0,
                    "selfplay_jobs": 2,
                }
            },
            "version": "1.0.0",
            "uptime_seconds": 3600,
        }

        with patch.object(client, '_request', return_value=mock_response):
            status = client.get_status()

            assert status.leader_id == "lambda-gh200-a"
            assert status.role == "leader"
            assert len(status.online_nodes) >= 1
            assert status.version == "1.0.0"

    def test_get_node(self, client):
        """Test getting specific node info."""
        mock_status = ClusterStatus(
            leader_id="node-1",
            online_nodes=[
                NodeInfo(node_id="node-1", host="100.1.1.1", online=True),
                NodeInfo(node_id="node-2", host="100.1.1.2", online=True),
            ],
        )

        with patch.object(client, 'get_status', return_value=mock_status):
            node = client.get_node("node-1")
            assert node is not None
            assert node.node_id == "node-1"

            missing = client.get_node("nonexistent")
            assert missing is None

    def test_submit_job(self, client):
        """Test submitting a job."""
        job = JobRequest(
            job_type="selfplay",
            config_key="square8_2p",
            num_games=1000,
        )
        mock_response = {"job_id": "job-abc123"}

        with patch.object(client, '_request', return_value=mock_response) as mock_req:
            job_id = client.submit_job(job)

            assert job_id == "job-abc123"
            mock_req.assert_called_once()
            call_args = mock_req.call_args
            assert call_args[0][0] == "/jobs"
            assert call_args[1]["method"] == "POST"

    def test_get_job(self, client):
        """Test getting job details."""
        mock_response = {
            "job_id": "job-123",
            "status": "completed",
            "node_id": "lambda-gh200-a",
            "started_at": "2025-01-01T00:00:00",
            "completed_at": "2025-01-01T01:00:00",
            "output_path": "/data/output",
            "metrics": {"games_played": 1000},
        }

        with patch.object(client, '_request', return_value=mock_response):
            result = client.get_job("job-123")

            assert result is not None
            assert result.job_id == "job-123"
            assert result.status == JobStatus.COMPLETED
            assert result.metrics["games_played"] == 1000

    def test_cancel_job(self, client):
        """Test cancelling a job."""
        with patch.object(client, '_request', return_value={"cancelled": True}):
            assert client.cancel_job("job-123") is True

        with patch.object(client, '_request', side_effect=P2PClientError("Not found")):
            assert client.cancel_job("nonexistent") is False


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    def test_get_client_singleton(self):
        """Test that get_client returns same instance."""
        client1 = get_client()
        client2 = get_client()
        assert client1 is client2

    def test_submit_selfplay_job(self):
        """Test submitting selfplay job via convenience function."""
        with patch('scripts.p2p.client.get_client') as mock_get_client:
            mock_client = MagicMock()
            mock_client.submit_job.return_value = "job-selfplay-123"
            mock_get_client.return_value = mock_client

            job_id = submit_selfplay_job(
                config_key="hex_2p",
                num_games=500,
            )

            assert job_id == "job-selfplay-123"
            mock_client.submit_job.assert_called_once()

    def test_submit_training_job(self):
        """Test submitting training job via convenience function."""
        with patch('scripts.p2p.client.get_client') as mock_get_client:
            mock_client = MagicMock()
            mock_client.submit_job.return_value = "job-training-456"
            mock_get_client.return_value = mock_client

            job_id = submit_training_job(
                config_key="square8_2p",
                epochs=100,
            )

            assert job_id == "job-training-456"
            mock_client.submit_job.assert_called_once()


class TestP2PClientError:
    """Tests for P2PClientError exception."""

    def test_error_message(self):
        """Test error message formatting."""
        error = P2PClientError("Connection refused")
        assert "Connection refused" in str(error)

    def test_error_is_exception(self):
        """Test that P2PClientError is an Exception."""
        error = P2PClientError("test")
        assert isinstance(error, Exception)
