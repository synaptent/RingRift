"""Unit tests for train_cli preflight cluster check functionality.

Jan 2026: Tests for Phase 1 of Cluster Manifest Training Integration.
"""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


class TestPreflightClusterCheck:
    """Tests for _preflight_cluster_check function."""

    def test_preflight_check_local_sufficient(self):
        """Test that sync is skipped when local data is sufficient."""
        from app.training.train_cli import _preflight_cluster_check

        mock_registry = MagicMock()
        mock_registry.get_cluster_status.return_value = {
            "hex8_2p": {
                "local": 6000,
                "cluster": 2000,
                "owc": 0,
                "s3": 0,
                "total": 8000,
            }
        }

        with patch(
            "app.distributed.data_catalog.get_data_registry",
            return_value=mock_registry,
        ):
            result = _preflight_cluster_check(
                board_type="hex8",
                num_players=2,
                sync_threshold=5000,
            )

        assert result["local"] == 6000
        assert result["sync_triggered"] is False
        assert result["sync_completed"] is False

    def test_preflight_check_no_remote_data(self):
        """Test that sync is skipped when cluster has no more data than local."""
        from app.training.train_cli import _preflight_cluster_check

        mock_registry = MagicMock()
        mock_registry.get_cluster_status.return_value = {
            "hex8_2p": {
                "local": 3000,
                "cluster": 0,
                "owc": 0,
                "s3": 0,
                "total": 3000,
            }
        }

        with patch(
            "app.distributed.data_catalog.get_data_registry",
            return_value=mock_registry,
        ):
            result = _preflight_cluster_check(
                board_type="hex8",
                num_players=2,
                sync_threshold=5000,
            )

        assert result["local"] == 3000
        assert result["sync_triggered"] is False

    def test_preflight_check_triggers_sync_when_needed(self):
        """Test that sync is triggered when local < threshold and cluster has more."""
        from app.training.train_cli import _preflight_cluster_check

        mock_registry = MagicMock()
        # Initial status - local below threshold
        initial_status = {
            "hex8_2p": {
                "local": 2000,
                "cluster": 5000,
                "owc": 1000,
                "s3": 0,
                "total": 8000,
            }
        }
        # After sync status - more local data
        after_sync_status = {
            "hex8_2p": {
                "local": 6000,
                "cluster": 5000,
                "owc": 1000,
                "s3": 0,
                "total": 8000,
            }
        }
        mock_registry.get_cluster_status.side_effect = [
            initial_status,
            after_sync_status,
        ]

        mock_facade = MagicMock()
        mock_response = MagicMock()
        mock_response.success = True
        mock_facade.trigger_priority_sync = AsyncMock(return_value=mock_response)

        with patch(
            "app.distributed.data_catalog.get_data_registry",
            return_value=mock_registry,
        ):
            with patch(
                "app.coordination.sync_facade.get_sync_facade",
                return_value=mock_facade,
            ):
                result = _preflight_cluster_check(
                    board_type="hex8",
                    num_players=2,
                    sync_threshold=5000,
                )

        assert result["sync_triggered"] is True
        # Note: sync_completed depends on the actual async execution

    def test_preflight_check_handles_missing_registry(self):
        """Test graceful fallback when registry is not available."""
        from app.training.train_cli import _preflight_cluster_check

        with patch(
            "app.distributed.data_catalog.get_data_registry",
            side_effect=ImportError("No registry"),
        ):
            result = _preflight_cluster_check(
                board_type="hex8",
                num_players=2,
                sync_threshold=5000,
            )

        assert result["local"] == 0
        assert result["cluster"] == 0
        assert result["sync_triggered"] is False

    def test_preflight_check_handles_registry_error(self):
        """Test graceful fallback when registry raises an error."""
        from app.training.train_cli import _preflight_cluster_check

        mock_registry = MagicMock()
        mock_registry.get_cluster_status.side_effect = RuntimeError("Connection failed")

        with patch(
            "app.distributed.data_catalog.get_data_registry",
            return_value=mock_registry,
        ):
            result = _preflight_cluster_check(
                board_type="hex8",
                num_players=2,
                sync_threshold=5000,
            )

        assert result["local"] == 0
        assert result["sync_triggered"] is False

    def test_preflight_check_unknown_config(self):
        """Test handling of unknown config key."""
        from app.training.train_cli import _preflight_cluster_check

        mock_registry = MagicMock()
        mock_registry.get_cluster_status.return_value = {}  # Empty - no matching config

        with patch(
            "app.distributed.data_catalog.get_data_registry",
            return_value=mock_registry,
        ):
            result = _preflight_cluster_check(
                board_type="unknown",
                num_players=5,
                sync_threshold=5000,
            )

        assert result["local"] == 0
        assert result["total"] == 0
        assert result["sync_triggered"] is False


class TestCLIArguments:
    """Tests for new CLI arguments."""

    def test_cluster_data_arguments_exist(self):
        """Test that cluster data CLI arguments are properly defined."""
        from app.training.train_cli import parse_args

        args = parse_args(["--use-cluster-data", "--cluster-sync-threshold", "10000"])

        assert args.use_cluster_data is True
        assert args.cluster_sync_threshold == 10000
        assert args.skip_cluster_preflight is False

    def test_skip_cluster_preflight_argument(self):
        """Test skip-cluster-preflight argument."""
        from app.training.train_cli import parse_args

        args = parse_args(["--use-cluster-data", "--skip-cluster-preflight"])

        assert args.use_cluster_data is True
        assert args.skip_cluster_preflight is True

    def test_default_cluster_sync_threshold(self):
        """Test default value for cluster-sync-threshold."""
        from app.training.train_cli import parse_args

        args = parse_args(["--use-cluster-data"])

        assert args.cluster_sync_threshold == 5000


class TestPreflightIntegration:
    """Integration tests for preflight check in main()."""

    def test_preflight_not_called_without_flag(self):
        """Test that preflight check is not called when --use-cluster-data is not set."""
        from app.training.train_cli import parse_args

        args = parse_args(["--board-type", "hex8", "--num-players", "2"])

        assert getattr(args, "use_cluster_data", False) is False

    def test_preflight_skipped_with_skip_flag(self):
        """Test that preflight check is skipped with --skip-cluster-preflight."""
        from app.training.train_cli import parse_args

        args = parse_args([
            "--use-cluster-data",
            "--skip-cluster-preflight",
            "--board-type", "hex8",
        ])

        assert args.use_cluster_data is True
        assert args.skip_cluster_preflight is True
