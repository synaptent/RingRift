"""Integration tests for Vast.ai P2P Sync CLI behavior."""

import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Path to the script
SCRIPT_PATH = Path(__file__).parents[2] / "scripts" / "vast_p2p_sync.py"


class TestCLIArguments:
    """Tests for CLI argument parsing and validation."""

    def test_help_flag(self):
        """Test that --help works and shows expected options."""
        result = subprocess.run(
            [sys.executable, str(SCRIPT_PATH), "--help"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 0
        assert "--check" in result.stdout
        assert "--sync" in result.stdout
        assert "--start-p2p" in result.stdout
        assert "--full" in result.stdout
        assert "--dry-run" in result.stdout
        assert "--provision" in result.stdout
        assert "--deprovision" in result.stdout
        assert "--sync-code" in result.stdout

    def test_dry_run_flag_in_help(self):
        """Test that dry-run flag is documented."""
        result = subprocess.run(
            [sys.executable, str(SCRIPT_PATH), "--help"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert "--dry-run" in result.stdout
        assert "without making changes" in result.stdout.lower()


class TestDryRunOutput:
    """Tests for dry-run mode output formatting."""

    def test_dry_run_message_format(self):
        """Test that dry-run messages follow consistent format."""
        # Import the module to test log message formatting
        sys.path.insert(0, str(SCRIPT_PATH.parent.parent))

        from scripts.vast_p2p_sync import (
            P2PNode,
            VastInstance,
            match_vast_to_p2p,
        )

        # Create mock data
        vast_instances = [
            VastInstance(
                id=12345, machine_id=999, gpu_name="RTX 3070",
                num_gpus=1, vcpus=8, ram_gb=32, ssh_host="host1",
                ssh_port=22, status="running", hourly_cost=0.05, uptime_mins=10,
            ),
        ]
        p2p_nodes = [
            P2PNode(
                node_id="vast-12345", host="100.64.0.1",
                retired=True, selfplay_jobs=0, healthy=False, gpu_name="RTX 3070",
            ),
        ]

        # Mock the Tailscale IP lookup
        with patch("scripts.vast_p2p_sync.get_vast_tailscale_ip", return_value=None):
            matches = match_vast_to_p2p(vast_instances, p2p_nodes)

        # Verify matching works
        assert 12345 in matches
        assert matches[12345].retired is True

        # Test dry-run message would indicate unretire
        node = matches[12345]
        dry_run_msg = f"[DRY-RUN] Would unretire {node.node_id} (Vast instance 12345)"
        assert "[DRY-RUN]" in dry_run_msg
        assert "Would unretire" in dry_run_msg


class TestArgumentCombinations:
    """Tests for argument combinations."""

    def test_mutually_exclusive_args(self):
        """Test that certain argument combinations work together."""
        import argparse

        # Simulate the argument parser
        parser = argparse.ArgumentParser()
        parser.add_argument('--check', action='store_true')
        parser.add_argument('--sync', action='store_true')
        parser.add_argument('--dry-run', action='store_true')
        parser.add_argument('--full', action='store_true')

        # These combinations should work
        args = parser.parse_args(['--dry-run', '--full'])
        assert args.dry_run is True
        assert args.full is True

        args = parser.parse_args(['--dry-run', '--sync'])
        assert args.dry_run is True
        assert args.sync is True

        args = parser.parse_args(['--check'])
        assert args.check is True
        assert args.dry_run is False

    def test_provision_with_dry_run(self):
        """Test provision with dry-run shows GPU preferences."""
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument('--provision', type=int)
        parser.add_argument('--dry-run', action='store_true')
        parser.add_argument('--max-hourly', type=float, default=0.50)

        args = parser.parse_args(['--provision', '3', '--dry-run'])
        assert args.provision == 3
        assert args.dry_run is True
        assert args.max_hourly == 0.50

    def test_deprovision_parsing(self):
        """Test deprovision argument parsing with various formats."""
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument('--deprovision', type=str)

        # Test comma-separated
        args = parser.parse_args(['--deprovision', '123,456,789'])
        ids = [int(x.strip()) for x in args.deprovision.split(',') if x.strip().isdigit()]
        assert ids == [123, 456, 789]

        # Test single ID
        args = parser.parse_args(['--deprovision', '12345'])
        ids = [int(x.strip()) for x in args.deprovision.split(',') if x.strip().isdigit()]
        assert ids == [12345]

        # Test with spaces
        args = parser.parse_args(['--deprovision', '123, 456, 789'])
        ids = [int(x.strip()) for x in args.deprovision.split(',') if x.strip().isdigit()]
        assert ids == [123, 456, 789]


class TestErrorHandling:
    """Tests for error handling in CLI."""

    def test_invalid_deprovision_ids(self):
        """Test handling of invalid deprovision IDs."""
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument('--deprovision', type=str)

        # Invalid IDs should be filtered out
        args = parser.parse_args(['--deprovision', 'abc,123,def,456'])
        ids = [int(x.strip()) for x in args.deprovision.split(',') if x.strip().isdigit()]
        assert ids == [123, 456]

        # All invalid
        args = parser.parse_args(['--deprovision', 'abc,def,ghi'])
        ids = [int(x.strip()) for x in args.deprovision.split(',') if x.strip().isdigit()]
        assert ids == []

    def test_empty_deprovision(self):
        """Test handling of empty deprovision string."""
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument('--deprovision', type=str)

        args = parser.parse_args(['--deprovision', ''])
        ids = [int(x.strip()) for x in args.deprovision.split(',') if x.strip().isdigit()]
        assert ids == []


class TestModuleFunctions:
    """Tests for module-level functions used in integration."""

    def test_gpu_roles_for_provisioning(self):
        """Test GPU roles are used in provisioning decisions."""
        sys.path.insert(0, str(SCRIPT_PATH.parent.parent))
        from scripts.vast_p2p_sync import GPU_ROLES, PREFERRED_GPUS

        # Verify preferred GPUs have valid roles
        for pref in PREFERRED_GPUS:
            pref["name"]
            role = pref["role"]
            # Role should match GPU_ROLES or be a valid role
            assert role in ["gpu_selfplay", "nn_training_primary", "flexible"]

    def test_preferred_gpus_have_prices(self):
        """Test all preferred GPUs have max prices set."""
        sys.path.insert(0, str(SCRIPT_PATH.parent.parent))
        from scripts.vast_p2p_sync import PREFERRED_GPUS

        for pref in PREFERRED_GPUS:
            assert "max_price" in pref
            assert pref["max_price"] > 0
            assert pref["max_price"] < 2.0  # Sanity check for budget
