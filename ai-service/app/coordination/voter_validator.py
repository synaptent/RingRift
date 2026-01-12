"""VoterValidator - Validate P2P voter configuration against Tailscale status.

Jan 12, 2026: Created for automated voter validation.

This module provides tools to:
1. Check all p2p_voters exist in Tailscale network
2. Warn if any voter is offline
3. Suggest voter list updates based on online status
4. Integrate with --sync-config to validate before sync

Usage:
    from app.coordination.voter_validator import VoterValidator, validate_voters_cli

    # Programmatic usage
    validator = VoterValidator()
    result = validator.validate()
    print(f"Valid: {result.is_valid}, Online: {result.online_count}/{result.total_count}")

    # CLI usage (returns exit code)
    exit_code = validate_voters_cli()
"""

from __future__ import annotations

import logging
import subprocess
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


@dataclass
class VoterInfo:
    """Information about a single voter node."""

    node_id: str
    tailscale_ip: str | None = None
    is_online: bool = False
    last_seen: str = ""
    hostname: str = ""
    error: str | None = None


@dataclass
class ValidationResult:
    """Result of voter validation."""

    is_valid: bool
    online_count: int
    total_count: int
    quorum_size: int
    voters: dict[str, VoterInfo] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)

    @property
    def has_quorum(self) -> bool:
        """Check if enough voters are online for quorum."""
        return self.online_count >= self.quorum_size


class VoterValidator:
    """Validates P2P voter configuration against Tailscale network status.

    Checks that:
    1. All voters have tailscale_ip configured
    2. All voters exist in the Tailscale network
    3. Enough voters are online for quorum
    """

    def __init__(
        self,
        config_path: Path | None = None,
        min_quorum_ratio: float = 0.5,
    ) -> None:
        """Initialize validator.

        Args:
            config_path: Path to distributed_hosts.yaml. Defaults to standard location.
            min_quorum_ratio: Minimum ratio of voters that must be online (default 0.5)
        """
        self._config_path = config_path or (
            Path(__file__).parent.parent.parent / "config" / "distributed_hosts.yaml"
        )
        self._min_quorum_ratio = min_quorum_ratio

    def validate(self) -> ValidationResult:
        """Validate voter configuration against Tailscale status.

        Returns:
            ValidationResult with validation status and details
        """
        result = ValidationResult(
            is_valid=False,
            online_count=0,
            total_count=0,
            quorum_size=0,
        )

        # Load config
        try:
            if not self._config_path.exists():
                result.errors.append(f"Config file not found: {self._config_path}")
                return result

            config = yaml.safe_load(self._config_path.read_text())
            voters = config.get("p2p_voters", [])
            hosts = config.get("hosts", {})
        except Exception as e:
            result.errors.append(f"Failed to load config: {e}")
            return result

        if not voters:
            result.errors.append("No p2p_voters configured")
            return result

        result.total_count = len(voters)
        result.quorum_size = max(1, int(len(voters) * self._min_quorum_ratio) + 1)

        # Get Tailscale status
        tailscale_peers = self._get_tailscale_status()

        # Check each voter
        online_count = 0
        offline_voters = []

        for voter_id in voters:
            host_cfg = hosts.get(voter_id, {})
            tailscale_ip = host_cfg.get("tailscale_ip")

            info = VoterInfo(
                node_id=voter_id,
                tailscale_ip=tailscale_ip,
            )

            if not tailscale_ip:
                info.error = "No tailscale_ip in config"
                result.errors.append(f"Voter '{voter_id}' has no tailscale_ip configured")
            elif tailscale_ip not in tailscale_peers:
                info.error = "Not found in Tailscale network"
                result.errors.append(
                    f"Voter '{voter_id}' ({tailscale_ip}) not found in Tailscale"
                )
                offline_voters.append(voter_id)
            else:
                peer_info = tailscale_peers[tailscale_ip]
                info.is_online = peer_info.get("online", False)
                info.hostname = peer_info.get("hostname", "")
                info.last_seen = peer_info.get("last_seen", "")

                if info.is_online:
                    online_count += 1
                else:
                    offline_voters.append(voter_id)

            result.voters[voter_id] = info

        result.online_count = online_count
        result.is_valid = result.has_quorum and len(result.errors) == 0

        # Generate suggestions
        if offline_voters:
            result.suggestions.append(
                f"Consider replacing offline voters: {', '.join(offline_voters)}"
            )

        # Suggest online nodes that could be voters
        online_non_voters = self._find_online_non_voters(
            voters, hosts, tailscale_peers
        )
        if online_non_voters and len(offline_voters) > 0:
            result.suggestions.append(
                f"Consider adding as voters: {', '.join(online_non_voters[:3])}"
            )

        return result

    def _get_tailscale_status(self) -> dict[str, dict]:
        """Get Tailscale peer status.

        Returns:
            Dict mapping IP to peer info (online, hostname, last_seen)
        """
        try:
            # Run tailscale status --json
            result = subprocess.run(
                ["tailscale", "status", "--json"],
                capture_output=True,
                text=True,
                timeout=30,
            )

            if result.returncode != 0:
                logger.warning(f"tailscale status failed: {result.stderr}")
                return {}

            status = json.loads(result.stdout)
            peers: dict[str, dict] = {}

            # Add self
            if status.get("Self"):
                self_info = status["Self"]
                for ip in self_info.get("TailscaleIPs", []):
                    peers[ip] = {
                        "online": True,
                        "hostname": self_info.get("HostName", ""),
                        "last_seen": "now",
                    }

            # Add peers
            for peer in status.get("Peer", {}).values():
                for ip in peer.get("TailscaleIPs", []):
                    peers[ip] = {
                        "online": peer.get("Online", False),
                        "hostname": peer.get("HostName", ""),
                        "last_seen": peer.get("LastSeen", ""),
                    }

            return peers

        except subprocess.TimeoutExpired:
            logger.warning("tailscale status timed out")
            return {}
        except FileNotFoundError:
            logger.warning("tailscale command not found")
            return {}
        except Exception as e:
            logger.warning(f"Failed to get Tailscale status: {e}")
            return {}

    def _find_online_non_voters(
        self,
        voters: list[str],
        hosts: dict[str, Any],
        tailscale_peers: dict[str, dict],
    ) -> list[str]:
        """Find hosts that are online but not currently voters.

        Args:
            voters: Current voter list
            hosts: All hosts from config
            tailscale_peers: Tailscale peer status

        Returns:
            List of node_ids that are online but not voters
        """
        online_non_voters = []

        for node_id, host_cfg in hosts.items():
            if node_id in voters:
                continue

            tailscale_ip = host_cfg.get("tailscale_ip")
            if not tailscale_ip:
                continue

            peer_info = tailscale_peers.get(tailscale_ip)
            if peer_info and peer_info.get("online"):
                online_non_voters.append(node_id)

        return online_non_voters

    def print_report(self, result: ValidationResult) -> None:
        """Print validation report to stdout."""
        print("\n" + "=" * 60)
        print("P2P Voter Validation Report")
        print("=" * 60)

        # Summary
        status = "PASS" if result.is_valid else "FAIL"
        print(f"\nStatus: {status}")
        print(f"Voters Online: {result.online_count}/{result.total_count}")
        print(f"Quorum Required: {result.quorum_size}")
        print(f"Has Quorum: {'Yes' if result.has_quorum else 'NO'}")

        # Voter details
        print("\nVoter Details:")
        print("-" * 40)
        for voter_id, info in result.voters.items():
            status_icon = "OK" if info.is_online else "OFFLINE"
            if info.error:
                status_icon = "ERROR"
            print(f"  {voter_id}: [{status_icon}]")
            if info.tailscale_ip:
                print(f"    IP: {info.tailscale_ip}")
            if info.hostname:
                print(f"    Hostname: {info.hostname}")
            if info.error:
                print(f"    Error: {info.error}")

        # Errors
        if result.errors:
            print("\nErrors:")
            print("-" * 40)
            for error in result.errors:
                print(f"  - {error}")

        # Suggestions
        if result.suggestions:
            print("\nSuggestions:")
            print("-" * 40)
            for suggestion in result.suggestions:
                print(f"  - {suggestion}")

        print("\n" + "=" * 60)


def validate_voters_cli(config_path: Path | None = None) -> int:
    """CLI entry point for voter validation.

    Args:
        config_path: Optional path to config file

    Returns:
        Exit code (0 = success, 1 = validation failed)
    """
    validator = VoterValidator(config_path=config_path)
    result = validator.validate()
    validator.print_report(result)

    return 0 if result.is_valid else 1


if __name__ == "__main__":
    import sys

    sys.exit(validate_voters_cli())
