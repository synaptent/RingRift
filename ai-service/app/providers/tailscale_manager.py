"""Tailscale network manager.

Manages Tailscale mesh network connectivity across all providers.
Tailscale is the backbone for reliable cross-provider communication.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from app.providers.base import HealthCheckResult

logger = logging.getLogger(__name__)


@dataclass
class TailscalePeer:
    """Tailscale peer status."""

    hostname: str
    tailscale_ip: str
    public_ip: str | None = None
    os: str | None = None
    online: bool = False
    last_seen: datetime | None = None
    exit_node: bool = False
    relay: str | None = None  # DERP relay if direct connection failed


@dataclass
class TailscaleStatus:
    """Overall Tailscale status."""

    self_hostname: str
    self_ip: str
    online: bool = True
    peers: list[TailscalePeer] | None = None
    backend_state: str = "unknown"
    derp_region: str | None = None


class TailscaleManager:
    """Manage Tailscale network connectivity.

    Provides utilities for checking Tailscale status locally and
    on remote hosts, as well as restart/reauth capabilities.

    Usage:
        manager = TailscaleManager()

        # Local status
        status = await manager.get_local_status()
        print(f"Self: {status.self_hostname} ({status.self_ip})")

        # Remote status
        peer = await manager.get_peer_status("gpu-node-1")
        if peer and peer.online:
            print(f"Peer online: {peer.tailscale_ip}")
    """

    async def get_local_status(self) -> TailscaleStatus | None:
        """Get local Tailscale status."""
        try:
            proc = await asyncio.create_subprocess_exec(
                "tailscale", "status", "--json",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=15)

            if proc.returncode != 0:
                logger.warning(f"tailscale status failed: {stderr.decode()}")
                return None

            data = json.loads(stdout.decode())
            return self._parse_status(data)
        except FileNotFoundError:
            logger.warning("tailscale CLI not found")
            return None
        except asyncio.TimeoutError:
            logger.warning("tailscale status timed out")
            return None
        except Exception as e:
            logger.error(f"tailscale status error: {e}")
            return None

    def _parse_status(self, data: dict[str, Any]) -> TailscaleStatus:
        """Parse tailscale status JSON."""
        self_data = data.get("Self", {})

        peers = []
        for peer_key, peer_data in data.get("Peer", {}).items():
            try:
                # Parse last seen timestamp
                last_seen = None
                if ls := peer_data.get("LastSeen"):
                    try:
                        last_seen = datetime.fromisoformat(ls.replace("Z", "+00:00"))
                    except Exception:
                        pass

                peers.append(TailscalePeer(
                    hostname=peer_data.get("HostName", peer_key),
                    tailscale_ip=peer_data.get("TailscaleIPs", [""])[0],
                    public_ip=peer_data.get("CurAddr", "").split(":")[0] or None,
                    os=peer_data.get("OS"),
                    online=peer_data.get("Online", False),
                    last_seen=last_seen,
                    exit_node=peer_data.get("ExitNode", False),
                    relay=peer_data.get("Relay"),
                ))
            except Exception as e:
                logger.debug(f"Failed to parse peer {peer_key}: {e}")

        return TailscaleStatus(
            self_hostname=self_data.get("HostName", "unknown"),
            self_ip=self_data.get("TailscaleIPs", [""])[0],
            online=True,  # If we got status, we're online
            peers=peers,
            backend_state=data.get("BackendState", "unknown"),
            derp_region=data.get("CurrentTailnet", {}).get("CurrentDERP"),
        )

    async def get_peer_status(self, hostname: str) -> TailscalePeer | None:
        """Get status of a specific peer by hostname."""
        status = await self.get_local_status()
        if not status or not status.peers:
            return None

        # Search by hostname (case-insensitive, partial match)
        hostname_lower = hostname.lower()
        for peer in status.peers:
            if hostname_lower in peer.hostname.lower():
                return peer

        return None

    async def get_peer_ip(self, hostname: str) -> str | None:
        """Get Tailscale IP of a peer."""
        peer = await self.get_peer_status(hostname)
        return peer.tailscale_ip if peer else None

    async def is_peer_online(self, hostname: str) -> bool:
        """Check if a peer is online in Tailscale."""
        peer = await self.get_peer_status(hostname)
        return peer.online if peer else False

    async def ping_peer(self, tailscale_ip: str, timeout: int = 5) -> bool:
        """Ping a Tailscale peer."""
        try:
            proc = await asyncio.create_subprocess_exec(
                "tailscale", "ping", "--c", "1", "--timeout", f"{timeout}s", tailscale_ip,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout + 5)
            return proc.returncode == 0
        except Exception:
            return False

    async def get_remote_status(
        self,
        ssh_host: str,
        ssh_user: str = "ubuntu",
        ssh_key: str = "~/.ssh/id_cluster",
        ssh_port: int = 22,
    ) -> TailscaleStatus | None:
        """Get Tailscale status from a remote host via SSH."""
        from pathlib import Path

        ssh_cmd = [
            "ssh",
            "-o", "ConnectTimeout=10",
            "-o", "StrictHostKeyChecking=accept-new",
            "-o", "BatchMode=yes",
            "-i", str(Path(ssh_key).expanduser()),
            "-p", str(ssh_port),
            f"{ssh_user}@{ssh_host}",
            "tailscale status --json",
        ]

        try:
            proc = await asyncio.create_subprocess_exec(
                *ssh_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)

            if proc.returncode != 0:
                logger.debug(f"Remote tailscale status failed: {stderr.decode()}")
                return None

            data = json.loads(stdout.decode())
            return self._parse_status(data)
        except Exception as e:
            logger.debug(f"Remote tailscale status error: {e}")
            return None

    async def get_remote_ip(
        self,
        ssh_host: str,
        ssh_user: str = "ubuntu",
        ssh_key: str = "~/.ssh/id_cluster",
        ssh_port: int = 22,
    ) -> str | None:
        """Get Tailscale IP of a remote host via SSH."""
        from pathlib import Path

        ssh_cmd = [
            "ssh",
            "-o", "ConnectTimeout=10",
            "-o", "StrictHostKeyChecking=accept-new",
            "-o", "BatchMode=yes",
            "-i", str(Path(ssh_key).expanduser()),
            "-p", str(ssh_port),
            f"{ssh_user}@{ssh_host}",
            "tailscale ip -4 2>/dev/null",
        ]

        try:
            proc = await asyncio.create_subprocess_exec(
                *ssh_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=20)

            if proc.returncode == 0:
                ip = stdout.decode().strip().split("\n")[0]
                if ip.startswith("100."):
                    return ip
            return None
        except Exception:
            return None

    async def restart_remote_tailscale(
        self,
        ssh_host: str,
        ssh_user: str = "ubuntu",
        ssh_key: str = "~/.ssh/id_cluster",
        ssh_port: int = 22,
        use_sudo: bool = True,
    ) -> bool:
        """Restart Tailscale on a remote host."""
        from pathlib import Path

        sudo = "sudo " if use_sudo else ""
        cmd = f"{sudo}systemctl restart tailscaled && sleep 3 && tailscale status"

        ssh_cmd = [
            "ssh",
            "-o", "ConnectTimeout=10",
            "-o", "StrictHostKeyChecking=accept-new",
            "-o", "BatchMode=yes",
            "-i", str(Path(ssh_key).expanduser()),
            "-p", str(ssh_port),
            f"{ssh_user}@{ssh_host}",
            cmd,
        ]

        try:
            proc = await asyncio.create_subprocess_exec(
                *ssh_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=45)

            if proc.returncode == 0:
                logger.info(f"Tailscale restarted on {ssh_host}")
                return True
            else:
                logger.error(f"Tailscale restart failed on {ssh_host}: {stderr.decode()}")
                return False
        except Exception as e:
            logger.error(f"Tailscale restart error on {ssh_host}: {e}")
            return False

    async def force_reauth_remote(
        self,
        ssh_host: str,
        auth_key: str,
        ssh_user: str = "ubuntu",
        ssh_key: str = "~/.ssh/id_cluster",
        ssh_port: int = 22,
        use_sudo: bool = True,
    ) -> bool:
        """Force re-authentication of Tailscale on a remote host."""
        from pathlib import Path

        sudo = "sudo " if use_sudo else ""
        cmd = f"{sudo}tailscale up --authkey {auth_key} --accept-routes"

        ssh_cmd = [
            "ssh",
            "-o", "ConnectTimeout=10",
            "-o", "StrictHostKeyChecking=accept-new",
            "-o", "BatchMode=yes",
            "-i", str(Path(ssh_key).expanduser()),
            "-p", str(ssh_port),
            f"{ssh_user}@{ssh_host}",
            cmd,
        ]

        try:
            proc = await asyncio.create_subprocess_exec(
                *ssh_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=60)

            if proc.returncode == 0:
                logger.info(f"Tailscale re-authenticated on {ssh_host}")
                return True
            else:
                logger.error(f"Tailscale reauth failed on {ssh_host}: {stderr.decode()}")
                return False
        except Exception as e:
            logger.error(f"Tailscale reauth error on {ssh_host}: {e}")
            return False

    async def check_health(
        self,
        tailscale_ip: str,
        hostname: str | None = None,
    ) -> HealthCheckResult:
        """Check Tailscale connectivity to a peer."""
        import time

        start = time.time()

        # First check if peer is in our network
        peer = await self.get_peer_status(hostname or tailscale_ip)
        latency = (time.time() - start) * 1000

        if peer:
            if peer.online:
                # Try to ping
                ping_ok = await self.ping_peer(tailscale_ip)
                return HealthCheckResult(
                    healthy=ping_ok,
                    check_type="tailscale",
                    message="Tailscale peer reachable" if ping_ok else "Tailscale peer offline",
                    latency_ms=latency,
                    details={
                        "peer_hostname": peer.hostname,
                        "online": peer.online,
                        "relay": peer.relay,
                    },
                )
            else:
                return HealthCheckResult(
                    healthy=False,
                    check_type="tailscale",
                    message=f"Tailscale peer offline (last seen: {peer.last_seen})",
                    latency_ms=latency,
                    details={"peer_hostname": peer.hostname},
                )

        return HealthCheckResult(
            healthy=False,
            check_type="tailscale",
            message="Tailscale peer not found in network",
            latency_ms=latency,
        )

    async def get_all_online_peers(self) -> list[TailscalePeer]:
        """Get list of all online Tailscale peers."""
        status = await self.get_local_status()
        if not status or not status.peers:
            return []
        return [p for p in status.peers if p.online]

    async def get_offline_peers(self) -> list[TailscalePeer]:
        """Get list of offline Tailscale peers."""
        status = await self.get_local_status()
        if not status or not status.peers:
            return []
        return [p for p in status.peers if not p.online]


async def test_tailscale():
    """Test Tailscale connectivity."""
    manager = TailscaleManager()

    print("Testing Tailscale...")
    status = await manager.get_local_status()

    if status:
        print(f"\nSelf: {status.self_hostname} ({status.self_ip})")
        print(f"Backend state: {status.backend_state}")

        if status.peers:
            online = [p for p in status.peers if p.online]
            offline = [p for p in status.peers if not p.online]

            print(f"\nOnline peers ({len(online)}):")
            for peer in online[:10]:
                relay = f" (via {peer.relay})" if peer.relay else ""
                print(f"  {peer.hostname}: {peer.tailscale_ip}{relay}")

            if offline:
                print(f"\nOffline peers ({len(offline)}):")
                for peer in offline[:5]:
                    print(f"  {peer.hostname}: {peer.tailscale_ip}")
    else:
        print("Tailscale not available or not configured")


if __name__ == "__main__":
    asyncio.run(test_tailscale())
