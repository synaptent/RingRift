"""Serf RPC client for Python.

This module provides a Python client for HashiCorp Serf's RPC interface.
Serf uses msgpack-based RPC over TCP for communication.

Usage:
    client = SerfClient()
    members = await client.members()
    await client.event("deploy", b"version=1.2.3")

Serf RPC Protocol:
- Connect to RPC port (default 7373)
- Send msgpack-encoded commands
- Receive msgpack-encoded responses
- Each command has a sequence number for request/response matching

References:
- https://www.serf.io/docs/agent/rpc.html
- https://github.com/hashicorp/serf
"""

import asyncio
import logging
import struct
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

try:
    import msgpack
except ImportError:
    msgpack = None  # type: ignore

logger = logging.getLogger(__name__)


class SerfError(Exception):
    """Base exception for Serf client errors."""
    pass


class SerfConnectionError(SerfError):
    """Failed to connect to Serf agent."""
    pass


class SerfRPCError(SerfError):
    """Serf RPC command failed."""
    pass


class MemberStatus(Enum):
    """Serf member status."""
    NONE = 0
    ALIVE = 1
    LEAVING = 2
    LEFT = 3
    FAILED = 4


@dataclass
class SerfMember:
    """Represents a Serf cluster member."""
    name: str
    addr: str
    port: int
    tags: dict[str, str] = field(default_factory=dict)
    status: MemberStatus = MemberStatus.NONE
    protocol_min: int = 0
    protocol_max: int = 0
    protocol_cur: int = 0
    delegate_min: int = 0
    delegate_max: int = 0
    delegate_cur: int = 0

    @classmethod
    def from_dict(cls, data: dict) -> "SerfMember":
        """Create SerfMember from RPC response dict."""
        return cls(
            name=data.get("Name", ""),
            addr=data.get("Addr", ""),
            port=data.get("Port", 0),
            tags=data.get("Tags", {}),
            status=MemberStatus(data.get("Status", 0)),
            protocol_min=data.get("ProtocolMin", 0),
            protocol_max=data.get("ProtocolMax", 0),
            protocol_cur=data.get("ProtocolCur", 0),
            delegate_min=data.get("DelegateMin", 0),
            delegate_max=data.get("DelegateMax", 0),
            delegate_cur=data.get("DelegateCur", 0),
        )

    def is_alive(self) -> bool:
        """Check if member is alive."""
        return self.status == MemberStatus.ALIVE


@dataclass
class SerfCoordinate:
    """Network coordinate for latency estimation."""
    adjustment: float = 0.0
    error: float = 0.0
    height: float = 0.0
    vec: list[float] = field(default_factory=list)


class SerfClient:
    """Async client for Serf RPC interface.

    Provides methods to:
    - Query cluster membership
    - Send events to the cluster
    - Join/leave the cluster
    - Query/respond for request-response patterns

    IMPORTANT: Always use this client as an async context manager to ensure
    proper socket cleanup:

        async with SerfClient() as client:
            members = await client.members()
            for m in members:
                print(f"{m.name}: {m.status.name}")

    If not using context manager, ensure close() is called in a finally block.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 7373,
        auth_key: str | None = None,
        timeout: float = 5.0,
    ):
        """Initialize Serf client.

        Args:
            host: Serf agent RPC host
            port: Serf agent RPC port (default 7373)
            auth_key: Optional authentication key
            timeout: RPC timeout in seconds
        """
        if msgpack is None:
            raise ImportError("msgpack is required for SerfClient. Install with: pip install msgpack")

        self.host = host
        self.port = port
        self.auth_key = auth_key
        self.timeout = timeout

        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None
        self._seq = 0
        self._lock = asyncio.Lock()
        self._connected = False
        self._closed = False  # Track if close() was called to prevent double-close

        # Event handlers for streaming responses
        self._event_handlers: dict[str, list[Callable]] = {}

    async def __aenter__(self) -> "SerfClient":
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        # Always close on exit, even on exceptions
        # close() is idempotent so safe to call multiple times
        await self.close()

    async def connect(self) -> None:
        """Connect to Serf agent.

        Can be called to reconnect after close() was called.
        """
        if self._connected:
            return

        try:
            # Reset closed flag to allow reconnection
            self._closed = False

            self._reader, self._writer = await asyncio.wait_for(
                asyncio.open_connection(self.host, self.port),
                timeout=self.timeout,
            )
            self._connected = True
            logger.info(f"Connected to Serf agent at {self.host}:{self.port}")

            # Authenticate if key provided
            if self.auth_key:
                await self._handshake()
                await self._auth()

        except asyncio.TimeoutError as e:
            self._closed = True  # Mark as closed on failure
            raise SerfConnectionError(f"Timeout connecting to Serf at {self.host}:{self.port}") from e
        except OSError as e:
            self._closed = True  # Mark as closed on failure
            raise SerfConnectionError(f"Failed to connect to Serf at {self.host}:{self.port}: {e}") from e

    async def close(self) -> None:
        """Close connection to Serf agent.

        This method is idempotent and safe to call multiple times.
        Ensures all socket resources are properly released.
        """
        # Guard against double-close
        if self._closed:
            return
        self._closed = True
        self._connected = False

        # Close writer (which also closes the underlying socket)
        writer = self._writer
        self._writer = None
        self._reader = None

        if writer is not None:
            try:
                # Check if transport is still open before closing
                if not writer.is_closing():
                    writer.close()
                # Wait for clean shutdown with timeout
                try:
                    await asyncio.wait_for(writer.wait_closed(), timeout=2.0)
                except asyncio.TimeoutError:
                    logger.warning(f"Timeout closing Serf connection to {self.host}:{self.port}")
            except (OSError, asyncio.CancelledError, BrokenPipeError, ConnectionResetError) as e:
                # Socket may already be closed - this is fine
                logger.debug(f"Exception during Serf client close (may be expected): {e}")

    async def _handshake(self) -> dict:
        """Perform RPC handshake."""
        return await self._rpc("handshake", {"Version": 1})

    async def _auth(self) -> dict:
        """Authenticate with Serf agent."""
        if not self.auth_key:
            return {}
        return await self._rpc("auth", {"AuthKey": self.auth_key})

    def _next_seq(self) -> int:
        """Get next sequence number."""
        self._seq += 1
        return self._seq

    async def _rpc(self, command: str, body: dict | None = None) -> dict:
        """Send RPC command and receive response.

        Args:
            command: RPC command name
            body: Command body (optional)

        Returns:
            Response dict
        """
        if not self._connected:
            await self.connect()

        async with self._lock:
            seq = self._next_seq()

            # Build header
            header = {
                "Command": command,
                "Seq": seq,
            }

            # Encode and send
            header_bytes = msgpack.packb(header)
            if self._writer is None:
                raise SerfConnectionError("Not connected")

            self._writer.write(header_bytes)
            if body:
                body_bytes = msgpack.packb(body)
                self._writer.write(body_bytes)
            await self._writer.drain()

            # Read response header
            if self._reader is None:
                raise SerfConnectionError("Not connected")

            unpacker = msgpack.Unpacker(raw=False)

            # Read response - Serf sends header + body
            resp_header = None
            resp_body = None

            try:
                while resp_header is None or resp_body is None:
                    data = await asyncio.wait_for(
                        self._reader.read(4096),
                        timeout=self.timeout,
                    )
                    if not data:
                        raise SerfConnectionError("Connection closed by Serf agent")

                    unpacker.feed(data)
                    for obj in unpacker:
                        if resp_header is None:
                            resp_header = obj
                        else:
                            resp_body = obj
                            break

            except asyncio.TimeoutError as e:
                raise SerfRPCError(f"Timeout waiting for response to {command}") from e

            # Check for errors
            if resp_header.get("Error"):
                raise SerfRPCError(f"Serf RPC error: {resp_header['Error']}")

            return resp_body or {}

    # ========== Membership Commands ==========

    async def members(self, filters: dict | None = None) -> list[SerfMember]:
        """Get list of cluster members.

        Args:
            filters: Optional filters (Status, Name, Tags)

        Returns:
            List of SerfMember objects
        """
        body = filters or {}
        resp = await self._rpc("members", body)
        members_data = resp.get("Members", [])
        return [SerfMember.from_dict(m) for m in members_data]

    async def members_filtered(
        self,
        status: str | None = None,
        name: str | None = None,
        tags: dict[str, str] | None = None,
    ) -> list[SerfMember]:
        """Get filtered list of cluster members.

        Args:
            status: Filter by status (alive, leaving, left, failed)
            name: Filter by name regex
            tags: Filter by tags

        Returns:
            List of matching SerfMember objects
        """
        filters: dict[str, Any] = {}
        if status:
            filters["Status"] = status
        if name:
            filters["Name"] = name
        if tags:
            filters["Tags"] = tags
        return await self.members(filters)

    async def alive_members(self) -> list[SerfMember]:
        """Get list of alive members only."""
        members = await self.members()
        return [m for m in members if m.is_alive()]

    # ========== Event Commands ==========

    async def event(
        self,
        name: str,
        payload: bytes | str = b"",
        coalesce: bool = True,
    ) -> None:
        """Send a user event to the cluster.

        Args:
            name: Event name
            payload: Event payload (bytes or string)
            coalesce: Whether to coalesce events with same name
        """
        if isinstance(payload, str):
            payload = payload.encode("utf-8")

        await self._rpc("event", {
            "Name": name,
            "Payload": payload,
            "Coalesce": coalesce,
        })
        logger.debug(f"Sent event: {name}")

    # ========== Join/Leave Commands ==========

    async def join(self, addresses: list[str], replay: bool = False) -> int:
        """Join a Serf cluster.

        Args:
            addresses: List of addresses to join (host:port)
            replay: Whether to replay old user events

        Returns:
            Number of nodes successfully joined
        """
        resp = await self._rpc("join", {
            "Existing": addresses,
            "Replay": replay,
        })
        return resp.get("Num", 0)

    async def leave(self) -> None:
        """Leave the Serf cluster gracefully."""
        await self._rpc("leave", {})

    async def force_leave(self, node: str) -> None:
        """Force a node to leave the cluster.

        Args:
            node: Node name to force leave
        """
        await self._rpc("force-leave", {"Node": node})

    # ========== Query Commands ==========

    async def query(
        self,
        name: str,
        payload: bytes | str = b"",
        filter_nodes: list[str] | None = None,
        filter_tags: dict[str, str] | None = None,
        request_ack: bool = False,
        timeout: float = 15.0,
    ) -> list[dict]:
        """Send a query to the cluster and collect responses.

        Args:
            name: Query name
            payload: Query payload
            filter_nodes: Only send to specific nodes
            filter_tags: Only send to nodes with matching tags
            request_ack: Request acknowledgment from nodes
            timeout: Query timeout in seconds

        Returns:
            List of response dicts with 'From' and 'Payload'
        """
        if isinstance(payload, str):
            payload = payload.encode("utf-8")

        body: dict[str, Any] = {
            "Name": name,
            "Payload": payload,
            "RequestAck": request_ack,
            "Timeout": int(timeout * 1_000_000_000),  # nanoseconds
        }
        if filter_nodes:
            body["FilterNodes"] = filter_nodes
        if filter_tags:
            body["FilterTags"] = filter_tags

        # Query returns streaming responses
        # For now, just return the initial response
        resp = await self._rpc("query", body)
        return resp.get("Responses", [])

    # ========== Stats Commands ==========

    async def stats(self) -> dict:
        """Get Serf agent statistics.

        Returns:
            Dict with agent, runtime, serf stats
        """
        return await self._rpc("stats", {})

    async def get_coordinate(self, node: str) -> SerfCoordinate | None:
        """Get network coordinate for a node.

        Args:
            node: Node name

        Returns:
            SerfCoordinate or None if not found
        """
        resp = await self._rpc("get-coordinate", {"Node": node})
        coord = resp.get("Coord")
        if not coord:
            return None
        return SerfCoordinate(
            adjustment=coord.get("Adjustment", 0.0),
            error=coord.get("Error", 0.0),
            height=coord.get("Height", 0.0),
            vec=coord.get("Vec", []),
        )

    # ========== Tag Commands ==========

    async def tags(self) -> dict[str, str]:
        """Get current node's tags.

        Returns:
            Dict of tag name to value
        """
        resp = await self._rpc("tags", {})
        return resp.get("Tags", {})

    async def set_tags(self, tags: dict[str, str], delete_tags: list[str] | None = None) -> None:
        """Update node's tags.

        Args:
            tags: Tags to set/update
            delete_tags: Tag names to delete
        """
        await self._rpc("tags", {
            "Tags": tags,
            "DeleteTags": delete_tags or [],
        })

    # ========== Utility Methods ==========

    def is_connected(self) -> bool:
        """Check if connected to Serf agent."""
        return self._connected

    async def ping(self) -> bool:
        """Check if Serf agent is responsive.

        Returns:
            True if agent responds to stats command
        """
        try:
            await self.stats()
            return True
        except SerfError:
            return False


class SerfMembershipMonitor:
    """Monitor Serf membership changes and provide callbacks.

    This class polls Serf for membership changes and calls registered
    handlers when members join, leave, or fail.

    Supports async context manager for guaranteed cleanup:

        async with SerfMembershipMonitor(client) as monitor:
            monitor.on_join(lambda m: print(f"Joined: {m.name}"))
            monitor.on_leave(lambda m: print(f"Left: {m.name}"))
            await asyncio.sleep(60)  # Monitor for 60 seconds
        # Monitor is automatically stopped

    Or traditional usage:
        monitor = SerfMembershipMonitor(client)
        monitor.on_join(lambda m: print(f"Joined: {m.name}"))
        await monitor.start()
        try:
            # ... do work ...
        finally:
            await monitor.stop()
    """

    def __init__(
        self,
        client: SerfClient,
        poll_interval: float = 5.0,
    ):
        """Initialize membership monitor.

        Args:
            client: SerfClient instance
            poll_interval: How often to poll for changes (seconds)
        """
        self.client = client
        self.poll_interval = poll_interval

        self._members: dict[str, SerfMember] = {}
        self._running = False
        self._task: asyncio.Task | None = None

        self._on_join: list[Callable[[SerfMember], None]] = []
        self._on_leave: list[Callable[[SerfMember], None]] = []
        self._on_fail: list[Callable[[SerfMember], None]] = []
        self._on_update: list[Callable[[SerfMember], None]] = []

    async def __aenter__(self) -> "SerfMembershipMonitor":
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.stop()

    def on_join(self, handler: Callable[[SerfMember], None]) -> None:
        """Register handler for member join events."""
        self._on_join.append(handler)

    def on_leave(self, handler: Callable[[SerfMember], None]) -> None:
        """Register handler for member leave events."""
        self._on_leave.append(handler)

    def on_fail(self, handler: Callable[[SerfMember], None]) -> None:
        """Register handler for member failure events."""
        self._on_fail.append(handler)

    def on_update(self, handler: Callable[[SerfMember], None]) -> None:
        """Register handler for any membership update."""
        self._on_update.append(handler)

    async def start(self) -> None:
        """Start monitoring membership changes."""
        if self._running:
            return

        self._running = True

        # Initial member fetch
        members = await self.client.members()
        for m in members:
            self._members[m.name] = m

        self._task = asyncio.create_task(self._poll_loop())
        logger.info(f"Started Serf membership monitor with {len(self._members)} members")

    async def stop(self) -> None:
        """Stop monitoring.

        Idempotent - safe to call multiple times.
        Uses timeout to prevent blocking indefinitely on stuck poll tasks.
        """
        self._running = False
        task = self._task
        self._task = None

        if task is not None and not task.done():
            task.cancel()
            try:
                # Wait with timeout - poll task should cancel quickly
                await asyncio.wait_for(task, timeout=5.0)
            except asyncio.TimeoutError:
                logger.warning("Timeout stopping SerfMembershipMonitor poll task")
            except asyncio.CancelledError:
                pass

    async def _poll_loop(self) -> None:
        """Poll for membership changes."""
        while self._running:
            try:
                await asyncio.sleep(self.poll_interval)
                await self._check_membership()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error polling Serf membership: {e}")

    async def _check_membership(self) -> None:
        """Check for membership changes."""
        try:
            current = await self.client.members()
        except SerfError as e:
            logger.warning(f"Failed to get Serf members: {e}")
            return

        current_by_name = {m.name: m for m in current}

        # Check for new/updated members
        for name, member in current_by_name.items():
            old = self._members.get(name)

            if old is None:
                # New member
                if member.status == MemberStatus.ALIVE:
                    for handler in self._on_join:
                        try:
                            handler(member)
                        except Exception as e:
                            logger.error(f"Error in join handler: {e}")

            elif old.status != member.status:
                # Status changed
                if member.status == MemberStatus.LEFT:
                    for handler in self._on_leave:
                        try:
                            handler(member)
                        except Exception as e:
                            logger.error(f"Error in leave handler: {e}")
                elif member.status == MemberStatus.FAILED:
                    for handler in self._on_fail:
                        try:
                            handler(member)
                        except Exception as e:
                            logger.error(f"Error in fail handler: {e}")

            # Always call update handlers
            for handler in self._on_update:
                try:
                    handler(member)
                except Exception as e:
                    logger.error(f"Error in update handler: {e}")

        # Check for removed members (shouldn't happen with Serf, but handle it)
        for name in list(self._members.keys()):
            if name not in current_by_name:
                old = self._members[name]
                for handler in self._on_leave:
                    try:
                        handler(old)
                    except Exception as e:
                        logger.error(f"Error in leave handler: {e}")

        self._members = current_by_name

    @property
    def members(self) -> dict[str, SerfMember]:
        """Get current member snapshot."""
        return self._members.copy()

    def get_alive_members(self) -> list[SerfMember]:
        """Get list of alive members."""
        return [m for m in self._members.values() if m.is_alive()]


# ========== Convenience Functions ==========

async def get_serf_members(
    host: str = "127.0.0.1",
    port: int = 7373,
    timeout: float = 5.0,
) -> list[SerfMember]:
    """Quick helper to get Serf cluster members.

    Args:
        host: Serf agent host
        port: Serf agent RPC port
        timeout: Connection timeout

    Returns:
        List of SerfMember objects
    """
    async with SerfClient(host, port, timeout=timeout) as client:
        return await client.members()


async def send_serf_event(
    name: str,
    payload: bytes | str = b"",
    host: str = "127.0.0.1",
    port: int = 7373,
) -> None:
    """Quick helper to send a Serf event.

    Args:
        name: Event name
        payload: Event payload
        host: Serf agent host
        port: Serf agent RPC port
    """
    async with SerfClient(host, port) as client:
        await client.event(name, payload)
