"""HTTP client session, auth header, leader lookup, and leader proxy helpers.

April 2026: Extracted from p2p_orchestrator.py (Part 3 Phase 2).
"""
from __future__ import annotations

from scripts.p2p.p2p_mixin_base import P2PMixinBase
from scripts.p2p.startup_infrastructure import *  # noqa: F401,F403


class HttpSessionMixin(P2PMixinBase):
    """Mixin for P2POrchestrator http client session, auth header, leader lookup, and leader proxy helpers."""

    MIXIN_TYPE = "http_session"

    def _auth_headers(self) -> dict[str, str]:
        if not self.auth_token:
            return {}
        return {"Authorization": f"Bearer {self.auth_token}"}

    @property
    def http_session(self) -> "aiohttp.ClientSession":
        """Shared HTTP client session for outbound requests.

        Used by loop_registry (manifest collection, peer recovery probes).
        Lazily created and re-created if closed.
        """
        if not hasattr(self, "_http_session") or self._http_session is None or self._http_session.closed:
            import time as _time

            timeout = aiohttp.ClientTimeout(total=30)
            self._http_session = aiohttp.ClientSession(
                timeout=timeout,
                headers=self._auth_headers(),
            )
            self._http_session_created_at = _time.time()
        return self._http_session

    @property
    def http_session_created_at(self) -> float:
        """Timestamp when the current HTTP session was created."""
        return getattr(self, "_http_session_created_at", 0.0)

    async def recreate_http_session(self) -> None:
        """Close the existing HTTP session and create a fresh one.

        March 2026: Called by HttpPoolMonitorLoop to prevent TIME_WAIT socket
        exhaustion during 7-day autonomous operation. After closing the old
        session, the next access to self.http_session will lazily create a new
        one via the property getter.
        """
        import time as _time

        old_session = getattr(self, "_http_session", None)
        if old_session is not None and not old_session.closed:
            try:
                await old_session.close()
                # Allow FIN/ACK handshake to complete
                await asyncio.sleep(0.25)
            except Exception as e:
                logger.debug(f"[P2P] Error closing old HTTP session: {e}")

        # Reset so the property creates a fresh session on next access
        self._http_session = None
        self._http_session_created_at = 0.0

        # Eagerly create the new session so callers don't hit a race
        _ = self.http_session

        logger.info(
            f"[P2P] HTTP session recreated at {_time.time():.0f}"
        )

    def _get_leader_peer(self) -> NodeInfo | None:
        if self.leadership.check_is_leader():
            return self.self_info

        # Jan 2026: Use lock-free PeerSnapshot for read-only access
        peers_snapshot = list(self._peer_snapshot.get_snapshot().values())

        conflict_keys = self._endpoint_conflict_keys([self.self_info, *peers_snapshot])

        leader_id = self.leader_id
        if leader_id and self._is_leader_lease_valid():
            for peer in peers_snapshot:
                if (
                    peer.node_id == leader_id
                    and peer.role == NodeRole.LEADER
                    and peer.is_alive()
                    and self._is_leader_eligible(peer, conflict_keys)
                ):
                    # Jan 8, 2026: Validate consensus - check that other peers agree
                    consensus_count = self.leadership.count_peers_reporting_leader(leader_id, peers_snapshot)
                    if consensus_count < 2 and len(peers_snapshot) >= 3:
                        # Low consensus - log warning but still return leader
                        logger.warning(
                            f"[LeaderConsensus] Low consensus for leader {leader_id}: "
                            f"only {consensus_count} peers agree out of {len(peers_snapshot)}"
                        )
                    return peer

        eligible_leaders = [
            peer for peer in peers_snapshot
            if peer.role == NodeRole.LEADER and self._is_leader_eligible(peer, conflict_keys)
        ]
        if eligible_leaders:
            return sorted(eligible_leaders, key=lambda p: p.node_id)[-1]

        return None

    async def _proxy_to_leader(self, request: web.Request) -> web.StreamResponse:
        """Best-effort proxy for leader-only APIs when the dashboard hits a follower."""
        leader = self._get_leader_peer()
        if not leader:
            return web.json_response(
                {"success": False, "error": "leader_unknown", "leader_id": self.leader_id},
                status=503,
            )

        candidate_urls = self._urls_for_peer(leader, request.raw_path)
        if not candidate_urls:
            candidate_urls = [self._url_for_peer(leader, request.raw_path)]
        forward_headers: dict[str, str] = {}
        for h in ("Authorization", "X-RingRift-Auth", "Content-Type"):
            if h in request.headers:
                forward_headers[h] = request.headers[h]

        body: bytes | None = None
        if request.method not in ("GET", "HEAD", "OPTIONS"):
            body = await request.read()

        # Keep leader-proxy responsive: unreachable "leaders" (often NAT/firewall)
        # should fail fast so the dashboard doesn't hang for a full minute.
        timeout = ClientTimeout(total=10)
        last_exc: Exception | None = None
        async with get_client_session(timeout) as session:
            for target_url in candidate_urls:
                try:
                    async with session.request(
                        request.method,
                        target_url,
                        data=body,
                        headers=forward_headers,
                    ) as resp:
                        payload = await resp.read()
                        content_type = resp.headers.get("Content-Type")
                        headers: dict[str, str] = {}
                        if content_type:
                            headers["Content-Type"] = content_type
                        headers["X-RingRift-Proxied-By"] = self.node_id
                        headers["X-RingRift-Proxied-To"] = target_url
                        return web.Response(body=payload, status=resp.status, headers=headers)
                except Exception as exc:
                    last_exc = exc
                    continue

        return web.json_response(
            {
                "success": False,
                "error": "leader_proxy_failed",
                "message": str(last_exc) if last_exc else "unknown_error",
                "leader_id": self.leader_id,
                "attempted_urls": candidate_urls,
            },
            status=502,
        )

    def _is_request_authorized(self, request: web.Request) -> bool:
        if not self.auth_token:
            return True

        auth_header = request.headers.get("Authorization", "")
        token = ""
        if auth_header.lower().startswith("bearer "):
            token = auth_header[7:].strip()
        if not token:
            token = request.headers.get("X-RingRift-Auth", "").strip()
        if not token:
            return False

        return secrets.compare_digest(token, self.auth_token)
