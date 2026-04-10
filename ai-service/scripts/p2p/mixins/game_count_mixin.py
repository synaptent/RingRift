"""Selfplay scheduler game-count seeding and refresh helpers.

April 2026: Extracted from p2p_orchestrator.py (Part 3 Phase 2).
"""
from __future__ import annotations

from scripts.p2p.p2p_mixin_base import P2PMixinBase
from scripts.p2p.startup_infrastructure import *  # noqa: F401,F403


class GameCountMixin(P2PMixinBase):
    """Mixin for P2POrchestrator selfplay scheduler game-count seeding and refresh helpers."""

    MIXIN_TYPE = "game_count"

    def _seed_selfplay_scheduler_game_counts_sync(self) -> dict[str, int]:
        """Seed game counts from canonical databases synchronously.

        IMPORTANT: This is a blocking operation. Call via asyncio.to_thread() from async code.
        Added Jan 2026 (Session 17.29) to fix bootstrap priority for underserved configs.

        Returns:
            Dict mapping config_key -> game_count from canonical databases
        """
        game_counts: dict[str, int] = {}
        # Jan 7, 2026: Use _get_ai_service_path() to avoid doubled ai-service/ path
        canonical_dir = Path(self._get_ai_service_path()) / "data" / "games"

        # Pattern: canonical_<board_type>_<num_players>p.db
        for db_path in canonical_dir.glob("canonical_*_*p.db"):
            try:
                # Extract config_key from filename: canonical_hex8_2p.db -> hex8_2p
                stem = db_path.stem  # canonical_hex8_2p
                if stem.startswith("canonical_"):
                    config_key = stem[len("canonical_"):]  # hex8_2p
                    # Inline: was _get_db_game_count_sync()
                    game_count = self.data_pipeline_manager.get_db_game_count_sync(db_path)
                    if game_count > 0:
                        game_counts[config_key] = game_count
            except (ValueError, AttributeError):
                continue

        return game_counts

    async def _fetch_game_counts_from_peers(self) -> dict[str, int]:
        """Fetch game counts from coordinator or other peers with canonical databases.

        Session 17.41: Cluster nodes don't have canonical databases, so they need to
        fetch game counts from the coordinator which has them. This enables the
        starvation multipliers to work correctly on all nodes.

        Returns:
            Dict mapping config_key -> game_count from peers
        """
        # Try coordinator nodes first (they have canonical databases)
        # Jan 2026: Use lock-free PeerSnapshot for read-only access
        peers_snapshot = self._peer_snapshot.get_snapshot()
        coordinator_candidates = []
        for peer_id, peer in peers_snapshot.items():
            # Coordinator nodes or nodes with role=coordinator
            role_str = getattr(peer.role, "value", str(peer.role)) if peer.role else ""
            if "coordinator" in role_str.lower() or "mac-studio" in peer_id.lower():
                coordinator_candidates.append(peer)

        # Fallback to any alive peer
        if not coordinator_candidates:
            coordinator_candidates = [p for p in peers_snapshot.values() if p.is_alive()]

        for peer in coordinator_candidates[:3]:  # Try up to 3 candidates
            try:
                # Get best endpoint for peer
                key = self._endpoint_key(peer)
                if not key:
                    continue
                scheme, host, port = key
                url = f"{scheme}://{host}:{port}/game_counts"

                async with aiohttp.ClientSession() as session:
                    async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            game_counts = data.get("game_counts", {})
                            if game_counts:
                                source_node = data.get("node_id", peer.node_id)
                                logger.info(f"[P2P] Fetched {len(game_counts)} game counts from {source_node}")
                                return game_counts
            except (aiohttp.ClientError, asyncio.TimeoutError, json.JSONDecodeError) as e:
                logger.debug(f"[P2P] Failed to fetch game counts from {peer.node_id}: {e}")
                continue

        # Session 17.48: Fallback to known coordinator IPs from config if peer discovery failed
        # This handles the case where P2P network hasn't converged yet (no heartbeats from coordinator)
        fallback_coordinator_ips = [
            "100.69.164.58",  # macbook-pro-2-1 Tailscale IP (has canonical DBs)
        ]
        for ip in fallback_coordinator_ips:
            try:
                url = f"http://{ip}:8770/game_counts"
                async with aiohttp.ClientSession() as session:
                    async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            game_counts = data.get("game_counts", {})
                            if game_counts:
                                source_node = data.get("node_id", "unknown")
                                logger.info(f"[P2P] Fetched {len(game_counts)} game counts from fallback {source_node}")
                                return game_counts
            except (aiohttp.ClientError, asyncio.TimeoutError, json.JSONDecodeError) as e:
                logger.debug(f"[P2P] Fallback fetch from {ip} failed: {e}")
                continue

        return {}

    async def _async_seed_game_counts_from_peers_if_needed(self) -> None:
        """Async fallback to seed game counts from peers if local seeding failed.

        Jan 9, 2026: Cluster nodes don't have local canonical databases, so
        the synchronous seeding during __init__ returns empty. This method
        fetches game counts from the coordinator/peers during async startup,
        enabling proper underserved config prioritization on worker nodes.

        Without this, all configs appear to have 0 games and get the same
        maximum bootstrap boost (+100), which neutralizes the prioritization.
        """
        try:
            # Check if game counts were already seeded during __init__
            if self.selfplay_scheduler:
                existing_counts = self.selfplay_scheduler._get_game_counts_per_config()
                if existing_counts and len(existing_counts) >= 6:
                    # Already have game counts from local canonical DBs
                    logger.debug(
                        f"[P2P] Game counts already seeded ({len(existing_counts)} configs), "
                        "skipping peer fetch"
                    )
                    return

            # Fetch from peers/coordinator
            logger.info("[P2P] Local canonical DBs empty, fetching game counts from peers...")
            peer_counts = await self._fetch_game_counts_from_peers()

            if peer_counts and self.selfplay_scheduler:
                self.selfplay_scheduler.update_p2p_game_counts(peer_counts)
                logger.info(
                    f"[P2P] Seeded SelfplayScheduler with {len(peer_counts)} config game counts from peers"
                )
                # Log underserved configs for visibility
                for config_key, count in sorted(peer_counts.items(), key=lambda x: x[1]):
                    if count < 5000:
                        logger.info(f"[P2P] Underserved config (from peers): {config_key} = {count} games")
            else:
                logger.warning(
                    "[P2P] Could not fetch game counts from peers - "
                    "bootstrap prioritization may not work correctly"
                )

        except Exception as e:  # noqa: BLE001
            logger.warning(f"[P2P] Async game count seeding failed: {e}")

    async def _game_count_refresh_loop(self) -> None:
        """Periodically refresh game counts from coordinator.

        Jan 9, 2026: Cluster nodes need to periodically refresh game counts
        as games are generated and consolidated. This ensures the scheduler
        always has accurate game counts for prioritization decisions.

        Interval: 5 minutes (300 seconds)
        """
        REFRESH_INTERVAL = 300  # 5 minutes
        await asyncio.sleep(60)  # Initial delay to let cluster stabilize

        while True:
            try:
                # Skip if this node has local canonical DBs (coordinator)
                local_counts = await asyncio.to_thread(self._seed_selfplay_scheduler_game_counts_sync)
                if local_counts and len(local_counts) >= 6:
                    # Has local DBs - update from local
                    if self.selfplay_scheduler:
                        self.selfplay_scheduler.update_p2p_game_counts(local_counts)
                        logger.debug(f"[P2P] Refreshed game counts from local DBs ({len(local_counts)} configs)")
                else:
                    # Fetch from peers
                    peer_counts = await self._fetch_game_counts_from_peers()
                    if peer_counts and self.selfplay_scheduler:
                        self.selfplay_scheduler.update_p2p_game_counts(peer_counts)
                        logger.debug(f"[P2P] Refreshed game counts from peers ({len(peer_counts)} configs)")

            except Exception as e:  # noqa: BLE001
                logger.debug(f"[P2P] Game count refresh failed: {e}")

            await asyncio.sleep(REFRESH_INTERVAL)
