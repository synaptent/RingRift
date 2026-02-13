"""PULL strategy operations mixin for AutoSyncDaemon.

December 2025: Extracted from auto_sync_daemon.py as part of mixin-based refactoring.
December 2025: Updated to inherit from SyncMixinBase for common functionality.

This mixin provides:
- Pull-based sync from cluster nodes to coordinator
- Remote database listing and discovery
- Rsync-based pull operations with checksum verification
- Database merging into canonical databases
"""

from __future__ import annotations

import asyncio
import logging
import re
import socket
import sqlite3
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from app.coordination.sync_mixin_base import SyncMixinBase
from app.coordination.event_emission_helpers import safe_emit_event
from app.config.coordination_defaults import build_ssh_options, build_ssh_options_list

# Threshold for using ResilientTransfer vs simple rsync
LARGE_FILE_THRESHOLD_BYTES = 50_000_000  # 50MB

if TYPE_CHECKING:
    from app.coordination.sync_strategies import SyncStats
    from app.distributed.circuit_breaker import CircuitBreaker

logger = logging.getLogger(__name__)


class SyncPullMixin(SyncMixinBase):
    """Mixin providing PULL strategy operations for AutoSyncDaemon.

    Inherits from SyncMixinBase for common error handling and logging utilities.

    The PULL strategy is used for coordinator recovery - pulling data FROM
    cluster nodes TO the coordinator, which is the reverse of normal sync.
    """

    # Note: All common attributes (node_id, _stats, _circuit_breaker) are
    # inherited from SyncMixinBase

    async def _pull_from_cluster_nodes(self) -> int:
        """Pull data FROM cluster nodes TO coordinator (reverse sync).

        December 2025: Implements PULL strategy for coordinator recovery.
        This is the inverse of normal sync - coordinator pulls data from
        generating nodes rather than pushing to receivers.

        Used for:
        - Coordinator data recovery after restart
        - Backfilling missing data from cluster
        - Consolidating distributed game databases

        Returns:
            Number of games pulled and validated.
        """
        # Only coordinators should use PULL strategy
        # Check env.is_coordinator from centralized config
        try:
            from app.config.env import env
            is_coordinator = env.is_coordinator
        except ImportError:
            # Fallback to checking hostname
            is_coordinator = "mac-studio" in socket.gethostname().lower() or "coordinator" in socket.gethostname().lower()

        if not is_coordinator:
            logger.debug("[AutoSyncDaemon] PULL strategy requires coordinator role")
            return 0

        # Get sync sources from SyncRouter
        try:
            from app.coordination.sync_router import get_sync_router, DataType

            sync_router = get_sync_router()
            if not sync_router:
                logger.warning("[AutoSyncDaemon] SyncRouter not available for PULL")
                return 0

            # Refresh node capabilities before routing
            sync_router.refresh_from_cluster_config()

            sources = sync_router.get_sync_sources(
                data_type=DataType.GAME,
                target_node=self.node_id,
                max_sources=5,  # Limit to top 5 sources per cycle
            )

            if not sources:
                logger.debug("[AutoSyncDaemon] No sync sources available for PULL")
                return 0

            logger.info(
                f"[AutoSyncDaemon] PULL: Found {len(sources)} sources: "
                f"{[s.node_id for s in sources]}"
            )

        except ImportError as e:
            logger.warning(f"[AutoSyncDaemon] SyncRouter import failed: {e}")
            return 0

        # Pull from each source
        total_pulled = 0
        for source in sources:
            try:
                pulled = await self._pull_from_node(source.node_id)
                total_pulled += pulled
                if pulled > 0:
                    logger.info(
                        f"[AutoSyncDaemon] Pulled {pulled} games from {source.node_id}"
                    )
            except Exception as e:
                logger.warning(
                    f"[AutoSyncDaemon] Failed to pull from {source.node_id}: {e}"
                )
                # Record failure for circuit breaker
                if self._circuit_breaker:
                    self._circuit_breaker.record_failure(source.node_id)

        # Update stats
        # Dec 28, 2025: Fixed sync_cycles -> syncs_completed (correct field name)
        # January 2026: Use _sync_stats (SyncStats) not _stats (HandlerStats) for HandlerBase migration
        self._sync_stats.syncs_completed += 1
        self._sync_stats.games_synced += total_pulled

        # Emit sync completion event
        if total_pulled > 0:
            await self._emit_pull_sync_completed(total_pulled, len(sources))

        return total_pulled

    async def _pull_from_node(self, source_node: str) -> int:
        """Pull game databases from a specific cluster node.

        Args:
            source_node: Node ID to pull from

        Returns:
            Number of games successfully pulled and validated.
        """
        from app.coordination.sync_integrity import check_sqlite_integrity

        # Get node SSH config
        try:
            from app.config.cluster_config import get_cluster_nodes

            nodes = get_cluster_nodes()
            node = nodes.get(source_node)
            if not node:
                logger.warning(f"[AutoSyncDaemon] Node {source_node} not found in config")
                return 0

            # Get SSH connection info
            ssh_host = node.best_ip
            ssh_user = node.ssh_user or "ubuntu"
            ssh_key = node.ssh_key or "~/.ssh/id_cluster"

            if not ssh_host:
                logger.warning(f"[AutoSyncDaemon] No SSH host for {source_node}")
                return 0

        except ImportError as e:
            logger.warning(f"[AutoSyncDaemon] cluster_config import failed: {e}")
            return 0

        import os
        ssh_key = os.path.expanduser(ssh_key)

        # List remote databases
        remote_games_path = self._get_remote_games_path(source_node)
        remote_dbs = await self._list_remote_databases(
            ssh_host, ssh_user, ssh_key, remote_games_path
        )

        if not remote_dbs:
            logger.debug(f"[AutoSyncDaemon] No databases found on {source_node}")
            return 0

        logger.debug(
            f"[AutoSyncDaemon] Found {len(remote_dbs)} databases on {source_node}"
        )

        # Prepare local temp directory for pulled databases
        base_dir = Path(__file__).resolve().parent.parent.parent
        pull_dir = base_dir / "data" / "games" / "pulled"
        pull_dir.mkdir(parents=True, exist_ok=True)

        games_pulled = 0
        for remote_db in remote_dbs:
            try:
                # Pull database (December 2025: pass source_node for ResilientTransfer)
                local_path = await self._rsync_pull(
                    ssh_host, ssh_user, ssh_key,
                    remote_games_path, remote_db, pull_dir,
                    source_node=source_node,
                )

                if not local_path or not local_path.exists():
                    continue

                # Validate completeness
                is_valid, msg = self._validate_database_completeness(local_path)
                if not is_valid:
                    logger.warning(
                        f"[AutoSyncDaemon] Skipping {remote_db} from {source_node}: {msg}"
                    )
                    local_path.unlink(missing_ok=True)
                    continue

                # Check integrity with adaptive timeout for large databases
                # Dec 29, 2025: Use fast check for DBs >100MB to prevent timeouts
                db_size_mb = local_path.stat().st_size / (1024 * 1024)
                use_fast = db_size_mb > 100  # Fast check for DBs > 100MB
                is_intact, errors = check_sqlite_integrity(
                    local_path,
                    use_fast_check=use_fast,
                    timeout_seconds=15.0 if use_fast else 30.0,
                )
                if not is_intact:
                    logger.warning(
                        f"[AutoSyncDaemon] {remote_db} from {source_node} failed integrity: {errors}"
                    )
                    local_path.unlink(missing_ok=True)
                    continue

                # Count games for stats (Dec 27, 2025: Use context manager to prevent leaks)
                # Dec 30, 2025: Wrap SQLite in asyncio.to_thread() to avoid blocking event loop
                def _count_games_sync(db_path: Path) -> int:
                    try:
                        with sqlite3.connect(str(db_path), timeout=10.0) as conn:
                            cursor = conn.cursor()
                            cursor.execute("SELECT COUNT(*) FROM games")
                            return cursor.fetchone()[0]
                    except sqlite3.Error:
                        return 0

                game_count = await asyncio.to_thread(_count_games_sync, local_path)
                games_pulled += game_count

                # Merge into canonical database
                await self._merge_into_canonical(local_path, source_node)

                # Clean up pulled file after merge
                local_path.unlink(missing_ok=True)

            except Exception as e:
                logger.warning(
                    f"[AutoSyncDaemon] Error pulling {remote_db} from {source_node}: {e}"
                )

        return games_pulled

    async def _list_remote_databases(
        self,
        ssh_host: str,
        ssh_user: str,
        ssh_key: str,
        remote_path: str,
    ) -> list[str]:
        """List database files on a remote node.

        Returns:
            List of database filenames (not full paths).
        """
        # Dec 30, 2025: Use centralized SSH config for consistent timeouts
        cmd = build_ssh_options_list(
            key_path=ssh_key,
            include_keepalive=False,  # Quick listing, no keepalive needed
        )
        cmd.extend([f"{ssh_user}@{ssh_host}", f"ls -1 {remote_path}/*.db 2>/dev/null || echo ''"])

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=30.0)

            output = stdout.decode().strip()
            if not output:
                return []

            # Extract just filenames
            dbs = []
            for line in output.split("\n"):
                line = line.strip()
                if line.endswith(".db"):
                    dbs.append(Path(line).name)

            return dbs

        except asyncio.TimeoutError:
            logger.warning(f"[AutoSyncDaemon] Timeout listing databases on {ssh_host}")
            return []
        except Exception as e:
            logger.debug(f"[AutoSyncDaemon] Error listing remote dbs: {e}")
            return []

    async def _resilient_pull(
        self,
        source_node: str,
        remote_path: str,
        db_name: str,
        local_dir: Path,
    ) -> Path | None:
        """Pull a file using ResilientTransfer for reliability.

        December 2025: Uses BitTorrent/aria2 for large files with piece-level
        verification. Falls back to rsync/base64/chunked for smaller files or
        when preferred transports are unavailable.

        Args:
            source_node: Node ID (matches distributed_hosts.yaml)
            remote_path: Remote directory path
            db_name: Database filename
            local_dir: Local directory to save to

        Returns:
            Local path to the pulled file, or None if failed.
        """
        try:
            from app.distributed.resilient_transfer import (
                ResilientTransfer,
                TransferRequest,
            )
        except ImportError:
            logger.debug("[AutoSyncDaemon] resilient_transfer not available, falling back to rsync")
            return None

        local_path = local_dir / db_name
        remote_file = f"{remote_path}/{db_name}"

        try:
            transfer = ResilientTransfer(
                prefer_bittorrent=True,
                verify_all=True,
                quarantine_on_failure=True,
            )

            request = TransferRequest(
                source_node=source_node,
                source_path=remote_file,
                target_path=local_path,
                file_type="db",  # SQLite validation
                priority="high",
            )

            result = await transfer.transfer(request)

            if result.success and result.verification_passed:
                logger.info(
                    f"[AutoSyncDaemon] Resilient pull succeeded: {db_name} via {result.transport_used} "
                    f"({result.bytes_transferred / 1024 / 1024:.1f}MB)"
                )
                self._stats.databases_verified += 1
                return local_path
            else:
                logger.warning(
                    f"[AutoSyncDaemon] Resilient pull failed for {db_name}: {result.error}"
                )
                self._stats.databases_verification_failed += 1
                return None

        except Exception as e:
            logger.warning(f"[AutoSyncDaemon] Resilient pull error for {db_name}: {e}")
            return None

    async def _rsync_pull(
        self,
        ssh_host: str,
        ssh_user: str,
        ssh_key: str,
        remote_path: str,
        db_name: str,
        local_dir: Path,
        verify_checksum: bool = True,
        source_node: str | None = None,
    ) -> Path | None:
        """Pull a single database file from a remote node.

        December 2025: Added checksum verification after pull.
        December 2025: Uses ResilientTransfer for large files (>50MB).

        Args:
            ssh_host: Remote host IP/hostname
            ssh_user: SSH username
            ssh_key: Path to SSH private key
            remote_path: Remote directory path
            db_name: Database filename
            local_dir: Local directory to save to
            verify_checksum: If True, verify checksum after pull (default: True)
            source_node: Node ID for ResilientTransfer (optional)

        Returns:
            Local path to the pulled file, or None if failed.
        """
        # Check if file is large enough to warrant ResilientTransfer
        if source_node:
            try:
                # Try to get remote file size
                # Dec 30, 2025: Use centralized SSH config for consistent timeouts
                size_cmd = build_ssh_options_list(
                    key_path=ssh_key,
                    include_keepalive=False,  # Quick size check, no keepalive needed
                )
                size_cmd.extend([
                    f"{ssh_user}@{ssh_host}",
                    f"stat -c%s '{remote_path}/{db_name}' 2>/dev/null || echo 0"
                ])
                proc = await asyncio.create_subprocess_exec(
                    *size_cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=15.0)
                remote_size = int(stdout.decode().strip() or "0")

                if remote_size >= LARGE_FILE_THRESHOLD_BYTES:
                    logger.info(
                        f"[AutoSyncDaemon] Large file detected ({remote_size / 1024 / 1024:.1f}MB), "
                        f"using ResilientTransfer for {db_name}"
                    )
                    result = await self._resilient_pull(
                        source_node=source_node,
                        remote_path=remote_path,
                        db_name=db_name,
                        local_dir=local_dir,
                    )
                    if result:
                        return result
                    # Fall through to rsync if resilient pull fails
                    logger.info("[AutoSyncDaemon] Falling back to rsync after resilient pull failure")

            except Exception as e:
                logger.debug(f"[AutoSyncDaemon] Could not check remote size, using rsync: {e}")

        local_path = local_dir / db_name

        # Feb 2026: Memory-aware transfer - try aria2/HTTP if memory is high
        try:
            from app.coordination.rsync_command_builder import should_use_rsync, aria2_pull_file
            if not should_use_rsync():
                http_url = f"http://{ssh_host}:8766/games/{db_name}"
                logger.info(
                    f"[AutoSyncDaemon] Memory-aware: using aria2 for {db_name} from {ssh_host}"
                )
                success, _bytes_dl, error = await aria2_pull_file(
                    http_url=http_url,
                    local_path=local_dir,
                    filename=db_name,
                )
                if success and (local_dir / db_name).exists():
                    return local_dir / db_name
                logger.info(f"[AutoSyncDaemon] aria2 fallback failed ({error}), using rsync")
        except ImportError:
            pass

        remote_full = f"{ssh_user}@{ssh_host}:{remote_path}/{db_name}"

        # Dec 30, 2025: Use centralized SSH config for rsync
        ssh_options = build_ssh_options(
            key_path=ssh_key,
            include_keepalive=False,  # rsync has its own timeout
        )
        cmd = [
            "rsync",
            "-az",
            "--timeout=60",
            "-e", ssh_options,
            remote_full,
            str(local_path),
        ]

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            _, stderr = await asyncio.wait_for(proc.communicate(), timeout=120.0)

            if proc.returncode == 0 and local_path.exists():
                # December 2025: Checksum verification after pull
                if verify_checksum:
                    try:
                        from app.coordination.sync_integrity import verify_sync_checksum

                        remote_file_path = f"{remote_path}/{db_name}"
                        checksum_ok, error = await verify_sync_checksum(
                            source_path=str(local_path),
                            dest_path=remote_file_path,
                            ssh_host=ssh_host,
                            ssh_user=ssh_user,
                            ssh_key=ssh_key,
                        )

                        # Note: For pulls, the "source" is remote and "dest" is local
                        # So we swap the comparison
                        if not checksum_ok:
                            logger.error(
                                f"[AutoSyncDaemon] Checksum mismatch after pull for {db_name} "
                                f"from {ssh_host}: {error}"
                            )
                            # Delete corrupted file and retry once
                            local_path.unlink(missing_ok=True)

                            # Retry the pull
                            proc = await asyncio.create_subprocess_exec(
                                *cmd,
                                stdout=asyncio.subprocess.PIPE,
                                stderr=asyncio.subprocess.PIPE,
                            )
                            _, _ = await asyncio.wait_for(proc.communicate(), timeout=120.0)

                            if proc.returncode == 0 and local_path.exists():
                                # Verify again
                                checksum_ok2, error2 = await verify_sync_checksum(
                                    source_path=str(local_path),
                                    dest_path=remote_file_path,
                                    ssh_host=ssh_host,
                                    ssh_user=ssh_user,
                                    ssh_key=ssh_key,
                                )
                                if not checksum_ok2:
                                    logger.error(
                                        f"[AutoSyncDaemon] Checksum still mismatched after "
                                        f"retry for {db_name}: {error2}"
                                    )
                                    local_path.unlink(missing_ok=True)
                                    self._stats.databases_verification_failed += 1
                                    return None
                                else:
                                    self._stats.databases_verified += 1
                                    logger.info(
                                        f"[AutoSyncDaemon] Pull verified after retry: {db_name}"
                                    )
                            else:
                                self._stats.databases_verification_failed += 1
                                return None
                        else:
                            self._stats.databases_verified += 1
                            logger.debug(
                                f"[AutoSyncDaemon] Pull checksum verified: {db_name}"
                            )

                    except ImportError:
                        logger.debug("[AutoSyncDaemon] sync_integrity not available, skipping verification")
                    except Exception as e:
                        logger.warning(f"[AutoSyncDaemon] Checksum verification error: {e}")
                        # Don't fail the pull if verification fails - just log it

                return local_path
            else:
                if stderr:
                    logger.debug(f"[AutoSyncDaemon] Rsync pull error: {stderr.decode()}")
                return None

        except asyncio.TimeoutError:
            logger.warning(f"[AutoSyncDaemon] Rsync pull timeout for {db_name}, trying fallback")
            # December 2025: Use robust_pull fallback on timeout
            fallback_path = await self._robust_pull_fallback(
                ssh_host, ssh_user, ssh_key, remote_path, db_name, local_dir
            )
            return fallback_path
        except Exception as e:
            logger.warning(f"[AutoSyncDaemon] Rsync pull error: {e}, trying fallback")
            # December 2025: Use robust_pull fallback on error
            fallback_path = await self._robust_pull_fallback(
                ssh_host, ssh_user, ssh_key, remote_path, db_name, local_dir
            )
            return fallback_path

    async def _robust_pull_fallback(
        self,
        ssh_host: str,
        ssh_user: str,
        ssh_key: str,
        remote_path: str,
        db_name: str,
        local_dir: Path,
    ) -> Path | None:
        """Fallback to robust_pull when rsync fails.

        December 2025: Added for improved sync reliability.
        Uses multi-transport fallback: rsync → scp → base64 → http

        Args:
            ssh_host: Remote host IP/hostname
            ssh_user: SSH username
            ssh_key: Path to SSH private key
            remote_path: Remote directory path
            db_name: Database filename
            local_dir: Local directory to save to

        Returns:
            Local path to the pulled file, or None if failed.
        """
        try:
            from scripts.lib.transfer import robust_pull, TransferConfig

            config = TransferConfig(
                ssh_key=ssh_key,
                ssh_user=ssh_user,
                ssh_port=22,
                connect_timeout=10,
                transfer_timeout=300,
                max_retries=2,
                compress=True,
                verify_checksum=True,
            )

            local_path = local_dir / db_name
            remote_file = f"{remote_path}/{db_name}"

            # Run in thread pool since robust_pull is synchronous
            result = await asyncio.get_running_loop().run_in_executor(
                None,
                lambda: robust_pull(ssh_host, 22, remote_file, local_path, config),
            )

            if result.success and local_path.exists():
                logger.info(
                    f"[AutoSyncDaemon] Fallback pull succeeded via {result.method}: "
                    f"{db_name} from {ssh_host}"
                )
                return local_path
            else:
                logger.warning(
                    f"[AutoSyncDaemon] All fallback transports failed for {db_name}: "
                    f"{result.error}"
                )
                return None

        except ImportError:
            logger.debug("[AutoSyncDaemon] robust_pull not available")
            return None
        except Exception as e:
            logger.warning(f"[AutoSyncDaemon] Fallback pull error for {db_name}: {e}")
            return None

    async def _merge_into_canonical(self, pulled_db: Path, source_node: str) -> None:
        """Merge a pulled database into the appropriate canonical database.

        Uses ATTACH DATABASE to copy games that don't already exist locally.

        Args:
            pulled_db: Path to the pulled database file
            source_node: Source node for logging
        """
        # Determine canonical database from pulled db name
        # e.g., "selfplay_hex8_2p.db" -> "canonical_hex8_2p.db"
        db_name = pulled_db.name
        canonical_name = self._get_canonical_name(db_name)

        base_dir = Path(__file__).resolve().parent.parent.parent
        canonical_path = base_dir / "data" / "games" / canonical_name

        # If no canonical exists, just rename the pulled file
        if not canonical_path.exists():
            pulled_db.rename(canonical_path)
            logger.info(
                f"[AutoSyncDaemon] Created canonical {canonical_name} from {source_node}"
            )
            return

        # Merge games from pulled into canonical (Dec 28, 2025: Context manager pattern)
        # Dec 30, 2025: Wrap blocking SQLite merge in asyncio.to_thread()
        def _merge_databases_sync(
            canonical: Path, pulled: Path
        ) -> tuple[int, str | None]:
            """Sync helper to merge databases without blocking event loop.

            CRITICAL: Only copies games that have move data to prevent orphan games.

            Returns:
                Tuple of (new_games_count, error_message_or_none)
            """
            from app.db.move_data_validator import MIN_MOVES_REQUIRED

            try:
                with sqlite3.connect(str(canonical), timeout=30.0) as conn:
                    cursor = conn.cursor()

                    # Attach pulled database
                    cursor.execute("ATTACH DATABASE ? AS pulled", (str(pulled),))

                    try:
                        # Check if pulled database has game_moves table
                        cursor.execute(
                            "SELECT name FROM pulled.sqlite_master "
                            "WHERE type='table' AND name='game_moves'"
                        )
                        has_game_moves = cursor.fetchone() is not None

                        if not has_game_moves:
                            logger.warning(
                                f"[AutoSyncDaemon] Pulled database has no game_moves table, "
                                f"skipping merge to prevent orphan games"
                            )
                            return 0, None

                        # Get count before merge
                        cursor.execute("SELECT COUNT(*) FROM games")
                        before_count = cursor.fetchone()[0]

                        # CRITICAL: Only copy games that have sufficient moves
                        # This prevents orphan games (metadata without moves)
                        cursor.execute(f"""
                            INSERT OR IGNORE INTO games
                            SELECT g.* FROM pulled.games g
                            INNER JOIN (
                                SELECT game_id, COUNT(*) as move_count
                                FROM pulled.game_moves
                                GROUP BY game_id
                                HAVING move_count >= {MIN_MOVES_REQUIRED}
                            ) gm ON g.game_id = gm.game_id
                            WHERE g.game_id NOT IN (SELECT game_id FROM games)
                        """)

                        # Copy moves for new games (only for games we just inserted)
                        cursor.execute("""
                            INSERT OR IGNORE INTO game_moves
                            SELECT * FROM pulled.game_moves
                            WHERE game_id IN (SELECT game_id FROM games)
                              AND game_id NOT IN (SELECT DISTINCT game_id FROM game_moves)
                        """)

                        conn.commit()

                        # Get count after merge
                        cursor.execute("SELECT COUNT(*) FROM games")
                        after_count = cursor.fetchone()[0]

                        return after_count - before_count, None
                    finally:
                        # Always detach pulled database
                        try:
                            cursor.execute("DETACH DATABASE pulled")
                        except sqlite3.Error:
                            pass  # May already be detached

            except sqlite3.Error as e:
                return 0, str(e)

        new_games, error = await asyncio.to_thread(
            _merge_databases_sync, canonical_path, pulled_db
        )

        if error:
            logger.warning(f"[AutoSyncDaemon] Merge failed for {db_name}: {error}")
        elif new_games > 0:
            logger.info(
                f"[AutoSyncDaemon] Merged {new_games} games from {source_node} "
                f"into {canonical_name}"
            )

    def _get_canonical_name(self, db_name: str) -> str:
        """Convert a database name to its canonical form.

        Examples:
            selfplay_hex8_2p.db -> canonical_hex8_2p.db
            games_square8_4p.db -> canonical_square8_4p.db
            hex8_2p_selfplay.db -> canonical_hex8_2p.db
        """
        # Extract board type and player count from name
        # Try common patterns
        patterns = [
            r"(hex8|square8|square19|hexagonal)_(\d)p",  # hex8_2p
            r"(\d)p_(hex8|square8|square19|hexagonal)",  # 2p_hex8
        ]

        for pattern in patterns:
            match = re.search(pattern, db_name)
            if match:
                groups = match.groups()
                if groups[0].isdigit():
                    # Pattern 2: 2p_hex8
                    board_type = groups[1]
                    num_players = groups[0]
                else:
                    # Pattern 1: hex8_2p
                    board_type = groups[0]
                    num_players = groups[1]
                return f"canonical_{board_type}_{num_players}p.db"

        # Fallback: just prefix with canonical_
        if db_name.startswith("canonical_"):
            return db_name
        return f"canonical_{db_name}"

    def _get_remote_games_path(self, node_id: str) -> str:
        """Get the remote games directory path for a node.

        Different providers have different paths.
        """
        # Check provider from config
        try:
            from app.config.cluster_config import get_host_provider

            provider = get_host_provider(node_id)
        except ImportError:
            provider = None

        # Common paths by provider
        if provider == "runpod":
            return "/workspace/ringrift/ai-service/data/games"
        elif provider == "vast":
            return "~/ringrift/ai-service/data/games"
        elif provider == "nebius":
            return "~/ringrift/ai-service/data/games"
        elif provider == "vultr":
            return "/root/ringrift/ai-service/data/games"
        else:
            return "~/ringrift/ai-service/data/games"

    async def _emit_pull_sync_completed(self, games_pulled: int, sources_count: int) -> None:
        """Emit event when PULL sync completes successfully."""
        safe_emit_event(
            "data_sync_completed",
            {
                "sync_type": "pull",
                "games_synced": games_pulled,
                "sources_count": sources_count,
                "node_id": self.node_id,
                "timestamp": time.time(),
            },
            context="SyncPullMixin",
        )

    # Note: _validate_database_completeness() is expected from main class
    # _emit_sync_failure() and _emit_sync_stalled() are inherited from SyncMixinBase
