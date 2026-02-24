"""Unified Loop Data Collection Services.

This module contains data collection services for the unified AI loop:
- StreamingDataCollector: Collects game data from remote hosts with incremental sync

Extracted from unified_ai_loop.py for better modularity (Phase 2 refactoring).
"""

from __future__ import annotations

import asyncio
import sqlite3
import time
from typing import TYPE_CHECKING, Any, Optional

from .config import DataEvent, DataEventType, DataIngestionConfig, HostState

if TYPE_CHECKING:
    from unified_ai_loop import EventBus, UnifiedLoopState

    from app.training.hot_data_buffer import HotDataBuffer

import contextlib

from app.utils.paths import AI_SERVICE_ROOT

# Unified game discovery - finds all game databases across all storage patterns
try:
    from app.utils.game_discovery import GameDiscovery
    HAS_GAME_DISCOVERY = True
except ImportError:
    HAS_GAME_DISCOVERY = False
    GameDiscovery = None

# Resource checking - disk threshold from canonical source
try:
    from scripts.p2p.constants import MAX_DISK_USAGE_PERCENT
    from scripts.p2p.resource import check_disk_has_capacity
except ImportError:
    try:
        from app.config.thresholds import DISK_SYNC_TARGET_PERCENT
        MAX_DISK_USAGE_PERCENT = float(DISK_SYNC_TARGET_PERCENT)
    except ImportError:
        MAX_DISK_USAGE_PERCENT = 70.0
    def check_disk_has_capacity():
        return (True, 0.0)

# Unified manifest for game deduplication
try:
    from app.distributed.unified_manifest import DataManifest
    HAS_UNIFIED_MANIFEST = True
except ImportError:
    HAS_UNIFIED_MANIFEST = False
    DataManifest = None

# Unified WAL for crash recovery
try:
    from app.distributed.unified_wal import UnifiedWAL
    HAS_UNIFIED_WAL = True
except ImportError:
    HAS_UNIFIED_WAL = False
    UnifiedWAL = None

# Host classification for ephemeral detection
try:
    from app.distributed.host_classification import (
        HostSyncProfile,
        create_sync_profile,
    )
    HAS_HOST_CLASSIFICATION = True
except ImportError:
    HAS_HOST_CLASSIFICATION = False
    HostSyncProfile = None
    create_sync_profile = None

# Circuit breaker for fault tolerance
try:
    from app.distributed.circuit_breaker import (
        CircuitState,
        get_host_breaker,
    )
    HAS_CIRCUIT_BREAKER = True
except ImportError:
    HAS_CIRCUIT_BREAKER = False
    CircuitState = None
    get_host_breaker = None

# Coordination features
try:
    from app.coordination import (
        TransferPriority,
        release_bandwidth,
        request_bandwidth,
        sync_lock,
    )
    HAS_COORDINATION = True
except ImportError:
    HAS_COORDINATION = False
    sync_lock = None
    request_bandwidth = None
    release_bandwidth = None
    TransferPriority = None

# Resource optimizer for config weights
try:
    from app.coordination.resource_optimizer import update_config_weights
    HAS_RESOURCE_OPTIMIZER = True
except ImportError:
    HAS_RESOURCE_OPTIMIZER = False
    update_config_weights = None

# Prometheus metrics - avoid duplicate registration
try:
    from prometheus_client import REGISTRY, Gauge
    HAS_PROMETHEUS = True
    # Check if metric already registered (e.g., by unified_ai_loop.py)
    if 'ringrift_config_weight' in REGISTRY._names_to_collectors:
        CONFIG_WEIGHT = REGISTRY._names_to_collectors['ringrift_config_weight']
    else:
        CONFIG_WEIGHT = Gauge('ringrift_config_weight', 'Training weight for config', ['config_key'])
except ImportError:
    HAS_PROMETHEUS = False
    CONFIG_WEIGHT = None


class StreamingDataCollector:
    """Collects game data from remote hosts with 60-second incremental sync.

    Uses consolidated infrastructure:
    - unified_manifest: Game deduplication and host state tracking
    - unified_wal: Crash-safe data collection
    - host_classification: Ephemeral host detection with aggressive sync
    """

    def __init__(
        self,
        config: DataIngestionConfig,
        state: UnifiedLoopState,
        event_bus: EventBus,
        hot_buffer: HotDataBuffer | None = None,
    ):
        self.config = config
        self.state = state
        self.event_bus = event_bus
        self.hot_buffer = hot_buffer
        self._known_game_ids: set[str] = set()

        # Initialize unified manifest for game deduplication
        self._manifest: DataManifest | None = None
        if HAS_UNIFIED_MANIFEST:
            try:
                manifest_path = AI_SERVICE_ROOT / "data" / "data_manifest.db"
                self._manifest = DataManifest(manifest_path)
                print(f"[DataCollector] Using unified manifest: {manifest_path}")
            except Exception as e:
                print(f"[DataCollector] Failed to initialize manifest: {e}")

        # Initialize unified WAL for crash recovery
        self._wal: UnifiedWAL | None = None
        if HAS_UNIFIED_WAL:
            try:
                wal_path = AI_SERVICE_ROOT / "data" / "unified_wal.db"
                self._wal = UnifiedWAL(wal_path)
                print(f"[DataCollector] Using unified WAL: {wal_path}")
            except Exception as e:
                print(f"[DataCollector] Failed to initialize WAL: {e}")

        # Initialize host sync profiles for ephemeral detection
        self._host_profiles: dict[str, HostSyncProfile] = {}
        if HAS_HOST_CLASSIFICATION:
            self._init_host_profiles()

    def set_hot_buffer(self, hot_buffer: HotDataBuffer) -> None:
        """Set or update the hot buffer for in-memory game caching."""
        self.hot_buffer = hot_buffer
        print(f"[DataCollector] Hot buffer attached (max_size={hot_buffer.max_size})")

    def _init_host_profiles(self) -> None:
        """Initialize host sync profiles from state with storage type classification."""
        if not HAS_HOST_CLASSIFICATION:
            return

        ephemeral_count = 0
        for host in self.state.hosts.values():
            # Create host config dict for classification
            host_config = {
                "ssh_host": host.ssh_host,
                "remote_path": getattr(host, "remote_db_path", ""),
                "storage_type": getattr(host, "storage_type", "persistent"),
            }
            profile = create_sync_profile(host.name, host_config)
            self._host_profiles[host.name] = profile

            if profile.is_ephemeral:
                ephemeral_count += 1
                print(f"[DataCollector] {host.name}: EPHEMERAL (15s sync)")

        if ephemeral_count > 0:
            print(f"[DataCollector] Detected {ephemeral_count} ephemeral hosts")

    def get_sync_interval(self, host_name: str) -> int:
        """Get sync interval for a host based on its storage type.

        Ephemeral hosts (RAM disk) use aggressive 15s sync.
        Persistent hosts use default 60s sync.
        """
        if host_name in self._host_profiles:
            return self._host_profiles[host_name].poll_interval_seconds
        return self.config.poll_interval_seconds

    def is_ephemeral_host(self, host_name: str) -> bool:
        """Check if a host has ephemeral storage (needs aggressive sync)."""
        if host_name in self._host_profiles:
            return self._host_profiles[host_name].is_ephemeral
        return False

    async def sync_host(self, host: HostState) -> int:
        """Sync games from a single host. Returns count of new games."""
        if not host.enabled:
            return 0

        # Circuit breaker check - prevent repeated failures
        if HAS_CIRCUIT_BREAKER:
            breaker = get_host_breaker()
            if not breaker.can_execute(host.ssh_host):
                state = breaker.get_state(host.ssh_host)
                if state == CircuitState.OPEN:
                    print(f"[DataCollector] Skipping {host.name}: circuit open (cooldown)")
                return 0

        try:
            # Query game count on remote host
            ssh_target = f"{host.ssh_user}@{host.ssh_host}"
            port_arg = f"-p {host.ssh_port}" if host.ssh_port != 22 else ""

            # Get game count from all DBs AND JSONL files (GPU selfplay outputs JSONL)
            python_script = (
                "import sqlite3, glob, os; "
                "os.chdir(os.path.expanduser('~/ringrift/ai-service')); "
                "total=0; "
                # Count from DB files
                "dbs=glob.glob('data/games/*.db'); "
                "[total := total + sqlite3.connect(db).execute('SELECT COUNT(*) FROM games').fetchone()[0] for db in dbs if 'schema' not in db]; "
                # Count from JSONL files (one game per line)
                "jsonls=glob.glob('data/games/gpu_selfplay/*/games.jsonl'); "
                "[total := total + sum(1 for _ in open(j)) for j in jsonls]; "
                "print(total)"
            )
            cmd = f'ssh -o ConnectTimeout=10 {port_arg} {ssh_target} "python3 -c \\"{python_script}\\"" 2>/dev/null'

            result = await asyncio.create_subprocess_shell(
                cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await asyncio.wait_for(result.communicate(), timeout=30)

            current_count = int(stdout.decode().strip() or "0")
            new_games = max(0, current_count - host.last_game_count)

            print(f"[DataCollector] {host.name}: {current_count} games (last: {host.last_game_count}, new: {new_games})")

            if new_games >= self.config.min_games_per_sync:
                # Trigger rsync for incremental sync
                if self.config.sync_method == "incremental":
                    await self._incremental_sync(host)
                else:
                    await self._full_sync(host)

                host.last_game_count = current_count
                host.last_sync_time = time.time()

                # Publish event
                await self.event_bus.publish(DataEvent(
                    event_type=DataEventType.NEW_GAMES_AVAILABLE,
                    payload={
                        "host": host.name,
                        "new_games": new_games,
                        "total_games": current_count,
                    }
                ))

            host.consecutive_failures = 0
            # Record success with circuit breaker
            if HAS_CIRCUIT_BREAKER:
                get_host_breaker().record_success(host.ssh_host)
            return new_games

        except Exception as e:
            host.consecutive_failures += 1
            # Record failure with circuit breaker
            if HAS_CIRCUIT_BREAKER:
                get_host_breaker().record_failure(host.ssh_host, e)
            print(f"[DataCollector] Failed to sync {host.name}: {e}")
            return 0

    async def _incremental_sync(self, host: HostState):
        """Perform incremental rsync of new data.

        Uses sync_lock and bandwidth management when available for coordinated
        data transfers across the cluster.
        """
        ssh_target = f"{host.ssh_user}@{host.ssh_host}"
        local_dir = AI_SERVICE_ROOT / "data" / "games" / "synced" / host.name
        local_dir.mkdir(parents=True, exist_ok=True)

        # Base rsync command
        base_cmd = 'rsync -avz --progress -e "ssh -o ConnectTimeout=10"'

        # Use new coordination if available: sync_lock + bandwidth
        if HAS_COORDINATION:
            bandwidth_alloc = None
            try:
                # Acquire sync lock to prevent concurrent rsync operations
                with sync_lock(host=host.ssh_host, operation="data_sync"):
                    # Request bandwidth allocation (estimate ~50MB for DB sync)
                    bandwidth_alloc = request_bandwidth(
                        host=host.ssh_host,
                        estimated_mb=50,  # 50MB estimate
                        priority=TransferPriority.NORMAL,
                    )

                    if bandwidth_alloc and not bandwidth_alloc.granted:
                        print(f"[DataCollector] Bandwidth not available for {host.name}: {bandwidth_alloc.reason}")
                        return

                    # Apply bandwidth limit to rsync command
                    if bandwidth_alloc and bandwidth_alloc.bwlimit_kbps > 0:
                        cmd = f'{base_cmd} --bwlimit={bandwidth_alloc.bwlimit_kbps} {ssh_target}:~/ringrift/ai-service/data/games/*.db {local_dir}/'
                        print(f"[DataCollector] Sync {host.name} with bwlimit={bandwidth_alloc.bwlimit_kbps}KB/s")
                    else:
                        cmd = f'{base_cmd} {ssh_target}:~/ringrift/ai-service/data/games/*.db {local_dir}/'

                    start_time = time.time()
                    process = await asyncio.create_subprocess_shell(
                        cmd,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                    )
                    await asyncio.wait_for(process.communicate(), timeout=300)
                    transfer_duration = time.time() - start_time

                    # Release bandwidth with actual transfer stats
                    if bandwidth_alloc and bandwidth_alloc.granted:
                        release_bandwidth(
                            bandwidth_alloc.allocation_id,
                            bytes_transferred=50 * 1024 * 1024,  # Estimate
                            duration_seconds=transfer_duration
                        )
            except Exception as e:
                print(f"[DataCollector] Sync error for {host.name}: {e}")
                raise
            finally:
                # Ensure bandwidth is released even on error
                if bandwidth_alloc and bandwidth_alloc.granted:
                    with contextlib.suppress(Exception):
                        release_bandwidth(bandwidth_alloc.allocation_id)
        else:
            # Fallback: no coordination - unlimited bandwidth
            cmd = f'{base_cmd} {ssh_target}:~/ringrift/ai-service/data/games/*.db {local_dir}/'
            process = await asyncio.create_subprocess_shell(
                cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await asyncio.wait_for(process.communicate(), timeout=300)

        # Also sync selfplay JSONL files during incremental sync
        await self._sync_selfplay_jsonl(host)

    async def _sync_selfplay_jsonl(self, host: HostState):
        """Sync JSONL training data files from remote host.

        This syncs JSONL files from selfplay and tournament directories that are
        not covered by the standard DB sync. Sources include:
        - GPU selfplay scripts (run_gpu_selfplay.py)
        - Tier tournaments (run_distributed_tournament.py with --record-training-data)
        - Diverse tournaments (run_diverse_tournaments.py)
        """
        ssh_target = f"{host.ssh_user}@{host.ssh_host}"

        # JSONL data patterns to sync (selfplay + tournaments)
        selfplay_patterns = [
            ("data/selfplay/gpu_*", "selfplay/gpu"),
            ("data/selfplay/p2p_gpu", "selfplay/p2p_gpu"),
            ("data/games/gpu_selfplay", "games/gpu_selfplay"),
            # Tournament data - high quality games from tier/Elo tournaments
            ("data/tournaments", "tournaments"),
        ]

        base_cmd = 'rsync -avz --include="*.jsonl" --include="*/" --exclude="*" -e "ssh -o ConnectTimeout=10"'

        for remote_pattern, local_subdir in selfplay_patterns:
            local_dir = AI_SERVICE_ROOT / "data" / local_subdir / "synced" / host.name
            local_dir.mkdir(parents=True, exist_ok=True)

            # Check if remote directory exists first
            check_cmd = f'ssh -o ConnectTimeout=10 {ssh_target} "ls -d ~/ringrift/ai-service/{remote_pattern} 2>/dev/null | head -1"'
            try:
                check_proc = await asyncio.create_subprocess_shell(
                    check_cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                stdout, _ = await asyncio.wait_for(check_proc.communicate(), timeout=30)
                if not stdout.strip():
                    continue  # Directory doesn't exist on remote

                # Sync the directory
                cmd = f'{base_cmd} {ssh_target}:~/ringrift/ai-service/{remote_pattern}/ {local_dir}/'
                process = await asyncio.create_subprocess_shell(
                    cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                await asyncio.wait_for(process.communicate(), timeout=300)
            except asyncio.TimeoutError:
                print(f"[DataCollector] Selfplay sync timeout for {host.name}/{remote_pattern}")
            except Exception as e:
                print(f"[DataCollector] Selfplay sync error for {host.name}/{remote_pattern}: {e}")

    async def _full_sync(self, host: HostState):
        """Perform full sync (same as incremental for now)."""
        await self._incremental_sync(host)
        # Also sync selfplay JSONL files
        await self._sync_selfplay_jsonl(host)

    def compute_quality_stats(self, sample_size: int = 500) -> dict[str, Any]:
        """Compute data quality statistics from all available databases.

        Uses unified GameDiscovery to find databases across all storage patterns,
        not just the synced directory.

        Returns:
            Dictionary with draw_rate, timeout_rate, game_lengths, etc.
        """
        total_games = 0
        draws = 0
        timeouts = 0
        game_lengths = []

        # Use unified GameDiscovery if available
        if HAS_GAME_DISCOVERY:
            discovery = GameDiscovery()
            db_paths = [db_info.path for db_info in discovery.find_all_databases()]
        else:
            # Fallback to synced directory only
            synced_dir = AI_SERVICE_ROOT / "data" / "games" / "synced"
            if not synced_dir.exists():
                return {"draw_rate": 0, "timeout_rate": 0, "game_lengths": []}
            db_paths = list(synced_dir.rglob("*.db"))

        for db_path in db_paths:
            try:
                conn = sqlite3.connect(str(db_path))
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()

                # Check table structure
                cursor.execute("PRAGMA table_info(games)")
                columns = {row['name'] for row in cursor.fetchall()}

                has_winner = 'winner' in columns
                has_total_moves = 'total_moves' in columns

                # Sample recent games
                if has_winner and has_total_moves:
                    cursor.execute("""
                        SELECT winner, total_moves FROM games
                        ORDER BY ROWID DESC LIMIT ?
                    """, (sample_size,))
                elif has_winner:
                    cursor.execute("""
                        SELECT winner, 0 as total_moves FROM games
                        ORDER BY ROWID DESC LIMIT ?
                    """, (sample_size,))
                else:
                    conn.close()
                    continue

                for row in cursor.fetchall():
                    total_games += 1
                    winner = row['winner']
                    moves = row['total_moves']

                    # Count draws (winner = -1 or winner is NULL typically means draw)
                    if winner is None or winner == -1:
                        draws += 1

                    # Count timeouts (games hitting move limits)
                    if moves >= 500:  # Common move limits
                        timeouts += 1

                    if moves > 0:
                        game_lengths.append(moves)

                conn.close()
            except Exception as e:
                print(f"[DataCollector] Quality stats error for {db_path}: {e}")
                continue

        return {
            "draw_rate": draws / total_games if total_games > 0 else 0,
            "timeout_rate": timeouts / total_games if total_games > 0 else 0,
            "game_lengths": game_lengths,
            "total_sampled": total_games,
        }

    async def run_collection_cycle(self) -> int:
        """Run one data collection cycle across all hosts.

        OPTIMIZED: Uses a fast parallel pre-query phase to identify hosts with new data,
        then only syncs those hosts. This reduces cycle time significantly.
        """
        print(f"[DataCollector] Starting collection cycle for {len(self.state.hosts)} hosts...", flush=True)

        # Check disk capacity before syncing
        has_capacity, disk_percent = check_disk_has_capacity()
        if not has_capacity:
            print(f"[DataCollector] SKIPPING SYNC - Disk usage {disk_percent:.1f}% exceeds limit {MAX_DISK_USAGE_PERCENT}%")
            return 0

        enabled_hosts = [h for h in self.state.hosts.values() if h.enabled]
        if not enabled_hosts:
            return 0

        # Phase 1: Fast parallel pre-query to get game counts (5s timeout per host)
        host_counts = await self._fast_parallel_query(enabled_hosts)

        # Phase 2: Only sync hosts that have new data
        hosts_with_new_data = []
        for host in enabled_hosts:
            current_count = host_counts.get(host.name, 0)
            if current_count > 0:
                new_games = max(0, current_count - host.last_game_count)
                if new_games >= self.config.min_games_per_sync:
                    hosts_with_new_data.append((host, current_count, new_games))

        if hosts_with_new_data:
            print(f"[DataCollector] {len(hosts_with_new_data)} hosts have new data to sync")

        # Phase 3: Sync hosts with new data (increased concurrency for faster ingestion)
        total_new = 0
        max_concurrent_syncs = 1  # Limited to 1 to prevent OOM from parallel rsyncs

        semaphore = asyncio.Semaphore(max_concurrent_syncs)

        async def sync_with_limit(host: HostState, current_count: int, new_games: int) -> int:
            async with semaphore:
                return await self._sync_host_data(host, current_count, new_games)

        if hosts_with_new_data:
            tasks = [
                sync_with_limit(host, count, new_games)
                for host, count, new_games in hosts_with_new_data
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for result in results:
                if isinstance(result, int):
                    total_new += result

        self.state.total_data_syncs += 1
        self.state.total_games_pending += total_new

        # Update per-config game counts from synced databases
        if total_new > 0:
            self._update_per_config_game_counts(total_new)

        return total_new

    async def _fast_parallel_query(self, hosts: list[HostState]) -> dict[str, int]:
        """Query all hosts in parallel with short timeout to get game counts.

        This is much faster than querying each host sequentially during sync_host.
        Uses a 5-second timeout to quickly identify unreachable hosts.

        Returns:
            Dict mapping host name to game count (0 for failed queries)
        """
        async def query_host_count(host: HostState) -> tuple[str, int]:
            # Circuit breaker check
            if HAS_CIRCUIT_BREAKER:
                breaker = get_host_breaker()
                if not breaker.can_execute(host.ssh_host):
                    return (host.name, 0)

            try:
                ssh_target = f"{host.ssh_user}@{host.ssh_host}"
                port_arg = f"-p {host.ssh_port}" if host.ssh_port != 22 else ""

                # Use base64 encoding to avoid shell quoting issues completely
                import base64
                python_script = (
                    "import sqlite3, glob, os; "
                    "os.chdir(os.path.expanduser('~/ringrift/ai-service')); "
                    "dbs=glob.glob('data/games/*.db'); "
                    "total=0; "
                    "[total := total + sqlite3.connect(db).execute('SELECT COUNT(*) FROM games').fetchone()[0] for db in dbs if 'schema' not in db]; "
                    "print(total)"
                )
                encoded_script = base64.b64encode(python_script.encode()).decode()
                cmd = f'ssh -o ConnectTimeout=5 -o BatchMode=yes {port_arg} {ssh_target} "python3 -c \'import base64; exec(base64.b64decode(\\"{encoded_script}\\").decode())\'" 2>/dev/null'

                result = await asyncio.create_subprocess_shell(
                    cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                stdout, _ = await asyncio.wait_for(result.communicate(), timeout=8)

                count = int(stdout.decode().strip() or "0")
                return (host.name, count)

            except (asyncio.TimeoutError, ValueError, OSError, UnicodeDecodeError):
                return (host.name, 0)

        # Run all queries in parallel
        start_time = time.time()
        tasks = [query_host_count(h) for h in hosts]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        elapsed = time.time() - start_time
        successful = sum(1 for r in results if isinstance(r, tuple) and r[1] > 0)
        print(f"[DataCollector] Pre-query completed in {elapsed:.1f}s ({successful}/{len(hosts)} hosts responded)")

        # Convert to dict
        counts = {}
        for result in results:
            if isinstance(result, tuple):
                counts[result[0]] = result[1]
        return counts

    async def _sync_host_data(self, host: HostState, current_count: int, new_games: int) -> int:
        """Sync data from a host that has new games.

        This is called after _fast_parallel_query has confirmed the host has new data.
        Uses unified_manifest for sync history logging.
        """
        # Re-check disk capacity before each sync (secondary safety)
        has_capacity, disk_percent = check_disk_has_capacity()
        if not has_capacity:
            print(f"[DataCollector] {host.name}: SKIPPING - Disk {disk_percent:.1f}% >= {MAX_DISK_USAGE_PERCENT}%")
            return 0

        print(f"[DataCollector] {host.name}: {current_count} games (last: {host.last_game_count}, new: {new_games})")

        sync_start = time.time()
        try:
            # Trigger rsync for incremental sync
            if self.config.sync_method == "incremental":
                await self._incremental_sync(host)
            else:
                await self._full_sync(host)

            host.last_game_count = current_count
            host.last_sync_time = time.time()

            # Manifest: Log successful sync
            if self._manifest and HAS_UNIFIED_MANIFEST:
                try:
                    duration = time.time() - sync_start
                    self._manifest.log_sync(host.name, new_games, duration, success=True, sync_method=self.config.sync_method)
                except (sqlite3.Error, OSError):
                    pass  # Non-fatal

            # Publish event
            await self.event_bus.publish(DataEvent(
                event_type=DataEventType.NEW_GAMES_AVAILABLE,
                payload={
                    "host": host.name,
                    "new_games": new_games,
                    "total_games": current_count,
                }
            ))

            host.consecutive_failures = 0
            # Record success with circuit breaker
            if HAS_CIRCUIT_BREAKER:
                get_host_breaker().record_success(host.ssh_host)
            return new_games

        except Exception as e:
            host.consecutive_failures += 1
            # Record failure with circuit breaker
            if HAS_CIRCUIT_BREAKER:
                get_host_breaker().record_failure(host.ssh_host, e)
            # Manifest: Log failed sync
            if self._manifest and HAS_UNIFIED_MANIFEST:
                try:
                    duration = time.time() - sync_start
                    self._manifest.log_sync(host.name, 0, duration, success=False, error_message=str(e))
                except (sqlite3.Error, OSError):
                    pass  # Non-fatal
            print(f"[DataCollector] Failed to sync {host.name}: {e}")
            return 0

    def _update_per_config_game_counts(self, new_games: int) -> None:
        """Update per-config games_since_training counters.

        Distributes new games across configs based on what's in synced databases.
        Falls back to proportional distribution if db parsing fails.
        """
        # Count games from multiple sources:
        # 1. Synced DB files: data/games/synced/*.db
        # 2. GPU selfplay JSONL: data/games/gpu_selfplay/{config}/games.jsonl
        config_counts: dict[str, int] = {}
        total_counted = 0

        # Source 1: Synced DB files
        synced_dir = AI_SERVICE_ROOT / "data" / "games" / "synced"
        if synced_dir.exists():
            for db_path in synced_dir.rglob("*.db"):
                try:
                    db_name = db_path.stem.lower()
                    # Parse config from filename patterns like "selfplay_square8_2p.db"
                    config_key = None
                    for ck in self.state.configs:
                        if ck.replace("_", "") in db_name.replace("_", "") or ck in db_name:
                            config_key = ck
                            break

                    if config_key:
                        conn = sqlite3.connect(str(db_path))
                        cursor = conn.cursor()
                        cursor.execute("SELECT COUNT(*) FROM games")
                        count = cursor.fetchone()[0]
                        conn.close()
                        config_counts[config_key] = config_counts.get(config_key, 0) + count
                        total_counted += count
                except (sqlite3.Error, OSError, ValueError, TypeError):
                    pass

        # Source 2: GPU selfplay JSONL files (primary source for GPU-generated games)
        gpu_selfplay_dir = AI_SERVICE_ROOT / "data" / "games" / "gpu_selfplay"
        if gpu_selfplay_dir.exists():
            for config_dir in gpu_selfplay_dir.iterdir():
                if config_dir.is_dir():
                    # Directory name is the config key (e.g., "hexagonal_3p", "square19_4p")
                    config_key = config_dir.name
                    if config_key in self.state.configs:
                        jsonl_path = config_dir / "games.jsonl"
                        if jsonl_path.exists():
                            try:
                                if jsonl_path.stat().st_size == 0:
                                    print(f"[DataCollector] GPU selfplay JSONL empty: {jsonl_path}")
                                    continue
                                # Count lines = count games
                                with open(jsonl_path) as f:
                                    count = sum(1 for _ in f)
                                config_counts[config_key] = config_counts.get(config_key, 0) + count
                                total_counted += count
                            except (OSError, IOError):
                                pass

        # Source 3: Tournament JSONL files (tier/Elo tournaments with training data)
        tournaments_dir = AI_SERVICE_ROOT / "data" / "tournaments"
        if tournaments_dir.exists():
            for jsonl_path in tournaments_dir.glob("*.jsonl"):
                try:
                    # Parse config from filename (e.g., "diverse_square8_2p_*.jsonl")
                    name = jsonl_path.stem
                    parts = name.split("_")
                    # Find board type and player count
                    config_key = None
                    for i, part in enumerate(parts):
                        if part in ("square8", "square19", "hexagonal", "hex8"):
                            if i + 1 < len(parts) and parts[i + 1].endswith("p"):
                                config_key = f"{part}_{parts[i + 1]}"
                                break
                    if config_key and config_key in self.state.configs:
                        with open(jsonl_path) as f:
                            count = sum(1 for _ in f)
                        config_counts[config_key] = config_counts.get(config_key, 0) + count
                        total_counted += count
                except (OSError, IOError):
                    pass

        # Source 4: Synced tournament data from remote hosts
        synced_tournaments_dir = AI_SERVICE_ROOT / "data" / "tournaments" / "synced"
        if synced_tournaments_dir.exists():
            for jsonl_path in synced_tournaments_dir.rglob("*.jsonl"):
                try:
                    name = jsonl_path.stem
                    parts = name.split("_")
                    config_key = None
                    for i, part in enumerate(parts):
                        if part in ("square8", "square19", "hexagonal", "hex8"):
                            if i + 1 < len(parts) and parts[i + 1].endswith("p"):
                                config_key = f"{part}_{parts[i + 1]}"
                                break
                    if config_key and config_key in self.state.configs:
                        with open(jsonl_path) as f:
                            count = sum(1 for _ in f)
                        config_counts[config_key] = config_counts.get(config_key, 0) + count
                        total_counted += count
                except (OSError, IOError):
                    pass

        # Distribute new games proportionally based on existing counts
        if total_counted > 0:
            for config_key, count in config_counts.items():
                if config_key in self.state.configs:
                    proportion = count / total_counted
                    added = int(new_games * proportion)
                    self.state.configs[config_key].games_since_training += added

            # Update data-aware config weights for balanced selfplay distribution
            if HAS_RESOURCE_OPTIMIZER and update_config_weights is not None:
                try:
                    new_weights = update_config_weights(config_counts)
                    # Update Prometheus metrics for config weights
                    if HAS_PROMETHEUS:
                        for ck, weight in new_weights.items():
                            CONFIG_WEIGHT.labels(config_key=ck).set(weight)
                except (AttributeError, KeyError, ValueError):
                    pass  # Non-critical
        else:
            # Fallback: distribute to square8_2p
            if "square8_2p" in self.state.configs:
                self.state.configs["square8_2p"].games_since_training += new_games
