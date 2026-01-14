"""Data Availability Daemon - Training nodes actively pull missing data.

This daemon runs on training-capable nodes to ensure they have the data
they need for pending training jobs. It discovers data from multiple sources
and pulls based on priority order.

Architecture:
    1. Query work queue for pending training jobs
    2. Check local data availability
    3. Find missing data from sources in priority order:
       - P2P peers (fastest, local network)
       - S3 (reliable, moderate speed)
       - OWC via coordinator (slowest but comprehensive)
    4. Pull missing data and emit events

Usage:
    # Via daemon manager
    from app.coordination.daemon_manager import get_daemon_manager
    dm = get_daemon_manager()
    await dm.start(DaemonType.DATA_AVAILABILITY)

    # Direct usage
    daemon = DataAvailabilityDaemon()
    await daemon.start()

Configuration:
    RINGRIFT_DATA_AVAILABILITY_ENABLED - Enable daemon (default: true on training nodes)
    RINGRIFT_DATA_AVAILABILITY_INTERVAL - Check interval in seconds (default: 300)
    RINGRIFT_S3_PULL_ENABLED - Allow S3 pulls (default: true)
    RINGRIFT_OWC_PULL_ENABLED - Allow OWC pulls via coordinator (default: true)

January 2026: Created as part of distributed architecture Phase 2.
"""

from __future__ import annotations

import asyncio
import logging
import os
import subprocess
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class DataSourcePriority(Enum):
    """Priority order for data sources."""

    P2P = 1      # Fastest - local network peers
    S3 = 2       # Reliable - cloud storage
    OWC = 3      # Comprehensive - external storage via coordinator
    NONE = 99    # Not available


@dataclass
class DataSource:
    """Information about a data source."""

    source_type: DataSourcePriority
    location: str  # peer_id, s3://path, or owc://path
    priority: int = field(default=99)
    last_seen: float = 0.0
    size_bytes: int = 0

    def __lt__(self, other: DataSource) -> bool:
        return self.priority < other.priority


@dataclass
class DataAvailabilityConfig:
    """Configuration for data availability daemon."""

    # Enabled by default on training nodes
    enabled: bool = field(
        default_factory=lambda: os.getenv(
            "RINGRIFT_DATA_AVAILABILITY_ENABLED", "true"
        ).lower() == "true"
    )

    # Check interval (default: 5 minutes)
    check_interval_seconds: float = field(
        default_factory=lambda: float(
            os.getenv("RINGRIFT_DATA_AVAILABILITY_INTERVAL", "300")
        )
    )

    # Source settings
    s3_pull_enabled: bool = field(
        default_factory=lambda: os.getenv("RINGRIFT_S3_PULL_ENABLED", "true").lower()
        == "true"
    )
    owc_pull_enabled: bool = field(
        default_factory=lambda: os.getenv("RINGRIFT_OWC_PULL_ENABLED", "true").lower()
        == "true"
    )
    p2p_pull_enabled: bool = field(
        default_factory=lambda: os.getenv("RINGRIFT_P2P_PULL_ENABLED", "true").lower()
        == "true"
    )

    # S3 settings
    s3_bucket: str = field(
        default_factory=lambda: os.getenv(
            "RINGRIFT_S3_BUCKET", "ringrift-models-20251214"
        )
    )

    # OWC settings (via coordinator)
    owc_coordinator: str = field(
        default_factory=lambda: os.getenv("RINGRIFT_OWC_COORDINATOR", "mac-studio")
    )
    owc_path: str = field(
        default_factory=lambda: os.getenv(
            "RINGRIFT_OWC_PATH", "/Volumes/RingRift-Data"
        )
    )

    # Local paths
    data_dir: Path = field(
        default_factory=lambda: Path(os.getenv("RINGRIFT_DATA_DIR", "data"))
    )
    models_dir: Path = field(
        default_factory=lambda: Path(os.getenv("RINGRIFT_MODELS_DIR", "models"))
    )
    training_dir: Path = field(
        default_factory=lambda: Path(
            os.getenv("RINGRIFT_TRAINING_DIR", "data/training")
        )
    )

    # Timeout settings
    pull_timeout_seconds: float = 300.0


@dataclass
class DataRequirement:
    """A data requirement for training."""

    data_type: str  # "npz", "model", "db"
    config_key: str  # e.g., "hex8_2p"
    file_pattern: str  # e.g., "hex8_2p.npz", "canonical_hex8_2p.pth"
    required_by: str = ""  # job_id or "training"
    priority: int = 1  # 1=high, 2=medium, 3=low


try:
    from app.coordination.handler_base import HandlerBase
    from app.coordination.contracts import HealthCheckResult

    HAS_HANDLER_BASE = True
except ImportError:
    HAS_HANDLER_BASE = False


class DataAvailabilityDaemon:
    """Training nodes actively pull missing data.

    This daemon ensures training nodes have the data they need by:
    1. Discovering what data is required for pending jobs
    2. Checking local availability
    3. Pulling from sources in priority order

    January 2026: Created for distributed architecture Phase 2.
    """

    DAEMON_TYPE = "DATA_AVAILABILITY"
    DEFAULT_INTERVAL = 300.0  # 5 minutes

    def __init__(self, config: DataAvailabilityConfig | None = None):
        """Initialize data availability daemon."""
        self.config = config or DataAvailabilityConfig()
        self._running = False
        self._task: asyncio.Task | None = None

        # Statistics
        self._last_check_time: float = 0.0
        self._requirements_found: int = 0
        self._data_pulled: int = 0
        self._pull_errors: int = 0
        self._sources_queried: int = 0

        # Node identity
        self._node_id = self._get_node_id()

        # Cached P2P client
        self._p2p_client: Any = None

    def _get_node_id(self) -> str:
        """Get node identifier."""
        try:
            from app.config.node_identity import get_node_id_safe

            return get_node_id_safe()
        except ImportError:
            import socket

            return os.getenv("RINGRIFT_NODE_ID", socket.gethostname())

    async def start(self) -> None:
        """Start the daemon."""
        if not self.config.enabled:
            logger.info(
                f"DataAvailabilityDaemon disabled on {self._node_id}"
            )
            return

        logger.info(f"DataAvailabilityDaemon starting on {self._node_id}")
        self._running = True
        self._task = asyncio.create_task(self._run_loop())

    async def stop(self) -> None:
        """Stop the daemon."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("DataAvailabilityDaemon stopped")

    async def _run_loop(self) -> None:
        """Main daemon loop."""
        while self._running:
            try:
                pulled = await self._run_cycle()
                logger.info(f"DataAvailabilityDaemon cycle complete: pulled {pulled} items")
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"DataAvailabilityDaemon error: {e}")
                self._pull_errors += 1

            await asyncio.sleep(self.config.check_interval_seconds)

    async def _run_cycle(self) -> int:
        """Run one availability check and pull cycle.

        Returns:
            Number of items successfully pulled
        """
        self._last_check_time = time.time()

        # 1. What do we need for pending training jobs?
        requirements = await self._get_training_requirements()
        self._requirements_found = len(requirements)

        if not requirements:
            logger.debug("No training requirements found")
            return 0

        # 2. What do we have locally?
        local_data = await self._scan_local_data()

        # 3. Find missing data
        missing = self._find_missing_data(requirements, local_data)
        if not missing:
            logger.debug("All required data available locally")
            return 0

        logger.info(f"Found {len(missing)} missing data items: {[m.file_pattern for m in missing]}")

        # 4. Pull missing data from sources
        pulled = 0
        for req in missing:
            sources = await self._find_data_sources(req)
            self._sources_queried += len(sources)

            for source in sorted(sources):
                success = await self._pull_from_source(req, source)
                if success:
                    pulled += 1
                    self._data_pulled += 1
                    self._emit_data_available_event(req)
                    break

        return pulled

    async def _get_training_requirements(self) -> list[DataRequirement]:
        """Get data requirements for pending training jobs.

        Checks the work queue and training config for what data is needed.
        """
        requirements: list[DataRequirement] = []

        # Check work queue for pending training jobs
        try:
            from app.coordination.work_queue import WorkQueue

            queue = WorkQueue.get_instance()
            pending_jobs = queue.get_pending_jobs(job_type="training")

            for job in pending_jobs:
                config_key = job.get("config_key", "")
                if config_key:
                    # NPZ training data
                    requirements.append(
                        DataRequirement(
                            data_type="npz",
                            config_key=config_key,
                            file_pattern=f"{config_key}.npz",
                            required_by=job.get("job_id", ""),
                            priority=1,
                        )
                    )
                    # Model for fine-tuning (if exists)
                    requirements.append(
                        DataRequirement(
                            data_type="model",
                            config_key=config_key,
                            file_pattern=f"canonical_{config_key}.pth",
                            required_by=job.get("job_id", ""),
                            priority=2,
                        )
                    )
        except Exception as e:
            logger.debug(f"Could not check work queue: {e}")

        # Also check for canonical configs that need models
        try:
            canonical_configs = [
                "hex8_2p", "hex8_3p", "hex8_4p",
                "square8_2p", "square8_3p", "square8_4p",
                "square19_2p", "square19_3p", "square19_4p",
                "hexagonal_2p", "hexagonal_3p", "hexagonal_4p",
            ]
            for config_key in canonical_configs:
                # Check if we're missing canonical model
                model_path = self.config.models_dir / f"canonical_{config_key}.pth"
                if not model_path.exists():
                    requirements.append(
                        DataRequirement(
                            data_type="model",
                            config_key=config_key,
                            file_pattern=f"canonical_{config_key}.pth",
                            required_by="canonical",
                            priority=3,
                        )
                    )
        except Exception as e:
            logger.debug(f"Could not check canonical configs: {e}")

        return requirements

    async def _scan_local_data(self) -> dict[str, Path]:
        """Scan local data directories.

        Returns:
            Dict mapping file patterns to local paths
        """
        local_data: dict[str, Path] = {}

        # Scan training data (NPZ files)
        if self.config.training_dir.exists():
            for npz_file in self.config.training_dir.glob("*.npz"):
                local_data[npz_file.name] = npz_file

        # Scan models
        if self.config.models_dir.exists():
            for model_file in self.config.models_dir.glob("*.pth"):
                local_data[model_file.name] = model_file

        # Scan game databases
        games_dir = self.config.data_dir / "games"
        if games_dir.exists():
            for db_file in games_dir.glob("*.db"):
                local_data[db_file.name] = db_file

        return local_data

    def _find_missing_data(
        self, requirements: list[DataRequirement], local_data: dict[str, Path]
    ) -> list[DataRequirement]:
        """Find requirements not satisfied by local data."""
        missing = []
        for req in requirements:
            if req.file_pattern not in local_data:
                missing.append(req)
        return missing

    async def _find_data_sources(self, req: DataRequirement) -> list[DataSource]:
        """Find all locations for required data, in priority order."""
        sources: list[DataSource] = []

        # Priority 1: P2P peers (fastest)
        if self.config.p2p_pull_enabled:
            p2p_sources = await self._find_p2p_sources(req)
            sources.extend(p2p_sources)

        # Priority 2: S3 (reliable, moderate speed)
        if self.config.s3_pull_enabled:
            s3_source = await self._find_s3_source(req)
            if s3_source:
                sources.append(s3_source)

        # Priority 3: OWC via coordinator (slowest but comprehensive)
        if self.config.owc_pull_enabled:
            owc_source = await self._find_owc_source(req)
            if owc_source:
                sources.append(owc_source)

        return sources

    async def _find_p2p_sources(self, req: DataRequirement) -> list[DataSource]:
        """Find P2P peers that have the required data."""
        sources: list[DataSource] = []

        try:
            # Query P2P for data manifests
            import aiohttp

            async with aiohttp.ClientSession() as session:
                # Query local P2P orchestrator for peer manifests
                url = "http://localhost:8770/data/manifests"
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        for peer_id, manifest in data.get("manifests", {}).items():
                            files = manifest.get("files", {})
                            if req.file_pattern in files:
                                sources.append(
                                    DataSource(
                                        source_type=DataSourcePriority.P2P,
                                        location=peer_id,
                                        priority=1,
                                        size_bytes=files[req.file_pattern].get("size", 0),
                                    )
                                )
        except Exception as e:
            logger.debug(f"P2P source discovery failed: {e}")

        return sources

    async def _find_s3_source(self, req: DataRequirement) -> DataSource | None:
        """Check if data exists in S3."""
        try:
            # Determine S3 path based on data type
            if req.data_type == "npz":
                s3_path = f"consolidated/training/{req.file_pattern}"
            elif req.data_type == "model":
                s3_path = f"consolidated/models/{req.file_pattern}"
            else:
                s3_path = f"consolidated/{req.data_type}/{req.file_pattern}"

            # Check if file exists in S3
            cmd = [
                "aws", "s3", "ls",
                f"s3://{self.config.s3_bucket}/{s3_path}",
            ]
            result = await asyncio.wait_for(
                asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                ),
                timeout=30,
            )
            stdout, _ = await result.communicate()

            if result.returncode == 0 and stdout:
                # Parse size from ls output
                parts = stdout.decode().strip().split()
                size = int(parts[2]) if len(parts) > 2 else 0

                return DataSource(
                    source_type=DataSourcePriority.S3,
                    location=f"s3://{self.config.s3_bucket}/{s3_path}",
                    priority=2,
                    size_bytes=size,
                )
        except Exception as e:
            logger.debug(f"S3 source check failed for {req.file_pattern}: {e}")

        return None

    async def _find_owc_source(self, req: DataRequirement) -> DataSource | None:
        """Check if data exists on OWC via coordinator."""
        try:
            # Determine OWC path based on data type
            if req.data_type == "npz":
                owc_subdir = "canonical_data"
            elif req.data_type == "model":
                owc_subdir = "canonical_models"
            else:
                owc_subdir = req.data_type

            owc_path = f"{self.config.owc_path}/{owc_subdir}/{req.file_pattern}"

            # Query coordinator for file existence via SSH
            cmd = [
                "ssh", "-o", "ConnectTimeout=10", "-o", "StrictHostKeyChecking=no",
                f"armand@{self.config.owc_coordinator}",
                f"test -f '{owc_path}' && stat -f%z '{owc_path}' 2>/dev/null || stat --format=%s '{owc_path}' 2>/dev/null",
            ]

            result = await asyncio.wait_for(
                asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                ),
                timeout=30,
            )
            stdout, _ = await result.communicate()

            if result.returncode == 0:
                size = int(stdout.decode().strip() or "0")
                return DataSource(
                    source_type=DataSourcePriority.OWC,
                    location=f"owc://{self.config.owc_coordinator}{owc_path}",
                    priority=3,
                    size_bytes=size,
                )
        except Exception as e:
            logger.debug(f"OWC source check failed for {req.file_pattern}: {e}")

        return None

    async def _pull_from_source(
        self, req: DataRequirement, source: DataSource
    ) -> bool:
        """Pull data from a specific source.

        Returns:
            True if pull succeeded
        """
        logger.info(f"Pulling {req.file_pattern} from {source.source_type.name}: {source.location}")

        try:
            if source.source_type == DataSourcePriority.P2P:
                return await self._pull_from_p2p(req, source)
            elif source.source_type == DataSourcePriority.S3:
                return await self._pull_from_s3(req, source)
            elif source.source_type == DataSourcePriority.OWC:
                return await self._pull_from_owc(req, source)
        except Exception as e:
            logger.warning(f"Pull failed from {source.location}: {e}")
            return False

        return False

    async def _pull_from_p2p(
        self, req: DataRequirement, source: DataSource
    ) -> bool:
        """Pull data from P2P peer."""
        try:
            import aiohttp

            peer_id = source.location
            url = f"http://{peer_id}:8770/data/download/{req.file_pattern}"

            # Determine local path
            if req.data_type == "npz":
                local_path = self.config.training_dir / req.file_pattern
            elif req.data_type == "model":
                local_path = self.config.models_dir / req.file_pattern
            else:
                local_path = self.config.data_dir / req.data_type / req.file_pattern

            local_path.parent.mkdir(parents=True, exist_ok=True)

            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url, timeout=aiohttp.ClientTimeout(total=self.config.pull_timeout_seconds)
                ) as resp:
                    if resp.status == 200:
                        with open(local_path, "wb") as f:
                            async for chunk in resp.content.iter_chunked(8192):
                                f.write(chunk)
                        logger.info(f"Pulled {req.file_pattern} from P2P peer {peer_id}")
                        return True
        except Exception as e:
            logger.debug(f"P2P pull failed: {e}")

        return False

    async def _pull_from_s3(
        self, req: DataRequirement, source: DataSource
    ) -> bool:
        """Pull data from S3."""
        try:
            # Determine local path
            if req.data_type == "npz":
                local_path = self.config.training_dir / req.file_pattern
            elif req.data_type == "model":
                local_path = self.config.models_dir / req.file_pattern
            else:
                local_path = self.config.data_dir / req.data_type / req.file_pattern

            local_path.parent.mkdir(parents=True, exist_ok=True)

            cmd = ["aws", "s3", "cp", source.location, str(local_path)]

            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            _, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=self.config.pull_timeout_seconds,
            )

            if process.returncode == 0:
                logger.info(f"Pulled {req.file_pattern} from S3")
                return True
            else:
                logger.warning(f"S3 pull failed: {stderr.decode()}")

        except asyncio.TimeoutError:
            logger.warning(f"S3 pull timed out for {req.file_pattern}")
        except Exception as e:
            logger.debug(f"S3 pull error: {e}")

        return False

    async def _pull_from_owc(
        self, req: DataRequirement, source: DataSource
    ) -> bool:
        """Pull data from OWC via coordinator."""
        try:
            # Parse OWC path from location
            # Format: owc://coordinator/path/to/file
            parts = source.location.replace("owc://", "").split("/", 1)
            if len(parts) != 2:
                return False

            coordinator, remote_path = parts[0], "/" + parts[1]

            # Determine local path
            if req.data_type == "npz":
                local_path = self.config.training_dir / req.file_pattern
            elif req.data_type == "model":
                local_path = self.config.models_dir / req.file_pattern
            else:
                local_path = self.config.data_dir / req.data_type / req.file_pattern

            local_path.parent.mkdir(parents=True, exist_ok=True)

            # Use rsync for large files
            cmd = [
                "rsync", "-avz", "--progress",
                f"armand@{coordinator}:{remote_path}",
                str(local_path),
            ]

            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            _, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=self.config.pull_timeout_seconds,
            )

            if process.returncode == 0:
                logger.info(f"Pulled {req.file_pattern} from OWC via {coordinator}")
                return True
            else:
                logger.warning(f"OWC pull failed: {stderr.decode()}")

        except asyncio.TimeoutError:
            logger.warning(f"OWC pull timed out for {req.file_pattern}")
        except Exception as e:
            logger.debug(f"OWC pull error: {e}")

        return False

    def _emit_data_available_event(self, req: DataRequirement) -> None:
        """Emit event when data becomes available."""
        try:
            from app.coordination.event_router import emit_event
            from app.coordination.data_events import DataEventType

            emit_event(
                DataEventType.DATA_SYNC_COMPLETED,
                {
                    "node_id": self._node_id,
                    "data_type": req.data_type,
                    "config_key": req.config_key,
                    "file": req.file_pattern,
                    "source": "data_availability_daemon",
                },
            )
        except ImportError:
            pass

    def health_check(self) -> dict[str, Any]:
        """Return health check result."""
        now = time.time()
        age = now - self._last_check_time if self._last_check_time > 0 else float("inf")
        healthy = age < self.config.check_interval_seconds * 2

        return {
            "healthy": healthy,
            "status": "healthy" if healthy else "stale",
            "details": {
                "node_id": self._node_id,
                "last_check_age_seconds": round(age, 1),
                "requirements_found": self._requirements_found,
                "data_pulled": self._data_pulled,
                "pull_errors": self._pull_errors,
                "sources_queried": self._sources_queried,
                "config": {
                    "s3_enabled": self.config.s3_pull_enabled,
                    "owc_enabled": self.config.owc_pull_enabled,
                    "p2p_enabled": self.config.p2p_pull_enabled,
                },
            },
        }


# Factory function for daemon manager
def create_data_availability_daemon() -> DataAvailabilityDaemon:
    """Create a DataAvailabilityDaemon instance."""
    return DataAvailabilityDaemon()


# If HandlerBase is available, create a HandlerBase subclass
if HAS_HANDLER_BASE:

    class DataAvailabilityHandler(HandlerBase):
        """HandlerBase wrapper for DataAvailabilityDaemon."""

        def __init__(self):
            super().__init__(
                name="data_availability",
                cycle_interval=300.0,
            )
            self._daemon = DataAvailabilityDaemon()

        async def _run_cycle(self) -> None:
            """Run one availability check cycle."""
            await self._daemon._run_cycle()

        def _get_event_subscriptions(self) -> dict:
            """Get event subscriptions."""
            return {
                "TRAINING_SCHEDULED": self._on_training_scheduled,
            }

        async def _on_training_scheduled(self, event: dict) -> None:
            """Handle training scheduled event - check data availability."""
            # Trigger immediate check when training is scheduled
            await self._daemon._run_cycle()

        def health_check(self) -> HealthCheckResult:
            """Return health check result."""
            result = self._daemon.health_check()
            return HealthCheckResult(
                healthy=result["healthy"],
                status=result["status"],
                details=result["details"],
            )
