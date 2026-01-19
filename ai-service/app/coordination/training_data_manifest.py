"""Training Data Manifest - Tracks available training data across all sources.

This module provides a unified view of training data available across:
- Local disk (data/training/*.npz)
- OWC external drive (mac-studio:/Volumes/RingRift-Data)
- Amazon S3 (s3://ringrift-models-20251214/consolidated/training/)

The manifest enables:
- Pre-training data sync to maximize training data availability
- Automatic discovery of the best/largest dataset for each config
- Background refresh to keep the manifest up-to-date

Usage:
    from app.coordination.training_data_manifest import (
        TrainingDataManifest,
        get_training_data_manifest,
        DataSource,
        TrainingDataEntry,
    )

    # Get the singleton manifest
    manifest = get_training_data_manifest()

    # Refresh data from all sources
    await manifest.refresh_all()

    # Get best data for a config
    best = manifest.get_best_data("hex8_2p")
    print(f"Best source: {best.source}, Size: {best.size_mb}MB, Path: {best.path}")

    # Get all available data for a config
    all_data = manifest.get_all_data("hex8_2p")
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shlex
import subprocess
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

from app.coordination.event_utils import make_config_key

# Dec 30, 2025: Centralized SSH configuration
try:
    from app.config.coordination_defaults import build_ssh_options
    SSH_CONFIG_AVAILABLE = True
except ImportError:
    SSH_CONFIG_AVAILABLE = False
    build_ssh_options = None  # type: ignore

logger = logging.getLogger(__name__)


class DataSource(str, Enum):
    """Available data sources."""

    LOCAL = "local"
    OWC = "owc"  # Mac-studio external drive
    S3 = "s3"


@dataclass
class TrainingDataEntry:
    """A training data file entry."""

    config_key: str  # e.g., "hex8_2p"
    source: DataSource
    path: str  # Full path (local), OWC path, or S3 URI
    size_bytes: int
    modified_time: datetime | None = None
    sample_count: int | None = None  # If known from NPZ metadata
    version: str | None = None  # Model version (v2, v3, v4, etc.)
    quality_score: float | None = None  # 0-1 quality metric
    newest_game_time: datetime | None = None  # Timestamp of newest game in dataset

    @property
    def size_mb(self) -> float:
        """Size in megabytes."""
        return self.size_bytes / (1024 * 1024)

    @property
    def size_gb(self) -> float:
        """Size in gigabytes."""
        return self.size_bytes / (1024 * 1024 * 1024)

    @property
    def age_hours(self) -> float | None:
        """Age in hours since newest game (or file modified time as fallback)."""
        ref_time = self.newest_game_time or self.modified_time
        if not ref_time:
            return None
        # Ensure ref_time is timezone-aware (fix for "can't subtract offset-naive
        # and offset-aware datetimes" error when deserializing old timestamps)
        if ref_time.tzinfo is None:
            ref_time = ref_time.replace(tzinfo=timezone.utc)
        return (datetime.now(tz=timezone.utc) - ref_time).total_seconds() / 3600

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "config_key": self.config_key,
            "source": self.source.value,
            "path": self.path,
            "size_bytes": self.size_bytes,
            "modified_time": (
                self.modified_time.isoformat() if self.modified_time else None
            ),
            "sample_count": self.sample_count,
            "version": self.version,
            "quality_score": self.quality_score,
            "newest_game_time": (
                self.newest_game_time.isoformat() if self.newest_game_time else None
            ),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TrainingDataEntry":
        """Deserialize from dictionary."""
        return cls(
            config_key=data["config_key"],
            source=DataSource(data["source"]),
            path=data["path"],
            size_bytes=data["size_bytes"],
            modified_time=(
                datetime.fromisoformat(data["modified_time"])
                if data.get("modified_time")
                else None
            ),
            sample_count=data.get("sample_count"),
            version=data.get("version"),
            quality_score=data.get("quality_score"),
            newest_game_time=(
                datetime.fromisoformat(data["newest_game_time"])
                if data.get("newest_game_time")
                else None
            ),
        )


# OWC drive configuration
OWC_HOST = os.environ.get("RINGRIFT_OWC_HOST", "100.107.168.125")
OWC_USER = os.environ.get("RINGRIFT_OWC_USER", "armand")
OWC_BASE_PATH = os.environ.get(
    "RINGRIFT_OWC_PATH", "/Volumes/RingRift-Data"
)
OWC_SSH_KEY = os.environ.get("RINGRIFT_OWC_SSH_KEY", "~/.ssh/id_ed25519")

# S3 configuration
S3_BUCKET = os.environ.get("RINGRIFT_S3_BUCKET", "ringrift-models-20251214")
S3_TRAINING_PREFIX = os.environ.get(
    "RINGRIFT_S3_TRAINING_PREFIX", "consolidated/training/"
)

# Local data paths
LOCAL_TRAINING_DIR = Path(
    os.environ.get("RINGRIFT_TRAINING_DIR", "data/training")
)

# Cache file for manifest persistence
MANIFEST_CACHE_PATH = Path(
    os.environ.get("RINGRIFT_MANIFEST_CACHE", "/tmp/ringrift_training_manifest.json")
)


# Config key patterns to extract from filenames
CONFIG_KEY_PATTERNS = [
    # Standard patterns: hex8_2p, square8_3p, hexagonal_4p, square19_2p
    r"(hex8|square8|square19|hexagonal)_([234])p",
    # Alternative patterns with underscores
    r"(hex8|sq8|sq19|hexagonal)_([234])p",
]


def extract_config_key(filename: str) -> str | None:
    """Extract config key (e.g., 'hex8_2p') from a filename."""
    # Normalize common abbreviations
    name = filename.lower()
    name = name.replace("sq8", "square8").replace("sq19", "square19")

    for pattern in CONFIG_KEY_PATTERNS:
        match = re.search(pattern, name)
        if match:
            board_type = match.group(1)
            num_players = int(match.group(2))
            return make_config_key(board_type, num_players)
    return None


@dataclass
class TrainingDataManifest:
    """Manifest tracking all available training data."""

    entries: dict[str, list[TrainingDataEntry]] = field(default_factory=dict)
    last_refresh: datetime | None = None
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)

    def add_entry(self, entry: TrainingDataEntry) -> None:
        """Add an entry to the manifest."""
        if entry.config_key not in self.entries:
            self.entries[entry.config_key] = []
        # Check for duplicate paths
        for existing in self.entries[entry.config_key]:
            if existing.path == entry.path and existing.source == entry.source:
                # Update existing entry
                self.entries[entry.config_key].remove(existing)
                break
        self.entries[entry.config_key].append(entry)

    def get_all_data(self, config_key: str) -> list[TrainingDataEntry]:
        """Get all available data for a config, sorted by size (largest first)."""
        entries = self.entries.get(config_key, [])
        return sorted(entries, key=lambda e: e.size_bytes, reverse=True)

    def get_best_data(
        self,
        config_key: str,
        prefer_source: DataSource | None = None,
        min_size_mb: float = 0,
        max_age_hours: float | None = None,
        freshness_weight: float = 0.4,
    ) -> TrainingDataEntry | None:
        """Get the best available training data for a config.

        Uses combined scoring that balances data size and freshness:
            score = size_weight * (size / max_size) + freshness_weight * (1 / (1 + age_hours))

        Args:
            config_key: Config key (e.g., 'hex8_2p')
            prefer_source: Preferred data source (optional)
            min_size_mb: Minimum file size in MB
            max_age_hours: Maximum acceptable data age in hours (optional)
            freshness_weight: Weight for freshness (0-1). Size weight = 1 - freshness_weight.
                              Default 0.4 means 40% freshness, 60% size.

        Returns:
            Best training data entry, or None if not found
        """
        all_data = self.get_all_data(config_key)
        if not all_data:
            return None

        # Filter by minimum size
        min_bytes = int(min_size_mb * 1024 * 1024)
        filtered = [e for e in all_data if e.size_bytes >= min_bytes]
        if not filtered:
            return None

        # Filter by max age if specified
        if max_age_hours is not None:
            filtered = [
                e for e in filtered
                if e.age_hours is None or e.age_hours <= max_age_hours
            ]
            if not filtered:
                return None

        # If preferred source specified, filter to that source
        if prefer_source:
            source_data = [e for e in filtered if e.source == prefer_source]
            if source_data:
                filtered = source_data

        # Single entry - no scoring needed
        if len(filtered) == 1:
            return filtered[0]

        # Combined scoring
        max_size = max(e.size_bytes for e in filtered)
        size_weight = 1.0 - freshness_weight

        def score(entry: TrainingDataEntry) -> float:
            # Size score: proportion of max size
            size_score = entry.size_bytes / max_size if max_size > 0 else 0

            # Freshness score: inverse decay with age
            # age=0h -> 1.0, age=1h -> 0.5, age=3h -> 0.25, etc.
            age = entry.age_hours
            if age is None:
                # Unknown age - assume moderately stale (equivalent to 12h)
                freshness_score = 1.0 / (1.0 + 12.0)
            else:
                freshness_score = 1.0 / (1.0 + age)

            return size_weight * size_score + freshness_weight * freshness_score

        return max(filtered, key=score)

    def get_configs(self) -> list[str]:
        """Get all config keys with available data."""
        return list(self.entries.keys())

    def get_summary(self) -> dict[str, dict[str, Any]]:
        """Get a summary of available data by config."""
        summary = {}
        for config_key in self.entries:
            all_data = self.get_all_data(config_key)
            best = all_data[0] if all_data else None
            summary[config_key] = {
                "count": len(all_data),
                "best_source": best.source.value if best else None,
                "best_size_mb": round(best.size_mb, 1) if best else 0,
                "best_path": best.path if best else None,
                "sources": list(set(e.source.value for e in all_data)),
            }
        return summary

    async def ensure_available(
        self,
        entry: TrainingDataEntry,
        target_dir: Path | None = None,
    ) -> Path | None:
        """Ensure training data is available locally, downloading if needed.

        December 30, 2025: Added to enable lazy download of S3/OWC data.
        This allows training pipeline to automatically fetch the best data
        regardless of where it's stored.

        Args:
            entry: Training data entry to make available
            target_dir: Directory to download to (default: LOCAL_TRAINING_DIR)

        Returns:
            Path to local file, or None if download failed
        """
        if target_dir is None:
            target_dir = LOCAL_TRAINING_DIR

        # LOCAL source - already available
        if entry.source == DataSource.LOCAL:
            local_path = Path(entry.path)
            if local_path.exists():
                return local_path
            logger.warning(f"Local file missing: {entry.path}")
            return None

        # Ensure target directory exists
        target_dir.mkdir(parents=True, exist_ok=True)

        # Generate local filename from config_key
        filename = f"{entry.config_key}.npz"
        local_path = target_dir / filename

        # S3 source - download using aws s3 cp
        if entry.source == DataSource.S3:
            try:
                logger.info(f"Downloading from S3: {entry.path}")
                cmd = ["aws", "s3", "cp", entry.path, str(local_path)]
                result = await asyncio.to_thread(
                    subprocess.run,
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=600,  # 10 minute timeout
                )
                if result.returncode == 0 and local_path.exists():
                    logger.info(f"Downloaded {local_path.stat().st_size / 1024 / 1024:.1f}MB from S3")
                    return local_path
                else:
                    logger.error(f"S3 download failed: {result.stderr}")
                    return None
            except subprocess.TimeoutExpired:
                logger.error(f"Timeout downloading from S3: {entry.path}")
                return None
            except (OSError, subprocess.SubprocessError) as e:
                logger.error(f"Error downloading from S3: {e}")
                return None

        # OWC source - rsync from external drive
        if entry.source == DataSource.OWC:
            try:
                logger.info(f"Syncing from OWC: {entry.path}")
                ssh_key = os.path.expanduser(OWC_SSH_KEY)
                remote_path = f"{OWC_USER}@{OWC_HOST}:{entry.path}"
                # Dec 30, 2025: Use centralized SSH config for consistent timeouts
                if SSH_CONFIG_AVAILABLE and build_ssh_options:
                    ssh_opts = build_ssh_options(
                        key_path=ssh_key,
                        include_keepalive=False,  # rsync has its own timeout
                    )
                else:
                    ssh_opts = f"ssh -i {ssh_key} -o StrictHostKeyChecking=no"
                cmd = [
                    "rsync", "-avz",
                    "-e", ssh_opts,
                    remote_path,
                    str(local_path),
                ]
                result = await asyncio.to_thread(
                    subprocess.run,
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=600,
                )
                if result.returncode == 0 and local_path.exists():
                    logger.info(f"Synced {local_path.stat().st_size / 1024 / 1024:.1f}MB from OWC")
                    return local_path
                else:
                    logger.error(f"OWC sync failed: {result.stderr}")
                    return None
            except subprocess.TimeoutExpired:
                logger.error(f"Timeout syncing from OWC: {entry.path}")
                return None
            except (OSError, subprocess.SubprocessError) as e:
                logger.error(f"Error syncing from OWC: {e}")
                return None

        logger.warning(f"Unknown data source: {entry.source}")
        return None

    async def get_or_download_best(
        self,
        config_key: str,
        target_dir: Path | None = None,
        prefer_source: DataSource | None = None,
        min_size_mb: float = 0,
    ) -> Path | None:
        """Get the best training data for a config, downloading if needed.

        December 30, 2025: High-level API for training pipeline to get data.

        Args:
            config_key: Config key (e.g., 'hex8_2p')
            target_dir: Directory to download to (default: LOCAL_TRAINING_DIR)
            prefer_source: Preferred data source
            min_size_mb: Minimum file size in MB

        Returns:
            Path to local file, or None if no data available
        """
        best = self.get_best_data(
            config_key,
            prefer_source=prefer_source,
            min_size_mb=min_size_mb,
        )
        if not best:
            logger.warning(f"No training data found for {config_key}")
            return None

        return await self.ensure_available(best, target_dir)

    def _extract_npz_freshness(self, npz_path: Path) -> datetime | None:
        """Extract newest_game_time from NPZ metadata.

        NPZ files exported by export_replay_dataset.py contain a 'metadata'
        array with 'newest_game_time' field indicating when the newest game
        in the dataset was completed.

        Args:
            npz_path: Path to the NPZ file

        Returns:
            Datetime of newest game, or None if unavailable
        """
        if not NUMPY_AVAILABLE:
            return None

        try:
            with np.load(npz_path, allow_pickle=True) as npz:
                if "metadata" in npz.files:
                    metadata = npz["metadata"].item()
                    if isinstance(metadata, dict) and "newest_game_time" in metadata:
                        newest_time_str = metadata["newest_game_time"]
                        if newest_time_str:
                            return datetime.fromisoformat(newest_time_str)
        except (OSError, ValueError, TypeError, KeyError, zipfile.BadZipFile) as e:
            logger.debug(f"Failed to extract freshness from {npz_path}: {e}")
        return None

    async def refresh_local(self, data_dir: Path | None = None) -> int:
        """Refresh entries from local disk."""
        data_dir = data_dir or LOCAL_TRAINING_DIR
        count = 0

        if not data_dir.exists():
            logger.warning(f"Local training directory not found: {data_dir}")
            return 0

        for npz_file in data_dir.glob("*.npz"):
            config_key = extract_config_key(npz_file.name)
            if not config_key:
                continue

            stat = npz_file.stat()

            # Extract freshness from NPZ metadata
            newest_game_time = self._extract_npz_freshness(npz_file)

            entry = TrainingDataEntry(
                config_key=config_key,
                source=DataSource.LOCAL,
                path=str(npz_file.absolute()),
                size_bytes=stat.st_size,
                modified_time=datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc),
                newest_game_time=newest_game_time,
            )
            self.add_entry(entry)
            count += 1

        logger.info(f"Found {count} local training files")
        return count

    async def refresh_owc(self) -> int:
        """Refresh entries from OWC external drive."""
        count = 0

        # Check connectivity to mac-studio
        ssh_key_path = Path(OWC_SSH_KEY).expanduser()

        # Directories to scan on OWC
        owc_dirs = [
            "canonical_data",
            "consolidated_training",
            "training_data/coordinator_backup",
        ]

        for owc_dir in owc_dirs:
            full_path = f"{OWC_BASE_PATH}/{owc_dir}"
            # Remote shell command runs on the OWC host
            remote_cmd = (
                f"find {shlex.quote(full_path)} -name '*.npz' -type f "
                f"-exec stat -f '%N %z %m' {{}} \\; 2>/dev/null"
            )
            cmd = [
                "ssh",
                "-i", str(ssh_key_path),
                "-o", "ConnectTimeout=10",
                "-o", "BatchMode=yes",
                f"{OWC_USER}@{OWC_HOST}",
                remote_cmd,
            ]

            try:
                # December 30, 2025: Wrap subprocess.run in asyncio.to_thread()
                # to avoid blocking the event loop during SSH operations
                def _run_owc_scan() -> subprocess.CompletedProcess[str]:
                    return subprocess.run(
                        cmd,
                        capture_output=True,
                        text=True,
                        timeout=60,
                    )

                result = await asyncio.to_thread(_run_owc_scan)
                if result.returncode != 0:
                    logger.debug(f"OWC scan failed for {owc_dir}: {result.stderr}")
                    continue

                for line in result.stdout.strip().split("\n"):
                    if not line.strip():
                        continue
                    parts = line.rsplit(" ", 2)
                    if len(parts) != 3:
                        continue

                    path, size_str, mtime_str = parts
                    config_key = extract_config_key(Path(path).name)
                    if not config_key:
                        continue

                    try:
                        size_bytes = int(size_str)
                        mtime = datetime.fromtimestamp(
                            int(mtime_str), tz=timezone.utc
                        )
                    except ValueError:
                        continue

                    entry = TrainingDataEntry(
                        config_key=config_key,
                        source=DataSource.OWC,
                        path=path,
                        size_bytes=size_bytes,
                        modified_time=mtime,
                    )
                    self.add_entry(entry)
                    count += 1

            except subprocess.TimeoutExpired:
                logger.warning(f"Timeout scanning OWC directory: {owc_dir}")
            except (OSError, subprocess.SubprocessError, PermissionError) as e:
                # SSH/rsync connection failures or permission issues
                logger.warning(f"Error scanning OWC directory {owc_dir}: {e}")

        logger.info(f"Found {count} OWC training files")
        return count

    async def refresh_s3(self) -> int:
        """Refresh entries from S3 bucket."""
        count = 0

        cmd = [
            "aws", "s3", "ls",
            f"s3://{S3_BUCKET}/{S3_TRAINING_PREFIX}",
            "--recursive",
        ]

        try:
            # December 30, 2025: Wrap subprocess.run in asyncio.to_thread()
            # to avoid blocking the event loop during S3 operations
            def _run_s3_scan() -> subprocess.CompletedProcess[str]:
                return subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=60,
                )

            result = await asyncio.to_thread(_run_s3_scan)
            if result.returncode != 0:
                logger.warning(f"S3 scan failed: {result.stderr}")
                return 0

            for line in result.stdout.strip().split("\n"):
                if not line.strip() or not line.endswith(".npz"):
                    continue
                # Format: 2025-12-27 01:17:47   12345678 path/to/file.npz
                parts = line.split()
                if len(parts) < 4:
                    continue

                date_str, time_str, size_str = parts[0], parts[1], parts[2]
                path = " ".join(parts[3:])

                config_key = extract_config_key(Path(path).name)
                if not config_key:
                    continue

                try:
                    size_bytes = int(size_str)
                    mtime = datetime.strptime(
                        f"{date_str} {time_str}", "%Y-%m-%d %H:%M:%S"
                    ).replace(tzinfo=timezone.utc)
                except ValueError:
                    continue

                s3_uri = f"s3://{S3_BUCKET}/{path}"
                entry = TrainingDataEntry(
                    config_key=config_key,
                    source=DataSource.S3,
                    path=s3_uri,
                    size_bytes=size_bytes,
                    modified_time=mtime,
                )
                self.add_entry(entry)
                count += 1

        except subprocess.TimeoutExpired:
            logger.warning("Timeout scanning S3 bucket")
        except (OSError, subprocess.SubprocessError, PermissionError) as e:
            # AWS CLI execution failures or permission issues
            logger.warning(f"Error scanning S3: {e}")

        logger.info(f"Found {count} S3 training files")
        return count

    async def refresh_all(self) -> dict[str, int]:
        """Refresh entries from all sources."""
        async with self._lock:
            results = {}

            # Clear existing entries
            self.entries.clear()

            # Refresh each source
            results["local"] = await self.refresh_local()
            results["owc"] = await self.refresh_owc()
            results["s3"] = await self.refresh_s3()

            self.last_refresh = datetime.now(tz=timezone.utc)

            # Save to cache
            await self.save_cache()

            logger.info(
                f"Manifest refresh complete: {sum(results.values())} total entries"
            )
            return results

    async def save_cache(self) -> None:
        """Save manifest to cache file."""
        try:
            data = {
                "last_refresh": (
                    self.last_refresh.isoformat() if self.last_refresh else None
                ),
                "entries": {
                    config: [e.to_dict() for e in entries]
                    for config, entries in self.entries.items()
                },
            }
            def _write_cache() -> None:
                MANIFEST_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
                with open(MANIFEST_CACHE_PATH, "w") as f:
                    json.dump(data, f, indent=2)

            await asyncio.to_thread(_write_cache)
            logger.debug(f"Saved manifest cache to {MANIFEST_CACHE_PATH}")
        except (OSError, IOError, PermissionError, TypeError) as e:
            # File write failures or JSON serialization issues
            logger.warning(f"Failed to save manifest cache: {e}")

    async def load_cache(self) -> bool:
        """Load manifest from cache file."""
        try:
            if not MANIFEST_CACHE_PATH.exists():
                return False

            def _read_cache() -> dict:
                with open(MANIFEST_CACHE_PATH) as f:
                    return json.load(f)

            data = await asyncio.to_thread(_read_cache)

            self.last_refresh = (
                datetime.fromisoformat(data["last_refresh"])
                if data.get("last_refresh")
                else None
            )
            self.entries.clear()
            for config_key, entries_data in data.get("entries", {}).items():
                self.entries[config_key] = [
                    TrainingDataEntry.from_dict(e) for e in entries_data
                ]

            logger.debug(f"Loaded manifest cache from {MANIFEST_CACHE_PATH}")
            return True
        except (OSError, IOError, json.JSONDecodeError, KeyError, ValueError) as e:
            # File read failures, malformed JSON, or missing required fields
            logger.warning(f"Failed to load manifest cache: {e}")
            return False


# Singleton instance
_manifest_instance: TrainingDataManifest | None = None
_manifest_lock = asyncio.Lock()


async def get_training_data_manifest(
    refresh_if_stale_hours: float = 1.0,
) -> TrainingDataManifest:
    """Get or create the singleton training data manifest.

    Args:
        refresh_if_stale_hours: Refresh if cache is older than this many hours

    Returns:
        TrainingDataManifest instance
    """
    global _manifest_instance

    async with _manifest_lock:
        if _manifest_instance is None:
            _manifest_instance = TrainingDataManifest()
            await _manifest_instance.load_cache()

        # Check if refresh needed
        if _manifest_instance.last_refresh:
            last_refresh = _manifest_instance.last_refresh
            # Ensure last_refresh is timezone-aware
            if last_refresh.tzinfo is None:
                last_refresh = last_refresh.replace(tzinfo=timezone.utc)
            age_hours = (
                datetime.now(tz=timezone.utc) - last_refresh
            ).total_seconds() / 3600
            if age_hours > refresh_if_stale_hours:
                logger.info(
                    f"Manifest is {age_hours:.1f}h old, refreshing..."
                )
                await _manifest_instance.refresh_all()
        else:
            # No previous refresh, do one now
            await _manifest_instance.refresh_all()

    return _manifest_instance


def get_training_data_manifest_sync() -> TrainingDataManifest:
    """Get manifest synchronously (without refresh)."""
    global _manifest_instance
    if _manifest_instance is None:
        _manifest_instance = TrainingDataManifest()
        # Try to load from cache synchronously
        try:
            if MANIFEST_CACHE_PATH.exists():
                with open(MANIFEST_CACHE_PATH) as f:
                    data = json.load(f)
                _manifest_instance.last_refresh = (
                    datetime.fromisoformat(data["last_refresh"])
                    if data.get("last_refresh")
                    else None
                )
                for config_key, entries_data in data.get("entries", {}).items():
                    _manifest_instance.entries[config_key] = [
                        TrainingDataEntry.from_dict(e) for e in entries_data
                    ]
        except (OSError, IOError, json.JSONDecodeError, KeyError, ValueError):
            # Cache load failures - proceed with empty manifest
            pass
    return _manifest_instance
