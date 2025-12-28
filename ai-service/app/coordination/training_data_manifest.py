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
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

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

    @property
    def size_mb(self) -> float:
        """Size in megabytes."""
        return self.size_bytes / (1024 * 1024)

    @property
    def size_gb(self) -> float:
        """Size in gigabytes."""
        return self.size_bytes / (1024 * 1024 * 1024)

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
            num_players = match.group(2)
            return f"{board_type}_{num_players}p"
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
    ) -> TrainingDataEntry | None:
        """Get the best available training data for a config.

        Priority:
        1. Preferred source (if specified)
        2. Largest file size
        3. Quality score (if available)

        Args:
            config_key: Config key (e.g., 'hex8_2p')
            prefer_source: Preferred data source (optional)
            min_size_mb: Minimum file size in MB

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

        # If preferred source specified, try to get from that source
        if prefer_source:
            source_data = [e for e in filtered if e.source == prefer_source]
            if source_data:
                return source_data[0]  # Already sorted by size

        # Otherwise return largest
        return filtered[0]

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
            entry = TrainingDataEntry(
                config_key=config_key,
                source=DataSource.LOCAL,
                path=str(npz_file.absolute()),
                size_bytes=stat.st_size,
                modified_time=datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc),
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
        ssh_opts = f"-i {ssh_key_path} -o ConnectTimeout=10 -o BatchMode=yes"

        # Directories to scan on OWC
        owc_dirs = [
            "canonical_data",
            "consolidated_training",
            "training_data/coordinator_backup",
        ]

        for owc_dir in owc_dirs:
            full_path = f"{OWC_BASE_PATH}/{owc_dir}"
            cmd = (
                f"ssh {ssh_opts} {OWC_USER}@{OWC_HOST} "
                f"'find {full_path} -name \"*.npz\" -type f "
                f"-exec stat -f \"%N %z %m\" {{}} \\; 2>/dev/null'"
            )

            try:
                result = subprocess.run(
                    cmd,
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=60,
                )
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
            except Exception as e:
                logger.warning(f"Error scanning OWC directory {owc_dir}: {e}")

        logger.info(f"Found {count} OWC training files")
        return count

    async def refresh_s3(self) -> int:
        """Refresh entries from S3 bucket."""
        count = 0

        cmd = (
            f"aws s3 ls s3://{S3_BUCKET}/{S3_TRAINING_PREFIX} "
            "--recursive 2>/dev/null"
        )

        try:
            result = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=60,
            )
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
        except Exception as e:
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
            MANIFEST_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(MANIFEST_CACHE_PATH, "w") as f:
                json.dump(data, f, indent=2)
            logger.debug(f"Saved manifest cache to {MANIFEST_CACHE_PATH}")
        except Exception as e:
            logger.warning(f"Failed to save manifest cache: {e}")

    async def load_cache(self) -> bool:
        """Load manifest from cache file."""
        try:
            if not MANIFEST_CACHE_PATH.exists():
                return False

            with open(MANIFEST_CACHE_PATH) as f:
                data = json.load(f)

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
        except Exception as e:
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
            age_hours = (
                datetime.now(tz=timezone.utc) - _manifest_instance.last_refresh
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
        except Exception:
            pass
    return _manifest_instance
