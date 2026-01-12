"""Data Availability Checking for Training Trigger Daemon.

This module contains the data availability checking logic extracted from
TrainingTriggerDaemon to reduce file size and improve maintainability.

January 2026: Extracted from training_trigger_daemon.py as part of
modularization effort.

Usage:
    from app.coordination.training_data_availability import (
        DataAvailabilityChecker,
        DataAvailabilityConfig,
        check_gpu_availability,
        check_cluster_availability,
    )

    # Quick standalone checks
    gpu_ok = await check_gpu_availability()
    cluster_ok = await check_cluster_availability()

    # Full checker with config
    config = DataAvailabilityConfig(local_only_mode=False)
    checker = DataAvailabilityChecker(config)
    total, remote_path = await checker.check_all_data_sources("hex8_2p", 5000)
"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.config.ports import get_local_p2p_status_url
from app.coordination.event_utils import make_config_key

logger = logging.getLogger(__name__)


@dataclass
class DataAvailabilityConfig:
    """Configuration for data availability checking."""

    local_only_mode: bool = False
    gpu_idle_threshold_percent: float = 50.0
    cluster_availability_timeout_seconds: float = 5.0
    max_data_age_hours: float = 1.0
    freshness_sync_timeout_seconds: float = 300.0


async def check_gpu_availability(
    gpu_idle_threshold_percent: float = 50.0,
) -> bool:
    """Check if any GPU is available for training.

    Args:
        gpu_idle_threshold_percent: GPU utilization threshold below which
                                     a GPU is considered available

    Returns:
        True if at least one GPU is available (or if check fails, assumes available)
    """
    try:
        process = await asyncio.create_subprocess_exec(
            "nvidia-smi",
            "--query-gpu=utilization.gpu",
            "--format=csv,noheader,nounits",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await asyncio.wait_for(process.communicate(), timeout=10)

        if process.returncode == 0:
            for line in stdout.decode().strip().split("\n"):
                try:
                    util = float(line.strip())
                    if util < gpu_idle_threshold_percent:
                        return True
                except ValueError:
                    continue
            return False

    except (FileNotFoundError, asyncio.TimeoutError):
        pass
    except Exception as e:
        logger.debug(f"[DataAvailability] GPU check failed: {e}")

    # Assume GPU available if we can't check
    return True


async def check_cluster_availability(
    timeout_seconds: float = 5.0,
) -> bool:
    """Check if cluster is available with fast timeout.

    Used to determine if we should fall back to local-only mode
    when cluster is unreachable.

    Args:
        timeout_seconds: Timeout for the cluster check

    Returns:
        True if cluster is reachable with alive peers, False otherwise
    """
    try:
        import aiohttp

        p2p_url = get_local_p2p_status_url()
        async with aiohttp.ClientSession() as session:
            async with session.get(
                p2p_url, timeout=aiohttp.ClientTimeout(total=timeout_seconds)
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    alive_peers = data.get("alive_peers", 0)
                    if alive_peers > 0:
                        return True
                    logger.debug("[DataAvailability] Cluster check: no alive peers")
                    return False

    except ImportError:
        logger.debug("[DataAvailability] aiohttp not available for cluster check")
    except asyncio.TimeoutError:
        logger.debug(
            f"[DataAvailability] Cluster check timed out after {timeout_seconds}s"
        )
    except Exception as e:
        logger.debug(f"[DataAvailability] Cluster check failed: {e}")

    return False


def parse_config_from_filename(name: str) -> tuple[str | None, int | None]:
    """Parse board type and num_players from an NPZ filename.

    Recognizes patterns like:
    - hex8_2p.npz
    - square8_4p_v2.npz
    - hexagonal_3p_filtered.npz

    Args:
        name: Filename stem (without .npz extension)

    Returns:
        Tuple of (board_type, num_players) or (None, None) if not parseable
    """
    patterns = [
        r"^(hex8|square8|square19|hexagonal)_(\d)p",
        r"^(hex|sq|square)(\d+)_(\d)p",
    ]

    for pattern in patterns:
        match = re.match(pattern, name)
        if match:
            groups = match.groups()
            if len(groups) == 2:
                return groups[0], int(groups[1])
            elif len(groups) == 3:
                # Handle short board names
                board_map = {"hex": "hex8", "sq": "square8", "square": "square8"}
                board = board_map.get(groups[0], groups[0])
                return board, int(groups[2])

    return None, None


def scan_local_npz_files(
    training_dir: Path | None = None,
) -> list[tuple[str, str, int, Path]]:
    """Scan local NPZ files for training.

    Args:
        training_dir: Directory to scan. Defaults to data/training/

    Returns:
        List of (config_key, board_type, num_players, npz_path) tuples
        for all valid NPZ files found.
    """
    results: list[tuple[str, str, int, Path]] = []

    if training_dir is None:
        training_dir = (
            Path(__file__).resolve().parent.parent.parent / "data" / "training"
        )

    if not training_dir.exists():
        return results

    for npz_path in training_dir.glob("*.npz"):
        board_type, num_players = parse_config_from_filename(npz_path.stem)
        if board_type is None or num_players is None:
            continue

        config_key = make_config_key(board_type, num_players)
        results.append((config_key, board_type, num_players, npz_path))

    return results


class DataAvailabilityChecker:
    """Checks data availability from multiple sources for training.

    Queries local NPZ files, TrainingDataManifest (S3/OWC), and
    ClusterManifest to find available training data.
    """

    def __init__(self, config: DataAvailabilityConfig | None = None):
        """Initialize the checker.

        Args:
            config: Configuration options. Uses defaults if not provided.
        """
        self.config = config or DataAvailabilityConfig()

    async def check_gpu_availability(self) -> bool:
        """Check if any GPU is available for training."""
        return await check_gpu_availability(self.config.gpu_idle_threshold_percent)

    async def check_cluster_availability(self) -> bool:
        """Check if cluster is available."""
        return await check_cluster_availability(
            self.config.cluster_availability_timeout_seconds
        )

    def scan_local_npz_files(
        self, training_dir: Path | None = None
    ) -> list[tuple[str, str, int, Path]]:
        """Scan local NPZ files for training."""
        return scan_local_npz_files(training_dir)

    async def ensure_fresh_data(
        self,
        board_type: str,
        num_players: int,
        training_states: dict[str, Any] | None = None,
    ) -> bool:
        """Ensure training data is fresh, triggering sync if needed.

        Uses training_freshness module to check data age and trigger sync
        if data is stale.

        Args:
            board_type: Board type for training
            num_players: Number of players
            training_states: Optional dict to update with fresh data info

        Returns:
            True if data is now fresh, False if sync failed or timed out
        """
        import time

        # In local-only mode, just check if local NPZ exists
        if self.config.local_only_mode:
            config_key = make_config_key(board_type, num_players)
            local_npz = Path(f"data/training/{config_key}.npz")
            if local_npz.exists():
                logger.debug(
                    f"[DataAvailability] Local-only mode: using existing NPZ for {config_key}"
                )
                return True
            logger.debug(
                f"[DataAvailability] Local-only mode: no NPZ for {config_key}"
            )
            return False

        try:
            from app.coordination.training_freshness import (
                DataFreshnessChecker,
                FreshnessConfig,
            )

            freshness_config = FreshnessConfig(
                max_age_hours=self.config.max_data_age_hours,
                trigger_sync=True,
                wait_for_sync=True,
                sync_timeout_seconds=self.config.freshness_sync_timeout_seconds,
            )

            checker = DataFreshnessChecker(freshness_config)
            result = await checker.ensure_fresh_data(board_type, num_players)

            if result.is_fresh:
                # Update state with fresh data info
                if training_states is not None:
                    config_key = make_config_key(board_type, num_players)
                    if config_key in training_states:
                        training_states[config_key].last_npz_update = time.time()
                        if result.games_available:
                            training_states[
                                config_key
                            ].npz_sample_count = result.games_available
                return True

            logger.warning(
                f"[DataAvailability] Data freshness check failed for "
                f"{board_type}_{num_players}p: {result.error}"
            )
            return False

        except ImportError:
            logger.debug("[DataAvailability] training_freshness module not available")
            return False
        except Exception as e:
            logger.warning(f"[DataAvailability] ensure_fresh_data failed: {e}")
            return False

    async def check_all_data_sources(
        self, config_key: str, min_samples_needed: int
    ) -> tuple[int, str | None]:
        """Check all sources for available training data.

        Queries local NPZ files, TrainingDataManifest (S3/OWC), and
        ClusterManifest to find total available samples.

        Args:
            config_key: Configuration identifier (e.g., "hex8_2p")
            min_samples_needed: Minimum samples required for training

        Returns:
            Tuple of (total_samples_available, best_remote_path_if_any)
        """
        total_samples = 0
        best_remote_path: str | None = None

        # 1. Check local NPZ files
        try:
            local_npz = Path(f"data/training/{config_key}.npz")
            if local_npz.exists():
                import numpy as np

                data = np.load(local_npz)
                local_count = len(data.get("features", data.get("states", [])))
                total_samples += local_count
                logger.debug(
                    f"[DataAvailability] Local NPZ for {config_key}: {local_count} samples"
                )
        except Exception as e:
            logger.debug(f"[DataAvailability] Local NPZ check failed: {e}")

        # Skip remote sources in local-only mode
        if self.config.local_only_mode:
            logger.debug(
                f"[DataAvailability] Local-only mode: skipping remote data sources for {config_key}"
            )
            return total_samples, best_remote_path

        # 2. Check TrainingDataManifest for S3/OWC data
        try:
            from app.coordination.training_data_manifest import (
                get_training_manifest,
                DataSource,
            )

            manifest = get_training_manifest()

            # Check S3
            s3_data = manifest.get_data_for_config(config_key, source=DataSource.S3)
            if s3_data and s3_data.sample_count > 0:
                logger.debug(
                    f"[DataAvailability] S3 has {s3_data.sample_count} samples for {config_key}"
                )
                if s3_data.sample_count > total_samples:
                    total_samples = s3_data.sample_count
                    best_remote_path = s3_data.path

            # Check OWC
            owc_data = manifest.get_data_for_config(config_key, source=DataSource.OWC)
            if owc_data and owc_data.sample_count > 0:
                logger.debug(
                    f"[DataAvailability] OWC has {owc_data.sample_count} samples for {config_key}"
                )
                if owc_data.sample_count > total_samples:
                    total_samples = owc_data.sample_count
                    best_remote_path = owc_data.path

        except ImportError:
            logger.debug("[DataAvailability] TrainingDataManifest not available")
        except Exception as e:
            logger.debug(f"[DataAvailability] Manifest check failed: {e}")

        # 3. Check ClusterManifest for games on other nodes
        try:
            from app.distributed.cluster_manifest import get_cluster_manifest

            cluster_manifest = get_cluster_manifest()
            remote_games = cluster_manifest.get_game_count(config_key)
            if remote_games > 0:
                # Estimate ~50 samples per game
                estimated_samples = remote_games * 50
                logger.debug(
                    f"[DataAvailability] Cluster has ~{remote_games} games "
                    f"(~{estimated_samples} samples) for {config_key}"
                )

        except ImportError:
            logger.debug("[DataAvailability] ClusterManifest not available")
        except Exception as e:
            logger.debug(f"[DataAvailability] Cluster check failed: {e}")

        return total_samples, best_remote_path

    async def fetch_remote_data_if_needed(
        self, config_key: str, local_count: int, min_samples_needed: int
    ) -> bool:
        """Fetch remote data if local is insufficient.

        When local training data is below threshold, attempts to download
        from S3 or OWC to enable training.

        Args:
            config_key: Configuration identifier (e.g., "hex8_2p")
            local_count: Current local sample count
            min_samples_needed: Minimum samples required for training

        Returns:
            True if data was fetched and is now available locally
        """
        if local_count >= min_samples_needed:
            return True  # Already have enough locally

        try:
            from app.coordination.training_data_manifest import (
                get_training_manifest,
                DataSource,
            )

            manifest = get_training_manifest()

            # Find best remote source
            best_source = None
            best_count = local_count

            for source in [DataSource.S3, DataSource.OWC]:
                data = manifest.get_data_for_config(config_key, source=source)
                if data and data.sample_count > best_count:
                    best_source = data
                    best_count = data.sample_count

            if best_source and best_count >= min_samples_needed:
                logger.info(
                    f"[DataAvailability] Fetching {config_key} from "
                    f"{best_source.source.value} ({best_count} samples)"
                )

                # Download to local training directory
                local_path = await manifest.download_to_local(best_source)
                if local_path and local_path.exists():
                    logger.info(
                        f"[DataAvailability] Downloaded {config_key} to {local_path}"
                    )
                    return True
                else:
                    logger.warning(
                        f"[DataAvailability] Download failed for {config_key}"
                    )
                    return False

            logger.debug(
                f"[DataAvailability] No remote source with enough data for {config_key}"
            )
            return False

        except ImportError:
            logger.debug("[DataAvailability] TrainingDataManifest not available")
            return False
        except Exception as e:
            logger.warning(f"[DataAvailability] Remote fetch failed: {e}")
            return False
