#!/usr/bin/env python3
"""Model archival daemon for automatic checkpoint cleanup.

This script implements an archival policy to manage the growing number of
model checkpoints (600+) by automatically archiving old, underperforming,
or superseded models.

Usage:
    # Dry run (show what would be archived)
    python scripts/model_archival_daemon.py --dry-run

    # Execute archival
    python scripts/model_archival_daemon.py --execute

    # Run as daemon (check every hour)
    python scripts/model_archival_daemon.py --daemon --interval 3600
"""

import argparse
import json
import logging
import os
import re
import shutil
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Optional

# Add ai-service to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class ArchivalReason(Enum):
    """Reasons for archiving a model."""
    AGE = "age"  # Model is too old without promotion
    LOW_ELO = "low_elo"  # Performance below threshold
    SUPERSEDED = "superseded"  # Replaced by better model
    EXPERIMENT = "experiment"  # Experiment checkpoint pattern
    DUPLICATE = "duplicate"  # Same checksum as another model
    MANUAL = "manual"  # Manually requested


@dataclass
class ArchivalDecision:
    """Decision about whether to archive a model."""
    path: Path
    should_archive: bool
    reason: Optional[ArchivalReason] = None
    details: str = ""
    target_subdir: str = "misc"


@dataclass
class ArchivalPolicy:
    """Policy configuration for model archival."""
    # Age-based archival
    max_age_days: int = 7

    # Performance-based archival
    min_elo_threshold: float = 1400.0
    min_games_for_elo: int = 100

    # Experiment pattern archival
    experiment_patterns: list = field(default_factory=lambda: [
        r".*_iter\d+.*",
        r".*_epoch\d+.*",
        r".*_trial\d+.*",
        r".*_step\d+.*",
        r"checkpoint_.*",
    ])
    experiment_max_age_days: int = 3

    # Protected patterns (never archive)
    protected_patterns: list = field(default_factory=lambda: [
        r"canonical_.*",
        r"production_.*",
        r".*_best.*",
    ])


class ModelArchiver:
    """Handles model archival based on configurable policies."""

    def __init__(
        self,
        models_dir: Path,
        archive_dir: Optional[Path] = None,
        policy: Optional[ArchivalPolicy] = None,
    ):
        self.models_dir = Path(models_dir)
        self.archive_dir = archive_dir or self.models_dir / "archive"
        self.policy = policy or ArchivalPolicy()

        # Try to load model registry for ELO data
        self.registry = self._load_registry()

    def _load_registry(self):
        """Load model registry if available."""
        try:
            from app.training.model_registry import ModelRegistry
            registry_path = self.models_dir.parent / "data" / "model_registry" / "registry.db"
            if registry_path.exists():
                return ModelRegistry(registry_path)
        except Exception as e:
            logger.debug(f"Could not load model registry: {e}")
        return None

    def _is_protected(self, path: Path) -> bool:
        """Check if model matches a protected pattern."""
        name = path.name
        for pattern in self.policy.protected_patterns:
            if re.match(pattern, name, re.IGNORECASE):
                return True
        return False

    def _matches_experiment_pattern(self, path: Path) -> bool:
        """Check if model matches an experiment pattern."""
        name = path.name
        for pattern in self.policy.experiment_patterns:
            if re.match(pattern, name, re.IGNORECASE):
                return True
        return False

    def _get_model_age_days(self, path: Path) -> float:
        """Get model age in days based on modification time."""
        mtime = path.stat().st_mtime
        age_seconds = time.time() - mtime
        return age_seconds / (24 * 3600)

    def _get_model_elo(self, path: Path) -> Optional[tuple[float, int]]:
        """Get model ELO and games played from registry."""
        if not self.registry:
            return None
        try:
            # Try to find model in registry by path
            model_info = self.registry.get_model_by_path(str(path))
            if model_info and model_info.get("elo_rating"):
                return (model_info["elo_rating"], model_info.get("games_played", 0))
        except Exception:
            pass
        return None

    def evaluate_model(self, path: Path) -> ArchivalDecision:
        """Evaluate whether a model should be archived."""
        # Check if protected
        if self._is_protected(path):
            return ArchivalDecision(
                path=path,
                should_archive=False,
                details="Protected pattern",
            )

        age_days = self._get_model_age_days(path)

        # Check experiment pattern with shorter age threshold
        if self._matches_experiment_pattern(path):
            if age_days > self.policy.experiment_max_age_days:
                return ArchivalDecision(
                    path=path,
                    should_archive=True,
                    reason=ArchivalReason.EXPERIMENT,
                    details=f"Experiment checkpoint, age={age_days:.1f} days",
                    target_subdir="iter_checkpoints",
                )

        # Check ELO performance
        elo_data = self._get_model_elo(path)
        if elo_data:
            elo, games = elo_data
            if games >= self.policy.min_games_for_elo:
                if elo < self.policy.min_elo_threshold:
                    return ArchivalDecision(
                        path=path,
                        should_archive=True,
                        reason=ArchivalReason.LOW_ELO,
                        details=f"ELO={elo:.0f} < {self.policy.min_elo_threshold} after {games} games",
                        target_subdir="low_elo",
                    )

        # Check age-based archival
        if age_days > self.policy.max_age_days:
            # Only archive if not in production/staging
            if self.registry:
                try:
                    model_info = self.registry.get_model_by_path(str(path))
                    if model_info and model_info.get("stage") in ["production", "staging"]:
                        return ArchivalDecision(
                            path=path,
                            should_archive=False,
                            details=f"In {model_info['stage']} stage",
                        )
                except Exception:
                    pass

            return ArchivalDecision(
                path=path,
                should_archive=True,
                reason=ArchivalReason.AGE,
                details=f"Age={age_days:.1f} days > {self.policy.max_age_days}",
                target_subdir="aged_out",
            )

        return ArchivalDecision(
            path=path,
            should_archive=False,
            details=f"Age={age_days:.1f} days, no archival criteria met",
        )

    def scan_models(self) -> list[ArchivalDecision]:
        """Scan all models and evaluate for archival."""
        decisions = []

        # Find all model files
        patterns = ["*.pt", "*.pth", "*.ckpt"]
        for pattern in patterns:
            for path in self.models_dir.glob(pattern):
                if path.is_file() and "archive" not in str(path):
                    decision = self.evaluate_model(path)
                    decisions.append(decision)

        # Also scan subdirectories (but not archive)
        for subdir in self.models_dir.iterdir():
            if subdir.is_dir() and subdir.name != "archive":
                for pattern in patterns:
                    for path in subdir.glob(pattern):
                        if path.is_file():
                            decision = self.evaluate_model(path)
                            decisions.append(decision)

        return decisions

    def execute_archival(
        self,
        decisions: list[ArchivalDecision],
        dry_run: bool = True,
    ) -> dict:
        """Execute archival decisions."""
        # Create archive directory with year-month subfolder
        date_folder = datetime.now().strftime("%Y%m")
        archive_base = self.archive_dir / date_folder

        stats = {
            "total_scanned": len(decisions),
            "to_archive": 0,
            "archived": 0,
            "skipped": 0,
            "errors": 0,
            "space_freed_mb": 0,
        }

        to_archive = [d for d in decisions if d.should_archive]
        stats["to_archive"] = len(to_archive)

        if not to_archive:
            logger.info("No models to archive")
            return stats

        # Group by reason for manifest
        manifest = {
            "archived_at": datetime.now().isoformat(),
            "dry_run": dry_run,
            "models": [],
        }

        for decision in to_archive:
            target_dir = archive_base / decision.target_subdir
            target_path = target_dir / decision.path.name

            size_mb = decision.path.stat().st_size / (1024 * 1024)

            manifest["models"].append({
                "original_path": str(decision.path),
                "archived_path": str(target_path),
                "reason": decision.reason.value if decision.reason else "unknown",
                "details": decision.details,
                "size_mb": round(size_mb, 2),
            })

            if dry_run:
                logger.info(f"[DRY RUN] Would archive: {decision.path.name}")
                logger.info(f"  → {target_path}")
                logger.info(f"  Reason: {decision.reason.value if decision.reason else 'unknown'}")
                logger.info(f"  Details: {decision.details}")
                stats["space_freed_mb"] += size_mb
            else:
                try:
                    target_dir.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(decision.path), str(target_path))
                    logger.info(f"Archived: {decision.path.name} → {target_path}")
                    stats["archived"] += 1
                    stats["space_freed_mb"] += size_mb
                except Exception as e:
                    logger.error(f"Failed to archive {decision.path}: {e}")
                    stats["errors"] += 1

        # Save manifest
        if not dry_run and stats["archived"] > 0:
            manifest_path = archive_base / "archived_manifest.json"
            manifest_path.parent.mkdir(parents=True, exist_ok=True)

            # Append to existing manifest if it exists
            if manifest_path.exists():
                with open(manifest_path) as f:
                    existing = json.load(f)
                if isinstance(existing, list):
                    existing.append(manifest)
                else:
                    existing = [existing, manifest]
                manifest = existing
            else:
                manifest = [manifest]

            with open(manifest_path, "w") as f:
                json.dump(manifest, f, indent=2)
            logger.info(f"Manifest saved to {manifest_path}")

        return stats


def main():
    parser = argparse.ArgumentParser(description="Model archival daemon")
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=Path(__file__).parent.parent / "models",
        help="Models directory to scan",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be archived without executing",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Execute archival (moves files)",
    )
    parser.add_argument(
        "--daemon",
        action="store_true",
        help="Run as daemon with periodic checks",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=3600,
        help="Daemon check interval in seconds (default: 3600)",
    )
    parser.add_argument(
        "--max-age-days",
        type=int,
        default=7,
        help="Max age in days before archival (default: 7)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    policy = ArchivalPolicy(max_age_days=args.max_age_days)
    archiver = ModelArchiver(args.models_dir, policy=policy)

    def run_scan():
        logger.info(f"Scanning {args.models_dir}...")
        decisions = archiver.scan_models()

        to_archive = [d for d in decisions if d.should_archive]
        to_keep = [d for d in decisions if not d.should_archive]

        logger.info(f"Scan complete: {len(decisions)} models found")
        logger.info(f"  To archive: {len(to_archive)}")
        logger.info(f"  To keep: {len(to_keep)}")

        if to_archive:
            dry_run = not args.execute
            stats = archiver.execute_archival(decisions, dry_run=dry_run)

            logger.info(f"\nArchival {'preview' if dry_run else 'complete'}:")
            logger.info(f"  Models archived: {stats['archived'] if not dry_run else stats['to_archive']}")
            logger.info(f"  Space freed: {stats['space_freed_mb']:.1f} MB")
            if stats["errors"] > 0:
                logger.warning(f"  Errors: {stats['errors']}")

    if args.daemon:
        logger.info(f"Starting daemon mode (interval: {args.interval}s)")
        while True:
            try:
                run_scan()
            except Exception as e:
                logger.error(f"Scan failed: {e}")
            time.sleep(args.interval)
    else:
        run_scan()


if __name__ == "__main__":
    main()
