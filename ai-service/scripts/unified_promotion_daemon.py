#!/usr/bin/env python3
"""Unified Model Promotion Daemon - Automated gauntlet evaluation and promotion.

Continuously monitors for new trained models, runs baseline gauntlet evaluation,
and promotes models that exceed Elo thresholds. Integrates with cluster sync
and notification systems.

Usage:
    # Run once (for cron)
    python scripts/unified_promotion_daemon.py --check-once

    # Run as daemon
    python scripts/unified_promotion_daemon.py --daemon

    # Dry run (no actual promotion)
    python scripts/unified_promotion_daemon.py --check-once --dry-run

    # Show status
    python scripts/unified_promotion_daemon.py --status
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import yaml

# Add project root
SCRIPT_DIR = Path(__file__).parent
AI_SERVICE_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(AI_SERVICE_ROOT))

LOG_DIR = AI_SERVICE_ROOT / "logs"
LOG_FILE = LOG_DIR / "promotion_daemon.log"
STATE_FILE = AI_SERVICE_ROOT / "data" / "promotion_daemon_state.json"
CONFIG_FILE = AI_SERVICE_ROOT / "config" / "promotion_daemon.yaml"

LOG_DIR.mkdir(parents=True, exist_ok=True)
STATE_FILE.parent.mkdir(parents=True, exist_ok=True)

from scripts.lib.logging_config import setup_script_logging

logger = setup_script_logging("unified_promotion_daemon")

# =============================================================================
# Configuration
# =============================================================================

@dataclass
class DaemonConfig:
    """Promotion daemon configuration."""
    check_interval_seconds: int = 300
    models_dir: str = "models"

    # Gauntlet settings
    games_per_baseline: int = 10
    baselines: List[str] = field(default_factory=lambda: ["random", "heuristic"])
    fast_mode: bool = True

    # Promotion thresholds
    elo_threshold: float = 25.0
    min_games: int = 50
    min_win_rate: float = 0.52
    confidence_level: float = 0.95

    # Cluster sync
    cluster_sync_enabled: bool = True
    restart_p2p_after_sync: bool = False

    # Notifications
    slack_webhook_url: Optional[str] = None
    notify_on_promotion: bool = True
    notify_on_rejection: bool = False

    @classmethod
    def from_yaml(cls, path: Path) -> "DaemonConfig":
        """Load config from YAML file."""
        if not path.exists():
            logger.warning(f"Config file not found: {path}, using defaults")
            return cls()

        with open(path) as f:
            data = yaml.safe_load(f) or {}

        # Flatten nested config
        daemon = data.get("daemon", {})
        gauntlet = data.get("gauntlet", {})
        promotion = data.get("promotion", {})
        cluster = data.get("cluster_sync", {})
        notifications = data.get("notifications", {})

        # Resolve environment variables
        slack_url = notifications.get("slack_webhook")
        if slack_url and slack_url.startswith("${"):
            env_var = slack_url[2:-1]
            slack_url = os.environ.get(env_var)

        return cls(
            check_interval_seconds=daemon.get("check_interval_seconds", 300),
            models_dir=daemon.get("models_dir", "models"),
            games_per_baseline=gauntlet.get("games_per_baseline", 10),
            baselines=gauntlet.get("baselines", ["random", "heuristic"]),
            fast_mode=gauntlet.get("fast_mode", True),
            elo_threshold=promotion.get("elo_threshold", 25.0),
            min_games=promotion.get("min_games", 50),
            min_win_rate=promotion.get("min_win_rate", 0.52),
            confidence_level=promotion.get("confidence_level", 0.95),
            cluster_sync_enabled=cluster.get("enabled", True),
            restart_p2p_after_sync=cluster.get("restart_p2p", False),
            slack_webhook_url=slack_url,
            notify_on_promotion=notifications.get("on_promotion", True),
            notify_on_rejection=notifications.get("on_rejection", False),
        )


@dataclass
class DaemonState:
    """Persistent daemon state."""
    known_models: Dict[str, str] = field(default_factory=dict)  # path -> hash
    evaluated_models: Dict[str, Dict] = field(default_factory=dict)  # path -> result
    last_check: Optional[str] = None
    promotions: List[Dict] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> "DaemonState":
        return cls(
            known_models=data.get("known_models", {}),
            evaluated_models=data.get("evaluated_models", {}),
            last_check=data.get("last_check"),
            promotions=data.get("promotions", []),
        )


# =============================================================================
# Model Watcher
# =============================================================================

class ModelWatcher:
    """Watch for new model files."""

    def __init__(self, models_dir: Path, patterns: List[str] = None):
        self.models_dir = models_dir
        self.patterns = patterns or ["*.pth"]

    def get_all_models(self) -> List[Path]:
        """Get all model files."""
        models = []
        for pattern in self.patterns:
            models.extend(self.models_dir.glob(pattern))
            models.extend(self.models_dir.glob(f"**/{pattern}"))
        return sorted(set(models))

    def get_model_hash(self, path: Path) -> str:
        """Get hash of model file (for change detection)."""
        # Use file size + mtime for fast hashing
        stat = path.stat()
        return hashlib.md5(f"{stat.st_size}:{stat.st_mtime}".encode()).hexdigest()[:16]

    def get_new_models(self, known: Dict[str, str]) -> List[Path]:
        """Get models not in known set."""
        new_models = []
        for model in self.get_all_models():
            # Skip broken symlinks
            if model.is_symlink() and not model.exists():
                continue
            path_str = str(model)
            try:
                current_hash = self.get_model_hash(model)
            except (FileNotFoundError, OSError):
                continue
            if path_str not in known or known[path_str] != current_hash:
                new_models.append(model)
        return new_models


# =============================================================================
# Gauntlet Runner
# =============================================================================

class GauntletRunner:
    """Run baseline gauntlet for models."""

    def __init__(self, config: DaemonConfig):
        self.config = config

    def run_gauntlet(self, model_path: Path) -> Optional[Dict]:
        """Run gauntlet for a single model."""
        try:
            # Import gauntlet functions
            from scripts.baseline_gauntlet import (
                run_gauntlet_for_model,
                GauntletResult,
            )
            from app.models import BoardType

            # Parse model info from path/filename
            model_info = self._parse_model_info(model_path)

            model_dict = {
                "path": str(model_path),
                "name": model_path.stem,
                "type": model_info.get("type", "nn"),
            }

            board_type = BoardType(model_info.get("board_type", "square8"))

            result = run_gauntlet_for_model(
                model=model_dict,
                num_games=self.config.games_per_baseline,
                board_type=board_type,
                fast_mode=self.config.fast_mode,
            )

            return {
                "model_path": str(model_path),
                "model_name": model_path.stem,
                "model_type": result.model_type,
                "vs_random": result.vs_random,
                "vs_heuristic": result.vs_heuristic,
                "vs_mcts": result.vs_mcts,
                "score": result.score,
                "games_played": result.games_played,
                "board_type": model_info.get("board_type", "square8"),
                "num_players": model_info.get("num_players", 2),
                "timestamp": datetime.now().isoformat(),
            }
        except Exception as e:
            logger.error(f"Gauntlet failed for {model_path}: {e}")
            return None

    def _parse_model_info(self, path: Path) -> Dict[str, Any]:
        """Parse model info from filename or sidecar JSON."""
        info = {"type": "nn", "board_type": "square8", "num_players": 2}

        # Check for sidecar JSON
        json_path = path.with_suffix(".json")
        if json_path.exists():
            try:
                with open(json_path) as f:
                    sidecar = json.load(f)
                    info["type"] = sidecar.get("model_type", info["type"])
                    info["board_type"] = sidecar.get("board_type", info["board_type"])
                    info["num_players"] = sidecar.get("num_players", info["num_players"])
            except Exception:
                pass

        # Parse from filename
        name = path.stem.lower()
        if "nnue" in name:
            info["type"] = "nnue"
        if "sq8" in name or "square8" in name:
            info["board_type"] = "square8"
        elif "sq19" in name or "square19" in name:
            info["board_type"] = "square19"
        elif "hex" in name:
            info["board_type"] = "hexagonal"

        for np in [2, 3, 4]:
            if f"_{np}p" in name or f"{np}p_" in name:
                info["num_players"] = np
                break

        return info


# =============================================================================
# Promotion Checker
# =============================================================================

class PromotionChecker:
    """Check if model should be promoted."""

    def __init__(self, config: DaemonConfig):
        self.config = config

    def should_promote(self, gauntlet_result: Dict) -> tuple[bool, str]:
        """Check if model meets promotion criteria."""
        # Check win rates
        vs_random = gauntlet_result.get("vs_random", 0)
        vs_heuristic = gauntlet_result.get("vs_heuristic", 0)
        score = gauntlet_result.get("score", 0)

        # Must beat random decisively
        if vs_random < 0.90:
            return False, f"Too low vs random: {vs_random:.1%} (need 90%)"

        # Must beat heuristic reasonably
        if vs_heuristic < self.config.min_win_rate:
            return False, f"Too low vs heuristic: {vs_heuristic:.1%} (need {self.config.min_win_rate:.0%})"

        # Check composite score (if available)
        if score < 0.5:
            return False, f"Low composite score: {score:.2f}"

        return True, "Passed all criteria"

    def get_current_best(self, board_type: str, num_players: int) -> Optional[Dict]:
        """Get current best model for config from Elo DB."""
        try:
            from app.tournament.unified_elo_db import get_elo_database
            db = get_elo_database()
            leaderboard = db.get_leaderboard(board_type, num_players, limit=1)
            if leaderboard:
                return {
                    "model_id": leaderboard[0].participant_id,
                    "elo": leaderboard[0].rating,
                }
        except Exception as e:
            logger.warning(f"Could not get current best: {e}")
        return None


# =============================================================================
# Cluster Sync
# =============================================================================

class ClusterSync:
    """Sync promoted models to cluster."""

    def __init__(self, config: DaemonConfig):
        self.config = config

    def sync_model(self, model_path: Path, dry_run: bool = False) -> bool:
        """Sync model to cluster nodes."""
        if not self.config.cluster_sync_enabled:
            logger.info("Cluster sync disabled")
            return True

        if dry_run:
            logger.info(f"[DRY RUN] Would sync {model_path} to cluster")
            return True

        try:
            # Use model_promotion_manager for sync
            from scripts.model_promotion_manager import sync_models_to_cluster

            logger.info(f"Syncing {model_path} to cluster...")
            sync_models_to_cluster([str(model_path)])
            return True
        except ImportError:
            logger.warning("model_promotion_manager not available, skipping cluster sync")
            return True
        except Exception as e:
            logger.error(f"Cluster sync failed: {e}")
            return False


# =============================================================================
# Notifier
# =============================================================================

class Notifier:
    """Send notifications for promotion events."""

    def __init__(self, config: DaemonConfig):
        self.config = config

    def notify_promotion(self, model_path: str, result: Dict) -> None:
        """Send notification for successful promotion."""
        if not self.config.notify_on_promotion:
            return

        message = (
            f"*Model Promoted*\n"
            f"Model: `{Path(model_path).stem}`\n"
            f"Board: {result.get('board_type', 'unknown')}\n"
            f"Score: {result.get('score', 0):.2f}\n"
            f"vs Random: {result.get('vs_random', 0):.1%}\n"
            f"vs Heuristic: {result.get('vs_heuristic', 0):.1%}"
        )
        self._send_slack(message, "good")

    def notify_rejection(self, model_path: str, reason: str) -> None:
        """Send notification for rejected model."""
        if not self.config.notify_on_rejection:
            return

        message = (
            f"*Model Rejected*\n"
            f"Model: `{Path(model_path).stem}`\n"
            f"Reason: {reason}"
        )
        self._send_slack(message, "warning")

    def _send_slack(self, message: str, color: str = "good") -> None:
        """Send Slack notification."""
        if not self.config.slack_webhook_url:
            return

        import urllib.request

        payload = {
            "attachments": [{
                "color": color,
                "text": message,
                "footer": "RingRift Promotion Daemon",
                "ts": int(time.time()),
            }]
        }

        try:
            req = urllib.request.Request(
                self.config.slack_webhook_url,
                data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                if resp.status == 200:
                    logger.info("Slack notification sent")
        except Exception as e:
            logger.warning(f"Failed to send Slack notification: {e}")


# =============================================================================
# Main Daemon
# =============================================================================

class UnifiedPromotionDaemon:
    """Main promotion daemon."""

    def __init__(self, config: DaemonConfig):
        self.config = config
        self.models_dir = AI_SERVICE_ROOT / config.models_dir

        self.watcher = ModelWatcher(self.models_dir)
        self.gauntlet = GauntletRunner(config)
        self.checker = PromotionChecker(config)
        self.sync = ClusterSync(config)
        self.notifier = Notifier(config)

        self.state = self._load_state()

    def _load_state(self) -> DaemonState:
        """Load persistent state."""
        if STATE_FILE.exists():
            try:
                with open(STATE_FILE) as f:
                    return DaemonState.from_dict(json.load(f))
            except Exception:
                pass
        return DaemonState()

    def _save_state(self) -> None:
        """Save persistent state."""
        try:
            with open(STATE_FILE, "w") as f:
                json.dump(self.state.to_dict(), f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save state: {e}")

    def check_once(self, dry_run: bool = False) -> Dict[str, Any]:
        """Run one check cycle."""
        logger.info("=" * 70)
        logger.info("PROMOTION DAEMON CHECK")
        logger.info("=" * 70)

        results = {
            "checked": 0,
            "new_models": 0,
            "evaluated": 0,
            "promoted": 0,
            "rejected": 0,
            "errors": 0,
        }

        # Find new models
        new_models = self.watcher.get_new_models(self.state.known_models)
        results["new_models"] = len(new_models)

        if not new_models:
            logger.info("No new models found")
            self.state.last_check = datetime.now().isoformat()
            self._save_state()
            return results

        logger.info(f"Found {len(new_models)} new model(s)")

        for model_path in new_models:
            results["checked"] += 1
            model_name = model_path.stem

            # Update known models
            self.state.known_models[str(model_path)] = self.watcher.get_model_hash(model_path)

            # Run gauntlet
            logger.info(f"Evaluating: {model_name}")
            gauntlet_result = self.gauntlet.run_gauntlet(model_path)

            if gauntlet_result is None:
                results["errors"] += 1
                continue

            results["evaluated"] += 1
            self.state.evaluated_models[str(model_path)] = gauntlet_result

            # Check promotion criteria
            should_promote, reason = self.checker.should_promote(gauntlet_result)

            if should_promote:
                logger.info(f"PROMOTING: {model_name} - {reason}")
                results["promoted"] += 1

                # Sync to cluster
                if not dry_run:
                    self.sync.sync_model(model_path, dry_run=dry_run)

                # Record promotion
                self.state.promotions.append({
                    "model": model_name,
                    "timestamp": datetime.now().isoformat(),
                    "result": gauntlet_result,
                })

                # Notify
                self.notifier.notify_promotion(str(model_path), gauntlet_result)
            else:
                logger.info(f"REJECTED: {model_name} - {reason}")
                results["rejected"] += 1
                self.notifier.notify_rejection(str(model_path), reason)

        self.state.last_check = datetime.now().isoformat()
        self._save_state()

        # Summary
        logger.info("-" * 70)
        logger.info(f"Summary: {results['evaluated']} evaluated, "
                   f"{results['promoted']} promoted, {results['rejected']} rejected")

        return results

    def run_daemon(self) -> None:
        """Run as continuous daemon."""
        logger.info(f"Starting promotion daemon (interval: {self.config.check_interval_seconds}s)")

        while True:
            try:
                self.check_once()
            except Exception as e:
                logger.error(f"Check cycle failed: {e}")

            time.sleep(self.config.check_interval_seconds)

    def show_status(self) -> None:
        """Show daemon status."""
        print("\n" + "=" * 70)
        print("PROMOTION DAEMON STATUS")
        print("=" * 70)

        print(f"\nLast check: {self.state.last_check or 'Never'}")
        print(f"Known models: {len(self.state.known_models)}")
        print(f"Evaluated models: {len(self.state.evaluated_models)}")
        print(f"Total promotions: {len(self.state.promotions)}")

        # Recent promotions
        if self.state.promotions:
            print("\nRecent promotions:")
            for p in self.state.promotions[-5:]:
                print(f"  - {p['model']} at {p['timestamp']}")

        # Config summary
        print(f"\nConfig:")
        print(f"  Check interval: {self.config.check_interval_seconds}s")
        print(f"  Games per baseline: {self.config.games_per_baseline}")
        print(f"  Min win rate: {self.config.min_win_rate:.0%}")
        print(f"  Cluster sync: {'enabled' if self.config.cluster_sync_enabled else 'disabled'}")
        print(f"  Slack notifications: {'configured' if self.config.slack_webhook_url else 'not configured'}")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Unified Model Promotion Daemon")
    parser.add_argument("--check-once", action="store_true", help="Run one check cycle")
    parser.add_argument("--daemon", action="store_true", help="Run as continuous daemon")
    parser.add_argument("--status", action="store_true", help="Show daemon status")
    parser.add_argument("--dry-run", action="store_true", help="Don't actually promote")
    parser.add_argument("--config", type=str, default=str(CONFIG_FILE), help="Config file path")

    args = parser.parse_args()

    # Load config
    config = DaemonConfig.from_yaml(Path(args.config))
    daemon = UnifiedPromotionDaemon(config)

    if args.status:
        daemon.show_status()
    elif args.daemon:
        daemon.run_daemon()
    elif args.check_once or not any([args.status, args.daemon]):
        daemon.check_once(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
