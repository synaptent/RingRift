#!/usr/bin/env python3
"""Automated Model Promotion Pipeline.

.. deprecated:: 2025-12-20
    This script is now consolidated into unified_promotion_daemon.py.
    Use the unified daemon instead::

        python scripts/unified_promotion_daemon.py --check-once     # Check and promote
        python scripts/unified_promotion_daemon.py --daemon         # Run continuously
        python scripts/unified_promotion_daemon.py --dry-run        # Preview without changes

    This standalone script remains for backward compatibility but will emit
    a deprecation warning when run directly.

Monitors ELO database and automatically promotes models that meet production criteria.
Handles:
- Detection of production-ready models
- Model file deployment to production paths
- Notification via Slack
- Rollback capability

Usage (deprecated):
    python scripts/auto_promote.py              # Check and promote
    python scripts/auto_promote.py --daemon     # Run continuously
    python scripts/auto_promote.py --dry-run    # Preview without changes

Preferred usage:
    python scripts/unified_promotion_daemon.py --check-once
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
AI_SERVICE_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(AI_SERVICE_ROOT))

from scripts.lib.elo_queries import (
    DEFAULT_DB,
    PRODUCTION_ELO_THRESHOLD,
    PRODUCTION_MIN_GAMES,
    get_production_candidates as _get_production_candidates,
    get_top_models,
)
PRODUCTION_DIR = AI_SERVICE_ROOT / "models" / "production"
PROMOTION_LOG = AI_SERVICE_ROOT / "data" / ".promotion_history.json"


def get_slack_webhook():
    """Get Slack webhook URL."""
    webhook = os.environ.get("RINGRIFT_SLACK_WEBHOOK") or os.environ.get("SLACK_WEBHOOK_URL")
    if webhook:
        return webhook
    webhook_file = Path.home() / ".ringrift_slack_webhook"
    if webhook_file.exists():
        return webhook_file.read_text().strip()
    return None


def send_notification(message: str, alert_type: str = "success"):
    """Send notification via Slack or stdout."""
    webhook = get_slack_webhook()
    emoji = {
        "success": ":rocket:",
        "warning": ":warning:",
        "error": ":x:",
    }.get(alert_type, ":robot_face:")
    
    if webhook:
        payload = {
            "text": f"{emoji} *Model Promotion*\n{message}",
            "username": "Promotion Bot",
        }
        try:
            subprocess.run(
                ["curl", "-s", "-X", "POST", "-H", "Content-Type: application/json",
                 "-d", json.dumps(payload), webhook],
                capture_output=True, timeout=10
            )
        except Exception as e:
            print(f"Slack notification failed: {e}")
    
    print(f"[{alert_type.upper()}] {message}")


def load_promotion_history() -> dict:
    """Load promotion history."""
    if PROMOTION_LOG.exists():
        try:
            return json.loads(PROMOTION_LOG.read_text())
        except Exception:
            pass
    return {"promoted": [], "last_check": None}


def save_promotion_history(history: dict):
    """Save promotion history."""
    PROMOTION_LOG.parent.mkdir(parents=True, exist_ok=True)
    PROMOTION_LOG.write_text(json.dumps(history, indent=2, default=str))


def find_model_file(model_id: str) -> Path | None:
    """Find model file for a given model ID.

    Searches for both .pt and .pth files in models/ and models/archived/.
    Uses fuzzy matching for timestamp-based model names.
    """
    # Common patterns for model files (both .pt and .pth extensions)
    patterns = [
        f"models/{model_id}.pth",
        f"models/{model_id}.pt",
        f"models/archived/{model_id}.pth",
        f"models/archived/{model_id}.pt",
        f"models/nnue/{model_id}.pt",
        f"models/nnue/{model_id}.pth",
        f"models/nnue_policy_{model_id}.pt",
        f"models/config_specific/{model_id}.pt",
    ]

    for pattern in patterns:
        path = AI_SERVICE_ROOT / pattern
        if path.exists():
            return path

    # Search recursively for .pth files (most common format)
    for pth_file in AI_SERVICE_ROOT.glob("models/**/*.pth"):
        if model_id in pth_file.stem:
            return pth_file

    # Also search for .pt files
    for pt_file in AI_SERVICE_ROOT.glob("models/**/*.pt"):
        if model_id in pt_file.stem:
            return pt_file

    # Fuzzy matching: try prefix match for timestamp-based names
    # e.g., policy_sq8_2p_20251217_205153 might match policy_sq8_2p_20251217_205331
    # Extract base pattern (everything before last timestamp segment)
    parts = model_id.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit() and len(parts[1]) == 6:
        prefix = parts[0]  # e.g., "policy_sq8_2p_20251217"
        candidates = list(AI_SERVICE_ROOT.glob(f"models/**/{prefix}_*.pth"))
        candidates.extend(AI_SERVICE_ROOT.glob(f"models/**/{prefix}_*.pt"))
        if candidates:
            # Return the closest timestamp match
            target_ts = int(parts[1])
            best_match = None
            best_diff = float("inf")
            for c in candidates:
                try:
                    c_parts = c.stem.rsplit("_", 1)
                    if len(c_parts) == 2 and c_parts[1].isdigit():
                        diff = abs(int(c_parts[1]) - target_ts)
                        if diff < best_diff:
                            best_diff = diff
                            best_match = c
                except (ValueError, IndexError):
                    continue
            if best_match and best_diff < 10000:  # Within ~1 hour tolerance
                return best_match

    return None


def get_production_candidates(db_path: Path) -> list[dict]:
    """Get models that meet production criteria using unified query library."""
    models = _get_production_candidates(db_path, include_baselines=False)
    return [
        {
            "model_id": m.participant_id,
            "rating": m.rating,
            "games": m.games_played,
        }
        for m in models
    ]


def promote_model(model_id: str, rating: float, dry_run: bool = False) -> bool:
    """Promote a model to production."""
    model_path = find_model_file(model_id)
    
    if not model_path:
        print(f"  Warning: Model file not found for {model_id}")
        return False
    
    # Determine config from model name
    config = "default"
    for cfg in ["square8_2p", "square8_3p", "square8_4p", 
                "square19_2p", "square19_3p", "square19_4p",
                "hex8_2p", "hex8_3p", "hex8_4p",
                "hexagonal_2p", "hexagonal_3p", "hexagonal_4p"]:
        if cfg.replace("_", "") in model_id.lower() or cfg in model_id.lower():
            config = cfg
            break
    
    dest_dir = PRODUCTION_DIR / config
    dest_path = dest_dir / f"model_elo{int(rating)}.pt"
    
    if dry_run:
        print(f"  [DRY-RUN] Would copy {model_path} -> {dest_path}")
        return True
    
    dest_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(model_path, dest_path)
    
    # Create symlink to latest
    latest_link = dest_dir / "latest.pt"
    if latest_link.exists():
        latest_link.unlink()
    latest_link.symlink_to(dest_path.name)
    
    return True


def run_promotion(db_path: Path, dry_run: bool = False) -> list[str]:
    """Run promotion check and promote eligible models."""
    history = load_promotion_history()
    promoted_ids = set(history.get("promoted", []))
    
    candidates = get_production_candidates(db_path)
    newly_promoted = []
    
    for candidate in candidates:
        model_id = candidate["model_id"]
        
        if model_id in promoted_ids:
            continue
        
        print(f"\nPromoting: {model_id}")
        print(f"  ELO: {candidate['rating']:.1f}")
        print(f"  Games: {candidate['games']}")
        
        if promote_model(model_id, candidate["rating"], dry_run):
            if not dry_run:
                promoted_ids.add(model_id)
                newly_promoted.append(model_id)
                
                send_notification(
                    f"Model `{model_id}` promoted to production!\n"
                    f"  ELO: {candidate['rating']:.1f}\n"
                    f"  Games: {candidate['games']}",
                    "success"
                )
    
    if not dry_run:
        history["promoted"] = list(promoted_ids)
        history["last_check"] = datetime.now().isoformat()
        save_promotion_history(history)
    
    return newly_promoted


def main():
    import warnings
    warnings.warn(
        "auto_promote.py is deprecated. "
        "Use 'unified_promotion_daemon.py --check-once' or '--daemon' instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    print("WARNING: This script is deprecated. Use 'unified_promotion_daemon.py' instead.\n")

    parser = argparse.ArgumentParser(description="Automated Model Promotion")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB, help="ELO database path")
    parser.add_argument("--daemon", action="store_true", help="Run continuously")
    parser.add_argument("--interval", type=int, default=300, help="Check interval (seconds)")
    parser.add_argument("--dry-run", action="store_true", help="Preview without changes")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("AUTOMATED MODEL PROMOTION")
    print("=" * 60)
    print(f"Database: {args.db}")
    print(f"Production threshold: ELO >= {PRODUCTION_ELO_THRESHOLD}, Games >= {PRODUCTION_MIN_GAMES}")
    print(f"Production directory: {PRODUCTION_DIR}")
    if args.dry_run:
        print("[DRY-RUN MODE]")
    print()
    
    if args.daemon:
        print(f"Running in daemon mode (interval: {args.interval}s)")
        while True:
            try:
                promoted = run_promotion(args.db, args.dry_run)
                if promoted:
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] Promoted {len(promoted)} model(s)")
                else:
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] No new promotions")
            except Exception as e:
                print(f"Error: {e}")
            time.sleep(args.interval)
    else:
        candidates = get_production_candidates(args.db)
        print(f"Production-ready candidates: {len(candidates)}")
        
        if candidates:
            promoted = run_promotion(args.db, args.dry_run)
            print(f"\nNewly promoted: {len(promoted)}")
        else:
            print("No models meet production criteria yet.")

            # Show closest candidates using unified query
            top_models = get_top_models(args.db, limit=5, include_baselines=False)
            print("\nClosest candidates:")
            for model in top_models:
                elo_needed = max(0, PRODUCTION_ELO_THRESHOLD - model.rating)
                games_needed = max(0, PRODUCTION_MIN_GAMES - model.games_played)
                print(f"  {model.participant_id[:50]}")
                print(f"    ELO: {model.rating:.1f} (need +{elo_needed:.1f}), Games: {model.games_played} (need +{games_needed})")


if __name__ == "__main__":
    main()
