#!/usr/bin/env python3
"""Automated Model Promotion Pipeline.

Monitors ELO database and automatically promotes models that meet production criteria.
Handles:
- Detection of production-ready models
- Model file deployment to production paths
- Notification via Slack
- Rollback capability

Usage:
    python scripts/auto_promote.py              # Check and promote
    python scripts/auto_promote.py --daemon     # Run continuously
    python scripts/auto_promote.py --dry-run    # Preview without changes
"""

import argparse
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
AI_SERVICE_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(AI_SERVICE_ROOT))

from app.config.thresholds import (
    PRODUCTION_ELO_THRESHOLD,
    PRODUCTION_MIN_GAMES,
)

DEFAULT_DB = AI_SERVICE_ROOT / "data" / "unified_elo.db"
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
    """Find model file for a given model ID."""
    # Common patterns for model files
    patterns = [
        f"models/{model_id}.pt",
        f"models/nnue/{model_id}.pt",
        f"models/nnue_policy_{model_id}.pt",
        f"models/config_specific/{model_id}.pt",
    ]
    
    for pattern in patterns:
        path = AI_SERVICE_ROOT / pattern
        if path.exists():
            return path
    
    # Search recursively
    for pt_file in AI_SERVICE_ROOT.glob("models/**/*.pt"):
        if model_id in pt_file.stem:
            return pt_file
    
    return None


def get_production_candidates(db_path: Path) -> list[dict]:
    """Get models that meet production criteria."""
    if not db_path.exists():
        return []
    
    conn = sqlite3.connect(str(db_path))
    cursor = conn.execute("""
        SELECT participant_id, rating, games_played
        FROM elo_ratings
        WHERE rating >= ? AND games_played >= ?
          AND participant_id NOT LIKE 'baseline_%'
        ORDER BY rating DESC
    """, (PRODUCTION_ELO_THRESHOLD, PRODUCTION_MIN_GAMES))
    
    candidates = []
    for row in cursor.fetchall():
        model_id, rating, games = row
        candidates.append({
            "model_id": model_id,
            "rating": rating,
            "games": games,
        })
    
    conn.close()
    return candidates


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
            
            # Show closest candidates
            conn = sqlite3.connect(str(args.db))
            cursor = conn.execute("""
                SELECT participant_id, rating, games_played
                FROM elo_ratings
                WHERE participant_id NOT LIKE 'baseline_%'
                ORDER BY rating DESC
                LIMIT 5
            """)
            print("\nClosest candidates:")
            for row in cursor.fetchall():
                model_id, rating, games = row
                elo_needed = max(0, PRODUCTION_ELO_THRESHOLD - rating)
                games_needed = max(0, PRODUCTION_MIN_GAMES - games)
                print(f"  {model_id[:50]}")
                print(f"    ELO: {rating:.1f} (need +{elo_needed:.1f}), Games: {games} (need +{games_needed})")
            conn.close()


if __name__ == "__main__":
    main()
