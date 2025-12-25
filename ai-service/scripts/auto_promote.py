#!/usr/bin/env python3
"""Automated Model Promotion Pipeline.

Monitors ELO database and/or runs gauntlet evaluation to promote models.
Handles:
- ELO-based promotion (monitors database for high-ELO models)
- Gauntlet-based promotion (runs evaluation against baselines)
- Model file deployment to production paths
- Cluster synchronization
- Notification via Slack
- Rollback capability

Usage:
    # ELO-based promotion (monitors database)
    python scripts/auto_promote.py              # Check and promote
    python scripts/auto_promote.py --daemon     # Run continuously
    python scripts/auto_promote.py --dry-run    # Preview without changes

    # Gauntlet-based promotion (post-training workflow)
    python scripts/auto_promote.py --gauntlet --model models/my_model.pth \\
        --board-type hex8 --num-players 4 --games 50

    # Gauntlet with cluster sync
    python scripts/auto_promote.py --gauntlet --model models/my_model.pth \\
        --board-type hex8 --num-players 4 --sync-to-cluster
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
    MIN_WIN_RATE_VS_RANDOM,
    MIN_WIN_RATE_VS_HEURISTIC,
)

# Import PromotionController for unified promotion logic (December 2025)
try:
    from app.training.promotion_controller import (
        PromotionController,
        PromotionType,
        PromotionDecision,
        PromotionCriteria,
    )
    HAS_PROMOTION_CONTROLLER = True
except ImportError:
    HAS_PROMOTION_CONTROLLER = False
    PromotionController = None
    PromotionType = None

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
          AND participant_id NOT LIKE 'none:%'
          AND participant_id NOT LIKE '%:heuristic:%'
          AND participant_id NOT LIKE '%:random:%'
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
    """Run promotion check and promote eligible models.

    Uses PromotionController when available for unified promotion logic,
    with fallback to direct database checks.
    """
    history = load_promotion_history()
    promoted_ids = set(history.get("promoted", []))

    # Try to use PromotionController for unified logic (December 2025)
    if HAS_PROMOTION_CONTROLLER:
        return _run_promotion_via_controller(db_path, dry_run, promoted_ids, history)

    # Fallback to direct database checks
    return _run_promotion_direct(db_path, dry_run, promoted_ids, history)


def _run_promotion_via_controller(
    db_path: Path,
    dry_run: bool,
    promoted_ids: set,
    history: dict,
) -> list[str]:
    """Run promotion using PromotionController.

    Delegates to the controller for evaluation and execution,
    providing unified criteria and event emission.
    """
    candidates = get_production_candidates(db_path)
    newly_promoted = []
    controller = PromotionController()

    for candidate in candidates:
        model_id = candidate["model_id"]

        if model_id in promoted_ids:
            continue

        # Parse board type and players from model_id
        board_type, num_players = _parse_config_from_model_id(model_id)

        # Use controller to evaluate
        decision = controller.evaluate_promotion(
            model_id=model_id,
            board_type=board_type,
            num_players=num_players,
            promotion_type=PromotionType.PRODUCTION,
        )

        print(f"\nEvaluating: {model_id}")
        print(f"  ELO: {candidate['rating']:.1f}")
        print(f"  Games: {candidate['games']}")
        print(f"  Decision: {'PROMOTE' if decision.should_promote else 'SKIP'}")
        print(f"  Reason: {decision.reason}")

        if decision.should_promote:
            # Use local promotion logic for file operations
            # (controller doesn't know about our file layout)
            if promote_model(model_id, candidate["rating"], dry_run):
                if not dry_run:
                    promoted_ids.add(model_id)
                    newly_promoted.append(model_id)

                    send_notification(
                        f"Model `{model_id}` promoted to production!\n"
                        f"  ELO: {candidate['rating']:.1f}\n"
                        f"  Games: {candidate['games']}\n"
                        f"  Via: PromotionController",
                        "success"
                    )

    if not dry_run:
        history["promoted"] = list(promoted_ids)
        history["last_check"] = datetime.now().isoformat()
        save_promotion_history(history)

    return newly_promoted


def _run_promotion_direct(
    db_path: Path,
    dry_run: bool,
    promoted_ids: set,
    history: dict,
) -> list[str]:
    """Run promotion using direct database checks (legacy fallback)."""
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


def _parse_config_from_model_id(model_id: str) -> tuple[str, int]:
    """Parse board type and player count from model ID.

    Args:
        model_id: Model identifier (e.g., "square8_2p_v42", "hex8:policy:v3")

    Returns:
        Tuple of (board_type, num_players)
    """
    # Common config patterns
    for board in ["square8", "square19", "hex8", "hexagonal"]:
        if board in model_id.lower():
            for players in [2, 3, 4]:
                if f"{players}p" in model_id.lower():
                    return board, players
            return board, 2  # Default to 2 players

    # Fallback
    return "square8", 2


# =============================================================================
# Gauntlet-Based Promotion
# =============================================================================

def detect_model_type(model_path: Path) -> str:
    """Detect model type from checkpoint structure.

    GNN models have 'conv_type' or 'gnn' in path/checkpoint.
    Hybrid models have 'hybrid' in path or both CNN and GNN components.

    Args:
        model_path: Path to model checkpoint

    Returns:
        "gnn", "hybrid", or "cnn"
    """
    import torch

    path_str = str(model_path).lower()

    # Check path for hints
    if "gnn" in path_str and "hybrid" not in path_str:
        return "gnn"
    if "hybrid" in path_str:
        return "hybrid"

    # Check checkpoint contents
    try:
        ckpt = torch.load(model_path, map_location="cpu")
        if isinstance(ckpt, dict):
            # GNN models have conv_type in checkpoint
            if "conv_type" in ckpt:
                return "gnn"
            # Check state dict for GNN layer names
            state_dict = ckpt.get("model_state_dict", ckpt.get("state_dict", {}))
            if any("sage" in k or "gat" in k or "message" in k for k in state_dict.keys()):
                return "gnn"
    except Exception:
        pass

    return "cnn"


def run_gauntlet_evaluation(
    model_path: Path,
    board_type: str,
    num_players: int,
    games_per_opponent: int = 20,
    model_type: str | None = None,
) -> dict:
    """Run gauntlet evaluation and return results.

    Args:
        model_path: Path to model file
        board_type: Board type (e.g., "hex8", "square8")
        num_players: Number of players (2, 3, or 4)
        games_per_opponent: Games per baseline opponent
        model_type: Model type ("cnn", "gnn", "hybrid") or None to auto-detect

    Returns:
        Dict with gauntlet results including pass/fail status
    """
    # Lazy import to avoid circular dependencies
    from app.training.game_gauntlet import run_baseline_gauntlet, BaselineOpponent
    from app.models import BoardType as BT

    # Auto-detect model type if not specified
    if model_type is None:
        model_type = detect_model_type(model_path)

    # Map string to BoardType enum
    board_type_map = {
        "square8": BT.SQUARE8,
        "square19": BT.SQUARE19,
        "hex8": BT.HEX8,
        "hexagonal": BT.HEXAGONAL,
    }
    bt = board_type_map.get(board_type.lower())
    if bt is None:
        raise ValueError(f"Unknown board type: {board_type}")

    print(f"\nRunning gauntlet evaluation...")
    print(f"  Model: {model_path}")
    print(f"  Model type: {model_type.upper()}")
    print(f"  Config: {board_type}_{num_players}p")
    print(f"  Games per opponent: {games_per_opponent}")
    print()

    results = run_baseline_gauntlet(
        model_path=str(model_path),
        board_type=bt,
        num_players=num_players,
        opponents=[BaselineOpponent.RANDOM, BaselineOpponent.HEURISTIC],
        games_per_opponent=games_per_opponent,
        model_type=model_type,
    )

    # Extract win rates
    random_wr = results.opponent_results.get("random", {}).get("win_rate", 0)
    heuristic_wr = results.opponent_results.get("heuristic", {}).get("win_rate", 0)

    return {
        "model_path": str(model_path),
        "board_type": board_type,
        "num_players": num_players,
        "total_games": results.total_games,
        "total_wins": results.total_wins,
        "total_losses": results.total_losses,
        "total_draws": results.total_draws,
        "win_rate": results.win_rate,
        "win_rate_vs_random": random_wr,
        "win_rate_vs_heuristic": heuristic_wr,
        "passes_random": random_wr >= MIN_WIN_RATE_VS_RANDOM,
        "passes_heuristic": heuristic_wr >= MIN_WIN_RATE_VS_HEURISTIC,
        "passes_gauntlet": results.passes_baseline_gating,
        "estimated_elo": results.estimated_elo,
    }


def promote_after_gauntlet(
    model_path: Path,
    board_type: str,
    num_players: int,
    gauntlet_results: dict,
    dry_run: bool = False,
) -> bool:
    """Promote model to production after passing gauntlet.

    Args:
        model_path: Path to model file
        board_type: Board type
        num_players: Number of players
        gauntlet_results: Results from run_gauntlet_evaluation
        dry_run: If True, only preview without making changes

    Returns:
        True if promotion successful
    """
    config = f"{board_type}_{num_players}p"
    dest_dir = PRODUCTION_DIR / config

    # Use estimated ELO in filename
    elo = int(gauntlet_results.get("estimated_elo", 1500))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dest_filename = f"model_elo{elo}_{timestamp}.pth"
    dest_path = dest_dir / dest_filename

    if dry_run:
        print(f"\n[DRY-RUN] Would promote:")
        print(f"  Source: {model_path}")
        print(f"  Destination: {dest_path}")
        return True

    print(f"\nPromoting model to production...")
    dest_dir.mkdir(parents=True, exist_ok=True)

    # Copy model file
    shutil.copy2(model_path, dest_path)
    print(f"  ✓ Copied to {dest_path}")

    # Update latest symlink
    latest_link = dest_dir / "latest.pth"
    if latest_link.exists() or latest_link.is_symlink():
        latest_link.unlink()
    latest_link.symlink_to(dest_filename)
    print(f"  ✓ Updated latest symlink")

    # Also copy to canonical location
    canonical_path = AI_SERVICE_ROOT / "models" / f"canonical_{config}.pth"
    shutil.copy2(model_path, canonical_path)
    print(f"  ✓ Updated canonical model: {canonical_path}")

    # Update promotion history
    history = load_promotion_history()
    history["promoted"].append({
        "model_id": str(model_path.name),
        "config": config,
        "elo": elo,
        "timestamp": datetime.now().isoformat(),
        "method": "gauntlet",
        "gauntlet_results": gauntlet_results,
    })
    save_promotion_history(history)

    return True


def sync_model_to_cluster(
    model_path: Path,
    config: str,
    hosts: list[str] | None = None,
) -> int:
    """Sync promoted model to cluster nodes.

    Args:
        model_path: Path to model file
        config: Configuration string (e.g., "hex8_4p")
        hosts: Optional list of hosts to sync to

    Returns:
        Number of successful syncs
    """
    if hosts is None:
        # Default cluster hosts from config
        config_path = AI_SERVICE_ROOT / "config" / "distributed_hosts.yaml"
        if config_path.exists():
            import yaml
            with open(config_path) as f:
                cluster_config = yaml.safe_load(f)
            hosts = [h.get("host") for h in cluster_config.get("hosts", []) if h.get("host")]
        else:
            print("  Warning: No cluster config found")
            return 0

    success_count = 0
    for host in hosts[:10]:  # Limit to first 10 hosts
        remote_path = f"ubuntu@{host}:~/ringrift/ai-service/models/"
        try:
            result = subprocess.run(
                ["rsync", "-avz", "--timeout=30",
                 "-e", "ssh -i ~/.ssh/id_cluster -o ConnectTimeout=10 -o StrictHostKeyChecking=no",
                 str(model_path), remote_path],
                capture_output=True, text=True, timeout=60
            )
            if result.returncode == 0:
                print(f"  ✓ Synced to {host}")
                success_count += 1
            else:
                print(f"  ✗ Failed sync to {host}: {result.stderr[:100]}")
        except subprocess.TimeoutExpired:
            print(f"  ✗ Timeout syncing to {host}")
        except Exception as e:
            print(f"  ✗ Error syncing to {host}: {e}")

    return success_count


def run_gauntlet_promotion(
    model_path: Path,
    board_type: str,
    num_players: int,
    games_per_opponent: int = 20,
    sync_to_cluster: bool = False,
    dry_run: bool = False,
    model_type: str | None = None,
) -> bool:
    """Full gauntlet-based promotion workflow.

    Args:
        model_path: Path to model file
        board_type: Board type
        num_players: Number of players
        games_per_opponent: Games per baseline opponent
        sync_to_cluster: Whether to sync to cluster after promotion
        dry_run: If True, only preview without making changes
        model_type: Model type ("cnn", "gnn", "hybrid") or None to auto-detect

    Returns:
        True if promotion successful, False otherwise
    """
    print("=" * 60)
    print("GAUNTLET-BASED MODEL PROMOTION")
    print("=" * 60)

    if not model_path.exists():
        print(f"Error: Model file not found: {model_path}")
        return False

    # Run gauntlet evaluation
    try:
        results = run_gauntlet_evaluation(
            model_path, board_type, num_players, games_per_opponent,
            model_type=model_type,
        )
    except Exception as e:
        print(f"Error running gauntlet: {e}")
        return False

    # Print results
    print("\nGauntlet Results:")
    print(f"  Total games: {results['total_games']}")
    print(f"  Wins: {results['total_wins']}, Losses: {results['total_losses']}, Draws: {results['total_draws']}")
    print(f"  Overall win rate: {results['win_rate']:.1%}")
    print()
    print(f"  vs RANDOM: {results['win_rate_vs_random']:.1%} "
          f"({'✓ PASS' if results['passes_random'] else '✗ FAIL'}, need {MIN_WIN_RATE_VS_RANDOM:.0%})")
    print(f"  vs HEURISTIC: {results['win_rate_vs_heuristic']:.1%} "
          f"({'✓ PASS' if results['passes_heuristic'] else '✗ FAIL'}, need {MIN_WIN_RATE_VS_HEURISTIC:.0%})")
    print(f"  Estimated ELO: {results['estimated_elo']:.0f}")
    print()

    if not results['passes_gauntlet']:
        print("✗ GAUNTLET FAILED - Model not promoted")
        if not results['passes_random']:
            print(f"  Need {MIN_WIN_RATE_VS_RANDOM:.0%} vs random, got {results['win_rate_vs_random']:.1%}")
        if not results['passes_heuristic']:
            print(f"  Need {MIN_WIN_RATE_VS_HEURISTIC:.0%} vs heuristic, got {results['win_rate_vs_heuristic']:.1%}")
        return False

    print("✓ GAUNTLET PASSED - Promoting model...")

    # Promote model
    success = promote_after_gauntlet(
        model_path, board_type, num_players, results, dry_run
    )

    if not success:
        return False

    # Optionally sync to cluster
    config = f"{board_type}_{num_players}p"
    if sync_to_cluster and not dry_run:
        print("\nSyncing to cluster...")
        canonical_path = AI_SERVICE_ROOT / "models" / f"canonical_{config}.pth"
        synced = sync_model_to_cluster(canonical_path, config)
        print(f"  Synced to {synced} nodes")

    # Send notification
    if not dry_run:
        send_notification(
            f"Model promoted via gauntlet!\n"
            f"  Config: {config}\n"
            f"  ELO: {results['estimated_elo']:.0f}\n"
            f"  vs Random: {results['win_rate_vs_random']:.1%}\n"
            f"  vs Heuristic: {results['win_rate_vs_heuristic']:.1%}",
            "success"
        )

    print("\n" + "=" * 60)
    print("✓ PROMOTION COMPLETE")
    print("=" * 60)

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Automated Model Promotion",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # ELO-based promotion (check database)
  python scripts/auto_promote.py

  # Gauntlet-based promotion (post-training)
  python scripts/auto_promote.py --gauntlet --model models/my_model.pth \\
      --board-type hex8 --num-players 4

  # Gauntlet with cluster sync
  python scripts/auto_promote.py --gauntlet --model models/my_model.pth \\
      --board-type hex8 --num-players 4 --sync-to-cluster --games 50
        """
    )

    # ELO-based promotion args
    parser.add_argument("--db", type=Path, default=DEFAULT_DB, help="ELO database path")
    parser.add_argument("--daemon", action="store_true", help="Run ELO check continuously")
    parser.add_argument("--interval", type=int, default=300, help="Check interval in seconds")
    parser.add_argument("--dry-run", action="store_true", help="Preview without changes")

    # Gauntlet-based promotion args
    parser.add_argument("--gauntlet", action="store_true",
                        help="Use gauntlet evaluation instead of ELO database")
    parser.add_argument("--model", type=Path, help="Model file to evaluate (required for --gauntlet)")
    parser.add_argument("--board-type", choices=["square8", "square19", "hex8", "hexagonal"],
                        help="Board type for gauntlet (required for --gauntlet)")
    parser.add_argument("--num-players", type=int, choices=[2, 3, 4],
                        help="Number of players for gauntlet (required for --gauntlet)")
    parser.add_argument("--games", type=int, default=20,
                        help="Games per opponent in gauntlet (default: 20)")
    parser.add_argument("--model-type", choices=["cnn", "gnn", "hybrid"],
                        help="Model type (auto-detected if not specified)")
    parser.add_argument("--sync-to-cluster", action="store_true",
                        help="Sync promoted model to cluster nodes")

    args = parser.parse_args()

    # Validate gauntlet args
    if args.gauntlet:
        if not args.model:
            parser.error("--gauntlet requires --model")
        if not args.board_type:
            parser.error("--gauntlet requires --board-type")
        if not args.num_players:
            parser.error("--gauntlet requires --num-players")

    # Branch between gauntlet mode and ELO mode
    if args.gauntlet:
        # Gauntlet-based promotion
        success = run_gauntlet_promotion(
            model_path=args.model,
            board_type=args.board_type,
            num_players=args.num_players,
            games_per_opponent=args.games,
            sync_to_cluster=args.sync_to_cluster,
            dry_run=args.dry_run,
            model_type=getattr(args, 'model_type', None),
        )
        sys.exit(0 if success else 1)

    # ELO-based promotion (original behavior)
    print("=" * 60)
    print("ELO-BASED MODEL PROMOTION")
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
        if not args.db.exists():
            print(f"Database not found: {args.db}")
            print("Use --gauntlet mode for post-training promotion without ELO database.")
            sys.exit(1)

        candidates = get_production_candidates(args.db)
        print(f"Production-ready candidates: {len(candidates)}")

        if candidates:
            promoted = run_promotion(args.db, args.dry_run)
            print(f"\nNewly promoted: {len(promoted)}")
        else:
            print("No models meet production criteria yet.")

            # Show closest candidates
            try:
                conn = sqlite3.connect(str(args.db))
                cursor = conn.execute("""
                    SELECT participant_id, rating, games_played
                    FROM elo_ratings
                    WHERE participant_id NOT LIKE 'baseline_%'
                      AND participant_id NOT LIKE 'none:%'
                      AND participant_id NOT LIKE '%:heuristic:%'
                      AND participant_id NOT LIKE '%:random:%'
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
            except sqlite3.OperationalError as e:
                print(f"Could not query database: {e}")


if __name__ == "__main__":
    main()
