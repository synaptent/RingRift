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
from __future__ import annotations


import argparse
import json
import os
import shutil
import signal
import sqlite3
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# Process management for singleton lock and signal handling
try:
    from scripts.lib.process import SingletonLock, SignalHandler
    HAS_PROCESS_UTILS = True
except ImportError:
    HAS_PROCESS_UTILS = False
    SingletonLock = None
    SignalHandler = None

SCRIPT_DIR = Path(__file__).parent
AI_SERVICE_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(AI_SERVICE_ROOT))

from app.config.thresholds import (
    PRODUCTION_ELO_THRESHOLD,
    PRODUCTION_MIN_GAMES,
    MIN_WIN_RATE_VS_RANDOM,
    MIN_WIN_RATE_VS_HEURISTIC,
    get_min_win_rate_vs_random,
    get_min_win_rate_vs_heuristic,
    should_promote_model,
    get_promotion_thresholds,
    get_minimum_thresholds,
    get_gauntlet_games_per_opponent,  # December 2025: Dynamic games per player count
)
from app.utils.name_generator import generate_model_name, name_from_checkpoint_hash
from app.utils.game_discovery import count_games_for_config

# Import gauntlet resource limits (Dec 2025)
try:
    from scripts.p2p.constants import (
        MAX_CONCURRENT_GAUNTLETS,
        MAX_GAUNTLET_PROCESSES,
        MIN_MEMORY_GB_FOR_GAUNTLET,
        MAX_CPU_LOAD_RATIO_FOR_GAUNTLET,
    )
except ImportError:
    # Fallback defaults if constants not available
    MAX_CONCURRENT_GAUNTLETS = 3
    MAX_GAUNTLET_PROCESSES = 64
    MIN_MEMORY_GB_FOR_GAUNTLET = 16
    MAX_CPU_LOAD_RATIO_FOR_GAUNTLET = 0.8

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

# Import event emission for feedback loop (December 2025)
try:
    from app.coordination.event_emission_helpers import safe_emit_event
    HAS_EVENT_EMITTERS = True
except ImportError:
    HAS_EVENT_EMITTERS = False
    safe_emit_event = None  # type: ignore

# Import EloService for hash-based model identity tracking (January 2026)
try:
    from app.training.elo_service import get_elo_service
    HAS_ELO_SERVICE = True
except ImportError:
    HAS_ELO_SERVICE = False
    get_elo_service = None

DEFAULT_DB = AI_SERVICE_ROOT / "data" / "unified_elo.db"
PRODUCTION_DIR = AI_SERVICE_ROOT / "models" / "production"
PROMOTION_LOG = AI_SERVICE_ROOT / "data" / ".promotion_history.json"

# Gauntlet timeout protection (Dec 2025)
# Prevents indefinitely hanging evaluations
GAUNTLET_TIMEOUT_SECONDS = int(os.environ.get("RINGRIFT_GAUNTLET_TIMEOUT", "1800"))  # 30 min default

# Track child processes for cleanup on signal (Dec 2025)
_active_child_processes: list = []


def _cleanup_child_processes():
    """Clean up any active child processes on shutdown.

    Called by SignalHandler when receiving SIGTERM/SIGINT.
    Prevents zombie processes from accumulating on cluster nodes.
    """
    global _active_child_processes

    for proc in _active_child_processes:
        try:
            if proc.poll() is None:  # Still running
                print(f"  Terminating child process {proc.pid}")
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    print(f"  Force-killing process {proc.pid}")
                    proc.kill()
                    proc.wait(timeout=2)
        except (ProcessLookupError, OSError) as e:
            print(f"  Process {getattr(proc, 'pid', 'unknown')} cleanup error: {e}")

    _active_child_processes.clear()
    print("  Child process cleanup complete")


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
        except (json.JSONDecodeError, OSError):
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
    # Use unique name instead of ELO (ELO can change over time)
    unique_name = generate_model_name(include_timestamp=True)
    dest_path = dest_dir / f"{unique_name}.pt"
    
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
        from app.utils.torch_utils import safe_load_checkpoint
        ckpt = safe_load_checkpoint(model_path, map_location="cpu")
        if isinstance(ckpt, dict):
            # GNN models have conv_type in checkpoint
            if "conv_type" in ckpt:
                return "gnn"
            # Check state dict for GNN layer names
            state_dict = ckpt.get("model_state_dict", ckpt.get("state_dict", {}))
            if any("sage" in k or "gat" in k or "message" in k for k in state_dict.keys()):
                return "gnn"
    except (RuntimeError, OSError, KeyError, AttributeError):
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

    # Use multi-tier promotion system from thresholds.py
    # This considers Elo-adaptive, game-count graduated, aspirational, AND minimum floor thresholds
    config_key = f"{board_type}_{num_players}p"

    # Check if this model beats the current best (for relative promotion)
    # For now, assume False - caller can override if known
    beats_current_best = False

    # Dec 31, 2025: Get training game count for graduated thresholds
    try:
        training_game_count = count_games_for_config(board_type, num_players)
    except Exception:
        training_game_count = None  # Fallback if count fails

    # Get promotion decision using sophisticated multi-tier system
    # Dec 30, 2025: Pass model_elo to enable Elo-adaptive thresholds
    # This allows bootstrap models (800-1200 Elo) to pass with lower thresholds
    model_elo = results.estimated_elo if results.estimated_elo > 0 else None
    should_promote, promotion_reason = should_promote_model(
        config_key=config_key,
        vs_random_rate=random_wr,
        vs_heuristic_rate=heuristic_wr,
        beats_current_best=beats_current_best,
        model_elo=model_elo,
        game_count=training_game_count,
    )

    # Get thresholds for display purposes
    aspirational = get_promotion_thresholds(config_key)
    minimum = get_minimum_thresholds(config_key)

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
        # Use two-tier threshold check instead of fixed constants
        "passes_random": random_wr >= minimum["vs_random"],
        "passes_heuristic": heuristic_wr >= minimum["vs_heuristic"],
        "passes_gauntlet": should_promote,  # Use two-tier decision
        "estimated_elo": results.estimated_elo,
        # Additional context for promotion decision
        "promotion_reason": promotion_reason,
        "aspirational_thresholds": aspirational,
        "minimum_thresholds": minimum,
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
    # Pre-promotion validation (added Dec 2025)
    # Validates that checkpoint metadata matches the target configuration
    try:
        from app.models import BoardType
        from app.training.model_config_contract import (
            ModelConfigContract,
            validate_checkpoint_for_promotion,
        )
        from app.training.model_versioning import load_versioned_checkpoint

        # Load checkpoint metadata
        _, metadata = load_versioned_checkpoint(str(model_path))
        metadata_dict = metadata.to_dict()

        # Validate against target configuration
        board_type_enum = BoardType[board_type.upper()]
        is_valid, violations = validate_checkpoint_for_promotion(
            metadata_dict, board_type_enum, num_players
        )

        if not is_valid:
            print(f"\n✗ Cannot promote - configuration violations:")
            for v in violations:
                print(f"  - {v}")
            print(f"\nModel configuration does not match target {board_type}_{num_players}p")
            print("Please retrain with correct configuration.")
            return False

        print(f"✓ Model config validated for {board_type}_{num_players}p")

    except Exception as e:
        # If validation fails due to missing metadata, warn but allow promotion
        # (for backwards compatibility with legacy checkpoints)
        print(f"⚠ Warning: Could not validate checkpoint metadata: {e}")
        print("  Proceeding with promotion (legacy checkpoint assumed)")

    config = f"{board_type}_{num_players}p"
    dest_dir = PRODUCTION_DIR / config

    # Use unique memorable name instead of ELO (ELO can change over time)
    # Generate deterministic name from model content for reproducibility
    unique_name = name_from_checkpoint_hash(str(model_path))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dest_filename = f"{unique_name}_{timestamp}.pth"
    dest_path = dest_dir / dest_filename

    # Store ELO in metadata file alongside model for reference
    elo = int(gauntlet_results.get("estimated_elo", 1500))

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

    # Save metadata alongside model (ELO, gauntlet results, etc.)
    metadata_path = dest_path.with_suffix(".json")
    metadata = {
        "name": unique_name,
        "config": config,
        "estimated_elo": elo,
        "timestamp": timestamp,
        "source_model": str(model_path.name),
        "gauntlet_results": gauntlet_results,
    }
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  ✓ Saved metadata to {metadata_path.name}")

    # Update latest symlink
    latest_link = dest_dir / "latest.pth"
    if latest_link.exists() or latest_link.is_symlink():
        latest_link.unlink()
    latest_link.symlink_to(dest_filename)
    print(f"  ✓ Updated latest symlink")

    # Also copy to canonical location
    canonical_path = AI_SERVICE_ROOT / "models" / f"canonical_{config}.pth"

    # January 2026: Hash-based model identity tracking before promotion
    # This fixes stale Elo where canonical models appear weak because their
    # Elo was computed with an older model version
    if HAS_ELO_SERVICE and get_elo_service is not None:
        try:
            elo_service = get_elo_service()
            # Source participant is the model being promoted (e.g., "ringrift_best_hex8_2p")
            source_participant_id = f"ringrift_best_{config}"
            # Target participant is the canonical model (use ringrift_best_ for stable Elo tracking)
            target_participant_id = f"ringrift_best_{config}"

            promotion_result = elo_service.handle_model_promotion(
                source_model_path=str(model_path),
                target_model_path=str(canonical_path),
                source_participant_id=source_participant_id,
                target_participant_id=target_participant_id,
                board_type=board_type,
                num_players=num_players,
            )

            if promotion_result["status"] == "error":
                print(f"  ⚠ Elo tracking warning: {promotion_result['message']}")
            elif promotion_result["elo_transferred"]:
                print(f"  ✓ Elo tracking: {promotion_result['message']}")
            elif promotion_result["elo_reset"]:
                print(f"  ⚠ Elo tracking: {promotion_result['message']}")
            # else: no change or new model, no special message needed
        except Exception as e:
            print(f"  ⚠ Elo tracking failed (non-fatal): {e}")

    shutil.copy2(model_path, canonical_path)
    print(f"  ✓ Updated canonical model: {canonical_path}")

    # January 2026: Update model identity tracking for target after copy
    if HAS_ELO_SERVICE and get_elo_service is not None:
        try:
            elo_service = get_elo_service()
            target_participant_id = f"ringrift_best_{config}"
            elo_service.register_model(
                model_id=target_participant_id,
                board_type=board_type,
                num_players=num_players,
                model_path=str(canonical_path),
            )
        except Exception as e:
            print(f"  ⚠ Elo identity update failed (non-fatal): {e}")

    # Create ringrift_best_* symlink for inference (December 2025)
    symlink_path = AI_SERVICE_ROOT / "models" / f"ringrift_best_{config}.pth"
    if symlink_path.exists() or symlink_path.is_symlink():
        symlink_path.unlink()
    symlink_path.symlink_to(f"canonical_{config}.pth")
    print(f"  ✓ Created inference symlink: {symlink_path.name} -> canonical_{config}.pth")

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

    # Emit PROMOTION_COMPLETE event for feedback loop (December 2025)
    # This notifies the curriculum system and triggers model distribution
    if HAS_EVENT_EMITTERS and safe_emit_event is not None:
        emitted = safe_emit_event(
            "PROMOTION_COMPLETE",
            {
                "model_id": str(model_path.name),
                "board_type": board_type,
                "num_players": num_players,
                "promotion_type": "gauntlet",
                "elo_improvement": elo - 1500.0 if elo else None,  # Delta from baseline
                "model_path": str(canonical_path),
                "win_rate_vs_random": gauntlet_results.get("win_rate_vs_random"),
                "win_rate_vs_heuristic": gauntlet_results.get("win_rate_vs_heuristic"),
            },
            context="auto_promote",
        )
        if emitted:
            print("  ✓ Emitted PROMOTION_COMPLETE event")

    return True


def sync_model_to_cluster(
    model_path: Path,
    config: str,
    hosts: list[str] | None = None,
) -> int:
    """Sync promoted model to cluster nodes with verification.

    December 2025: Updated to use rsync_push_file_verified for integrity
    verification after transfer. Prevents silent corruption on flaky connections.

    Args:
        model_path: Path to model file
        config: Configuration string (e.g., "hex8_4p")
        hosts: Optional list of hosts to sync to

    Returns:
        Number of successful syncs (with verification)
    """
    from app.distributed.hosts import HostConfig
    from app.distributed.sync_utils import rsync_push_file_verified

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
        try:
            # Create HostConfig for verified rsync
            host_config = HostConfig(
                name=host,
                ssh_host=host,
                ssh_user="ubuntu",
                ssh_port=22,
                ssh_key="~/.ssh/id_cluster",
            )

            # Use verified rsync with checksum validation
            result = rsync_push_file_verified(
                host=host_config,
                local_path=model_path,
                remote_path=f"~/ringrift/ai-service/models/{model_path.name}",
                timeout=120,
            )

            if result.success and result.verified:
                print(f"  ✓ Synced to {host} (checksum verified)")
                success_count += 1
            elif result.success:
                print(f"  ⚠ Synced to {host} (unverified: {result.error})")
            else:
                print(f"  ✗ Failed sync to {host}: {result.error}")

        except subprocess.TimeoutExpired:
            print(f"  ✗ Timeout syncing to {host}")
        except Exception as e:
            print(f"  ✗ Error syncing to {host}: {e}")

    return success_count


# ============================================
# Resource Check Functions (Dec 2025)
# ============================================
# Prevent resource exhaustion from concurrent gauntlet evaluations.
# These checks run before starting a gauntlet to ensure the node has capacity.


def count_gauntlet_processes() -> int:
    """Count running gauntlet-related processes.

    Looks for auto_promote.py, game_gauntlet, and related patterns
    to determine how many gauntlet evaluations are currently running.
    """
    try:
        import psutil
    except ImportError:
        return 0

    patterns = ["auto_promote", "game_gauntlet", "run_gauntlet"]
    count = 0

    try:
        for proc in psutil.process_iter(['pid', 'cmdline']):
            try:
                cmdline = " ".join(proc.info.get('cmdline') or [])
                if any(p in cmdline for p in patterns):
                    count += 1
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
    except (RuntimeError, OSError):
        pass

    return count


def check_gauntlet_resources(skip_check: bool = False) -> tuple[bool, str]:
    """Check if system has resources for gauntlet execution.

    Args:
        skip_check: If True, skip all checks and return OK

    Returns:
        Tuple of (ok, message) where ok is True if resources are available
    """
    if skip_check:
        return True, "Resource check skipped"

    try:
        import psutil
    except ImportError:
        # psutil not available, allow gauntlet to run
        return True, "psutil not available, skipping resource check"

    # Check available memory
    try:
        mem = psutil.virtual_memory()
        available_gb = mem.available / (1024**3)
        if available_gb < MIN_MEMORY_GB_FOR_GAUNTLET:
            return False, f"Insufficient memory: {available_gb:.1f}GB < {MIN_MEMORY_GB_FOR_GAUNTLET}GB required"
    except Exception as e:
        print(f"Warning: Could not check memory: {e}")

    # Check CPU load
    try:
        load_avg = os.getloadavg()[0]
        cpu_count = os.cpu_count() or 1
        load_ratio = load_avg / cpu_count
        if load_ratio > MAX_CPU_LOAD_RATIO_FOR_GAUNTLET:
            return False, f"High CPU load: {load_ratio:.2f} > {MAX_CPU_LOAD_RATIO_FOR_GAUNTLET} (load={load_avg:.1f}, cpus={cpu_count})"
    except Exception as e:
        print(f"Warning: Could not check CPU load: {e}")

    # Check concurrent gauntlets
    try:
        gauntlet_count = count_gauntlet_processes()
        if gauntlet_count >= MAX_CONCURRENT_GAUNTLETS:
            return False, f"Too many concurrent gauntlets: {gauntlet_count} >= {MAX_CONCURRENT_GAUNTLETS}"
    except Exception as e:
        print(f"Warning: Could not count gauntlet processes: {e}")

    return True, "OK"


class GauntletTimeoutError(Exception):
    """Raised when gauntlet evaluation exceeds timeout."""
    pass


def _run_gauntlet_with_timeout(
    model_path: Path,
    board_type: str,
    num_players: int,
    games_per_opponent: int,
    model_type: str | None = None,
    timeout_seconds: int | None = None,
) -> dict | None:
    """Run gauntlet evaluation with timeout protection.

    Uses SIGALRM on Unix systems to enforce timeout.

    Args:
        model_path: Path to model file
        board_type: Board type
        num_players: Number of players
        games_per_opponent: Games per baseline opponent
        model_type: Model type or None to auto-detect
        timeout_seconds: Timeout in seconds (default: GAUNTLET_TIMEOUT_SECONDS)

    Returns:
        Gauntlet results dict

    Raises:
        GauntletTimeoutError: If evaluation exceeds timeout
    """
    timeout = timeout_seconds or GAUNTLET_TIMEOUT_SECONDS

    def _timeout_handler(signum, frame):
        raise GauntletTimeoutError(f"Gauntlet evaluation timed out after {timeout}s")

    # Set up alarm on Unix systems
    old_handler = None
    if hasattr(signal, 'SIGALRM'):
        old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(timeout)

    try:
        return run_gauntlet_evaluation(
            model_path, board_type, num_players, games_per_opponent,
            model_type=model_type,
        )
    finally:
        # Cancel alarm and restore handler
        if hasattr(signal, 'SIGALRM'):
            signal.alarm(0)
            if old_handler:
                signal.signal(signal.SIGALRM, old_handler)


def run_gauntlet_with_retry(
    model_path: Path,
    board_type: str,
    num_players: int,
    games_per_opponent: int = 20,
    model_type: str | None = None,
    max_retries: int = 3,
    retry_delay: float = 5.0,
    timeout_seconds: int | None = None,
) -> dict | None:
    """Run gauntlet evaluation with retry and timeout logic.

    On failure, retries with increased game count to reduce variance.
    This helps overcome transient failures and statistical variance.

    Args:
        model_path: Path to model file
        board_type: Board type
        num_players: Number of players
        games_per_opponent: Base games per baseline opponent
        model_type: Model type or None to auto-detect
        max_retries: Maximum retry attempts (default: 3)
        retry_delay: Delay between retries in seconds
        timeout_seconds: Timeout per attempt in seconds (default: GAUNTLET_TIMEOUT_SECONDS)

    Returns:
        Gauntlet results dict or None if all attempts failed
    """
    last_error = None
    timeout = timeout_seconds or GAUNTLET_TIMEOUT_SECONDS

    for attempt in range(max_retries):
        # Increase games on retry to reduce variance
        # Attempt 0: base games, Attempt 1: 1.5x, Attempt 2: 2x
        retry_games = int(games_per_opponent * (1 + 0.5 * attempt))

        if attempt > 0:
            print(f"\n[Retry {attempt}/{max_retries - 1}] Retrying with {retry_games} games per opponent...")
            time.sleep(retry_delay)

        try:
            results = _run_gauntlet_with_timeout(
                model_path, board_type, num_players, retry_games,
                model_type=model_type,
                timeout_seconds=timeout,
            )

            # If we got valid results (even if gauntlet failed), return them
            if results:
                if attempt > 0:
                    print(f"  Evaluation completed on attempt {attempt + 1}")
                return results

        except GauntletTimeoutError as e:
            last_error = e
            print(f"  Attempt {attempt + 1} timed out after {timeout}s")
            # Timeout is retriable - evaluation may have hung due to transient issue
            continue

        except Exception as e:
            last_error = e
            print(f"  Attempt {attempt + 1} failed: {e}")

            # Only continue retrying for transient errors
            # For fatal errors like model not found, don't retry
            error_msg = str(e).lower()
            if any(fatal in error_msg for fatal in ["model not found", "file not found", "invalid"]):
                print(f"  Fatal error - not retrying")
                break

    print(f"Error: All {max_retries} gauntlet attempts failed")
    if last_error:
        print(f"  Last error: {last_error}")
    return None


def run_gauntlet_promotion(
    model_path: Path,
    board_type: str,
    num_players: int,
    games_per_opponent: int = 20,
    sync_to_cluster: bool = False,
    dry_run: bool = False,
    model_type: str | None = None,
    skip_resource_check: bool = False,
    max_retries: int = 3,
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
        skip_resource_check: If True, skip pre-execution resource checks
        max_retries: Maximum retry attempts for gauntlet evaluation (default: 3)

    Returns:
        True if promotion successful, False otherwise
    """
    print("=" * 60)
    print("GAUNTLET-BASED MODEL PROMOTION")
    print("=" * 60)

    # Pre-execution resource check (Dec 2025)
    # Prevents resource exhaustion from concurrent gauntlet evaluations
    ok, msg = check_gauntlet_resources(skip_check=skip_resource_check)
    if not ok:
        print(f"\n⚠ RESOURCE CHECK FAILED: {msg}")
        print("  Use --skip-resource-check to bypass this check")
        print("  Or wait for other gauntlets to complete")
        return False
    elif not skip_resource_check:
        print(f"  Resource check: {msg}")

    if not model_path.exists():
        print(f"Error: Model file not found: {model_path}")
        return False

    # Run gauntlet evaluation with retry logic (Dec 2025)
    results = run_gauntlet_with_retry(
        model_path, board_type, num_players, games_per_opponent,
        model_type=model_type,
        max_retries=max_retries,
    )

    if results is None:
        print("Error: Gauntlet evaluation failed after all retries")
        return False

    # Print results
    print("\nGauntlet Results:")
    print(f"  Total games: {results['total_games']}")
    print(f"  Wins: {results['total_wins']}, Losses: {results['total_losses']}, Draws: {results['total_draws']}")
    print(f"  Overall win rate: {results['win_rate']:.1%}")
    print()

    # Get threshold information from results (two-tier system)
    aspirational = results.get('aspirational_thresholds', {})
    minimum = results.get('minimum_thresholds', {})

    # Display thresholds in two-tier format
    print("  Promotion Thresholds (Two-Tier System):")
    print(f"    Aspirational: {aspirational.get('vs_random', 0.85):.0%} random, "
          f"{aspirational.get('vs_heuristic', 0.60):.0%} heuristic")
    print(f"    Minimum floor: {minimum.get('vs_random', 0.70):.0%} random, "
          f"{minimum.get('vs_heuristic', 0.40):.0%} heuristic")
    print()

    print(f"  vs RANDOM: {results['win_rate_vs_random']:.1%} "
          f"({'✓ PASS' if results['passes_random'] else '✗ FAIL'})")
    print(f"  vs HEURISTIC: {results['win_rate_vs_heuristic']:.1%} "
          f"({'✓ PASS' if results['passes_heuristic'] else '✗ FAIL'})")
    print(f"  Estimated ELO: {results['estimated_elo']:.0f}")
    print()

    # Display promotion decision and reason
    promotion_reason = results.get('promotion_reason', 'No reason provided')
    if not results['passes_gauntlet']:
        print("✗ GAUNTLET FAILED - Model not promoted")
        print(f"  Reason: {promotion_reason}")
        return False

    print("✓ GAUNTLET PASSED - Promoting model...")
    print(f"  Reason: {promotion_reason}")

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
    parser.add_argument("--games", type=int, default=None,
                        help="Games per opponent in gauntlet (default: dynamic based on player count)")
    parser.add_argument("--model-type", choices=["cnn", "gnn", "hybrid"],
                        help="Model type (auto-detected if not specified)")
    parser.add_argument("--sync-to-cluster", action="store_true",
                        help="Sync promoted model to cluster nodes")
    parser.add_argument("--skip-resource-check", action="store_true",
                        help="Skip pre-execution resource checks (memory, CPU, concurrent gauntlets)")
    parser.add_argument("--max-retries", type=int, default=3,
                        help="Max retry attempts for gauntlet evaluation (default: 3)")

    args = parser.parse_args()

    # Validate gauntlet args
    if args.gauntlet:
        if not args.model:
            parser.error("--gauntlet requires --model")
        if not args.board_type:
            parser.error("--gauntlet requires --board-type")
        if not args.num_players:
            parser.error("--gauntlet requires --num-players")

        # December 2025: Compute dynamic games per opponent if not explicitly provided
        # 3p/4p games have higher variance (33%/25% random baseline vs 50%)
        # requiring more games for statistical significance
        if args.games is None:
            args.games = get_gauntlet_games_per_opponent(args.num_players)
            print(f"Using dynamic gauntlet games: {args.games} (for {args.num_players}p)")

    # Branch between gauntlet mode and ELO mode
    if args.gauntlet:
        # December 2025: SingletonLock prevents concurrent gauntlet invocations
        # This avoids 65+ zombie processes accumulating on cluster nodes
        lock = None
        signal_handler = None

        if HAS_PROCESS_UTILS and SingletonLock:
            config_key = f"{args.board_type}_{args.num_players}p"
            lock = SingletonLock(
                f"auto_promote_gauntlet_{config_key}",
                auto_cleanup_stale=True,
            )

            if not lock.acquire():
                holder_pid = lock.get_holder_pid()
                print(f"Error: Another gauntlet for {config_key} is already running (PID {holder_pid})")
                print("  Use 'ps aux | grep auto_promote' to see running processes")
                print("  Or wait for the existing gauntlet to complete")
                sys.exit(1)

            print(f"Acquired singleton lock for gauntlet: {config_key}")

            # SignalHandler ensures child processes are cleaned up on SIGTERM/SIGINT
            signal_handler = SignalHandler(on_shutdown=_cleanup_child_processes)

        try:
            # Gauntlet-based promotion with retry logic (Dec 2025)
            success = run_gauntlet_promotion(
                model_path=args.model,
                board_type=args.board_type,
                num_players=args.num_players,
                games_per_opponent=args.games,
                sync_to_cluster=args.sync_to_cluster,
                dry_run=args.dry_run,
                model_type=getattr(args, 'model_type', None),
                skip_resource_check=args.skip_resource_check,
                max_retries=args.max_retries,
            )
        finally:
            # Ensure lock is released and processes cleaned up
            if lock:
                lock.release()
            if signal_handler:
                _cleanup_child_processes()

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
