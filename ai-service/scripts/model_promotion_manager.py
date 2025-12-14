#!/usr/bin/env python
"""Model Promotion Manager - Manages promoted model symlinks and cluster-wide deployment.

This script provides:
1. Stable alias publishing: Creates/updates aliases like `ringrift_best_sq8_2p.pth`
2. Symlink management: Optionally maintains `models/promoted/*_best.pth` links
3. Cluster sync: Rsyncs published aliases to all cluster nodes
4. Optional sandbox config emission (opt-in)

Usage:
    # Update symlinks and config for all promoted models
    python scripts/model_promotion_manager.py --update-all

    # Update specific config
    python scripts/model_promotion_manager.py --update square8 2

    # Sync models to cluster
    python scripts/model_promotion_manager.py --sync-cluster

    # Full pipeline: promote, update symlinks, sync cluster
    python scripts/model_promotion_manager.py --full-pipeline
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add project root to path
SCRIPT_DIR = Path(__file__).parent
AI_SERVICE_ROOT = SCRIPT_DIR.parent
PROJECT_ROOT = AI_SERVICE_ROOT.parent
sys.path.insert(0, str(AI_SERVICE_ROOT))

# Paths
MODELS_DIR = AI_SERVICE_ROOT / "models"
PROMOTED_DIR = MODELS_DIR / "promoted"
PROMOTED_CONFIG_PATH = AI_SERVICE_ROOT / "data" / "promoted_models.json"
ELO_DB_PATH = AI_SERVICE_ROOT / "data" / "elo_leaderboard.db"
PROMOTION_LOG_PATH = AI_SERVICE_ROOT / "data" / "model_promotion_history.json"

# Sandbox config path (TypeScript side)
SANDBOX_CONFIG_PATH = PROJECT_ROOT / "src" / "shared" / "config" / "ai_models.json"

# All board/player configurations
ALL_CONFIGS = [
    ("square8", 2),
    ("square8", 3),
    ("square8", 4),
    ("square19", 2),
    ("square19", 3),
    ("square19", 4),
    ("hexagonal", 2),
    ("hexagonal", 3),
    ("hexagonal", 4),
]

BOARD_ALIAS_TOKENS = {
    "square8": "sq8",
    "square19": "sq19",
    "hexagonal": "hex",
}


@dataclass
class PromotedModel:
    """Information about a promoted model."""
    board_type: str
    num_players: int
    model_path: str  # Relative path from ai-service/models/
    model_id: str
    elo_rating: float
    games_played: int
    promoted_at: str
    symlink_name: str  # e.g., "square8_2p_best.pth"
    alias_id: str  # e.g., "ringrift_best_sq8_2p"
    alias_paths: List[str]  # absolute file paths under ai-service/models


def best_alias_id(board_type: str, num_players: int) -> str:
    token = BOARD_ALIAS_TOKENS.get(board_type)
    if not token:
        raise ValueError(f"Unsupported board_type for alias: {board_type!r}")
    return f"ringrift_best_{token}_{num_players}p"


def _atomic_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + f".tmp_{uuid.uuid4().hex}")
    shutil.copy2(src, tmp)
    os.replace(tmp, dst)


def _write_json_atomic(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp_{uuid.uuid4().hex}")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
    os.replace(tmp, path)


def get_best_model_from_elo(board_type: str, num_players: int) -> Optional[Dict[str, Any]]:
    """Get the best model from Elo leaderboard for a given config."""
    if not ELO_DB_PATH.exists():
        return None

    try:
        conn = sqlite3.connect(str(ELO_DB_PATH))
        cursor = conn.cursor()

        cursor.execute("""
            SELECT model_id, rating, games_played, wins, losses, draws
            FROM elo_ratings
            WHERE board_type = ? AND num_players = ?
            ORDER BY rating DESC
            LIMIT 1
        """, (board_type, num_players))

        row = cursor.fetchone()
        conn.close()

        if row:
            return {
                "model_id": row[0],
                "elo_rating": row[1],
                "games_played": row[2],
                "wins": row[3],
                "losses": row[4],
                "draws": row[5],
            }
        return None
    except Exception as e:
        print(f"[model_promotion] Error querying Elo DB: {e}")
        return None


def find_model_file(model_id: str) -> Optional[Path]:
    """Find the actual model file for a given model ID."""
    # Prefer exact filenames, then fall back to best-effort prefix matches.
    candidates: List[Path] = [
        MODELS_DIR / f"{model_id}_mps.pth",
        MODELS_DIR / f"{model_id}.pth",
        MODELS_DIR / model_id,
    ]

    matches = sorted(MODELS_DIR.glob(f"{model_id}_*.pth"), key=lambda p: p.stat().st_mtime)
    candidates.extend(reversed(matches[-25:]))  # newest-first, cap for speed

    for candidate in candidates:
        try:
            if candidate.exists() and candidate.is_file() and candidate.stat().st_size > 0:
                return candidate
        except OSError:
            continue

    return None


def publish_best_alias(
    *,
    board_type: str,
    num_players: int,
    best_model_path: Path,
    best_model_id: str,
    elo_rating: float,
    games_played: int,
    verbose: bool,
) -> List[Path]:
    alias = best_alias_id(board_type, num_players)
    published_at = datetime.utcnow().isoformat() + "Z"

    # Prefer an explicit _mps variant for the MPS alias if present.
    mps_src = MODELS_DIR / f"{best_model_id}_mps.pth"
    if not (mps_src.exists() and mps_src.is_file() and mps_src.stat().st_size > 0):
        mps_src = best_model_path

    cpu_dst = MODELS_DIR / f"{alias}.pth"
    mps_dst = MODELS_DIR / f"{alias}_mps.pth"
    meta_dst = MODELS_DIR / f"{alias}.meta.json"

    _atomic_copy(best_model_path, cpu_dst)
    _atomic_copy(mps_src, mps_dst)
    _write_json_atomic(
        meta_dst,
        {
            "alias_id": alias,
            "board_type": board_type,
            "num_players": int(num_players),
            "source_model_id": best_model_id,
            "source_checkpoint": str(best_model_path),
            "source_checkpoint_mps": str(mps_src),
            "elo_rating": float(elo_rating),
            "games_played": int(games_played),
            "published_at": published_at,
        },
    )

    if verbose:
        print(f"[model_promotion] Published alias {alias} -> {best_model_path.name}")

    return [cpu_dst, mps_dst, meta_dst]


def create_symlink(model_path: Path, symlink_name: str) -> bool:
    """Create or update a symlink in the promoted directory."""
    PROMOTED_DIR.mkdir(parents=True, exist_ok=True)
    symlink_path = PROMOTED_DIR / symlink_name

    # Remove existing symlink if it exists
    if symlink_path.exists() or symlink_path.is_symlink():
        symlink_path.unlink()

    # Create relative symlink
    try:
        rel_path = os.path.relpath(model_path, PROMOTED_DIR)
        symlink_path.symlink_to(rel_path)
        print(f"[model_promotion] Created symlink: {symlink_name} -> {rel_path}")
        return True
    except Exception as e:
        print(f"[model_promotion] Error creating symlink: {e}")
        return False


def update_promoted_config(promoted_models: List[PromotedModel]) -> bool:
    """Update the promoted_models.json config file."""
    config = {
        "updated_at": datetime.utcnow().isoformat() + "Z",
        "models": {
            f"{m.board_type}_{m.num_players}p": {
                "path": f"promoted/{m.symlink_name}",
                "alias_id": m.alias_id,
                "alias_paths": m.alias_paths,
                "model_id": m.model_id,
                "elo_rating": m.elo_rating,
                "games_played": m.games_played,
                "promoted_at": m.promoted_at,
            }
            for m in promoted_models
        }
    }

    try:
        PROMOTED_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(PROMOTED_CONFIG_PATH, "w") as f:
            json.dump(config, f, indent=2)
        print(f"[model_promotion] Updated config: {PROMOTED_CONFIG_PATH}")
        return True
    except Exception as e:
        print(f"[model_promotion] Error updating config: {e}")
        return False


def update_sandbox_config(promoted_models: List[PromotedModel]) -> bool:
    """Update the TypeScript sandbox config with promoted models."""
    config = {
        "_comment": "Auto-generated by model_promotion_manager.py - DO NOT EDIT",
        "updated_at": datetime.utcnow().isoformat() + "Z",
        "models": {
            f"{m.board_type}_{m.num_players}p": {
                "path": f"ai-service/models/{m.alias_id}.pth",
                "elo_rating": m.elo_rating,
            }
            for m in promoted_models
        }
    }

    try:
        SANDBOX_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(SANDBOX_CONFIG_PATH, "w") as f:
            json.dump(config, f, indent=2)
        print(f"[model_promotion] Updated sandbox config: {SANDBOX_CONFIG_PATH}")
        return True
    except Exception as e:
        print(f"[model_promotion] Error updating sandbox config: {e}")
        return False


def log_promotion(promoted_model: PromotedModel) -> None:
    """Append promotion to history log."""
    try:
        history = []
        if PROMOTION_LOG_PATH.exists():
            with open(PROMOTION_LOG_PATH) as f:
                history = json.load(f)

        history.append(asdict(promoted_model))

        # Keep last 1000 entries
        if len(history) > 1000:
            history = history[-1000:]

        with open(PROMOTION_LOG_PATH, "w") as f:
            json.dump(history, f, indent=2)
    except Exception as e:
        print(f"[model_promotion] Warning: Could not log promotion: {e}")


def sync_to_cluster_ssh(
    promoted_models: List[PromotedModel],
    *,
    verbose: bool = True,
    restart_p2p: bool = False,
) -> bool:
    """Sync published best-model aliases to cluster nodes via SSH+rsync."""
    # Read cluster hosts from config (gitignored, local).
    hosts_file = AI_SERVICE_ROOT / "config" / "distributed_hosts.yaml"
    if not hosts_file.exists():
        if verbose:
            print("[model_promotion] No distributed_hosts.yaml found, skipping cluster sync")
        return False

    try:
        import yaml
        with open(hosts_file) as f:
            config = yaml.safe_load(f)

        hosts = config.get("hosts", {})
        success_count = 0
        files: List[Path] = []
        for m in promoted_models:
            for raw in m.alias_paths:
                p = Path(raw)
                if p.exists() and p.is_file() and p.stat().st_size > 0:
                    files.append(p)

        # De-duplicate by resolved path.
        uniq: List[Path] = []
        seen: set[Path] = set()
        for p in files:
            rp = p.resolve()
            if rp in seen:
                continue
            seen.add(rp)
            uniq.append(rp)
        files = uniq

        if not files:
            if verbose:
                print("[model_promotion] No published alias files found to sync")
            return False

        for host_name, host_config in hosts.items():
            ssh_host = host_config.get("ssh_host")
            ssh_user = host_config.get("ssh_user", "root")
            ssh_port = host_config.get("ssh_port", 22)
            ssh_key = host_config.get("ssh_key")
            ringrift_path = host_config.get("ringrift_path", "~/ringrift")
            status = host_config.get("status", "ready")

            if not ssh_host:
                continue
            if status != "ready":
                continue

            # Normalize ringrift_path: allow configs that point at .../ai-service.
            ringrift_path_str = str(ringrift_path).rstrip("/")
            if ringrift_path_str.endswith("/ai-service"):
                ringrift_path_str = ringrift_path_str[: -len("/ai-service")]
            remote_models_dir = f"{ringrift_path_str}/ai-service/models"

            # Build SSH command
            ssh_cmd = ["ssh", "-o", "ConnectTimeout=10", "-o", "BatchMode=yes"]
            if ssh_port != 22:
                ssh_cmd.extend(["-p", str(ssh_port)])
            if ssh_key:
                ssh_cmd.extend(["-i", os.path.expanduser(str(ssh_key))])
            ssh_cmd.append(f"{ssh_user}@{ssh_host}")

            # Ensure remote dir exists.
            mkdir_cmd = f"mkdir -p {remote_models_dir}"

            try:
                mkdir_res = subprocess.run(
                    ssh_cmd + [mkdir_cmd],
                    capture_output=True,
                    timeout=30,
                    text=True,
                )
                if mkdir_res.returncode != 0:
                    if verbose:
                        print(f"[model_promotion] Failed to mkdir on {host_name}: {mkdir_res.stderr[:200]}")
                    continue

                ssh_opts = ["-o", "ConnectTimeout=10", "-o", "BatchMode=yes"]
                if ssh_port != 22:
                    ssh_opts.extend(["-p", str(ssh_port)])
                if ssh_key:
                    ssh_opts.extend(["-i", os.path.expanduser(str(ssh_key))])

                rsync_cmd = [
                    "rsync",
                    "-az",
                    "--timeout=120",
                    "-e",
                    "ssh " + " ".join(ssh_opts),
                    *[str(p) for p in files],
                    f"{ssh_user}@{ssh_host}:{remote_models_dir}/",
                ]
                rsync_res = subprocess.run(
                    rsync_cmd,
                    capture_output=True,
                    timeout=3600,
                    text=True,
                )
                if rsync_res.returncode != 0:
                    if verbose:
                        print(f"[model_promotion] rsync failed for {host_name}: {rsync_res.stderr[:200]}")
                    continue

                if restart_p2p:
                    # Best-effort restart of the P2P orchestrator (if installed as systemd or launchd).
                    restart_cmd = (
                        "if command -v systemctl >/dev/null 2>&1; then "
                        "sudo systemctl restart ringrift-p2p.service ringrift-resilience.service >/dev/null 2>&1 || true; "
                        "sudo systemctl restart ringrift-p2p-orchestrator.service >/dev/null 2>&1 || true; "
                        "fi; "
                        "if command -v launchctl >/dev/null 2>&1; then "
                        "launchctl kickstart -k gui/$(id -u)/com.ringrift.p2p-orchestrator >/dev/null 2>&1 || true; "
                        "launchctl kickstart -k system/com.ringrift.p2p-orchestrator >/dev/null 2>&1 || true; "
                        "fi"
                    )
                    subprocess.run(
                        ssh_cmd + [restart_cmd],
                        capture_output=True,
                        timeout=60,
                        text=True,
                    )

                if verbose:
                    print(f"[model_promotion] Synced aliases to: {host_name}")
                success_count += 1
            except subprocess.TimeoutExpired:
                if verbose:
                    print(f"[model_promotion] Timeout syncing {host_name}")
            except Exception as e:
                if verbose:
                    print(f"[model_promotion] Error syncing {host_name}: {e}")

        if verbose:
            print(f"[model_promotion] Cluster sync complete: {success_count}/{len(hosts)} hosts")
        return success_count > 0
    except Exception as e:
        if verbose:
            print(f"[model_promotion] Cluster sync error: {e}")
        return False


def update_all_promotions(
    min_games: int = 20,
    *,
    verbose: bool = True,
    update_sandbox: bool = False,
) -> List[PromotedModel]:
    """Publish best-model aliases (and optional symlinks/config) for all configs."""
    promoted_models = []

    for board_type, num_players in ALL_CONFIGS:
        if verbose:
            print(f"\n[model_promotion] Checking {board_type} {num_players}p...")

        best = get_best_model_from_elo(board_type, num_players)
        if not best:
            if verbose:
                print(f"  No Elo data available")
            continue

        if best["games_played"] < min_games:
            if verbose:
                print(f"  Best model has only {best['games_played']} games (min: {min_games})")
            continue

        model_path = find_model_file(best["model_id"])
        if not model_path:
            if verbose:
                print(f"  Model file not found: {best['model_id']}")
            continue

        alias_id = best_alias_id(board_type, int(num_players))
        alias_paths = publish_best_alias(
            board_type=board_type,
            num_players=int(num_players),
            best_model_path=model_path,
            best_model_id=best["model_id"],
            elo_rating=best["elo_rating"],
            games_played=best["games_played"],
            verbose=verbose,
        )

        symlink_name = f"{board_type}_{num_players}p_best.pth"

        if create_symlink(model_path, symlink_name):
            promoted = PromotedModel(
                board_type=board_type,
                num_players=num_players,
                model_path=str(model_path.relative_to(MODELS_DIR)),
                model_id=best["model_id"],
                elo_rating=best["elo_rating"],
                games_played=best["games_played"],
                promoted_at=datetime.utcnow().isoformat() + "Z",
                symlink_name=symlink_name,
                alias_id=alias_id,
                alias_paths=[str(p) for p in alias_paths],
            )
            promoted_models.append(promoted)
            log_promotion(promoted)

            if verbose:
                print(f"  Promoted: {best['model_id']} (Elo: {best['elo_rating']:.0f}, games: {best['games_played']})")

    if promoted_models:
        update_promoted_config(promoted_models)
        if update_sandbox:
            update_sandbox_config(promoted_models)

    return promoted_models


def run_full_pipeline(
    min_games: int = 20,
    sync_cluster: bool = True,
    update_sandbox: bool = False,
    restart_p2p: bool = False,
    verbose: bool = True,
) -> bool:
    """Run the full promotion pipeline: update symlinks, config, and sync cluster."""
    if verbose:
        print("[model_promotion] Starting full promotion pipeline...")

    # Step 1: Update all promotions
    promoted = update_all_promotions(min_games=min_games, verbose=verbose, update_sandbox=update_sandbox)

    if not promoted:
        if verbose:
            print("[model_promotion] No models promoted")
        return False

    if verbose:
        print(f"\n[model_promotion] Promoted {len(promoted)} models")

    # Step 2: Sync to cluster
    if sync_cluster:
        if verbose:
            print("\n[model_promotion] Syncing to cluster...")
        sync_to_cluster_ssh(promoted, verbose=verbose, restart_p2p=restart_p2p)

    if verbose:
        print("\n[model_promotion] Pipeline complete!")

    return True


def sync_staging_artifacts(
    *,
    restart: bool,
    validate_health: bool,
    fail_on_missing: bool,
    verbose: bool,
) -> bool:
    """Best-effort sync of published artifacts to staging via SSH."""
    host = os.environ.get("RINGRIFT_STAGING_SSH_HOST")
    remote_root = os.environ.get("RINGRIFT_STAGING_ROOT")
    if not host or not remote_root:
        if verbose:
            print(
                "[model_promotion] Staging sync requested but missing "
                "RINGRIFT_STAGING_SSH_HOST / RINGRIFT_STAGING_ROOT"
            )
        return False

    cmd = [sys.executable, "scripts/sync_staging_ai_artifacts.py"]
    if restart:
        cmd.append("--restart")
    if validate_health:
        cmd.append("--validate-health")
    if fail_on_missing:
        cmd.append("--fail-on-missing")

    proc = subprocess.run(
        cmd,
        cwd=str(AI_SERVICE_ROOT),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if verbose and proc.stdout:
        print(proc.stdout)
    return proc.returncode == 0


def main():
    parser = argparse.ArgumentParser(
        description="Model Promotion Manager - Manage promoted model symlinks and deployment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--update-all",
        action="store_true",
        help="Update symlinks and config for all configurations",
    )
    parser.add_argument(
        "--update",
        nargs=2,
        metavar=("BOARD", "PLAYERS"),
        help="Update specific configuration (e.g., --update square8 2)",
    )
    parser.add_argument(
        "--sync-cluster",
        action="store_true",
        help="Sync models to all cluster nodes",
    )
    parser.add_argument(
        "--restart-p2p",
        action="store_true",
        help="After syncing, best-effort restart of the P2P orchestrator on each host.",
    )
    parser.add_argument(
        "--full-pipeline",
        action="store_true",
        help="Run full pipeline: update all, sync cluster",
    )
    parser.add_argument(
        "--min-games",
        type=int,
        default=20,
        help="Minimum games required for promotion (default: 20)",
    )
    parser.add_argument(
        "--update-sandbox-config",
        action="store_true",
        help="Also write src/shared/config/ai_models.json (opt-in).",
    )
    parser.add_argument(
        "--sync-staging",
        action="store_true",
        help="After publishing aliases, sync the promoted artifacts to staging via scripts/sync_staging_ai_artifacts.py.",
    )
    parser.add_argument(
        "--sync-staging-no-restart",
        action="store_true",
        help="When used with --sync-staging, do not restart docker compose services after syncing.",
    )
    parser.add_argument(
        "--sync-staging-validate-health",
        action="store_true",
        help="When used with --sync-staging, validate /internal/ladder/health on staging after sync.",
    )
    parser.add_argument(
        "--sync-staging-fail-on-missing",
        action="store_true",
        help="When used with --sync-staging-validate-health, exit non-zero if staging reports missing artifacts.",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress verbose output",
    )

    args = parser.parse_args()
    verbose = not args.quiet

    promoted_models: List[PromotedModel] = []
    did_publish = False

    if args.full_pipeline:
        promoted_models = update_all_promotions(
            min_games=args.min_games,
            verbose=verbose,
            update_sandbox=bool(args.update_sandbox_config),
        )
        did_publish = bool(promoted_models)
        if did_publish and verbose:
            print(f"\n[model_promotion] Promoted {len(promoted_models)} models")

        if did_publish:
            if verbose:
                print("\n[model_promotion] Syncing to cluster...")
            sync_to_cluster_ssh(promoted_models, verbose=verbose, restart_p2p=bool(args.restart_p2p))

            if verbose:
                print("\n[model_promotion] Pipeline complete!")
    elif args.update_all:
        promoted_models = update_all_promotions(
            min_games=args.min_games,
            verbose=verbose,
            update_sandbox=bool(args.update_sandbox_config),
        )
        did_publish = bool(promoted_models)
    elif args.sync_cluster:
        promoted_models = update_all_promotions(
            min_games=args.min_games,
            verbose=False,
            update_sandbox=False,
        )
        did_publish = bool(promoted_models)
        sync_to_cluster_ssh(promoted_models, verbose=verbose, restart_p2p=bool(args.restart_p2p))
    elif args.update:
        board_type, num_players = args.update
        promoted_models = update_all_promotions(
            min_games=args.min_games,
            verbose=verbose,
            update_sandbox=bool(args.update_sandbox_config),
        )
        did_publish = bool(promoted_models)
        for p in promoted_models:
            if p.board_type == board_type and p.num_players == int(num_players):
                print(f"Updated: {p.symlink_name}")
    else:
        parser.print_help()

    if args.sync_staging and did_publish:
        restart = not bool(args.sync_staging_no_restart)
        validate = bool(args.sync_staging_validate_health)
        fail_on_missing = bool(args.sync_staging_fail_on_missing)
        if fail_on_missing:
            validate = True
        ok = sync_staging_artifacts(
            restart=restart,
            validate_health=validate,
            fail_on_missing=fail_on_missing,
            verbose=verbose,
        )
        if not ok:
            raise SystemExit(3)


if __name__ == "__main__":
    main()
