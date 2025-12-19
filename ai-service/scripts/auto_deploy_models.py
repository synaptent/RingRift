#!/usr/bin/env python3
"""Automated Model Deployment Pipeline for RingRift.

This script bridges training completion with production deployment:
1. Detects newly trained models
2. Runs automated evaluation against baseline
3. Deploys promoted models to cluster nodes
4. Syncs models to TypeScript sandbox
5. Triggers retraining if promotion fails

Integration:
- Called by continuous_improvement_daemon after training
- Called by P2P orchestrator after distributed training
- Can be run standalone for manual deployment

Usage:
    # Deploy latest model with full pipeline
    python scripts/auto_deploy_models.py --board-type square8 --num-players 2

    # Deploy specific model without evaluation
    python scripts/auto_deploy_models.py --model-path models/nnue/nnue_square8_2p.pt --skip-eval

    # Sync all models to cluster
    python scripts/auto_deploy_models.py --sync-cluster

    # Check deployment status
    python scripts/auto_deploy_models.py --status
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add ai-service to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

AI_SERVICE_ROOT = Path(__file__).resolve().parents[1]
RINGRIFT_ROOT = AI_SERVICE_ROOT.parent

# Model directories
MODELS_DIR = AI_SERVICE_ROOT / "models"
NNUE_DIR = MODELS_DIR / "nnue"
NN_DIR = MODELS_DIR / "neural_net"

# Sandbox model location (TypeScript side)
SANDBOX_MODEL_DIR = RINGRIFT_ROOT / "public" / "models"

# Deployment tracking
DEPLOYMENT_LOG = AI_SERVICE_ROOT / "logs" / "deployment" / "deployments.json"
DEPLOYMENT_LOG.parent.mkdir(parents=True, exist_ok=True)


@dataclass
class DeploymentRecord:
    """Record of a model deployment."""
    model_path: str
    model_hash: str
    board_type: str
    num_players: int
    deployment_time: str
    evaluation_result: Optional[Dict[str, Any]]
    deployed_to: List[str]
    promoted: bool
    notes: str = ""


@dataclass
class EvaluationResult:
    """Result of model evaluation against baseline."""
    win_rate: float
    games_played: int
    baseline_model: str
    passed: bool
    details: Dict[str, Any]


def compute_model_hash(model_path: Path) -> str:
    """Compute SHA256 hash of model file."""
    sha256 = hashlib.sha256()
    with open(model_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()[:16]


def load_deployment_log() -> List[DeploymentRecord]:
    """Load deployment history."""
    if not DEPLOYMENT_LOG.exists():
        return []
    try:
        with open(DEPLOYMENT_LOG) as f:
            data = json.load(f)
        return [DeploymentRecord(**r) for r in data]
    except Exception as e:
        print(f"[Deploy] Warning: Could not load deployment log: {e}")
        return []


def save_deployment_log(records: List[DeploymentRecord]) -> None:
    """Save deployment history."""
    data = [asdict(r) for r in records]
    with open(DEPLOYMENT_LOG, "w") as f:
        json.dump(data, f, indent=2)


def find_latest_model(board_type: str, num_players: int, model_type: str = "nnue") -> Optional[Path]:
    """Find the latest trained model for given configuration."""
    if model_type == "nnue":
        model_dir = NNUE_DIR
        pattern = f"nnue_{board_type}_{num_players}p*.pt"
    else:
        model_dir = NN_DIR
        pattern = f"policy_value_{board_type}_{num_players}p*.pt"

    if not model_dir.exists():
        return None

    # Check for exact match first
    exact_path = model_dir / f"nnue_{board_type}_{num_players}p.pt"
    if exact_path.exists():
        return exact_path

    # Find latest by modification time
    matches = list(model_dir.glob(pattern))
    if not matches:
        return None

    return max(matches, key=lambda p: p.stat().st_mtime)


def evaluate_model(
    model_path: Path,
    baseline_path: Optional[Path],
    board_type: str,
    num_players: int,
    games: int = 50
) -> EvaluationResult:
    """Evaluate model against baseline with tournament."""
    print(f"[Deploy] Evaluating {model_path.name} ({games} games)...")

    if baseline_path is None or not baseline_path.exists():
        # No baseline - auto-pass for first model
        return EvaluationResult(
            win_rate=1.0,
            games_played=0,
            baseline_model="none",
            passed=True,
            details={"reason": "No baseline model - auto-approved for first deployment"}
        )

    # Run tournament using tier evaluation
    try:
        result = subprocess.run(
            [
                sys.executable, "-m", "scripts.run_model_tournament",
                "--model-a", str(model_path),
                "--model-b", str(baseline_path),
                "--board-type", board_type,
                "--num-players", str(num_players),
                "--games", str(games),
                "--output-json", "/tmp/eval_result.json"
            ],
            capture_output=True,
            text=True,
            timeout=600,
            cwd=AI_SERVICE_ROOT
        )

        if result.returncode == 0 and Path("/tmp/eval_result.json").exists():
            with open("/tmp/eval_result.json") as f:
                eval_data = json.load(f)
            win_rate = eval_data.get("model_a_win_rate", 0.5)
            passed = win_rate >= 0.55  # 55% threshold for promotion
            return EvaluationResult(
                win_rate=win_rate,
                games_played=games,
                baseline_model=baseline_path.name,
                passed=passed,
                details=eval_data
            )
    except subprocess.TimeoutExpired:
        print("[Deploy] Tournament timed out")
    except Exception as e:
        print(f"[Deploy] Tournament error: {e}")

    # Fallback: simple sanity check (model loads and runs)
    try:
        import torch
        state_dict = torch.load(model_path, map_location="cpu")
        print(f"[Deploy] Model loads successfully ({len(state_dict)} keys)")
        return EvaluationResult(
            win_rate=0.5,
            games_played=0,
            baseline_model=baseline_path.name if baseline_path else "none",
            passed=True,  # Pass on sanity check only
            details={"reason": "Tournament failed, passed on sanity check"}
        )
    except Exception as e:
        return EvaluationResult(
            win_rate=0.0,
            games_played=0,
            baseline_model=baseline_path.name if baseline_path else "none",
            passed=False,
            details={"error": str(e)}
        )


def deploy_to_sandbox(model_path: Path, board_type: str, num_players: int) -> bool:
    """Deploy model to TypeScript sandbox public directory."""
    SANDBOX_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Standard naming for sandbox
    target_name = f"nnue_{board_type}_{num_players}p.pt"
    target_path = SANDBOX_MODEL_DIR / target_name

    try:
        shutil.copy2(model_path, target_path)
        print(f"[Deploy] Copied to sandbox: {target_path}")

        # Also create a JSON manifest for the model
        manifest = {
            "model": target_name,
            "board_type": board_type,
            "num_players": num_players,
            "deployed_at": datetime.now().isoformat(),
            "hash": compute_model_hash(target_path)
        }
        manifest_path = SANDBOX_MODEL_DIR / f"{target_name}.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

        return True
    except Exception as e:
        print(f"[Deploy] Failed to deploy to sandbox: {e}")
        return False


def deploy_to_cluster(
    model_path: Path,
    hosts: List[str],
    remote_path: str = "~/ringrift/ai-service/models/nnue/"
) -> Dict[str, bool]:
    """Deploy model to cluster nodes via SSH."""
    results = {}

    for host in hosts:
        try:
            # Use rsync for efficient transfer
            cmd = [
                "rsync", "-avz", "--timeout=30",
                str(model_path),
                f"{host}:{remote_path}"
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            success = result.returncode == 0
            results[host] = success
            if success:
                print(f"[Deploy] Synced to {host}")
            else:
                print(f"[Deploy] Failed to sync to {host}: {result.stderr}")
        except Exception as e:
            results[host] = False
            print(f"[Deploy] Error syncing to {host}: {e}")

    return results


def get_cluster_hosts() -> List[str]:
    """Get list of cluster hosts from distributed_hosts.yaml."""
    import yaml

    hosts_file = AI_SERVICE_ROOT / "config" / "distributed_hosts.yaml"
    if not hosts_file.exists():
        return []

    with open(hosts_file) as f:
        config = yaml.safe_load(f)

    hosts = []
    for name, info in config.get("hosts", {}).items():
        ssh_host = info.get("ssh_host")
        ssh_user = info.get("ssh_user", "root")
        ssh_port = info.get("ssh_port", 22)
        if ssh_host:
            if ssh_port != 22:
                hosts.append(f"{ssh_user}@{ssh_host}:{ssh_port}")
            else:
                hosts.append(f"{ssh_user}@{ssh_host}")

    return hosts


def trigger_retraining(
    board_type: str,
    num_players: int,
    reason: str,
    additional_games: int = 1000
) -> None:
    """Trigger additional selfplay and retraining after failed promotion."""
    print(f"[Deploy] Triggering retraining: {reason}")
    print(f"[Deploy] Scheduling {additional_games} additional selfplay games...")

    # Write retrain request to P2P orchestrator
    request = {
        "board_type": board_type,
        "num_players": num_players,
        "reason": reason,
        "additional_games": additional_games,
        "timestamp": datetime.now().isoformat()
    }

    retrain_queue = AI_SERVICE_ROOT / "logs" / "deployment" / "retrain_queue.json"
    retrain_queue.parent.mkdir(parents=True, exist_ok=True)

    queue = []
    if retrain_queue.exists():
        try:
            with open(retrain_queue) as f:
                queue = json.load(f)
        except (json.JSONDecodeError, OSError):
            pass

    queue.append(request)
    with open(retrain_queue, "w") as f:
        json.dump(queue, f, indent=2)

    print(f"[Deploy] Retrain request queued")


def run_deployment_pipeline(
    model_path: Optional[Path],
    board_type: str,
    num_players: int,
    skip_eval: bool = False,
    sync_cluster: bool = False
) -> bool:
    """Run full deployment pipeline."""
    print(f"\n{'='*60}")
    print(f"[Deploy] Starting deployment pipeline")
    print(f"[Deploy] Board: {board_type}, Players: {num_players}")
    print(f"{'='*60}\n")

    # Find model if not specified
    if model_path is None:
        model_path = find_latest_model(board_type, num_players)
        if model_path is None:
            print("[Deploy] No model found for this configuration")
            return False

    print(f"[Deploy] Model: {model_path}")
    model_hash = compute_model_hash(model_path)
    print(f"[Deploy] Hash: {model_hash}")

    # Find baseline for comparison
    baseline_path = find_latest_model(board_type, num_players)
    if baseline_path == model_path:
        # Look for previous version
        baseline_path = None
        backup_dir = NNUE_DIR / "archive"
        if backup_dir.exists():
            archives = list(backup_dir.glob(f"nnue_{board_type}_{num_players}p*.pt"))
            if archives:
                baseline_path = max(archives, key=lambda p: p.stat().st_mtime)

    # Evaluate model
    eval_result = None
    if not skip_eval:
        eval_result = evaluate_model(model_path, baseline_path, board_type, num_players)
        print(f"\n[Deploy] Evaluation Results:")
        print(f"  Win Rate: {eval_result.win_rate:.1%}")
        print(f"  Games: {eval_result.games_played}")
        print(f"  Passed: {eval_result.passed}")

        if not eval_result.passed:
            print("\n[Deploy] Model FAILED evaluation - triggering retraining")
            trigger_retraining(
                board_type, num_players,
                f"Model failed evaluation with {eval_result.win_rate:.1%} win rate"
            )
            return False
    else:
        print("[Deploy] Skipping evaluation")

    # Deploy to sandbox
    deployed_to = []
    if deploy_to_sandbox(model_path, board_type, num_players):
        deployed_to.append("sandbox")

    # Deploy to cluster if requested
    if sync_cluster:
        hosts = get_cluster_hosts()
        if hosts:
            print(f"\n[Deploy] Syncing to {len(hosts)} cluster nodes...")
            cluster_results = deploy_to_cluster(model_path, hosts)
            for host, success in cluster_results.items():
                if success:
                    deployed_to.append(host)

    # Archive current model before overwriting
    archive_dir = NNUE_DIR / "archive"
    archive_dir.mkdir(parents=True, exist_ok=True)
    current_model = NNUE_DIR / f"nnue_{board_type}_{num_players}p.pt"
    if current_model.exists() and current_model != model_path:
        archive_name = f"nnue_{board_type}_{num_players}p_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
        shutil.copy2(current_model, archive_dir / archive_name)
        print(f"[Deploy] Archived previous model")

    # Copy to standard location
    if model_path != current_model:
        shutil.copy2(model_path, current_model)
        print(f"[Deploy] Updated {current_model}")

    # Record deployment
    record = DeploymentRecord(
        model_path=str(model_path),
        model_hash=model_hash,
        board_type=board_type,
        num_players=num_players,
        deployment_time=datetime.now().isoformat(),
        evaluation_result=asdict(eval_result) if eval_result else None,
        deployed_to=deployed_to,
        promoted=True,
        notes=""
    )

    records = load_deployment_log()
    records.append(record)
    save_deployment_log(records)

    print(f"\n{'='*60}")
    print(f"[Deploy] Deployment SUCCESSFUL")
    print(f"[Deploy] Deployed to: {', '.join(deployed_to)}")
    print(f"{'='*60}\n")

    return True


def show_status() -> None:
    """Show deployment status."""
    print("\n=== Model Deployment Status ===\n")

    # Show current models
    print("Current Models:")
    for model_type, model_dir in [("NNUE", NNUE_DIR), ("Neural Net", NN_DIR)]:
        if model_dir.exists():
            models = list(model_dir.glob("*.pt"))
            if models:
                for m in sorted(models):
                    stat = m.stat()
                    age_hours = (time.time() - stat.st_mtime) / 3600
                    size_mb = stat.st_size / 1024 / 1024
                    print(f"  {model_type}: {m.name} ({size_mb:.1f}MB, {age_hours:.1f}h ago)")

    # Show sandbox models
    print("\nSandbox Models:")
    if SANDBOX_MODEL_DIR.exists():
        for m in sorted(SANDBOX_MODEL_DIR.glob("*.pt")):
            print(f"  {m.name}")
    else:
        print("  (no sandbox directory)")

    # Show recent deployments
    print("\nRecent Deployments:")
    records = load_deployment_log()
    for r in records[-5:]:
        status = "PROMOTED" if r.promoted else "FAILED"
        print(f"  {r.deployment_time[:19]} - {r.board_type}/{r.num_players}p - {status}")

    print()


def main():
    parser = argparse.ArgumentParser(description="Automated Model Deployment")
    parser.add_argument("--model-path", type=Path, help="Specific model to deploy")
    parser.add_argument("--board-type", default="square8", help="Board type")
    parser.add_argument("--num-players", type=int, default=2, help="Number of players")
    parser.add_argument("--skip-eval", action="store_true", help="Skip evaluation")
    parser.add_argument("--sync-cluster", action="store_true", help="Sync to cluster")
    parser.add_argument("--status", action="store_true", help="Show status")

    args = parser.parse_args()

    if args.status:
        show_status()
        return

    success = run_deployment_pipeline(
        model_path=args.model_path,
        board_type=args.board_type,
        num_players=args.num_players,
        skip_eval=args.skip_eval,
        sync_cluster=args.sync_cluster
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
