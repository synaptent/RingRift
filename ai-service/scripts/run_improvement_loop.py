#!/usr/bin/env python
"""AlphaZero-style continuous improvement loop with checkpointing and rollback.

This script implements the core self-improvement loop for RingRift AI:
1. Run selfplay games with current best models
2. Export training data from new games
3. Fine-tune neural network on new data
4. Evaluate new model against previous best
5. Promote if improved, rollback if consecutive failures

Features:
- Checkpointing: Resume from any iteration after interruption
- Rollback: Automatically revert to previous model after N failures
- Dry-run: Preview commands without execution
- Model validation: Sanity checks before promotion

Usage:
    # Basic run
    python scripts/run_improvement_loop.py --board square8 --players 2 --iterations 50

    # Resume from checkpoint
    python scripts/run_improvement_loop.py --board square8 --players 2 --resume

    # Dry run to see what would execute
    python scripts/run_improvement_loop.py --board square8 --players 2 --dry-run

    # Custom thresholds
    python scripts/run_improvement_loop.py --board square8 --players 2 \
        --promotion-threshold 0.55 --max-consecutive-failures 5
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Allow imports from app/
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Light-weight significance helper (used for promotion gating).
from app.training.significance import wilson_score_interval

AI_SERVICE_ROOT = Path(__file__).resolve().parents[1]


def _resolve_ai_service_path(raw: str) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path
    return (AI_SERVICE_ROOT / path).resolve()


@dataclass
class LoopState:
    """Checkpoint state for resumable improvement loop."""

    iteration: int = 0
    best_model_path: Optional[str] = None
    best_winrate: float = 0.0
    consecutive_failures: int = 0
    total_improvements: int = 0
    total_games_generated: int = 0
    history: List[dict] = field(default_factory=list)


def load_state(state_path: Path) -> LoopState:
    """Load checkpoint state or return fresh state."""
    if state_path.exists():
        try:
            data = json.loads(state_path.read_text())
            # Handle history field migration
            if "history" not in data:
                data["history"] = []
            return LoopState(**data)
        except (json.JSONDecodeError, TypeError) as e:
            print(f"WARNING: Could not load state from {state_path}: {e}")
    return LoopState()


def save_state(state: LoopState, state_path: Path) -> None:
    """Persist checkpoint state."""
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(asdict(state), indent=2))


def parse_winrate(output: str) -> float:
    """Extract win rate from tournament output.

    Handles multiple output formats:
    - "P1 wins: 12/20 (60.0%)"
    - "Win rate: 0.60"
    - "Winner: P1" counting
    """
    # Match percentage format: "60.0%" or "60%"
    match = re.search(r"(\d+\.?\d*)%", output)
    if match:
        return float(match.group(1)) / 100.0

    # Match decimal format: "Win rate: 0.60"
    match = re.search(r"[Ww]in\s*rate[:\s]+(\d+\.?\d*)", output)
    if match:
        return float(match.group(1))

    # Match fraction format: "12/20"
    match = re.search(r"(\d+)\s*/\s*(\d+)", output)
    if match:
        wins, total = int(match.group(1)), int(match.group(2))
        if total > 0:
            return wins / total

    # Fallback: count explicit win lines
    p1_wins = len(re.findall(r"(P1 wins|Winner:\s*P1|Player 1 wins)", output, re.IGNORECASE))
    p2_wins = len(re.findall(r"(P2 wins|Winner:\s*P2|Player 2 wins)", output, re.IGNORECASE))
    total = p1_wins + p2_wins
    if total > 0:
        return p1_wins / total

    return 0.5  # Default to 50% if parsing fails


def _promotion_gate(
    *,
    wins: int,
    losses: int,
    draws: int,
    threshold: float,
    confidence: float,
) -> dict:
    """Compute promotion decision using a Wilson lower-bound gate."""
    total_games = wins + losses + draws
    if total_games <= 0:
        return {
            "games": 0,
            "wins": wins,
            "losses": losses,
            "draws": draws,
            "win_rate": 0.0,
            "win_rate_ci_low": None,
            "win_rate_ci_high": None,
            "threshold": threshold,
            "confidence": confidence,
            "promote": False,
        }

    win_rate = wins / float(total_games)
    ci_low, ci_high = wilson_score_interval(
        wins,
        total_games,
        confidence=confidence,
    )
    promote = win_rate >= threshold and ci_low >= threshold
    return {
        "games": total_games,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "win_rate": win_rate,
        "win_rate_ci_low": ci_low,
        "win_rate_ci_high": ci_high,
        "threshold": threshold,
        "confidence": confidence,
        "promote": promote,
    }


def validate_model(model_path: Path) -> bool:
    """Quick sanity check that model file is valid.

    Checks:
    - File exists
    - File size is reasonable (> 1KB, < 1GB)
    - File is readable
    """
    if not model_path.exists():
        return False

    try:
        size = model_path.stat().st_size
        # Model should be at least 1KB and less than 1GB
        if not (1024 < size < 1_000_000_000):
            return False

        # Try to read first few bytes to verify it's accessible
        with open(model_path, "rb") as f:
            header = f.read(16)
            return len(header) == 16
    except (OSError, IOError):
        return False


def run_command(
    cmd: List[str],
    dry_run: bool = False,
    capture: bool = False,
    timeout: Optional[int] = None,
    cwd: Optional[Path] = None,
    env_overrides: Optional[Dict[str, str]] = None,
) -> Tuple[int, str, str]:
    """Run a command with optional dry-run mode.

    Returns: (return_code, stdout, stderr)
    """
    if dry_run:
        print(f"[DRY-RUN] Would execute: {' '.join(cmd)}")
        return 0, "", ""

    try:
        env = None
        if env_overrides:
            env = os.environ.copy()
            env.update(env_overrides)
        result = subprocess.run(
            cmd,
            capture_output=capture,
            text=True,
            timeout=timeout,
            cwd=str(cwd) if cwd else None,
            env=env,
        )
        stdout = result.stdout if capture else ""
        stderr = result.stderr if capture else ""
        return result.returncode, stdout, stderr
    except subprocess.TimeoutExpired:
        return 1, "", "Command timed out"
    except Exception as e:
        return 1, "", str(e)


def _validate_canonical_training_source(
    db_path: Path,
    registry_path: Path,
    *,
    allow_pending_gate: bool,
) -> Tuple[bool, List[str]]:
    """Validate that db_path is a canonical training source per registry + gate summary."""
    from scripts.validate_canonical_training_sources import (  # type: ignore[import]
        validate_canonical_sources,
    )

    allowed_statuses = ["canonical", "pending_gate"] if allow_pending_gate else ["canonical"]
    result = validate_canonical_sources(
        registry_path=registry_path,
        db_paths=[db_path],
        allowed_statuses=allowed_statuses,
    )
    ok = bool(result.get("ok"))
    problems = [str(p) for p in (result.get("problems") or [])]
    return ok, problems


def run_selfplay(
    config: dict,
    iteration: int,
    dry_run: bool = False,
) -> Tuple[bool, int]:
    """Run selfplay to generate training data.

    Returns: (success, games_generated)
    """
    board = config["board"]
    players = config["players"]
    games = config["games_per_iter"]
    max_moves = config["max_moves"]
    seed = iteration * 1000

    log_dir = AI_SERVICE_ROOT / "logs" / "improvement"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"selfplay_iter{iteration}_{board}_{players}p.jsonl"

    replay_db_path = Path(config["replay_db"])
    canonical_mode = bool(config.get("canonical_mode", False))

    if canonical_mode:
        gate_summary_path = Path(config["gate_summary"])
        cmd = [
            "python",
            "scripts/generate_canonical_selfplay.py",
            "--board-type",
            board,
            "--num-games",
            str(games),
            "--num-players",
            str(players),
            "--db",
            str(replay_db_path),
            "--summary",
            str(gate_summary_path),
        ]
        print(f"Running canonical selfplay+gate: {games} games on {board} {players}p...")
        code, _, stderr = run_command(cmd, dry_run=dry_run, timeout=7200, cwd=AI_SERVICE_ROOT)
        if code != 0:
            print(f"Canonical selfplay gate failed: {stderr[:200]}")
            return False, 0

        if not dry_run:
            ok, problems = _validate_canonical_training_source(
                db_path=replay_db_path,
                registry_path=Path(config["registry_path"]),
                allow_pending_gate=bool(config.get("allow_pending_gate", False)),
            )
            if not ok:
                for issue in problems:
                    print(f"[canonical-source-error] {issue}", file=sys.stderr)
                print(
                    "[canonical-source-error] Refusing to proceed with training on a non-canonical source DB.\n"
                    "Fix TRAINING_DATA_REGISTRY.md status and/or rerun scripts/generate_canonical_selfplay.py.",
                    file=sys.stderr,
                )
                return False, 0

        return True, games

    cmd = [
        "python",
        "scripts/run_self_play_soak.py",
        "--num-games",
        str(games),
        "--board-type",
        board,
        "--engine-mode",
        "mixed",
        "--num-players",
        str(players),
        "--max-moves",
        str(max_moves),
        "--seed",
        str(seed),
        "--record-db",
        str(replay_db_path),
        "--log-jsonl",
        str(log_file),
    ]

    print(f"Running selfplay: {games} games on {board} {players}p...")
    code, _, stderr = run_command(cmd, dry_run=dry_run, timeout=3600, cwd=AI_SERVICE_ROOT)

    if code != 0:
        print(f"Selfplay failed: {stderr[:200]}")
        return False, 0

    # Count completed games from log file
    games_completed = 0
    if not dry_run and log_file.exists():
        try:
            with open(log_file) as f:
                for line in f:
                    if '"status": "completed"' in line or '"status":"completed"' in line:
                        games_completed += 1
        except Exception:
            games_completed = games  # Assume success if can't count

    return True, games_completed if not dry_run else games


def export_training_data(
    config: dict,
    iteration: int,
    dry_run: bool = False,
) -> Tuple[bool, Path]:
    """Export training data from selfplay database.

    Returns: (success, output_path)
    """
    board = config["board"]
    players = config["players"]

    data_dir = AI_SERVICE_ROOT / "data" / "training"
    data_dir.mkdir(parents=True, exist_ok=True)
    output_path = data_dir / f"iter_{iteration}_{board}_{players}p.npz"

    replay_db_path = Path(config["replay_db"])
    canonical_mode = bool(config.get("canonical_mode", False))
    if canonical_mode and not dry_run:
        ok, problems = _validate_canonical_training_source(
            db_path=replay_db_path,
            registry_path=Path(config["registry_path"]),
            allow_pending_gate=bool(config.get("allow_pending_gate", False)),
        )
        if not ok:
            for issue in problems:
                print(f"[canonical-source-error] {issue}", file=sys.stderr)
            return False, output_path

    cmd = [
        "python",
        "scripts/export_replay_dataset.py",
        "--db",
        str(replay_db_path),
        "--board-type",
        board,
        "--num-players",
        str(players),
        "--output",
        str(output_path),
        "--require-completed",
        "--min-moves",
        "20",
    ]

    print(f"Exporting training data to {output_path}...")
    code, _, stderr = run_command(cmd, dry_run=dry_run, timeout=600, cwd=AI_SERVICE_ROOT)

    if code != 0:
        print(f"Export failed: {stderr[:200]}")
        return False, output_path

    return True, output_path


def train_model(
    config: dict,
    iteration: int,
    data_path: Path,
    dry_run: bool = False,
) -> Tuple[bool, Path]:
    """Fine-tune neural network on new data.

    Returns: (success, model_path)
    """
    board = config["board"]
    players = config["players"]

    models_dir = AI_SERVICE_ROOT / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    iter_model = models_dir / f"{board}_{players}p_iter{iteration}.pth"
    best_model = models_dir / f"{board}_{players}p_best.pth"

    cmd = [
        "python",
        "app/training/train.py",
        "--data-path",
        str(data_path),
        "--board-type",
        board,
        "--num-players",
        str(players),
        "--epochs",
        "10",
        "--batch-size",
        "256",
        "--save-path",
        str(iter_model),
    ]

    # Resume from best model if it exists
    if best_model.exists():
        cmd.extend(["--resume-from", str(best_model)])
        print(f"Fine-tuning from {best_model}...")
    else:
        print("Training new model from scratch...")

    code, _, stderr = run_command(cmd, dry_run=dry_run, timeout=3600, cwd=AI_SERVICE_ROOT)

    if code != 0:
        print(f"Training failed: {stderr[:200]}")
        return False, iter_model

    # Validate the model was created
    if not dry_run and not validate_model(iter_model):
        print(f"Model validation failed for {iter_model}")
        return False, iter_model

    return True, iter_model


def evaluate_model(
    config: dict,
    iteration: int,
    iter_model: Path,
    dry_run: bool = False,
) -> Tuple[bool, dict]:
    """Evaluate new model against baseline.

    Returns: (success, evaluation_summary)
    """
    board = config["board"]
    players = config["players"]
    best_model = AI_SERVICE_ROOT / "models" / f"{board}_{players}p_best.pth"

    if players != 2:
        print(
            "Evaluation currently supports only 2-player games "
            f"(requested players={players}).",
            file=sys.stderr,
        )
        return False, {}

    eval_games = int(config.get("eval_games", 100))
    seed = int(config.get("eval_seed_base", 10_000)) + int(iteration)
    max_moves = int(config.get("max_moves", 200))

    log_dir = AI_SERVICE_ROOT / "logs" / "improvement"
    log_dir.mkdir(parents=True, exist_ok=True)
    eval_out = log_dir / f"eval_iter{iteration}_{board}_{players}p.json"

    # Determine opponent (prefer NN-vs-NN when a best model exists).
    if best_model.exists():
        opponent = "neural_network"
        opponent_args = ["--checkpoint2", str(best_model)]
    else:
        opponent = "baseline_heuristic"
        opponent_args = []

    cmd = [
        "python",
        "scripts/evaluate_ai_models.py",
        "--player1",
        "neural_network",
        "--player2",
        opponent,
        *opponent_args,
        "--board",
        board,
        "--games",
        str(eval_games),
        "--max-moves",
        str(max_moves),
        "--seed",
        str(seed),
        "--checkpoint",
        str(iter_model),
        "--output",
        str(eval_out),
        "--quiet",
    ]

    print(f"Evaluating {iter_model.name} vs {opponent}...")
    code, _, stderr = run_command(
        cmd,
        dry_run=dry_run,
        capture=False,
        timeout=7200,
        cwd=AI_SERVICE_ROOT,
    )

    if code != 0:
        print(f"Evaluation failed: {stderr[:200]}")
        return False, {}

    if dry_run:
        return True, {
            "games": eval_games,
            "wins": int(eval_games * 0.6),
            "losses": int(eval_games * 0.4),
            "draws": 0,
            "win_rate": 0.6,
        }

    if not eval_out.exists():
        print(f"Evaluation did not produce output file: {eval_out}", file=sys.stderr)
        return False, {}

    payload = json.loads(eval_out.read_text())
    res = payload.get("results", {}) if isinstance(payload, dict) else {}
    wins = int(res.get("player1_wins", 0))
    losses = int(res.get("player2_wins", 0))
    draws = int(res.get("draws", 0))

    summary = _promotion_gate(
        wins=wins,
        losses=losses,
        draws=draws,
        threshold=float(config.get("promotion_threshold", 0.55)),
        confidence=float(config.get("promotion_confidence", 0.95)),
    )
    print(
        f"Win rate: {summary['win_rate']:.1%} "
        f"(CI_low={summary['win_rate_ci_low']:.1%} "
        f"@ {summary['confidence']:.0%})"
    )
    return True, summary


def promote_model(
    config: dict,
    iter_model: Path,
    dry_run: bool = False,
) -> bool:
    """Promote iteration model to best model.

    Creates a backup of the previous best model before overwriting.
    """
    board = config["board"]
    players = config["players"]
    models_dir = AI_SERVICE_ROOT / "models"

    best_model = models_dir / f"{board}_{players}p_best.pth"
    backup_model = models_dir / f"{board}_{players}p_prev_best.pth"

    if dry_run:
        print(f"[DRY-RUN] Would promote {iter_model} to {best_model}")
        return True

    try:
        # Backup previous best
        if best_model.exists():
            shutil.copy2(best_model, backup_model)
            print(f"Backed up previous best to {backup_model}")

        # Promote new model
        shutil.copy2(iter_model, best_model)
        print(f"Promoted {iter_model.name} to {best_model.name}")
        return True
    except Exception as e:
        print(f"Promotion failed: {e}")
        return False


def rollback_model(config: dict, dry_run: bool = False) -> bool:
    """Rollback to previous best model if available."""
    board = config["board"]
    players = config["players"]
    models_dir = AI_SERVICE_ROOT / "models"

    best_model = models_dir / f"{board}_{players}p_best.pth"
    backup_model = models_dir / f"{board}_{players}p_prev_best.pth"

    if not backup_model.exists():
        print("No backup model available for rollback")
        return False

    if dry_run:
        print(f"[DRY-RUN] Would rollback {best_model} from {backup_model}")
        return True

    try:
        shutil.copy2(backup_model, best_model)
        print(f"Rolled back to previous best: {backup_model.name}")
        return True
    except Exception as e:
        print(f"Rollback failed: {e}")
        return False


def run_improvement_iteration(
    iteration: int,
    config: dict,
    state: LoopState,
    dry_run: bool = False,
) -> Tuple[bool, float, int]:
    """Run a single iteration of the improvement loop.

    Returns: (improved, winrate, games_generated)
    """
    # Step 1: Selfplay
    print(f"\n--- Step 1: Selfplay ---")
    success, games = run_selfplay(config, iteration, dry_run)
    if not success:
        return False, 0.0, 0

    # Step 2: Export training data
    print(f"\n--- Step 2: Export Data ---")
    success, data_path = export_training_data(config, iteration, dry_run)
    if not success:
        return False, 0.0, games

    # Step 3: Train model
    print(f"\n--- Step 3: Train Model ---")
    success, iter_model = train_model(config, iteration, data_path, dry_run)
    if not success:
        return False, 0.0, games

    # Step 4: Evaluate
    print(f"\n--- Step 4: Evaluate ---")
    success, eval_summary = evaluate_model(
        config,
        iteration,
        iter_model,
        dry_run,
    )
    if not success:
        return False, 0.0, games

    # Step 5: Promote if improved
    print(f"\n--- Step 5: Promotion Decision ---")
    threshold = config.get("promotion_threshold", 0.55)
    if eval_summary.get("promote", False):
        if promote_model(config, iter_model, dry_run):
            winrate = float(eval_summary.get("win_rate") or 0.0)
            ci_low = eval_summary.get("win_rate_ci_low")
            ci_text = f"{ci_low:.1%}" if isinstance(ci_low, float) else "N/A"
            print(
                "Model promoted! "
                f"Win rate {winrate:.1%} "
                f"(CI_low={ci_text}) >= {threshold:.1%}"
            )
            return True, winrate, games
        else:
            return False, float(eval_summary.get("win_rate") or 0.0), games
    else:
        winrate = float(eval_summary.get("win_rate") or 0.0)
        ci_low = eval_summary.get("win_rate_ci_low")
        ci_text = f"{ci_low:.1%}" if isinstance(ci_low, float) else "N/A"
        print(
            "No improvement: "
            f"win={winrate:.1%}, CI_low={ci_text} < {threshold:.1%}"
        )
        return False, winrate, games


def main():
    parser = argparse.ArgumentParser(
        description="AlphaZero-style continuous improvement loop for RingRift AI"
    )
    parser.add_argument(
        "--board",
        type=str,
        default="square8",
        choices=["square8", "square19", "hexagonal"],
        help="Board type to train on",
    )
    parser.add_argument(
        "--players",
        type=int,
        default=2,
        choices=[2, 3, 4],
        help="Number of players",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=100,
        help="Number of improvement iterations",
    )
    parser.add_argument(
        "--games-per-iter",
        type=int,
        default=100,
        help="Selfplay games per iteration",
    )
    parser.add_argument(
        "--replay-db",
        type=str,
        default=None,
        help=(
            "Path to the GameReplayDB used to record selfplay and export training data. "
            "Default: data/games/canonical_<board>.db (canonical mode) or data/games/selfplay.db (--allow-legacy)."
        ),
    )
    parser.add_argument(
        "--registry",
        type=str,
        default="TRAINING_DATA_REGISTRY.md",
        help="Path to TRAINING_DATA_REGISTRY.md for canonical-source validation (default: TRAINING_DATA_REGISTRY.md).",
    )
    parser.add_argument(
        "--allow-pending-gate",
        action="store_true",
        help=(
            "Allow registry Status=pending_gate as long as the referenced gate summary JSON "
            "reports canonical_ok=true and a passing parity gate."
        ),
    )
    parser.add_argument(
        "--allow-legacy",
        action="store_true",
        help=(
            "Allow training from non-canonical replay DBs. This bypasses canonical gating and uses "
            "the legacy selfplay soak; use only for ablations/debugging."
        ),
    )
    parser.add_argument(
        "--promotion-threshold",
        type=float,
        default=0.55,
        help="Win rate threshold for model promotion",
    )
    parser.add_argument(
        "--promotion-confidence",
        type=float,
        default=0.95,
        help="Wilson CI confidence level for promotion gate (default: 0.95).",
    )
    parser.add_argument(
        "--eval-games",
        type=int,
        default=100,
        help="Number of evaluation games per iteration (default: 100).",
    )
    parser.add_argument(
        "--max-consecutive-failures",
        type=int,
        default=5,
        help="Rollback after this many consecutive non-improvements",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoint",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing",
    )
    parser.add_argument(
        "--state-file",
        type=Path,
        default=None,
        help="Path to state checkpoint file",
    )
    args = parser.parse_args()

    canonical_mode = not bool(args.allow_legacy)
    if args.replay_db:
        replay_db_path = _resolve_ai_service_path(args.replay_db)
    else:
        default_name = f"canonical_{args.board}.db" if canonical_mode else "selfplay.db"
        replay_db_path = (AI_SERVICE_ROOT / "data" / "games" / default_name).resolve()

    registry_path = _resolve_ai_service_path(args.registry)
    gate_summary_path = (AI_SERVICE_ROOT / f"db_health.{replay_db_path.stem}.json").resolve()

    # Configure
    config = {
        "board": args.board,
        "players": args.players,
        "games_per_iter": args.games_per_iter,
        "max_moves": 500 if args.board == "square8" else 1000,
        "promotion_threshold": args.promotion_threshold,
        "promotion_confidence": args.promotion_confidence,
        "eval_games": args.eval_games,
        "replay_db": str(replay_db_path),
        "canonical_mode": canonical_mode,
        "allow_pending_gate": bool(args.allow_pending_gate),
        "registry_path": str(registry_path),
        "gate_summary": str(gate_summary_path),
    }

    # State file path
    if args.state_file:
        state_path = args.state_file
    else:
        state_path = AI_SERVICE_ROOT / "logs" / "improvement" / f"{args.board}_{args.players}p_state.json"

    # Load or initialize state
    state = load_state(state_path) if args.resume else LoopState()
    start_iter = state.iteration if args.resume else 0

    print("=" * 60)
    print("RingRift AI Improvement Loop")
    print("=" * 60)
    print(f"Board: {args.board}, Players: {args.players}")
    print(f"Replay DB: {replay_db_path}")
    print(f"Data policy: {'canonical' if canonical_mode else 'legacy (UNSAFE)'}")
    if canonical_mode:
        print(f"Gate summary: {gate_summary_path}")
    print(f"Iterations: {start_iter + 1} to {args.iterations}")
    print(f"Games per iteration: {args.games_per_iter}")
    print(f"Eval games per iteration: {args.eval_games}")
    print(f"Promotion threshold: {args.promotion_threshold:.0%}")
    print(f"Promotion confidence: {args.promotion_confidence:.0%}")
    print(f"State file: {state_path}")
    if args.dry_run:
        print("*** DRY RUN MODE - No commands will be executed ***")
    print("=" * 60)

    for i in range(start_iter, args.iterations):
        print(f"\n{'='*60}")
        print(f"=== Improvement Iteration {i+1}/{args.iterations} ===")
        print(f"{'='*60}")

        try:
            improved, winrate, games = run_improvement_iteration(
                i, config, state, args.dry_run
            )

            # Update state
            state.iteration = i + 1
            state.total_games_generated += games
            state.history.append({
                "iteration": i + 1,
                "improved": improved,
                "winrate": winrate,
                "games": games,
            })

            if improved:
                state.total_improvements += 1
                state.best_winrate = winrate
                state.consecutive_failures = 0
                state.best_model_path = f"models/{args.board}_{args.players}p_best.pth"
                print(f"\nModel improved! (Total: {state.total_improvements})")
            else:
                state.consecutive_failures += 1
                print(
                    f"\nNo improvement (consecutive failures: "
                    f"{state.consecutive_failures})"
                )

                # Rollback if too many failures
                if state.consecutive_failures >= args.max_consecutive_failures:
                    print(
                        f"\nWARNING: {state.consecutive_failures} consecutive "
                        f"failures, attempting rollback..."
                    )
                    if rollback_model(config, args.dry_run):
                        state.consecutive_failures = 0

            save_state(state, state_path)

        except KeyboardInterrupt:
            print("\n\nInterrupted by user. Saving state...")
            save_state(state, state_path)
            print(f"State saved to {state_path}. Resume with --resume")
            sys.exit(0)

        except Exception as e:
            print(f"\nERROR in iteration {i+1}: {e}")
            state.consecutive_failures += 1
            save_state(state, state_path)

        # Brief pause between iterations
        if not args.dry_run:
            time.sleep(5)

    # Final summary
    print(f"\n{'='*60}")
    print("Improvement Loop Complete!")
    print(f"{'='*60}")
    print(f"Total iterations: {state.iteration}")
    print(f"Total improvements: {state.total_improvements}")
    print(f"Total games generated: {state.total_games_generated}")
    print(f"Best win rate achieved: {state.best_winrate:.1%}")
    if state.best_model_path:
        print(f"Best model: {state.best_model_path}")


if __name__ == "__main__":
    main()
