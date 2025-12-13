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


def _resolve_default_reanalysis_nn_model_id(
    board: str,
    num_players: int,
) -> Optional[str]:
    """Best-effort nn_model_id prefix for reanalysis search.

    When the improvement loop has not produced a `*_best.pth` yet, we still
    want reanalysis to work on square8 2-player runs by falling back to
    known-good baseline prefixes shipped in ai-service/models.
    """
    if board != "square8" or num_players != 2:
        return None

    models_dir = AI_SERVICE_ROOT / "models"
    candidates = [
        "ringrift_v5_sq8_2p_2xh100",
        "ringrift_v4_sq8_2p",
        "ringrift_v3_sq8_2p",
        "sq8_2p_nn_baseline",
    ]
    for prefix in candidates:
        matches = list(models_dir.glob(f"{prefix}*.pth"))
        matches = [
            p
            for p in matches
            if p.is_file()
            and p.stat().st_size > 0
        ]
        if matches:
            return prefix
    return None


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
    ingested_db_fingerprints: Dict[str, str] = field(default_factory=dict)


def load_state(state_path: Path) -> LoopState:
    """Load checkpoint state or return fresh state."""
    if state_path.exists():
        try:
            data = json.loads(state_path.read_text())
            # Handle history field migration
            if "history" not in data:
                data["history"] = []
            if not isinstance(data.get("ingested_db_fingerprints"), dict):
                data["ingested_db_fingerprints"] = {}
            return LoopState(**data)
        except (json.JSONDecodeError, TypeError) as e:
            print(f"WARNING: Could not load state from {state_path}: {e}")
    return LoopState()


def save_state(state: LoopState, state_path: Path) -> None:
    """Persist checkpoint state."""
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(asdict(state), indent=2))


def _fingerprint_path(path: Path) -> str:
    """Return a stable fingerprint for change detection (mtime+size)."""
    stat = path.stat()
    return f"{int(stat.st_size)}:{int(getattr(stat, 'st_mtime_ns', int(stat.st_mtime * 1e9)))}"


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
) -> Tuple[bool, int, Optional[Path]]:
    """Run selfplay to generate training data.

    Returns: (success, games_generated, staging_db_path)
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

    staging_db_path: Optional[Path] = None
    record_db_path = replay_db_path
    if canonical_mode:
        staging_dir = _resolve_ai_service_path(str(config.get("staging_db_dir") or "data/games/staging/improvement_loop"))
        staging_dir.mkdir(parents=True, exist_ok=True)
        staging_db_path = (staging_dir / f"selfplay_iter{iteration}_{board}_{players}p.db").resolve()
        record_db_path = staging_db_path
        if record_db_path.exists() and not dry_run:
            archived = Path(f"{record_db_path.as_posix()}.archived_{time.strftime('%Y%m%d_%H%M%S')}")
            record_db_path.rename(archived)
            print(f"[selfplay] archived existing staging DB -> {archived}", file=sys.stderr)

    cmd = [
        sys.executable,
        "scripts/run_self_play_soak.py",
        "--num-games",
        str(games),
        "--board-type",
        board,
        "--engine-mode",
        "mixed",
        "--difficulty-band",
        str(config.get("selfplay_difficulty_band", "canonical")),
        "--num-players",
        str(players),
        "--max-moves",
        str(max_moves),
        "--seed",
        str(seed),
        "--record-db",
        str(record_db_path),
        "--log-jsonl",
        str(log_file),
    ]

    print(f"Running selfplay: {games} games on {board} {players}p...")
    code, _, stderr = run_command(cmd, dry_run=dry_run, timeout=3600, cwd=AI_SERVICE_ROOT)

    if code != 0:
        print(f"Selfplay failed: {stderr[:200]}")
        return False, 0, staging_db_path

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

    return True, games_completed if not dry_run else games, staging_db_path


def ingest_training_pool(
    config: dict,
    iteration: int,
    state: LoopState,
    *,
    staging_db_path: Optional[Path],
    dry_run: bool = False,
) -> bool:
    """Ingest newly discovered staging DBs into the canonical training pool DB."""
    if not bool(config.get("canonical_mode", False)):
        return True

    output_db = Path(config["replay_db"]).resolve()
    board = str(config["board"])
    players = int(config["players"])

    holdout_db_raw = config.get("holdout_db")
    holdout_db = _resolve_ai_service_path(str(holdout_db_raw)).resolve() if holdout_db_raw else None
    quarantine_db_raw = config.get("quarantine_db")
    quarantine_db = _resolve_ai_service_path(str(quarantine_db_raw)).resolve() if quarantine_db_raw else None

    excluded = {output_db}
    if holdout_db is not None:
        excluded.add(holdout_db)
    if quarantine_db is not None:
        excluded.add(quarantine_db)

    scan_dirs: List[Path] = []
    for raw in (config.get("ingest_scan_dirs") or []):
        try:
            scan_dirs.append(_resolve_ai_service_path(str(raw)).resolve())
        except Exception:
            continue

    candidates: List[Path] = []
    if staging_db_path is not None:
        candidates.append(staging_db_path.resolve())
    for scan_dir in scan_dirs:
        if not scan_dir.exists():
            continue
        for path in scan_dir.rglob("*.db"):
            if path.is_file():
                candidates.append(path.resolve())

    # De-dupe while preserving stable ordering.
    seen: set[Path] = set()
    unique_candidates: List[Path] = []
    for path in candidates:
        if path in seen:
            continue
        seen.add(path)
        unique_candidates.append(path)

    inputs: List[Path] = []
    for path in unique_candidates:
        if path in excluded:
            continue
        try:
            fingerprint = _fingerprint_path(path)
        except OSError:
            continue
        key = str(path)
        if state.ingested_db_fingerprints.get(key) == fingerprint:
            continue
        inputs.append(path)

    if not inputs:
        return True

    log_dir = AI_SERVICE_ROOT / "logs" / "improvement"
    log_dir.mkdir(parents=True, exist_ok=True)
    report_json = log_dir / f"training_pool_ingest_iter{iteration}_{board}_{players}p.json"

    cmd: List[str] = [
        sys.executable,
        "scripts/build_canonical_training_pool_db.py",
        "--output-db",
        str(output_db),
        "--board-type",
        board,
        "--num-players",
        str(players),
        "--require-completed",
        "--report-json",
        str(report_json),
    ]
    if holdout_db is not None:
        cmd += ["--holdout-db", str(holdout_db)]
    if quarantine_db is not None:
        cmd += ["--quarantine-db", str(quarantine_db)]
    for path in inputs:
        cmd += ["--input-db", str(path)]

    print(f"\n--- Training Pool Ingest ({len(inputs)} DBs) ---")
    code, stdout, stderr = run_command(
        cmd,
        dry_run=dry_run,
        capture=True,
        timeout=7200,
        cwd=AI_SERVICE_ROOT,
    )
    if code != 0:
        print(f"[ingest] failed: {stderr[:400]}")
        return False

    if not dry_run:
        try:
            report = json.loads(report_json.read_text())
        except Exception:
            report = {}
        totals = report.get("totals") or {}
        print(f"[ingest] totals: {json.dumps(totals, sort_keys=True)}")

        for path in inputs:
            try:
                state.ingested_db_fingerprints[str(path)] = _fingerprint_path(path)
            except OSError:
                continue

        if staging_db_path is not None and int(totals.get("training_passed") or 0) <= 0:
            print("[ingest] No training games passed gates from staging DB; aborting iteration.", file=sys.stderr)
            return False

    return True


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

    dataset_policy_target = str(config.get("dataset_policy_target", "played")).strip()
    dataset_max_games = config.get("dataset_max_games", None)
    legacy_maxn_encoding = bool(config.get("legacy_maxn_encoding", False))
    use_board_aware_encoding = board in {"square8", "square19"} and not legacy_maxn_encoding

    cmd: List[str]
    timeout_sec = 600
    if dataset_policy_target == "played":
        cmd = [
            sys.executable,
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
    else:
        policy_search_think_time_ms = int(config.get("policy_search_think_time_ms", 50))
        policy_temperature = float(config.get("policy_temperature", 1.0))

        nn_model_id = config.get("policy_nn_model_id", None)
        if nn_model_id is None:
            best_model = AI_SERVICE_ROOT / "models" / f"{board}_{players}p_best.pth"
            if best_model.exists():
                nn_model_id = best_model.stem
        if nn_model_id is None:
            nn_model_id = _resolve_default_reanalysis_nn_model_id(board, int(players))

        if not nn_model_id:
            print(
                "[reanalysis] No usable NN checkpoint found; "
                "falling back to played policy targets. "
                "(Pass --policy-nn-model-id or train a *_best.pth to enable reanalysis.)",
                file=sys.stderr,
            )
            dataset_policy_target = "played"
            cmd = [
                sys.executable,
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
        else:
            cmd = [
                sys.executable,
                "scripts/reanalyze_replay_dataset.py",
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
                "--policy-target",
                dataset_policy_target,
                "--policy-search-think-time-ms",
                str(policy_search_think_time_ms),
                "--policy-temperature",
                str(policy_temperature),
                "--nn-model-id",
                str(nn_model_id),
            ]
            timeout_sec = 7200

    if dataset_max_games is not None:
        cmd += ["--max-games", str(int(dataset_max_games))]
    if use_board_aware_encoding:
        cmd.append("--board-aware-encoding")

    if dataset_policy_target == "played":
        print(f"Exporting training data to {output_path}...")
    else:
        print(
            f"Reanalyzing training data ({dataset_policy_target}) to {output_path}..."
        )
    code, _, stderr = run_command(
        cmd,
        dry_run=dry_run,
        timeout=timeout_sec,
        cwd=AI_SERVICE_ROOT,
    )

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
        sys.executable,
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
    max_moves = int(config.get("max_moves", 10000))

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
        sys.executable,
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
    success, games, staging_db_path = run_selfplay(config, iteration, dry_run)
    if not success:
        return False, 0.0, 0

    # Step 2: Ingest staging sources (canonical mode only)
    if bool(config.get("canonical_mode", False)):
        print(f"\n--- Step 2: Ingest Training Pool ---")
        if not ingest_training_pool(
            config,
            iteration,
            state,
            staging_db_path=staging_db_path,
            dry_run=dry_run,
        ):
            return False, 0.0, games

    # Step 3: Export training data
    print(f"\n--- Step 3: Export Data ---")
    success, data_path = export_training_data(config, iteration, dry_run)
    if not success:
        return False, 0.0, games

    # Step 4: Train model
    print(f"\n--- Step 4: Train Model ---")
    success, iter_model = train_model(config, iteration, data_path, dry_run)
    if not success:
        return False, 0.0, games

    # Step 5: Evaluate
    print(f"\n--- Step 5: Evaluate ---")
    success, eval_summary = evaluate_model(
        config,
        iteration,
        iter_model,
        dry_run,
    )
    if not success:
        return False, 0.0, games

    # Step 6: Promote if improved
    print(f"\n--- Step 6: Promotion Decision ---")
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
            "Path to the canonical training-pool GameReplayDB used for dataset export/training. "
            "In canonical mode, self-play games are recorded into per-iteration staging DBs and then "
            "ingested into this DB (per-game parity + canonical-history gates). "
            "Default: data/games/canonical_<board>[_<players>p].db (canonical mode) or data/games/selfplay.db (--allow-legacy)."
        ),
    )
    parser.add_argument(
        "--staging-db-dir",
        type=str,
        default="data/games/staging/improvement_loop",
        help=(
            "Directory for per-iteration staging GameReplayDBs (canonical mode only). "
            "These DBs are gated and ingested into --replay-db."
        ),
    )
    parser.add_argument(
        "--ingest-scan-dir",
        action="append",
        default=[],
        help=(
            "Additional directories to scan for staging GameReplayDBs to ingest each iteration "
            "(repeatable). Use this to include CMA-ES, hybrid, soak, and other sources as long as "
            "they pass per-game parity + canonical-history gates."
        ),
    )
    parser.add_argument(
        "--holdout-db",
        type=str,
        default=None,
        help=(
            "Optional path to write holdout (tournament/eval) games into a separate DB. "
            "Default: data/games/holdouts/holdout_<board>_<players>p.db"
        ),
    )
    parser.add_argument(
        "--quarantine-db",
        type=str,
        default=None,
        help=(
            "Optional path to write games that fail gates into a quarantine DB. "
            "Default: data/games/quarantine/quarantine_<board>_<players>p.db"
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
        "--max-moves",
        type=int,
        default=10000,
        help="Maximum moves per game before applying evaluation tie-breaks (default: 10000).",
    )
    parser.add_argument(
        "--selfplay-difficulty-band",
        type=str,
        choices=["canonical", "light"],
        default="canonical",
        help=(
            "Difficulty band for canonical self-play. "
            "'canonical' samples across the full ladder (includes MCTS/Descent+NN); "
            "'light' restricts to Random/Heuristic/Minimax for speed. "
            "(default: canonical)"
        ),
    )
    parser.add_argument(
        "--dataset-policy-target",
        type=str,
        choices=["played", "mcts_visits", "descent_softmax"],
        default="mcts_visits",
        help=(
            "Policy target source for exported datasets. "
            "'played' exports 1-hot played moves (fast). "
            "'mcts_visits' or 'descent_softmax' runs search-based reanalysis "
            "to produce soft policy targets (stronger). "
            "(default: mcts_visits)"
        ),
    )
    parser.add_argument(
        "--dataset-max-games",
        type=int,
        default=None,
        help=(
            "Optional cap on number of most-recent games to include when exporting/reanalyzing datasets. "
            "When unset, uses up to 10k games from the DB."
        ),
    )
    parser.add_argument(
        "--legacy-maxn-encoding",
        action="store_true",
        help=(
            "Export/reanalyze datasets using the legacy MAX_N policy encoding "
            "(larger action space). Default is board-aware encoding for square boards "
            "to support v3 training."
        ),
    )
    parser.add_argument(
        "--policy-search-think-time-ms",
        type=int,
        default=50,
        help="Search think time (ms) for reanalysis policy targets (default: 50).",
    )
    parser.add_argument(
        "--policy-temperature",
        type=float,
        default=1.0,
        help="Softmax temperature for descent_softmax reanalysis (default: 1.0).",
    )
    parser.add_argument(
        "--policy-nn-model-id",
        type=str,
        default=None,
        help=(
            "Optional NeuralNetAI nn_model_id prefix for reanalysis search. "
            "When unset, uses the current best model file stem if available."
        ),
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
        "max_moves": int(args.max_moves),
        "promotion_threshold": args.promotion_threshold,
        "promotion_confidence": args.promotion_confidence,
        "eval_games": args.eval_games,
        "selfplay_difficulty_band": args.selfplay_difficulty_band,
        "dataset_policy_target": args.dataset_policy_target,
        "dataset_max_games": args.dataset_max_games,
        "legacy_maxn_encoding": bool(args.legacy_maxn_encoding),
        "policy_search_think_time_ms": args.policy_search_think_time_ms,
        "policy_temperature": args.policy_temperature,
        "policy_nn_model_id": args.policy_nn_model_id,
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
    print(f"Self-play difficulty band: {args.selfplay_difficulty_band}")
    print(f"Eval games per iteration: {args.eval_games}")
    print(f"Promotion threshold: {args.promotion_threshold:.0%}")
    print(f"Promotion confidence: {args.promotion_confidence:.0%}")
    print(f"Dataset policy target: {args.dataset_policy_target}")
    if args.dataset_max_games is not None:
        print(f"Dataset max games: {args.dataset_max_games}")
    if args.dataset_policy_target != "played":
        print(f"Reanalysis think time: {args.policy_search_think_time_ms}ms")
        print(f"Reanalysis temperature: {args.policy_temperature}")
        if args.policy_nn_model_id:
            print(f"Reanalysis nn_model_id: {args.policy_nn_model_id}")
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
