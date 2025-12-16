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
import asyncio
import json
import os
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Allow imports from app/
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Light-weight significance helper (used for promotion gating).
from app.training.significance import wilson_score_interval

# Evaluation caching for pool-based promotion gating
try:
    from app.training.eval_cache import EvalCache, get_eval_cache
    HAS_EVAL_CACHE = True
except ImportError:
    HAS_EVAL_CACHE = False
    EvalCache = None  # type: ignore
    get_eval_cache = None  # type: ignore

# Diverse tournament orchestrator integration (for periodic Elo calibration)
try:
    from scripts.run_diverse_tournaments import (
        run_tournament_round_distributed,
        run_tournament_round_local,
        load_cluster_hosts,
        filter_available_hosts,
        build_tournament_configs,
        TournamentConfig,
        TournamentResult,
        DEFAULT_GAMES_PER_CONFIG,
    )
    HAS_DIVERSE_TOURNAMENTS = True
except ImportError:
    HAS_DIVERSE_TOURNAMENTS = False

# Adaptive controller for dynamic loop behavior
try:
    from app.training.adaptive_controller import (
        AdaptiveController,
        create_adaptive_controller,
    )
    HAS_ADAPTIVE_CONTROLLER = True
except ImportError:
    HAS_ADAPTIVE_CONTROLLER = False
    AdaptiveController = None  # type: ignore
    create_adaptive_controller = None  # type: ignore

# Hot data buffer for streaming training
try:
    from app.training.hot_data_buffer import (
        HotDataBuffer,
        create_hot_buffer,
    )
    HAS_HOT_BUFFER = True
except ImportError:
    HAS_HOT_BUFFER = False
    HotDataBuffer = None  # type: ignore
    create_hot_buffer = None  # type: ignore

# Unified resource guard - 80% utilization limits (enforced 2025-12-16)
try:
    from app.utils.resource_guard import (
        can_proceed as resource_can_proceed,
        check_disk_space,
        check_memory,
        check_cpu,
        check_gpu_memory,
        wait_for_resources,
        get_resource_status,
        LIMITS as RESOURCE_LIMITS,
    )
    HAS_RESOURCE_GUARD = True
except ImportError:
    HAS_RESOURCE_GUARD = False
    resource_can_proceed = lambda **kwargs: True  # type: ignore
    check_disk_space = lambda *args, **kwargs: True  # type: ignore
    check_memory = lambda *args, **kwargs: True  # type: ignore
    check_cpu = lambda *args, **kwargs: True  # type: ignore
    check_gpu_memory = lambda *args, **kwargs: True  # type: ignore
    wait_for_resources = lambda *args, **kwargs: True  # type: ignore
    get_resource_status = lambda: {}  # type: ignore
    RESOURCE_LIMITS = None  # type: ignore

AI_SERVICE_ROOT = Path(__file__).resolve().parents[1]


# =============================================================================
# Load-based Job Throttling
# =============================================================================
# Prevents running jobs when system is overloaded

# Max load average (relative to CPU count) before refusing jobs
MAX_LOAD_FACTOR = float(os.environ.get("RINGRIFT_MAX_LOAD_FACTOR", "2.0"))
# Absolute max load average regardless of CPU count
MAX_LOAD_ABSOLUTE = float(os.environ.get("RINGRIFT_MAX_LOAD_ABSOLUTE", "100.0"))
# How long to wait when overloaded before rechecking (seconds)
LOAD_BACKOFF_SECONDS = float(os.environ.get("RINGRIFT_LOAD_BACKOFF_SECONDS", "30.0"))


def get_system_load() -> Tuple[float, float, float]:
    """Get system load averages (1min, 5min, 15min)."""
    try:
        return os.getloadavg()
    except (OSError, AttributeError):
        return (0.0, 0.0, 0.0)


def get_cpu_count() -> int:
    """Get number of CPU cores."""
    try:
        return os.cpu_count() or 1
    except Exception:
        return 1


def is_system_overloaded(verbose: bool = False) -> bool:
    """Check if system is too overloaded to run jobs.

    Returns True if load average exceeds thresholds.
    """
    load_1min, load_5min, _ = get_system_load()
    cpu_count = get_cpu_count()

    relative_threshold = MAX_LOAD_FACTOR * cpu_count
    is_overloaded = load_5min > relative_threshold or load_5min > MAX_LOAD_ABSOLUTE

    if is_overloaded and verbose:
        print(f"[improvement] System overloaded: load={load_5min:.1f}, "
              f"threshold={min(relative_threshold, MAX_LOAD_ABSOLUTE):.1f}", file=sys.stderr)

    return is_overloaded


def wait_for_load_decrease(max_wait_seconds: float = 300.0, verbose: bool = True) -> bool:
    """Wait for system load to decrease. Returns True if load decreased."""
    start_time = time.time()

    while is_system_overloaded(verbose=False):
        elapsed = time.time() - start_time
        if elapsed >= max_wait_seconds:
            if verbose:
                _, load_5min, _ = get_system_load()
                print(f"[improvement] Load still high after {elapsed:.0f}s "
                      f"(load={load_5min:.1f}). Giving up.", file=sys.stderr)
            return False

        if verbose:
            _, load_5min, _ = get_system_load()
            print(f"[improvement] Waiting for load to decrease "
                  f"(current={load_5min:.1f}, waited={elapsed:.0f}s)...", file=sys.stderr)

        time.sleep(LOAD_BACKOFF_SECONDS)

    return True


class OverlappedPipelineManager:
    """Manages overlapped pipeline stages for faster iteration.

    Enables selfplay for iteration N+1 to run concurrently with training/eval for N.
    This reduces total iteration time by overlapping data generation with training.
    """

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self._selfplay_future: Optional[asyncio.Task] = None
        self._selfplay_result: Optional[Tuple[bool, int, Optional[Path]]] = None
        self._loop = None

    def _get_loop(self):
        """Get or create event loop."""
        if self._loop is None or self._loop.is_closed():
            try:
                self._loop = asyncio.get_event_loop()
            except RuntimeError:
                self._loop = asyncio.new_event_loop()
                asyncio.set_event_loop(self._loop)
        return self._loop

    def start_selfplay_async(
        self,
        config: dict,
        iteration: int,
        dry_run: bool = False,
    ) -> None:
        """Start selfplay for next iteration in background.

        This allows training to proceed while data generation happens.
        """
        if not self.enabled:
            return

        async def _run_selfplay_async():
            # Run selfplay in executor to not block event loop
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: run_selfplay(config, iteration, dry_run)
            )
            return result

        loop = self._get_loop()
        self._selfplay_future = loop.create_task(_run_selfplay_async())

    def get_selfplay_result(
        self,
        config: dict,
        iteration: int,
        dry_run: bool = False,
    ) -> Tuple[bool, int, Optional[Path]]:
        """Get selfplay result, waiting if async task is running.

        If no async task was started, runs selfplay synchronously.
        """
        if self._selfplay_future is not None:
            loop = self._get_loop()
            try:
                # Wait for async selfplay to complete
                result = loop.run_until_complete(self._selfplay_future)
                self._selfplay_future = None
                return result
            except Exception as e:
                print(f"[overlapped] Async selfplay failed: {e}")
                self._selfplay_future = None
                # Fall through to synchronous execution

        # Run synchronously if no async task or it failed
        return run_selfplay(config, iteration, dry_run)

    def cancel_pending(self) -> None:
        """Cancel any pending async tasks."""
        if self._selfplay_future is not None:
            self._selfplay_future.cancel()
            self._selfplay_future = None


def _atomic_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + f".tmp_{int(time.time() * 1e6)}")
    shutil.copy2(src, tmp)
    os.replace(tmp, dst)


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp_{int(time.time() * 1e6)}")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
    os.replace(tmp, path)


def _alias_token_for_board(board: str) -> str:
    board = str(board).strip().lower()
    if board == "square8":
        return "sq8"
    if board == "square19":
        return "sq19"
    if board == "hexagonal":
        return "hex"
    raise ValueError(f"Unsupported board for alias: {board!r}")


def publish_ringrift_best_alias(
    *,
    board: str,
    num_players: int,
    best_model_path: Path,
    promotion_summary: Optional[dict] = None,
) -> Path:
    """Publish a ringrift_best_* alias so the canonical ladder can consume the model."""
    token = _alias_token_for_board(board)
    alias_id = f"ringrift_best_{token}_{int(num_players)}p"

    models_dir = AI_SERVICE_ROOT / "models"
    cpu_dst = models_dir / f"{alias_id}.pth"
    mps_dst = models_dir / f"{alias_id}_mps.pth"
    meta_dst = models_dir / f"{alias_id}.meta.json"

    _atomic_copy(best_model_path, cpu_dst)
    # Best-effort: if no MPS-specific checkpoint exists, reuse the CPU weights.
    _atomic_copy(best_model_path, mps_dst)

    summary = promotion_summary if isinstance(promotion_summary, dict) else {}
    _write_json_atomic(
        meta_dst,
        {
            "alias_id": alias_id,
            "board_type": str(board),
            "num_players": int(num_players),
            "source_model_id": best_model_path.stem,
            "source_checkpoint": str(best_model_path),
            "source_checkpoint_mps": str(best_model_path),
            "elo_rating": 0.0,
            "games_played": int(summary.get("games", 0) or 0),
            "published_at": datetime.utcnow().isoformat() + "Z",
            "promotion_method": "improvement_loop",
            "promotion_summary": summary,
        },
    )

    return cpu_dst


def maybe_sync_staging(
    *,
    enabled: bool,
    restart: bool,
    validate_health: bool,
    fail_on_missing: bool,
    dry_run: bool,
    reason: str,
) -> bool:
    """Best-effort: push promoted artifacts to staging via SSH."""
    if not enabled:
        return False
    if dry_run:
        print(f"[DRY-RUN] Would sync staging artifacts ({reason})")
        return True

    if not os.environ.get("RINGRIFT_STAGING_SSH_HOST") or not os.environ.get("RINGRIFT_STAGING_ROOT"):
        print(
            f"[staging_sync] Requested ({reason}) but missing "
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

    code, stdout, stderr = run_command(
        cmd,
        dry_run=False,
        capture=True,
        timeout=900,
        cwd=AI_SERVICE_ROOT,
    )
    output = stdout or stderr
    if output:
        print(output)
    return code == 0


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


def _log_promotion_decision(
    *,
    config: dict,
    iteration: int,
    summary: dict,
    model_path: Optional[Path] = None,
) -> None:
    """Log promotion decision to JSONL file for audit trail.

    This allows tracking of all promotion decisions over time, enabling
    analysis of model quality trends and debugging promotion issues.
    """
    logs_dir = AI_SERVICE_ROOT / "logs" / "improvement"
    logs_dir.mkdir(parents=True, exist_ok=True)

    board = config.get("board", "unknown")
    players = config.get("players", 2)
    log_file = logs_dir / f"{board}_{players}p_promotion_gates.jsonl"

    record = {
        "timestamp": datetime.now().isoformat(),
        "board": board,
        "players": players,
        "iteration": iteration,
        "model_path": str(model_path) if model_path else None,
        **summary,
    }

    # Append atomically (write to temp, then append)
    try:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
            f.flush()
    except OSError as e:
        print(f"Warning: Could not log promotion decision: {e}", file=sys.stderr)


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
    # Check system load before starting
    if is_system_overloaded(verbose=True):
        if not wait_for_load_decrease(max_wait_seconds=300.0, verbose=True):
            print("[selfplay] Skipping due to system overload", file=sys.stderr)
            return False, 0, None

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

    engine_mode = str(config.get("selfplay_engine_mode", "mixed"))
    cmd = [
        sys.executable,
        "scripts/run_self_play_soak.py",
        "--num-games",
        str(games),
        "--board-type",
        board,
        "--engine-mode",
        engine_mode,
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
    nn_pool_size = int(config.get("selfplay_nn_pool_size", 0) or 0)
    nn_pool_dir = config.get("selfplay_nn_pool_dir")
    if nn_pool_size > 0:
        cmd += ["--nn-pool-size", str(nn_pool_size)]
    if nn_pool_dir:
        cmd += ["--nn-pool-dir", str(nn_pool_dir)]

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
    parallel_workers: int = 8,
) -> bool:
    """Ingest newly discovered staging DBs into the canonical training pool DB.

    Args:
        parallel_workers: Number of parallel workers for validation.
            Default 8 reduces validation time from ~30min to ~5min.
    """
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
    if parallel_workers > 1:
        cmd += ["--parallel-workers", str(parallel_workers)]
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
    # Check system load before starting
    if is_system_overloaded(verbose=True):
        if not wait_for_load_decrease(max_wait_seconds=300.0, verbose=True):
            print("[training] Skipping due to system overload", file=sys.stderr)
            return False, Path("")

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

    # Resume from best model if it exists (warm-start fine-tuning)
    if best_model.exists():
        cmd.extend(["--resume", str(best_model)])
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


def _run_pool_opponent_eval(
    *,
    iter_model: Path,
    opponent: Path,
    board: str,
    pool_games: int,
    max_moves: int,
    seed: int,
    out_path: Path,
    dry_run: bool,
) -> Tuple[bool, int, int, int, str]:
    """Run evaluation against a single pool opponent.

    Returns: (success, wins, losses, draws, error_msg)
    """
    if dry_run:
        return True, int(pool_games * 0.55), int(pool_games * 0.45), 0, ""

    cmd = [
        sys.executable,
        "scripts/evaluate_ai_models.py",
        "--player1",
        "neural_network",
        "--player2",
        "neural_network",
        "--checkpoint",
        str(iter_model),
        "--checkpoint2",
        str(opponent),
        "--board",
        board,
        "--games",
        str(pool_games),
        "--max-moves",
        str(max_moves),
        "--seed",
        str(seed),
        "--output",
        str(out_path),
        "--quiet",
    ]

    code, _, stderr = run_command(
        cmd,
        dry_run=False,
        capture=False,
        timeout=7200,
        cwd=AI_SERVICE_ROOT,
    )
    if code != 0:
        return False, 0, 0, 0, f"eval failed: {stderr[:200]}"

    if not out_path.exists():
        return False, 0, 0, 0, f"missing output file: {out_path}"

    try:
        payload = json.loads(out_path.read_text())
        res = payload.get("results", {}) if isinstance(payload, dict) else {}
        wins = int(res.get("player1_wins", 0))
        losses = int(res.get("player2_wins", 0))
        draws = int(res.get("draws", 0))
        return True, wins, losses, draws, ""
    except Exception as e:
        return False, 0, 0, 0, f"parse error: {e}"


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

    # Adaptive evaluation: use fewer games for clear results, more for uncertain
    # Starts with initial_games (default 50), extends to max_games if win rate is 45-55%
    initial_games = int(config.get("eval_initial_games", 50))
    max_eval_games = int(config.get("eval_games", 100))
    adaptive_eval = bool(config.get("adaptive_eval", True))
    eval_games = initial_games if adaptive_eval else max_eval_games
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

    # Adaptive evaluation: extend if result is uncertain (45-55% win rate)
    total_games = wins + losses + draws
    if adaptive_eval and total_games > 0 and total_games < max_eval_games:
        win_rate = wins / total_games
        uncertain_lower = float(config.get("eval_uncertain_lower", 0.45))
        uncertain_upper = float(config.get("eval_uncertain_upper", 0.55))
        if uncertain_lower < win_rate < uncertain_upper:
            extra_games = max_eval_games - total_games
            print(f"  Win rate {win_rate:.1%} is uncertain, extending with {extra_games} more games...")
            eval_out_ext = log_dir / f"eval_iter{iteration}_{board}_{players}p_ext.json"
            cmd_ext = cmd.copy()
            # Update games and seed in command
            for i, arg in enumerate(cmd_ext):
                if arg == "--games" and i + 1 < len(cmd_ext):
                    cmd_ext[i + 1] = str(extra_games)
                if arg == "--seed" and i + 1 < len(cmd_ext):
                    cmd_ext[i + 1] = str(seed + 1000)
                if arg == "--output" and i + 1 < len(cmd_ext):
                    cmd_ext[i + 1] = str(eval_out_ext)
            code_ext, _, _ = run_command(cmd_ext, dry_run=dry_run, capture=False, timeout=7200, cwd=AI_SERVICE_ROOT)
            if code_ext == 0 and eval_out_ext.exists():
                try:
                    payload_ext = json.loads(eval_out_ext.read_text())
                    res_ext = payload_ext.get("results", {}) if isinstance(payload_ext, dict) else {}
                    wins += int(res_ext.get("player1_wins", 0))
                    losses += int(res_ext.get("player2_wins", 0))
                    draws += int(res_ext.get("draws", 0))
                    total_games = wins + losses + draws
                    print(f"  Extended evaluation: {wins}W-{losses}L-{draws}D ({total_games} games)")
                except Exception:
                    pass
        else:
            print(f"  Win rate {win_rate:.1%} is decisive, skipping extended evaluation")

    summary = _promotion_gate(
        wins=wins,
        losses=losses,
        draws=draws,
        threshold=float(config.get("promotion_threshold", 0.55)),
        confidence=float(config.get("promotion_confidence", 0.95)),
    )

    # Optional pool gate: require the candidate to be non-regressing vs a
    # recent pool of opponents (best-effort, defaults to "newest checkpoints").
    pool_size = int(config.get("promotion_pool_size", 0) or 0)
    pool_games = int(config.get("promotion_pool_games_per_opponent", 20) or 20)
    pool_threshold = float(config.get("promotion_pool_threshold", 0.5) or 0.5)

    pool_eval: dict | None = None
    if pool_size > 0:
        opponents = _select_promotion_pool_opponents(
            board=board,
            players=int(players),
            models_dir=(AI_SERVICE_ROOT / "models"),
            iter_model=iter_model,
            best_model=best_model if best_model.exists() else None,
            pool_size=pool_size,
        )
        if opponents:
            agg_wins = 0
            agg_losses = 0
            agg_draws = 0
            per_opponent: list[dict] = []

            # Initialize evaluation cache if available
            eval_cache = None
            if HAS_EVAL_CACHE and get_eval_cache is not None:
                try:
                    eval_cache = get_eval_cache(str(log_dir))
                except Exception as e:
                    print(f"[pool-eval] cache init failed: {e}", file=sys.stderr)

            # Check if parallel pool evaluation is enabled
            parallel_pool_eval = bool(config.get("parallel_pool_eval", False))

            # Prepare evaluation tasks
            eval_tasks: list[tuple[int, Path, Path, int, bool]] = []  # (idx, opp, out_path, seed, cached)
            cached_results: dict[int, tuple[int, int, int]] = {}  # idx -> (wins, losses, draws)

            for idx, opp in enumerate(opponents):
                out_path = log_dir / f"eval_iter{iteration}_{board}_{players}p_pool{idx+1}.json"
                opp_seed = seed + 10_000 * (idx + 1)

                # Check cache first
                cached_result = None
                if eval_cache is not None and not dry_run:
                    try:
                        cached_result = eval_cache.get(
                            iter_model, opp, board, int(players), pool_games
                        )
                        if cached_result is not None:
                            print(f"[pool-eval] cache hit vs {opp.name}")
                            cached_results[idx] = (
                                cached_result.model_a_wins,
                                cached_result.model_b_wins,
                                cached_result.draws,
                            )
                    except Exception as e:
                        print(f"[pool-eval] cache lookup failed: {e}", file=sys.stderr)

                if cached_result is None:
                    eval_tasks.append((idx, opp, out_path, opp_seed, False))

            # Run evaluations (parallel or sequential)
            eval_results: dict[int, tuple[bool, int, int, int, str]] = {}

            if eval_tasks:
                if parallel_pool_eval and len(eval_tasks) > 1:
                    print(f"[pool-eval] Running {len(eval_tasks)} evaluations in parallel...")

                    def run_eval_task(task: tuple) -> tuple[int, tuple[bool, int, int, int, str]]:
                        idx, opp, out_path, opp_seed, _ = task
                        result = _run_pool_opponent_eval(
                            iter_model=iter_model,
                            opponent=opp,
                            board=board,
                            pool_games=pool_games,
                            max_moves=max_moves,
                            seed=opp_seed,
                            out_path=out_path,
                            dry_run=dry_run,
                        )
                        return idx, result

                    # Use ThreadPoolExecutor for parallel execution
                    max_workers = min(len(eval_tasks), 4)  # Cap at 4 to avoid resource exhaustion
                    with ThreadPoolExecutor(max_workers=max_workers) as executor:
                        futures = {executor.submit(run_eval_task, task): task[0] for task in eval_tasks}
                        for future in as_completed(futures):
                            try:
                                idx, result = future.result()
                                eval_results[idx] = result
                            except Exception as e:
                                idx = futures[future]
                                eval_results[idx] = (False, 0, 0, 0, f"ThreadPool error: {e}")
                else:
                    # Sequential execution
                    for task in eval_tasks:
                        idx, opp, out_path, opp_seed, _ = task
                        result = _run_pool_opponent_eval(
                            iter_model=iter_model,
                            opponent=opp,
                            board=board,
                            pool_games=pool_games,
                            max_moves=max_moves,
                            seed=opp_seed,
                            out_path=out_path,
                            dry_run=dry_run,
                        )
                        eval_results[idx] = result

            # Aggregate results
            for idx, opp in enumerate(opponents):
                if idx in cached_results:
                    opp_wins, opp_losses, opp_draws = cached_results[idx]
                    was_cached = True
                elif idx in eval_results:
                    success, opp_wins, opp_losses, opp_draws, err_msg = eval_results[idx]
                    if not success:
                        print(f"[pool-eval] failed vs {opp.name}: {err_msg}", file=sys.stderr)
                        return False, {}
                    was_cached = False

                    # Store in cache
                    if eval_cache is not None and not dry_run:
                        try:
                            eval_cache.put(
                                iter_model, opp, board, int(players), pool_games,
                                opp_wins, opp_losses, opp_draws
                            )
                        except Exception as e:
                            print(f"[pool-eval] cache store failed: {e}", file=sys.stderr)
                else:
                    print(f"[pool-eval] no result for opponent {idx}", file=sys.stderr)
                    return False, {}

                agg_wins += opp_wins
                agg_losses += opp_losses
                agg_draws += opp_draws
                per_opponent.append(
                    {
                        "opponent_checkpoint": str(opp),
                        "games": pool_games,
                        "wins": opp_wins,
                        "losses": opp_losses,
                        "draws": opp_draws,
                        "cached": was_cached,
                    }
                )

            pool_gate = _promotion_gate(
                wins=agg_wins,
                losses=agg_losses,
                draws=agg_draws,
                threshold=pool_threshold,
                confidence=float(config.get("promotion_confidence", 0.95)),
            )
            pool_eval = {
                "pool_size": len(opponents),
                "games_per_opponent": pool_games,
                "aggregate": pool_gate,
                "opponents": per_opponent,
            }

            # Tighten the overall promotion decision with the pool gate.
            summary["promote"] = bool(summary.get("promote")) and bool(pool_gate.get("promote"))
            summary["pool_gate"] = pool_gate
            summary["pool_eval"] = pool_eval
        else:
            pool_eval = {
                "pool_size": 0,
                "games_per_opponent": pool_games,
                "aggregate": None,
                "opponents": [],
                "note": "no_pool_opponents_found",
            }
            summary["pool_eval"] = pool_eval

    print(
        f"Win rate: {summary['win_rate']:.1%} "
        f"(CI_low={summary['win_rate_ci_low']:.1%} "
        f"@ {summary['confidence']:.0%})"
    )

    # Log promotion decision for audit trail and analysis
    _log_promotion_decision(
        config=config,
        iteration=iteration,
        summary=summary,
        model_path=iter_model,
    )

    return True, summary


def _select_promotion_pool_opponents(
    *,
    board: str,
    players: int,
    models_dir: Path,
    iter_model: Path,
    best_model: Optional[Path],
    pool_size: int,
    diverse_sampling: bool = True,
) -> List[Path]:
    """Select a diverse pool of opponent checkpoints for promotion gating.

    When diverse_sampling=True (default), selects opponents from different
    time periods rather than just the newest checkpoints. This prevents
    overfitting to recent model styles and tests generalization.

    Selection priority:
    1. Previous best model (if exists)
    2. Canonical v5 model (baseline anchor)
    3. Diverse historical checkpoints (evenly spaced by time)
    """
    if pool_size <= 0:
        return []

    board = str(board).strip().lower()
    players = int(players)

    board_tokens: list[str]
    if board == "square8":
        board_tokens = ["sq8", "square8"]
    elif board == "square19":
        board_tokens = ["sq19", "square19", "19x19"]
    elif board == "hexagonal":
        board_tokens = ["hex", "hexagonal"]
    else:
        board_tokens = [board]

    def _is_candidate(path: Path) -> bool:
        try:
            if not path.is_file() or path.stat().st_size <= 0:
                return False
        except OSError:
            return False
        name = path.name.lower()
        if name.endswith("_mps.pth"):
            return False
        if name.startswith("ringrift_best_"):
            return False
        if f"{players}p" not in name:
            return False
        return any(token in name for token in board_tokens)

    exclude: set[Path] = {iter_model.resolve()}
    if best_model is not None:
        exclude.add(best_model.resolve())

    pool: list[Path] = []

    # Priority 1: Previous best model
    backup = models_dir / f"{board}_{players}p_prev_best.pth"
    if backup.exists():
        try:
            resolved = backup.resolve()
        except Exception:
            resolved = backup
        if resolved not in exclude and backup.stat().st_size > 0:
            pool.append(backup)

    # Priority 2: Canonical v5 model (baseline anchor)
    v5_candidates = [
        models_dir / f"ringrift_v5_{board}_{players}p.pth",
        models_dir / f"ringrift_v5_sq8_{players}p_2xh100.pth" if board == "square8" else None,
    ]
    for v5_path in v5_candidates:
        if v5_path is None:
            continue
        if v5_path.exists() and len(pool) < pool_size:
            try:
                resolved = v5_path.resolve()
            except Exception:
                resolved = v5_path
            if resolved not in exclude and v5_path.stat().st_size > 0:
                if not any(v5_path.samefile(p) for p in pool):
                    pool.append(v5_path)
                    break

    # Collect all candidates for diverse sampling
    candidates = [p for p in models_dir.glob("*.pth") if _is_candidate(p)]
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)

    # Filter out excluded paths
    filtered_candidates = []
    for path in candidates:
        try:
            resolved = path.resolve()
        except Exception:
            resolved = path
        if resolved in exclude:
            continue
        if any(path.samefile(existing) for existing in pool):
            continue
        filtered_candidates.append(path)

    # Priority 3: Diverse historical checkpoints
    remaining_slots = pool_size - len(pool)
    if remaining_slots > 0 and filtered_candidates:
        if diverse_sampling and len(filtered_candidates) > remaining_slots:
            # Sample evenly across the time range for diversity
            step = len(filtered_candidates) // remaining_slots
            indices = [i * step for i in range(remaining_slots)]
            for idx in indices:
                if idx < len(filtered_candidates):
                    pool.append(filtered_candidates[idx])
        else:
            # Just take the newest if not enough for diverse sampling
            pool.extend(filtered_candidates[:remaining_slots])

    return pool[:pool_size]


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

        if bool(config.get("publish_ringrift_best_alias")):
            alias_path = publish_ringrift_best_alias(
                board=board,
                num_players=int(players),
                best_model_path=best_model,
                promotion_summary=config.get("last_eval_summary"),
            )
            print(f"Published ladder alias: {alias_path.name}")

            maybe_sync_staging(
                enabled=bool(config.get("sync_staging")),
                restart=bool(config.get("sync_staging_restart", True)),
                validate_health=bool(config.get("sync_staging_validate_health", False)),
                fail_on_missing=bool(config.get("sync_staging_fail_on_missing", False)),
                dry_run=dry_run,
                reason=f"promotion_{board}_{players}p",
            )

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


def run_diverse_tournaments(
    board_types: List[str],
    player_counts: List[int],
    games_per_config: int,
    cluster_config: Optional[str] = None,
    use_distributed: bool = True,
    dry_run: bool = False,
    output_base: Optional[str] = None,
) -> Tuple[bool, int, int]:
    """Run diverse tournaments across board/player configurations.

    Returns: (success, total_games, total_samples)
    """
    if not HAS_DIVERSE_TOURNAMENTS:
        print("Diverse tournament module not available - skipping")
        return True, 0, 0

    if dry_run:
        print(f"[DRY-RUN] Would run diverse tournaments:")
        print(f"  Board types: {board_types}")
        print(f"  Player counts: {player_counts}")
        print(f"  Games per config: {games_per_config}")
        return True, 0, 0

    print("\n" + "=" * 60)
    print("Running Diverse Tournament Round (Elo Calibration)")
    print("=" * 60)

    # Default output base
    if output_base is None:
        output_base = str(AI_SERVICE_ROOT / "data" / "tournaments")

    # Build tournament configs
    configs = build_tournament_configs(
        board_types=board_types,
        player_counts=player_counts,
        games_per_config=games_per_config,
        output_base=output_base,
    )

    if not configs:
        print("No tournament configs to run")
        return True, 0, 0

    print(f"Configurations: {len(configs)} total")
    for cfg in configs:
        print(f"  - {cfg.board_type} {cfg.num_players}p x {cfg.num_games} games")

    results: List[TournamentResult] = []

    try:
        if use_distributed:
            # Load cluster hosts
            hosts = load_cluster_hosts(cluster_config)
            if not hosts:
                print("No cluster hosts configured - falling back to local execution")
                results = run_tournament_round_local(configs)
            else:
                # Filter available hosts
                available = asyncio.get_event_loop().run_until_complete(
                    filter_available_hosts(hosts)
                )
                if not available:
                    print("No cluster hosts available - falling back to local execution")
                    results = run_tournament_round_local(configs)
                else:
                    print(f"Cluster hosts available: {len(available)}/{len(hosts)}")
                    results = asyncio.get_event_loop().run_until_complete(
                        run_tournament_round_distributed(configs, available)
                    )
        else:
            results = run_tournament_round_local(configs)
    except Exception as e:
        print(f"Tournament execution failed: {e}")
        return False, 0, 0

    # Summarize results
    total_games = sum(r.games_completed for r in results)
    total_samples = sum(r.samples_generated for r in results)
    success_count = sum(1 for r in results if r.success)

    print(f"\nTournament Summary:")
    print(f"  Successful configs: {success_count}/{len(results)}")
    print(f"  Total games: {total_games}")
    print(f"  Total samples: {total_samples}")

    return success_count == len(results), total_games, total_samples


def run_improvement_iteration(
    iteration: int,
    config: dict,
    state: LoopState,
    dry_run: bool = False,
    pipeline_manager: Optional[OverlappedPipelineManager] = None,
) -> Tuple[bool, float, int]:
    """Run a single iteration of the improvement loop.

    Args:
        iteration: Current iteration number (0-indexed)
        config: Loop configuration dictionary
        state: Current loop state
        dry_run: If True, print commands without executing
        pipeline_manager: Optional overlapped pipeline manager for async selfplay

    Returns: (improved, winrate, games_generated)
    """
    # Step 1: Selfplay - use pipeline manager if async selfplay was started
    print(f"\n--- Step 1: Selfplay ---")
    if pipeline_manager is not None and pipeline_manager.enabled:
        # Get result from async selfplay (or run sync if no task pending)
        success, games, staging_db_path = pipeline_manager.get_selfplay_result(
            config, iteration, dry_run
        )
    else:
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
            parallel_workers=int(config.get("parallel_validation_workers", 1)),
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

    # Thread evaluation summary into config so promotion hooks can attach metadata.
    config["last_eval_summary"] = eval_summary

    # Step 6: Promote if improved
    print(f"\n--- Step 6: Promotion Decision ---")
    threshold = config.get("promotion_threshold", 0.55)
    pool_gate = eval_summary.get("pool_gate") if isinstance(eval_summary, dict) else None
    pool_gate_text = ""
    if isinstance(pool_gate, dict):
        pool_wr = pool_gate.get("win_rate")
        pool_ci_low = pool_gate.get("win_rate_ci_low")
        pool_thr = pool_gate.get("threshold")
        if isinstance(pool_wr, float) and isinstance(pool_ci_low, float) and isinstance(pool_thr, float):
            pool_gate_text = (
                f" | pool win={pool_wr:.1%} (CI_low={pool_ci_low:.1%} >= {pool_thr:.1%})"
            )
    if eval_summary.get("promote", False):
        if promote_model(config, iter_model, dry_run):
            winrate = float(eval_summary.get("win_rate") or 0.0)
            ci_low = eval_summary.get("win_rate_ci_low")
            ci_text = f"{ci_low:.1%}" if isinstance(ci_low, float) else "N/A"
            print(
                "Model promoted! "
                f"Win rate {winrate:.1%} "
                f"(CI_low={ci_text}) >= {threshold:.1%}"
                f"{pool_gate_text}"
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
            f"{pool_gate_text}"
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
        "--promotion-pool-size",
        type=int,
        default=5,
        help=(
            "Number of additional pool opponents to evaluate against before promoting. "
            "Set to 0 to disable the pool gate (default: 5)."
        ),
    )
    parser.add_argument(
        "--promotion-pool-games-per-opponent",
        type=int,
        default=20,
        help="Games per pool opponent when --promotion-pool-size > 0 (default: 20).",
    )
    parser.add_argument(
        "--promotion-pool-threshold",
        type=float,
        default=0.5,
        help=(
            "Win-rate threshold (Wilson CI lower bound) for the aggregated pool gate. "
            "Default: 0.5 (non-regression)."
        ),
    )
    parser.add_argument(
        "--publish-ringrift-best-alias",
        action="store_true",
        help=(
            "After promoting a new best model, also publish ringrift_best_* aliases under ai-service/models/ "
            "so the canonical difficulty ladder can pick up the new checkpoint."
        ),
    )
    parser.add_argument(
        "--sync-staging",
        action="store_true",
        help=(
            "After publishing ringrift_best_* aliases, sync promoted artifacts to staging via "
            "scripts/sync_staging_ai_artifacts.py (requires RINGRIFT_STAGING_* env vars)."
        ),
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
        "--selfplay-engine-mode",
        type=str,
        default="mixed",
        choices=[
            "descent-only",
            "mixed",
            "random-only",
            "heuristic-only",
            "minimax-only",
            "mcts-only",
            "nn-only",
            "best-vs-pool",
        ],
        help=(
            "Self-play engine selection strategy forwarded to scripts/run_self_play_soak.py "
            "(default: mixed). Use 'best-vs-pool' for best-vs-recent-checkpoint "
            "self-play (with seat rotation for 3–4p)."
        ),
    )
    parser.add_argument(
        "--selfplay-nn-pool-size",
        type=int,
        default=0,
        help=(
            "When set, diversify neural tiers by sampling from up to N recent "
            "NN checkpoints (excluding the best alias). "
            "(default: 0)"
        ),
    )
    parser.add_argument(
        "--selfplay-nn-pool-dir",
        type=str,
        default=None,
        help=(
            "Optional directory to scan for NN checkpoints to populate the pool "
            "(defaults to ai-service/models/ when unset)."
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
    # Parallelization options
    parser.add_argument(
        "--parallel-validation-workers",
        type=int,
        default=1,
        help=(
            "Number of parallel workers for config/history validation during ingest. "
            "Set to 8+ to reduce validation time from ~30min to ~5min. "
            "(default: 1 = sequential)"
        ),
    )
    parser.add_argument(
        "--parallel-pool-eval",
        action="store_true",
        help=(
            "Run pool opponent evaluations in parallel using concurrent.futures. "
            "Reduces pool eval time from 30-60min to ~15min."
        ),
    )
    # Overlapped pipeline options (enabled by default for faster iterations)
    parser.add_argument(
        "--overlapped-selfplay",
        action="store_true",
        default=True,
        help=(
            "Enable overlapped pipeline: start selfplay for iteration N+1 while "
            "training/eval for N is running. Reduces total iteration time by ~30%%. (default: enabled)"
        ),
    )
    parser.add_argument(
        "--no-overlapped-selfplay",
        action="store_false",
        dest="overlapped_selfplay",
        help="Disable overlapped pipeline (run stages sequentially).",
    )
    # Adaptive controller options
    parser.add_argument(
        "--adaptive",
        action="store_true",
        help=(
            "Enable adaptive controller for dynamic loop behavior. "
            "Includes plateau detection, dynamic game counts, and early stopping."
        ),
    )
    parser.add_argument(
        "--plateau-threshold",
        type=int,
        default=5,
        help=(
            "Number of iterations without improvement before plateau detection "
            "(default: 5). Only used with --adaptive."
        ),
    )
    parser.add_argument(
        "--adaptive-state-file",
        type=str,
        default=None,
        help=(
            "Path to adaptive controller state file for persistence across runs. "
            "Default: logs/improvement/<board>_<players>p_adaptive.json"
        ),
    )
    # Diverse tournament scheduling
    parser.add_argument(
        "--tournament-every-n-iterations",
        type=int,
        default=0,
        help=(
            "Run diverse tournaments every N iterations (0 = disabled). "
            "Tournaments generate games across all board/player configs for Elo calibration."
        ),
    )
    parser.add_argument(
        "--tournament-on-promotion",
        action="store_true",
        help="Run diverse tournaments after each model promotion.",
    )
    parser.add_argument(
        "--tournament-board-types",
        type=str,
        default="square8",
        help="Comma-separated board types for tournaments (default: square8). Use 'all' for square8,square19,hexagonal.",
    )
    parser.add_argument(
        "--tournament-player-counts",
        type=str,
        default="2",
        help="Comma-separated player counts for tournaments (default: 2). Use 'all' for 2,3,4.",
    )
    parser.add_argument(
        "--tournament-games-per-config",
        type=int,
        default=10,
        help="Games per board/player configuration in tournaments (default: 10).",
    )
    parser.add_argument(
        "--tournament-cluster-config",
        type=str,
        default=None,
        help="Path to cluster config for distributed tournaments (default: config/distributed_hosts.yaml).",
    )
    parser.add_argument(
        "--tournament-local-only",
        action="store_true",
        help="Run tournaments locally only, without distributed execution.",
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
        db_token = "hex" if args.board == "hexagonal" else args.board
        suffix = "" if int(args.players) == 2 else f"_{int(args.players)}p"
        default_name = f"canonical_{db_token}{suffix}.db" if canonical_mode else "selfplay.db"
        replay_db_path = (AI_SERVICE_ROOT / "data" / "games" / default_name).resolve()

    registry_path = _resolve_ai_service_path(args.registry)
    gate_summary_path = (AI_SERVICE_ROOT / f"db_health.{replay_db_path.stem}.json").resolve()
    staging_db_dir = _resolve_ai_service_path(args.staging_db_dir)
    ingest_scan_dirs = [str(staging_db_dir)] + [str(x) for x in (args.ingest_scan_dir or [])]
    db_label = ("hex" if args.board == "hexagonal" else args.board)
    holdout_db = args.holdout_db or f"data/games/holdouts/holdout_{db_label}_{int(args.players)}p.db"
    quarantine_db = args.quarantine_db or f"data/games/quarantine/quarantine_{db_label}_{int(args.players)}p.db"

    # Configure
    config = {
        "board": args.board,
        "players": args.players,
        "games_per_iter": args.games_per_iter,
        "max_moves": int(args.max_moves),
        "promotion_threshold": args.promotion_threshold,
        "promotion_confidence": args.promotion_confidence,
        "promotion_pool_size": int(args.promotion_pool_size),
        "promotion_pool_games_per_opponent": int(args.promotion_pool_games_per_opponent),
        "promotion_pool_threshold": float(args.promotion_pool_threshold),
        "publish_ringrift_best_alias": bool(args.publish_ringrift_best_alias),
        "sync_staging": bool(args.sync_staging),
        "sync_staging_restart": not bool(args.sync_staging_no_restart),
        "sync_staging_validate_health": bool(args.sync_staging_validate_health)
        or bool(args.sync_staging_fail_on_missing),
        "sync_staging_fail_on_missing": bool(args.sync_staging_fail_on_missing),
        "eval_games": args.eval_games,
        "selfplay_difficulty_band": args.selfplay_difficulty_band,
        "selfplay_engine_mode": args.selfplay_engine_mode,
        "selfplay_nn_pool_size": int(args.selfplay_nn_pool_size),
        "selfplay_nn_pool_dir": args.selfplay_nn_pool_dir,
        "dataset_policy_target": args.dataset_policy_target,
        "dataset_max_games": args.dataset_max_games,
        "legacy_maxn_encoding": bool(args.legacy_maxn_encoding),
        "policy_search_think_time_ms": args.policy_search_think_time_ms,
        "policy_temperature": args.policy_temperature,
        "policy_nn_model_id": args.policy_nn_model_id,
        "replay_db": str(replay_db_path),
        "staging_db_dir": str(staging_db_dir),
        "ingest_scan_dirs": ingest_scan_dirs,
        "holdout_db": holdout_db if canonical_mode else None,
        "quarantine_db": quarantine_db if canonical_mode else None,
        "canonical_mode": canonical_mode,
        "allow_pending_gate": bool(args.allow_pending_gate),
        "registry_path": str(registry_path),
        "gate_summary": str(gate_summary_path),
        "parallel_validation_workers": int(args.parallel_validation_workers),
        "parallel_pool_eval": bool(args.parallel_pool_eval),
        "overlapped_selfplay": bool(args.overlapped_selfplay),
        "adaptive": bool(args.adaptive),
        "plateau_threshold": int(args.plateau_threshold),
    }

    # Tournament configuration
    tournament_board_types = (
        ["square8", "square19", "hexagonal"]
        if args.tournament_board_types == "all"
        else [b.strip() for b in args.tournament_board_types.split(",") if b.strip()]
    )
    tournament_player_counts = (
        [2, 3, 4]
        if args.tournament_player_counts == "all"
        else [int(p.strip()) for p in args.tournament_player_counts.split(",") if p.strip()]
    )
    tournament_config = {
        "every_n_iterations": args.tournament_every_n_iterations,
        "on_promotion": args.tournament_on_promotion,
        "board_types": tournament_board_types,
        "player_counts": tournament_player_counts,
        "games_per_config": args.tournament_games_per_config,
        "cluster_config": args.tournament_cluster_config,
        "local_only": args.tournament_local_only,
    }
    tournaments_enabled = (
        args.tournament_every_n_iterations > 0 or args.tournament_on_promotion
    )

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
    print(f"Self-play engine mode: {args.selfplay_engine_mode}")
    if int(args.selfplay_nn_pool_size or 0) > 0:
        pool_dir = args.selfplay_nn_pool_dir or "(default)"
        print(f"Self-play NN pool: size={int(args.selfplay_nn_pool_size)} dir={pool_dir}")
    print(f"Eval games per iteration: {args.eval_games}")
    print(f"Promotion threshold: {args.promotion_threshold:.0%}")
    print(f"Promotion confidence: {args.promotion_confidence:.0%}")
    if int(args.promotion_pool_size or 0) > 0:
        print(
            "Promotion pool gate: "
            f"size={int(args.promotion_pool_size)} "
            f"games_per_opp={int(args.promotion_pool_games_per_opponent)} "
            f"threshold={float(args.promotion_pool_threshold):.0%}"
        )
    else:
        print("Promotion pool gate: disabled")
    if args.publish_ringrift_best_alias:
        print("Publish ringrift_best aliases: enabled")
        if args.sync_staging:
            restart_text = "no-restart" if args.sync_staging_no_restart else "restart"
            validate_text = (
                "validate"
                if args.sync_staging_validate_health or args.sync_staging_fail_on_missing
                else "no-validate"
            )
            missing_text = "fail-on-missing" if args.sync_staging_fail_on_missing else "no-fail-on-missing"
            print(f"Staging sync: enabled ({restart_text}, {validate_text}, {missing_text})")
    print(f"Dataset policy target: {args.dataset_policy_target}")
    if args.dataset_max_games is not None:
        print(f"Dataset max games: {args.dataset_max_games}")
    if args.dataset_policy_target != "played":
        print(f"Reanalysis think time: {args.policy_search_think_time_ms}ms")
        print(f"Reanalysis temperature: {args.policy_temperature}")
        if args.policy_nn_model_id:
            print(f"Reanalysis nn_model_id: {args.policy_nn_model_id}")
    print(f"State file: {state_path}")
    if tournaments_enabled:
        triggers = []
        if args.tournament_every_n_iterations > 0:
            triggers.append(f"every {args.tournament_every_n_iterations} iterations")
        if args.tournament_on_promotion:
            triggers.append("on promotion")
        mode = "local" if args.tournament_local_only else "distributed"
        print(f"Diverse tournaments: {', '.join(triggers)} ({mode})")
        print(f"  Boards: {tournament_board_types}, Players: {tournament_player_counts}")
        print(f"  Games per config: {args.tournament_games_per_config}")
    if args.dry_run:
        print("*** DRY RUN MODE - No commands will be executed ***")
    if args.overlapped_selfplay:
        print("Overlapped selfplay: enabled (30% faster iterations)")
    if args.adaptive:
        print(f"Adaptive controller: enabled (plateau threshold: {args.plateau_threshold})")
    print("=" * 60)

    # Initialize overlapped pipeline manager
    pipeline_manager = OverlappedPipelineManager(enabled=bool(args.overlapped_selfplay))

    # Initialize adaptive controller
    adaptive_controller = None
    if args.adaptive and HAS_ADAPTIVE_CONTROLLER:
        adaptive_state_file = args.adaptive_state_file
        if adaptive_state_file is None:
            adaptive_state_file = AI_SERVICE_ROOT / "logs" / "improvement" / f"{args.board}_{args.players}p_adaptive.json"
        else:
            adaptive_state_file = Path(adaptive_state_file)

        adaptive_controller = create_adaptive_controller(
            plateau_threshold=args.plateau_threshold,
            min_games=50,
            max_games=200,
            state_path=adaptive_state_file,
        )
        print(f"Adaptive state file: {adaptive_state_file}")
    elif args.adaptive:
        print("Warning: --adaptive requested but adaptive_controller module not available")

    for i in range(start_iter, args.iterations):
        print(f"\n{'='*60}")
        print(f"=== Improvement Iteration {i+1}/{args.iterations} ===")
        print(f"{'='*60}")

        # Check adaptive controller for plateau detection
        if adaptive_controller is not None:
            if not adaptive_controller.should_continue():
                print("\n[adaptive] Plateau detected - stopping loop")
                print(f"[adaptive] {adaptive_controller.get_statistics()}")
                break

            # Get dynamic game count from adaptive controller
            dynamic_games = adaptive_controller.compute_games()
            if dynamic_games != config["games_per_iter"]:
                print(f"[adaptive] Adjusting games_per_iter: {config['games_per_iter']} -> {dynamic_games}")
                config["games_per_iter"] = dynamic_games

        try:
            # Use pipeline manager for overlapped selfplay if enabled
            improved, winrate, games = run_improvement_iteration(
                i, config, state, args.dry_run,
                pipeline_manager=pipeline_manager,
            )

            # Start selfplay for next iteration in background (if overlapped enabled)
            if args.overlapped_selfplay and i + 1 < args.iterations:
                print("[overlapped] Starting selfplay for next iteration in background...")
                pipeline_manager.start_selfplay_async(config, i + 1, args.dry_run)

            # Update state
            state.iteration = i + 1
            state.total_games_generated += games
            state.history.append({
                "iteration": i + 1,
                "improved": improved,
                "winrate": winrate,
                "games": games,
            })

            # Update adaptive controller
            if adaptive_controller is not None:
                adaptive_controller.record_iteration(
                    iteration=i + 1,
                    win_rate=winrate,
                    promoted=improved,
                    games_played=games,
                    eval_games=int(config.get("eval_games", 100)),
                )
                # Save adaptive state
                if args.adaptive_state_file:
                    adaptive_controller.save(Path(args.adaptive_state_file))
                else:
                    adaptive_controller.save(
                        AI_SERVICE_ROOT / "logs" / "improvement" / f"{args.board}_{args.players}p_adaptive.json"
                    )

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

            # Run diverse tournaments if configured
            run_tournament_now = False
            if tournaments_enabled:
                # Trigger on promotion
                if improved and tournament_config["on_promotion"]:
                    print("\nTriggering tournament: model was promoted")
                    run_tournament_now = True
                # Trigger every N iterations (1-indexed, so iteration i+1)
                elif (
                    tournament_config["every_n_iterations"] > 0
                    and (i + 1) % tournament_config["every_n_iterations"] == 0
                ):
                    print(f"\nTriggering tournament: iteration {i+1} (every {tournament_config['every_n_iterations']})")
                    run_tournament_now = True

            if run_tournament_now:
                try:
                    t_success, t_games, t_samples = run_diverse_tournaments(
                        board_types=tournament_config["board_types"],
                        player_counts=tournament_config["player_counts"],
                        games_per_config=tournament_config["games_per_config"],
                        cluster_config=tournament_config["cluster_config"],
                        use_distributed=not tournament_config["local_only"],
                        dry_run=args.dry_run,
                    )
                    if t_success:
                        print(f"Tournament completed: {t_games} games, {t_samples} samples")
                    else:
                        print("Tournament completed with some failures")
                except Exception as te:
                    print(f"Tournament error (non-fatal): {te}")

            save_state(state, state_path)

        except KeyboardInterrupt:
            print("\n\nInterrupted by user. Saving state...")
            pipeline_manager.cancel_pending()
            save_state(state, state_path)
            if adaptive_controller is not None:
                adaptive_controller.save(
                    Path(args.adaptive_state_file) if args.adaptive_state_file
                    else AI_SERVICE_ROOT / "logs" / "improvement" / f"{args.board}_{args.players}p_adaptive.json"
                )
            print(f"State saved to {state_path}. Resume with --resume")
            sys.exit(0)

        except Exception as e:
            print(f"\nERROR in iteration {i+1}: {e}")
            state.consecutive_failures += 1
            save_state(state, state_path)

        # Brief pause between iterations
        if not args.dry_run:
            time.sleep(5)

    # Cleanup
    pipeline_manager.cancel_pending()

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

    # Print adaptive controller stats if enabled
    if adaptive_controller is not None:
        stats = adaptive_controller.get_statistics()
        print(f"\nAdaptive Controller Statistics:")
        print(f"  Plateau count: {stats.get('plateau_count', 0)}")
        print(f"  Promotion rate: {stats.get('promotion_rate', 0):.1%}")
        print(f"  Trend: {stats.get('trend', 'unknown')}")


if __name__ == "__main__":
    main()
