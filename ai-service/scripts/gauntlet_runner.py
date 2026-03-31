#!/usr/bin/env python3
"""Dedicated gauntlet evaluation runner for Lambda GPU nodes.

Runs as a separate systemd service from P2P, so it survives P2P restarts.
Polls S3 for candidate models and evaluates them against strong baselines.
Results are written to a local JSON file and pushed to S3 for coordinator pickup.

Mar 20, 2026: Created because MCTS-based gauntlets take 30-60 min and
were always killed by P2P restarts (KillMode=control-group).
"""
import json
import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
os.chdir(str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/gauntlet_runner.log"),
    ],
)
logger = logging.getLogger("gauntlet_runner")

# Config
POLL_INTERVAL = 600  # Check for new candidates every 10 min
S3_BUCKET = os.environ.get("RINGRIFT_S3_BUCKET", "ringrift-models-20251214")
S3_PREFIX = "consolidated/models"
RESULTS_PREFIX = "consolidated/gauntlet_results"
MODELS_DIR = Path("models")
RESULTS_DIR = Path("data/gauntlet_results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
STATUS_FILE = Path("/tmp/gauntlet_status.json")

# Mar 30, 2026: Expanded to all 12 configs. Previous narrowing to hex8_2p+square8_2p
# left 10 configs with no gauntlet evaluation, blocking promotions cluster-wide.
CONFIGS = [
    ("hex8", 2), ("hex8", 3), ("hex8", 4),
    ("square8", 2), ("square8", 3), ("square8", 4),
    ("square19", 2), ("square19", 3), ("square19", 4),
    ("hexagonal", 2), ("hexagonal", 3), ("hexagonal", 4),
]

# Track what we've already evaluated (by SHA256 checksum)
evaluated: dict[str, str] = {}
running = True


def signal_handler(signum, _frame):
    global running
    logger.info(f"Signal {signum} received, shutting down gracefully...")
    running = False


signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)


def _is_missing_object_error(stderr_text: str) -> bool:
    """Best-effort check for missing S3 object responses."""
    lowered = stderr_text.lower()
    return (
        "nosuchkey" in lowered
        or "not found" in lowered
        or "404" in lowered
        or "does not exist" in lowered
    )


def pull_candidate(config_key: str) -> Path | None:
    """Pull candidate from S3 if newer than local."""
    s3_path = f"s3://{S3_BUCKET}/{S3_PREFIX}/candidate_{config_key}.pth"
    local_path = MODELS_DIR / f"candidate_{config_key}.pth"
    sidecar = local_path.with_suffix(local_path.suffix + ".sha256")

    def cleanup_local_candidate() -> None:
        local_path.unlink(missing_ok=True)
        sidecar.unlink(missing_ok=True)

    try:
        result = subprocess.run(
            ["aws", "s3", "ls", s3_path],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode != 0:
            err_text = (result.stderr or "").strip()
            if err_text and not _is_missing_object_error(err_text):
                logger.warning(
                    f"Failed listing candidate in S3 for {config_key}: "
                    f"exit={result.returncode}, error={err_text}"
                )
            return None
        parts = result.stdout.strip().split()
        if len(parts) < 3:
            logger.warning(
                f"Unexpected `aws s3 ls` output for {config_key}: {result.stdout.strip()!r}"
            )
            return None
        try:
            s3_size = int(parts[2])
        except ValueError:
            logger.warning(
                f"Could not parse candidate size from `aws s3 ls` for {config_key}: "
                f"{result.stdout.strip()!r}"
            )
            return None

        # Try to fetch SHA256 sidecar from S3 for precise change detection
        s3_sha = None
        sha_result = subprocess.run(
            ["aws", "s3", "cp", f"{s3_path}.sha256", "/dev/stdout", "--quiet"],
            capture_output=True, text=True, timeout=15,
        )
        if sha_result.returncode == 0 and sha_result.stdout.strip():
            s3_sha = sha_result.stdout.strip().split()[0]
        elif sha_result.returncode != 0:
            sha_err = (sha_result.stderr or "").strip()
            if sha_err and not _is_missing_object_error(sha_err):
                logger.warning(
                    f"Failed fetching SHA sidecar for {config_key}: "
                    f"exit={sha_result.returncode}, error={sha_err}"
                )

        # Skip if already evaluated this exact version (SHA256 or size fallback)
        check_val = s3_sha or str(s3_size)
        if evaluated.get(config_key) == check_val:
            return None

        # Delete stale sidecar before downloading new candidate
        # (stale sidecar from previous version causes false checksum mismatch)
        cleanup_local_candidate()

        # Pull from S3
        cp_result = subprocess.run(
            ["aws", "s3", "cp", s3_path, str(local_path), "--quiet"],
            timeout=120, capture_output=True, text=True,
        )
        if cp_result.returncode != 0:
            err_text = (cp_result.stderr or cp_result.stdout or "").strip()
            logger.warning(
                f"Failed downloading candidate for {config_key}: "
                f"exit={cp_result.returncode}, error={err_text}"
            )
            cleanup_local_candidate()
            return None
        if not local_path.exists() or local_path.stat().st_size <= 0:
            logger.warning(f"Downloaded candidate missing or empty for {config_key}: {local_path}")
            cleanup_local_candidate()
            return None
        logger.info(f"Pulled {config_key} from S3 ({s3_size} bytes)")

        # Verify model integrity after S3 download
        try:
            from app.utils.torch_utils import validate_checkpoint_magic_bytes
            is_valid, format_type = validate_checkpoint_magic_bytes(str(local_path))
            if not is_valid:
                logger.error(f"Invalid checkpoint for {config_key}: {format_type}")
                cleanup_local_candidate()
                return None
        except Exception as e:
            logger.warning(f"Could not verify {config_key} integrity: {e}")

        return local_path
    except Exception as e:
        logger.warning(f"Failed to pull {config_key}: {e}")
        cleanup_local_candidate()
        return None


def _write_status(config_key: str, status: str, **extra):
    """Write machine-readable status to /tmp/gauntlet_status.json."""
    try:
        data = {"config": config_key, "status": status, "timestamp": time.time()}
        data.update(extra)
        STATUS_FILE.write_text(json.dumps(data, indent=2))
    except Exception:
        pass


def run_gauntlet(board_type: str, num_players: int, model_path: Path) -> dict | None:
    """Run gauntlet evaluation with strong baselines."""
    config_key = f"{board_type}_{num_players}p"
    logger.info(f"Starting gauntlet: {config_key} ({model_path})")
    _write_status(config_key, "starting", model=str(model_path))

    try:
        # Model Identity Contract: verify checkpoint matches expected config
        try:
            from app.utils.torch_utils import safe_load_checkpoint
            safe_load_checkpoint(
                model_path,
                expected_board_type=board_type,
                expected_num_players=num_players,
            )
        except ValueError as e:
            logger.error(f"[Identity] {config_key}: {e}")
            return None
        except Exception:
            pass  # Other load errors handled by gauntlet itself

        from app.training.game_gauntlet import run_baseline_gauntlet, BaselineOpponent

        import torch
        has_cuda = torch.cuda.is_available()

        if has_cuda:
            opponents = [
                BaselineOpponent.RANDOM,
                BaselineOpponent.HEURISTIC,
                BaselineOpponent.HEURISTIC_STRONG,
                BaselineOpponent.MCTS_MEDIUM,
                BaselineOpponent.MCTS_STRONG,
            ]
        else:
            opponents = [
                BaselineOpponent.RANDOM,
                BaselineOpponent.HEURISTIC,
                BaselineOpponent.HEURISTIC_STRONG,
            ]

        t0 = time.time()
        result = run_baseline_gauntlet(
            model_path=str(model_path),
            board_type=board_type,
            num_players=num_players,
            games_per_opponent=10,
            opponents=opponents,
            verbose=True,
            use_search=has_cuda,
            harness_type="gumbel_mcts" if has_cuda else "policy_only",
            early_stopping=False,
            parallel_games=2,
        )
        elapsed = time.time() - t0

        elo = getattr(result, "estimated_elo", 0)
        passed = getattr(result, "passed", False)
        total = getattr(result, "total_games", 0)
        win_rate = getattr(result, "win_rate", 0)

        logger.info(
            f"Gauntlet complete: {config_key} Elo={elo:.0f} "
            f"win_rate={win_rate:.1%} games={total} time={elapsed:.0f}s"
        )
        _write_status(config_key, "complete",
                      elo=elo, win_rate=win_rate, games=total, elapsed=elapsed)

        opp_results = {}
        for name, stats in getattr(result, "opponent_results", {}).items():
            if isinstance(stats, dict):
                opp_results[str(name)] = {
                    "win_rate": stats.get("win_rate", 0),
                    "games": stats.get("games", 0),
                }

        return {
            "config_key": config_key,
            "board_type": board_type,
            "num_players": num_players,
            "model_path": str(model_path),
            "estimated_elo": elo,
            "win_rate": win_rate,
            "passed": passed,
            "total_games": total,
            "opponent_results": opp_results,
            "elapsed_seconds": elapsed,
            "timestamp": time.time(),
            "node_id": os.environ.get("NODE_ID", "unknown"),
        }
    except Exception as e:
        logger.error(f"Gauntlet failed for {config_key}: {e}", exc_info=True)
        return None


def push_result(config_key: str, result: dict):
    """Save result locally and push to S3."""
    result_file = RESULTS_DIR / f"gauntlet_{config_key}.json"
    with open(result_file, "w") as f:
        json.dump(result, f, indent=2)

    s3_path = f"s3://{S3_BUCKET}/{RESULTS_PREFIX}/gauntlet_{config_key}.json"
    try:
        cp_result = subprocess.run(
            ["aws", "s3", "cp", str(result_file), s3_path, "--quiet"],
            timeout=30, capture_output=True, text=True,
        )
        if cp_result.returncode != 0:
            err_text = (cp_result.stderr or cp_result.stdout or "").strip()
            logger.warning(
                f"Failed to push gauntlet result for {config_key}: "
                f"exit={cp_result.returncode}, error={err_text}"
            )
            return
        logger.info(f"Pushed result to S3: {s3_path}")
    except Exception as e:
        logger.warning(f"Failed to push result to S3: {e}")


def main():
    logger.info("Gauntlet runner starting...")
    Path("logs").mkdir(exist_ok=True)

    while running:
        for board_type, num_players in CONFIGS:
            if not running:
                break

            config_key = f"{board_type}_{num_players}p"
            model_path = pull_candidate(config_key)

            if model_path is None:
                continue

            result = run_gauntlet(board_type, num_players, model_path)
            if result:
                push_result(config_key, result)
                # Track by SHA256 for precise change detection
                try:
                    from app.utils.torch_utils import compute_model_checksum
                    evaluated[config_key] = compute_model_checksum(str(model_path))
                except Exception:
                    evaluated[config_key] = str(model_path.stat().st_size)

        if running:
            _write_status("", "polling", next_poll_in=POLL_INTERVAL,
                          evaluated=list(evaluated.keys()))
            logger.info(f"Sleeping {POLL_INTERVAL}s before next poll...")
            for _ in range(POLL_INTERVAL):
                if not running:
                    break
                time.sleep(1)

    logger.info("Gauntlet runner stopped.")


if __name__ == "__main__":
    main()
