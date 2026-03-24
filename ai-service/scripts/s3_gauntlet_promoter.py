#!/usr/bin/env python3
"""Bridge between gauntlet_runner (Lambda/S3) and promotion (coordinator).

Polls S3 for gauntlet evaluation results written by gauntlet_runner.py on
Lambda GPU nodes. When a candidate meets promotion thresholds, pulls the
candidate model from S3 and promotes it to canonical.

Mar 23, 2026: Created to close the critical gap where gauntlet_runner
evaluates candidates and writes JSON to S3, but nothing on the coordinator
reads those results or triggers promotion.

Usage:
    # One-shot (check and promote once)
    python scripts/s3_gauntlet_promoter.py

    # Daemon mode (poll every 10 min)
    python scripts/s3_gauntlet_promoter.py --daemon

    # Dry run (show what would be promoted)
    python scripts/s3_gauntlet_promoter.py --dry-run
"""
import argparse
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
os.chdir(str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("s3_gauntlet_promoter")

S3_BUCKET = os.environ.get("RINGRIFT_S3_BUCKET", "ringrift-models-20251214")
RESULTS_PREFIX = "consolidated/gauntlet_results"
MODELS_PREFIX = "consolidated/models"
MODELS_DIR = Path("models")
POLL_INTERVAL = 600  # 10 min

# Promotion thresholds (must match or be stricter than gauntlet_runner)
MIN_WIN_RATE_VS_RANDOM = 0.70  # Must beat random 70%+
MIN_TOTAL_GAMES = 30  # At least 30 games played
MIN_ESTIMATED_ELO = 1300  # Above random baseline (~1200)

# Track what we've already processed
promoted: dict[str, float] = {}  # config_key -> timestamp of last promotion


def fetch_gauntlet_results() -> dict[str, dict]:
    """Fetch all gauntlet result JSONs from S3."""
    results = {}
    try:
        proc = subprocess.run(
            ["aws", "s3", "ls", f"s3://{S3_BUCKET}/{RESULTS_PREFIX}/"],
            capture_output=True, text=True, timeout=30,
        )
        if proc.returncode != 0:
            return results

        for line in proc.stdout.strip().splitlines():
            parts = line.split()
            if len(parts) < 4 or not parts[3].endswith(".json"):
                continue
            filename = parts[3]
            config_key = filename.replace("gauntlet_", "").replace(".json", "")

            # Download the JSON
            tmp_path = Path(f"/tmp/gauntlet_{config_key}.json")
            dl = subprocess.run(
                ["aws", "s3", "cp",
                 f"s3://{S3_BUCKET}/{RESULTS_PREFIX}/{filename}",
                 str(tmp_path), "--quiet"],
                capture_output=True, timeout=30,
            )
            if dl.returncode == 0 and tmp_path.exists():
                try:
                    with open(tmp_path) as f:
                        results[config_key] = json.load(f)
                except (json.JSONDecodeError, OSError):
                    pass
    except Exception as e:
        logger.warning(f"Failed to fetch gauntlet results: {e}")
    return results


def pull_candidate_from_s3(config_key: str) -> Path | None:
    """Pull candidate model from S3 to local models dir."""
    s3_path = f"s3://{S3_BUCKET}/{MODELS_PREFIX}/candidate_{config_key}.pth"
    local_path = MODELS_DIR / f"candidate_{config_key}.pth"

    try:
        proc = subprocess.run(
            ["aws", "s3", "cp", s3_path, str(local_path), "--quiet"],
            capture_output=True, timeout=300,
        )
        if proc.returncode == 0 and local_path.exists():
            logger.info(f"Pulled candidate_{config_key}.pth from S3 "
                        f"({local_path.stat().st_size / 1e6:.1f}MB)")
            return local_path
    except Exception as e:
        logger.warning(f"Failed to pull candidate for {config_key}: {e}")
    return None


def should_promote(config_key: str, result: dict) -> tuple[bool, str]:
    """Check if gauntlet result meets promotion thresholds."""
    win_rate = result.get("win_rate", 0)
    total_games = result.get("total_games", 0)
    elo = result.get("estimated_elo", 0)
    timestamp = result.get("timestamp", 0)

    # Skip if already promoted this version
    if promoted.get(config_key) == timestamp:
        return False, "already processed"

    # Check vs random specifically
    opp_results = result.get("opponent_results", {})
    random_wr = 0
    for key in ("random", "RANDOM", "RandomAI"):
        if key in opp_results:
            random_wr = opp_results[key].get("win_rate", 0)
            break

    if total_games < MIN_TOTAL_GAMES:
        return False, f"too few games ({total_games} < {MIN_TOTAL_GAMES})"

    if random_wr < MIN_WIN_RATE_VS_RANDOM:
        return False, f"weak vs random ({random_wr:.0%} < {MIN_WIN_RATE_VS_RANDOM:.0%})"

    if elo < MIN_ESTIMATED_ELO:
        return False, f"Elo too low ({elo:.0f} < {MIN_ESTIMATED_ELO})"

    return True, f"Elo={elo:.0f} wr={win_rate:.0%} random={random_wr:.0%} games={total_games}"


def execute_promotion(config_key: str, result: dict, dry_run: bool = False) -> bool:
    """Promote candidate to canonical using PromotionController."""
    if dry_run:
        logger.info(f"[DRY RUN] Would promote {config_key}")
        return True

    # Pull candidate from S3
    candidate_path = pull_candidate_from_s3(config_key)
    if not candidate_path:
        logger.error(f"Cannot promote {config_key}: candidate not available")
        return False

    # Verify model integrity
    try:
        from app.utils.torch_utils import validate_checkpoint_magic_bytes
        is_valid, fmt = validate_checkpoint_magic_bytes(str(candidate_path))
        if not is_valid:
            logger.error(f"Candidate {config_key} has invalid format: {fmt}")
            return False
    except Exception:
        pass

    # Use PromotionController for atomic promotion with checksum
    try:
        from app.training.promotion_controller import (
            PromotionController, PromotionDecision, PromotionType,
        )
        controller = PromotionController()
        decision = PromotionDecision(
            model_id=f"candidate_{config_key}",
            promotion_type=PromotionType.ELO_IMPROVEMENT,
            should_promote=True,
            reason=f"S3 gauntlet: Elo={result.get('estimated_elo', 0):.0f}",
            current_elo=result.get("estimated_elo"),
            win_rate=result.get("win_rate"),
            games_played=result.get("total_games", 0),
            model_path=str(candidate_path),
        )
        success = controller._execute_elo_promotion(decision)
        if success:
            logger.info(f"PROMOTED {config_key} from S3 gauntlet result")
            # Push promoted canonical + checksum back to S3
            canonical_path = MODELS_DIR / f"canonical_{config_key}.pth"
            if canonical_path.exists():
                subprocess.run(
                    ["aws", "s3", "cp", str(canonical_path),
                     f"s3://{S3_BUCKET}/{MODELS_PREFIX}/canonical_{config_key}.pth",
                     "--quiet"],
                    timeout=300,
                )
                sha_path = Path(f"{canonical_path}.sha256")
                if sha_path.exists():
                    subprocess.run(
                        ["aws", "s3", "cp", str(sha_path),
                         f"s3://{S3_BUCKET}/{MODELS_PREFIX}/canonical_{config_key}.pth.sha256",
                         "--quiet"],
                        timeout=30,
                    )
        return success
    except Exception as e:
        logger.error(f"Promotion failed for {config_key}: {e}")
        return False


def run_once(dry_run: bool = False) -> int:
    """Check all gauntlet results and promote eligible candidates."""
    results = fetch_gauntlet_results()
    if not results:
        logger.info("No gauntlet results found on S3")
        return 0

    promotions = 0
    for config_key, result in sorted(results.items()):
        eligible, reason = should_promote(config_key, result)
        if eligible:
            logger.info(f"ELIGIBLE: {config_key} — {reason}")
            if execute_promotion(config_key, result, dry_run=dry_run):
                promoted[config_key] = result.get("timestamp", 0)
                promotions += 1
        else:
            logger.debug(f"SKIP: {config_key} — {reason}")

    logger.info(f"Processed {len(results)} results, {promotions} promotions")
    return promotions


def main():
    parser = argparse.ArgumentParser(description="S3 gauntlet results → promotion")
    parser.add_argument("--daemon", action="store_true", help="Run as polling daemon")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be promoted")
    parser.add_argument("--interval", type=int, default=POLL_INTERVAL, help="Poll interval (s)")
    args = parser.parse_args()

    logger.info("S3 gauntlet promoter starting...")
    MODELS_DIR.mkdir(exist_ok=True)

    if args.daemon:
        while True:
            try:
                run_once(dry_run=args.dry_run)
            except Exception as e:
                logger.error(f"Poll cycle failed: {e}")
            logger.info(f"Sleeping {args.interval}s...")
            time.sleep(args.interval)
    else:
        run_once(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
