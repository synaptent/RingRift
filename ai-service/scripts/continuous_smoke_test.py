#!/usr/bin/env python3
"""Continuous smoke test for Lambda node health checks.

Lightweight script designed to run every hour via cron on GPU nodes.
Verifies the canonical model, encoding contract, and inference pipeline
are all healthy for the node's current configuration.

Exit codes:
    0 = healthy
    1 = failure (check stderr for details)

Usage:
    cd ~/ringrift/ai-service && PYTHONPATH=.
    python scripts/continuous_smoke_test.py --board-type hex8 --num-players 2

Cron example (every hour):
    0 * * * * cd ~/ringrift/ai-service && PYTHONPATH=. python3 scripts/continuous_smoke_test.py --board-type hex8 --num-players 2 >> /tmp/ringrift_smoke.log 2>&1
"""
from __future__ import annotations

import argparse
import io
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Resolve project root so imports work regardless of cwd
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Disable torch compile to avoid slow JIT warmup
if not os.environ.get("RINGRIFT_DISABLE_TORCH_COMPILE"):
    os.environ["RINGRIFT_DISABLE_TORCH_COMPILE"] = "1"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [SMOKE] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("smoke_test")


def run_smoke_test(board_type_str: str, num_players: int) -> bool:
    """Run all smoke test checks. Returns True if healthy."""
    from app.models import AIConfig, BoardType
    from app.training.board_encoding_contract import get_expected_channels

    # --- Resolve board type ---
    try:
        board_type = BoardType(board_type_str)
    except ValueError:
        logger.error("Unknown board type: %s (valid: %s)", board_type_str,
                      ", ".join(bt.value for bt in BoardType))
        return False

    config_key = f"{board_type.value}_{num_players}p"
    canonical_path = PROJECT_ROOT / "models" / f"canonical_{board_type.value}_{num_players}p.pth"

    logger.info("Config: %s, model: %s", config_key, canonical_path.name)

    # --- Check 1: Encoding contract ---
    try:
        expected_channels = get_expected_channels(board_type, "v2")
        logger.info("PASS encoding contract: %d channels for %s v2", expected_channels, board_type.value)
    except Exception as e:
        logger.error("FAIL encoding contract: %s", e)
        return False

    # --- Check 2: Canonical model exists ---
    if not canonical_path.exists():
        logger.error("FAIL canonical model not found: %s", canonical_path)
        return False

    # --- Check 3: Model file freshness ---
    model_mtime = canonical_path.stat().st_mtime
    model_age_hours = (time.time() - model_mtime) / 3600
    process_start = time.time()
    if model_age_hours > 168:  # 7 days
        logger.warning(
            "WARNING model is %.0f hours old (%.1f days) -- may be stale after deploy",
            model_age_hours,
            model_age_hours / 24,
        )
    else:
        logger.info("PASS model age: %.1f hours", model_age_hours)

    # --- Check 4: Load into GumbelMCTSAI without fallback warnings ---
    import warnings
    captured_warnings: list[str] = []
    original_warn = warnings.warn

    def capturing_warn(message, *args, **kwargs):
        captured_warnings.append(str(message))
        original_warn(message, *args, **kwargs)

    # Capture WARNING-level log messages
    log_capture = io.StringIO()
    handler = logging.StreamHandler(log_capture)
    handler.setLevel(logging.WARNING)
    root_logger = logging.getLogger()
    root_logger.addHandler(handler)

    try:
        warnings.warn = capturing_warn  # type: ignore[assignment]

        from app.ai.gumbel_mcts_ai import GumbelMCTSAI

        cfg = AIConfig(
            difficulty=9,
            randomness=0.0,
            use_neural_net=True,
            gumbel_simulation_budget=32,
            nn_model_id=str(canonical_path),
            nn_model_version="v2",
            allow_fresh_weights=False,
            use_gpu_tree=False,  # CPU-only for smoke test
        )
        ai = GumbelMCTSAI(1, cfg, board_type)
    except Exception as e:
        logger.error("FAIL GumbelMCTSAI construction: %s", e)
        return False
    finally:
        warnings.warn = original_warn  # type: ignore[assignment]
        root_logger.removeHandler(handler)

    # Check for fallback warnings
    log_output = log_capture.getvalue()
    fallback_indicators = ["fallback", "fresh_weights", "failed to load", "heuristic", "random play"]
    for indicator in fallback_indicators:
        for w in captured_warnings:
            if indicator.lower() in w.lower():
                logger.error("FAIL model loading triggered fallback warning: %s", w)
                return False
        if indicator.lower() in log_output.lower():
            logger.error("FAIL model loading produced fallback log containing: %r", indicator)
            return False

    if ai.neural_net is None:
        logger.error("FAIL GumbelMCTSAI.neural_net is None -- model did not load")
        return False

    logger.info("PASS model loaded into GumbelMCTSAI without fallback")

    # --- Check 5: Run 3 moves of inference ---
    try:
        import random as _random

        from app.models import GameStatus
        from app.training.env import TrainingEnvConfig, get_theoretical_max_moves, make_env

        tmax = get_theoretical_max_moves(board_type, num_players)
        env = make_env(
            TrainingEnvConfig(
                board_type=board_type,
                num_players=num_players,
                max_moves=int(tmax * 1.5),
            )
        )
        state = env.reset(seed=42)
        rng = _random.Random(42)

        # Cache AIs per player to avoid re-loading the model each time
        ai_cache: dict[int, GumbelMCTSAI] = {1: ai}

        moves_completed = 0
        nn_moves = 0
        for _ in range(10):  # allow up to 10 steps to get 3 NN moves
            if state.game_status != GameStatus.ACTIVE:
                break
            legal = env.legal_moves()
            if not legal:
                break

            current_player = state.current_player
            if current_player not in ai_cache:
                player_cfg = AIConfig(
                    difficulty=9,
                    randomness=0.0,
                    use_neural_net=True,
                    gumbel_simulation_budget=32,
                    nn_model_id=str(canonical_path),
                    nn_model_version="v2",
                    allow_fresh_weights=False,
                    use_gpu_tree=False,
                )
                ai_cache[current_player] = GumbelMCTSAI(
                    current_player, player_cfg, board_type,
                )

            move = ai_cache[current_player].select_move(state)
            if move is None:
                # Fallback to random legal move (some phases may not be NN-supported)
                move = rng.choice(legal)
            else:
                nn_moves += 1

            state, _, done, _ = env.step(move)
            moves_completed += 1

            if nn_moves >= 3 or done:
                break

        if nn_moves == 0:
            logger.error("FAIL no neural network inference moves completed (tried %d steps)", moves_completed)
            return False

        logger.info("PASS %d NN inference moves completed (%d total steps)", nn_moves, moves_completed)

    except Exception as e:
        logger.error("FAIL inference: %s", e)
        return False

    # --- Check 6: Model file timestamp vs process start ---
    if model_mtime > process_start:
        logger.warning(
            "WARNING model was modified AFTER process started "
            "(deploy may have updated it -- restart recommended)"
        )

    logger.info("ALL CHECKS PASSED for %s", config_key)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Continuous smoke test for Lambda node health",
    )
    parser.add_argument(
        "--board-type", required=True,
        help="Board type (hex8, hexagonal, square8, square19)",
    )
    parser.add_argument(
        "--num-players", type=int, required=True,
        help="Number of players (2, 3, or 4)",
    )
    args = parser.parse_args()

    t0 = time.time()
    healthy = run_smoke_test(args.board_type, args.num_players)
    elapsed = time.time() - t0

    logger.info("Smoke test completed in %.1fs", elapsed)

    sys.exit(0 if healthy else 1)


if __name__ == "__main__":
    main()
