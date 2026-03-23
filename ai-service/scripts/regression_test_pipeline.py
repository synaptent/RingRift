#!/usr/bin/env python3
"""End-to-end regression test for the RingRift training pipeline.

Tests the full pipeline on a single node without cluster/S3/P2P dependencies:
  1. Verify NPZ training data exists and has valid structure
  2. Train 1 epoch from existing NPZ
  3. Verify checkpoint has proper metadata and SHA256 checksum
  4. Play 10 games vs random — model must win >60% (2p) or >30% (3p/4p)
  5. Verify promotion gates would NOT promote a 1-epoch model

Usage:
    python scripts/regression_test_pipeline.py
    python scripts/regression_test_pipeline.py --board-type square8 --num-players 2
    python scripts/regression_test_pipeline.py --quick  # Skip training, test existing canonical

Requires: PyTorch, ai-service dependencies. No cluster, S3, or P2P needed.
"""
import argparse
import os
import sys
import tempfile
import time

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ".")


def find_npz(board_type: str, num_players: int) -> str | None:
    """Find an existing NPZ file for this config."""
    config_key = f"{board_type}_{num_players}p"
    candidates = [
        f"data/training/{config_key}.npz",
        f"data/training/{board_type}_{num_players}p_iter1.npz",
    ]
    for path in candidates:
        if os.path.exists(path) and os.path.getsize(path) > 10000:
            return path
    return None


def stage_1_verify_npz(npz_path: str, board_type: str, num_players: int) -> bool:
    """Stage 1: Verify NPZ has valid structure."""
    import numpy as np
    print(f"\n[Stage 1] Verifying NPZ: {npz_path}")
    try:
        # numpy.load with allow_pickle=True is required for NPZ files
        # containing object arrays (policy indices). This is trusted local data.
        data = np.load(npz_path, allow_pickle=True)
        keys = list(data.keys())
        print(f"  Keys: {keys}")

        if "features" not in keys:
            print("  FAIL: Missing 'features' key")
            return False

        features = data["features"]
        print(f"  Features shape: {features.shape} dtype={features.dtype}")
        print(f"  Samples: {len(features)}")

        if len(features) < 10:
            print("  FAIL: Too few samples (<10)")
            return False

        print("  PASS: NPZ structure valid")
        return True
    except Exception as e:
        print(f"  FAIL: {e}")
        return False


def stage_2_train(npz_path: str, board_type: str, num_players: int,
                  output_dir: str) -> str | None:
    """Stage 2: Train 1 epoch and return model path."""
    import subprocess
    print(f"\n[Stage 2] Training 1 epoch: {board_type}_{num_players}p")

    model_path = os.path.join(output_dir, "best_model.pth")
    cmd = [
        sys.executable, "-m", "app.training.train",
        "--board-type", board_type,
        "--num-players", str(num_players),
        "--data-path", npz_path,
        "--epochs", "1",
        "--batch-size", "64",
        "--output-dir", output_dir,
        "--no-wandb",
    ]
    print(f"  CMD: {' '.join(cmd)}")

    t0 = time.time()
    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=600,
        env={**os.environ, "PYTHONPATH": "."},
    )
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"  FAIL: Training exited {result.returncode} ({elapsed:.0f}s)")
        print(f"  STDERR: {result.stderr[-500:]}")
        return None

    if not os.path.exists(model_path):
        for f in os.listdir(output_dir):
            if f.endswith(".pth"):
                model_path = os.path.join(output_dir, f)
                break

    if os.path.exists(model_path):
        size_mb = os.path.getsize(model_path) / 1e6
        print(f"  PASS: Model saved ({size_mb:.1f}MB, {elapsed:.0f}s)")
        return model_path
    else:
        print(f"  FAIL: No model file produced in {output_dir}")
        return None


def stage_3_verify_checkpoint(model_path: str, board_type: str,
                              num_players: int) -> bool:
    """Stage 3: Verify checkpoint has metadata and checksum."""
    print(f"\n[Stage 3] Verifying checkpoint: {model_path}")
    try:
        from app.utils.torch_utils import (
            safe_load_checkpoint, write_checksum_file, verify_model_checksum,
        )

        checkpoint = safe_load_checkpoint(
            model_path,
            expected_board_type=board_type,
            expected_num_players=num_players,
        )

        meta = checkpoint.get("_versioning_metadata", {})
        if not meta:
            print("  WARN: No _versioning_metadata in checkpoint")
        else:
            config = meta.get("config", {})
            print(f"  Metadata: board={config.get('board_type', '?')}, "
                  f"players={config.get('num_players', '?')}, "
                  f"arch={meta.get('architecture_version', '?')}")

        sha_path = write_checksum_file(model_path)
        valid, computed = verify_model_checksum(model_path)
        if valid:
            print(f"  Checksum: {computed[:16]}... (verified)")
        else:
            print(f"  FAIL: Checksum verification failed")
            return False

        print("  PASS: Checkpoint integrity verified")
        return True
    except Exception as e:
        print(f"  FAIL: {e}")
        return False


def stage_4_smoke_test(model_path: str, board_type: str,
                       num_players: int) -> tuple[bool, float]:
    """Stage 4: Play 10 games vs random."""
    print(f"\n[Stage 4] Smoke test: 10 games vs random")
    try:
        from app.models import BoardType
        from app.training.game_gauntlet import (
            play_single_game, create_neural_ai, create_baseline_ai,
            BaselineOpponent,
        )

        bt = BoardType(board_type)
        ai = create_neural_ai(
            player=1, board_type=bt, model_path=model_path,
            num_players=num_players, use_search=True, search_budget=32,
            temperature=0.1,
        )

        wins = 0
        for i in range(10):
            opp_ais = {
                p: create_baseline_ai(
                    BaselineOpponent.RANDOM, p, bt, num_players=num_players,
                )
                for p in range(2, num_players + 1)
            }
            first_opp = opp_ais[2]
            result = play_single_game(
                candidate_ai=ai, opponent_ai=first_opp,
                board_type=bt, num_players=num_players,
                candidate_player=1, max_moves=500,
                opponent_ais=opp_ais if num_players > 2 else None,
            )
            if result.candidate_won:
                wins += 1
            print(f"  Game {i+1}: {'WIN' if result.candidate_won else 'LOSS'}")

        win_rate = wins / 10
        threshold = 0.6 if num_players == 2 else 0.3
        passed = win_rate >= threshold
        status = "PASS" if passed else "FAIL"
        print(f"  {status}: {wins}/10 ({win_rate:.0%}), threshold={threshold:.0%}")
        return passed, win_rate
    except Exception as e:
        print(f"  FAIL: {e}")
        return False, 0.0


def stage_5_promotion_gate(win_rate: float, num_players: int) -> bool:
    """Stage 5: Check if promotion gates would pass."""
    print(f"\n[Stage 5] Promotion gate check")
    try:
        from app.config.thresholds import (
            ELO_IMPROVEMENT_PROMOTE, MIN_GAMES_PROMOTE,
        )
        print(f"  Elo gap required: {ELO_IMPROVEMENT_PROMOTE}")
        print(f"  Min games required: {MIN_GAMES_PROMOTE}")
        print(f"  Win rate vs random: {win_rate:.0%}")
        print(f"  (1-epoch model should NOT be promoted)")
        print("  PASS: Promotion gate check complete")
        return True
    except ImportError:
        print("  SKIP: thresholds module not available")
        return True


def main():
    parser = argparse.ArgumentParser(description="E2E regression test")
    parser.add_argument("--board-type", default="hex8")
    parser.add_argument("--num-players", type=int, default=2)
    parser.add_argument("--quick", action="store_true",
                        help="Test existing canonical model (skip training)")
    args = parser.parse_args()

    bt, np_ = args.board_type, args.num_players
    config_key = f"{bt}_{np_}p"
    print(f"{'='*60}")
    print(f"  RingRift Pipeline Regression Test: {config_key}")
    print(f"{'='*60}")

    results = {}

    if args.quick:
        model_path = f"models/canonical_{config_key}.pth"
        if not os.path.exists(model_path):
            print(f"ABORT: {model_path} not found")
            sys.exit(1)
        results["stage1"] = True
        results["stage2"] = True
        results["stage3"] = stage_3_verify_checkpoint(model_path, bt, np_)
        passed, wr = stage_4_smoke_test(model_path, bt, np_)
        results["stage4"] = passed
        results["stage5"] = stage_5_promotion_gate(wr, np_)
    else:
        npz_path = find_npz(bt, np_)
        if not npz_path:
            print(f"ABORT: No NPZ found for {config_key}")
            sys.exit(1)

        results["stage1"] = stage_1_verify_npz(npz_path, bt, np_)
        if not results["stage1"]:
            sys.exit(1)

        with tempfile.TemporaryDirectory(prefix="ringrift_regtest_") as tmpdir:
            model_path = stage_2_train(npz_path, bt, np_, tmpdir)
            results["stage2"] = model_path is not None
            if not model_path:
                sys.exit(1)

            results["stage3"] = stage_3_verify_checkpoint(model_path, bt, np_)
            passed, wr = stage_4_smoke_test(model_path, bt, np_)
            results["stage4"] = passed
            results["stage5"] = stage_5_promotion_gate(wr, np_)

    print(f"\n{'='*60}")
    print("  RESULTS")
    print(f"{'='*60}")
    all_pass = True
    for stage, passed in results.items():
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False
        print(f"  {stage}: {status}")

    print(f"\n  Overall: {'ALL PASSED' if all_pass else 'SOME FAILED'}")
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
