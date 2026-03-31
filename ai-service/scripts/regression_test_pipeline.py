#!/usr/bin/env python3
"""Pipeline regression test: end-to-end AlphaZero pipeline validation in <10 minutes.

Catches every class of silent failure found in production:
- JSONL export producing no data
- NPZ conversion failing silently
- Training loading wrong weights
- Model loading with architecture mismatch
- Evaluation falling back to random play
- Missing files/symlinks

Usage:
    cd ~/ringrift/ai-service && export PYTHONPATH=.
    python scripts/regression_test_pipeline.py           # Full run (~8 min)
    python scripts/regression_test_pipeline.py --quick    # Skip selfplay + eval (~30s)
"""
from __future__ import annotations

import argparse
import io
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("regression_test")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BOARD_TYPE = "hex8"
NUM_PLAYERS = 2
MODEL_VERSION = "v2"
CANONICAL_MODEL = PROJECT_ROOT / "models" / f"canonical_{BOARD_TYPE}_{NUM_PLAYERS}p.pth"
# hex8 v2 encoder: 40 channels, 9x9 grid
EXPECTED_FEATURE_CHANNELS = 40
EXPECTED_BOARD_H = 9
EXPECTED_BOARD_W = 9
SELFPLAY_GAMES = 5
SELFPLAY_BUDGET = 32
TRAIN_EPOCHS = 1
TRAIN_BATCH_SIZE = 64
TRAIN_LR = 3e-4
EVAL_GAMES = 2
EVAL_BUDGET = 32
QUICK_FIXTURE_GAMES = 8

# Supported checkpoint naming families across the canonical CNN variants.
# The smoke test should validate that a trained checkpoint looks like a real
# model, without assuming only one internal module naming convention.
REQUIRED_STATE_DICT_GROUPS = {
    "stem": ("conv_block", "conv1", "initial_conv"),
    "trunk": ("res_blocks", "backbone", "trunk"),
    "policy": ("policy_head", "policy_conv", "policy_fc", "spatial_policy_conv"),
    "value": ("value_head", "value_conv", "value_fc", "value_bn"),
}


def _python() -> str:
    """Return the Python executable to use for subprocesses."""
    return sys.executable


# ---------------------------------------------------------------------------
# Result tracking
# ---------------------------------------------------------------------------
class TestResult:
    def __init__(self, name: str):
        self.name = name
        self.passed = False
        self.error: str | None = None
        self.details: dict[str, Any] = {}
        self.elapsed_s: float = 0.0
        self.investigate: str | None = None

    def fail(self, error: str, investigate: str | None = None) -> "TestResult":
        self.passed = False
        self.error = error
        self.investigate = investigate
        return self

    def ok(self, **details: Any) -> "TestResult":
        self.passed = True
        self.details.update(details)
        return self


results: list[TestResult] = []


def _run_test(name: str):
    """Decorator to run a test function, capture results, and print status."""
    def decorator(fn):
        def wrapper(*args, **kwargs):
            r = TestResult(name)
            t0 = time.time()
            try:
                fn(r, *args, **kwargs)
            except Exception as e:
                r.fail(f"Exception: {e}", f"Traceback:\n{traceback.format_exc()}")
            r.elapsed_s = time.time() - t0
            results.append(r)
            status = "PASS" if r.passed else "FAIL"
            logger.info(f"  [{status}] {name} ({r.elapsed_s:.1f}s)")
            if not r.passed:
                logger.error(f"    Error: {r.error}")
                if r.investigate:
                    logger.error(f"    Investigate: {r.investigate}")
            return r
        return wrapper
    return decorator


def _subprocess_env() -> dict[str, str]:
    """Build a clean environment dict for subprocess calls."""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT)
    env["RINGRIFT_DISABLE_TORCH_COMPILE"] = "1"
    return env


def _canonical_model_available() -> bool:
    return CANONICAL_MODEL.exists()


# ===================================================================
# Test 1: Selfplay
# ===================================================================
@_run_test("1. Selfplay (5 gumbel games, hex8_2p, budget=32)")
def test_selfplay(r: TestResult, tmpdir: Path, jsonl_path: Path):
    """Generate games via the real selfplay code path."""
    if not CANONICAL_MODEL.exists():
        r.fail(
            f"Canonical model not found: {CANONICAL_MODEL}",
            "Run: python scripts/sync_models.py or download from S3",
        )
        return

    # Invoke selfplay through subprocess to test the real code path.
    # Uses minimal_alphazero_loop.run_selfplay which exercises
    # GumbelMCTSAI, the game engine, MCTS policy extraction, and JSONL output.
    script = f"""\
import json, sys, os
os.environ["RINGRIFT_DISABLE_TORCH_COMPILE"] = "1"
sys.path.insert(0, {str(PROJECT_ROOT)!r})
from scripts.minimal_alphazero_loop import run_selfplay
from pathlib import Path
result = run_selfplay(
    model_path={str(CANONICAL_MODEL)!r},
    n_games={SELFPLAY_GAMES},
    out=Path({str(jsonl_path)!r}),
    budget={SELFPLAY_BUDGET},
)
print(json.dumps(result))
"""
    proc = subprocess.run(
        [_python(), "-c", script],
        capture_output=True, text=True,
        timeout=300,
        cwd=str(PROJECT_ROOT),
        env=_subprocess_env(),
    )
    if proc.returncode != 0:
        r.fail(
            f"Selfplay subprocess failed (exit {proc.returncode})",
            f"stderr (last 500 chars): {proc.stderr[-500:]}",
        )
        return

    # Parse result JSON from stdout (last JSON line)
    sp_result = _parse_last_json_line(proc.stdout)
    if sp_result is None:
        r.fail("Could not parse selfplay result JSON from stdout",
               f"stdout (last 300 chars): {proc.stdout[-300:]}")
        return

    # Assert: JSONL file exists and >1KB
    if not jsonl_path.exists():
        r.fail("JSONL file was not created")
        return
    fsize = jsonl_path.stat().st_size
    if fsize < 1024:
        r.fail(f"JSONL file too small: {fsize} bytes (expected >1KB)",
               "Selfplay may have produced empty/corrupt games")
        return

    # Assert: has correct number of lines
    with open(jsonl_path) as f:
        lines = f.readlines()
    if len(lines) < SELFPLAY_GAMES:
        r.fail(f"JSONL has {len(lines)} lines, expected {SELFPLAY_GAMES}",
               "Some games may have failed silently")
        return

    # Assert: each line has 'winner' and 'num_moves' > 10
    for i, line in enumerate(lines):
        game = json.loads(line)
        if "winner" not in game:
            r.fail(f"Game {i} missing 'winner' field",
                   "Selfplay output format may have changed")
            return
        nm = game.get("num_moves", 0)
        if nm < 10:
            r.fail(f"Game {i} has only {nm} moves (expected >10)",
                   "Games may be terminating too early -- check engine rules")
            return

    r.ok(
        file_size=fsize,
        num_games=len(lines),
        completed=sp_result.get("completed", 0),
    )


# ===================================================================
# Test 2: JSONL -> NPZ Export
# ===================================================================
@_run_test("2. JSONL->NPZ Export (jsonl_to_npz.py --gpu-selfplay)")
def test_export(r: TestResult, jsonl_path: Path, npz_path: Path):
    """Convert JSONL to NPZ using the real CLI script."""
    if not jsonl_path.exists():
        r.fail("JSONL input file missing (selfplay test may have failed)")
        return

    cmd = [
        _python(), str(SCRIPT_DIR / "jsonl_to_npz.py"),
        "--input", str(jsonl_path),
        "--output", str(npz_path),
        "--board-type", BOARD_TYPE,
        "--num-players", str(NUM_PLAYERS),
        "--gpu-selfplay",
    ]
    proc = subprocess.run(
        cmd, capture_output=True, text=True,
        timeout=120,
        cwd=str(PROJECT_ROOT),
        env=_subprocess_env(),
    )
    if proc.returncode != 0:
        r.fail(
            f"jsonl_to_npz.py failed (exit {proc.returncode})",
            f"stderr: {proc.stderr[-500:]}",
        )
        return

    # Assert: NPZ exists and >1KB
    if not npz_path.exists():
        r.fail("NPZ file was not created (silent failure)")
        return
    fsize = npz_path.stat().st_size
    if fsize < 1024:
        r.fail(f"NPZ file too small: {fsize} bytes",
               "Export may have produced no valid samples")
        return

    # Assert: has 'features' array with >50 samples
    import numpy as np
    data = np.load(npz_path, allow_pickle=True)
    if "features" not in data:
        r.fail(
            f"NPZ missing 'features' key. Keys found: {list(data.keys())}",
            "NPZ key must be 'features' not 'boards'/'states' -- check export script",
        )
        return
    features = data["features"]
    if len(features) < 50:
        r.fail(f"Only {len(features)} samples (expected >50 from {SELFPLAY_GAMES} games)",
               "Export may be dropping most positions")
        return

    # Assert: features shape matches (N, 40, 9, 9) for hex8 v2
    expected_shape = (EXPECTED_FEATURE_CHANNELS, EXPECTED_BOARD_H, EXPECTED_BOARD_W)
    if features.shape[1:] != expected_shape:
        r.fail(
            f"Feature shape mismatch: got {features.shape[1:]}, expected {expected_shape}",
            "Encoder channel count may not match model architecture (v2=40ch)",
        )
        return

    r.ok(
        file_size=fsize,
        num_samples=len(features),
        feature_shape=list(features.shape),
        keys=list(data.keys()),
    )


# ===================================================================
# Test 3: Training (1 epoch)
# ===================================================================
@_run_test("3. Training (1 epoch on NPZ with optional canonical init weights)")
def test_training(r: TestResult, npz_path: Path, candidate_path: Path):
    """Train 1 epoch using the real training CLI."""
    if not npz_path.exists():
        r.fail("NPZ input missing (export test may have failed)")
        return

    cmd = [
        _python(), "-m", "app.training.train",
        "--data-path", str(npz_path),
        "--save-path", str(candidate_path),
        "--board-type", BOARD_TYPE,
        "--num-players", str(NUM_PLAYERS),
        "--model-version", MODEL_VERSION,
        "--epochs", str(TRAIN_EPOCHS),
        "--batch-size", str(TRAIN_BATCH_SIZE),
        "--learning-rate", str(TRAIN_LR),
        "--no-auto-tune-batch-size",
        "--lr-scheduler", "cosine",
        "--skip-freshness-check",
        "--sampling-weights", "uniform",
        "--early-stopping-patience", "0",
    ]
    if _canonical_model_available():
        cmd.extend(["--init-weights", str(CANONICAL_MODEL)])
    proc = subprocess.run(
        cmd, capture_output=True, text=True,
        timeout=600,
        cwd=str(PROJECT_ROOT),
        env=_subprocess_env(),
    )
    if proc.returncode != 0:
        r.fail(
            f"Training failed (exit {proc.returncode})",
            f"stderr (last 500 chars): {proc.stderr[-500:]}",
        )
        return

    # Assert: candidate model exists and >1MB
    if not candidate_path.exists():
        r.fail("Candidate model was not saved",
               "Check --save-path handling in train_cli.py")
        return
    fsize = candidate_path.stat().st_size
    if fsize < 1_000_000:
        r.fail(f"Candidate model too small: {fsize} bytes (expected >1MB)",
               "Model may not have saved correctly")
        return

    # Assert: can be loaded by torch.load
    import torch
    try:
        checkpoint = torch.load(str(candidate_path), map_location="cpu", weights_only=False)
    except Exception as e:
        r.fail(f"torch.load failed: {e}", "Model file may be corrupt")
        return

    # Assert: has expected state_dict structure for a supported architecture.
    state_dict = _extract_state_dict(checkpoint)
    if state_dict is None:
        r.fail(
            "Could not find state_dict in checkpoint",
            f"Checkpoint keys: {list(checkpoint.keys()) if isinstance(checkpoint, dict) else type(checkpoint)}",
        )
        return

    found_groups = _detect_state_dict_groups(state_dict)
    missing = sorted(name for name in REQUIRED_STATE_DICT_GROUPS if name not in found_groups)
    if missing:
        r.fail(
            f"State dict missing expected structure groups: {missing}",
            f"Found keys (first 10): {list(state_dict.keys())[:10]}",
        )
        return

    r.ok(
        file_size=fsize,
        num_keys=len(state_dict),
        state_dict_groups=found_groups,
    )


# ===================================================================
# Test 4: Model Loading into GumbelMCTSAI
# ===================================================================
@_run_test("4. Model Loading (load candidate into GumbelMCTSAI)")
def test_model_loading(r: TestResult, candidate_path: Path):
    """Load the trained candidate into the real inference engine."""
    if not candidate_path.exists():
        r.fail("Candidate model missing (training test may have failed)")
        return

    # Capture warnings to detect silent fallback to heuristic
    import warnings
    captured_warnings: list[str] = []
    original_warn = warnings.warn

    def capturing_warn(message, *args, **kwargs):
        captured_warnings.append(str(message))
        original_warn(message, *args, **kwargs)

    # Also capture WARNING-level log messages
    log_capture = io.StringIO()
    handler = logging.StreamHandler(log_capture)
    handler.setLevel(logging.WARNING)
    root_logger = logging.getLogger()
    root_logger.addHandler(handler)

    try:
        warnings.warn = capturing_warn  # type: ignore[assignment]

        from app.ai.gumbel_mcts_ai import GumbelMCTSAI
        from app.models import AIConfig, BoardType

        cfg = AIConfig(
            difficulty=9,
            randomness=0.0,
            use_neural_net=True,
            gumbel_simulation_budget=EVAL_BUDGET,
            nn_model_id=str(candidate_path),
            nn_model_version=MODEL_VERSION,
            allow_fresh_weights=False,
            use_gpu_tree=True,
        )
        ai = GumbelMCTSAI(1, cfg, BoardType.HEX8)
    except RuntimeError as e:
        r.fail(f"GumbelMCTSAI construction failed: {e}",
               "Model may have architecture mismatch or corrupt weights")
        return
    finally:
        warnings.warn = original_warn  # type: ignore[assignment]
        root_logger.removeHandler(handler)

    # Check for fallback warnings
    log_output = log_capture.getvalue()
    fallback_indicators = [
        "fallback", "fresh_weights", "failed to load",
        "heuristic", "random play",
    ]
    for indicator in fallback_indicators:
        for w in captured_warnings:
            if indicator.lower() in w.lower():
                r.fail(
                    f"Model loading triggered fallback warning: {w}",
                    "Model may not be loading correctly -- check architecture match",
                )
                return
        if indicator.lower() in log_output.lower():
            r.fail(
                f"Model loading produced fallback log containing: {indicator!r}",
                "Neural net may have silently failed to load",
            )
            return

    # Assert: neural net is actually loaded (not None)
    if ai.neural_net is None:
        r.fail("GumbelMCTSAI.neural_net is None -- model did not load",
               "Check allow_fresh_weights and model path")
        return

    # Verify encoder channels match model input channels
    nn = ai.neural_net
    if hasattr(nn, "model") and nn.model is not None:
        model = nn.model
        # Find first conv layer input channels
        first_conv = None
        for _name, module in model.named_modules():
            if hasattr(module, "in_channels"):
                first_conv = module
                break
        if first_conv is not None:
            in_ch = first_conv.in_channels
            if in_ch != EXPECTED_FEATURE_CHANNELS:
                r.fail(
                    f"Model input channels ({in_ch}) != expected ({EXPECTED_FEATURE_CHANNELS})",
                    "Architecture/encoder mismatch -- this causes silent fallback to heuristic",
                )
                return
            r.details["model_input_channels"] = in_ch

    r.ok(
        neural_net_loaded=ai.neural_net is not None,
        warnings_captured=len(captured_warnings),
    )


# ===================================================================
# Test 5: Evaluation (head-to-head)
# ===================================================================
@_run_test("5. Evaluation (2 head-to-head games, candidate vs canonical)")
def test_evaluation(r: TestResult, candidate_path: Path):
    """Play head-to-head games using the real inference engine."""
    if not candidate_path.exists():
        r.fail("Candidate model missing")
        return
    if not CANONICAL_MODEL.exists():
        r.fail("Canonical model missing")
        return

    # Use minimal_alphazero_loop.evaluate via subprocess
    script = f"""\
import json, sys, os
os.environ["RINGRIFT_DISABLE_TORCH_COMPILE"] = "1"
sys.path.insert(0, {str(PROJECT_ROOT)!r})
from scripts.minimal_alphazero_loop import evaluate
result = evaluate(
    cand={str(candidate_path)!r},
    best={str(CANONICAL_MODEL)!r},
    n_games={EVAL_GAMES},
    budget={EVAL_BUDGET},
)
print(json.dumps(result))
"""
    proc = subprocess.run(
        [_python(), "-c", script],
        capture_output=True, text=True,
        timeout=300,
        cwd=str(PROJECT_ROOT),
        env=_subprocess_env(),
    )
    if proc.returncode != 0:
        r.fail(
            f"Evaluation subprocess failed (exit {proc.returncode})",
            f"stderr: {proc.stderr[-500:]}",
        )
        return

    # Check for known failure signatures in output
    combined_output = (proc.stdout + proc.stderr).lower()
    if "fallback to heuristic" in combined_output:
        r.fail(
            "Evaluation fell back to heuristic play",
            "Neural net failed to load during evaluation -- check MPS/CUDA device placement",
        )
        return
    if "input type" in combined_output and "weight type" in combined_output:
        r.fail(
            "Device type mismatch detected during evaluation",
            "MPS type mismatch bug -- model weights on wrong device after load_state_dict",
        )
        return

    # Parse result
    ev_result = _parse_last_json_line(proc.stdout)
    if ev_result is None:
        r.fail("Could not parse evaluation result",
               f"stdout (last 300 chars): {proc.stdout[-300:]}")
        return

    # Assert: games completed with outcomes
    total_decided = ev_result.get("candidate_wins", 0) + ev_result.get("best_wins", 0)
    total_draws = ev_result.get("draws", 0)
    if total_decided + total_draws < EVAL_GAMES:
        r.fail(
            f"Only {total_decided + total_draws}/{EVAL_GAMES} games completed",
            "Games may be hitting max move limit or crashing",
        )
        return

    r.ok(
        candidate_wins=ev_result.get("candidate_wins", 0),
        best_wins=ev_result.get("best_wins", 0),
        draws=ev_result.get("draws", 0),
        win_rate=ev_result.get("win_rate"),
    )


# ===================================================================
# Test 6: NPZ Data Quality
# ===================================================================
@_run_test("6. NPZ Data Quality (validate features, policy, values)")
def test_npz_quality(r: TestResult, npz_path: Path):
    """Validate the NPZ file contents for numerical correctness."""
    if not npz_path.exists():
        r.fail("NPZ file missing (export test may have failed)")
        return

    import numpy as np

    data = np.load(npz_path, allow_pickle=True)

    # --- Features: no NaN, no Inf, not all zeros ---
    features = data["features"]
    if np.any(np.isnan(features)):
        nan_count = int(np.sum(np.isnan(features)))
        r.fail(f"Features contain {nan_count} NaN values",
               "Encoder may be producing invalid outputs")
        return
    if np.any(np.isinf(features)):
        inf_count = int(np.sum(np.isinf(features)))
        r.fail(f"Features contain {inf_count} Inf values")
        return
    if np.all(features == 0):
        r.fail("Features are ALL zeros",
               "Encoder is producing blank tensors -- check state encoding")
        return

    # Check that most samples have nonzero features
    nonzero_per_sample = np.count_nonzero(features.reshape(len(features), -1), axis=1)
    all_zero_samples = int(np.sum(nonzero_per_sample == 0))
    if all_zero_samples > len(features) * 0.5:
        r.fail(f"{all_zero_samples}/{len(features)} samples are all-zero (>50%)",
               "Encoder is failing on most positions")
        return

    # --- Values: in [-1, 1], no NaN ---
    if "values" in data:
        values = data["values"]
        if np.any(np.isnan(values)):
            r.fail("Values contain NaN")
            return
        if np.any(np.abs(values) > 1.01):
            r.fail(f"Values out of range: min={values.min():.3f}, max={values.max():.3f}",
                   "Value targets should be in [-1, 1]")
            return

    # --- Multi-player values ---
    if "values_mp" in data:
        values_mp = data["values_mp"]
        if np.any(np.isnan(values_mp)):
            r.fail("Multi-player values contain NaN")
            return

    # --- Policy targets: sparse format, probabilities sum to ~1.0 ---
    if "policy_indices" in data and "policy_values" in data:
        policy_indices = data["policy_indices"]
        policy_values = data["policy_values"]
        n_checked = min(20, len(policy_indices))
        bad_sums = 0
        empty_policies = 0
        for i in range(n_checked):
            pv = policy_values[i]
            if pv is None or len(pv) == 0:
                empty_policies += 1
                continue
            pv_arr = np.array(pv, dtype=np.float64)
            if np.any(np.isnan(pv_arr)):
                r.fail(f"Policy values at sample {i} contain NaN")
                return
            total = float(np.sum(pv_arr))
            if abs(total - 1.0) > 0.05:
                bad_sums += 1
        if bad_sums > n_checked * 0.5:
            r.fail(f"{bad_sums}/{n_checked} checked policy targets don't sum to ~1.0",
                   "MCTS policy extraction may be broken")
            return
        if empty_policies > n_checked * 0.5:
            r.fail(f"{empty_policies}/{n_checked} samples have empty policy targets",
                   "Policy is not being recorded during selfplay")
            return

    r.ok(
        num_samples=len(features),
        feature_range=(float(features.min()), float(features.max())),
        all_zero_samples=all_zero_samples,
        value_range=(
            (float(data["values"].min()), float(data["values"].max()))
            if "values" in data else None
        ),
    )


# ===================================================================
# Test 7: Promotion Metadata
# ===================================================================
@_run_test("7. Promotion Logic (verify candidate model metadata)")
def test_promotion_metadata(r: TestResult, candidate_path: Path):
    """Verify the candidate model has correct structure for promotion."""
    if not candidate_path.exists():
        r.fail("Candidate model missing")
        return

    import torch

    checkpoint = torch.load(str(candidate_path), map_location="cpu", weights_only=False)

    if isinstance(checkpoint, dict):
        # Check for metadata fields that promotion logic relies on.
        # May be in the checkpoint dict itself or in a nested 'metadata' key.
        meta = checkpoint.get("metadata", checkpoint)

        bt = meta.get("board_type")
        if bt is not None and bt != BOARD_TYPE:
            r.fail(f"Checkpoint board_type={bt!r}, expected {BOARD_TYPE!r}",
                   "Model may be trained for wrong board")
            return

        np_val = meta.get("num_players")
        if np_val is not None and np_val != NUM_PLAYERS:
            r.fail(f"Checkpoint num_players={np_val}, expected {NUM_PLAYERS}",
                   "Model may be trained for wrong player count")
            return

        mv = meta.get("model_version") or meta.get("version")
        if mv is not None and mv != MODEL_VERSION:
            r.fail(f"Checkpoint model_version={mv!r}, expected {MODEL_VERSION!r}")
            return

    # Verify the candidate is different from canonical (training actually changed weights)
    if _canonical_model_available():
        canonical_ckpt = torch.load(str(CANONICAL_MODEL), map_location="cpu", weights_only=False)
        cand_sd = _extract_state_dict(checkpoint)
        canon_sd = _extract_state_dict(canonical_ckpt)

        if cand_sd is not None and canon_sd is not None:
            identical_keys = 0
            total_keys = 0
            for key in list(cand_sd.keys())[:20]:
                if key in canon_sd:
                    total_keys += 1
                    if torch.equal(cand_sd[key], canon_sd[key]):
                        identical_keys += 1
            if total_keys > 0 and identical_keys == total_keys:
                r.fail(
                    "Candidate weights are IDENTICAL to canonical (training had no effect)",
                    "Training may have loaded wrong weights or gradient updates failed",
                )
                return
            r.details["weight_diff_ratio"] = (
                f"{total_keys - identical_keys}/{total_keys} keys differ"
            )

    r.ok(
        has_metadata=isinstance(checkpoint, dict) and (
            "metadata" in checkpoint or "board_type" in checkpoint
        ),
        checkpoint_type=type(checkpoint).__name__,
    )


# ===================================================================
# Helpers
# ===================================================================
def _parse_last_json_line(stdout: str) -> dict | None:
    """Parse the last JSON-parseable line from subprocess stdout."""
    for line in reversed(stdout.strip().split("\n")):
        try:
            return json.loads(line)
        except (json.JSONDecodeError, ValueError):
            continue
    return None


def _extract_state_dict(checkpoint: Any) -> dict | None:
    """Extract state_dict from a checkpoint, handling various formats."""
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "state_dict"):
            if key in checkpoint:
                return checkpoint[key]
        # If the dict itself looks like a state_dict (keys are parameter names)
        if any("." in k for k in list(checkpoint.keys())[:5]):
            return checkpoint
    return None


def _detect_state_dict_groups(state_dict: dict[str, Any]) -> dict[str, str]:
    """Return the matched architecture group -> prefix for a checkpoint."""
    found: dict[str, str] = {}
    for group, prefixes in REQUIRED_STATE_DICT_GROUPS.items():
        for prefix in prefixes:
            if any(key.startswith(prefix) for key in state_dict):
                found[group] = prefix
                break
    return found


def _find_or_create_quick_jsonl(tmpdir: Path) -> Path:
    """For --quick mode: find an existing JSONL or generate a deterministic one."""
    quick_path = tmpdir / "quick_test.jsonl"

    # Try existing test fixtures
    test_fixtures = [
        PROJECT_ROOT / "tests" / "fixtures" / "selfplay_hex8_2p.jsonl",
        PROJECT_ROOT / "data" / "selfplay" / "regression_test.jsonl",
    ]
    for fixture in test_fixtures:
        if fixture.exists() and fixture.stat().st_size > 1024:
            shutil.copy2(fixture, quick_path)
            logger.info(f"  Using existing test fixture: {fixture}")
            return quick_path

    logger.info(
        "  No test fixture found, generating %d deterministic random games...",
        QUICK_FIXTURE_GAMES,
    )
    try:
        _write_quick_fixture_jsonl(quick_path)
    except Exception as e:
        logger.warning(f"  Quick JSONL generation failed: {e}")
        logger.warning("  Cannot proceed without test data")
    return quick_path


def _write_quick_fixture_jsonl(path: Path) -> None:
    """Generate a tiny deterministic JSONL fixture without external model files."""
    import random

    from app.models import BoardType, GameStatus, Move
    from app.training.env import TrainingEnvConfig, get_theoretical_max_moves, make_env

    board_enum = BoardType.HEX8
    tmax = get_theoretical_max_moves(board_enum, NUM_PLAYERS)
    env = make_env(
        TrainingEnvConfig(
            board_type=board_enum,
            num_players=NUM_PLAYERS,
            max_moves=int(tmax * 1.5),
        )
    )
    rng = random.Random(12345)

    def serialize_move(move: Move, phase: str, move_number: int) -> dict[str, Any]:
        payload = move.model_dump(by_alias=True, exclude_none=True, mode="json")
        if phase and "phase" not in payload:
            payload["phase"] = phase
        payload["moveNumber"] = move_number
        return payload

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for game_idx in range(QUICK_FIXTURE_GAMES):
            state = env.reset(seed=1234 + game_idx)
            initial_state = state.model_dump(by_alias=True, exclude_none=True, mode="json")
            moves: list[dict[str, Any]] = []
            move_count = 0

            while state.game_status == GameStatus.ACTIVE and move_count < 200:
                legal = env.legal_moves()
                if not legal:
                    break
                move = rng.choice(legal)
                phase = (
                    state.current_phase.value
                    if hasattr(state.current_phase, "value")
                    else str(state.current_phase)
                )
                moves.append(serialize_move(move, phase, move_count + 1))
                state, _, done, _ = env.step(move)
                move_count += 1
                if done:
                    break

            winner = state.winner if state.game_status == GameStatus.COMPLETED else None
            record = {
                "game_id": str(uuid.uuid4()),
                "board_type": BOARD_TYPE,
                "num_players": NUM_PLAYERS,
                "winner": winner,
                "status": state.game_status.value,
                "num_moves": move_count,
                "moves": moves,
                "initial_state": initial_state,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            f.write(json.dumps(record) + "\n")


# ===================================================================
# Main
# ===================================================================
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pipeline regression test: end-to-end AlphaZero validation",
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Skip selfplay + evaluation and use a deterministic JSONL fixture.",
    )
    parser.add_argument(
        "--keep-tmpdir", action="store_true",
        help="Don't clean up temp directory after test (for debugging).",
    )
    args = parser.parse_args()

    # Pre-flight check
    if not args.quick and not _canonical_model_available():
        logger.error(f"FATAL: Canonical model not found: {CANONICAL_MODEL}")
        logger.error("Download it first: python scripts/sync_models.py or from S3")
        sys.exit(1)

    logger.info("=" * 70)
    logger.info("PIPELINE REGRESSION TEST")
    logger.info(f"  Config: {BOARD_TYPE}_{NUM_PLAYERS}p, model_version={MODEL_VERSION}")
    logger.info(
        f"  Canonical model: {CANONICAL_MODEL} "
        f"({'present' if _canonical_model_available() else 'missing'})"
    )
    logger.info(f"  Mode: {'QUICK (skip selfplay + eval)' if args.quick else 'FULL (~8 min)'}")
    logger.info("=" * 70)

    tmpdir = Path(tempfile.mkdtemp(prefix="ringrift_regression_"))
    logger.info(f"  Temp dir: {tmpdir}")

    jsonl_path = tmpdir / "selfplay.jsonl"
    npz_path = tmpdir / "training_data.npz"
    candidate_path = tmpdir / "candidate.pth"

    t0 = time.time()

    try:
        # ---------------------------------------------------------------
        # Test 1: Selfplay
        # ---------------------------------------------------------------
        if args.quick:
            logger.info("\n[SKIP] Test 1: Selfplay (--quick mode)")
            jsonl_path = _find_or_create_quick_jsonl(tmpdir)
            if not jsonl_path.exists():
                logger.error("  Could not create quick test JSONL, aborting")
                sys.exit(1)
            skip_result = TestResult("1. Selfplay (SKIPPED -- quick mode)")
            skip_result.ok(skipped=True)
            results.append(skip_result)
        else:
            logger.info("\n[TEST 1] Selfplay")
            test_selfplay(tmpdir, jsonl_path)

        # ---------------------------------------------------------------
        # Test 2: JSONL -> NPZ Export
        # ---------------------------------------------------------------
        logger.info("\n[TEST 2] JSONL -> NPZ Export")
        test_export(jsonl_path, npz_path)

        # ---------------------------------------------------------------
        # Test 3: Training
        # ---------------------------------------------------------------
        logger.info("\n[TEST 3] Training")
        test_training(npz_path, candidate_path)

        # ---------------------------------------------------------------
        # Test 4: Model Loading
        # ---------------------------------------------------------------
        logger.info("\n[TEST 4] Model Loading")
        test_model_loading(candidate_path)

        # ---------------------------------------------------------------
        # Test 5: Evaluation
        # ---------------------------------------------------------------
        if args.quick:
            logger.info("\n[SKIP] Test 5: Evaluation (--quick mode)")
            skip_result = TestResult("5. Evaluation (SKIPPED -- quick mode)")
            skip_result.ok(skipped=True)
            results.append(skip_result)
        else:
            logger.info("\n[TEST 5] Evaluation")
            test_evaluation(candidate_path)

        # ---------------------------------------------------------------
        # Test 6: NPZ Data Quality
        # ---------------------------------------------------------------
        logger.info("\n[TEST 6] NPZ Data Quality")
        test_npz_quality(npz_path)

        # ---------------------------------------------------------------
        # Test 7: Promotion Metadata
        # ---------------------------------------------------------------
        logger.info("\n[TEST 7] Promotion Metadata")
        test_promotion_metadata(candidate_path)

    finally:
        if not args.keep_tmpdir:
            try:
                shutil.rmtree(tmpdir)
                logger.info(f"\n  Cleaned up: {tmpdir}")
            except OSError as e:
                logger.warning(f"  Cleanup failed: {e}")
        else:
            logger.info(f"\n  Keeping temp dir: {tmpdir}")

    # ---------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------
    elapsed = time.time() - t0
    logger.info("\n" + "=" * 70)
    logger.info("RESULTS SUMMARY")
    logger.info("=" * 70)

    passed = sum(1 for r in results if r.passed)
    failed = sum(1 for r in results if not r.passed)

    for r in results:
        status = "PASS" if r.passed else "FAIL"
        time_str = f"{r.elapsed_s:.1f}s" if r.elapsed_s > 0 else "---"
        logger.info(f"  [{status}] {r.name} ({time_str})")
        if not r.passed:
            logger.info(f"         Error: {r.error}")
            if r.investigate:
                logger.info(f"         Investigate: {r.investigate}")
        elif r.details:
            # Show key details for passed tests (omit internal flags)
            detail_str = ", ".join(
                f"{k}={v}" for k, v in r.details.items()
                if k not in ("skipped",)
            )
            if detail_str:
                logger.info(f"         {detail_str}")

    logger.info(f"\n  Total: {passed}/{passed + failed} passed in {elapsed:.0f}s")

    if failed > 0:
        logger.error(f"\n  {failed} TEST(S) FAILED -- pipeline has regressions!")
        sys.exit(1)
    else:
        logger.info("\n  ALL TESTS PASSED -- pipeline is healthy")
        sys.exit(0)


if __name__ == "__main__":
    main()
