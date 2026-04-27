#!/usr/bin/env python3
"""Minimal standalone AlphaZero training loop for a single GPU node.

Runs the complete self-improvement cycle with ZERO external dependencies
(no S3, P2P, coordinator, daemons, work queue, event bus).

Usage:
    cd ~/ringrift/ai-service && export PYTHONPATH=.
    python scripts/minimal_alphazero_loop.py \
        --model models/canonical_hex8_2p.pth \
        --iterations 20 --games-per-iter 300 --eval-games 100 --budget 128
"""
from __future__ import annotations

import argparse, json, logging, math, os, random, shutil, socket, subprocess
import sys, tempfile, time, uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

if not os.environ.get("RINGRIFT_DISABLE_TORCH_COMPILE"):
    os.environ["RINGRIFT_DISABLE_TORCH_COMPILE"] = "1"

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.ai.gumbel_mcts_ai import GumbelMCTSAI
from app.models import AIConfig, BoardType, GameStatus, Move
from app.training.env import TrainingEnvConfig, get_theoretical_max_moves, make_env
from scripts.lib.loop_self_healing import (
    FailureContext,
    attempt_recovery,
    reset_recovery_counts,
)
from scripts.lib.minimal_loop_strategy import recommend_transfer_source, resolve_loop_profile
from scripts.lib.model_quality_gate import QualityGateTracker, check_model_quality
from scripts.lib.plateau_detector import detect_plateau
from scripts.lib.training_probes import run_training_probes

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger("minimal_alphazero")

# Defaults — overridden by --board-type and --num-players CLI args
BOARD_TYPE = "hex8"
BOARD_ENUM = BoardType.HEX8
NUM_PLAYERS = 2
MODEL_VERSION = "v2"  # Overridden by --model-version CLI arg
FEATURE_VERSION = 2  # Overridden by --feature-version CLI arg; 3 disables placement-validity shortcut (c790d339f)
MAX_MOVES = 800
INITIAL_ESTIMATED_ELO = 1500.0

BOARD_TYPE_MAP = {
    "hex8": BoardType.HEX8,
    "hexagonal": BoardType.HEXAGONAL,
    "square8": BoardType.SQUARE8,
    "square19": BoardType.SQUARE19,
}


def _make_ai(player: int, model_path: str, budget: int,
             randomness: float = 0.0) -> GumbelMCTSAI:
    cfg = AIConfig(difficulty=9, randomness=randomness, use_neural_net=True,
                   gumbel_simulation_budget=budget, nn_model_id=model_path,
                   allow_fresh_weights=False, use_gpu_tree=True,
                   nn_model_version=MODEL_VERSION if MODEL_VERSION != "v2" else None,
                   feature_version=FEATURE_VERSION)
    return GumbelMCTSAI(player, cfg, BOARD_ENUM)


def _serialize_move(move: Move, policy: dict | None, phase: str, num: int) -> dict:
    d = move.model_dump(by_alias=True, exclude_none=True, mode="json")
    if phase and "phase" not in d:
        d["phase"] = phase
    d["moveNumber"] = num
    if policy:
        d["mcts_policy"] = policy
    d["policy_target"] = bool(policy)
    return d


def _extract_policy(ai: GumbelMCTSAI) -> dict[str, float]:
    if not hasattr(ai, "_last_search_actions") or ai._last_search_actions is None:
        return {}
    total = sum(a.visit_count for a in ai._last_search_actions)
    if total == 0:
        return {}
    pol = {}
    for a in ai._last_search_actions:
        if a.visit_count > 0:
            m = a.move
            key = m.type.value
            if m.from_pos:
                key += f"_{m.from_pos.x},{m.from_pos.y}"
            if m.to:
                key += f"_{m.to.x},{m.to.y}"
            pol[key] = a.visit_count / total
    return pol


def _make_env():
    tmax = get_theoretical_max_moves(BOARD_ENUM, NUM_PLAYERS)
    return make_env(TrainingEnvConfig(board_type=BOARD_ENUM, num_players=NUM_PLAYERS,
                                     max_moves=int(tmax * 1.5)))


def _play_game(env, ai_players: dict[int, GumbelMCTSAI], idx: int, seed: int):
    gseed = (seed + idx * 1_000_003) & 0xFFFFFFFF
    for p, ai in ai_players.items():
        if hasattr(ai, "reset_for_new_game"):
            ai.reset_for_new_game(rng_seed=(gseed + p * 97_911) & 0xFFFFFFFF)
    state = env.reset(seed=gseed)
    init = state.model_dump(by_alias=True, exclude_none=True, mode="json")
    moves, mc = [], 0
    while state.game_status == GameStatus.ACTIVE and mc < MAX_MOVES:
        cp = state.current_player
        ai = ai_players.get(cp)
        if ai is None:
            break
        ai.player_number = cp
        legal = env.legal_moves()
        if not legal:
            break
        sel = ai.select_move(state)
        if sel is None:
            break
        if sel not in legal:
            sel = legal[random.randint(0, len(legal) - 1)]
        phase = state.current_phase.value if hasattr(state.current_phase, "value") else str(state.current_phase)
        moves.append(_serialize_move(sel, _extract_policy(ai), phase, mc + 1))
        state, _, done, _ = env.step(sel)
        mc += 1
        if done:
            break
    winner = state.winner if state.game_status == GameStatus.COMPLETED else None
    return {"game_id": str(uuid.uuid4()), "board_type": BOARD_TYPE,
            "num_players": NUM_PLAYERS, "winner": winner,
            "status": state.game_status.value, "num_moves": mc,
            "moves": moves, "initial_state": init,
            "final_state": state.model_dump(by_alias=True, exclude_none=True, mode="json"),
            "timestamp": datetime.now(timezone.utc).isoformat()}


def _count_valid_games(path: Path) -> tuple[int, dict[int, int]]:
    """Count well-formed JSONL game records. Stops at first corrupt line.

    Returns (game_count, wins_by_player). Caller rewrites the file truncated
    to game_count lines so appended resumes don't leave garbage trailing a
    corrupt line.
    """
    wins: dict[int, int] = {p: 0 for p in range(1, NUM_PLAYERS + 1)}
    count = 0
    if not path.exists():
        return 0, wins
    try:
        with open(path) as f:
            for line in f:
                if not line.strip():
                    break
                try:
                    g = json.loads(line)
                except (json.JSONDecodeError, ValueError):
                    break
                w = g.get("winner")
                if w in wins:
                    wins[w] += 1
                count += 1
    except OSError:
        return 0, {p: 0 for p in range(1, NUM_PLAYERS + 1)}
    return count, wins


def _truncate_jsonl_to(path: Path, keep_count: int) -> None:
    """Rewrite path keeping only the first `keep_count` valid JSON lines.

    Called before append-mode resume so partial/corrupt trailing data is
    discarded and the file is a clean prefix of what's needed.
    """
    if keep_count == 0:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass
        return
    try:
        kept: list[str] = []
        with open(path) as f:
            for line in f:
                if len(kept) >= keep_count:
                    break
                try:
                    json.loads(line)
                except (json.JSONDecodeError, ValueError):
                    break
                kept.append(line if line.endswith("\n") else line + "\n")
        with open(path, "w") as f:
            f.writelines(kept)
    except OSError:
        pass


def _promotion_elo_delta(win_rate: float, num_players: int) -> float:
    """Estimate Elo gain from a promotion against the fair-seat baseline."""
    if not 0.0 < win_rate < 1.0 or num_players < 2:
        return 0.0

    # Preserve the legacy 2p arithmetic exactly so existing 2p Elo history
    # remains stable while multiplayer configs get the fair-baseline correction.
    if num_players == 2:
        return 400.0 * math.log10(win_rate / (1.0 - win_rate))

    fair_win_rate = 1.0 / float(num_players)
    if not 0.0 < fair_win_rate < 1.0:
        return 0.0

    fair_odds = fair_win_rate / (1.0 - fair_win_rate)
    odds = win_rate / (1.0 - win_rate)
    return 400.0 * math.log10(odds / fair_odds)


def _extract_metric_win_rate(metric: dict[str, Any]) -> float | None:
    evaluation = metric.get("evaluation")
    if not isinstance(evaluation, dict):
        return None
    win_rate = evaluation.get("win_rate")
    if isinstance(win_rate, bool):
        return None
    if isinstance(win_rate, (int, float)):
        return float(win_rate)
    return None


def _recompute_progress_from_metrics(
    metrics_history: list[dict[str, Any]],
    num_players: int,
    *,
    initial_elo: float = INITIAL_ESTIMATED_ELO,
) -> tuple[int, float]:
    """Replay historical promotions so resumed Elo uses the current formula."""
    promos = 0
    elo = initial_elo
    saw_promotion_marker = False

    for metric in metrics_history:
        if "promoted" in metric:
            saw_promotion_marker = True
        if not metric.get("promoted"):
            continue
        promos += 1
        win_rate = _extract_metric_win_rate(metric)
        if win_rate is None:
            continue
        elo += _promotion_elo_delta(win_rate, num_players)

    if saw_promotion_marker or not metrics_history:
        return promos, elo

    latest = metrics_history[-1]
    fallback_promos = latest.get("total_promotions")
    fallback_elo = latest.get("estimated_elo")
    if isinstance(fallback_promos, int) and isinstance(fallback_elo, (int, float)):
        return fallback_promos, float(fallback_elo)

    return promos, elo


def run_selfplay(model_path: str, n_games: int, out: Path, budget: int,
                  randomness: float = 0.25) -> dict:
    out.parent.mkdir(parents=True, exist_ok=True)

    # Game-granular resume: reuse any completed games already on disk from
    # a prior run of this same iteration. Truncate the JSONL to the last
    # valid line before appending, so restart never duplicates or corrupts
    # games.
    resume_start, wins = _count_valid_games(out)
    _truncate_jsonl_to(out, resume_start)

    if resume_start >= n_games:
        logger.info(f"  selfplay already complete: {resume_start}/{n_games} games on disk")
        result: dict[str, Any] = {"completed": resume_start, "elapsed_s": 0.0}
        for p in wins:
            result[f"p{p}_wins"] = wins[p]
        return result

    env = _make_env()
    # Use exploration noise (randomness > 0) for training data diversity
    ais = {p: _make_ai(p, model_path, budget, randomness=randomness)
           for p in range(1, NUM_PLAYERS + 1)}
    # Use os.urandom for entropy instead of Python random (which may share state)
    seed = int.from_bytes(os.urandom(4), "big")
    if resume_start > 0:
        logger.info(f"  selfplay resume: {resume_start}/{n_games} already done, continuing from game {resume_start}")
    logger.info(f"  selfplay seed={seed}")
    done_n, t0 = resume_start, time.time()
    with open(out, "a") as f:
        for i in range(resume_start, n_games):
            try:
                g = _play_game(env, ais, i, seed)
                f.write(json.dumps(g) + "\n")
                f.flush()  # Durable per-game so SIGTERM loses at most the in-flight game
                w = g.get("winner")
                if w in wins:
                    wins[w] += 1
                done_n += 1
                if (i + 1) % max(1, n_games // 10) == 0:
                    wstr = " ".join(f"P{p}={wins[p]}" for p in sorted(wins))
                    logger.info(f"  selfplay {i+1}/{n_games} {wstr} ({time.time()-t0:.0f}s)")
            except Exception as e:
                logger.warning(f"  game {i} failed: {e}")
    el = time.time() - t0
    logger.info(f"  selfplay done: {done_n}/{n_games} in {el:.0f}s")
    result = {"completed": done_n, "elapsed_s": el}
    for p in wins:
        result[f"p{p}_wins"] = wins[p]
    return result


def export_npz(jsonl: Path, npz: Path) -> bool:
    # --gpu-selfplay is needed because GumbelMCTSAI records only player actions,
    # not bookkeeping phase transitions (line processing, territory, etc.).
    # The converter auto-injects these when --gpu-selfplay is set.
    # Delete existing NPZ first to avoid stale data if export fails
    if npz.exists():
        npz.unlink()
    # v4 and v5-heavy need v3 encoder (64 channels); v2 uses v2 encoder (40 channels)
    encoder = "v3" if MODEL_VERSION in ("v4", "v5-heavy") else "v2"
    cmd = [sys.executable, str(SCRIPT_DIR / "jsonl_to_npz.py"),
           "--input", str(jsonl), "--output", str(npz),
           "--board-type", BOARD_TYPE, "--num-players", str(NUM_PLAYERS),
           "--encoder-version", encoder,
           "--feature-version", str(FEATURE_VERSION),
           "--gpu-selfplay"]
    if MODEL_VERSION in ("v5", "v5-gnn", "v5-heavy"):
        cmd.append("--include-heuristics")
    logger.info(f"  exporting {jsonl.name} -> {npz.name}")
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        logger.error(f"  export failed: {r.stderr[-500:]}")
        return False
    if not npz.exists():
        logger.error("  export produced no file")
        return False
    try:
        d = np.load(npz, allow_pickle=True)
        n_samples = len(d["features"])
        n_channels = d["features"].shape[1] if len(d["features"].shape) >= 2 else 0
        fsum = float(d["features"].sum())
        logger.info(f"  exported {n_samples} samples, {n_channels}ch (checksum={fsum:.1f})")
        # Contract validation: catch encoding mismatches immediately
        # v4/v5-heavy use v3 encoder (64ch for hex); v2 uses default (40ch)
        try:
            from app.training.board_encoding_contract import get_expected_channels
            expected = get_expected_channels(BOARD_ENUM)
            if MODEL_VERSION in ("v4", "v5-heavy"):
                # v3 encoder produces 64ch for hex boards
                expected_v3 = 64 if "hex" in BOARD_TYPE else expected
                if n_channels != expected_v3:
                    logger.error(
                        f"  ENCODING MISMATCH: NPZ has {n_channels}ch but "
                        f"{MODEL_VERSION} expects {expected_v3}ch for {BOARD_TYPE}."
                    )
                    return False
            elif n_channels != expected:
                logger.error(
                    f"  ENCODING MISMATCH: NPZ has {n_channels}ch but contract "
                    f"expects {expected}ch for {BOARD_TYPE}. Training will fail!"
                )
                return False
        except ImportError:
            pass
    except Exception as e:
        logger.error(f"  NPZ validation FAILED: {e}")
        return False
    return True


def _combine_npz_files(npz_files: list[Path], output_path: Path) -> tuple[Path, int] | None:
    """Merge multiple local/supplemental NPZ shards into one training file."""

    if len(npz_files) <= 1:
        return None

    arrays: dict[str, list[np.ndarray]] = {}
    scalars: dict[str, np.ndarray] = {}
    for npz_file in npz_files:
        with np.load(str(npz_file), allow_pickle=True) as data:  # trusted local training data
            for key in data.files:
                value = data[key]
                if hasattr(value, "shape") and len(value.shape) > 0:
                    arrays.setdefault(key, []).append(value)
                else:
                    scalars[key] = value

    merged: dict[str, np.ndarray] = {}
    for key, values in arrays.items():
        merged[key] = np.concatenate(values) if len(values) > 1 else values[0]
    for key, value in scalars.items():
        if key not in merged:
            merged[key] = value

    np.savez_compressed(str(output_path), **merged)
    n_samples = int(len(merged.get("features", [])))
    return output_path, n_samples


def _summarize_training_targets(npz_path: Path) -> dict[str, Any]:
    """Return compact value-target stats for probe failure diagnostics."""
    try:
        with np.load(str(npz_path), allow_pickle=True) as data:
            stats: dict[str, Any] = {}
            if "values" in data:
                values = np.asarray(data["values"], dtype=np.float32).reshape(-1)
                if values.size:
                    rounded, counts = np.unique(np.round(values, 3), return_counts=True)
                    stats["values"] = {
                        "samples": int(values.size),
                        "mean": round(float(values.mean()), 6),
                        "std": round(float(values.std()), 6),
                        "min": round(float(values.min()), 6),
                        "max": round(float(values.max()), 6),
                        "histogram": {
                            f"{float(value):.3f}": int(count)
                            for value, count in zip(rounded, counts, strict=False)
                        },
                    }
            if "values_mp" in data:
                values_mp = np.asarray(data["values_mp"], dtype=np.float32).reshape(-1)
                if values_mp.size:
                    rounded, counts = np.unique(np.round(values_mp, 3), return_counts=True)
                    stats["values_mp"] = {
                        "samples": int(values_mp.size),
                        "mean": round(float(values_mp.mean()), 6),
                        "std": round(float(values_mp.std()), 6),
                        "histogram": {
                            f"{float(value):.3f}": int(count)
                            for value, count in zip(rounded, counts, strict=False)
                        },
                    }
            return stats
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def train_model(
    npz: Path,
    out: Path,
    init: Path,
    epochs: int,
    bs: int,
    lr: float,
    train_lr_scheduler: str,
    *,
    policy_weight: float | None = None,
    value_weight: float | None = None,
    rank_dist_weight: float | None = None,
    gradient_clip_max_norm: float | None = None,
) -> dict:
    """Train candidate. Retries with halved batch on OOM."""
    for attempt in range(3):
        b = bs // (2 ** attempt)
        cmd = [sys.executable, "-m", "app.training.train",
               "--data-path", str(npz), "--save-path", str(out),
               "--board-type", BOARD_TYPE, "--num-players", str(NUM_PLAYERS),
               "--model-version", MODEL_VERSION,
               "--feature-version", str(FEATURE_VERSION),
               "--epochs", str(epochs),
               "--batch-size", str(b), "--learning-rate", str(lr),
               "--init-weights", str(init), "--no-auto-tune-batch-size",
               "--lr-scheduler", train_lr_scheduler, "--skip-freshness-check",
               "--sampling-weights", "uniform"]
        if policy_weight is not None:
            cmd.extend(["--policy-weight", str(policy_weight)])
        if value_weight is not None:
            cmd.extend(["--value-weight", str(value_weight)])
        if rank_dist_weight is not None:
            cmd.extend(["--rank-dist-weight", str(rank_dist_weight)])
        if gradient_clip_max_norm is not None:
            cmd.extend(["--gradient-clip-max-norm", str(gradient_clip_max_norm)])
        logger.info(
            f"  training epochs={epochs} bs={b} lr={lr} "
            f"scheduler={train_lr_scheduler} policy_w={policy_weight} "
            f"value_w={value_weight} rank_w={rank_dist_weight} "
            f"grad_clip={gradient_clip_max_norm} (attempt {attempt+1})"
        )
        t0 = time.time()
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
        el = time.time() - t0
        if r.returncode == 0:
            info: dict[str, Any] = {
                "elapsed_s": el,
                "batch_size": b,
                "learning_rate": lr,
                "epochs": epochs,
                "scheduler": train_lr_scheduler,
                "policy_weight": policy_weight,
                "value_weight": value_weight,
                "rank_dist_weight": rank_dist_weight,
                "gradient_clip_max_norm": gradient_clip_max_norm,
            }
            for line in (r.stdout + r.stderr).split("\n"):
                if "val_loss" in line.lower():
                    info["log_line"] = line.strip()
                if "Epoch" in line and "loss" in line.lower():
                    info["last_epoch_line"] = line.strip()
            logger.info(f"  training done in {el:.0f}s")
            return info
        err = r.stderr[-800:] if r.stderr else ""
        if "CUDA out of memory" in err or "OutOfMemoryError" in err:
            logger.warning(f"  OOM at bs={b}, retrying with {b//2}")
            continue
        logger.error(f"  training failed (exit {r.returncode}): {err[-300:]}")
        return {"error": err[-500:]}
    logger.error("  training failed after 3 OOM retries")
    return {"error": "CUDA out of memory after 3 retries"}


def evaluate(cand: str, best: str, n_games: int, budget: int,
             tracker: QualityGateTracker | None = None) -> dict:
    """Head-to-head: candidate vs best, rotating the candidate seat fairly.

    If *tracker* is provided, records the candidate's moves and value head
    outputs for the model quality gate (behavioral diversity + value health).
    """
    env = _make_env()
    cw, bw, dr, t0 = 0, 0, 0, time.time()
    for i in range(n_games):
        gseed = 42_000 + i * 7919
        num_p = env.num_players if hasattr(env, "num_players") else 2
        ais = {}
        candidate_player = (i % num_p) + 1
        for p in range(1, num_p + 1):
            model = cand if p == candidate_player else best
            ais[p] = _make_ai(p, model, budget)
        for p, ai in ais.items():
            if hasattr(ai, "reset_for_new_game"):
                ai.reset_for_new_game(rng_seed=(gseed + p * 97_911) & 0xFFFFFFFF)
        state, mc = env.reset(seed=gseed), 0
        cand_move_num = 0  # per-game move counter for candidate only
        while state.game_status == GameStatus.ACTIVE and mc < MAX_MOVES:
            c = state.current_player
            if c not in ais:
                break  # unexpected player index
            ais[c].player_number = c
            legal = env.legal_moves()
            if not legal:
                break
            mv = ais[c].select_move(state)
            if mv is None:
                break
            if mv not in legal:
                mv = legal[random.randint(0, len(legal) - 1)]
            # Track candidate moves for the quality gate
            if tracker is not None and c == candidate_player:
                root_value = None
                stats = getattr(ais[c], "_last_search_stats", None)
                if isinstance(stats, dict):
                    root_value = stats.get("root_value")
                tracker.record_move(i, cand_move_num, mv, legal,
                                    root_value=root_value)
                cand_move_num += 1
            state, _, done, _ = env.step(mv)
            mc += 1
            if done:
                break
        if tracker is not None:
            tracker.finish_game(i)
        w = state.winner if state.game_status == GameStatus.COMPLETED else None
        if w is None:
            dr += 1
        elif w == candidate_player:
            cw += 1
        else:
            bw += 1
        if tracker is not None:
            tracker.record_game_outcome(
                i, candidate_player, w == candidate_player,
            )
        if (i + 1) % max(1, n_games // 5) == 0:
            logger.info(f"  eval {i+1}/{n_games} cand={cw} best={bw} draws={dr}")
    el = time.time() - t0
    dec = cw + bw
    wr = cw / dec if dec > 0 else 0.5
    logger.info(f"  eval done: cand {cw}-{bw} best (wr={wr:.1%}, {dr} draws, {el:.0f}s)")
    return {"candidate_wins": cw, "best_wins": bw, "draws": dr,
            "win_rate": wr, "elapsed_s": el}


# ---------------------------------------------------------------------------
# Staged Evaluation: play games in batches, exit early on clear wins/losses
# ---------------------------------------------------------------------------
# Stage 1 (50 games):  promote if >60%, reject if <42%, else continue
# Stage 2 (100 total): promote if >56%, reject if <46%, else continue
# Stage 3 (200 total): promote if >53%, reject if <48%, else continue
# Stage 4 (400 total): promote if >50%, reject otherwise
#
# Detects true 53% models 83% of the time (vs 50% with old 50-game eval).
# Clear wins/losses resolve in 50 games.

_EVAL_STAGES_2P = [
    # (cumulative_games, promote_threshold, reject_threshold)
    (50,  0.60, 0.42),
    (100, 0.56, 0.46),
    (200, 0.53, 0.48),
    (400, 0.501, 0.0),  # final: any improvement promotes
]
# Multiplayer evaluation: candidate plays 1 seat vs (N-1) copies of best.
# Random baseline is 1/N, not 50%.  Lower thresholds accordingly.
# 3-player: random WR ≈ 33%, so 42-45% is a meaningful improvement.
_EVAL_STAGES_3P = [
    (50,  0.45, 0.25),
    (100, 0.42, 0.28),
    (200, 0.39, 0.31),
    (400, 0.341, 0.0),  # beat random (33%) = real improvement in 1-vs-2
]
# 4-player: random WR ≈ 25%, so 35-38% is a meaningful improvement.
_EVAL_STAGES_4P = [
    (50,  0.45, 0.22),
    (100, 0.42, 0.25),
    (200, 0.38, 0.28),
    (400, 0.334, 0.0),  # beat random chance = promote
]

AUTO_PLATEAU_RELAX_PROMOTE_THRESHOLD = 0.52
AUTO_PLATEAU_RELAX_ITERATIONS = 3


def _get_eval_stages() -> list:
    if NUM_PLAYERS == 3:
        return _EVAL_STAGES_3P
    elif NUM_PLAYERS >= 4:
        return _EVAL_STAGES_4P
    return _EVAL_STAGES_2P


def _cap_promote_thresholds(eval_stages: list, cap: float | None) -> list:
    """Return eval stages with promotion thresholds capped for plateau recovery."""
    if cap is None:
        return eval_stages
    return [(games, min(promote_thr, cap), reject_thr)
            for games, promote_thr, reject_thr in eval_stages]


def staged_evaluate(
    cand: str, best: str, budget: int,
    *, tracker: "QualityGateTracker | None" = None,
    promote_threshold_cap: float | None = None,
    checkpoint_path: Path | None = None,
) -> dict:
    """Staged head-to-head evaluation with early exit.

    Plays games in batches. After each batch, checks if the result is
    decisive enough to promote or reject early. This saves GPU time on
    clear wins/losses while giving marginal improvements up to 400 games
    of evidence.

    When ``checkpoint_path`` is provided, per-game progress (candidate_wins,
    best_wins, draws, games_played, per-seat outcomes) is persisted after
    every game. On restart, the function resumes from that state rather than
    replaying the full stage. Tracker move-level data from pre-restart is
    not recovered — only game outcomes are replayed into the tracker — so
    per-seat WR remains accurate but policy/value diversity metrics will
    reflect only the post-resume portion.
    """
    eval_stages = _get_eval_stages()
    eval_stages = _cap_promote_thresholds(eval_stages, promote_threshold_cap)
    env = _make_env()
    cw, bw, dr = 0, 0, 0
    t0 = time.time()
    games_played = 0
    decision = None
    decision_stage = 0
    seat_outcomes: list[dict[str, Any]] = []

    if checkpoint_path is not None and checkpoint_path.exists():
        try:
            ckpt = json.loads(checkpoint_path.read_text())
            cw = int(ckpt.get("candidate_wins", 0))
            bw = int(ckpt.get("best_wins", 0))
            dr = int(ckpt.get("draws", 0))
            games_played = int(ckpt.get("games_played", 0))
            seat_outcomes = list(ckpt.get("seat_outcomes", []))
            logger.info(
                f"  eval resume: {games_played} games already played "
                f"(cand={cw} best={bw} draws={dr})"
            )
            if tracker is not None:
                for so in seat_outcomes:
                    try:
                        tracker.record_game_outcome(
                            int(so["i"]), int(so["candidate_player"]), bool(so["won"]),
                        )
                    except (KeyError, TypeError, ValueError):
                        continue
        except (OSError, json.JSONDecodeError, ValueError):
            logger.warning("  eval checkpoint corrupt, starting eval from scratch")
            cw = bw = dr = games_played = 0
            seat_outcomes = []

    def _save_eval_checkpoint() -> None:
        if checkpoint_path is None:
            return
        try:
            payload = {
                "candidate_wins": cw,
                "best_wins": bw,
                "draws": dr,
                "games_played": games_played,
                "seat_outcomes": seat_outcomes,
            }
            tmp = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
            tmp.write_text(json.dumps(payload))
            tmp.replace(checkpoint_path)
        except OSError:
            pass

    for stage_idx, (target_games, promote_thr, reject_thr) in enumerate(eval_stages):
        if games_played >= target_games:
            # This stage was fully covered by the resumed checkpoint.
            # Check its thresholds and continue to the next stage.
            dec = cw + bw
            wr = cw / dec if dec > 0 else 0.5
            if wr >= promote_thr:
                decision = "promote"
                decision_stage = stage_idx + 1
                break
            elif wr <= reject_thr:
                decision = "reject"
                decision_stage = stage_idx + 1
                break
            continue
        games_this_stage = target_games - games_played
        for i in range(games_played, target_games):
            gseed = 42_000 + i * 7919
            num_p = env.num_players if hasattr(env, "num_players") else 2
            ais = {}
            candidate_player = (i % num_p) + 1
            for p in range(1, num_p + 1):
                model = cand if p == candidate_player else best
                ais[p] = _make_ai(p, model, budget)
            for p, ai in ais.items():
                if hasattr(ai, "reset_for_new_game"):
                    ai.reset_for_new_game(rng_seed=(gseed + p * 97_911) & 0xFFFFFFFF)
            state, mc = env.reset(seed=gseed), 0
            cand_move_num = 0
            while state.game_status == GameStatus.ACTIVE and mc < MAX_MOVES:
                c = state.current_player
                if c not in ais:
                    break
                ais[c].player_number = c
                legal = env.legal_moves()
                if not legal:
                    break
                mv = ais[c].select_move(state)
                if mv is None:
                    break
                if mv not in legal:
                    mv = legal[random.randint(0, len(legal) - 1)]
                if tracker is not None and c == candidate_player:
                    root_value = None
                    stats = getattr(ais[c], "_last_search_stats", None)
                    if isinstance(stats, dict):
                        root_value = stats.get("root_value")
                    tracker.record_move(i, cand_move_num, mv, legal,
                                        root_value=root_value)
                    cand_move_num += 1
                state, _, done, _ = env.step(mv)
                mc += 1
                if done:
                    break
            if tracker is not None:
                tracker.finish_game(i)
            w = state.winner if state.game_status == GameStatus.COMPLETED else None
            if w is None:
                dr += 1
            elif w == candidate_player:
                cw += 1
            else:
                bw += 1
            won = (w == candidate_player)
            if tracker is not None:
                tracker.record_game_outcome(i, candidate_player, won)
            seat_outcomes.append({"i": i, "candidate_player": candidate_player, "won": won})
            games_played += 1
            _save_eval_checkpoint()

        # Check stage thresholds
        dec = cw + bw
        wr = cw / dec if dec > 0 else 0.5
        logger.info(f"  eval stage {stage_idx+1}: {games_played} games, "
                     f"cand={cw} best={bw} wr={wr:.1%}")

        if wr >= promote_thr:
            decision = "promote"
            decision_stage = stage_idx + 1
            break
        elif wr <= reject_thr:
            decision = "reject"
            decision_stage = stage_idx + 1
            break

    if decision is None:
        # Reached final stage without early exit
        decision = "reject"
        decision_stage = len(eval_stages)

    # Decision reached — drop the resume checkpoint so the next iteration
    # starts clean. Leave it on disk if we somehow exited without a decision
    # (should never happen, but fail-safe: file will be overwritten next run).
    if checkpoint_path is not None and decision is not None:
        try:
            checkpoint_path.unlink(missing_ok=True)
        except OSError:
            pass

    el = time.time() - t0
    dec = cw + bw
    wr = cw / dec if dec > 0 else 0.5
    logger.info(f"  eval done: cand {cw}-{bw} best (wr={wr:.1%}, "
                 f"{dr} draws, {el:.0f}s, stage {decision_stage}/{len(eval_stages)})")
    result = {
        "candidate_wins": cw, "best_wins": bw, "draws": dr,
        "win_rate": wr, "elapsed_s": el,
        "games_played": games_played,
        "decision": decision,
        "decision_stage": decision_stage,
    }
    if promote_threshold_cap is not None:
        result["promote_threshold_cap"] = promote_threshold_cap
    return result


S3_HEARTBEAT_BUCKET = "s3://ringrift-models-20251214/consolidated/heartbeats"


def _push_heartbeat_s3(
    node_id: str,
    config_key: str,
    iteration: int,
    elo: float,
    promos: int,
    data_quality_score: float | None = None,
    *,
    stage: str = "iteration_done",
    experiment_params: dict | None = None,
) -> None:
    """Push a <1KB heartbeat JSON to S3 for coordinator fleet monitoring.

    Best-effort: never blocks training on S3 failure.
    """
    heartbeat = {
        "node_id": node_id,
        "config_key": config_key,
        "iteration": iteration,
        "estimated_elo": elo,
        "promotions": promos,
        "timestamp": time.time(),
        "data_quality_score": data_quality_score,
        "stage": stage,
    }
    if experiment_params:
        heartbeat.update(experiment_params)
    tmp = None
    try:
        fd, tmp = tempfile.mkstemp(suffix=".json")
        with os.fdopen(fd, "w") as f:
            json.dump(heartbeat, f)
        subprocess.run(
            ["aws", "s3", "cp", tmp, f"{S3_HEARTBEAT_BUCKET}/{node_id}.json"],
            timeout=30,
            capture_output=True,
        )
    except Exception:
        pass  # Best-effort, don't block training
    finally:
        if tmp:
            try:
                os.unlink(tmp)
            except OSError:
                pass


def main() -> None:
    global BOARD_TYPE, BOARD_ENUM, NUM_PLAYERS, MODEL_VERSION, FEATURE_VERSION

    ap = argparse.ArgumentParser(description="Minimal AlphaZero loop")
    ap.add_argument("--model", required=True, help="Starting model checkpoint")
    ap.add_argument("--board-type", type=str, default="hex8",
                    choices=list(BOARD_TYPE_MAP.keys()),
                    help="Board type (default: hex8)")
    ap.add_argument("--num-players", type=int, default=2, choices=[2, 3, 4])
    ap.add_argument("--model-version", type=str, default="v2",
                    choices=["v2", "v3", "v4", "v5-heavy"],
                    help="Neural network architecture version (default: v2)")
    ap.add_argument("--feature-version", type=int, default=2, choices=[1, 2, 3],
                    help="Hex encoder feature version (default: 2; set 3 to disable "
                         "placement-validity shortcut for v5-heavy/v4 retries, see c790d339f)")
    ap.add_argument(
        "--profile",
        type=str,
        default="auto",
        choices=["auto", "standard", "large-board"],
        help="Loop preset profile (default: auto)",
    )
    ap.add_argument("--iterations", type=int, default=20)
    ap.add_argument("--games-per-iter", type=int, default=None)
    ap.add_argument("--eval-games", type=int, default=None)
    ap.add_argument("--budget", type=int, default=None,
                    help="MCTS sims for both selfplay and eval (default from profile)")
    ap.add_argument("--selfplay-budget", type=int, default=None,
                    help="MCTS sims for selfplay only (overrides --budget for selfplay)")
    ap.add_argument("--eval-budget", type=int, default=None,
                    help="MCTS sims for eval only (overrides --budget for eval)")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lr-schedule", type=str, default="sqrt_decay",
                    choices=["sqrt_decay", "fixed"],
                    help="Loop-level LR schedule: sqrt_decay (lr/sqrt(iter)) or fixed")
    ap.add_argument("--train-lr-scheduler", type=str, default="auto",
                    choices=["auto", "none", "cosine", "step", "plateau", "warmrestart"],
                    help="Epoch LR scheduler for app.training.train "
                         "(auto=none for fixed loop LR, cosine otherwise)")
    ap.add_argument("--lr-floor", type=float, default=1e-5,
                    help="Minimum LR for sqrt_decay schedule (default: 1e-5)")
    ap.add_argument("--policy-weight", type=float, default=None,
                    help="Optional policy loss weight passed to app.training.train")
    ap.add_argument("--value-weight", type=float, default=None,
                    help="Optional value loss weight passed to app.training.train")
    ap.add_argument("--rank-dist-weight", type=float, default=None,
                    help="Optional rank-distribution loss weight passed to app.training.train")
    ap.add_argument("--gradient-clip-max-norm", type=float, default=None,
                    help="Optional gradient clip max norm passed to app.training.train")
    ap.add_argument("--train-window", type=int, default=10,
                    help="Number of recent iterations to combine for training data (default: 10)")
    ap.add_argument("--promote-threshold", type=float, default=0.55)
    ap.add_argument("--selfplay-randomness", type=float, default=0.25,
                    help="Exploration noise for selfplay (0=deterministic, 0.25=default)")
    ap.add_argument("--work-dir", type=str, default="data/minimal_loop")
    ap.add_argument(
        "--supplemental-data-dir",
        type=str,
        default="",
        help="Optional directory of supplemental NPZ shards to merge at train time",
    )
    ap.add_argument("--log-file", type=str, default=None)
    ap.add_argument("--node-id", type=str, default=None,
                    help="Node identifier for fleet heartbeats (default: hostname)")
    ap.add_argument("--skip-quality-check", action="store_true",
                    help="Skip data quality sentinel check before training")
    ap.add_argument("--skip-probes", action="store_true",
                    help="Skip training effectiveness probes after training")
    ap.add_argument("--skip-quality-gate", action="store_true",
                    help="Skip model quality gate (behavioral diversity + value health) after evaluation")
    ap.add_argument(
        "--auto-plateau-relax",
        action="store_true",
        help=(
            "After PLATEAU_DETECTED, cap staged promotion thresholds at "
            "52% for the next 3 iterations"
        ),
    )
    ap.add_argument("--no-self-heal", action="store_true",
                    help="Disable automatic recovery on circuit breaker trips")
    args = ap.parse_args()

    node_id = args.node_id or socket.gethostname()

    # Cache git SHA once at startup (not per-heartbeat)
    try:
        _git_sha = subprocess.run(
            ["git", "-C", str(SCRIPT_DIR.parent), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5
        ).stdout.strip()
    except Exception:
        _git_sha = "unknown"

    def _resolve_train_lr_scheduler() -> str:
        if args.train_lr_scheduler != "auto":
            return args.train_lr_scheduler
        return "none" if args.lr_schedule == "fixed" else "cosine"

    # Set globals from CLI args
    BOARD_TYPE = args.board_type
    BOARD_ENUM = BOARD_TYPE_MAP[args.board_type]
    NUM_PLAYERS = args.num_players
    MODEL_VERSION = args.model_version
    FEATURE_VERSION = args.feature_version
    profile_info = resolve_loop_profile(
        BOARD_TYPE,
        NUM_PLAYERS,
        args.profile,
        games_per_iter=args.games_per_iter,
        eval_games=args.eval_games,
        budget=args.budget,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
    loop_settings = profile_info["settings"]
    config_key = profile_info["config_key"]
    games_per_iter = int(loop_settings["games_per_iter"])
    eval_games = int(loop_settings["eval_games"])
    budget = int(loop_settings["budget"])
    selfplay_budget = args.selfplay_budget or budget
    eval_budget = args.eval_budget or budget
    train_lr_scheduler = _resolve_train_lr_scheduler()
    epochs = int(loop_settings["epochs"])
    batch_size = int(loop_settings["batch_size"])
    train_window = args.train_window

    wdir = Path(args.work_dir)
    wdir.mkdir(parents=True, exist_ok=True)
    supplemental_data_dir = Path(args.supplemental_data_dir).expanduser() if args.supplemental_data_dir else None
    if supplemental_data_dir is not None:
        supplemental_data_dir.mkdir(parents=True, exist_ok=True)
    mdir = wdir / "models"
    mdir.mkdir(parents=True, exist_ok=True)

    best = mdir / "best.pth"
    if not best.exists():
        src = Path(args.model)
        if not src.exists():
            logger.error(f"Starting model not found: {src}")
            sys.exit(1)
        shutil.copy2(src, best)
        logger.info(f"Copied starting model -> {best}")

    logf = Path(args.log_file) if args.log_file else wdir / "metrics.jsonl"
    logf.parent.mkdir(parents=True, exist_ok=True)
    promos, elo = 0, INITIAL_ESTIMATED_ELO
    consec_failures = 0  # Circuit breaker: stop after repeated failures
    MAX_CONSEC_FAILURES = 3
    last_error = ""  # Captured for self-healing diagnostics
    last_error_stage = ""
    plateau_relax_until_iter = 0
    reset_recovery_counts()  # Fresh recovery budget for this loop run

    # Cached metrics history: read once at startup, appended to in-memory on
    # each iteration.  Fixes #84: previously the plateau detector re-read the
    # full metrics.jsonl every iteration, which is O(N^2) and has a race with
    # its own just-completed append.  Keeping history in-memory is both
    # cheaper and correct-by-construction.
    metrics_history: list[dict] = []
    last_metrics_iter = 0
    logged_promos = promos
    logged_elo = elo
    if logf.exists():
        for line in logf.read_text().strip().split("\n"):
            if not line.strip():
                continue
            try:
                m = json.loads(line)
            except (json.JSONDecodeError, ValueError):
                # Partial/corrupt line — tolerate and continue; resume is best-effort.
                continue
            metrics_history.append(m)
            it_val = m.get("iteration")
            if isinstance(it_val, int):
                last_metrics_iter = max(last_metrics_iter, it_val)
            total_promotions = m.get("total_promotions")
            if isinstance(total_promotions, int):
                logged_promos = total_promotions
            estimated_elo = m.get("estimated_elo")
            if isinstance(estimated_elo, (int, float)):
                logged_elo = float(estimated_elo)

    promos, elo = _recompute_progress_from_metrics(metrics_history, NUM_PLAYERS)
    if metrics_history and (
        promos != logged_promos or not math.isclose(elo, logged_elo, rel_tol=0.0, abs_tol=0.05)
    ):
        logger.info(
            "Recomputed resume Elo from promotion history "
            "(logged elo=%.1f -> corrected elo=%.1f, logged promos=%d -> corrected promos=%d)",
            logged_elo,
            elo,
            logged_promos,
            promos,
        )

    # Resume: the metrics.jsonl line is appended only after training + eval
    # finish, so last_metrics_iter is the authoritative "fully completed"
    # marker. Falling back to the npz count only for pre-existing runs that
    # lack a metrics log. Using npz count alone was wrong because iter_N.npz
    # is written after training but before eval completes — so an eval
    # interrupted by SIGTERM would have been silently skipped on restart.
    if last_metrics_iter > 0:
        start_iter = last_metrics_iter + 1
    else:
        existing_npz = sorted(wdir.glob("iter_*.npz"))
        start_iter = len(existing_npz) + 1 if existing_npz else 1
    if start_iter > 1:
        logger.info(f"Resuming from iteration {start_iter} (elo={elo:.0f}, promos={promos})")

    logger.info("=" * 70)
    logger.info("MINIMAL ALPHAZERO LOOP")
    logger.info(f"  board={BOARD_TYPE} {NUM_PLAYERS}p | model={args.model}")
    logger.info(f"  profile={profile_info['profile']} config={profile_info['config_key']}")
    logger.info(f"  iters={args.iterations} games={games_per_iter} eval={eval_games}")
    logger.info(f"  selfplay_budget={selfplay_budget} eval_budget={eval_budget}")
    logger.info(
        f"  epochs={epochs} bs={batch_size} lr={args.lr} "
        f"lr_schedule={args.lr_schedule} train_lr_scheduler={train_lr_scheduler} "
        f"lr_floor={args.lr_floor}"
    )
    logger.info(f"  train_window={train_window} promote_thr={args.promote_threshold:.0%} work_dir={wdir}")
    if supplemental_data_dir is not None:
        logger.info(f"  supplemental_data_dir={supplemental_data_dir}")
    logger.info(f"  selfplay_randomness={args.selfplay_randomness}")
    transfer_hint = recommend_transfer_source(BOARD_TYPE, NUM_PLAYERS)
    if transfer_hint:
        logger.info(
            "  bootstrap_hint=%s (recommended same-board transfer init for weak/slow large-board configs)",
            transfer_hint,
        )
    logger.info("=" * 70)

    # Static experiment params — shared across heartbeats and metrics.
    # effective_lr is updated per iteration (inside loop).
    _static_exp_params = {
        "selfplay_budget": selfplay_budget,
        "eval_budget": eval_budget,
        "base_lr": args.lr,
        "lr_schedule": args.lr_schedule,
        "train_lr_scheduler": train_lr_scheduler,
        "policy_weight": args.policy_weight,
        "value_weight": args.value_weight,
        "rank_dist_weight": args.rank_dist_weight,
        "gradient_clip_max_norm": args.gradient_clip_max_norm,
        "lr_floor": args.lr_floor,
        "train_window": train_window,
        "supplemental_data_dir": str(supplemental_data_dir) if supplemental_data_dir is not None else "",
        "git_sha": _git_sha,
    }

    for it in range(start_iter, start_iter + args.iterations):
        it0 = time.time()
        logger.info(f"\n{'='*70}\nITERATION {it}/{args.iterations}\n{'='*70}")
        jpath = wdir / f"iter_{it:03d}.jsonl"
        npath = wdir / f"iter_{it:03d}.npz"
        cpath = mdir / f"candidate_{it:03d}.pth"

        # Circuit breaker: stop wasting GPU on repeated failures
        if consec_failures >= MAX_CONSEC_FAILURES:
            if not args.no_self_heal:
                recovery = attempt_recovery(FailureContext(
                    error_message=last_error,
                    stage=last_error_stage,
                    config_key=config_key,
                    work_dir=str(wdir),
                    model_path=str(best),
                    batch_size=batch_size,
                    selfplay_randomness=args.selfplay_randomness,
                    model_version=MODEL_VERSION,
                ))
                if recovery.recovered:
                    logger.info(f"AUTO-RECOVERY: {recovery.action} - {recovery.message}")
                    # Apply adjustments from recovery
                    if "batch_size" in recovery.adjustments:
                        batch_size = recovery.adjustments["batch_size"]
                    if "selfplay_randomness" in recovery.adjustments:
                        args.selfplay_randomness = recovery.adjustments["selfplay_randomness"]
                    consec_failures = 0
                    continue
                logger.error(f"CIRCUIT BREAKER: {consec_failures} consecutive failures, "
                             f"recovery failed: {recovery.message}")
            else:
                logger.error(f"CIRCUIT BREAKER: {consec_failures} consecutive failures. "
                             f"Stopping to avoid wasting GPU. Fix the issue and restart.")
            break

        # Write progress at iteration start
        try:
            (wdir / "progress.json").write_text(json.dumps({
                "iteration": it, "stage": "selfplay_started",
                "estimated_elo": round(elo, 1),
                "total_promotions": promos,
                "games_target": games_per_iter,
                "selfplay_budget": selfplay_budget,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }, indent=2) + "\n")
        except OSError:
            pass

        # 1. SELFPLAY
        logger.info(f"[1/5] Selfplay: {games_per_iter} games, budget={selfplay_budget}, "
                     f"randomness={args.selfplay_randomness}")
        sp = run_selfplay(str(best), games_per_iter, jpath, selfplay_budget,
                          randomness=args.selfplay_randomness)
        if sp["completed"] == 0:
            logger.error("No games completed, skipping")
            last_error = "No games completed in selfplay"
            last_error_stage = "selfplay"
            consec_failures += 1; continue
        selfplay_seat_wins = {
            seat: int(sp.get(f"p{seat}_wins", 0))
            for seat in range(1, NUM_PLAYERS + 1)
        }

        _push_heartbeat_s3(node_id, config_key, it, elo, promos,
                           stage="selfplay_done", experiment_params=_static_exp_params)

        # Update progress file after selfplay
        try:
            (wdir / "progress.json").write_text(json.dumps({
                "iteration": it, "stage": "selfplay_done",
                "selfplay_games": sp["completed"],
                "selfplay_elapsed_s": round(sp["elapsed_s"], 1),
                "estimated_elo": round(elo, 1),
                "total_promotions": promos,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }, indent=2) + "\n")
        except OSError:
            pass

        # 2. EXPORT
        logger.info("[2/5] Export JSONL -> NPZ")
        if not export_npz(jpath, npath):
            logger.error("Export failed, skipping")
            last_error = "Export JSONL to NPZ failed"
            last_error_stage = "export"
            consec_failures += 1; continue

        # 2.5 DATA QUALITY CHECK
        if not args.skip_quality_check:
            try:
                from scripts.lib.data_quality_sentinel import check_data_quality

                verdict = check_data_quality(str(npath), work_dir=str(wdir))
                if verdict.critical:
                    logger.error(f"DATA QUALITY CRITICAL: {verdict.summary}")
                    last_error = verdict.summary
                    last_error_stage = "data_quality"
                    consec_failures += 1; continue
                elif verdict.warnings:
                    logger.warning(f"Data quality: {verdict.summary}")
            except Exception as e:
                logger.warning(f"DQS check failed (non-fatal): {e}")

        # 3. TRAIN (using sliding window of recent NPZ files for more data)
        # Accumulate data from recent iterations — much better than single-iteration training.
        # NPZ files are trusted local training data generated by this script.
        recent_npz = sorted(wdir.glob("iter_*.npz"))[-train_window:]
        recent_supplemental = (
            sorted(supplemental_data_dir.glob("*.npz"))[-train_window:]
            if supplemental_data_dir is not None and supplemental_data_dir.exists()
            else []
        )
        merge_inputs = [*recent_npz, *recent_supplemental]
        if len(merge_inputs) > 1:
            combined = wdir / f"combined_{it:03d}.npz"
            try:
                combined_result = _combine_npz_files(merge_inputs, combined)
                if combined_result is None:
                    train_npz = npath
                else:
                    train_npz, n_samples = combined_result
                    logger.info(
                        "  combined %s local + %s supplemental NPZ files -> %s samples",
                        len(recent_npz),
                        len(recent_supplemental),
                        n_samples,
                    )
            except Exception as e:
                logger.warning(f"  NPZ merge failed ({e}), using single iteration")
                train_npz = npath
        else:
            train_npz = npath
        # Iteration-aware LR decay: prevents catastrophic forgetting in later
        # iterations when fine-tuning from increasingly strong checkpoints.
        # Without this, small models (square8 ~3.8M params) overfit at lr=1e-4,
        # producing candidates with lower val_loss but WORSE play strength.
        if args.lr_schedule == "fixed":
            effective_lr = args.lr
        else:
            effective_lr = max(args.lr_floor, args.lr / math.sqrt(max(1, it)))
        logger.info(
            f"[3/5] Train (epochs={epochs}, bs={batch_size}, lr={effective_lr:.1e}, "
            f"scheduler={train_lr_scheduler})"
        )
        target_stats = _summarize_training_targets(train_npz)
        ti = train_model(
            train_npz,
            cpath,
            best,
            epochs,
            batch_size,
            effective_lr,
            train_lr_scheduler,
            policy_weight=args.policy_weight,
            value_weight=args.value_weight,
            rank_dist_weight=args.rank_dist_weight,
            gradient_clip_max_norm=args.gradient_clip_max_norm,
        )
        ti["target_stats"] = target_stats
        if "error" in ti or not cpath.exists():
            logger.error("Training failed, skipping")
            last_error = ti.get("error", "") or "Training produced no output"
            last_error_stage = "training"
            consec_failures += 1; continue

        training_exp_params = {**_static_exp_params, "effective_lr": effective_lr}

        # Training succeeded — reset circuit breaker
        consec_failures = 0
        _push_heartbeat_s3(node_id, config_key, it, elo, promos,
                           stage="training_done", experiment_params=training_exp_params)

        # Update progress file after training
        try:
            (wdir / "progress.json").write_text(json.dumps({
                "iteration": it, "stage": "training_done",
                "training_elapsed_s": round(ti.get("elapsed_s", 0), 1),
                "estimated_elo": round(elo, 1),
                "total_promotions": promos,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }, indent=2) + "\n")
        except OSError:
            pass

        # 3.5 PROBE: Verify training actually worked
        if not args.skip_probes:
            probe = run_training_probes(
                str(cpath), str(best), ti, BOARD_ENUM, NUM_PLAYERS, eval_budget,
                model_version=MODEL_VERSION,
                feature_version=FEATURE_VERSION,
            )
            if probe.critical:
                logger.error(f"TRAINING PROBE FAILED: {probe.summary}")
                logger.error("TRAINING PROBE DETAILS: %s", json.dumps(probe.details, sort_keys=True))
                last_error = probe.summary
                last_error_stage = "probe"
                consec_failures += 1; continue
            elif probe.warnings:
                logger.warning(f"Training probe warnings: {probe.summary}")
                logger.warning("Training probe details: %s", json.dumps(probe.details, sort_keys=True))
            else:
                logger.info(f"  probes passed ({probe.elapsed_s:.1f}s)")

        # 4. EVALUATE — staged evaluation with early exit for clear wins/losses
        logger.info(f"[4/5] Evaluate (staged, up to 400 games, budget={eval_budget})")

        # Update progress before evaluation (the longest stage)
        try:
            (wdir / "progress.json").write_text(json.dumps({
                "iteration": it, "stage": "evaluation_started",
                "estimated_elo": round(elo, 1),
                "total_promotions": promos,
                "eval_budget": eval_budget,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }, indent=2) + "\n")
        except OSError:
            pass
        qg_tracker = None if args.skip_quality_gate else QualityGateTracker()
        if qg_tracker is not None:
            qg_tracker.set_selfplay_baseline(selfplay_seat_wins)
        relax_active = args.auto_plateau_relax and it <= plateau_relax_until_iter
        promote_threshold_cap = (
            AUTO_PLATEAU_RELAX_PROMOTE_THRESHOLD if relax_active else None
        )
        if relax_active:
            logger.warning(
                "PLATEAU_RELAX_ACTIVE config=%s iter=%d through_iter=%d "
                "promote_threshold_cap=%.0f%%",
                config_key,
                it,
                plateau_relax_until_iter,
                AUTO_PLATEAU_RELAX_PROMOTE_THRESHOLD * 100,
            )
        eval_ckpt_path = wdir / f"iter_{it:03d}_eval.json"
        ev = staged_evaluate(str(cpath), str(best), eval_budget,
                             tracker=qg_tracker,
                             promote_threshold_cap=promote_threshold_cap,
                             checkpoint_path=eval_ckpt_path)

        # 4.5 MODEL QUALITY GATE — reject degenerate candidates before promotion
        quality_blocked = False
        quality_gate_record: dict[str, Any] | None = None
        if qg_tracker is not None:
            quality = check_model_quality(qg_tracker)
            if quality.critical:
                logger.error(f"MODEL QUALITY GATE: {quality.summary}")
                quality_blocked = True
            elif quality.warnings:
                logger.warning(f"Quality gate warnings: {quality.summary}")
            else:
                logger.info(f"  quality gate passed")
            # Fold the full verdict into metrics so downstream consumers
            # (refresh_experiment_status.py, plateau detector, eval evidence
            # dashboards) can see per-seat WR and other diagnostics without
            # having to re-derive them from logs.  Attempt JSON round-trip to
            # guard against any non-serializable debug payload leaking in.
            try:
                raw_record = {
                    "passed": bool(quality.passed),
                    "critical": bool(quality.critical),
                    "warnings": list(quality.warnings),
                    "summary": str(quality.summary),
                    "details": dict(quality.details),
                }
                quality_gate_record = json.loads(json.dumps(raw_record, default=str))
            except (TypeError, ValueError) as exc:
                logger.debug("quality gate serialization skipped: %s", exc)
                quality_gate_record = {
                    "passed": bool(quality.passed),
                    "critical": bool(quality.critical),
                    "warnings": list(quality.warnings),
                }

        # 5. PROMOTE / REJECT — decision comes from staged evaluation
        wr = ev["win_rate"]
        promoted = ev.get("decision") == "promote" and not quality_blocked
        if promoted:
            shutil.copy2(cpath, best)
            promos += 1
            eg = _promotion_elo_delta(wr, NUM_PLAYERS)
            elo += eg
            logger.info(f"[5/5] PROMOTED (wr={wr:.1%}, +{eg:.0f} -> ~{elo:.0f} Elo)")
        else:
            logger.info(f"[5/5] REJECTED (wr={wr:.1%}, need {args.promote_threshold:.0%})")

        iel = time.time() - it0
        # Experiment params for metrics and heartbeats
        exp_params = training_exp_params
        metrics = {
            "iteration": it, "timestamp": datetime.now(timezone.utc).isoformat(),
            "selfplay": sp, "training": {k: v for k, v in ti.items() if k != "log_line"},
            "evaluation": ev, "promoted": promoted, "estimated_elo": round(elo, 1),
            "total_promotions": promos, "iteration_time_s": round(iel, 1),
            **exp_params,
        }
        if quality_gate_record is not None:
            metrics["quality_gate"] = quality_gate_record

        # Plateau detection (A2 / plan #79). Diagnostic by default; when the
        # explicit opt-in flag is set, a detected plateau arms relaxed staged
        # promotion thresholds for the next three iterations.
        # #84 fix: use the in-memory metrics_history list rather than
        # re-reading metrics.jsonl from disk — same detection behaviour but
        # O(1) per iteration and no partial-read race with the append below.
        if it % 10 == 0:
            try:
                plateau = detect_plateau([*metrics_history, metrics])
                metrics["plateau"] = {
                    "detected": plateau.detected,
                    "recent_rejection_rate": plateau.recent_rejection_rate,
                    "iterations_since_promotion": plateau.iterations_since_promotion,
                    "window_size": plateau.window_size,
                    "total_iterations": plateau.total_iterations,
                    "last_promoted_iteration": plateau.last_promoted_iteration,
                    "reason": plateau.reason,
                    "auto_relax_enabled": args.auto_plateau_relax,
                }
                if plateau.detected:
                    logger.warning(
                        "%s config=%s iter=%d last_promoted=%s total_iters=%d",
                        plateau.reason,
                        config_key,
                        it,
                        plateau.last_promoted_iteration,
                        plateau.total_iterations,
                    )
                    if args.auto_plateau_relax:
                        plateau_relax_until_iter = max(
                            plateau_relax_until_iter,
                            it + AUTO_PLATEAU_RELAX_ITERATIONS,
                        )
                        metrics["plateau"]["relax_until_iteration"] = plateau_relax_until_iter
                        logger.warning(
                            "PLATEAU_RELAX_ARMED config=%s active_iters=%d-%d "
                            "promote_threshold_cap=%.0f%%",
                            config_key,
                            it + 1,
                            plateau_relax_until_iter,
                            AUTO_PLATEAU_RELAX_PROMOTE_THRESHOLD * 100,
                        )
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                logger.debug("plateau detector skipped: %s", exc)

        with open(logf, "a") as f:
            f.write(json.dumps(metrics) + "\n")

        # Keep the in-memory metrics cache in lockstep with metrics.jsonl so
        # the next plateau detection (every 10 iterations) sees this entry
        # without a disk re-read.
        metrics_history.append(metrics)

        # Write a human-readable progress file so observers don't need to parse JSONL
        try:
            progress = wdir / "progress.json"
            progress.write_text(json.dumps({
                "iteration": it,
                "stage": "complete",
                "estimated_elo": round(elo, 1),
                "total_promotions": promos,
                "last_decision": ev.get("decision"),
                "last_win_rate": ev.get("win_rate"),
                "iteration_time_s": round(iel, 1),
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }, indent=2) + "\n")
        except OSError:
            pass
        logger.info(f"  iter {it} done in {iel/60:.1f}min | elo~{elo:.0f} | promos={promos}/{it}")

        # Push S3 heartbeat for fleet health monitoring (best-effort)
        _push_heartbeat_s3(node_id, config_key, it, elo, promos,
                           experiment_params=exp_params)

        try:
            jpath.unlink()  # cleanup JSONL, keep NPZ
        except OSError:
            pass

    logger.info(f"\n{'='*70}\nLOOP COMPLETE\n{'='*70}")
    logger.info(f"  iterations={args.iterations} promotions={promos} elo~{elo:.0f}")
    logger.info(f"  best_model={best}  metrics={logf}")


if __name__ == "__main__":
    main()
