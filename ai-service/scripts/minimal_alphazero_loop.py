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
from scripts.lib.training_probes import run_training_probes

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger("minimal_alphazero")

# Defaults — overridden by --board-type and --num-players CLI args
BOARD_TYPE = "hex8"
BOARD_ENUM = BoardType.HEX8
NUM_PLAYERS = 2
MODEL_VERSION = "v2"
MAX_MOVES = 800

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
                   allow_fresh_weights=False, use_gpu_tree=True)
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
            "timestamp": datetime.now(timezone.utc).isoformat()}


def run_selfplay(model_path: str, n_games: int, out: Path, budget: int,
                  randomness: float = 0.25) -> dict:
    env = _make_env()
    # Use exploration noise (randomness > 0) for training data diversity
    ais = {p: _make_ai(p, model_path, budget, randomness=randomness)
           for p in range(1, NUM_PLAYERS + 1)}
    out.parent.mkdir(parents=True, exist_ok=True)
    # Use os.urandom for entropy instead of Python random (which may share state)
    seed = int.from_bytes(os.urandom(4), "big")
    logger.info(f"  selfplay seed={seed}")
    wins = {p: 0 for p in range(1, NUM_PLAYERS + 1)}
    done_n, t0 = 0, time.time()
    with open(out, "w") as f:
        for i in range(n_games):
            try:
                g = _play_game(env, ais, i, seed)
                f.write(json.dumps(g) + "\n")
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
    result: dict[str, Any] = {"completed": done_n, "elapsed_s": el}
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
    cmd = [sys.executable, str(SCRIPT_DIR / "jsonl_to_npz.py"),
           "--input", str(jsonl), "--output", str(npz),
           "--board-type", BOARD_TYPE, "--num-players", str(NUM_PLAYERS),
           "--gpu-selfplay"]
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
        try:
            from app.training.board_encoding_contract import get_expected_channels
            expected = get_expected_channels(BOARD_ENUM)
            if n_channels != expected:
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


def train_model(
    npz: Path,
    out: Path,
    init: Path,
    epochs: int,
    bs: int,
    lr: float,
    train_lr_scheduler: str,
) -> dict:
    """Train candidate. Retries with halved batch on OOM."""
    for attempt in range(3):
        b = bs // (2 ** attempt)
        cmd = [sys.executable, "-m", "app.training.train",
               "--data-path", str(npz), "--save-path", str(out),
               "--board-type", BOARD_TYPE, "--num-players", str(NUM_PLAYERS),
               "--epochs", str(epochs),
               "--batch-size", str(b), "--learning-rate", str(lr),
               "--init-weights", str(init), "--no-auto-tune-batch-size",
               "--lr-scheduler", train_lr_scheduler, "--skip-freshness-check",
               "--sampling-weights", "uniform"]
        logger.info(
            f"  training epochs={epochs} bs={b} lr={lr} "
            f"scheduler={train_lr_scheduler} (attempt {attempt+1})"
        )
        t0 = time.time()
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
        el = time.time() - t0
        if r.returncode == 0:
            info: dict[str, Any] = {"elapsed_s": el, "batch_size": b}
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
    (50,  0.50, 0.30),
    (100, 0.47, 0.33),
    (200, 0.44, 0.36),
    (400, 0.401, 0.0),
]
# 4-player: random WR ≈ 25%, so 35-38% is a meaningful improvement.
_EVAL_STAGES_4P = [
    (50,  0.45, 0.22),
    (100, 0.42, 0.25),
    (200, 0.38, 0.28),
    (400, 0.334, 0.0),  # beat random chance = promote
]

def _get_eval_stages() -> list:
    if NUM_PLAYERS == 3:
        return _EVAL_STAGES_3P
    elif NUM_PLAYERS >= 4:
        return _EVAL_STAGES_4P
    return _EVAL_STAGES_2P


def staged_evaluate(
    cand: str, best: str, budget: int,
    *, tracker: "QualityGateTracker | None" = None,
) -> dict:
    """Staged head-to-head evaluation with early exit.

    Plays games in batches. After each batch, checks if the result is
    decisive enough to promote or reject early. This saves GPU time on
    clear wins/losses while giving marginal improvements up to 400 games
    of evidence.
    """
    eval_stages = _get_eval_stages()
    env = _make_env()
    cw, bw, dr = 0, 0, 0
    t0 = time.time()
    games_played = 0
    decision = None
    decision_stage = 0

    for stage_idx, (target_games, promote_thr, reject_thr) in enumerate(eval_stages):
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
            games_played += 1

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

    el = time.time() - t0
    dec = cw + bw
    wr = cw / dec if dec > 0 else 0.5
    logger.info(f"  eval done: cand {cw}-{bw} best (wr={wr:.1%}, "
                 f"{dr} draws, {el:.0f}s, stage {decision_stage}/{len(eval_stages)})")
    return {
        "candidate_wins": cw, "best_wins": bw, "draws": dr,
        "win_rate": wr, "elapsed_s": el,
        "games_played": games_played,
        "decision": decision,
        "decision_stage": decision_stage,
    }


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
    global BOARD_TYPE, BOARD_ENUM, NUM_PLAYERS

    ap = argparse.ArgumentParser(description="Minimal AlphaZero loop")
    ap.add_argument("--model", required=True, help="Starting model checkpoint")
    ap.add_argument("--board-type", type=str, default="hex8",
                    choices=list(BOARD_TYPE_MAP.keys()),
                    help="Board type (default: hex8)")
    ap.add_argument("--num-players", type=int, default=2, choices=[2, 3, 4])
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
    promos, elo = 0, 1500.0
    consec_failures = 0  # Circuit breaker: stop after repeated failures
    MAX_CONSEC_FAILURES = 3
    last_error = ""  # Captured for self-healing diagnostics
    last_error_stage = ""
    reset_recovery_counts()  # Fresh recovery budget for this loop run

    # Resume: find the last completed iteration to avoid overwriting data
    existing = sorted(wdir.glob("iter_*.npz"))
    start_iter = len(existing) + 1 if existing else 1
    if start_iter > 1:
        # Reload state from metrics
        if logf.exists():
            for line in logf.read_text().strip().split("\n"):
                try:
                    m = json.loads(line)
                    promos = m.get("total_promotions", promos)
                    elo = m.get("estimated_elo", elo)
                except Exception:
                    pass
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

        _push_heartbeat_s3(node_id, config_key, it, elo, promos,
                           stage="selfplay_done", experiment_params=_static_exp_params)

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
        ti = train_model(
            train_npz,
            cpath,
            best,
            epochs,
            batch_size,
            effective_lr,
            train_lr_scheduler,
        )
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

        # 3.5 PROBE: Verify training actually worked
        if not args.skip_probes:
            probe = run_training_probes(
                str(cpath), str(best), ti, BOARD_ENUM, NUM_PLAYERS, eval_budget,
            )
            if probe.critical:
                logger.error(f"TRAINING PROBE FAILED: {probe.summary}")
                last_error = probe.summary
                last_error_stage = "probe"
                consec_failures += 1; continue
            elif probe.warnings:
                logger.warning(f"Training probe warnings: {probe.summary}")
            else:
                logger.info(f"  probes passed ({probe.elapsed_s:.1f}s)")

        # 4. EVALUATE — staged evaluation with early exit for clear wins/losses
        logger.info(f"[4/5] Evaluate (staged, up to 400 games, budget={eval_budget})")
        qg_tracker = None if args.skip_quality_gate else QualityGateTracker()
        ev = staged_evaluate(str(cpath), str(best), eval_budget,
                             tracker=qg_tracker)

        # 4.5 MODEL QUALITY GATE — reject degenerate candidates before promotion
        quality_blocked = False
        if qg_tracker is not None:
            quality = check_model_quality(qg_tracker)
            if quality.critical:
                logger.error(f"MODEL QUALITY GATE: {quality.summary}")
                quality_blocked = True
            elif quality.warnings:
                logger.warning(f"Quality gate warnings: {quality.summary}")
            else:
                logger.info(f"  quality gate passed")

        # 5. PROMOTE / REJECT — decision comes from staged evaluation
        wr = ev["win_rate"]
        promoted = ev.get("decision") == "promote" and not quality_blocked
        if promoted:
            shutil.copy2(cpath, best)
            promos += 1
            eg = 400.0 * math.log10(wr / (1 - wr)) if 0 < wr < 1 else 0
            elo += eg
            logger.info(f"[5/5] PROMOTED (wr={wr:.1%}, +{eg:.0f} -> ~{elo:.0f} Elo)")
        else:
            logger.info(f"[5/5] REJECTED (wr={wr:.1%}, need {args.promote_threshold:.0%})")

        iel = time.time() - it0
        # Experiment params for metrics and heartbeats
        exp_params = training_exp_params
        metrics = {"iteration": it, "timestamp": datetime.now(timezone.utc).isoformat(),
                   "selfplay": sp, "training": {k: v for k, v in ti.items() if k != "log_line"},
                   "evaluation": ev, "promoted": promoted, "estimated_elo": round(elo, 1),
                   "total_promotions": promos, "iteration_time_s": round(iel, 1),
                   **exp_params}
        with open(logf, "a") as f:
            f.write(json.dumps(metrics) + "\n")
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
