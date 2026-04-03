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

import argparse, json, logging, math, os, random, shutil, subprocess
import sys, time, uuid
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
        fsum = float(d["features"].sum())
        logger.info(f"  exported {len(d['features'])} samples (checksum={fsum:.1f})")
    except Exception:
        pass
    return True


def train_model(npz: Path, out: Path, init: Path,
                epochs: int, bs: int, lr: float) -> dict:
    """Train candidate. Retries with halved batch on OOM."""
    for attempt in range(3):
        b = bs // (2 ** attempt)
        cmd = [sys.executable, "-m", "app.training.train",
               "--data-path", str(npz), "--save-path", str(out),
               "--board-type", BOARD_TYPE, "--num-players", str(NUM_PLAYERS),
               "--epochs", str(epochs),
               "--batch-size", str(b), "--learning-rate", str(lr),
               "--init-weights", str(init), "--no-auto-tune-batch-size",
               "--lr-scheduler", "cosine", "--skip-freshness-check",
               "--sampling-weights", "uniform"]
        logger.info(f"  training epochs={epochs} bs={b} lr={lr} (attempt {attempt+1})")
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
        return {}
    logger.error("  training failed after 3 OOM retries")
    return {}


def evaluate(cand: str, best: str, n_games: int, budget: int) -> dict:
    """Head-to-head: candidate vs best, alternating colors."""
    env = _make_env()
    cw, bw, dr, t0 = 0, 0, 0, time.time()
    for i in range(n_games):
        gseed = 42_000 + i * 7919
        if i % 2 == 0:
            p1m, p2m = cand, best
        else:
            p1m, p2m = best, cand
        ais = {1: _make_ai(1, p1m, budget), 2: _make_ai(2, p2m, budget)}
        for p, ai in ais.items():
            if hasattr(ai, "reset_for_new_game"):
                ai.reset_for_new_game(rng_seed=(gseed + p * 97_911) & 0xFFFFFFFF)
        state, mc = env.reset(seed=gseed), 0
        while state.game_status == GameStatus.ACTIVE and mc < MAX_MOVES:
            c = state.current_player
            ais[c].player_number = c
            legal = env.legal_moves()
            if not legal:
                break
            mv = ais[c].select_move(state)
            if mv is None:
                break
            if mv not in legal:
                mv = legal[random.randint(0, len(legal) - 1)]
            state, _, done, _ = env.step(mv)
            mc += 1
            if done:
                break
        w = state.winner if state.game_status == GameStatus.COMPLETED else None
        if w is None:
            dr += 1
        elif (i % 2 == 0 and w == 1) or (i % 2 == 1 and w == 2):
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


def main() -> None:
    global BOARD_TYPE, BOARD_ENUM, NUM_PLAYERS

    ap = argparse.ArgumentParser(description="Minimal AlphaZero loop")
    ap.add_argument("--model", required=True, help="Starting model checkpoint")
    ap.add_argument("--board-type", type=str, default="hex8",
                    choices=list(BOARD_TYPE_MAP.keys()),
                    help="Board type (default: hex8)")
    ap.add_argument("--num-players", type=int, default=2, choices=[2, 3, 4])
    ap.add_argument("--iterations", type=int, default=20)
    ap.add_argument("--games-per-iter", type=int, default=300)
    ap.add_argument("--eval-games", type=int, default=100)
    ap.add_argument("--budget", type=int, default=128, help="MCTS sims (selfplay+eval)")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--promote-threshold", type=float, default=0.55)
    ap.add_argument("--selfplay-randomness", type=float, default=0.25,
                    help="Exploration noise for selfplay (0=deterministic, 0.25=default)")
    ap.add_argument("--work-dir", type=str, default="data/minimal_loop")
    ap.add_argument("--log-file", type=str, default=None)
    args = ap.parse_args()

    # Set globals from CLI args
    BOARD_TYPE = args.board_type
    BOARD_ENUM = BOARD_TYPE_MAP[args.board_type]
    NUM_PLAYERS = args.num_players

    wdir = Path(args.work_dir)
    wdir.mkdir(parents=True, exist_ok=True)
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
    logger.info(f"  iters={args.iterations} games={args.games_per_iter} eval={args.eval_games}")
    logger.info(f"  budget={args.budget} epochs={args.epochs} bs={args.batch_size} lr={args.lr}")
    logger.info(f"  promote_thr={args.promote_threshold:.0%} work_dir={wdir}")
    logger.info(f"  selfplay_randomness={args.selfplay_randomness}")
    logger.info("=" * 70)

    for it in range(start_iter, start_iter + args.iterations):
        it0 = time.time()
        logger.info(f"\n{'='*70}\nITERATION {it}/{args.iterations}\n{'='*70}")
        jpath = wdir / f"iter_{it:03d}.jsonl"
        npath = wdir / f"iter_{it:03d}.npz"
        cpath = mdir / f"candidate_{it:03d}.pth"

        # Circuit breaker: stop wasting GPU on repeated failures
        if consec_failures >= MAX_CONSEC_FAILURES:
            logger.error(f"CIRCUIT BREAKER: {consec_failures} consecutive failures. "
                         f"Stopping to avoid wasting GPU. Fix the issue and restart.")
            break

        # 1. SELFPLAY
        logger.info(f"[1/5] Selfplay: {args.games_per_iter} games, budget={args.budget}, "
                     f"randomness={args.selfplay_randomness}")
        sp = run_selfplay(str(best), args.games_per_iter, jpath, args.budget,
                          randomness=args.selfplay_randomness)
        if sp["completed"] == 0:
            logger.error("No games completed, skipping")
            consec_failures += 1; continue

        # 2. EXPORT
        logger.info("[2/5] Export JSONL -> NPZ")
        if not export_npz(jpath, npath):
            logger.error("Export failed, skipping")
            consec_failures += 1; continue

        # 3. TRAIN (using sliding window of recent NPZ files for more data)
        # Accumulate data from recent iterations — much better than single-iteration training.
        # NPZ files are trusted local training data generated by this script.
        recent_npz = sorted(wdir.glob("iter_*.npz"))[-10:]  # last 10 iterations
        if len(recent_npz) > 1:
            combined = wdir / f"combined_{it:03d}.npz"
            try:
                arrays: dict[str, list] = {}
                for npf in recent_npz:
                    d = np.load(str(npf), allow_pickle=True)  # trusted local training data
                    for k in d.files:
                        arrays.setdefault(k, []).append(d[k])
                merged = {k: np.concatenate(v) for k, v in arrays.items()
                          if all(hasattr(a, "shape") and len(a.shape) > 0 for a in v)}
                for k in d.files:
                    if k not in merged:
                        merged[k] = d[k]
                np.savez_compressed(str(combined), **merged)
                train_npz = combined
                n_samples = len(merged.get("features", []))
                logger.info(f"  combined {len(recent_npz)} NPZ files -> {n_samples} samples")
            except Exception as e:
                logger.warning(f"  NPZ merge failed ({e}), using single iteration")
                train_npz = npath
        else:
            train_npz = npath
        logger.info(f"[3/5] Train (epochs={args.epochs}, bs={args.batch_size})")
        ti = train_model(train_npz, cpath, best, args.epochs, args.batch_size, args.lr)
        if not ti or not cpath.exists():
            logger.error("Training failed, skipping")
            consec_failures += 1; continue

        # Training succeeded — reset circuit breaker
        consec_failures = 0

        # 4. EVALUATE
        logger.info(f"[4/5] Evaluate ({args.eval_games} games)")
        ev = evaluate(str(cpath), str(best), args.eval_games, args.budget)

        # 5. PROMOTE / REJECT
        wr = ev["win_rate"]
        promoted = wr >= args.promote_threshold
        if promoted:
            shutil.copy2(cpath, best)
            promos += 1
            eg = 400.0 * math.log10(wr / (1 - wr)) if 0 < wr < 1 else 0
            elo += eg
            logger.info(f"[5/5] PROMOTED (wr={wr:.1%}, +{eg:.0f} -> ~{elo:.0f} Elo)")
        else:
            logger.info(f"[5/5] REJECTED (wr={wr:.1%}, need {args.promote_threshold:.0%})")

        iel = time.time() - it0
        metrics = {"iteration": it, "timestamp": datetime.now(timezone.utc).isoformat(),
                   "selfplay": sp, "training": {k: v for k, v in ti.items() if k != "log_line"},
                   "evaluation": ev, "promoted": promoted, "estimated_elo": round(elo, 1),
                   "total_promotions": promos, "iteration_time_s": round(iel, 1)}
        with open(logf, "a") as f:
            f.write(json.dumps(metrics) + "\n")
        logger.info(f"  iter {it} done in {iel/60:.1f}min | elo~{elo:.0f} | promos={promos}/{it}")

        try:
            jpath.unlink()  # cleanup JSONL, keep NPZ
        except OSError:
            pass

    logger.info(f"\n{'='*70}\nLOOP COMPLETE\n{'='*70}")
    logger.info(f"  iterations={args.iterations} promotions={promos} elo~{elo:.0f}")
    logger.info(f"  best_model={best}  metrics={logf}")


if __name__ == "__main__":
    main()
