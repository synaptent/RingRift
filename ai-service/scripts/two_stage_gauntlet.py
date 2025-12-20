#!/usr/bin/env python3
"""Two-Stage Gauntlet with Statistical Confidence.

Stage 1: Quick screen (10 games) - eliminate models with <40% win rate
Stage 2: Deep evaluation (50 games) - full statistical analysis for top candidates

Uses Wilson score interval for confidence bounds, enabling early exit
for clearly weak models and confident ranking of strong ones.

Designed for distributed execution across cluster nodes.

Features:
- Game recording: Records games to SQLite database for training data extraction
- Quality filtering: Only records games where model wins (high-quality games)
- Auto-promotion: Automatically promote top performers to production

Usage:
    # Run full two-stage gauntlet
    python scripts/two_stage_gauntlet.py --run --board square8 --players 2

    # Stage 1 only (screening)
    python scripts/two_stage_gauntlet.py --stage1 --board square8 --players 2

    # Stage 2 only (use after collecting stage1 results)
    python scripts/two_stage_gauntlet.py --stage2 --board square8 --players 2

    # Distributed mode - process subset of models
    python scripts/two_stage_gauntlet.py --run --shard 0 --num-shards 10

    # Disable game recording
    python scripts/two_stage_gauntlet.py --run --no-record

    # Aggregate results from distributed run
    python scripts/two_stage_gauntlet.py --aggregate --board square8 --players 2

    # Promote top gauntlet performers
    python scripts/two_stage_gauntlet.py --promote --board square8 --players 2

    # Promote with custom threshold (80% confidence lower bound, top 5 models)
    python scripts/two_stage_gauntlet.py --promote --promote-threshold 0.80 --promote-top-n 5

    # Run gauntlet with auto-promotion at end (single shard only)
    python scripts/two_stage_gauntlet.py --run --promote --board square8 --players 2
"""

from __future__ import annotations

import argparse
import json
import math
import socket
import sys
import uuid
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from app.models import GameState, Move

# Add project root
SCRIPT_DIR = Path(__file__).parent
AI_SERVICE_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(AI_SERVICE_ROOT))

from app.models import BoardType
from app.models.discovery import discover_models as unified_discover_models

# Results directories
STAGE1_DIR = AI_SERVICE_ROOT / "data" / "gauntlet_stage1"
STAGE2_DIR = AI_SERVICE_ROOT / "data" / "gauntlet_stage2"
FINAL_RESULTS = AI_SERVICE_ROOT / "data" / "gauntlet_final_results.json"

# Game recording directory
GAUNTLET_GAMES_DIR = AI_SERVICE_ROOT / "data" / "gauntlet_games"

STAGE1_DIR.mkdir(parents=True, exist_ok=True)
STAGE2_DIR.mkdir(parents=True, exist_ok=True)
GAUNTLET_GAMES_DIR.mkdir(parents=True, exist_ok=True)

# Global flag for game recording (set by --no-record flag)
RECORD_GAMES = True


@dataclass
class WilsonScore:
    """Wilson score interval for binomial proportion confidence."""
    wins: int
    total: int
    confidence: float = 0.95  # 95% confidence

    @property
    def point_estimate(self) -> float:
        """Simple win rate."""
        return self.wins / self.total if self.total > 0 else 0.0

    @property
    def z(self) -> float:
        """Z-score for confidence level."""
        # 95% -> 1.96, 99% -> 2.576
        if self.confidence >= 0.99:
            return 2.576
        elif self.confidence >= 0.95:
            return 1.96
        else:
            return 1.645  # 90%

    def interval(self) -> Tuple[float, float]:
        """Return (lower, upper) Wilson score interval."""
        if self.total == 0:
            return (0.0, 1.0)

        n = self.total
        p = self.wins / n
        z = self.z
        z2 = z * z

        denom = 1 + z2 / n
        center = (p + z2 / (2 * n)) / denom
        margin = (z / denom) * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n))

        return (max(0.0, center - margin), min(1.0, center + margin))

    @property
    def lower_bound(self) -> float:
        """Conservative estimate (lower bound of CI)."""
        return self.interval()[0]

    @property
    def upper_bound(self) -> float:
        """Optimistic estimate (upper bound of CI)."""
        return self.interval()[1]

    def is_confident_above(self, threshold: float) -> bool:
        """Is lower bound confidently above threshold?"""
        return self.lower_bound > threshold

    def is_confident_below(self, threshold: float) -> bool:
        """Is upper bound confidently below threshold?"""
        return self.upper_bound < threshold


@dataclass
class ModelResult:
    """Result for a single model."""
    model_path: str
    model_name: str
    model_type: str
    board_type: str
    num_players: int

    # Stage 1 results
    stage1_games: int = 0
    stage1_wins_random: int = 0
    stage1_wins_heuristic: int = 0
    stage1_passed: bool = False
    stage1_early_exit: bool = False

    # Stage 2 results (only if passed stage 1)
    stage2_games: int = 0
    stage2_wins_random: int = 0
    stage2_wins_heuristic: int = 0
    stage2_wins_mcts: int = 0

    # Final scores
    final_score: float = 0.0
    confidence_lower: float = 0.0
    confidence_upper: float = 0.0

    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def stage1_score(self) -> WilsonScore:
        """Combined Wilson score for stage 1."""
        total_wins = self.stage1_wins_random + self.stage1_wins_heuristic
        total_games = self.stage1_games * 2  # played vs both baselines
        return WilsonScore(total_wins, total_games)

    def stage2_score(self) -> WilsonScore:
        """Combined Wilson score for stage 2."""
        total_wins = self.stage2_wins_random + self.stage2_wins_heuristic + self.stage2_wins_mcts
        total_games = self.stage2_games * 3  # played vs all baselines
        return WilsonScore(total_wins, total_games)

    def compute_final_score(self) -> None:
        """Compute final score and confidence bounds."""
        if self.stage2_games > 0:
            ws = self.stage2_score()
        else:
            ws = self.stage1_score()

        self.final_score = ws.point_estimate
        self.confidence_lower = ws.lower_bound
        self.confidence_upper = ws.upper_bound


def get_gauntlet_db_path(board_type: str, num_players: int, shard: int = 0) -> Path:
    """Get path to gauntlet games database for this shard."""
    hostname = socket.gethostname()
    return GAUNTLET_GAMES_DIR / f"gauntlet_{board_type}_{num_players}p_{hostname}_shard{shard}.db"


def play_game(
    model_path: str,
    model_type: str,
    opponent_type: str,
    board_type: BoardType,
    num_players: int = 2,
    model_plays_first: bool = True,
    record_games: bool = True,
    db_path: Optional[str] = None,
    mcts_difficulty: int = 3,
) -> Optional[int]:
    """Play a single game, optionally record to database, return winner or None for error.

    Args:
        model_path: Path to the model file
        model_type: "nn" or "nnue"
        opponent_type: "random", "heuristic", or "mcts"
        board_type: Board type
        num_players: Number of players
        model_plays_first: Whether the model plays first
        record_games: Whether to record games to database (only model wins)
        db_path: Path to games database (optional)
        mcts_difficulty: MCTS opponent difficulty level (1-10)

    Returns:
        Winner player number, or None if error
    """
    try:
        from app.game_engine import GameEngine
        from app.main import _create_ai_instance
        from app.models import AIType, AIConfig, GameStatus
        from app.training.generate_data import create_initial_state

        state = create_initial_state(board_type=board_type, num_players=num_players)
        initial_state = state  # Keep reference for recording
        engine = GameEngine()

        # Opponent setup
        opp_player = 2 if model_plays_first else 1
        OPPONENT_MAP = {
            "random": (AIType.RANDOM, 1),
            "heuristic": (AIType.HEURISTIC, 3),
            "mcts": (AIType.MCTS, mcts_difficulty),  # Use configurable difficulty
        }
        opp_type, difficulty = OPPONENT_MAP.get(opponent_type, (AIType.RANDOM, 1))
        opponent = _create_ai_instance(
            opp_type, opp_player,
            AIConfig(ai_type=opp_type, difficulty=difficulty)
        )

        # Model setup
        model_player = 1 if model_plays_first else 2
        model_name = Path(model_path).stem

        if model_type == "nnue":
            # NNUE models use minimax with NNUE evaluation
            model_ai = _create_ai_instance(
                AIType.MINIMAX, model_player,
                AIConfig(ai_type=AIType.MINIMAX, difficulty=4, nnue_model_path=model_path)
            )
        else:
            # NN models use MCTS with neural network guidance
            model_ai = _create_ai_instance(
                AIType.MCTS, model_player,
                AIConfig(
                    ai_type=AIType.MCTS,
                    difficulty=5,
                    think_time=500,  # 500ms per move
                    nn_model_id=model_name,
                    use_neural_net=True,
                )
            )

        ais = {model_player: model_ai, opp_player: opponent}

        # Collect moves and states for recording
        game_moves = []
        game_states = [state]

        for _ in range(300):
            if state.game_status == GameStatus.COMPLETED:
                break
            if hasattr(state.game_status, 'value'):
                if state.game_status.value not in ('active', 'in_progress'):
                    break

            current = state.current_player
            ai = ais.get(current)
            if ai is None:
                break

            move = ai.select_move(state)
            if move is None:
                break

            state_before = state
            state = engine.apply_move(state, move)

            # Record move and state
            game_moves.append((move, state_before, state))
            game_states.append(state)

        winner = state.winner

        # Record game if model won (high-quality training data)
        if record_games and RECORD_GAMES and winner == model_player and db_path:
            try:
                _record_game(
                    db_path=db_path,
                    initial_state=initial_state,
                    final_state=state,
                    moves=game_moves,
                    model_name=model_name,
                    model_type=model_type,
                    opponent_type=opponent_type,
                    model_player=model_player,
                    winner=winner,
                )
            except Exception as e:
                # Don't fail the game if recording fails
                pass

        return winner

    except Exception as e:
        return None


def _record_game(
    db_path: str,
    initial_state: "GameState",
    final_state: "GameState",
    moves: List[Tuple["Move", "GameState", "GameState"]],
    model_name: str,
    model_type: str,
    opponent_type: str,
    model_player: int,
    winner: int,
) -> None:
    """Record a completed game to the database.

    Args:
        db_path: Path to the database
        initial_state: Initial game state
        final_state: Final game state
        moves: List of (move, state_before, state_after) tuples
        model_name: Name of the model
        model_type: "nn" or "nnue"
        opponent_type: "random", "heuristic", or "mcts"
        model_player: Which player the model was (1 or 2)
        winner: Winner player number
    """
    from app.db.game_replay import GameReplayDB

    db = GameReplayDB(db_path, snapshot_interval=10)
    game_id = f"gauntlet_{model_name}_{opponent_type}_{uuid.uuid4().hex[:8]}"

    writer = db.store_game_incremental(
        game_id=game_id,
        initial_state=initial_state,
        all_snapshots=False,
        store_history_entries=True,
    )

    for move, state_before, state_after in moves:
        writer.add_move(
            move=move,
            state_after=state_after,
            state_before=state_before,
        )

    # Finalize with metadata
    metadata = {
        "source": "gauntlet",
        "model_name": model_name,
        "model_type": model_type,
        "opponent_type": opponent_type,
        "model_player": model_player,
        "winner": winner,
        "hostname": socket.gethostname(),
        "recorded_at": datetime.now().isoformat(),
    }

    writer.finalize(final_state=final_state, metadata=metadata)


def run_stage1_for_model(
    model: Dict[str, Any],
    board_type: BoardType,
    num_players: int,
    games_per_baseline: int = 10,
    early_exit_threshold: float = 0.3,
    db_path: Optional[str] = None,
    mcts_difficulty: int = 3,
) -> ModelResult:
    """Run stage 1 screening for a model.

    Early exit if model is clearly below threshold after initial games.
    """
    result = ModelResult(
        model_path=str(model["path"]),
        model_name=model["name"],
        model_type=model["type"],
        board_type=board_type.value,
        num_players=num_players,
    )

    # Test vs random first (easiest)
    for i in range(games_per_baseline):
        model_first = (i % 2 == 0)
        model_player = 1 if model_first else 2

        winner = play_game(
            model_path=str(model["path"]),
            model_type=model["type"],
            opponent_type="random",
            board_type=board_type,
            num_players=num_players,
            model_plays_first=model_first,
            db_path=db_path,
            mcts_difficulty=mcts_difficulty,
        )

        if winner is not None:
            result.stage1_games += 1
            if winner == model_player:
                result.stage1_wins_random += 1

        # Early exit check after 5 games vs random
        if i == 4:
            ws = WilsonScore(result.stage1_wins_random, 5)
            if ws.is_confident_below(early_exit_threshold):
                result.stage1_early_exit = True
                result.stage1_passed = False
                return result

    # Test vs heuristic
    for i in range(games_per_baseline):
        model_first = (i % 2 == 0)
        model_player = 1 if model_first else 2

        winner = play_game(
            model_path=str(model["path"]),
            model_type=model["type"],
            opponent_type="heuristic",
            board_type=board_type,
            num_players=num_players,
            model_plays_first=model_first,
            db_path=db_path,
            mcts_difficulty=mcts_difficulty,
        )

        if winner is not None:
            result.stage1_games += 1
            if winner == model_player:
                result.stage1_wins_heuristic += 1

    # Determine if passed stage 1
    ws = result.stage1_score()
    # Pass if lower bound > 0.35 (confidently better than random)
    result.stage1_passed = ws.lower_bound > 0.35

    return result


def run_stage2_for_model(
    result: ModelResult,
    board_type: BoardType,
    games_per_baseline: int = 50,
    db_path: Optional[str] = None,
    mcts_difficulty: int = 3,
) -> ModelResult:
    """Run stage 2 deep evaluation for a model that passed stage 1."""

    if not result.stage1_passed:
        return result

    for opponent in ["random", "heuristic", "mcts"]:
        for i in range(games_per_baseline):
            model_first = (i % 2 == 0)
            model_player = 1 if model_first else 2

            winner = play_game(
                model_path=result.model_path,
                model_type=result.model_type,
                opponent_type=opponent,
                board_type=board_type,
                num_players=result.num_players,
                model_plays_first=model_first,
                db_path=db_path,
                mcts_difficulty=mcts_difficulty,
            )

            if winner is not None:
                result.stage2_games += 1
                if winner == model_player:
                    if opponent == "random":
                        result.stage2_wins_random += 1
                    elif opponent == "heuristic":
                        result.stage2_wins_heuristic += 1
                    else:
                        result.stage2_wins_mcts += 1

    result.compute_final_score()
    return result


def run_two_stage_gauntlet(
    models: List[Dict[str, Any]],
    board_type: BoardType,
    num_players: int,
    stage1_games: int = 10,
    stage2_games: int = 50,
    parallel: int = 16,
    shard: int = 0,
    num_shards: int = 1,
    record_games: bool = True,
    mcts_difficulty: int = 3,
) -> List[ModelResult]:
    """Run two-stage gauntlet with parallelization.

    Args:
        models: List of model dicts with path, name, type
        board_type: Board type enum
        num_players: Number of players
        stage1_games: Games per baseline in stage 1
        stage2_games: Games per baseline in stage 2
        parallel: Number of parallel workers
        shard: Shard index for distributed execution
        num_shards: Total number of shards
        record_games: Whether to record winning games to database
        mcts_difficulty: MCTS opponent difficulty level (1-10)
    """

    # Shard models for distributed execution
    if num_shards > 1:
        models = [m for i, m in enumerate(models) if i % num_shards == shard]
        print(f"Shard {shard}/{num_shards}: Processing {len(models)} models")

    # Set up game recording database
    db_path = None
    if record_games and RECORD_GAMES:
        db_path = str(get_gauntlet_db_path(board_type.value, num_players, shard))
        print(f"Recording winning games to: {db_path}")

    results = []
    passed_stage1 = []

    # Stage 1: Parallel screening
    print(f"\n{'='*60}")
    print(f"STAGE 1: Screening {len(models)} models ({stage1_games} games each)")
    print(f"{'='*60}\n")

    with ProcessPoolExecutor(max_workers=parallel) as executor:
        futures = {
            executor.submit(
                run_stage1_for_model, model, board_type, num_players, stage1_games,
                0.3,  # early_exit_threshold
                db_path,
                mcts_difficulty,
            ): model
            for model in models
        }

        for i, future in enumerate(as_completed(futures)):
            result = future.result()
            results.append(result)

            status = "PASS" if result.stage1_passed else ("EARLY_EXIT" if result.stage1_early_exit else "FAIL")
            ws = result.stage1_score()
            print(f"[{i+1}/{len(models)}] {result.model_name}: {status} "
                  f"(score={ws.point_estimate:.2f}, CI=[{ws.lower_bound:.2f}, {ws.upper_bound:.2f}])")

            if result.stage1_passed:
                passed_stage1.append(result)

            # Save intermediate results
            if (i + 1) % 10 == 0:
                save_stage1_results(results, board_type, num_players, shard)

    save_stage1_results(results, board_type, num_players, shard)

    print(f"\nStage 1 complete: {len(passed_stage1)}/{len(models)} passed")

    # Stage 2: Deep evaluation of passed models
    if passed_stage1:
        print(f"\n{'='*60}")
        print(f"STAGE 2: Deep evaluation of {len(passed_stage1)} models ({stage2_games} games each)")
        print(f"{'='*60}\n")

        with ProcessPoolExecutor(max_workers=parallel) as executor:
            futures = {
                executor.submit(run_stage2_for_model, r, board_type, stage2_games, db_path, mcts_difficulty): r
                for r in passed_stage1
            }

            for i, future in enumerate(as_completed(futures)):
                result = future.result()
                ws = result.stage2_score()
                print(f"[{i+1}/{len(passed_stage1)}] {result.model_name}: "
                      f"score={ws.point_estimate:.2f} CI=[{ws.lower_bound:.2f}, {ws.upper_bound:.2f}]")

        save_stage2_results(passed_stage1, board_type, num_players, shard)

    # Print game recording stats
    if db_path:
        try:
            from app.db.game_replay import GameReplayDB
            db = GameReplayDB(db_path)
            with db._get_conn() as conn:
                count = conn.execute("SELECT COUNT(*) FROM games").fetchone()[0]
                print(f"\nRecorded {count} winning games to database")
        except Exception:
            pass

    return results


def save_stage1_results(results: List[ModelResult], board_type: BoardType, num_players: int, shard: int):
    """Save stage 1 results to file."""
    filename = STAGE1_DIR / f"{board_type.value}_{num_players}p_shard{shard}.json"
    data = {
        "board_type": board_type.value,
        "num_players": num_players,
        "shard": shard,
        "timestamp": datetime.now().isoformat(),
        "results": [asdict(r) for r in results],
    }
    with open(filename, "w") as f:
        json.dump(data, f, indent=2)


def save_stage2_results(results: List[ModelResult], board_type: BoardType, num_players: int, shard: int):
    """Save stage 2 results to file."""
    filename = STAGE2_DIR / f"{board_type.value}_{num_players}p_shard{shard}.json"
    data = {
        "board_type": board_type.value,
        "num_players": num_players,
        "shard": shard,
        "timestamp": datetime.now().isoformat(),
        "results": [asdict(r) for r in results],
    }
    with open(filename, "w") as f:
        json.dump(data, f, indent=2)


def aggregate_results(board_type: str, num_players: int) -> List[ModelResult]:
    """Aggregate results from all shards."""
    all_results = []

    # Load stage 2 results (preferred)
    for f in STAGE2_DIR.glob(f"{board_type}_{num_players}p_shard*.json"):
        with open(f) as fp:
            data = json.load(fp)
            for r in data["results"]:
                mr = ModelResult(**{k: v for k, v in r.items() if k in ModelResult.__dataclass_fields__})
                all_results.append(mr)

    # Sort by final score (lower bound for conservative ranking)
    all_results.sort(key=lambda r: r.confidence_lower, reverse=True)

    return all_results


# Promotion directory for gauntlet winners
PROMOTED_DIR = AI_SERVICE_ROOT / "models" / "gauntlet_promoted"
PROMOTED_DIR.mkdir(parents=True, exist_ok=True)


def promote_gauntlet_winners(
    board_type: str,
    num_players: int,
    threshold: float = 0.70,
    top_n: int = 3,
    dry_run: bool = False,
) -> List[Dict[str, Any]]:
    """Promote top gauntlet performers to production.

    Args:
        board_type: Board type to promote for
        num_players: Number of players
        threshold: Minimum confidence lower bound for promotion (default 70%)
        top_n: Maximum number of models to promote (default 3)
        dry_run: If True, only report what would be promoted

    Returns:
        List of promoted model info dicts
    """
    import shutil

    results = aggregate_results(board_type, num_players)
    if not results:
        print(f"No gauntlet results found for {board_type} {num_players}p")
        return []

    # Filter by threshold and take top N
    candidates = [r for r in results if r.confidence_lower >= threshold]
    to_promote = candidates[:top_n]

    if not to_promote:
        print(f"No models meet threshold {threshold:.1%} for promotion")
        print(f"Top models:")
        for i, r in enumerate(results[:5]):
            print(f"  {i+1}. {r.model_name}: CI=[{r.confidence_lower:.3f}, {r.confidence_upper:.3f}]")
        return []

    promoted = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for i, result in enumerate(to_promote):
        src_path = Path(result.model_path)
        if not src_path.exists():
            print(f"  [SKIP] Source not found: {src_path}")
            continue

        # Create promoted filename with rank
        promoted_name = f"gauntlet_top{i+1}_{board_type}_{num_players}p_{timestamp}{src_path.suffix}"
        dst_path = PROMOTED_DIR / promoted_name

        info = {
            "rank": i + 1,
            "model_name": result.model_name,
            "model_type": result.model_type,
            "source_path": str(src_path),
            "promoted_path": str(dst_path),
            "confidence_lower": result.confidence_lower,
            "confidence_upper": result.confidence_upper,
            "final_score": result.final_score,
            "stage2_games": result.stage2_games,
            "timestamp": timestamp,
        }

        if dry_run:
            print(f"  [DRY RUN] Would promote: {result.model_name}")
            print(f"            CI=[{result.confidence_lower:.3f}, {result.confidence_upper:.3f}]")
            print(f"            -> {dst_path}")
        else:
            shutil.copy2(src_path, dst_path)
            print(f"  [PROMOTED] {result.model_name}")
            print(f"             CI=[{result.confidence_lower:.3f}, {result.confidence_upper:.3f}]")
            print(f"             -> {dst_path}")

            # Try to register in model registry
            try:
                from app.training.model_registry import ModelRegistry, ModelStage
                registry = ModelRegistry()
                registry.register_model(
                    model_path=str(dst_path),
                    board_type=board_type,
                    num_players=num_players,
                    stage=ModelStage.PRODUCTION,
                    metadata={
                        "source": "gauntlet",
                        "gauntlet_rank": i + 1,
                        "confidence_lower": result.confidence_lower,
                        "confidence_upper": result.confidence_upper,
                        "original_model": result.model_name,
                    },
                )
                print(f"             Registered in model registry")
            except Exception as e:
                print(f"             Warning: Could not register in registry: {e}")

        promoted.append(info)

    # Save promotion manifest
    if promoted and not dry_run:
        manifest_path = PROMOTED_DIR / f"promotion_{board_type}_{num_players}p_{timestamp}.json"
        with open(manifest_path, "w") as f:
            json.dump({
                "board_type": board_type,
                "num_players": num_players,
                "threshold": threshold,
                "timestamp": timestamp,
                "promoted": promoted,
            }, f, indent=2)
        print(f"\nPromotion manifest: {manifest_path}")

    return promoted


def discover_models_for_gauntlet(
    board_type: str,
    num_players: int,
) -> List[Dict[str, Any]]:
    """Discover models for gauntlet."""
    models_dir = AI_SERVICE_ROOT / "models"

    discovered = unified_discover_models(
        models_dir=models_dir,
        board_type=board_type,
        num_players=num_players,
        include_unknown=True,
    )

    models = []
    for m in discovered:
        models.append({
            "path": m.path,
            "name": m.name,
            "type": m.model_type,
        })

    return models


def main():
    global RECORD_GAMES

    parser = argparse.ArgumentParser(description="Two-Stage Gauntlet with Statistical Confidence")
    parser.add_argument("--run", action="store_true", help="Run full two-stage gauntlet")
    parser.add_argument("--stage1", action="store_true", help="Run stage 1 only")
    parser.add_argument("--stage2", action="store_true", help="Run stage 2 only")
    parser.add_argument("--aggregate", action="store_true", help="Aggregate results from shards")
    parser.add_argument("--promote", action="store_true", help="Promote top gauntlet performers")
    parser.add_argument("--board", type=str, default="square8", help="Board type")
    parser.add_argument("--players", type=int, default=2, help="Number of players")
    parser.add_argument("--stage1-games", type=int, default=10, help="Games per baseline in stage 1")
    parser.add_argument("--stage2-games", type=int, default=50, help="Games per baseline in stage 2")
    parser.add_argument("-j", "--parallel", type=int, default=16, help="Parallel workers")
    parser.add_argument("--shard", type=int, default=0, help="Shard index for distributed execution")
    parser.add_argument("--num-shards", type=int, default=1, help="Total number of shards")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of models (0 = no limit)")
    parser.add_argument("--no-record", action="store_true", help="Disable game recording")
    parser.add_argument("--promote-threshold", type=float, default=0.70,
                        help="Minimum confidence lower bound for promotion (default 0.70)")
    parser.add_argument("--promote-top-n", type=int, default=3,
                        help="Maximum number of models to promote (default 3)")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without making changes")
    parser.add_argument("--difficulty", type=int, default=3,
                        help="MCTS opponent difficulty (1-10, default 3 for faster evaluation)")

    args = parser.parse_args()

    # Set global recording flag
    if args.no_record:
        RECORD_GAMES = False

    board_type = BoardType(args.board) if args.board in [bt.value for bt in BoardType] else BoardType.SQUARE8

    if args.aggregate:
        results = aggregate_results(args.board, args.players)
        print(f"\nAggregated {len(results)} models:")
        print(f"\n{'Rank':<5} {'Model':<50} {'Score':<8} {'CI Lower':<10} {'CI Upper':<10}")
        print("-" * 90)
        for i, r in enumerate(results[:20]):
            print(f"{i+1:<5} {r.model_name[:48]:<50} {r.final_score:.3f}    {r.confidence_lower:.3f}      {r.confidence_upper:.3f}")
        return

    if args.promote:
        print(f"\n{'='*60}")
        print(f"GAUNTLET PROMOTION: {args.board} {args.players}p")
        print(f"Threshold: {args.promote_threshold:.0%} confidence lower bound")
        print(f"Top N: {args.promote_top_n}")
        print(f"{'='*60}\n")

        promoted = promote_gauntlet_winners(
            board_type=args.board,
            num_players=args.players,
            threshold=args.promote_threshold,
            top_n=args.promote_top_n,
            dry_run=args.dry_run,
        )

        if promoted:
            print(f"\nPromoted {len(promoted)} models to {PROMOTED_DIR}")
        return

    # Discover models
    models = discover_models_for_gauntlet(args.board, args.players)
    print(f"Discovered {len(models)} models for {args.board} {args.players}p")

    if args.limit > 0:
        models = models[:args.limit]

    if not models:
        print("No models found!")
        return

    # Run gauntlet
    results = run_two_stage_gauntlet(
        models=models,
        board_type=board_type,
        num_players=args.players,
        stage1_games=args.stage1_games,
        stage2_games=args.stage2_games,
        parallel=args.parallel,
        shard=args.shard,
        num_shards=args.num_shards,
        record_games=not args.no_record,
        mcts_difficulty=args.difficulty,
    )

    # Print summary
    passed = [r for r in results if r.stage1_passed]
    early_exit = [r for r in results if r.stage1_early_exit]

    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Total models: {len(results)}")
    print(f"Passed stage 1: {len(passed)} ({100*len(passed)/len(results):.1f}%)")
    print(f"Early exit: {len(early_exit)} ({100*len(early_exit)/len(results):.1f}%)")

    if passed:
        print(f"\nTop 10 models:")
        passed.sort(key=lambda r: r.confidence_lower, reverse=True)
        for i, r in enumerate(passed[:10]):
            print(f"  {i+1}. {r.model_name}: {r.final_score:.3f} [{r.confidence_lower:.3f}, {r.confidence_upper:.3f}]")

    # Auto-promote if requested and this is a single-shard run (to avoid race conditions)
    if args.promote and args.num_shards == 1:
        print(f"\n{'='*60}")
        print(f"AUTO-PROMOTION")
        print(f"{'='*60}")
        promoted = promote_gauntlet_winners(
            board_type=args.board,
            num_players=args.players,
            threshold=args.promote_threshold,
            top_n=args.promote_top_n,
            dry_run=args.dry_run,
        )
        if promoted:
            print(f"Promoted {len(promoted)} models")


if __name__ == "__main__":
    main()
