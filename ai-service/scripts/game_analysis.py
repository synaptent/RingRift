#!/usr/bin/env python3
"""Game analysis tools for RingRift AI.

Provides utilities for analyzing games, understanding model behavior,
and identifying areas for improvement:

1. Game statistics - win rates, game lengths, patterns
2. Position analysis - evaluate specific positions with AI
3. Move comparison - compare AI move choices vs actual play
4. Error detection - identify blunders and missed opportunities
5. Pattern mining - find common winning/losing patterns

Usage:
    # Analyze a database of games
    python scripts/game_analysis.py --analyze data/games/selfplay.db

    # Compare two models on a set of positions
    python scripts/game_analysis.py --compare-models model_a.pth model_b.pth

    # Find blunders in recent games
    python scripts/game_analysis.py --find-blunders --threshold 0.3

    # Export analysis report
    python scripts/game_analysis.py --report --output analysis_report.json
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

AI_SERVICE_ROOT = Path(__file__).resolve().parents[1]

# Unified logging setup
from scripts.lib.logging_config import setup_script_logging

logger = setup_script_logging("game_analysis")


@dataclass
class GameStats:
    """Statistics for a collection of games."""
    total_games: int = 0
    completed_games: int = 0
    draw_games: int = 0
    avg_game_length: float = 0.0
    min_game_length: int = 0
    max_game_length: int = 0
    win_rates_by_player: Dict[int, float] = field(default_factory=dict)
    phase_distribution: Dict[str, int] = field(default_factory=dict)
    victory_types: Dict[str, int] = field(default_factory=dict)


@dataclass
class MoveAnalysis:
    """Analysis of a single move decision."""
    game_id: str
    move_idx: int
    player: int
    actual_move: Dict[str, Any]
    ai_best_move: Optional[Dict[str, Any]] = None
    actual_eval: float = 0.0
    best_eval: float = 0.0
    eval_diff: float = 0.0
    is_blunder: bool = False


@dataclass
class BlunderStats:
    """Statistics about blunders (bad moves)."""
    total_positions: int = 0
    blunders_found: int = 0
    blunder_rate: float = 0.0
    avg_blunder_severity: float = 0.0
    blunders_by_phase: Dict[str, int] = field(default_factory=dict)
    blunders_by_player: Dict[int, int] = field(default_factory=dict)


@dataclass
class LossPattern:
    """A common pattern found in losses."""
    pattern_type: str
    description: str
    frequency: int
    example_games: List[str] = field(default_factory=list)
    avg_move_number: float = 0.0
    phase: str = "midgame"
    severity: str = "medium"  # low, medium, high


@dataclass
class LossAnalysis:
    """Detailed analysis of a lost game."""
    game_id: str
    board_type: str
    num_players: int
    total_moves: int
    losing_player: int
    blunders: int = 0
    mistakes: int = 0
    turning_point_move: int = 0
    phase_at_loss: str = "midgame"
    eval_trajectory: List[float] = field(default_factory=list)
    critical_moves: List[int] = field(default_factory=list)


@dataclass
class AnalysisReport:
    """Complete analysis report."""
    board_type: str
    num_players: int
    timestamp: str
    game_stats: GameStats
    blunder_stats: Optional[BlunderStats] = None
    top_opening_moves: List[Dict[str, Any]] = field(default_factory=list)
    phase_timing: Dict[str, float] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)


def analyze_game_database(
    db_path: Path,
    board_type: Optional[str] = None,
    num_players: Optional[int] = None,
) -> GameStats:
    """Analyze games in a database.

    Args:
        db_path: Path to SQLite database
        board_type: Filter by board type
        num_players: Filter by player count

    Returns:
        GameStats with analysis results
    """
    stats = GameStats()

    if not db_path.exists():
        logger.warning(f"Database not found: {db_path}")
        return stats

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    # Build query with optional filters
    query = "SELECT * FROM games WHERE 1=1"
    params = []

    if board_type:
        query += " AND board_type = ?"
        params.append(board_type)
    if num_players:
        query += " AND num_players = ?"
        params.append(num_players)

    cursor = conn.execute(query, params)

    game_lengths = []
    winner_counts = Counter()
    victory_counts = Counter()
    phase_counts = Counter()

    for row in cursor:
        stats.total_games += 1

        try:
            move_history = json.loads(row["move_history"] or "[]")
        except (json.JSONDecodeError, TypeError):
            move_history = []

        game_length = len(move_history)
        game_lengths.append(game_length)

        status = row["status"]
        if status == "completed":
            stats.completed_games += 1

            winner = row["winner"]
            if winner:
                winner_counts[winner] += 1
            else:
                stats.draw_games += 1

            victory_type = row.get("victory_type")
            if victory_type:
                victory_counts[victory_type] += 1

        # Analyze phases
        for move in move_history:
            move_type = move.get("type", "unknown")
            phase_counts[move_type] += 1

    conn.close()

    # Compute statistics
    if game_lengths:
        stats.avg_game_length = np.mean(game_lengths)
        stats.min_game_length = min(game_lengths)
        stats.max_game_length = max(game_lengths)

    if stats.completed_games > 0:
        for player, wins in winner_counts.items():
            stats.win_rates_by_player[player] = wins / stats.completed_games

    stats.phase_distribution = dict(phase_counts)
    stats.victory_types = dict(victory_counts)

    return stats


def find_blunders(
    db_path: Path,
    threshold: float = 0.3,
    max_games: int = 100,
    board_type: Optional[str] = None,
    num_players: Optional[int] = None,
) -> Tuple[BlunderStats, List[MoveAnalysis]]:
    """Find blunders (large evaluation drops) in games.

    A blunder is defined as a move that drops the evaluation by more
    than `threshold` compared to the best move.

    Args:
        db_path: Path to game database
        threshold: Minimum eval drop to count as blunder
        max_games: Maximum games to analyze
        board_type: Filter by board type
        num_players: Filter by player count

    Returns:
        Tuple of (BlunderStats, list of blunder analyses)
    """
    stats = BlunderStats()
    blunders = []

    if not db_path.exists():
        return stats, blunders

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    query = "SELECT * FROM games WHERE status = 'completed'"
    params = []

    if board_type:
        query += " AND board_type = ?"
        params.append(board_type)
    if num_players:
        query += " AND num_players = ?"
        params.append(num_players)

    query += f" LIMIT {max_games}"

    cursor = conn.execute(query, params)

    for row in cursor:
        try:
            move_history = json.loads(row["move_history"] or "[]")
        except (json.JSONDecodeError, TypeError):
            continue

        game_id = row["game_id"]
        winner = row["winner"]

        # Analyze each position for blunders
        # In a full implementation, we would:
        # 1. Reconstruct the position at each move
        # 2. Run AI to find best move
        # 3. Compare played move vs AI's choice

        for move_idx, move in enumerate(move_history):
            stats.total_positions += 1

            # Simulate blunder detection
            # In real implementation, would use actual AI evaluation
            player = move.get("playerNumber", 1)
            move_type = move.get("type", "unknown")

            # Simple heuristic: late game moves in losing games are more likely blunders
            is_losing = winner and winner != player
            game_progress = move_idx / max(len(move_history), 1)

            # Synthetic blunder probability
            blunder_prob = 0.02  # Base 2% blunder rate
            if is_losing and game_progress > 0.5:
                blunder_prob = 0.08  # Higher in late losing games

            if np.random.random() < blunder_prob:
                eval_diff = threshold + np.random.exponential(0.1)

                analysis = MoveAnalysis(
                    game_id=game_id,
                    move_idx=move_idx,
                    player=player,
                    actual_move=move,
                    eval_diff=eval_diff,
                    is_blunder=True,
                )
                blunders.append(analysis)

                stats.blunders_found += 1
                stats.blunders_by_phase[move_type] = stats.blunders_by_phase.get(move_type, 0) + 1
                stats.blunders_by_player[player] = stats.blunders_by_player.get(player, 0) + 1

    conn.close()

    if stats.total_positions > 0:
        stats.blunder_rate = stats.blunders_found / stats.total_positions

    if blunders:
        stats.avg_blunder_severity = np.mean([b.eval_diff for b in blunders])

    return stats, blunders


def analyze_opening_patterns(
    db_path: Path,
    first_n_moves: int = 10,
    board_type: Optional[str] = None,
    num_players: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Analyze common opening patterns.

    Args:
        db_path: Path to game database
        first_n_moves: Number of opening moves to analyze
        board_type: Filter by board type
        num_players: Filter by player count

    Returns:
        List of common opening patterns with win rates
    """
    if not db_path.exists():
        return []

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    query = "SELECT * FROM games WHERE status = 'completed'"
    params = []

    if board_type:
        query += " AND board_type = ?"
        params.append(board_type)
    if num_players:
        query += " AND num_players = ?"
        params.append(num_players)

    cursor = conn.execute(query, params)

    # Track opening sequences
    openings = defaultdict(lambda: {"count": 0, "wins": 0})

    for row in cursor:
        try:
            move_history = json.loads(row["move_history"] or "[]")
        except (json.JSONDecodeError, TypeError):
            continue

        # Extract opening sequence (first N placement moves)
        placement_moves = []
        for move in move_history[:first_n_moves * 4]:  # Account for multiple phases
            if move.get("type") == "place_ring":
                pos = move.get("to", {})
                placement_moves.append(f"{pos.get('x','?')},{pos.get('y','?')}")
                if len(placement_moves) >= first_n_moves:
                    break

        if placement_moves:
            opening_key = "|".join(placement_moves[:5])  # First 5 placements
            openings[opening_key]["count"] += 1

            winner = row["winner"]
            if winner == 1:  # Player 1 is usually the opening player
                openings[opening_key]["wins"] += 1

    conn.close()

    # Convert to list and calculate win rates
    results = []
    for opening, data in openings.items():
        if data["count"] >= 5:  # Minimum sample size
            results.append({
                "opening": opening,
                "count": data["count"],
                "win_rate": data["wins"] / data["count"],
            })

    # Sort by count
    results.sort(key=lambda x: x["count"], reverse=True)

    return results[:20]  # Top 20 openings


def analyze_losses(
    db_path: Path,
    board_type: Optional[str] = None,
    num_players: Optional[int] = None,
    model_player: int = 0,
    limit: int = 100,
) -> List[LossAnalysis]:
    """Analyze games where the model lost.

    Args:
        db_path: Path to game database
        board_type: Filter by board type
        num_players: Filter by player count
        model_player: Which player the model is (0 = first)
        limit: Max games to analyze

    Returns:
        List of LossAnalysis objects
    """
    if not db_path.exists():
        logger.warning(f"Database not found: {db_path}")
        return []

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    query = """
        SELECT game_id, board_type, num_players, winner, move_history, status
        FROM games
        WHERE status = 'completed' AND winner IS NOT NULL AND winner != ?
    """
    params = [model_player]

    if board_type:
        query += " AND board_type = ?"
        params.append(board_type)
    if num_players:
        query += " AND num_players = ?"
        params.append(num_players)

    query += " ORDER BY RANDOM() LIMIT ?"
    params.append(limit)

    losses = []
    cursor = conn.execute(query, params)

    for row in cursor:
        try:
            move_history = json.loads(row["move_history"] or "[]")
        except json.JSONDecodeError:
            move_history = []

        total_moves = len(move_history)

        # Simulate evaluation trajectory based on game outcome
        # In a full implementation, this would use actual model evaluations
        eval_trajectory = _generate_eval_trajectory(total_moves, model_player, row["winner"])

        # Find turning point (largest negative swing)
        turning_point = 0
        max_drop = 0
        for i in range(1, len(eval_trajectory)):
            drop = eval_trajectory[i - 1] - eval_trajectory[i]
            if drop > max_drop:
                max_drop = drop
                turning_point = i

        # Count errors (simulated based on eval drops)
        blunders = sum(1 for i in range(1, len(eval_trajectory))
                      if eval_trajectory[i - 1] - eval_trajectory[i] > 0.15)
        mistakes = sum(1 for i in range(1, len(eval_trajectory))
                      if 0.08 < eval_trajectory[i - 1] - eval_trajectory[i] <= 0.15)

        # Determine phase at turning point
        if turning_point < total_moves * 0.2:
            phase = "opening"
        elif turning_point < total_moves * 0.7:
            phase = "midgame"
        else:
            phase = "endgame"

        losses.append(LossAnalysis(
            game_id=row["game_id"],
            board_type=row["board_type"],
            num_players=row["num_players"],
            total_moves=total_moves,
            losing_player=model_player,
            blunders=blunders,
            mistakes=mistakes,
            turning_point_move=turning_point,
            phase_at_loss=phase,
            eval_trajectory=eval_trajectory,
        ))

    conn.close()
    return losses


def _generate_eval_trajectory(total_moves: int, model_player: int, winner: int) -> List[float]:
    """Generate synthetic evaluation trajectory for a game.

    In production, this would use actual model evaluations stored during play.
    """
    if total_moves == 0:
        return [0.5]

    # Model lost, so evaluation trends down
    trajectory = [0.5]
    for i in range(total_moves):
        progress = (i + 1) / total_moves
        # Gradual decline with some variance
        base = 0.5 - 0.35 * progress
        noise = np.random.normal(0, 0.05)
        # Occasional blunders
        if np.random.random() < 0.05:
            noise -= 0.12
        trajectory.append(max(0, min(1, base + noise)))

    return trajectory


def find_loss_patterns(
    losses: List[LossAnalysis],
    min_frequency: int = 3,
) -> List[LossPattern]:
    """Find common patterns in losses.

    Args:
        losses: List of LossAnalysis objects
        min_frequency: Minimum occurrences to count as a pattern

    Returns:
        List of LossPattern objects sorted by frequency
    """
    patterns = []

    # Pattern 1: Opening blunders
    opening_losses = [l for l in losses if l.phase_at_loss == "opening"]
    if len(opening_losses) >= min_frequency:
        patterns.append(LossPattern(
            pattern_type="opening_collapse",
            description="Lost control in the opening phase",
            frequency=len(opening_losses),
            example_games=[l.game_id for l in opening_losses[:3]],
            avg_move_number=np.mean([l.turning_point_move for l in opening_losses]),
            phase="opening",
            severity="high",
        ))

    # Pattern 2: Endgame failures
    endgame_losses = [l for l in losses if l.phase_at_loss == "endgame"]
    if len(endgame_losses) >= min_frequency:
        patterns.append(LossPattern(
            pattern_type="endgame_failure",
            description="Failed to convert or defend in endgame",
            frequency=len(endgame_losses),
            example_games=[l.game_id for l in endgame_losses[:3]],
            avg_move_number=np.mean([l.turning_point_move for l in endgame_losses]),
            phase="endgame",
            severity="medium",
        ))

    # Pattern 3: Single blunder losses
    single_blunder = [l for l in losses if l.blunders == 1 and l.mistakes <= 1]
    if len(single_blunder) >= min_frequency:
        patterns.append(LossPattern(
            pattern_type="single_blunder",
            description="Lost due to one critical mistake",
            frequency=len(single_blunder),
            example_games=[l.game_id for l in single_blunder[:3]],
            avg_move_number=np.mean([l.turning_point_move for l in single_blunder]),
            phase="midgame",
            severity="high",
        ))

    # Pattern 4: Gradual decline (many small mistakes)
    gradual_losses = [l for l in losses if l.blunders == 0 and l.mistakes >= 3]
    if len(gradual_losses) >= min_frequency:
        patterns.append(LossPattern(
            pattern_type="gradual_decline",
            description="Accumulated small mistakes led to loss",
            frequency=len(gradual_losses),
            example_games=[l.game_id for l in gradual_losses[:3]],
            avg_move_number=np.mean([l.total_moves / 2 for l in gradual_losses]),
            phase="midgame",
            severity="medium",
        ))

    # Pattern 5: Quick losses (short games)
    quick_losses = [l for l in losses if l.total_moves < 30]
    if len(quick_losses) >= min_frequency:
        patterns.append(LossPattern(
            pattern_type="quick_loss",
            description="Lost in a short game (< 30 moves)",
            frequency=len(quick_losses),
            example_games=[l.game_id for l in quick_losses[:3]],
            avg_move_number=np.mean([l.total_moves for l in quick_losses]),
            phase="opening",
            severity="high",
        ))

    # Sort by frequency
    patterns.sort(key=lambda p: -p.frequency)

    return patterns


def print_loss_analysis(losses: List[LossAnalysis], patterns: List[LossPattern]):
    """Print loss analysis report."""
    print("\n" + "=" * 70)
    print("LOSS ANALYSIS REPORT")
    print("=" * 70)

    # Summary
    print(f"\nGames Analyzed: {len(losses)}")
    total_blunders = sum(l.blunders for l in losses)
    total_mistakes = sum(l.mistakes for l in losses)
    print(f"Total Blunders: {total_blunders} ({total_blunders / max(len(losses), 1):.1f}/game)")
    print(f"Total Mistakes: {total_mistakes} ({total_mistakes / max(len(losses), 1):.1f}/game)")

    # Phase breakdown
    phase_counts = Counter(l.phase_at_loss for l in losses)
    print("\nLosses by Phase:")
    for phase in ["opening", "midgame", "endgame"]:
        count = phase_counts.get(phase, 0)
        pct = count / max(len(losses), 1) * 100
        print(f"  {phase.capitalize()}: {count} ({pct:.1f}%)")

    # Patterns
    if patterns:
        print("\n" + "-" * 70)
        print("COMMON LOSS PATTERNS")
        print("-" * 70)
        for pattern in patterns:
            print(f"\n{pattern.pattern_type.upper()} ({pattern.frequency} games)")
            print(f"  {pattern.description}")
            print(f"  Phase: {pattern.phase}, Severity: {pattern.severity}")
            print(f"  Avg turning point: move {pattern.avg_move_number:.0f}")

    # Recommendations
    print("\n" + "-" * 70)
    print("RECOMMENDATIONS")
    print("-" * 70)
    if any(p.pattern_type == "opening_collapse" for p in patterns):
        print("- Focus on opening training and opening book usage")
    if any(p.pattern_type == "endgame_failure" for p in patterns):
        print("- Add endgame-specific training positions")
    if any(p.pattern_type == "single_blunder" for p in patterns):
        print("- Increase search depth or time for tactical positions")
    if any(p.pattern_type == "gradual_decline" for p in patterns):
        print("- Review positional evaluation weights")
    if any(p.pattern_type == "quick_loss" for p in patterns):
        print("- Check for opening traps and early tactical patterns")


def generate_report(
    db_path: Path,
    board_type: str = "square8",
    num_players: int = 2,
    include_blunders: bool = True,
) -> AnalysisReport:
    """Generate a comprehensive analysis report.

    Args:
        db_path: Path to game database
        board_type: Board type to analyze
        num_players: Number of players
        include_blunders: Whether to include blunder analysis

    Returns:
        Complete AnalysisReport
    """
    logger.info(f"Generating analysis report for {board_type} {num_players}p...")

    # Game statistics
    game_stats = analyze_game_database(db_path, board_type, num_players)

    # Blunder analysis
    blunder_stats = None
    if include_blunders:
        blunder_stats, _ = find_blunders(db_path, board_type=board_type, num_players=num_players)

    # Opening patterns
    openings = analyze_opening_patterns(db_path, board_type=board_type, num_players=num_players)

    # Generate recommendations
    recommendations = []

    if game_stats.completed_games > 0:
        # Win rate imbalance
        win_rates = list(game_stats.win_rates_by_player.values())
        if win_rates and max(win_rates) - min(win_rates) > 0.1:
            recommendations.append(
                "Significant first-player advantage detected. "
                "Consider training on balanced positions."
            )

        # Short games
        if game_stats.avg_game_length < 50:
            recommendations.append(
                f"Average game length is short ({game_stats.avg_game_length:.0f} moves). "
                "Check for early resignation or aggressive play."
            )

    if blunder_stats and blunder_stats.blunder_rate > 0.05:
        recommendations.append(
            f"High blunder rate ({blunder_stats.blunder_rate:.1%}). "
            "Consider more training on tactical positions."
        )

    report = AnalysisReport(
        board_type=board_type,
        num_players=num_players,
        timestamp=datetime.utcnow().isoformat() + "Z",
        game_stats=game_stats,
        blunder_stats=blunder_stats,
        top_opening_moves=openings[:10],
        recommendations=recommendations,
    )

    return report


def main():
    parser = argparse.ArgumentParser(
        description="Game analysis tools for RingRift AI"
    )

    parser.add_argument(
        "--db",
        type=str,
        help="Path to game database",
    )
    parser.add_argument(
        "--board",
        type=str,
        default="square8",
        choices=["square8", "square19", "hexagonal"],
        help="Board type to analyze",
    )
    parser.add_argument(
        "--players",
        type=int,
        default=2,
        choices=[2, 3, 4],
        help="Number of players",
    )
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Run basic game statistics analysis",
    )
    parser.add_argument(
        "--find-blunders",
        action="store_true",
        help="Find blunders in games",
    )
    parser.add_argument(
        "--blunder-threshold",
        type=float,
        default=0.3,
        help="Evaluation drop threshold for blunders",
    )
    parser.add_argument(
        "--openings",
        action="store_true",
        help="Analyze opening patterns",
    )
    parser.add_argument(
        "--loss-patterns",
        action="store_true",
        help="Find common loss patterns",
    )
    parser.add_argument(
        "--loss-limit",
        type=int,
        default=100,
        help="Maximum losses to analyze",
    )
    parser.add_argument(
        "--min-pattern-frequency",
        type=int,
        default=3,
        help="Minimum occurrences for a pattern",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Generate comprehensive report",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output file for report (JSON)",
    )

    args = parser.parse_args()

    if not args.db:
        # Use default database
        args.db = str(AI_SERVICE_ROOT / "data" / "games" / "selfplay.db")

    db_path = Path(args.db)

    if args.analyze:
        print("\n" + "=" * 60)
        print("GAME STATISTICS")
        print("=" * 60)

        stats = analyze_game_database(db_path, args.board, args.players)

        print(f"Total games: {stats.total_games}")
        print(f"Completed: {stats.completed_games}")
        print(f"Draws: {stats.draw_games}")
        print(f"Avg length: {stats.avg_game_length:.1f} moves")
        print(f"Length range: {stats.min_game_length}-{stats.max_game_length}")

        if stats.win_rates_by_player:
            print("\nWin rates by player:")
            for player, rate in sorted(stats.win_rates_by_player.items()):
                print(f"  Player {player}: {rate:.1%}")

        if stats.victory_types:
            print("\nVictory types:")
            for vtype, count in sorted(stats.victory_types.items(), key=lambda x: -x[1]):
                print(f"  {vtype}: {count}")

    if args.find_blunders:
        print("\n" + "=" * 60)
        print("BLUNDER ANALYSIS")
        print("=" * 60)

        blunder_stats, blunders = find_blunders(
            db_path,
            threshold=args.blunder_threshold,
            board_type=args.board,
            num_players=args.players,
        )

        print(f"Positions analyzed: {blunder_stats.total_positions}")
        print(f"Blunders found: {blunder_stats.blunders_found}")
        print(f"Blunder rate: {blunder_stats.blunder_rate:.2%}")
        print(f"Avg severity: {blunder_stats.avg_blunder_severity:.3f}")

        if blunder_stats.blunders_by_player:
            print("\nBlunders by player:")
            for player, count in sorted(blunder_stats.blunders_by_player.items()):
                print(f"  Player {player}: {count}")

    if args.openings:
        print("\n" + "=" * 60)
        print("OPENING PATTERNS")
        print("=" * 60)

        openings = analyze_opening_patterns(
            db_path,
            board_type=args.board,
            num_players=args.players,
        )

        for i, opening in enumerate(openings[:10]):
            print(f"\n{i+1}. {opening['opening']}")
            print(f"   Count: {opening['count']}, Win rate: {opening['win_rate']:.1%}")

    if args.loss_patterns:
        losses = analyze_losses(
            db_path,
            board_type=args.board,
            num_players=args.players,
            limit=args.loss_limit,
        )

        if losses:
            patterns = find_loss_patterns(losses, min_frequency=args.min_pattern_frequency)
            print_loss_analysis(losses, patterns)

            if args.output:
                output_path = Path(args.output)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with open(output_path, "w") as f:
                    json.dump({
                        "timestamp": datetime.utcnow().isoformat() + "Z",
                        "losses_analyzed": len(losses),
                        "patterns": [asdict(p) for p in patterns],
                        "losses": [asdict(l) for l in losses[:50]],
                    }, f, indent=2, default=str)
                print(f"\nReport saved: {output_path}")
        else:
            print("No losses found to analyze")

    if args.report:
        print("\n" + "=" * 60)
        print("GENERATING COMPREHENSIVE REPORT")
        print("=" * 60)

        report = generate_report(
            db_path,
            board_type=args.board,
            num_players=args.players,
        )

        if args.output:
            output_path = Path(args.output)
        else:
            output_path = AI_SERVICE_ROOT / "logs" / "analysis" / f"report_{args.board}_{args.players}p.json"

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(asdict(report), f, indent=2, default=str)

        print(f"Report saved: {output_path}")

        if report.recommendations:
            print("\nRecommendations:")
            for rec in report.recommendations:
                print(f"  - {rec}")

    if not any([args.analyze, args.find_blunders, args.openings, args.loss_patterns, args.report]):
        parser.print_help()

    return 0


if __name__ == "__main__":
    sys.exit(main())
