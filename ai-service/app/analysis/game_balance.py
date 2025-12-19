"""Game Balance Analysis for RingRift.

Analyzes game outcomes to identify balance issues such as:
- First-player advantage
- Position/start location advantage
- Win rate disparities
- Game length distribution
- Draw rates

Provides statistical analysis and visualization support.

Usage:
    from app.analysis.game_balance import GameBalanceAnalyzer

    analyzer = GameBalanceAnalyzer(db_path="data/games/selfplay.db")

    # Generate balance report
    report = analyzer.analyze()
    print(report.summary)

    # Check for significant imbalances
    issues = analyzer.find_balance_issues()
"""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from app.utils.datetime_utils import iso_now

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class WinRateStats:
    """Win rate statistics for a category."""
    wins: int
    losses: int
    draws: int
    total_games: int
    win_rate: float  # 0-1
    confidence_interval: Tuple[float, float]  # 95% CI
    expected_rate: float  # Expected under fair conditions
    is_significant: bool  # Statistically significant deviation

    @property
    def advantage(self) -> float:
        """Advantage over expected (positive = better than expected)."""
        return self.win_rate - self.expected_rate


@dataclass
class GameLengthStats:
    """Game length statistics."""
    mean: float
    median: float
    std: float
    min: int
    max: int
    percentiles: Dict[int, float]  # 10, 25, 50, 75, 90


@dataclass
class BalanceIssue:
    """A detected balance issue."""
    category: str  # "first_player", "position", "board_type", etc.
    severity: str  # "minor", "moderate", "major"
    description: str
    affected_group: str
    win_rate: float
    expected_rate: float
    p_value: float
    confidence_interval: Tuple[float, float]
    recommendation: str


@dataclass
class BalanceReport:
    """Complete game balance analysis report."""
    # Metadata
    analysis_date: str
    total_games: int
    board_type: str
    num_players: int

    # Overall statistics
    draw_rate: float
    avg_game_length: float

    # Per-player statistics
    player_win_rates: Dict[int, WinRateStats]

    # First-player analysis
    first_player_advantage: float
    first_player_stats: WinRateStats

    # Game length analysis
    game_length_stats: GameLengthStats

    # Detected issues
    balance_issues: List[BalanceIssue]

    # Summary
    summary: str
    is_balanced: bool

    # Raw data for visualization
    win_distribution: Dict[int, int]  # player -> win count
    length_distribution: List[int]  # list of game lengths


@dataclass
class CrossConfigAnalysis:
    """Analysis across different configurations."""
    configs: List[Dict[str, Any]]  # board_type, num_players combos
    per_config_stats: Dict[str, BalanceReport]
    cross_config_issues: List[BalanceIssue]


class GameBalanceAnalyzer:
    """Analyzes game balance from historical data."""

    # Statistical thresholds
    SIGNIFICANCE_LEVEL = 0.05
    MINOR_DEVIATION = 0.05  # 5% deviation
    MODERATE_DEVIATION = 0.10  # 10% deviation
    MAJOR_DEVIATION = 0.15  # 15% deviation

    def __init__(
        self,
        db_path: Path,
        board_type: Optional[str] = None,
        num_players: Optional[int] = None,
    ):
        """Initialize analyzer.

        Args:
            db_path: Path to game database
            board_type: Optional filter by board type
            num_players: Optional filter by player count
        """
        self.db_path = Path(db_path)
        self.board_type = board_type
        self.num_players = num_players

    def _get_connection(self) -> sqlite3.Connection:
        """Get database connection."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _load_games(
        self,
        board_type: Optional[str] = None,
        num_players: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Load completed games from database."""
        conn = self._get_connection()
        cursor = conn.cursor()

        query = """
            SELECT game_id, board_type, num_players, winner,
                   move_history, created_at, completed_at
            FROM games
            WHERE status = 'completed'
        """
        params = []

        bt = board_type or self.board_type
        np_ = num_players or self.num_players

        if bt:
            query += " AND board_type = ?"
            params.append(bt)

        if np_:
            query += " AND num_players = ?"
            params.append(np_)

        query += " ORDER BY completed_at DESC"

        if limit:
            query += " LIMIT ?"
            params.append(limit)

        cursor.execute(query, params)
        games = [dict(row) for row in cursor.fetchall()]
        conn.close()

        return games

    def _calculate_win_rate_stats(
        self,
        wins: int,
        losses: int,
        draws: int,
        expected_rate: float,
    ) -> WinRateStats:
        """Calculate win rate statistics with confidence interval."""
        total = wins + losses + draws
        if total == 0:
            return WinRateStats(
                wins=0, losses=0, draws=0, total_games=0,
                win_rate=0.0, confidence_interval=(0.0, 0.0),
                expected_rate=expected_rate, is_significant=False,
            )

        win_rate = wins / total

        # Wilson score interval for proportion
        z = 1.96  # 95% CI
        n = total
        p = win_rate

        denominator = 1 + z**2 / n
        center = (p + z**2 / (2 * n)) / denominator
        margin = z * np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denominator

        ci = (max(0, center - margin), min(1, center + margin))

        # Statistical significance test (binomial test)
        # H0: win_rate = expected_rate
        p_value = stats.binom_test(wins, total, expected_rate, alternative='two-sided')
        is_significant = p_value < self.SIGNIFICANCE_LEVEL

        return WinRateStats(
            wins=wins,
            losses=losses,
            draws=draws,
            total_games=total,
            win_rate=win_rate,
            confidence_interval=ci,
            expected_rate=expected_rate,
            is_significant=is_significant,
        )

    def _calculate_game_length_stats(
        self,
        games: List[Dict[str, Any]],
    ) -> GameLengthStats:
        """Calculate game length statistics."""
        lengths = []
        for game in games:
            try:
                history = json.loads(game.get("move_history") or "[]")
                lengths.append(len(history))
            except (json.JSONDecodeError, TypeError):
                continue

        if not lengths:
            return GameLengthStats(
                mean=0, median=0, std=0, min=0, max=0,
                percentiles={10: 0, 25: 0, 50: 0, 75: 0, 90: 0}
            )

        return GameLengthStats(
            mean=float(np.mean(lengths)),
            median=float(np.median(lengths)),
            std=float(np.std(lengths)),
            min=int(np.min(lengths)),
            max=int(np.max(lengths)),
            percentiles={
                10: float(np.percentile(lengths, 10)),
                25: float(np.percentile(lengths, 25)),
                50: float(np.percentile(lengths, 50)),
                75: float(np.percentile(lengths, 75)),
                90: float(np.percentile(lengths, 90)),
            }
        )

    def analyze(
        self,
        board_type: Optional[str] = None,
        num_players: Optional[int] = None,
    ) -> BalanceReport:
        """Perform comprehensive balance analysis.

        Args:
            board_type: Board type to analyze (uses instance default if None)
            num_players: Number of players (uses instance default if None)

        Returns:
            BalanceReport with detailed analysis
        """
        bt = board_type or self.board_type or "square8"
        np_ = num_players or self.num_players or 2

        games = self._load_games(bt, np_)

        if not games:
            return BalanceReport(
                analysis_date=iso_now(),
                total_games=0,
                board_type=bt,
                num_players=np_,
                draw_rate=0.0,
                avg_game_length=0.0,
                player_win_rates={},
                first_player_advantage=0.0,
                first_player_stats=self._calculate_win_rate_stats(0, 0, 0, 1/np_),
                game_length_stats=self._calculate_game_length_stats([]),
                balance_issues=[],
                summary="No games available for analysis.",
                is_balanced=True,
                win_distribution={},
                length_distribution=[],
            )

        # Calculate basic statistics
        total_games = len(games)
        draws = sum(1 for g in games if g.get("winner") is None)
        draw_rate = draws / total_games

        # Win distribution per player
        win_distribution: Dict[int, int] = {p: 0 for p in range(np_)}
        for game in games:
            winner = game.get("winner")
            if winner is not None and winner in win_distribution:
                win_distribution[winner] += 1

        # Expected win rate under fair conditions
        expected_rate = 1 / np_ if np_ > 0 else 0.5

        # Player win rates
        player_win_rates = {}
        for player in range(np_):
            wins = win_distribution.get(player, 0)
            losses = total_games - wins - draws
            player_win_rates[player] = self._calculate_win_rate_stats(
                wins, losses, draws, expected_rate
            )

        # First-player analysis (player 0)
        first_player_stats = player_win_rates.get(0, self._calculate_win_rate_stats(
            0, 0, 0, expected_rate
        ))
        first_player_advantage = first_player_stats.advantage

        # Game length statistics
        game_length_stats = self._calculate_game_length_stats(games)

        # Length distribution
        length_distribution = []
        for game in games:
            try:
                history = json.loads(game.get("move_history") or "[]")
                length_distribution.append(len(history))
            except (json.JSONDecodeError, TypeError):
                continue

        # Detect balance issues
        balance_issues = self._detect_balance_issues(
            player_win_rates, first_player_advantage, draw_rate,
            game_length_stats, np_
        )

        # Generate summary
        summary = self._generate_summary(
            total_games, bt, np_, player_win_rates,
            first_player_advantage, draw_rate, balance_issues
        )

        # Overall balance assessment
        is_balanced = len([i for i in balance_issues if i.severity != "minor"]) == 0

        return BalanceReport(
            analysis_date=iso_now(),
            total_games=total_games,
            board_type=bt,
            num_players=np_,
            draw_rate=draw_rate,
            avg_game_length=game_length_stats.mean,
            player_win_rates=player_win_rates,
            first_player_advantage=first_player_advantage,
            first_player_stats=first_player_stats,
            game_length_stats=game_length_stats,
            balance_issues=balance_issues,
            summary=summary,
            is_balanced=is_balanced,
            win_distribution=win_distribution,
            length_distribution=length_distribution,
        )

    def _detect_balance_issues(
        self,
        player_win_rates: Dict[int, WinRateStats],
        first_player_advantage: float,
        draw_rate: float,
        game_length_stats: GameLengthStats,
        num_players: int,
    ) -> List[BalanceIssue]:
        """Detect potential balance issues."""
        issues = []

        # Check first-player advantage
        if abs(first_player_advantage) > self.MAJOR_DEVIATION:
            issues.append(BalanceIssue(
                category="first_player",
                severity="major",
                description=f"First player has {abs(first_player_advantage):.1%} {'advantage' if first_player_advantage > 0 else 'disadvantage'}",
                affected_group="Player 0 (first player)",
                win_rate=player_win_rates[0].win_rate,
                expected_rate=player_win_rates[0].expected_rate,
                p_value=0.0,  # Calculated from stats
                confidence_interval=player_win_rates[0].confidence_interval,
                recommendation="Consider alternating starting positions or implementing pie rule",
            ))
        elif abs(first_player_advantage) > self.MODERATE_DEVIATION:
            issues.append(BalanceIssue(
                category="first_player",
                severity="moderate",
                description=f"First player has {abs(first_player_advantage):.1%} {'advantage' if first_player_advantage > 0 else 'disadvantage'}",
                affected_group="Player 0 (first player)",
                win_rate=player_win_rates[0].win_rate,
                expected_rate=player_win_rates[0].expected_rate,
                p_value=0.0,
                confidence_interval=player_win_rates[0].confidence_interval,
                recommendation="Monitor trend; may need balancing adjustments",
            ))
        elif abs(first_player_advantage) > self.MINOR_DEVIATION:
            issues.append(BalanceIssue(
                category="first_player",
                severity="minor",
                description=f"First player has slight {abs(first_player_advantage):.1%} {'advantage' if first_player_advantage > 0 else 'disadvantage'}",
                affected_group="Player 0 (first player)",
                win_rate=player_win_rates[0].win_rate,
                expected_rate=player_win_rates[0].expected_rate,
                p_value=0.0,
                confidence_interval=player_win_rates[0].confidence_interval,
                recommendation="Within acceptable range; continue monitoring",
            ))

        # Check for significant player imbalances (other than first player)
        for player, stats in player_win_rates.items():
            if player == 0:
                continue  # Already checked

            deviation = abs(stats.advantage)
            if stats.is_significant and deviation > self.MODERATE_DEVIATION:
                issues.append(BalanceIssue(
                    category="player_position",
                    severity="moderate" if deviation < self.MAJOR_DEVIATION else "major",
                    description=f"Player {player} has {deviation:.1%} {'advantage' if stats.advantage > 0 else 'disadvantage'}",
                    affected_group=f"Player {player}",
                    win_rate=stats.win_rate,
                    expected_rate=stats.expected_rate,
                    p_value=0.0,
                    confidence_interval=stats.confidence_interval,
                    recommendation="Investigate position-specific advantages",
                ))

        # Check draw rate
        expected_draw_rate = 0.1  # Generally expect low draw rate
        if draw_rate > 0.3:
            issues.append(BalanceIssue(
                category="draw_rate",
                severity="moderate",
                description=f"High draw rate ({draw_rate:.1%})",
                affected_group="All players",
                win_rate=draw_rate,
                expected_rate=expected_draw_rate,
                p_value=0.0,
                confidence_interval=(0, 0),
                recommendation="Games may be too passive or lack decisive endgames",
            ))

        # Check game length variance
        if game_length_stats.std > game_length_stats.mean * 0.5:
            issues.append(BalanceIssue(
                category="game_length",
                severity="minor",
                description=f"High variance in game length (std={game_length_stats.std:.1f}, mean={game_length_stats.mean:.1f})",
                affected_group="All players",
                win_rate=0,
                expected_rate=0,
                p_value=0.0,
                confidence_interval=(0, 0),
                recommendation="Some games may be too short or too long",
            ))

        return issues

    def _generate_summary(
        self,
        total_games: int,
        board_type: str,
        num_players: int,
        player_win_rates: Dict[int, WinRateStats],
        first_player_advantage: float,
        draw_rate: float,
        balance_issues: List[BalanceIssue],
    ) -> str:
        """Generate human-readable summary."""
        summary_parts = [
            f"Analysis of {total_games} games ({board_type}, {num_players} players)",
            "",
        ]

        # Win rates
        summary_parts.append("Win Rates:")
        for player, stats in sorted(player_win_rates.items()):
            marker = " *" if stats.is_significant else ""
            summary_parts.append(
                f"  Player {player}: {stats.win_rate:.1%} "
                f"(CI: {stats.confidence_interval[0]:.1%}-{stats.confidence_interval[1]:.1%}){marker}"
            )

        summary_parts.append("")
        summary_parts.append(f"Draw Rate: {draw_rate:.1%}")
        summary_parts.append(f"First Player Advantage: {first_player_advantage:+.1%}")

        # Issues
        if balance_issues:
            summary_parts.append("")
            summary_parts.append("Balance Issues Detected:")
            for issue in balance_issues:
                summary_parts.append(f"  [{issue.severity.upper()}] {issue.description}")
        else:
            summary_parts.append("")
            summary_parts.append("No significant balance issues detected.")

        return "\n".join(summary_parts)

    def find_balance_issues(
        self,
        board_type: Optional[str] = None,
        num_players: Optional[int] = None,
        severity_filter: Optional[str] = None,
    ) -> List[BalanceIssue]:
        """Find balance issues matching criteria.

        Args:
            board_type: Board type to analyze
            num_players: Number of players
            severity_filter: Only return issues of this severity

        Returns:
            List of balance issues
        """
        report = self.analyze(board_type, num_players)

        if severity_filter:
            return [i for i in report.balance_issues if i.severity == severity_filter]

        return report.balance_issues

    def analyze_all_configs(self) -> CrossConfigAnalysis:
        """Analyze balance across all board/player configurations."""
        configs = [
            {"board_type": "square8", "num_players": 2},
            {"board_type": "square8", "num_players": 3},
            {"board_type": "square8", "num_players": 4},
            {"board_type": "square19", "num_players": 2},
            {"board_type": "hexagonal", "num_players": 2},
        ]

        per_config_stats = {}
        all_issues = []

        for config in configs:
            bt = config["board_type"]
            np_ = config["num_players"]
            key = f"{bt}_{np_}p"

            try:
                report = self.analyze(bt, np_)
                per_config_stats[key] = report
                all_issues.extend(report.balance_issues)
            except Exception as e:
                logger.warning(f"Error analyzing {key}: {e}")

        # Cross-config issues (e.g., one config much harder than others)
        cross_issues = []

        # Compare first-player advantages across configs
        fpa_values = {k: r.first_player_advantage for k, r in per_config_stats.items()}
        if fpa_values:
            fpa_mean = np.mean(list(fpa_values.values()))
            fpa_std = np.std(list(fpa_values.values()))

            for config, fpa in fpa_values.items():
                if fpa_std > 0 and abs(fpa - fpa_mean) > 2 * fpa_std:
                    cross_issues.append(BalanceIssue(
                        category="cross_config",
                        severity="moderate",
                        description=f"{config} has unusual first-player advantage ({fpa:+.1%} vs avg {fpa_mean:+.1%})",
                        affected_group=config,
                        win_rate=fpa + 0.5,  # Approximate
                        expected_rate=0.5,
                        p_value=0.0,
                        confidence_interval=(0, 0),
                        recommendation="This configuration may need specific balancing",
                    ))

        return CrossConfigAnalysis(
            configs=configs,
            per_config_stats=per_config_stats,
            cross_config_issues=cross_issues,
        )

    def generate_report(
        self,
        output_path: Optional[Path] = None,
        board_type: Optional[str] = None,
        num_players: Optional[int] = None,
    ) -> str:
        """Generate and optionally save a balance report.

        Args:
            output_path: Optional path to save report
            board_type: Board type to analyze
            num_players: Number of players

        Returns:
            Report as string
        """
        report = self.analyze(board_type, num_players)

        # Build detailed report
        lines = [
            "=" * 70,
            "GAME BALANCE ANALYSIS REPORT",
            "=" * 70,
            "",
            f"Generated: {report.analysis_date}",
            f"Board Type: {report.board_type}",
            f"Players: {report.num_players}",
            f"Total Games: {report.total_games}",
            "",
            "-" * 70,
            "SUMMARY",
            "-" * 70,
            report.summary,
            "",
            "-" * 70,
            "GAME LENGTH STATISTICS",
            "-" * 70,
            f"Mean: {report.game_length_stats.mean:.1f} moves",
            f"Median: {report.game_length_stats.median:.1f} moves",
            f"Std Dev: {report.game_length_stats.std:.1f}",
            f"Range: {report.game_length_stats.min} - {report.game_length_stats.max} moves",
            "",
        ]

        if report.balance_issues:
            lines.extend([
                "-" * 70,
                "BALANCE ISSUES",
                "-" * 70,
            ])
            for issue in report.balance_issues:
                lines.extend([
                    f"\n[{issue.severity.upper()}] {issue.category}",
                    f"  Description: {issue.description}",
                    f"  Affected: {issue.affected_group}",
                    f"  Win Rate: {issue.win_rate:.1%} (expected: {issue.expected_rate:.1%})",
                    f"  Recommendation: {issue.recommendation}",
                ])

        lines.extend([
            "",
            "-" * 70,
            f"Overall Assessment: {'BALANCED' if report.is_balanced else 'IMBALANCED'}",
            "-" * 70,
        ])

        report_text = "\n".join(lines)

        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                f.write(report_text)
            logger.info(f"Saved balance report to {output_path}")

        return report_text

    def track_balance_over_time(
        self,
        window_size: int = 1000,
        step_size: int = 100,
    ) -> List[Dict[str, Any]]:
        """Track balance metrics over time to detect trends.

        Args:
            window_size: Number of games per window
            step_size: Step between windows

        Returns:
            List of balance snapshots over time
        """
        games = self._load_games()
        if not games:
            return []

        snapshots = []
        n_games = len(games)

        for start in range(0, n_games - window_size + 1, step_size):
            window = games[start:start + window_size]

            # Calculate statistics for this window
            draws = sum(1 for g in window if g.get("winner") is None)
            draw_rate = draws / len(window)

            # First player wins
            first_player_wins = sum(1 for g in window if g.get("winner") == 0)
            fpa = first_player_wins / len(window) - 0.5  # Deviation from 50%

            # Get timestamp
            first_game = window[0]
            last_game = window[-1]

            snapshots.append({
                "start_index": start,
                "end_index": start + window_size,
                "first_game_date": first_game.get("created_at"),
                "last_game_date": last_game.get("created_at"),
                "first_player_advantage": fpa,
                "draw_rate": draw_rate,
                "games_in_window": len(window),
            })

        return snapshots


def analyze_game_balance(
    db_path: Path,
    board_type: str = "square8",
    num_players: int = 2,
    output_path: Optional[Path] = None,
) -> BalanceReport:
    """Convenience function to analyze game balance.

    Args:
        db_path: Path to game database
        board_type: Board type
        num_players: Number of players
        output_path: Optional path to save report

    Returns:
        BalanceReport
    """
    analyzer = GameBalanceAnalyzer(db_path, board_type, num_players)

    if output_path:
        analyzer.generate_report(output_path)

    return analyzer.analyze()
