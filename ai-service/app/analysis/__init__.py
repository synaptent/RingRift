"""Game Analysis Module for RingRift AI service.

This module provides comprehensive game analysis capabilities:
- Game balance analysis (first-player advantage, win rates)
- Victory type distribution
- Recovery and forced elimination tracking
- Cluster-wide data aggregation
- AI type breakdown

Quick Start:
    from app.analysis import GameBalanceAnalyzer, analyze_game_balance

    # Local database analysis
    report = analyze_game_balance(
        db_path="data/games/selfplay.db",
        board_type="hex8",
        num_players=2,
    )
    print(report.summary)

    # Full analysis with GameBalanceAnalyzer
    analyzer = GameBalanceAnalyzer(db_path="data/games/selfplay.db")
    report = analyzer.analyze()
    issues = analyzer.find_balance_issues()

For cluster-wide analysis:
    from app.analysis import ClusterGameAnalyzer

    analyzer = ClusterGameAnalyzer()
    report = analyzer.analyze_cluster(
        board_type="hex8",
        num_players=2,
        ai_type_filter="gumbel",
    )

See Also:
    - scripts/analyze_game_statistics.py - CLI for detailed statistics
    - scripts/analyze_recovery_across_games.py - Recovery eligibility analysis
"""

from app.analysis.game_balance import (
    BalanceIssue,
    BalanceReport,
    CrossConfigAnalysis,
    GameBalanceAnalyzer,
    GameLengthStats,
    WinRateStats,
    analyze_game_balance,
)

__all__ = [
    # Core analyzer
    "GameBalanceAnalyzer",
    # Report types
    "BalanceReport",
    "BalanceIssue",
    "WinRateStats",
    "GameLengthStats",
    "CrossConfigAnalysis",
    # Convenience functions
    "analyze_game_balance",
]
