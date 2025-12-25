# Analysis Module

Comprehensive game analysis for RingRift selfplay data.

## Overview

This module provides game analysis capabilities:

- Game balance analysis (win rates, first-player advantage)
- Victory type distribution (territory, elimination, LPS)
- Recovery and forced elimination tracking
- Cluster-wide data aggregation
- AI type breakdown (heuristic, MCTS, Gumbel, neural net)

## Quick Start

```python
from app.analysis import GameBalanceAnalyzer, analyze_game_balance

# Quick analysis
report = analyze_game_balance(
    db_path="data/games/selfplay.db",
    board_type="hex8",
    num_players=2,
)
print(report.summary)
```

## Key Components

### GameBalanceAnalyzer

```python
from app.analysis import GameBalanceAnalyzer

analyzer = GameBalanceAnalyzer(
    db_path="data/games/selfplay.db",
    board_type="hex8",
    num_players=2,
)

# Full analysis
report = analyzer.analyze()
print(f"Total games: {report.total_games}")
print(f"First-player advantage: {report.first_player_advantage:+.1%}")
print(f"Draw rate: {report.draw_rate:.1%}")

# Find balance issues
issues = analyzer.find_balance_issues()
for issue in issues:
    print(f"[{issue.severity}] {issue.description}")

# Generate report
report_text = analyzer.generate_report(output_path="balance_report.txt")

# Analyze all configurations
cross_config = analyzer.analyze_all_configs()
```

### Balance Report

```python
from app.analysis import BalanceReport

report: BalanceReport = analyzer.analyze()

# Access statistics
report.total_games
report.board_type
report.num_players
report.draw_rate
report.avg_game_length

# Player win rates
for player, stats in report.player_win_rates.items():
    print(f"Player {player}: {stats.win_rate:.1%}")

# First-player analysis
report.first_player_advantage  # +5% = first player advantage
report.first_player_stats.confidence_interval

# Game length statistics
report.game_length_stats.mean
report.game_length_stats.median
report.game_length_stats.percentiles[90]

# Balance issues
for issue in report.balance_issues:
    print(f"{issue.category}: {issue.description}")
```

## CLI Scripts

### analyze_game_statistics.py

Comprehensive statistics with victory types, recovery, AI breakdown:

```bash
# Local analysis
python scripts/analyze_game_statistics.py --data-dir data/selfplay

# Filter by AI type
python scripts/analyze_game_statistics.py --data-dir data/selfplay --ai-types gumbel gumbel-mcts

# Recursive scan
python scripts/analyze_game_statistics.py --jsonl-dir /path/to/data --recursive

# Output formats
python scripts/analyze_game_statistics.py --data-dir data/selfplay --format markdown
python scripts/analyze_game_statistics.py --data-dir data/selfplay --format json --output report.json
```

### analyze_cluster_games.py

Cluster-wide analysis across all nodes:

```bash
# Scan all cluster nodes
python scripts/analyze_cluster_games.py --board-type hex8 --num-players 2

# Filter by Gumbel MCTS
python scripts/analyze_cluster_games.py --ai-type gumbel --board-type hex8

# Specific nodes
python scripts/analyze_cluster_games.py --nodes lambda-gh200-b lambda-gh200-d
```

### analyze_recovery_across_games.py

Recovery eligibility analysis:

```bash
python scripts/analyze_recovery_across_games.py --data-dir data/selfplay
```

## AI Type Categories

| AI Type      | Description         | Patterns                         |
| ------------ | ------------------- | -------------------------------- |
| `gumbel`     | Gumbel MCTS + NN    | gumbel, gumbel-mcts, gumbel_mcts |
| `mcts`       | Standard MCTS       | mcts, mcts-only                  |
| `heuristic`  | Rule-based          | heuristic, heuristic-only        |
| `neural_net` | Policy network only | nn-only, nn_only                 |
| `random`     | Random moves        | random, random-only              |
| `hybrid`     | Multiple algorithms | hybrid, mixed                    |

## Victory Types

| Type          | Description                  |
| ------------- | ---------------------------- |
| `territory`   | Won by territory threshold   |
| `elimination` | Opponent eliminated (forced) |
| `lps`         | Last player standing         |
| `stalemate`   | No valid moves (tiebreaker)  |
| `timeout`     | Game timeout                 |

## Metrics Tracked

- **Win Rates**: Per-player win rates with confidence intervals
- **First-Player Advantage**: Deviation from expected 50% (2p) or 33%/25% (3p/4p)
- **Victory Distribution**: Breakdown by victory type
- **Game Length**: Mean, median, std, percentiles
- **Recovery Usage**: Games with recovery slides, win rate impact
- **Forced Elimination**: Games with FE, comeback rate
- **AI Type Breakdown**: Games per AI type, victory types per AI

## See Also

- `app.utils.game_discovery` - Database discovery
- `app.training.data_quality` - Training data validation
- `app.db.game_replay` - Game replay database
