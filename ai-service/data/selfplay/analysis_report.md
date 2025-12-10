# RingRift Game Statistics Analysis Report

**Generated:** 2025-12-10T17:38:14.596171+00:00
**Total Games Analyzed:** 1,200
**Total Moves:** 24,970
**Data Sources:** 6

## 1. Victory Type Distribution

| Board/Players | Games | Territory | LPS | Stalemate | Ring Elim | Draw |
|---------------|-------|-----------|-----|-----------|-----------|------|
| hexagonal 2p | 300 | 0% (0) | 0% (0) | 0% (0) | 0% (0) | 0% |
| square19 2p | 50 | 8.0% (4) | 0% (0) | 88.0% (44) | 4.0% (2) | 0% |
| square8 2p | 520 | 3.5% (18) | 0% (0) | 0% (0) | 0.4% (2) | 0% |
| square8 3p | 300 | 0% (0) | 0% (0) | 0% (0) | 0% (0) | 0% |
| square8 4p | 30 | 63.3% (19) | 30.0% (9) | 6.7% (2) | 0% (0) | 0% |

## 2. Win Distribution by Player Position

### Hexagonal 2-player (300 games)

| Player | Wins | Win Rate |
|--------|------|----------|
| Player 1 | 17 | 5.7% |
| Player 2 | 23 | 7.7% |

*Expected win rate (uniform): 50.0%*

### Square19 2-player (50 games)

| Player | Wins | Win Rate |
|--------|------|----------|
| Player 1 | 20 | 40.0% |
| Player 2 | 30 | 60.0% |

*Expected win rate (uniform): 50.0%*

### Square8 2-player (520 games)

| Player | Wins | Win Rate |
|--------|------|----------|
| Player 1 | 232 | 44.6% |
| Player 2 | 277 | 53.3% |

*Expected win rate (uniform): 50.0%*

### Square8 3-player (300 games)

| Player | Wins | Win Rate |
|--------|------|----------|
| Player 1 | 79 | 26.3% |
| Player 2 | 111 | 37.0% |
| Player 3 | 94 | 31.3% |

*Expected win rate (uniform): 33.3%*

### Square8 4-player (30 games)

| Player | Wins | Win Rate |
|--------|------|----------|
| Player 1 | 5 | 16.7% |
| Player 2 | 6 | 20.0% |
| Player 3 | 12 | 40.0% |
| Player 4 | 7 | 23.3% |

*Expected win rate (uniform): 25.0%*

## 3. Game Length Statistics

| Configuration | Avg Moves/Game | Games/Second | Total Time |
|--------------|----------------|--------------|------------|
| hexagonal 2p | 0.0 | 0.000 | 0.0s |
| square19 2p | 332.6 | 0.045 | 18.7m |
| square8 2p | 4.5 | 21.041 | 24.7s |
| square8 3p | 0.0 | 0.000 | 0.0s |
| square8 4p | 199.5 | 0.541 | 55.4s |

## 4. Recovery Action Analysis

- **Games Analyzed:** 92
- **Games with Forced Elimination:** 22 (23.9%)
- **Games with Recovery Used:** 0
- **States Analyzed:** 30,446

### Recovery Condition Frequencies

| Condition | Frequency | % of States |
|-----------|-----------|-------------|
| has_markers | 27,161 | 89.2% |
| has_buried_rings | 23,009 | 75.6% |
| zero_controlled_stacks | 7,668 | 25.2% |
| zero_rings_in_hand | 1,940 | 6.4% |

### Conditions Met Distribution

| # Conditions | States | % |
|--------------|--------|---|
| 0_conditions | 648 | 2.1% |
| 1_condition | 6,734 | 22.1% |
| 2_conditions | 16,211 | 53.2% |
| 3_conditions | 6,790 | 22.3% |
| 4_conditions | 63 | 0.2% |

## 5. Key Findings

### Position Advantages
- **square8 4p:** Player 3 has advantage (40.0% win rate)
- **square19 2p:** Player 2 has advantage (60.0% win rate)

### Victory Type Patterns
- **square19 2p:** stalemate dominant (88%)
- **square8 2p:** territory dominant (3%)
- **square8 4p:** territory dominant (63%)

### Recovery Mechanic Status
- **Recovery is NOT being used** despite 63 states meeting all 4 conditions
- This suggests turn-skipping prevents recovery-eligible players from taking turns
