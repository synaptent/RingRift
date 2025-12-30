# Composite ELO System: NN + Algorithm Strength Measurement

## Overview

This document describes the design and implementation plan for a coherent ELO strength measurement system that tracks **combinations of Neural Networks (NNs) with AI search algorithms** as separate participants.

**Problem Statement:**
The same neural network performs very differently with different search algorithms:

- `ringrift_v5` + Policy-Only -> ~1400 Elo
- `ringrift_v5` + MCTS (800 sims) -> ~1800 Elo
- `ringrift_v5` + Gumbel MCTS (budget=200) -> ~2200 Elo

These represent fundamentally different playing strengths but were previously tracked as a single entity.

**Solution:**
Track `(NN, Algorithm, Config)` tuples as distinct participants in the ELO system.

**Implementation note (Dec 2025):**
Composite gauntlet baselines and phase-2 algorithms are now fixed in code. The
baseline list and default algorithm set below reflect the current implementation.
Algorithm-specific baselines remain a planned extension (not yet wired).

---

## 1. Composite Participant Identity

### 1.1 Participant ID Schema

```
participant_id = {nn_id}:{ai_type}:{config_hash}
```

**Components:**

- `nn_id`: Neural network identifier (e.g., `ringrift_v5_sq8_2p`) or `none` for non-NN
- `ai_type`: Search algorithm (e.g., `gumbel_mcts`, `mcts`, `descent`, `policy_only`)
- `config_hash`: Short encoding of algorithm configuration

**Config Hash Encoding:**
| Prefix | Meaning | Example |
|--------|---------|---------|
| `b{N}` | Budget/simulations | `b200` = 200 budget |
| `s{N}` | Search simulations | `s800` = 800 MCTS sims |
| `d{N}` | Difficulty level | `d6` = difficulty 6 |
| `t{N}` | Temperature | `t0.3` = temp 0.3 |
| `k{N}` | K-factor override | `k32` = K=32 |

**Examples:**

```
ringrift_v5_sq8_2p:gumbel_mcts:b200     # Gumbel with budget=200
ringrift_v5_sq8_2p:mcts:s800            # MCTS with 800 simulations
ringrift_v5_sq8_2p:descent:d6           # Descent at difficulty 6
ringrift_v5_sq8_2p:policy_only:t0.3     # Policy-only, temperature=0.3
none:heuristic:d2                       # Heuristic baseline (no NN)
none:random:d1                          # Random baseline
```

### 1.2 Participant Categories

| Category        | ID Pattern              | Description                  |
| --------------- | ----------------------- | ---------------------------- |
| **Baseline**    | `none:{algo}:d{N}`      | Fixed anchor points, no NN   |
| **Pure NN**     | `{nn}:policy_only:t{N}` | NN strength without search   |
| **NN+Search**   | `{nn}:{algo}:{config}`  | Full NN + search combination |
| **Search-Only** | `none:{algo}:{config}`  | Heuristic eval with search   |

### 1.3 Standard Configurations

```python
STANDARD_ALGORITHM_CONFIGS = {
    "random": {"difficulty": 1},
    "heuristic": {"difficulty": 2, "randomness": 0.3},
    "policy_only": {"temperature": 0.3},
    "mcts": {"simulations": 800, "c_puct": 1.5},
    "gumbel_mcts": {"budget": 200, "m": 16},
    "descent": {"difficulty": 6, "time_ms": 5000},
    "ebmo": {"direct_eval": True},
    "gmo": {"optim_steps": 50},
    "gmo_gumbel": {"budget": 150, "m": 16},
}
```

---

## 2. Database Schema Extensions

### 2.1 Participants Table Extensions

```sql
-- Add columns to existing participants table
ALTER TABLE participants ADD COLUMN nn_model_id TEXT;
ALTER TABLE participants ADD COLUMN nn_model_path TEXT;
ALTER TABLE participants ADD COLUMN ai_algorithm TEXT;
ALTER TABLE participants ADD COLUMN algorithm_config TEXT;  -- JSON
ALTER TABLE participants ADD COLUMN is_composite BOOLEAN DEFAULT FALSE;

-- Index for efficient queries
CREATE INDEX idx_participants_nn ON participants(nn_model_id);
CREATE INDEX idx_participants_algo ON participants(ai_algorithm);
CREATE INDEX idx_participants_nn_algo ON participants(nn_model_id, ai_algorithm);
```

### 2.2 Algorithm Baselines Table

```sql
-- Track algorithm-specific baseline ratings
CREATE TABLE IF NOT EXISTS algorithm_baselines (
    ai_algorithm TEXT NOT NULL,
    board_type TEXT NOT NULL,
    num_players INTEGER NOT NULL,
    baseline_elo REAL DEFAULT 1500.0,
    games_played INTEGER DEFAULT 0,
    last_updated REAL,
    PRIMARY KEY (ai_algorithm, board_type, num_players)
);
```

### 2.3 NN Performance Summary Table

```sql
-- Aggregate NN performance across algorithms
CREATE TABLE IF NOT EXISTS nn_performance_summary (
    nn_model_id TEXT NOT NULL,
    board_type TEXT NOT NULL,
    num_players INTEGER NOT NULL,
    best_algorithm TEXT,
    best_elo REAL,
    avg_elo REAL,
    algorithms_tested INTEGER DEFAULT 0,
    last_updated REAL,
    PRIMARY KEY (nn_model_id, board_type, num_players)
);
```

---

## 3. Gauntlet System

### 3.1 Two-Phase Gauntlet

**Phase 1: NN Quick Evaluation (Policy-Only)**

- Test all NNs using policy_only (fast, no search overhead)
- ~50 games each vs baselines
- Purpose: Eliminate weak NNs early
- Gate: Top 50% proceed to Phase 2

**Phase 2: Search Amplification**

- Surviving NNs tested with each search algorithm
- Algorithms: gumbel_mcts, mcts, descent, gmo_gumbel
- ~20 games per (NN, algorithm) pair
- Records separate Elo for each combination

### 3.2 Gauntlet Baselines (Current Implementation)

Composite gauntlet uses a fixed baseline set (not per-algorithm yet):

```
Baseline composite IDs (pinned):
|--- none:random:d1    # 400 Elo anchor
|--- none:heuristic:d2 # 1200 Elo heuristic baseline
|--- none:mcts:d4      # 1500 Elo MCTS_LIGHT baseline
`--- none:mcts:d6      # 1700 Elo MCTS_MEDIUM baseline
```

Algorithm-specific baselines are a planned extension; the composite gauntlet
currently uses the fixed baseline set above for all phases.

### 3.3 Gauntlet Efficiency

```
Naive approach: 50 NNs x 4 algorithms x 50 games = 10,000 games

Two-phase approach:
  Phase 1: 50 NNs x 50 games = 2,500 games
  Phase 2: 25 NNs x 4 algorithms x 20 games = 2,000 games
  Total: 4,500 games (55% reduction)
```

---

## 4. Tournament System

### 4.1 Tournament Types

**Algorithm Tournament (Weekly)**

```
Purpose: Rank search algorithms fairly
Method:
|--- Fix NN to "current best" model
|--- Round-robin (defaults): gumbel_mcts vs mcts vs descent vs policy_only
|--- Optional: include ebmo/gmo/gmo_gumbel via explicit algorithm list
|--- Same NN for all participants
`--- Isolates algorithm strength from NN quality

Output: Algorithm Elo ladder
```

**NN Tournament (Daily)**

```
Purpose: Rank neural networks fairly
Method:
|--- Fix algorithm to "standard" (Gumbel budget=200)
|--- Round-robin: NN_a vs NN_b vs NN_c...
|--- Same algorithm for all participants
`--- Isolates NN quality from algorithm choice

Output: NN Elo ladder
```

**Combined Tournament (Continuous)**

```
Purpose: Absolute strength ranking
Method:
|--- All (NN, algorithm) pairs compete
|--- Elo-based matchmaking (pair similar ratings)
|--- Swiss-style pairing for efficiency
`--- Full combinatorial space

Output: Global composite Elo leaderboard
```

### 4.2 Tournament Scheduling

```
+---------------------------------------------------------+
|                 Tournament Cycle                         |
|----------------------------------------------------------|
|                                                          |
|  Continuous:  Combined Tournament                        |
|               |--- ~10 games/hour (background)           |
|               `--- Updates global Elo continuously       |
|                                                          |
|  Daily:       NN Tournament (triggered by new models)   |
|               |--- Tests new NNs from training           |
|               |--- 50 games per new model                |
|               `--- Gates promotion to Combined           |
|                                                          |
|  Weekly:      Algorithm Tournament                       |
|               |--- Tests algorithm improvements          |
|               |--- 100 games per algorithm               |
|               `--- Updates algorithm baselines           |
|                                                          |
`----------------------------------------------------------+
```

---

## 5. Culling System

### 5.1 Hierarchical Culling Strategy

**Level 1: Cull Weak NNs**

- Condition: ALL (NN, \*) combinations are in bottom 50%
- Action: Archive the NN entirely (all algorithm variants)
- Impact: Removes ~40% of NNs

**Level 2: Cull Weak Algorithm Combinations**

- For surviving NNs, rank algorithm pairings
- Keep top 2 algorithms per NN
- Archive remaining combinations
- Ensures diversity while reducing clutter

**Level 3: Standard Elo Culling**

- Apply to remaining participants
- Standard 75% culling rule
- Based on absolute Elo ranking

### 5.2 Culling Safeguards

```python
CULLING_CONFIG = {
    "min_games_for_cull": 30,        # Don't cull high-uncertainty
    "min_participants_keep": 25,      # Always keep at least 25
    "protect_baselines": True,        # Never cull baselines
    "protect_best_per_algo": True,    # Keep best NN per algorithm
    "diversity_min_algos": 3,         # Keep at least 3 algorithm types
}
```

### 5.3 Archive vs Delete

- **Archive**: Move to `archived/` directory, mark in DB
- **Recoverable**: Can restore if needed
- **Metadata preserved**: Elo history, match records retained

---

## 6. Consistency Invariants

### 6.1 NN Ranking Consistency

```
Invariant: Same NN should rank similarly across algorithms

If NN_a > NN_b with Gumbel (70% win rate)
Then NN_a > NN_b with MCTS (expected ~65% win rate)

Significant violations indicate:
- Algorithm-specific NN properties
- Training data bias
- Architecture incompatibility
```

### 6.2 Algorithm Ranking Stability

```
Expected ranking (given same NN and time budget):
1. Gumbel MCTS (highest)
2. MCTS
3. Descent
4. Policy-Only (lowest)

Measured via Algorithm Tournament
```

### 6.3 Elo Transitivity

```
If A beats B (60%) and B beats C (60%)
Then A should beat C (~70%)

Track prediction accuracy:
- predicted_outcome = elo_expected(A, C)
- actual_outcome = empirical_win_rate(A, C)
- accuracy = 1 - |predicted - actual|
```

---

## 7. Implementation Sprints

### Sprint 1: Core Infrastructure (Week 1)

- [ ] Implement composite participant ID system
- [ ] Database schema migrations
- [ ] Participant parsing and creation utilities
- [ ] Unit tests for ID system

### Sprint 2: Gauntlet Enhancement (Week 2)

- [ ] Two-phase gauntlet implementation
- [ ] Algorithm-specific baselines
- [ ] Gauntlet result aggregation
- [ ] Integration with existing gauntlet runner

### Sprint 3: Tournament Types (Week 3)

- [ ] Algorithm Tournament implementation
- [ ] NN Tournament implementation
- [ ] Tournament scheduler
- [ ] Cross-algorithm matchup recording

### Sprint 4: Culling & Maintenance (Week 4)

- [ ] Hierarchical culling implementation
- [ ] NN performance summary aggregation
- [ ] Archive/restore utilities
- [ ] Consistency check monitoring

### Sprint 5: Integration & Polish (Week 5)

- [ ] Integration with training pipeline
- [ ] Event emission for composite ratings
- [ ] Dashboard/reporting updates
- [ ] Documentation and cleanup

---

## 8. API Changes

### 8.1 New Functions

```python
# Participant ID utilities
def make_composite_participant_id(
    nn_id: str | None,
    ai_type: str,
    config: dict
) -> str

def parse_composite_participant_id(
    participant_id: str
) -> tuple[str | None, str, dict]

def get_standard_config(ai_type: str) -> dict

# Gauntlet
def run_two_phase_gauntlet(
    nn_paths: list[str],
    board_type: str,
    num_players: int
) -> dict[str, dict[str, float]]  # nn_id -> {algo: elo}

# Tournaments
def run_algorithm_tournament(
    reference_nn: str,
    algorithms: list[str],
    board_type: str,
    num_players: int,
    games_per_matchup: int = 50
) -> dict[str, float]  # algo -> elo

def run_nn_tournament(
    nn_paths: list[str],
    reference_algo: str,
    board_type: str,
    num_players: int,
    games_per_matchup: int = 50
) -> dict[str, float]  # nn_id -> elo

# Culling
def run_hierarchical_culling(
    board_type: str,
    num_players: int,
    dry_run: bool = True
) -> CullingReport
```

### 8.2 Event Types

```python
# New events for composite system
EloCompositeUpdated = namedtuple('EloCompositeUpdated', [
    'nn_id', 'ai_type', 'config_hash',
    'board_type', 'num_players',
    'old_elo', 'new_elo', 'games_played'
])

NNPerformanceSummaryUpdated = namedtuple('NNPerformanceSummaryUpdated', [
    'nn_id', 'board_type', 'num_players',
    'best_algorithm', 'best_elo', 'avg_elo'
])

AlgorithmBaselineUpdated = namedtuple('AlgorithmBaselineUpdated', [
    'ai_type', 'board_type', 'num_players',
    'baseline_elo', 'games_played'
])
```

---

## 9. Migration Strategy

### 9.1 Backwards Compatibility

- Existing `participant_id` values remain valid
- Legacy IDs treated as `{id}:mcts:s800` (default algorithm)
- Gradual migration as new games played

### 9.2 Migration Script

```python
def migrate_to_composite_ids():
    """Migrate existing participants to composite ID format."""
    for p in get_all_participants():
        if ':' not in p.participant_id:
            # Legacy format - migrate
            new_id = f"{p.participant_id}:mcts:s800"
            update_participant_id(p.participant_id, new_id)

            # Add metadata
            update_participant_metadata(new_id, {
                'nn_model_id': p.participant_id,
                'ai_algorithm': 'mcts',
                'algorithm_config': '{"simulations": 800}',
                'is_composite': True,
            })
```

---

## 10. Success Metrics

| Metric                      | Target               | Measurement                              |
| --------------------------- | -------------------- | ---------------------------------------- |
| Elo prediction accuracy     | > 65%                | Match outcome vs Elo prediction          |
| Algorithm ranking stability | < 5% variance        | Weekly tournament results                |
| NN ranking consistency      | > 80% agreement      | Cross-algorithm correlation              |
| Gauntlet efficiency         | > 50% reduction      | Games played vs naive                    |
| Culling precision           | < 5% false positives | Archived models that would have improved |

---

## Appendix A: Configuration Reference

### A.1 Standard Algorithm Configurations

```python
STANDARD_ALGORITHM_CONFIGS = {
    "random": {
        "difficulty": 1,
    },
    "heuristic": {
        "difficulty": 2,
        "randomness": 0.3,
    },
    "policy_only": {
        "temperature": 0.3,
        "top_k": None,
    },
    "mcts": {
        "simulations": 800,
        "c_puct": 1.5,
        "use_neural_net": True,
    },
    "gumbel_mcts": {
        "budget": 200,
        "m": 16,
        "use_neural_net": True,
    },
    "descent": {
        "difficulty": 6,
        "time_ms": 5000,
        "use_neural_net": True,
    },
    "ebmo": {
        "direct_eval": True,
        "optim_steps": 20,
        "num_restarts": 3,
    },
    "gmo": {
        "optim_steps": 50,
        "learning_rate": 0.1,
    },
}
```

### A.2 Board-Specific Adjustments

```python
BOARD_ALGORITHM_ADJUSTMENTS = {
    "square19": {
        "mcts": {"simulations": 400},  # Larger board, fewer sims
        "gumbel_mcts": {"budget": 100},
        "descent": {"time_ms": 10000},
    },
    "hexagonal": {
        "mcts": {"simulations": 600},
        "gumbel_mcts": {"budget": 150},
    },
}
```

---

## Appendix B: Example Queries

```sql
-- Get all participants for a specific NN
SELECT * FROM participants
WHERE nn_model_id = 'ringrift_v5_sq8_2p';

-- Get best algorithm per NN
SELECT nn_model_id, ai_algorithm, MAX(rating) as best_elo
FROM participants p
JOIN elo_ratings e ON p.participant_id = e.participant_id
WHERE board_type = 'square8' AND num_players = 2
GROUP BY nn_model_id;

-- Algorithm leaderboard (using best NN)
SELECT ai_algorithm, AVG(rating) as avg_elo, COUNT(*) as nn_count
FROM participants p
JOIN elo_ratings e ON p.participant_id = e.participant_id
WHERE board_type = 'square8' AND num_players = 2
GROUP BY ai_algorithm
ORDER BY avg_elo DESC;

-- NN consistency check (variance across algorithms)
SELECT nn_model_id,
       MIN(rating) as min_elo,
       MAX(rating) as max_elo,
       MAX(rating) - MIN(rating) as elo_spread
FROM participants p
JOIN elo_ratings e ON p.participant_id = e.participant_id
WHERE board_type = 'square8' AND num_players = 2
GROUP BY nn_model_id
HAVING COUNT(*) >= 3
ORDER BY elo_spread DESC;
```

---

_Document Version: 1.0_
_Created: 2025-12-20_
_Author: Claude Code_
