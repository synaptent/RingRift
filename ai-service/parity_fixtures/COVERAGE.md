# Parity Fixture Coverage

This document tracks the TS↔Python parity fixture coverage for the RingRift game engine.

## Summary

| Metric                                       | Count | Target | Status     |
| -------------------------------------------- | ----- | ------ | ---------- |
| **Total Fixtures**                           | 274   | 100+   | ✅ Exceeds |
| Main directory (`parity_fixtures/*.json`)    | 274   | -      | -          |
| Generated (`generated/*.json`)               | 5     | -      | -          |
| Test vectors (`tests/parity/vectors/*.json`) | 7     | -      | -          |

**Last Updated:** 2025-12-07

## Board Type Distribution

| Board Type          | Fixtures | Percentage |
| ------------------- | -------- | ---------- |
| Square8             | 145      | 52.9%      |
| Square19            | 77       | 28.1%      |
| Hexagonal           | 32       | 11.7%      |
| Other (legacy/test) | 20       | 7.3%       |

All three primary board types are well represented.

## Player Count Distribution

| Player Count | Fixtures | Percentage |
| ------------ | -------- | ---------- |
| 2-player     | 137      | 50.0%      |
| 3-player     | 62       | 22.6%      |
| 4-player     | 49       | 17.9%      |
| Unspecified  | 26       | 9.5%       |

## Move Type Coverage

| Move Type                  | Fixtures | Description                |
| -------------------------- | -------- | -------------------------- |
| `move_stack`               | 153      | Basic stack movement       |
| `overtaking_capture`       | 61       | Capture mechanics          |
| `continue_capture_segment` | 48       | Chain capture continuation |
| `process_territory_region` | 5        | Territory processing       |
| `forced_elimination`       | 4        | Forced elimination events  |
| `swap_sides`               | 2        | Side swap moves            |
| `place_ring`               | 1        | Ring placement phase       |

## Phase Coverage

### Python Engine Phases

| Phase                  | Occurrences | Status          |
| ---------------------- | ----------- | --------------- |
| `chain_capture`        | 65          | ✅ Covered      |
| `capture`              | 63          | ✅ Covered      |
| `movement`             | 48          | ✅ Covered      |
| `ring_placement`       | 48          | ✅ Covered      |
| `line_processing`      | 48          | ✅ Covered      |
| `territory_processing` | 2           | ⚠️ Low coverage |
| `forced_elimination`   | 4           | ⚠️ Low coverage |

### TypeScript Engine Phases

| Phase                  | Occurrences | Status          |
| ---------------------- | ----------- | --------------- |
| `movement`             | 125         | ✅ Covered      |
| `territory_processing` | 51          | ✅ Covered      |
| `line_processing`      | 48          | ✅ Covered      |
| `ring_placement`       | 35          | ✅ Covered      |
| `capture`              | 15          | ⚠️ Low coverage |

## Game Status Coverage

| Status      | Occurrences | Description     |
| ----------- | ----------- | --------------- |
| `active`    | 489         | Mid-game states |
| `completed` | 43          | Game end states |
| `finished`  | 16          | Game end states |

## FAQ Scenario Coverage

Based on typical FAQ scenarios (Q1-Q24 style):

| Category                         | Scenarios           | Coverage Status  |
| -------------------------------- | ------------------- | ---------------- |
| **Basic Movement (Q1-Q3)**       | Movement, placement | ✅ 202+ fixtures |
| **Capture Mechanics (Q4-Q7)**    | Captures, chains    | ✅ 109+ fixtures |
| **Line Formation (Q8-Q10)**      | Line scoring        | ✅ 48+ fixtures  |
| **Territory Control (Q11-Q14)**  | Territory claims    | ✅ 7+ fixtures   |
| **Forced Elimination (Q15-Q17)** | FE events           | ⚠️ 4 fixtures    |
| **Multi-player (Q18-Q21)**       | 3p/4p games         | ✅ 111+ fixtures |
| **Game End (Q22-Q24)**           | Victory conditions  | ✅ 59+ fixtures  |

### Key Position Types

| Position Type        | Status     | Notes                                   |
| -------------------- | ---------- | --------------------------------------- |
| Captures             | ✅ Good    | 109+ fixtures with capture moves        |
| Chain captures       | ✅ Good    | 48 `continue_capture_segment` moves     |
| Territory processing | ⚠️ Limited | Only 5 `process_territory_region` moves |
| Forced elimination   | ⚠️ Limited | 4 fixtures, consider expansion          |
| Game endings         | ✅ Good    | 59+ completed/finished states           |

## Fixture Sources

Fixtures are organized by source database prefix:

| Source Pattern         | Count | Description               |
| ---------------------- | ----- | ------------------------- |
| `selfplay_square8_*`   | ~125  | Square8 self-play games   |
| `selfplay_square19_*`  | ~55   | Square19 self-play games  |
| `selfplay_hexagonal_*` | ~30   | Hexagonal self-play games |
| `canonical_square*`    | 3     | Canonical test games      |
| `minimal_test_*`       | 12    | Minimal test scenarios    |
| `coverage_selfplay_*`  | 3     | Coverage-focused games    |
| Other                  | ~46   | Various test sources      |

## Recommendations

### Areas with Good Coverage

- ✅ Basic movement and ring placement
- ✅ Capture and chain capture mechanics
- ✅ Multi-player games (2p, 3p, 4p)
- ✅ All three board types
- ✅ Game end states

### Areas Needing More Coverage

- ⚠️ **Territory processing**: Only 5 fixtures with `process_territory_region` moves
- ⚠️ **Forced elimination**: Only 4 fixtures with FE moves
- ⚠️ **LPS (Last Player Standing) rounds**: Not explicitly tracked

### Future Generation Commands

To expand coverage in specific areas:

```bash
cd ai-service

# Generate from canonical square8 DB with key positions
python scripts/generate_parity_vectors.py \
  --db data/games/canonical_square8.db \
  --output parity_fixtures/expanded/ \
  --strategy key_positions \
  --max-vectors 30

# Generate from canonical square19 DB
python scripts/generate_parity_vectors.py \
  --db data/games/canonical_square19.db \
  --output parity_fixtures/expanded/ \
  --strategy key_positions \
  --max-vectors 30
```

## Test Verification

Run parity regression tests:

```bash
cd ai-service
python -m pytest tests/parity/test_hash_parity.py tests/parity/test_chain_capture_parity.py tests/parity/test_forced_elimination_sequences_parity.py tests/parity/test_line_and_territory_scenario_parity.py -v
```

### Latest Test Results (2025-12-07)

| Test Suite                | Passed | Failed | Notes                        |
| ------------------------- | ------ | ------ | ---------------------------- |
| Hash parity               | 7      | 0      | ✅ All pass                  |
| Chain capture parity      | 11     | 0      | ✅ All pass                  |
| Forced elimination parity | 11     | 0      | ✅ All pass                  |
| Line & territory parity   | 11     | 3      | ⚠️ Overlength line edge case |
| **Total**                 | **40** | **3**  | 93% pass rate                |

The 3 failures are related to `test_overlength_line_option2_segments_exhaustive` across all board types - a known edge case being addressed separately.

Note: The full fixture regression tests (`test_replay_parity_fixtures_regression.py`) are currently marked as `xfail` because they capture known divergences between TS and Python engines that are being addressed.

---

_Generated: 2025-12-07_
