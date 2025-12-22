# Hex8 Board Variant Design

> **Status:** Draft
> **Author:** Claude Code
> **Date:** 2025-12-16
> **Related:** RULES_CANONICAL_SPEC.md, AI_ARCHITECTURE.md

## 1. Overview

This document specifies the **hex8** board variant, a smaller hexagonal board designed to parallel `square8` the same way `hexagonal` (radius-12) parallels `square19`.

### 1.1 Board Type Naming

| Board Type  | Grid Type  | Size      | Cells | Parallel To |
| ----------- | ---------- | --------- | ----- | ----------- |
| `square8`   | Orthogonal | 8×8       | 64    | -           |
| `square19`  | Orthogonal | 19×19     | 361   | -           |
| `hex8`      | Hexagonal  | Radius 4  | 61    | `square8`   |
| `hexagonal` | Hexagonal  | Radius 12 | 469   | `square19`  |

### 1.2 Design Rationale

- **Faster games:** ~10x fewer cells than full hexagonal (61 vs 469)
- **Faster training:** Comparable complexity to square8 for rapid iteration
- **Complete hex mechanics:** All hexagonal movement, capture, line, and territory rules preserved
- **Testing/development:** Quick validation of hex-specific logic

---

## 2. Hex8 Specification

### 2.1 Board Geometry

| Parameter       | Hex8 Value  | Formula/Notes                   |
| --------------- | ----------- | ------------------------------- |
| `boardType`     | `hex8`      | New enum value                  |
| `radius`        | 4           | Playable radius from center     |
| `size`          | 9           | Bounding box dimension (2r + 1) |
| `totalSpaces`   | 61          | 3r² + 3r + 1 = 3(16) + 12 + 1   |
| `boardGeometry` | `hexagonal` | Same geometry as full hex       |

### 2.2 Game Parameters

| Parameter            | Hex8 Value        | Comparison                        |
| -------------------- | ----------------- | --------------------------------- |
| `ringsPerPlayer`     | 18                | Same as square8                   |
| `lineLength`         | 4 (2p) / 3 (3-4p) | Same as square8 per RR-CANON-R120 |
| `movementAdjacency`  | 6 (hex dirs)      | Same as hexagonal                 |
| `lineAdjacency`      | 6 (hex dirs)      | Same as hexagonal                 |
| `territoryAdjacency` | 6 (hex dirs)      | Same as hexagonal                 |

### 2.3 Size Semantics (Unified Convention)

**IMPORTANT:** The `size` field in BOARD_CONFIGS represents the **bounding box dimension**, not the radius.

| Board       | Radius | Size (Bounding Box) | Formula         |
| ----------- | ------ | ------------------- | --------------- |
| `hex8`      | 4      | 9                   | 2 × 4 + 1 = 9   |
| `hexagonal` | 12     | 25                  | 2 × 12 + 1 = 25 |

**Conversion formulas:**

- `size = 2 * radius + 1` (radius to bounding box)
- `radius = (size - 1) / 2` (bounding box to radius)

This convention is used **consistently** across:

- TypeScript: `src/shared/types/game.ts` (BOARD_CONFIGS)
- Python: `ai-service/app/rules/core.py` (BOARD_CONFIGS)
- All radius calculations throughout the codebase

**Rationale:** Using bounding box as `size` eliminates confusion between different size semantics and matches the natural tensor allocation dimensions for GPU/neural network code.

### 2.4 Coordinate System

Uses same axial coordinate system as full hexagonal:

- Cube coordinates `(x, y, z)` where `x + y + z = 0`
- Valid cells: `max(|x|, |y|, |z|) <= 4` (radius)
- Bounding box embedding: 9×9 grid with center at (4, 4)

### 2.5 Movement Directions

Same 6 hex directions as full hexagonal:

```
HEX_DIRS = [
    (+1,  0, -1),  # East
    (+1, -1,  0),  # Northeast
    ( 0, -1, +1),  # Northwest
    (-1,  0, +1),  # West
    (-1, +1,  0),  # Southwest
    ( 0, +1, -1),  # Southeast
]
```

---

## 3. Neural Network Policy Size

### 3.1 Policy Layout

```
HEX8_BOARD_SIZE = 9
HEX8_MAX_DIST = 8  # Maximum distance on radius-4 board

# Policy spans:
Placements:      9 × 9 × 3 = 243
Movement:        9 × 9 × 6 × 8 = 3,888
Special:         1 (skip_placement)
─────────────────────────────────────
Total:           4,132 → POLICY_SIZE_HEX8 = 4,500 (padded)
```

### 3.2 Comparison

| Board     | Policy Size | Ratio to Hex8 |
| --------- | ----------- | ------------- |
| hex8      | 4,500       | 1.0x          |
| square8   | 7,000       | 1.6x          |
| square19  | 67,000      | 14.9x         |
| hexagonal | 91,876      | 20.4x         |

---

## 4. Implementation Plan

### 4.1 Phase 1: Core Types (Priority: High)

**Files to modify:**

1. **`src/shared/types/game.ts`**
   - Add `HEX8 = 'hex8'` to `BoardType` enum

2. **`ai-service/app/models.py`**
   - Add `HEX8 = "hex8"` to `BoardType` enum

3. **`RULES_CANONICAL_SPEC.md`**
   - Add hex8 to RR-CANON-R001 board type configuration table
   - Add hex8 parameters to RR-CANON-R020 (ringsPerPlayer = 18)

### 4.2 Phase 2: Geometry & Rules (Priority: High)

**Files to modify:**

1. **`src/shared/engine/board/geometry.ts`**
   - Add hex8 case to `createBoardGeometry()`
   - Implement radius-4 hex grid generation

2. **`ai-service/app/rules/geometry.py`**
   - Add hex8 support to `BoardGeometry` class
   - Add `HEX8_RADIUS = 4` constant

3. **`src/shared/engine/rules/config.ts`**
   - Add hex8 configuration object with lineLength=4 (base), ringsPerPlayer=18
   - Note: Effective lineLength is 4 for 2-player, 3 for 3-4 player (per RR-CANON-R120)

### 4.3 Phase 3: Neural Network (Priority: Medium)

**Files to modify:**

1. **`ai-service/app/ai/neural_net.py`**
   - Add `HEX8_BOARD_SIZE = 9`
   - Add `POLICY_SIZE_HEX8 = 4500`
   - Add hex8 to `BOARD_POLICY_SIZES` and `BOARD_SPATIAL_SIZES` dicts
   - Add policy encoding/decoding for hex8

2. **`ai-service/app/training/config.py`**
   - Add hex8 to `BOARD_TRAINING_CONFIGS`
   - Add `get_training_config_for_board()` hex8 case

3. **`ai-service/app/training/encoding.py`**
   - Add hex8 spatial encoding (9×9 grid)

### 4.4 Phase 4: Selfplay & Training (Priority: Medium)

**Files to modify:**

1. **`ai-service/scripts/p2p_orchestrator.py`**
   - Add hex8 to board type routing
   - Configure hex8 game counts (similar to square8)

2. **`ai-service/scripts/run_hybrid_selfplay.py`**
   - Add hex8 support

3. **`ai-service/data/unified_elo.db`**
   - Add hex8 baseline participants
   - Initialize hex8 ELO ratings

### 4.5 Phase 5: UI & Client (Priority: Low)

**Files to modify:**

1. **`src/client/components/Board/HexBoard.tsx`**
   - Support hex8 rendering (smaller radius)

2. **`src/client/game/config.ts`**
   - Add hex8 game configuration

---

## 5. Testing Strategy

### 5.1 Unit Tests

- Hex8 geometry generation (61 cells, correct adjacencies)
- Coordinate conversion (axial ↔ bounding box)
- Policy encoding/decoding roundtrip
- Line detection on hex8 (length=4)
- Territory processing on hex8

### 5.2 Integration Tests

- Full game simulation (hex8, 2-4 players)
- TS/Python parity validation
- Selfplay smoke test (100 games)

### 5.3 Performance Benchmarks

- Game completion time vs square8 (should be comparable)
- Training throughput vs square8 (should be comparable)
- Policy inference latency (should be faster than hex due to smaller policy)

---

## 6. Migration Notes

### 6.1 Backward Compatibility

- No changes to existing board types
- Hex8 is additive (new enum value)
- Existing hexagonal games/models unaffected

### 6.2 Database Schema

No schema changes required. Hex8 games stored with `board_type = 'hex8'`.

### 6.3 Model Naming Convention

```
ringrift_v5_hex8_2p.pth
ringrift_v5_hex8_3p.pth
ringrift_v5_hex8_4p.pth
```

---

## 7. Resolved Design Questions

1. **Line length:** ✅ RESOLVED (RR-CANON-R120)
   - hex8 2-player: lineLength=4
   - hex8 3-4 player: lineLength=3 (same as square8 3-4p)
   - Rationale: Consistency with square8 rules - both 61-64 cell boards use the same
     player-count-dependent line length thresholds. The S-invariant (S = M + C + E)
     guarantees termination regardless of line formation rate per RR-CANON-R191.

2. **Victory thresholds:** Should hex8 use same victory thresholds as square8?
   - **Recommendation:** Yes, same thresholds for parallel behavior

3. **Tournament priority:** Should hex8 be included in standard tournament rotations?
   - **Recommendation:** Yes, after initial validation

---

## 8. References

- [RULES_CANONICAL_SPEC.md](../../RULES_CANONICAL_SPEC.md) - Canonical rules specification
- [AI_ARCHITECTURE.md](../architecture/AI_ARCHITECTURE.md) - Neural network architecture
- [neural_net.py](../../ai-service/app/ai/neural_net.py) - Policy size definitions
