# Training Pipeline Status Report

> **Date:** 2025-12-12 (Updated 05:10 UTC)
> **Purpose:** Document training data pipeline gap and cluster sync status

---

## Executive Summary

Successfully synced 5.2GB of selfplay data from cloud cluster to Mac Studio. **Fixed:** Rewrote `jsonl_to_npz.py` to properly convert JSONL to training-compatible NPZ format using partial replay.

**Final Results:**

- **Data Synced:** 5.2GB SQLite DBs + 639MB JSONL (5,418 games)
- **Training Data Generated:** 250,715 positions from 5,000 games (25.27 MB NPZ)
- **Format:** Correct 56-channel features with sparse policy encoding
- **Previous Dataset:** 11,752 samples → **New Dataset:** 250,715 samples (21x increase)

---

## Resolution: jsonl_to_npz.py Rewrite

### Problem Solved

The original JSONL games had incompatible move sequences due to automatic line processing in selfplay. Full replay would fail when the game engine expected different moves than recorded.

### Solution: Partial Replay

The updated `jsonl_to_npz.py` script:

1. Replays games until first error (typically at line formation moves)
2. Extracts training data from successful portion (~50 positions per game)
3. Uses NeuralNetAI.\_extract_features() for proper 56-channel encoding
4. Produces sparse policy format compatible with train.py

### Key Changes

```python
# Replay until error, use last successful state
final_state = initial_state
moves_succeeded = 0
for move in moves:
    try:
        final_state = GameEngine.apply_move(final_state, move)
        moves_succeeded += 1
    except Exception:
        break  # Stop at first error

# Require at least 10 successful moves
if moves_succeeded < 10:
    raise ValueError(f"Only {moves_succeeded} moves succeeded")

# Extract features only from successful moves
for move_idx, move in enumerate(moves[:moves_succeeded]):
    # ... extract and store training samples
```

---

## Data Generated (2025-12-12 05:09 UTC)

### New Training Dataset

```
File: data/training/cloud_sq8_2p_partial.npz
Size: 25.27 MB
Games: 5,000
Positions: 250,715

Format:
- features: (250715, 56, 8, 8) float32
- globals: (250715, 20) float32
- values: (250715,) float32
- policy_indices: (250715,) object (sparse)
- policy_values: (250715,) object (sparse)
- values_mp: (250715, 4) float32
- num_players: (250715,) int32
```

### Comparison

| Dataset                        | Samples | Size     | Status                 |
| ------------------------------ | ------- | -------- | ---------------------- |
| Old (selfplay_square8_2p.npz)  | 11,752  | 1.4 MB   | Replaced               |
| New (cloud_sq8_2p_partial.npz) | 250,715 | 25.27 MB | **Ready for training** |

---

## Cluster Sync Status (2025-12-12 03:30 UTC)

### Selfplay DBs Synced to Mac Studio

| Source         | Size  | Games (24h) | Status                                    |
| -------------- | ----- | ----------- | ----------------------------------------- |
| Lambda H100    | 1.9GB | -           | Corrupted (active writes during transfer) |
| Lambda A10     | 1.3GB | 1,766       | Valid                                     |
| Vast 3090      | 1.5GB | 2,353       | Valid                                     |
| Vast 5090-Dual | 234MB | 831         | Valid                                     |
| Vast 5090-Quad | 258MB | 964         | Valid                                     |

**Location:** `~/Development/RingRift/ai-service/data/games/cloud_sync_20251212/`

### JSONL Data Synced

| Source                         | Games | Size  |
| ------------------------------ | ----- | ----- |
| Vast 3090 extra_w\*.jsonl      | 2,500 | 299MB |
| Vast 5090-Dual extra_w\*.jsonl | 2,500 | 301MB |
| H100 sq8_2p                    | 63    | 35MB  |
| A10 selfplay                   | 355   | 3.7MB |

**Combined:** `data/selfplay/combined_with_initial_state.jsonl` (5,000 games)

---

## Known Limitations

### Partial Replay Data Quality

- Games truncated at first replay error (typically ~12% into game)
- Values computed from partial game state, not true final outcome
- Training signal may be noisier than full-game data

### Root Cause (Unfixed)

The selfplay JSONL was recorded with automatic line processing (`FORCE_NO_LINE_FORMATION_CHOICE`), but move sequences include explicit `process_line` moves that don't replay correctly with the current engine.

**Future Fix:** Update selfplay to:

1. Record `game_history_entries` in SQLite for proper export
2. Or ensure JSONL move sequences are engine-version compatible

---

## Python Compatibility Fix

Added `from __future__ import annotations` to `neural_net.py` to support Python 3.9 on Mac Studio (type hints use `X | Y` syntax from Python 3.10+).

---

## Next Steps

1. ✅ **Completed:** Generate 250K training dataset from JSONL
2. **Next:** Start training `ringrift_v4_*` model with new dataset
   - Note: “v4” is a checkpoint lineage / `nn_model_id` prefix, not a Python model class.
   - There is no `RingRiftCNN_v4`; inspect actual architectures via `scripts/inspect_nn_checkpoint.py` and `docs/MPS_ARCHITECTURE.md`.
3. **Future:** Fix selfplay to record proper history or compatible moves
4. **Future:** Convert remaining JSONL from other board types (square19, hexagonal)

---

## Data Locations Summary

| Type                     | Path                                              | Size        | Status                                                                                |
| ------------------------ | ------------------------------------------------- | ----------- | ------------------------------------------------------------------------------------- |
| Cloud DBs                | `data/games/cloud_sync_20251212/`                 | 5.2GB       | Synced, no history                                                                    |
| JSONL with initial_state | `data/selfplay/combined_with_initial_state.jsonl` | ~600MB      | Processed                                                                             |
| **New Training NPZ**     | `data/training/cloud_sq8_2p_partial.npz`          | 25.27 MB    | **Ready**                                                                             |
| Old training NPZ         | `data/training/selfplay_square8_2p.npz`           | 1.4 MB      | Superseded                                                                            |
| v3 Models                | `models/ringrift_v3_*.pth`                        | ~112MB each | Training active (note: “v3” is a checkpoint prefix, not necessarily `RingRiftCNN_v3`) |

---

## Appendix: Mac Studio IP Change

Mac Studio moved from `192.168.1.69` to `10.0.0.62`.
Use mDNS: `ssh armand@mac-studio.local`

Use Homebrew Python for 3.10+ compatibility:

```bash
/opt/homebrew/bin/python3.11 scripts/jsonl_to_npz.py ...
```
