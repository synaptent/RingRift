# Hex Data Deprecation Notice

⚠️ **DEPRECATED & REMOVED: All hex artifacts generated before 2025-12-06 are incompatible with the current hex board geometry. Legacy files have been deleted from the repo.**

## What Changed

As of December 6, 2025, the canonical hex board geometry was updated:

**Old (DEPRECATED):**

- Radius: 10
- Side length: 11
- Total cells: 331
- Rings per player: 36
- Neural network: 21×21 grid, P_HEX = 54,244

**New (CURRENT):**

- Radius: 12
- Side length: 13
- Total cells: 469
- Rings per player: 96
- Neural network: 25×25 grid, P_HEX = 91,876

## Deprecated Artifacts

The following hex-related artifacts are **DEPRECATED** and should not be used for training or evaluation:

### Game Replay Databases (\*.db) — **Removed**

- `data/games/canonical_hex.db` (radius 10, 331 cells) — **deleted**
- `data/games/golden_hexagonal.db` (radius 10, 331 cells) — **deleted**
- `data/games/selfplay_hexagonal_2p.db` (radius 10, 331 cells) — **deleted**
- `data/games/selfplay_hexagonal_3p.db` (radius 10, 331 cells) — **deleted**
- `data/games/selfplay_hexagonal_4p.db` (radius 10, 331 cells) — **deleted**
- `data/games/selfplay_hex_mps_smoke.db` (radius 10, 331 cells) — **deleted**

### Training Data (\*.npz) — **Removed**

- `data/training/from_replays.hexagonal.npz` (21×21 grid, 36 rings) — **deleted**

### Neural Network Models (\*.pth)

- Any `ringrift_v1_hex.pth` or similar hex models — **deprecated** (21×21 input, wrong policy size)

### Eval Pools — **Removed**

- `data/eval_pools/hex/pool_v1.jsonl` (radius 10, 331 cells) — **deleted**
- `data/eval_pools/hex_3p/pool_v1.jsonl` (radius 10, 331 cells) — **deleted**
- `data/eval_pools/hex_4p/pool_v1.jsonl` (radius 10, 331 cells) — **deleted**

## Action Required

1. **DO NOT** use deprecated hex artifacts for new training runs (files are no longer present).
2. **DO NOT** load deprecated hex models for evaluation.
3. **Regenerate** all hex-specific data:
   - Run new self-play soaks with the updated geometry
   - Retrain hex models with 25×25 input and P_HEX=91,876
   - Regenerate contract vectors and parity fixtures for hex

## Migration Path

To regenerate hex training data with the new geometry:

```bash
# Generate new hex self-play data (2-player)
python -m app.training.generate_data \
  --board-type hexagonal \
  --num-players 2 \
  --num-games 1000 \
  --record-db data/games/canonical_hex_r12.db \
  --output data/training/hex_r12.npz

# Verify new geometry
# Expected: 469 total valid positions, radius 12
```

## Preservation for Historical Analysis

Legacy paths above are listed for reference only; the files have been removed from the repo. Use historical git commits if you need to inspect old geometry data for ablation or forensic analysis, but do not reintroduce them into training or evaluation pipelines.

## References

- Core config update: `src/shared/types/game.ts`, `ai-service/app/models/core.py`
- Neural network update: `ai-service/app/ai/neural_net.py`
- Encoding update: `ai-service/app/training/encoding.py`
- Rules update: `RULES_CANONICAL_SPEC.md`, `ringrift_complete_rules.md`

---

**Last Updated:** 2025-12-06  
**Deprecation Reason:** Hex board geometry redesign (radius 10→12, "ultimate" variant)
