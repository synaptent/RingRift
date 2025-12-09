# Hex Artifacts Deprecation Notice

This file documents **hexagonal** artifacts that were produced under the **old** board geometry (all of which have now been **removed** from the repo):

- **Radius 10** (13 sides total size = 21×21 embedding)
- **331 cells**
- **36 rings per player**

The current "ultimate hex" specification is **radius 12 (13‑side board)** with **469 cells** and **72 rings per player**, embedded in a **25×25** grid. All hex artifacts listed below are **deprecated** and **must not** be used for training, evaluation, or parity after the geometry change.

## Deprecated hex artifacts (old geometry, now deleted)

- `data/training/from_replays.hexagonal.npz` (deleted)
- `data/games/golden_hexagonal.db` (deleted)
- `data/games/selfplay_hex_mps_smoke.db` (deleted)
- `data/games/selfplay_hexagonal_4p.db` (deleted)
- `data/games/selfplay_hexagonal_3p.db` (deleted)
- `data/games/selfplay_hexagonal_2p.db` (deleted)
- `data/games/canonical_hex.db` (deleted)

## Policy going forward

- These artifacts were removed from the repo. Use historical git commits only for forensic/historical inspection; do **not** retrain or evaluate models with them.
- Regenerate new hex training data, parity fixtures, and models under the **radius 12 / 469‑cell / 72‑ring / 25×25** geometry.
- Any hex model trained on the old geometry is also **deprecated**; it cannot ingest the new 25×25 encoding or action space (`P_HEX = 91,876`).
