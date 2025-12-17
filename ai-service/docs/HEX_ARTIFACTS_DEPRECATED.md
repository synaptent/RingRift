# Hex Artifacts Deprecation Notice

> **Updated:** 2025-12-14

This file documents **hexagonal** artifacts that were produced under the **old** board geometry (radius-10) which have been **deprecated and removed**.

## Old Geometry (DEPRECATED)

- **Radius 10** (21×21 embedding)
- **331 cells**
- **36 rings per player**

## Current Geometry (CANONICAL)

- **Radius 12** (25×25 embedding)
- **469 cells**
- **96 rings per player**
- **Policy size:** 91,876

## Deprecated hex artifacts (old radius-10 geometry, deleted)

The following artifacts were produced under the old radius-10 geometry and have been removed:

- `data/training/from_replays.hexagonal.npz` (deleted)
- `data/games/golden_hexagonal.db` (deleted)
- `data/games/selfplay_hex_mps_smoke.db` (deleted)
- `data/games/selfplay_hexagonal_4p.db` (deleted)
- `data/games/selfplay_hexagonal_3p.db` (deleted)
- `data/games/selfplay_hexagonal_2p.db` (deleted)
- Old `canonical_hex.db` from radius-10 era (archived as `canonical_hex.db.archived_*`)

## Current Canonical Hex Data

A new `canonical_hex.db` was regenerated on 2025-12-13 under the radius-12 geometry:

- Location: `data/games/canonical_hex.db`
- Gate summary: `db_health.canonical_hex.json`
- Status: `canonical_ok=true`, `fe_territory_fixtures_ok=true`

See `TRAINING_DATA_REGISTRY.md` for canonical status of all databases.

## Policy going forward

- **Do not use** any hex artifacts from the old radius-10 geometry
- All new hex training, evaluation, and models must use **radius-12 / 469-cell / 96-ring / 25×25** geometry
- Neural network models trained on old geometry cannot be used; retrain from scratch
- Policy size for hex is now `54,244` (was different under old geometry)
