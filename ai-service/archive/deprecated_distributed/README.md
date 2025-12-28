# Deprecated Distributed Modules

Archived December 28, 2025 as part of dead code cleanup.

## Archived Files

### registries/ (2,452 LOC)

These registry modules were never imported from outside the registries folder:

- `base.py` - Base registry class (162 LOC)
- `game_registry.py` - Game location registry (400 LOC)
- `model_registry.py` - Model registry (289 LOC)
- `npz_registry.py` - NPZ file registry (325 LOC)
- `checkpoint_registry.py` - Checkpoint registry (366 LOC)
- `node_inventory.py` - Node inventory manager (363 LOC)
- `replication.py` - Replication manager (517 LOC)

**Replaced by:** `app/distributed/data_catalog.py`, `app/distributed/cluster_manifest.py`

### \_deprecated_serf_client.py (751 LOC)

SWIM protocol client that was never used (optional dependency not installed).

### \_deprecated_torrent_manager.py (548 LOC)

BitTorrent manager class that was never imported. BitTorrent functionality is handled by `aria2_transport.py` directly.

## Total Archived: 3,751 LOC

## Verification

All files had zero imports from outside their module.

## Migration

No migration needed - these modules were never used by any production code.
