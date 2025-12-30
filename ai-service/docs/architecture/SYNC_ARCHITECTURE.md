# Sync Architecture Documentation (Alias)

**Status:** Deprecated alias
**Canonical Doc:** `../sync_architecture.md`

This file is retained for backward links. The canonical sync architecture lives in
`../sync_architecture.md` and reflects the December 2025 consolidation:

- `cluster_data_sync.py` and `ephemeral_sync.py` were removed from `app/coordination/`.
- Their behavior is now implemented in `auto_sync_daemon.py` via strategy modes
  (BROADCAST and EPHEMERAL).

If you need historical details about the pre-consolidation modules, consult the
archive or dated consolidation reports under `../archive/`.
