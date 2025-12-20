# Data Sync Integration Plan (Lane: Cluster Model/Data Sync)

## Goals

- Standardize model synchronization on `SyncCoordinator` (aria2/SSH/P2P) with NFS-aware skips.
- Align scripts and docs so `sync_models.py` usage is consistent and correct.
- Ensure cluster-wide model sync runs by default on nodes with shared storage.

## Scope

- `scripts/sync_models.py` CLI and config handling.
- Call sites that invoke `sync_models.py` (cron and external drive sync).
- Operational docs and runbooks that instruct model sync usage.

## Plan

1. Add explicit `--config` support to `sync_models.py` and thread it into daemon usage.
2. Update cron + external sync daemon to use `--use-sync-coordinator`.
3. Refresh docs/runbooks to recommend coordinator-backed sync (and remove stale flags).
4. Update promotion sync to call `sync_models.py` with supported flags.
5. Validate no stale references remain to unsupported flags in these docs.

## Status

- [x] CLI config support + daemon threading
- [x] Script call sites updated (cron + external drive)
- [x] Docs updated for coordinator usage
- [x] Promotion sync updated to supported flags
- [x] Docs checked for stale flags
