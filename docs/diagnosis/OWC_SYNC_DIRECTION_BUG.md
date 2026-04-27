# OWC Sync Direction Bug

Date: 2026-04-27

## Summary

The unified `OWC_SYNC_MANAGER` is not the direct code path that rehydrates OWC
databases back into `ai-service/data/games`. Its documented pull direction is
cluster/S3 to OWC. The internal disk refill is caused by separate pull/import
paths that treated remote OWC/S3 gauntlet data as default-restorable local data.

The most direct gauntlet rehydration path is:

- `scripts/sync_gauntlet_for_training.sh`: `aws s3 sync ... --include "gauntlet_*.db" data/games/`

Other internal rehydration paths:

- `app/coordination/owc_import_daemon.py`: OWC to `data/games/owc_imports`
- `app/coordination/training_data_sync_daemon.py`: OWC to local training files
- `app/coordination/data_availability_daemon.py`: OWC/S3/P2P to local data dirs
- `scripts/consolidate_owc_data.py`: OWC to local staging

## Lifecycle of gauntlet DBs

- Creation: gauntlet/tournament/evaluation runners write `gauntlet_*.db`,
  `baseline_calibration_*.db`, and `tournament_*.db` under `data/games`.
- Backup: S3 and OWC backup paths preserve those DBs for cold storage and
  possible offline analysis.
- Cleanup: `DiskSpaceManagerDaemon._rotate_large_evaluation_dbs()` rotates large
  evaluation DBs over 10 GB and expires old `.bak` files.
- Rehydration before this patch: `sync_gauntlet_for_training.sh` and OWC import
  flows could copy large gauntlet DBs back to internal storage even after an
  operator intentionally deleted them.
- Consumption: current GH200 experiments do not require mac-studio to keep
  square19/hexagonal inactive gauntlet DBs locally.

## Intended Direction Semantics

Default behavior is now push-only for internal rehydration:

- Local canonical DBs, NPZ, and models may be pushed to OWC/S3.
- Cluster/S3 may still be pulled to OWC external storage.
- Remote-to-internal pulls require an explicit downstream consumer signal and
  an allowlist entry in `ai-service/data/sync_policy.yaml`.
- Gauntlet/evaluation DBs are denied by default even if present on OWC/S3.

## Policy File

`ai-service/data/sync_policy.yaml` controls internal rehydration:

```yaml
internal_write_min_free_gb: 10

pull:
  default_allowed: false
  require_consumer_signal: true
  gauntlet_allowed: false
  allowlist: []
```

This means deleting an inactive `gauntlet_*.db` from internal storage remains a
durable operator decision unless an operator explicitly changes policy and
provides a consumer signal.

## Disk Pressure Gap

Before this patch, some sync paths checked percentage thresholds or staging
sizes, but they did not share a hard absolute free-space backoff for internal
writes. A 926 GB disk can lose tens of GB to one rehydration cycle before a
percentage-based cleanup policy reacts.

Remediation:

- `sync_policy.should_backoff_internal_write()` blocks internal pulls below the
  policy threshold.
- `DiskSpaceManagerDaemon` exposes `sync_backoff_active` and blocks writes when
  free space drops below `RINGRIFT_DISK_SPACE_SYNC_BACKOFF_FREE_GB` (default 10).
- `sync_gauntlet_for_training.sh` now exits unless
  `RINGRIFT_GAUNTLET_PULL_FOR_TRAINING=true`.
