# Mac Studio Coordinator Memory Leak Root Causes

Date: 2026-04-27

## Summary

The observed `master_loop.py --profile lean` RSS growth to 53.6 GB is consistent
with unbounded subprocess output capture and unbounded per-file sync caches inside
long-lived coordinator daemons, not with the lean profile running too many daemons.
The lean profile should remain restartable and small; a long-lived coordinator
process must not depend on perfect daemon memory behavior.

## Findings

### 1. Sync subprocesses captured large progress output in memory

Several sync paths used `asyncio.create_subprocess_exec(... stdout=PIPE,
stderr=PIPE)` or `subprocess.run(... capture_output=True)` while running
`rsync --progress` or `aws s3 sync`. For multi-GB DB transfers, progress output
can become large and is retained until `communicate()` returns. When these runs
live inside the master-loop process, that memory pressure is charged to
`com.ringrift.master-loop`.

Patched paths:

- `app/coordination/owc_sync_manager.py`
- `app/coordination/owc_import_daemon.py`
- `app/coordination/training_data_sync_daemon.py`
- `app/coordination/data_availability_daemon.py`
- `scripts/consolidate_owc_data.py`

Remediation: remove `--progress` from large transfers and route normal stdout to
`DEVNULL`, keeping only bounded stderr snippets for failure diagnostics.

### 2. OWC sync dedupe caches were unbounded

`OWCSyncManager` stores checksums and mtimes in `_file_checksums` and
`_file_mtimes`. These dictionaries had no size cap. A coordinator that sees
rotated gauntlet/tournament DB names, staging imports, and backup variants over
days can accumulate entries forever.

Remediation: cap the OWC dedupe caches with
`RINGRIFT_OWC_SYNC_MAX_CACHE_ENTRIES` (default 2048), retaining newest entries.

### 3. Existing OOM watchdog watched system RAM, not master-loop RSS

`_oom_watchdog_check()` uses system memory percentage. On a 192 GB machine, a
single 53.6 GB process can be pathological while still below the global 85%
threshold. This explains why the master loop could hoard memory without the
watchdog acting.

Remediation: add a per-process RSS guard:

- `RINGRIFT_MASTER_LOOP_RSS_BUDGET_GB` default `20`
- `RINGRIFT_MASTER_LOOP_MAX_UPTIME_HOURS` default `24`
- `RINGRIFT_MASTER_LOOP_SELF_GUARD_ENABLED` default `true`

When the guard trips, the master loop writes a restart-request heartbeat and
cooperatively exits so launchd can restart it.

## Non-root-cause checks

- `HandlerBase` event dedupe is bounded (`DEDUP_MAX_SIZE=1000`).
- `FeedbackLoopController` Elo history is trimmed.
- The lean profile count is not itself sufficient to explain 53 GB RSS.

## Residual Risk

Other sync modules may still contain isolated `capture_output=True` usage. This
patch covers the paths involved in OWC/internal rehydration and the unified OWC
manager, but it is not a repo-wide subprocess audit.
