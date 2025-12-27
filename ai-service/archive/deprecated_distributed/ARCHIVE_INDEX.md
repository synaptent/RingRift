# Archive Index - Deprecated Distributed Infrastructure

This directory contains archived distributed infrastructure modules that have been superseded.

## Archive Date: December 26, 2025

## Archived Files

### 1. unified_data_sync.py (81KB)

- **Original Location**: `app/distributed/unified_data_sync.py`
- **Lines of Code**: 2,170
- **Deprecated**: December 2025
- **Will Be Removed**: Q2 2026
- **Replacement**: `AutoSyncDaemon` + `SyncCoordinator` + `SyncFacade`
- **Migration Guide**: See `README.md`

### 2. sync_orchestrator.py (37KB)

- **Original Location**: `app/distributed/sync_orchestrator.py`
- **Archived**: Earlier (date TBD)
- **Replacement**: TBD

## File Retention Policy

Archived files will be:

- **Preserved** until Q2 2026 for reference and legacy support
- **Not updated** with new features
- **Bug fixes only** for critical issues
- **Removed** in Q2 2026 cleanup

## Import Status

These modules are still available via `app.distributed.__init__.py` for backward compatibility.
They emit deprecation warnings when imported.

## Documentation

- `README.md` - Detailed migration guide
- Original module docstrings preserved in archived files
- Archive header added to each file explaining deprecation

## Questions?

See:

- `../../app/coordination/README.md` - New coordination framework
- `../../docs/CONSOLIDATION_STATUS_2025_12_19.md` - Architecture decisions
- `../../CLAUDE.md` - Migration examples and patterns
