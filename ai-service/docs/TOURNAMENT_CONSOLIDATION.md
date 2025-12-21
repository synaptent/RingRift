# Tournament Script Consolidation

This document tracks the consolidation of tournament-related scripts.

## Current State (December 2025)

### Canonical Implementations

| Script                                  | Purpose                                      | Status                               |
| --------------------------------------- | -------------------------------------------- | ------------------------------------ |
| `scripts/run_tournament.py`             | Unified entry point with modes               | **CANONICAL**                        |
| `scripts/run_distributed_tournament.py` | Tier difficulty ladder, training data export | **CANONICAL** for tier tournaments   |
| `scripts/unified_loop/tournament.py`    | Unified loop integration                     | **CANONICAL** for automated pipeline |

### Deprecated (With Notices)

| Script                                 | Replacement                      | Notes                             |
| -------------------------------------- | -------------------------------- | --------------------------------- |
| `scripts/shadow_tournament_service.py` | `unified_loop/tournament.py`     | Deprecation notice added Dec 2025 |
| `scripts/pipeline_dashboard.py`        | `scripts/dashboard.py pipeline`  | Deprecation notice added Dec 2025 |
| `scripts/elo_dashboard.py`             | `scripts/dashboard.py elo`       | Deprecation notice added Dec 2025 |
| `scripts/composite_elo_dashboard.py`   | `scripts/dashboard.py composite` | Deprecation notice added Dec 2025 |

### Specialized (Keep As-Is)

| Script                                      | Purpose                    | Notes                    |
| ------------------------------------------- | -------------------------- | ------------------------ |
| `scripts/run_ssh_distributed_tournament.py` | SSH-based remote execution | Used by p2p_orchestrator |
| `scripts/run_p2p_elo_tournament.py`         | Peer-to-peer evaluation    | Specialized P2P protocol |
| `scripts/auto_elo_tournament.py`            | Autonomous daemon          | Runs on cluster nodes    |
| `scripts/run_profile_tournament.py`         | Performance profiling      | Development tool         |

## Usage Guide

### For Tier Tournaments (D1-D10)

```bash
# Canonical approach
python scripts/run_distributed_tournament.py \
    --tiers D1,D2,D3,D4,D5 \
    --games-per-matchup 20 \
    --workers 8 \
    --record-training-data
```

### For Model Comparison

```bash
python scripts/run_tournament.py models \
    --model-a path/to/model_a.pt \
    --model-b path/to/model_b.pt \
    --games 50
```

### For Unified Loop Integration

```python
from scripts.unified_loop.tournament import ShadowTournamentService

service = ShadowTournamentService(config, state, event_bus)
await service.run_shadow_tournament(config_key)
```

## Future Consolidation

1. **Phase 1** (Completed): Add deprecation notices
2. **Phase 2** (Planned): Route all CLI usage through `run_tournament.py`
3. **Phase 3** (Planned): Merge common tournament logic into `app/tournament/`

## Related Files

- `app/tournament/runner.py` - Core tournament execution logic
- `app/tournament/orchestrator.py` - Tournament orchestration
- `app/tournament/unified_elo_db.py` - Elo database interface
