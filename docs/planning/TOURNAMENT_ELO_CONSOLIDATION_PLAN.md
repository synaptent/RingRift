# Tournament + Elo Consolidation Plan

## Goals

- Consolidate tournament + Elo entrypoints behind a small, consistent CLI surface.
- Guarantee canonical replay recording (RR-CANON-R075) with full metadata.
- Ensure distributed execution is sharded, fault tolerant, resumable, and consistent across nodes.
- Automatically ingest tournament games into training datasets with clear provenance.
- Use shared storage (NFS when available) and fast sync (aria2/P2P) for data distribution.

## Scope (What This Plan Touches)

- Core tournament engine: `ai-service/app/tournament/**`
- Tournament CLIs: `ai-service/scripts/run_tournament.py` and related runners
- Elo DB + dashboards: `ai-service/app/tournament/unified_elo_db.py`, `ai-service/scripts/*elo*`
- Sync + storage: `ai-service/scripts/aria2_data_sync.py`, `ai-service/app/sync/**`
- Training ingestion: `ai-service/scripts/export_replay_dataset.py`, `ai-service/scripts/aggregate_jsonl_to_db.py`
- Documentation and runbooks

## Non-Goals

- Rewriting AI algorithms or search logic
- Changing canonical rules or move semantics
- Replacing the TS engine as the SSoT

## Inventory (Current Assets To Preserve)

- `app/tournament/runner.py`: match execution + metadata + recording
- `app/tournament/distributed_gauntlet.py`: parallelized gauntlet with worker support
- `app/tournament/composite_gauntlet.py`: model+algorithm cross evaluation
- `scripts/run_tournament.py`: unified CLI wrapper
- `scripts/run_distributed_tournament.py`: resumable distributed ladder + training ingest
- `scripts/auto_elo_tournament.py`: scheduled tournaments with alerts
- `scripts/unified_loop/tournament.py`: remote host selection + orchestration
- `scripts/aria2_data_sync.py`: fast sync across nodes
- `scripts/aggregate_jsonl_to_db.py`: tournament JSONL -> GameReplayDB

## Plan (Phased)

### Phase 1 - Canonical Recording Hardening (Immediate)

1. Ensure all tournament execution paths use trace-mode application
   so forced elimination remains explicit and recorded.
2. Centralize recording options and metadata schema:
   - board_type, num_players, model_id, agent metadata, seed, config_key
   - move diversity + phase coverage scores
3. Add lightweight validation checks for canonical phase history on
   tournament DBs (opt-in or sampled).
4. Enforce canonical MoveType names at write time (no legacy aliases in
   tournament output; legacy replay only via explicit adapters).

### Phase 2 - Consolidate Entry Points

1. Make `scripts/run_tournament.py` the only supported CLI entrypoint.
2. Wrap legacy scripts as thin delegates that route into `run_tournament.py`
   and emit deprecation warnings.
3. Extract a shared `TournamentConfig` dataclass for:
   - matchmaking, board/player settings, seeds
   - recording + metadata
   - distributed execution parameters (workers, shard size, retry policy)

### Phase 3 - Distributed + Fault Tolerant Execution

1. Unify distributed orchestration under one scheduler layer
   (either extend `distributed_gauntlet` or `tournament/orchestrator`).
2. Add checkpoint + resume to all tournament modes.
3. Standardize worker heartbeat and retry policy:
   - per-match timeouts
   - worker-level retries
   - deterministic seeding per shard

### Phase 4 - Training Ingest + Provenance

1. Standardize tournament outputs in `GameReplayDB` format
   (not just JSON/JSONL).
2. Auto-ingest tournament DBs via `export_replay_dataset.py` or
   a dedicated ingestion wrapper, including:
   - source_db
   - canonical flag
   - parity hash
3. Track tournament-derived datasets in `TRAINING_DATA_REGISTRY.md`.

### Phase 5 - Data Sync + Shared Storage

1. Prefer shared NFS paths when available for:
   - unified_elo.db
   - tournament DBs
   - training datasets
2. Add an opt-in aria2 sync step post-tournament:
   - use `scripts/aria2_data_sync.py` for shards + summary outputs
3. Standardize output locations:
   - `ai-service/data/tournaments/`
   - `ai-service/data/games/`
   - `ai-service/data/training/`

### Phase 6 - Observability + Runbooks

1. Emit metrics for tournaments (duration, games/hr, win-rate variance).
2. Add parity + canonical history checks to the runbook.
3. Integrate into dashboards (Elo, throughput, failures).

## Immediate Next Steps (Starting Now)

1. Update tournament execution to use trace-mode moves to preserve
   explicit forced elimination and canonical phase histories.
2. Add a small unit test or smoke validation to ensure tournament runs
   can be replayed via canonical history checks (opt-in/sampled).
3. Add a tournament-time guard to reject legacy MoveTypes in new records.

---

## Implementation Status (2025-12-21)

### Phase 1: Canonical Recording Hardening - COMPLETED

See CODEBASE_CONSOLIDATION_PLAN.md Phase 3g: Tournament Recording Hardening

- [x] `trace_mode=True` enforced in all critical tournament scripts
- [x] 7 scripts updated to use trace_mode for canonical recording

### Phase 2: Consolidate Entry Points - SUBSTANTIALLY COMPLETE

- [x] `app/tournament/__init__.py` provides unified public API
- [x] `EloService` from `app.training.elo_service` marked as preferred SSoT
- [x] `EloDatabase` retained for backwards compatibility with deprecation guidance
- [x] High-level APIs available: `run_quick_tournament`, `create_tournament_runner`
- [x] `TournamentConfig` dataclass available for configuration

Scripts status:

- `run_tournament.py` - Primary entry point
- `run_model_elo_tournament.py` - Uses consolidated imports from `scripts/lib/tournament_cli.py`
- Legacy scripts have deprecation wrappers (see CODEBASE_CONSOLIDATION_PLAN.md Phase 3)

### Phase 3: Distributed + Fault Tolerant Execution - PARTIAL

- [x] `TournamentOrchestrator` provides distributed coordination
- [x] Checkpoint/resume available via `run_distributed_tournament.py`
- [ ] Full unification of distributed orchestration layers (deferred)

### Phase 4: Training Ingest + Provenance - COMPLETE

- [x] `GameReplayDB` format standardized
- [x] `export_replay_dataset.py` handles tournament DB ingestion
- [x] Canonical flag and parity hash tracked
- [x] `TRAINING_DATA_REGISTRY.md` tracks canonical databases

### Phase 5: Data Sync + Shared Storage - COMPLETE

- [x] NFS paths supported for unified_elo.db
- [x] aria2 sync available via `scripts/aria2_data_sync.py`
- [x] Standard output locations documented

### Phase 6: Observability + Runbooks - PARTIAL

- [x] Metrics emitted for tournaments
- [x] Elo dashboards integrated
- [ ] Full runbook documentation (lower priority)

**Summary:** Tournament consolidation is 85%+ complete. Main remaining work is unifying
distributed orchestration layers and completing observability runbooks.
