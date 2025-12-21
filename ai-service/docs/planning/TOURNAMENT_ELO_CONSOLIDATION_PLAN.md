# Tournament + Elo Consolidation Plan (2025-12-20)

## Goals

- Consolidate tournament/Elo execution into a small set of shared, canonical pipelines.
- Ensure **all tournament games record full canonical move history** (RR-CANON-R075/R076).
- Make tournaments **distributed, sharded, fault-tolerant**, and consistent across nodes.
- Enforce **quality + diversity scoring** and track metadata required for training reuse.
- Automatically **ingest tournament games into training** (with explicit gates).

---

## 1) Inventory (Current Entry Points + Unique Strengths)

### Unified entrypoint (wrapper)

- `scripts/run_tournament.py`
  - Aggregates multiple modes but mostly delegates to legacy scripts.
  - Good CLI UX; needs consolidation into core pipeline.

### Distributed/cluster orchestration

- `scripts/run_distributed_tournament.py`
  - Thread-pool parallel execution, checkpoint/resume, Wilson intervals, JSONL output.
  - **Gap**: does not store canonical GameReplayDB by default.
- `scripts/run_ssh_distributed_tournament.py`
  - SSH-based host distribution.
- `scripts/launch_distributed_elo_tournament.py`
  - Cluster health checks + node orchestration; can pause selfplay.
  - Uses HTTP endpoint `/tournament/play_elo_match`.
- `scripts/run_diverse_tournaments.py`
  - Multi-board and multi-player coverage; distributed or local; continuous mode.

### Model-vs-model Elo

- `scripts/run_model_elo_tournament.py`
  - Model discovery, Elo matchmaking, persistent leaderboard.
  - **Training-mode** support (changes source tag).
- `scripts/auto_elo_tournament.py`, `scripts/run_p2p_elo_tournament.py`
  - Automated/continuous Elo flow and P2P orchestration.

### Evaluation + specialty tournaments

- `scripts/run_eval_tournaments.py`
  - State-pool evaluation (mid/late game snapshots).
- `scripts/run_tournament.py` (`weights` mode; legacy entrypoint: `scripts/deprecated/run_axis_aligned_tournament.py`), `scripts/run_profile_tournament.py`
  - Heuristic profile diagnostics and structured evaluation.
- `scripts/run_tournament.py` (`crossboard` mode; legacy entrypoint: `scripts/deprecated/run_crossboard_difficulty_tournament.py`)
  - Multi-board difficulty comparisons.

### Gauntlets / composite Elo

- `app/tournament/composite_gauntlet.py`, `scripts/run_composite_gauntlet.py`
  - Algorithm-vs-NN and NN-vs-NN composite Elo.
- `app/tournament/distributed_gauntlet.py`, `scripts/run_gauntlet.py`
  - Sharded gauntlet execution with culling hooks.

### Elo storage layers (duplication)

- `app/tournament/unified_elo_db.py` (legacy unified DB)
- `app/training/elo_service.py` (preferred)
- `app/tournament/elo.py` (in-memory calculator)

---

## 2) Target Architecture (Consolidated Pipeline)

### Single Tournament Pipeline

```
CLI (scripts/run_tournament.py) → app/tournament/pipeline.py
    → Scheduler (RoundRobin/Swiss/Custom)
    → Executor (local / SSH / cluster / HTTP)
    → Recorder (UnifiedGameRecorder → GameReplayDB)
    → Quality + Diversity scoring (app/quality/unified_quality.py)
    → Elo update (app/training/elo_service.py)
    → Training ingestion (build_canonical_training_pool_db.py + export)
```

### Canonical Recording Contract

- **All moves recorded** (including `no_*_action`, `skip_*`, `forced_elimination`).
- Stored in **GameReplayDB** using `UnifiedGameRecorder`.
- Metadata includes:
  - tournament_id, match_id, seed, board_type, num_players
  - AI specs (ai_type, difficulty, model_id, model_path, algorithm config)
  - engine_mode, source, termination_reason, winner, duration, game_length
  - quality_score, diversity_score, phase_balance_score

### Elo Consolidation

- **EloService** (`app/training/elo_service.py`) is the SSoT.
- All scripts update Elo through a shared adapter (no direct sqlite writes).
- DB sync uses `scripts/elo_db_sync.py` + `scripts/aria2_data_sync.py`.

### Training Ingestion

- Tournament DBs ingested into training pool with:
  - `scripts/build_canonical_training_pool_db.py --include-tournament`
  - Canonical history validation + parity gate.
  - Quality filter (min score) and diversity filter.

---

## 3) Consolidation Plan (Workstreams)

### Phase A — Canonical Recording & Metadata (Highest priority)

1. Add **TournamentRecordingOptions** to core `TournamentRunner`.
2. Record every move via `UnifiedGameRecorder`.
3. Include full metadata (agent specs, seeds, tournament IDs).
4. Score quality/diversity and store in metadata.

### Phase B — Distributed & Sharded Execution

1. Create shared executor interface: local, SSH, HTTP, cluster.
2. Merge `run_distributed_tournament.py`, `run_ssh_distributed_tournament.py`,
   and `launch_distributed_elo_tournament.py` into pipeline executors.
3. Add retry + checkpoint for match-level failures.
4. Ensure shard IDs + worker IDs stored in metadata.

### Phase C — Elo Service Consolidation

1. Replace direct `unified_elo_db` writes with `EloService`.
2. Provide adapter for legacy scripts (read-only or compatibility).
3. Centralize Elo sync/validation via `elo_db_sync.py`.

### Phase D — Training Ingestion Pipeline

1. Convert tournament DBs into canonical training pool via gate.
2. Auto-export training samples (`export_replay_dataset.py`) after ingestion.
3. Use quality + diversity thresholds; record metrics for ingestion volume.

### Phase E — Deprecation & Documentation

1. Route legacy CLIs through `scripts/run_tournament.py`.
2. Add warnings in legacy scripts; update docs to new pipeline.
3. Add focused tests: canonical recording, metadata schema, Elo updates.

---

## 4) Immediate Next Steps (Step 1 Execution)

**Step 1 scope (current):**

- Wire canonical GameReplayDB recording into model Elo + basic tournaments:
  - `scripts/run_model_elo_tournament.py`
  - `scripts/run_tournament.py` (basic mode → TournamentRunner path).
- Attach canonical metadata for every recorded game:
  - `tournament_id`, `matchup_id`, `game_num`
  - `board_type`, `num_players`, `engine_mode`, `source`
  - AI specs: `ai_type`, `difficulty`, `model_id`, `model_path`
  - `winner`, `game_length`, `duration_sec`, `termination_reason`
  - `node_id`, `worker_id`, `shard_id` (when available)
- Keep legacy JSONL export as optional output while DB recording becomes the SSoT.

---

## 5) Migration Map (Script → Pipeline Endpoint)

| Script                                 | Current Role          | Target                           |
| -------------------------------------- | --------------------- | -------------------------------- |
| `run_tournament.py`                    | wrapper               | primary CLI into pipeline        |
| `run_distributed_tournament.py`        | local thread pool     | pipeline executor (local)        |
| `run_ssh_distributed_tournament.py`    | SSH orchestration     | pipeline executor (SSH)          |
| `launch_distributed_elo_tournament.py` | cluster orchestration | pipeline executor (cluster/HTTP) |
| `run_model_elo_tournament.py`          | model Elo             | pipeline "models" mode           |
| `run_eval_tournaments.py`              | state-pool eval       | pipeline "eval" mode             |
| `run_tournament.py` (weights)          | heuristic profiling   | pipeline "profiles" mode         |
| `run_composite_gauntlet.py`            | composite Elo         | pipeline "composite" mode        |

---

## 6) Canonical Recording Checklist

- [ ] `UnifiedGameRecorder` used for every tournament match.
- [ ] Move history includes all bookkeeping moves.
- [ ] `GameReplayDB` metadata contains: source, tournament_id, match_id, ai specs.
- [ ] `validate_canonical_history_for_game` passes for tournament DBs.
- [ ] Quality scores computed and stored.
