# Engine, Tooling, and Parity Research – Weeks 1–3

This document captures the Week 1–3 artefacts for the
engine/tooling/parity workstream. It is structured so later weeks can
build on it without re‑deriving context from the codebase.

The focus areas are:

- Parity invariant surfaces (TS ↔ Python) and replay harnesses.
- GameReplayDB schema and recording surfaces.
- AI failure modes and invariants.
- Onboarding / HUD / teaching copy relevant to rules surfaces.

Week 1 focused on mapping surfaces and gaps. Weeks 2–3 implemented:

- Richer TS↔Python replay parity tagging and aggregation.
- Replay DB health metrics and JSON summaries.
- Schema v5 with full‑fidelity recording metadata.
- A golden‑game differential replay test hook for CI/regressions.

---

## A. TS↔Python Parity – Current Signals and Surfaces

This section catalogues the existing parity dimensions and where they are
computed and compared.

### A.1 TS replay harness (`scripts/selfplay-db-ts-replay.ts`)

**Purpose**

- Replays a recorded self‑play game from a SQLite DB into the
  `ClientSandboxEngine` and logs per‑move TS state summaries in JSON
  lines for downstream tooling (Python parity harness, ad‑hoc analysis).

**Key outputs**

For each game, the harness emits:

- `ts-replay-initial`
  - Fields:
    - `dbPath`: absolute DB path.
    - `gameId`: game UUID.
    - `totalRecordedMoves`: length of `detail.moves`.
    - `summary`:
      - `label`: `"initial"`.
      - `currentPlayer`
      - `currentPhase`
      - `gameStatus`
      - `moveHistoryLength`
      - `stateHash` – from `hashGameState(state)`.

- `ts-replay-step` (for each applied move `k = 1..N`)
  - Fields:
    - `k`: 1‑based index of applied move.
    - `moveType`: canonical `Move.type`.
    - `movePlayer`: `Move.player`.
    - `moveNumber`: `Move.moveNumber` after normalization.
    - `summary`:
      - `label`: e.g. `"after_move_7"`.
      - `currentPlayer`
      - `currentPhase`
      - `gameStatus`
      - `moveHistoryLength`
      - `stateHash` – `hashGameState` again.

- `ts-replay-final`
  - Fields:
    - `appliedMoves`: number of moves actually applied.
    - `summary`:
      - `currentPlayer`
      - `currentPhase`
      - `gameStatus`
      - `moveHistoryLength`
      - `stateHash`.

**Parity‑relevant invariants surfaced by TS**

- At each step:
  - `currentPlayer`
  - `currentPhase`
  - `gameStatus`
  - `stateHash` (TS flavour – human‑readable but deterministic).
- Metadata:
  - Move types and players.
  - Applied move count vs `totalRecordedMoves`.

### A.2 Python parity checker (`ai-service/scripts/check_ts_python_replay_parity.py`)

**Purpose**

- For each game in each discovered GameReplayDB:
  - Classifies the game’s recording structure (`good`, `mid_snapshot`,
    `invalid`, `error`).
  - Replays the game in Python (`GameReplayDB.get_state_at_move`) and
    in TS (via `selfplay-db-ts-replay.ts`).
  - Compares state summaries per move index.

**Current dimensions compared**

For each move index:

- Python:
  - `current_player`
  - `current_phase` (string, via `.value` if enum)
  - `game_status` (string)
  - `state_hash` (SHA‑256 of canonical JSON, truncated)
- TS:
  - `currentPlayer`
  - `currentPhase`
  - `gameStatus`
  - `stateHash` (TS `hashGameState` string)

**Structure classification**

- `classify_game_structure(db, game_id)`:
  - `initial = db.get_initial_state(game_id)`; if missing → `"invalid"`.
  - Computes:
    - `move_hist_len = len(initial.move_history or [])`
    - Board content counts:
      - `stacks` length.
      - `markers` length.
      - `collapsed_spaces` length.
  - Rules:
    - If any of these counts are non‑zero → `"mid_snapshot"` with reason
      `"initial_state contains history/board: ..."`.
    - Else attempts `get_state_at_move(game_id, 0)`; if `None` →
      `"invalid"`.
    - Else → `"good"`.

**Per‑move parity comparison logic**

Index alignment comment (simplified):

- TS:
  - `k = 0  →` initial state BEFORE moves.
  - `k = 1  →` state AFTER move 0.
  - `k = N  →` state AFTER move N‒1.
- Python:
  - `get_state_at_move(0)` = state AFTER move 0.
  - `get_state_at_move(N)` = state AFTER move N.

Comparison steps:

1. **Initial state (TS k=0 vs Python initial_state)**
   - Compares:
     - `current_player`
     - `current_phase`
     - `game_status`
   - Does **not** compare `state_hash` here because Python uses SHA‑256
     and TS uses a custom string format.
   - On first mismatch → `diverged_at = 0`.

2. **Post‑move states (TS k ≥ 1 vs Python get_state_at_move(k−1))**
   - For `ts_k in [1..total_moves_ts]` while `py_move_index = ts_k - 1 < total_moves_py`:
     - Fetch Python summary at `py_move_index`.
     - Fetch TS summary at `ts_k`.
     - Compare:
       - `current_player`
       - `current_phase`
       - `game_status`.
     - First mismatch sets `diverged_at = ts_k`.
   - Also track `total_moves_python` vs `total_moves_ts`.

In Week 2 this logic was extended to:

- Classify _what_ diverged at the first mismatch via `mismatch_kinds`:
  - e.g. `["current_player"]`, `["current_phase", "game_status"]`,
    `["move_count"]`, `["ts_missing_step"]`.
- Attach a high‑level `mismatch_context`:
  - `"initial_state"` for mismatches at TS `k=0` vs Python initial.
  - `"post_move"` for per‑move divergences.
  - `"global"` when only total move counts differ.
- Aggregate mismatch counts by dimension across all games:
  - `mismatch_counts_by_dimension` in the JSON summary payload.
  - Dimensions currently include:
    - `"current_player"`, `"current_phase"`, `"game_status"`,
      `"move_count"`, `"ts_missing_step"`.

These extensions make it possible to answer:

- “Are divergences dominated by phase vs status vs player vs move‑count?”
- “Are we mostly missing TS steps, or disagreeing on post‑move state?”

**Outputs**

- JSON summary (default mode):
  - Counts:
    - Total DBs.
    - Total games checked.
    - Games with semantic divergences (parity issues).
    - Games with structural issues.
  - Lists:
    - `semantic_divergences`: full `GameParityResult` objects with
      optional `python_summary` and `ts_summary`.
    - `structural_issues`: classification summary per game.

- Compact mode (`--compact`):
  - One line per **semantic** divergence only:
    - `db`, `game`, `diverged_at`, phases/statuses, TS/Python hashes.

### A.3 Running the parity + DB health toolchain

For day-to-day use, the replay/parity tooling is wired up as a small CLI suite:

- **Classify and summarise DB health**
  - Script: `ai-service/scripts/cleanup_useless_replay_dbs.py`
  - Typical usage (from `ai-service/`):
    - Dry-run + JSON summary:
      - `PYTHONPATH=. python scripts/cleanup_useless_replay_dbs.py --summary-json db_health.before_cleanup.json`
    - Delete structurally useless DBs and write a post-cleanup summary:
      - `PYTHONPATH=. python scripts/cleanup_useless_replay_dbs.py --delete --summary-json db_health.after_cleanup.json`
  - Summaries include per-DB structure classifications (`good`, `mid_snapshot`, etc.), board/players/source/termination distributions, and a `marked_useless` flag.

- **Check TS↔Python replay parity**
  - Script: `ai-service/scripts/check_ts_python_replay_parity.py`
  - Typical usage (from `ai-service/`):
    - Full JSON report:
      - `PYTHONPATH=. python scripts/check_ts_python_replay_parity.py > parity_summary.json`
    - Grep-friendly compact log of semantic divergences:
      - `PYTHONPATH=. python scripts/check_ts_python_replay_parity.py --compact > parity_semantic.log`
  - The JSON summary exposes:
    - `semantic_divergences` with `mismatch_kinds` and `mismatch_context` for each game (for example `["current_phase"]`, `["ts_missing_step"]`, `["move_count"]`).
    - `mismatch_counts_by_dimension` aggregating per-dimension counts across all DBs.

- **Merge and mine replay DBs for evaluation pools**
  - Scripts:
    - `ai-service/scripts/merge_game_dbs.py` – merge many small `games.db` files into a consolidated DB.
    - `ai-service/scripts/export_state_pool.py` – sample mid-game states into JSONL eval pools.
    - `ai-service/scripts/find_golden_candidates.py` – scan one or more DBs for structurally healthy, metadata-rich games (zero invariant violations, optional pie-rule usage) as candidate inputs to the golden replay suite.
    - `ai-service/scripts/extract_golden_games.py` – copy specific `(db_path, game_id)` pairs into small, dedicated golden fixture DBs under `ai-service/tests/fixtures/golden_games/`.
  - Typical usage (from `ai-service/`):
    - Merge:
      - `PYTHONPATH=. python scripts/merge_game_dbs.py --output data/games/combined.db --db logs/cmaes/runs/*/games.db`
    - Export a balanced state pool:
      - `PYTHONPATH=. python scripts/export_state_pool.py --db data/games/combined.db --output data/eval_pools/combined_pool.jsonl --sample-moves 20,40,60 --min-game-length 50 --winners-only --source cmaes`
    - List candidate golden games for further inspection:
      - `PYTHONPATH=. python scripts/find_golden_candidates.py --db data/games/combined.db --min-moves 40 --output golden_candidates.json`
    - Promote chosen candidates into golden fixture DBs:
      - `PYTHONPATH=. python scripts/extract_golden_games.py --db data/games/combined.db --game-id <id1> --game-id <id2> --output ai-service/tests/fixtures/golden_games/golden_line_territory.db`
  - These JSONL eval pools are consumed by `app.training.eval_pools.load_state_pool(...)` and referenced by CMA-ES configs via `state_pool_id`.

---

## B. GameReplayDB – Schema and Recording Surfaces

This section summarises the DB schema and how recordings are produced
from Python scripts.

### B.1 Core tables (`ai-service/app/db/game_replay.py`)

**Schema version**

- `SCHEMA_VERSION = 6`
  - v1: initial schema.
  - v2: time control + engine eval fields on moves.
  - v3: `game_history_entries` and state hash fields.
  - v4: full pre/post state snapshots in `game_history_entries`.
  - v5: `metadata_json` on `games` for full recording metadata.
  - v6: `available_moves_json` and `available_moves_count` on
    `game_history_entries` for per-step move enumeration parity/debugging.

**Tables and key columns**

1. `games`
   - Core per‑game metadata:
     - `game_id` (PK)
     - `board_type` (TEXT)
     - `num_players` (INTEGER)
     - `rng_seed` (INTEGER, optional)
     - `created_at`, `completed_at`
     - `game_status` (TEXT)
     - `winner` (INTEGER, optional)
     - `termination_reason` (TEXT, optional)
     - `total_moves`, `total_turns`
     - `duration_ms` (optional)
     - `source` (TEXT, e.g. `'self_play'`, `'selfplay_soak'`,
       `'training_data_generation'`, `'sensitivity_test'`, `'cmaes'`, etc.)
     - `schema_version` (INTEGER)
     - Time control fields (v2):
       - `time_control_type` (TEXT, default `'none'`)
       - `initial_time_ms`, `time_increment_ms`.
     - Recording metadata (v5):
       - `metadata_json` (TEXT) – JSON‑encoded metadata dict supplied at
         recording time, including:
         - Script‑specific fields:
           - `engine_mode`, `difficulty_band`, `termination_reason`,
             `rng_seed`, `weight_key`, `weight_value`, etc.
         - Environment‑driven version tags (when set):
           - `rules_engine_version` from `RINGRIFT_RULES_ENGINE_VERSION`.
           - `ts_engine_version` from `RINGRIFT_TS_ENGINE_VERSION`.
           - `ai_service_version` from `RINGRIFT_AI_SERVICE_VERSION`.

2. `game_players`
   - Per‑player metadata:
     - `game_id` (FK → games)
     - `player_number`
     - `player_type` (e.g. `human` vs `ai`)
     - `ai_type`, `ai_difficulty`, `ai_profile_id`
     - `final_eliminated_rings`
     - `final_territory_spaces`
     - `final_rings_in_hand`.

3. `game_initial_state`
   - Stored canonical initial state:
     - `game_id` (PK)
     - `initial_state_json` (TEXT)
     - `compressed` (INTEGER, 0/1)
   - This JSON is expected to represent **start of recorded sequence**:
     - `move_history` cleared by recorders (see recording helpers).
     - Board should be empty for “true full games”; may be mid‑snapshot
       for certain older/iterative DBs (now classified as `mid_snapshot`
       by tooling and generally excluded from parity).

4. `game_moves`
   - Linear move log:
     - `game_id` (FK → games)
     - `move_number` (0‑based)
     - `turn_number` (0‑based → `total_turns`)
     - `player`, `phase`, `move_type`
     - `move_json` (serialized move)
     - `timestamp`, `think_time_ms`
     - Time/eval metadata:
       - `time_remaining_ms`
       - `engine_eval`, `engine_eval_type`
       - `engine_depth`, `engine_nodes`
       - `engine_pv` (JSON of best line)
       - `engine_time_ms`.

5. `game_state_snapshots`
   - For fast reconstruction:
     - `game_id` (FK)
     - `move_number` (snapshot taken AFTER this move)
     - `state_json` (GameState)
     - `compressed` (0/1)
     - `state_hash` (v3+; always populated when storing snapshot
       now, even in interval‑snapshot mode).

6. `game_choices`
   - Decision‑phase choices:
     - `game_id`, `move_number`
     - `choice_type`
     - `player`
     - `options_json` (full choice set)
     - `selected_option_json`
     - `ai_reasoning` (optional).

7. `game_history_entries`
   - GameTrace‑style structured history:
     - `game_id`, `move_number`
     - `player`
     - `phase_before`, `phase_after`
     - `status_before`, `status_after`
     - `progress_before_json`, `progress_after_json` (S‑like snapshots)
     - `state_hash_before`, `state_hash_after`
     - `board_summary_before_json`, `board_summary_after_json`
     - v4:
       - `state_before_json`, `state_after_json`
       - `compressed_states` (0/1).
     - v6:
       - `available_moves_json` – optional JSON‑encoded list of valid
         moves at `state_before` (for deep parity debugging).
       - `available_moves_count` – integer count of valid moves
         (lightweight summary when full enumeration isn’t captured).

### B.2 Recording helpers (`ai-service/app/db/recording.py`)

**GameRecorder (context manager)**

- Constructor:
  - Accepts `GameReplayDB` and an initial `GameState`.
  - Generates a `game_id` if none provided.

- `__enter__`:
  - Deep‑copies `initial_state`.
  - Clears any pre‑populated `move_history` on the copy.
  - Calls `db.store_game_incremental(game_id, initial_for_recording)`:
    - Ensures `game_initial_state.initial_state_json` always represents
      the start of the recorded sequence, not a mid‑game snapshot.

- `add_move(move)`:
  - Delegates to `GameWriter.add_move` to append to `game_moves` and,
    depending on configuration, snapshots/history entries.

- `finalize(final_state, metadata)`:
  - Delegates to `GameWriter.finalize`:
    - Stores final snapshot.
    - Finalises the `games` row (status, winner, termination_reason,
      totals, duration, metadata).
  - Before finalisation, metadata is enriched with version tags:
    - `rules_engine_version`, `ts_engine_version`,
      `ai_service_version` from the environment (if present and not
      already set in the metadata dict).

**record_completed_game (one‑shot)**

- Takes `initial_state`, `final_state`, full `moves` list, plus optional
  metadata and `game_id`.
- Deep‑copies `initial_state` and clears any `move_history` before
  storing.
- Enriches metadata with environment‑driven version tags as above.
- Calls `db.store_game(...)` to populate:
  - `games`
  - `game_initial_state`
  - `game_moves`
  - `game_players` (if applicable in upstream callers).

### B.3 Script entry points (high‑level)

**Self‑play soak harness (`ai-service/scripts/run_self_play_soak.py`)**

- Uses:
  - `get_or_create_db` from `app.db` (aliased import).
  - `record_completed_game` to persist completed self‑play games.
- Per‑game metadata recorded (and now persisted via `metadata_json`):
  - `source = "selfplay_soak"`.
  - `engine_mode` (`"mixed"` / `"descent-only"`).
  - `difficulty_band` (e.g. `"canonical"`, `"light"`).
  - `termination_reason` (explicit reason string).
  - `rng_seed` (per‑game seed).
  - `invariant_violations_by_type` (per-invariant counters) and pie-rule
    diagnostics (`swap_sides_moves`, `used_pie_rule`) – used when mining
    interesting candidate games for the golden replay suite.
  - Any global version tags from the environment.
- Companion DB‑health tool:
  - `ai-service/scripts/cleanup_useless_replay_dbs.py`:
    - Classifies DBs by structural health (`good`, `mid_snapshot`,
      `invalid`, etc.).
    - With `--summary-json PATH` (added in Week 2) writes:
      - `total_databases`.
      - Per‑DB:
        - `db_path`, `total_games`, `games_inspected`.
        - `structure_counts` (good/mid_snapshot/invalid/internal_inconsistent).
        - `board_type_counts`, `num_players_counts`.
        - `source_counts`, `termination_reason_counts`.
        - `marked_useless` (whether the DB would be / was deleted).

Other scripts (CMA‑ES, tournaments, training) follow a similar pattern
and now get full metadata persistence “for free” via
`record_completed_game` and the enriched recording helpers.

---

---

## C. AI Failure Modes and Invariants – Current Inventory

This section defines a draft taxonomy of AI failure modes and highlights
existing signals in TS and Python.

### C.1 Existing Python invariants (`run_self_play_soak.py`)

Core invariants surfaced via `VIOLATION_TYPE_TO_INVARIANT_ID`:

- `S_INVARIANT_DECREASED` → `INV-S-MONOTONIC`
- `TOTAL_RINGS_ELIMINATED_DECREASED` → `INV-ELIMINATION-MONOTONIC`
- `ACTIVE_NO_MOVES` → `INV-ACTIVE-NO-MOVES`
- `ACTIVE_NO_CANDIDATE_MOVES` → `INV-ACTIVE-NO-MOVES`

Violations are:

- Counted per‑game (`invariant_violations_by_type`).
- Optionally sampled into a bounded diagnostics list with:
  - `board_type`, `game_status`, `current_player`, `current_phase`.
  - Optional “before”/“after” progress snapshots.
- Exported via Prometheus metric `PYTHON_INVARIANT_VIOLATIONS` with
  `invariant_id` and `type` labels.

### C.2 Sandbox AI diagnostics (`src/client/sandbox/sandboxAI.ts`)

Key features:

- Hash‑based stall detection:
  - For each AI turn:
    - Compute `beforeHashForHistory = hashGameState(beforeState)`.
    - After the AI turn, compute `afterHashForHistory`.
  - `stateUnchanged = beforeHash === afterHash`.
  - `samePlayer` = current player unchanged.
  - `stillActive` = `gameStatus === 'active'`.
  - A module‑level counter `sandboxConsecutiveNoopAITurns` increments
    when `stateUnchanged && samePlayer && stillActive`, resets otherwise.
  - After `SANDBOX_NOOP_STALL_THRESHOLD` (8) consecutive no‑ops:
    - Logs a stall diagnostic entry.
  - After the same window (or reaching `SANDBOX_NOOP_MAX_THRESHOLD`):
    - Marks game `gameStatus: 'completed'` and normalises phase/capture
      cursor.

- Trace buffer (`__RINGRIFT_SANDBOX_TRACE__`):
  - Per‑turn entries include:
    - `boardType`
    - `playerNumber`
    - `currentPhaseBefore` / `currentPhaseAfter`
    - `gameStatusBefore` / `gameStatusAfter`
    - `beforeHash` / `afterHash`
    - `lastAIMoveType` / `lastAIMovePlayer`
    - Optionally:
      - `captureCount`
      - `simpleMoveCount`
      - `placementCandidateCount`
      - `forcedEliminationAttempted` / `forcedEliminationEliminated`
      - `consecutiveNoopAITurns`.

- Parity mode:
  - When enabled, movement decisions delegate directly to the shared
    `chooseLocalMoveFromCandidates` for closer backend alignment.

### C.3 Draft AI failure‑mode taxonomy

**Structural failures**

- **Stalls / non‑termination**
  - Repeated AI turns with unchanged state hash and same player while
    game remains `active`.
  - Already tracked in TS via `sandboxConsecutiveNoopAITurns`.
  - In Python, partially reflected via `ACTIVE_NO_MOVES` and `ACTIVE_NO_CANDIDATE_MOVES`.

- **ACTIVE state with no legal moves**
  - Invariants:
    - `STRICT_NO_MOVE_INVARIANT` (backend).
    - `INV-ACTIVE-NO-MOVES` / `INV-ACTIVE-NO-CANDIDATE-MOVES`.

**Semantic failures**

- TS↔Python divergence:
  - Detected by parity tools (Track A).
  - Often correlated with invariant violations (e.g. S‑invariant drifts).

**Strategic weaknesses (future)**

- Losing material without compensation.
- Missing obvious captures or lines.
- Easily‑detectable blunders in self‑play.

At present, we primarily have strong structural and some semantic
coverage; strategic weaknesses would likely rely on evaluation
differences or heuristics over S/eval traces.

---

## N. Onboarding & Teaching UX – Current Copy and Surfaces

This section summarises the current onboarding, HUD, and teaching copy
as a basis for later “minimal onboarding” design.

### N.1 GameHUD phase descriptions (`src/client/components/GameHUD.tsx`)

`getPhaseInfo(phase: GamePhase)` provides:

- `ring_placement`
  - Label: **Ring Placement**
  - Description:
    - “Place a ring on an empty space or on top of one of your own stacks.”

- `movement`
  - Label: **Movement Phase**
  - Description:
    - “Select one of your stacks and move it to a legal destination.”

- `capture`
  - Label: **Capture Phase**
  - Description:
    - “Overtake an adjacent stack by jumping over it to a legal landing space.”

- `chain_capture`
  - Label: **Chain Capture**
  - Description:
    - “Continue the capture by choosing the next target and landing space.”

- `line_processing`
  - Label: **Line Processing**
  - Description:
    - “Choose which completed line to resolve and what reward to take.”

- `territory_processing`
  - Label: **Territory Processing**
  - Description:
    - “Choose a disconnected region to resolve; some choices may cost you rings.”

These are short, phase‑oriented descriptions used in legacy HUD paths;
the newer HUD view models add action/spectator hints around them.

### N.2 Teaching overlay content (`src/client/components/TeachingOverlay.tsx`)

Structured `TEACHING_CONTENT` for topics:

- `ring_placement`
  - Title: Ring Placement.
  - Description:
    - Emphasises placing rings from hand onto empty spaces; stacking on
      existing rings.
  - Tips:
    - “Placing adjacent to your existing rings helps build territory.”
    - “Stacking on opponents can set up future captures.”
    - “The placement phase continues until all players have placed all rings.”
  - Related phases: `['ring_placement']`.

- `stack_movement`
  - Title: Stack Movement.
  - Description:
    - Move a stack you control exactly as many spaces as its height,
      straight lines (H/V/diagonal).
  - Tips:
    - “Taller stacks move further but are harder to control.”
    - “Moving onto an opponent creates a capture opportunity.”
    - “You cannot pass through other stacks.”
  - Related phases: `['movement']`.

- `capturing`
  - Title: Capturing.
  - Description:
    - Landing on an opponent’s stack eliminates one bottom ring; your
      ring stays on top.
  - Tips:
    - Emphasise permanent elimination and >50% win condition.

- `chain_capture`
  - Title: Chain Capture.
  - Description:
    - Continue capturing as long as new valid captures exist.
  - Tips:
    - Focus on planning, multiple eliminations per turn, and when chains
      end.

- `line_bonus`
  - Title: Line Bonus.
  - Description:
    - Form lines of 3+ and choose rewards (retrieve ring vs claim territory).
  - Tips:
    - All orientations count.
    - Permanent territory vs recycled rings.

- `territory`
  - Title: Territory Control.
  - Description:
    - Permanent ownership of spaces; >50% wins.
  - Tips:
    - Territory cannot be removed.
    - Emphasis on building territory over time.

- `victory_elimination`, `victory_territory`, `victory_stalemate`
  - Three victory modes, each with:
    - Title + icon + description.
    - Tips emphasising thresholds and scenarios.

This overlay is a strong, concept‑by‑concept teaching surface, though
some details (e.g. “placement phase continues until all rings placed”)
reflect earlier rules and may need to be reconciled with actual
engine/phase behaviour.

### N.3 Onboarding modal (`src/client/components/OnboardingModal.tsx`)

Multi‑step onboarding flow with four steps:

1. **WelcomeStep**
   - Visual: 🎮
   - Summary:
     - “Welcome to RingRift!”
     - “A strategic board game where you place rings, build stacks, and compete for territory.”

2. **PhasesStep**
   - Presents three “simple phases”:
     - Ring Placement:
       - “Place rings on the board to control territory.”
     - Movement:
       - “Move your stacks based on their height.”
     - Capture:
       - “Land on opponents to capture their rings.”
   - (Does not yet mention `chain_capture`, line/territory processing; they’re left for later learning.)

3. **VictoryStep**
   - Outlines three ways to win:
     - Ring Elimination:
       - “Capture more than half of any opponent’s rings.”
     - Territory Control:
       - “Control more than half the board spaces.”
     - Last Standing:
       - “Be the only player who can still move.”

4. **ReadyStep**
   - Encourages starting a “Learn the Basics” game.
   - Tips:
     - Use `?` for keyboard shortcuts and controls.

This onboarding is conceptually aligned with the rules but intentionally
high level, and it omits some details (markers, chain capture depth,
territory processing) that are handled later via overlays or HUD.

---

## Summary of Week 1–3 Outputs

The sections above now provide:

- A mapping of **parity surfaces** between TS and Python (what is
  compared, how state indices align, and what is logged), plus
  dimension‑tagged mismatch statistics and a compact debug format.
- A summary of **GameReplayDB schema and recording pathways** including:
  - Schema v5 with `metadata_json` capturing full recording metadata
    (sources, seeds, engine modes, experiment tags, version IDs).
  - Enriched recording helpers that automatically attach version tags.
  - Replay DB health‑check tooling with JSON summaries for aggregate
    analysis.
- An initial **taxonomy of AI failure modes**, with current invariant
  and diagnostic signals on both TS and Python sides.
- An **audit of onboarding and teaching copy**, tied to specific
  components, to inform later work on “minimal viable” onboarding and
  phase explanations.

Week 4+ can now:

- Use the parity checker and DB‑health summaries as a standard harness
  for:
  - Victory mix/balance analysis across board types and AI profiles.
  - Replay robustness and failure‑mode clustering.
  - Regression detection via golden‑game differential replay (see
    `tests/parity/test_differential_replay.py` and the
    `RINGRIFT_PARITY_GOLDEN_*` environment hooks).

## D. Golden Replay Suite (Planned)

To complement broad parity soaks and DB‑health checks, later waves will define a small, **curated “golden games” suite** used for strict, end‑to‑end replay guarantees across both TS and Python engines.

### D.1 Goals and scope

- Exercise the **hard corners of the rules** in a controlled way:
  - Line formation and collapse, including multiple concurrent lines and both `process_line` / `choose_line_option` decisions (legacy alias: `choose_line_reward`).
  - Territory processing: disconnected regions, `choose_territory_option` (legacy alias: `process_territory_region`), `eliminate_rings_from_stack`, automatic elimination from hand, and structural stalemates.
  - Capture and chain capture: long chains (3+ segments), forced continuation vs legal termination, and correct phase transitions.
  - Pie rule: `swap_sides` offered and taken/declined in 2‑player games.
  - LPS and multi‑player: 3p/4p games ending via `last_player_standing` (R172) rather than pure ring/territory thresholds.
  - Structural invariants: INV‑ACTIVE‑NO‑MOVES, elimination/territory monotonicity, and S‑invariant behaviour during complex turns.
- Provide **frozen, versioned traces** that CI can treat as “must not regress”:
  - Both engines must agree on every move (phases, statuses, basic summaries).
  - All invariants must hold for the entire trace.

### D.2 Representation and storage

- **Python / DB form (authoritative recordings)**
  - Location: `ai-service/tests/fixtures/golden_games/`.
  - Content: small `GameReplayDB` files (one or a few games each), for example:
    - `golden_line_territory.db`
    - `golden_chain_capture.db`
    - `golden_pie_rule.db`
    - `golden_lps_3p.db`
  - Each DB contains structurally “good” recordings only (per `cleanup_useless_replay_dbs.py` classification).
- **TS / GameRecord form (shared-engine replay)**
  - Location: `tests/fixtures/golden-games/`.
  - Content: JSONL files with one `GameRecord` per line, exported via the existing GameRecord exporters on either side:
    - Python: `generate_data.py --game-records-jsonl` or a focused helper for `GameReplayDB` → `GameRecord`.
    - TS/Node: `scripts/export-game-records-jsonl.ts` from Postgres.
  - These GameRecords are consumed by the shared engine’s `reconstructStateAtMove(record, moveIndex)` helper for TS‑only replay checks.

### D.3 Selection and promotion process

- **Scripted scenarios first** (line/territory, capture chains):
  - Extend existing scenario/parity tests (for example `ai-service/tests/parity/test_line_and_territory_scenario_parity.py`) to:
    - Drive the engines through carefully chosen move sequences that hit specific rule axes.
    - Record those games to `GameReplayDB` via `record_completed_game(...)` into the golden fixtures directory.
  - Only promote a scenario to “golden” once:
    - DB health is `good` (no mid‑snapshot or schema anomalies).
    - TS↔Python replay parity is clean for all moves (via `check_ts_python_replay_parity.py`).
    - Invariants remain satisfied over the full trace.
- **Mined rare events second** (LPS, complex territory, swap usage):
  - Use self‑play soaks (`run_self_play_soak.py`) and training runs to generate large corpora with rich metadata (`source`, `termination_reason`, `used_pie_rule`, etc.).
  - Filter to structurally sound, parity‑clean games and promote a small number of representative traces into the golden suite by copying or extracting them into dedicated golden DBs.

### D.4 Test integration (TS + Python)

- **Python golden replay parity test (planned)**
  - New test module (for example `ai-service/tests/parity/test_golden_replay_parity.py`) will:
    - Iterate over golden DBs and known `game_id`s.
    - Reconstruct Python states via `GameReplayDB.get_state_at_move(...)`.
    - Compare against TS replay summaries produced by `scripts/selfplay-db-ts-replay.ts` for the same games.
    - Assert equality of `current_player`, `current_phase`, `game_status`, and selected progress snapshots at every move; any mismatch is a test failure.
- **TS shared-engine golden replay test (planned)**
  - New Jest suite (for example `tests/unit/GoldenGames.replay.shared.test.ts`) will:
    - Load GameRecords from `tests/fixtures/golden-games/*.jsonl`.
    - Use `reconstructStateAtMove(record, k)` to step through each trace.
    - Assert expected high-level behaviour (phase transitions, status changes, final outcome/score coherence) for each golden axis.

This “golden replay suite” plan is tracked as part of **Wave 10.4 – Replay System** in `IMPROVEMENT_PLAN.md` and will evolve alongside the existing differential replay and invariant soaks as the primary, high‑fidelity guardrail over rules behaviour across engines.
