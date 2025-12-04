# Strict-invariant self-play soaks (Python GameEngine)

**Scope:** Operational guidance for running Python self-play soaks with the strict no-move invariant enabled, under memory-safe configurations, and mining resulting failures into regression tests.

This doc complements:

- [`AI_TRAINING_AND_DATASETS.md`](AI_TRAINING_AND_DATASETS.md) – general self-play and dataset generation.
- [`RULES_ENGINE_ARCHITECTURE.md`](RULES_ENGINE_ARCHITECTURE.md) / [`RULES_IMPLEMENTATION_MAPPING.md`](../RULES_IMPLEMENTATION_MAPPING.md) – TS↔Python rules alignment.

---

## 1. Strict no-move invariant (Python GameEngine)

The Python rules engine now enforces a **TS-aligned active-player invariant** behind a feature flag:

```bash
RINGRIFT_STRICT_NO_MOVE_INVARIANT=1
```

When enabled, every state produced by `GameEngine.apply_move` must satisfy:

> If `game_status == ACTIVE` for `current_player`, then that player has at least one legal **action**, defined as:
>
> - any interactive move returned by `GameEngine.get_valid_moves(...)` for the current phase (placements, movements, captures, line/territory decisions), **or**
> - at least one forced-elimination move from `_get_forced_elimination_moves(...)` when the player is blocked but still controls stacks.

Implementation (Python): `GameEngine._assert_active_player_has_legal_action` in [`ai-service/app/game_engine.py`](../ai-service/app/game_engine.py).

Violations are logged under:

- `ai-service/logs/invariant_failures/active_no_moves_p{player}_{timestamp}.json`

These snapshots are the canonical source for constructing invariant regression tests in `ai-service/tests/invariants/`.

In addition, the **self-play harness** logs env-level failures (even if the engine invariant is disabled):

- `ai-service/logs/selfplay/failures/failure_*.json`

These encode situations where `env.legal_moves()` returns an empty list while `GameStatus.ACTIVE` is still set, or where `env.step` raises.

---

## 2. Soak harness: memory-safe options

The long self-play soak harness lives in:

- [`ai-service/scripts/run_self_play_soak.py`](../ai-service/scripts/run_self_play_soak.py)

It runs N-player self-play over `RingRiftEnv` + Python `GameEngine`, and writes per-game summaries to a JSONL log.

### 2.1 Key CLI options

From `ai-service/`:

```bash
python scripts/run_self_play_soak.py \
  --num-games 100 \
  --board-type square8|square19|hexagonal \
  --engine-mode descent-only|mixed \
  --difficulty-band canonical|light \
  --num-players 2..4 \
  --max-moves 200 \
  --seed 42 \
  --log-jsonl logs/selfplay/soak.square8_2p.mixed.jsonl \
  --summary-json logs/selfplay/soak.square8_2p.mixed.summary.json \
  [--gc-interval 50] [--fail-on-anomaly]
```

When invoked as a CLI, `run_self_play_soak.py` prints a JSON configuration
summary (including whether `STRICT_NO_MOVE_INVARIANT` is enabled) at startup;
capturing this banner in logs is recommended when analysing failures from
long-running soaks or comparing runs across different machines. When
`--fail-on-anomaly` is passed, the script will exit with a non-zero status
code if any game terminates with an invariant/engine anomaly such as
`no_legal_moves_for_current_player` or `step_exception:...`, which is useful
for automated gates or scheduled jobs.

Important knobs for memory/CPU safety:

- `--engine-mode`:
  - `descent-only` – all players use `DescentAI` (heavy tree search + NN; highest memory/CPU).
  - `mixed` – per-player sampling across the canonical difficulty ladder.
- `--difficulty-band` (for `engine-mode=mixed`):
  - `canonical` (default) – difficulties `[1,2,4,5,6,7,8,9,10]` (Random, Heuristic, Minimax, MCTS, Descent).
  - `light` – **memory-conscious band** `[1,2,4,5]` (Random, Heuristic, low-depth Minimax).
- `--gc-interval` (int, default `0`):
  - If `> 0`, clears the `GameEngine` move cache and runs `gc.collect()` every _N_ games.
  - This bounds long-run memory by periodically tearing down cached move lists and unreachable search structures.
- `--max-moves`:
  - Caps the length of each game. Lower values reduce worst-case search depth and log size.

### 2.2 Recommended memory-safe strict-invariant profiles

Use the environment flag to enable the invariant:

```bash
export RINGRIFT_STRICT_NO_MOVE_INVARIANT=1
```

Then invoke the soak harness with a **light** band and periodic GC. The examples below are tuned for local debugging on moderate hardware; adjust `--num-games` upward for deeper soaks.

#### 2.2.1 2‑player, square8, mixed AI (strict, light band)

```bash
cd ai-service
RINGRIFT_STRICT_NO_MOVE_INVARIANT=1 \
python scripts/run_self_play_soak.py \
  --num-games 20 \
  --board-type square8 \
  --engine-mode mixed \
  --difficulty-band light \
  --num-players 2 \
  --max-moves 150 \
  --seed 42 \
  --gc-interval 20 \
  --log-jsonl logs/selfplay/soak.square8_2p.mixed.strict_light.jsonl \
  --summary-json logs/selfplay/soak.square8_2p.mixed.strict_light.summary.json
```

#### 2.2.2 3‑player and 4‑player, square8

```bash
# 3-player square8
RINGRIFT_STRICT_NO_MOVE_INVARIANT=1 \
python scripts/run_self_play_soak.py \
  --num-games 10 \
  --board-type square8 \
  --engine-mode mixed \
  --difficulty-band light \
  --num-players 3 \
  --max-moves 180 \
  --seed 101 \
  --gc-interval 10 \
  --log-jsonl logs/selfplay/soak.square8_3p.mixed.strict_light.jsonl

# 4-player square8
RINGRIFT_STRICT_NO_MOVE_INVARIANT=1 \
python scripts/run_self_play_soak.py \
  --num-games 10 \
  --board-type square8 \
  --engine-mode mixed \
  --difficulty-band light \
  --num-players 4 \
  --max-moves 200 \
  --seed 202 \
  --gc-interval 10 \
  --log-jsonl logs/selfplay/soak.square8_4p.mixed.strict_light.jsonl
```

#### 2.2.3 3‑player, hexagonal board

```bash
RINGRIFT_STRICT_NO_MOVE_INVARIANT=1 \
python scripts/run_self_play_soak.py \
  --num-games 10 \
  --board-type hexagonal \
  --engine-mode mixed \
  --difficulty-band light \
  --num-players 3 \
  --max-moves 200 \
  --seed 303 \
  --gc-interval 10 \
  --log-jsonl logs/selfplay/soak.hex_3p.mixed.strict_light.jsonl
```

For **deeper offline soaks**, prefer running several independent small batches (e.g. 20–50 games each) rather than a single monolithic run, and aggregate their JSONL logs via `cat` or simple scripts.

### 2.3 Orchestrator Soak Harness (`soak:orchestrator`)

In addition to the Python self-play harness, the TS shared-engine orchestrator has a lightweight soak harness that drives many random games through a real backend host (`GameEngine` with the orchestrator adapter enabled) while checking core invariants and emitting a JSON summary suitable for local inspection or future CI/monitoring.

> **Note:** As of this QA task the soak harness exercises the orchestrator via the backend `GameEngine` only. A symmetric sandbox-based entrypoint (`ClientSandboxEngine`) can be reintroduced in a later task once the client build surface under `ts-node` is fully clean.

- Commands:
  - `npm run soak:orchestrator` – generic entrypoint, parameters controlled via `--boardTypes`, `--gamesPerBoard`, etc.
  - `npm run soak:orchestrator:smoke` – **CI smoke profile** (single short backend game on `square8`, fails on invariant violation). Useful as a fast invariant check on PRs or before running heavier soaks.
  - `npm run soak:orchestrator:short` – **CI short-soak profile** (multiple short backend games on `square8` with `--failOnViolation=true`). This is the concrete implementation of the `SLO-CI-ORCH-SHORT-SOAK` gate referenced in `ORCHESTRATOR_ROLLOUT_PLAN.md` and wired into the `orchestrator-short-soak` job in `.github/workflows/ci.yml`.
  - `npm run soak:orchestrator:nightly` – **longer multi-board profile** used for scheduled or on-demand deeper soaks (20 games per board type across `square8`, `square19`, and `hexagonal`, with `--failOnViolation` enabled by default).
  - Examples (variants of the generic entrypoint):
    - `npm run soak:orchestrator -- --boardTypes=square8,square19 --gamesPerBoard=25`
    - `npm run soak:orchestrator -- --boardTypes=hexagonal --gamesPerBoard=10 --maxTurns=750 --randomSeed=1234 --outputPath=results/orchestrator_soak_smoke.json`
    - `npm run soak:orchestrator -- --failOnViolation` (non-zero exit if any invariant violation is detected)
    - `npm run soak:orchestrator -- --boardTypes=square8 --gamesPerBoard=5 --debug` (short run with strict S-invariant trace logs enabled via `RINGRIFT_TRACE_DEBUG=1`)

- Behaviour:
  - Runs multiple short self-play games per board type against the TS orchestrator via the backend `GameEngine`, selecting random legal moves from `getValidMoves`.
  - Enforces an **ACTIVE-no-move invariant**: whenever `gameStatus == 'active'`, `getValidMoves` for the current player must return at least one move, and there must never be an `ACTIVE` state with zero legal actions.
  - Verifies **orchestrator move validation**: every move returned by `getValidMoves` is re-validated via the shared orchestrator’s `validateMove`; any disagreement between host and orchestrator is recorded.
  - Checks **S-invariant** and elimination monotonicity on each turn (`S = markers + collapsedSpaces + eliminatedRings`, `totalRingsEliminated` non-decreasing), plus basic board sanity (no negative stack heights, `stackHeight === rings.length`, `0 ≤ capHeight ≤ stackHeight`, and non-negative `eliminatedRings` per player).
    - When invoked with `--debug` (or `--traceDebug`), the harness also forces `RINGRIFT_TRACE_DEBUG=1` for the run so that `GameEngine.appendHistoryEntry` and related hooks emit detailed `STRICT_S_INVARIANT_DECREASE` and elimination-bookkeeping trace logs without requiring manual env wiring.
  - Treats hitting `--maxTurns` without termination as a soft anomaly (recorded in the summary as a max-turns timeout).

- Output:
  - Writes a machine-readable summary to `results/orchestrator_soak_summary.json` (or the configured `--outputPath`) with per-board/per-host stats:
    - games run,
    - completed vs `maxTurns` timeouts,
    - game-length distribution (min / median / p95 / max),
    - total invariant violations.
  - Includes a bounded set of violation traces (up to 50 per run), each containing:
    - `id`, `message`,
    - `boardType`, `hostMode`, `gameId`, `gameIndex`, `seed`, `turnIndex`,
    - `gameStatus`, `currentPlayer`, `currentPhase`,
    - S components (`markers`, `collapsed`, `eliminatedRings`, `totalRingsEliminated`, `sInvariant`),
    - `movesTail`: the last few moves leading into the violation.

- Expected outcome on a healthy build:
  - 0 invariant violations (including **no** `S_INVARIANT_DECREASED` events).
  - Most games terminate naturally well before `--maxTurns`, with only a small tail of long-running games.

Debugging: if violations occur, use the summary fields `boardType`, `hostMode`, `gameIndex`, `seed`, `turnIndex`, and `movesTail` to reproduce or triage the failing game via:

- Orchestrator scenario tests and backend vs sandbox orchestrator parity suites.
- Python contract tests and parity fixtures (for cross-language confirmation).
- **Targeted Jest regression harness**:
  - `tests/unit/OrchestratorSInvariant.regression.test.ts` contains one or
    more seeded backend‑orchestrator games (see the `REGRESSION_SEEDS`
    constant in that file) that were promoted directly from
    `S_INVARIANT_DECREASED` entries in `results/orchestrator_soak_smoke.json`.
    Each test replays the exact move sequence via `GameEngine` +
    `TurnEngineAdapter` and asserts that `computeProgressSnapshot(state).S`
    never decreases.
  - All regression cases are initially marked `it.skip` so CI remains green
    while the underlying bug is being investigated. When working locally,
    you can temporarily un‑skip an individual seed (or use `it.only`) and run
    the dedicated script:

    ```bash
    npm run test:orchestrator:s-invariant
    ```

    which wraps:

    ```bash
    RINGRIFT_TRACE_DEBUG=1 \
    jest --runInBand tests/unit/OrchestratorSInvariant.regression.test.ts
    ```

    to emit detailed `STRICT_S_INVARIANT_DECREASE` logs from
    `GameEngine.appendHistoryEntry` for the failing turn. Once the
    S‑invariant bug is fixed, remove `.skip` on these tests to turn them into
    permanent guardrails.

### 2.4 How orchestrator soaks relate to rollout SLOs

The TS orchestrator soak harness is a **gating tool**, not just a diagnostic:

- **Pre-release / pre-deploy gate (short soak):**
  - A short run such as:
    - `npm run soak:orchestrator -- --boardTypes=square8 --gamesPerBoard=5 --failOnViolation=true`
  - acts as a concrete implementation of the
    `SLO-CI-ORCH-SHORT-SOAK` SLO defined in
    [`ORCHESTRATOR_ROLLOUT_PLAN.md`](ORCHESTRATOR_ROLLOUT_PLAN.md:1):
    - Exit code must be `0`.
    - `results/orchestrator_soak_summary.json` must report
      `totalInvariantViolations == 0`.
  - If this short soak fails, the corresponding build **must not** be promoted
    to staging or production until the invariant violations are triaged and
    fixed or explicitly waived.

- **Staging / production guardrail (scheduled or on-demand):**
  - Optional longer or nightly soaks (larger `--gamesPerBoard`, additional
    board types) against staging or production images provide early warning
    for regressions tied to specific seeds or board configurations.
  - These runs back the `SLO-STAGE-ORCH-INVARIANTS` and
    `SLO-PROD-ORCH-INVARIANTS` SLOs in
    [`ORCHESTRATOR_ROLLOUT_PLAN.md`](ORCHESTRATOR_ROLLOUT_PLAN.md:1):
    - Any **new** invariant violation pattern discovered in these soaks should
      be turned into a regression test (TS or Python) and investigated before
      further rollout.
    - Multiple soak failures in a short period are a signal to pause rollout
      or roll back to a safer phase (see the environment phases in
      [`ORCHESTRATOR_ROLLOUT_PLAN.md`](ORCHESTRATOR_ROLLOUT_PLAN.md:1)).

You do **not** need to re-state the full orchestrator rollout design here; treat
[`ORCHESTRATOR_ROLLOUT_PLAN.md`](ORCHESTRATOR_ROLLOUT_PLAN.md:1) as the SLO and
gating SSoT, and this document as the operational recipe for running strict
invariant and soak jobs.

### 2.5 Named Python soak profiles (`run_self_play_soak.py --profile=…`)

The Python self-play harness under `ai-service/scripts/run_self_play_soak.py` exposes two
named profiles that bundle strict-invariant and AI-health checks into small,
repeatable runs:

- `--profile python-strict` – **strict invariant mini‑soak**:
  - Configuration (see `_parse_args` and `main` in `run_self_play_soak.py`):
    - `num_games=6`, `board_type=square8`, `engine_mode=mixed`,
      `difficulty_band=light`, `num_players=2`, `max_moves=150`,
      `gc_interval=10`, deterministic `seed` default of `1764142864`.
  - Intended use:
    - Local or CI‑adjacent **Python strict-invariant preflight**, mirroring
      the spirit of the TS short orchestrator soak on `square8`.
    - When combined with `RINGRIFT_STRICT_NO_MOVE_INVARIANT=1` it treats
      any `no_legal_moves_for_current_player` or `step_exception:*` termination
      as an anomaly and can exit non‑zero via `--fail-on-anomaly`.
  - Example:
    ```bash
    cd ai-service
    RINGRIFT_STRICT_NO_MOVE_INVARIANT=1 \
    python scripts/run_self_play_soak.py \
      --log-jsonl logs/selfplay/soak.python_strict.jsonl \
      --summary-json logs/selfplay/soak.python_strict.summary.json \
      --profile python-strict \
      --fail-on-anomaly
    ```

- `--profile ai-healthcheck` – **AI self‑play healthcheck**:
  - Configuration (see `run_ai_healthcheck_profile`):
    - Runs a small mixed-engine job across `square8`, `square19`, and `hexagonal`
      with a light difficulty band (`mixed_(light)_2p`), bounded `num_games` per board
      (default `RINGRIFT_AI_HEALTHCHECK_GAMES` or `2`), and an aggregate JSON summary.
  - Output:
    - Per‑board JSONL logs (one file per board type) plus a summary JSON with:
      - `invariant_violations_by_id` keyed by `INV-*`,
      - `invariant_violations_total`,
      - a `parity_mismatches` placeholder block reserved for future TS↔Python checks.
  - Intended use:
    - Scheduled or on‑demand **AI healthcheck** that exercises the Python AI + rules
      stack under mixed engines and surfaces invariant regressions before they affect
      orchestrator/production rollout (see the Python invariant SLO notes in
      `INVARIANTS_AND_PARITY_FRAMEWORK.md`).
  - Example:
    ```bash
    cd ai-service
    RINGRIFT_STRICT_NO_MOVE_INVARIANT=1 \
    python scripts/run_self_play_soak.py \
      --log-jsonl logs/selfplay/healthcheck.log.jsonl \
      --summary-json logs/selfplay/healthcheck.summary.json \
      --profile ai-healthcheck
    ```

These profiles do **not** define rules semantics; they are diagnostics/health tools
that complement the TS orchestrator soaks and SLOs described above. Treat any
new invariant pattern discovered by these Python soaks as input to:

- promote representative snapshots into `ai-service/tests/invariants/**`, and
- correlate with TS orchestrator soaks and parity jobs before altering rollout.

---

## 3. Self-play stability tests under strict invariant

The CI-friendly stability test lives in:

- [`ai-service/tests/test_self_play_stability.py`](../ai-service/tests/test_self_play_stability.py)

Key properties:

- Uses `RingRiftEnv` + mixed AI (2‑player, square8) with the strict invariant force-enabled via `RINGRIFT_STRICT_NO_MOVE_INVARIANT`.
- Default game count:
  - `RINGRIFT_SELFPLAY_STABILITY_GAMES` (env var) – defaults to `3` if unset.
- AI difficulty band:
  - `RINGRIFT_SELFPLAY_STABILITY_DIFFICULTY_BAND` – defaults to `light`.
    - `light`: difficulties `[1,2,4,5]` (Random, Heuristic, low-depth Minimax).
    - `canonical`: full ladder `[1,2,4,5,6,7,8,9,10]` including MCTS/Descent.

Example invocations:

```bash
# CI-style, light band, few games (default)
cd ai-service
pytest tests/test_self_play_stability.py::test_self_play_mixed_2p_square8_stability -q

# Local deeper soak: more games, full difficulty ladder
RINGRIFT_SELFPLAY_STABILITY_GAMES=20 \
RINGRIFT_SELFPLAY_STABILITY_DIFFICULTY_BAND=canonical \
pytest tests/test_self_play_stability.py::test_self_play_mixed_2p_square8_stability -q
```

The test asserts that:

- No exceptions escape from `env.step` or AI selection.
- If `env.legal_moves()` returns empty, the state is **not** left in `GameStatus.ACTIVE`.
- Long games cannot exceed `env.max_moves` without terminating.

---

## 4. Failure triage and regression mining

### 4.1 Log locations

After any strict-invariant soak or stability run, inspect:

- **Engine invariant violations** (Python `GameEngine`):
  - `ai-service/logs/invariant_failures/*.json`
  - Each file contains:
    - `current_player`, `game_status`, `current_phase`.
    - `state`: full `GameState` snapshot.
    - `move`: last `Move` applied.
- **Env-level self-play failures** (soak harness):
  - `ai-service/logs/selfplay/failures/failure_*.json`
  - Each file contains:
    - `game_index`, `termination_reason`.
    - `state`: final `GameState` (if dump succeeded).
    - `last_move`: last `Move` (if dump succeeded).

Typical `termination_reason` values:

- `no_legal_moves_for_current_player` – `env.legal_moves()` returned an empty list while `GameStatus.ACTIVE`.
- `step_exception:RuntimeError` – an engine invariant blew up inside `GameEngine.apply_move`.

### 4.2 Promoting a snapshot to a regression test

1. **Copy or note** the snapshot file name and location, e.g.:
   - `ai-service/logs/invariant_failures/active_no_moves_p1_1764000947.json`
   - `ai-service/logs/selfplay/failures/failure_0_no_legal_moves_for_current_player.json`

2. **Create a new test** under `ai-service/tests/invariants/`, following the pattern of:
   - `test_active_no_moves_movement_forced_elimination_regression.py`
   - `test_active_no_moves_movement_fully_eliminated_regression.py`
   - `test_active_no_moves_movement_placements_only_regression.py`
   - `test_active_no_moves_territory_processing_regression.py`

3. Inside the test:
   - Deserialize `GameState` / `Move` from the snapshot.
   - Force-enable the strict invariant via `monkeypatch` (and/or env vars).
   - Re-apply the move through `GameEngine.apply_move`.
   - Assert that any resulting ACTIVE state for `current_player` exposes at least one interactive move or forced-elimination move.

4. Example pseudocode skeleton:

   ```python
   def test_strict_invariant_regression_from_snapshot(monkeypatch: pytest.MonkeyPatch) -> None:
       from app.models import GameState, Move
       import app.game_engine as game_engine

       # Enable strict invariant for this test
       monkeypatch.setenv("RINGRIFT_STRICT_NO_MOVE_INVARIANT", "1")
       monkeypatch.setattr(game_engine, "STRICT_NO_MOVE_INVARIANT", True, raising=False)

       payload = json.loads(Path(SNAPSHOT_PATH).read_text())
       state = GameState.model_validate(payload["state"])
       move = Move.model_validate(payload["move"])

       new_state = GameEngine.apply_move(state, move)

       if new_state.game_status == GameStatus.ACTIVE:
           legal_moves = GameEngine.get_valid_moves(new_state, new_state.current_player)
           forced = GameEngine._get_forced_elimination_moves(new_state, new_state.current_player)
           assert legal_moves or forced
   ```

5. Once the regression is added and green, **keep the original snapshot** in `logs/` so that future soaks can confirm no _new_ patterns are emerging.

---

## 5. CI strategy and recommended defaults

To keep CI runs bounded while still exercising strict-invariant semantics:

- **Unit/slow tests**:
  - `tests/test_self_play_stability.py` is marked `@pytest.mark.slow`. Configure CI to:
    - Run it with the default `RINGRIFT_SELFPLAY_STABILITY_GAMES=3` and `band=light` on at least one job.
- **Heavier soaks**:
  - Prefer running `run_self_play_soak.py` locally or in dedicated long-running jobs (outside the primary CI path).
  - Use `--difficulty-band light`, `--gc-interval`, and moderate `--max-moves` for invariant coverage.
  - When wiring these soaks into scheduled jobs or non-blocking CI lanes, pass `--fail-on-anomaly` so runs exit non-zero on strict-invariant / engine anomalies; for the TS orchestrator harness, use `--failOnViolation` on `scripts/run-orchestrator-soak.ts` (or `npm run soak:orchestrator:smoke`) to get equivalent gating behaviour.

When tuning these parameters for a new environment, monitor:

- Wall-clock runtime per 10–20 games.
- RSS / memory usage over time (to ensure `gc_interval` and light-band AIs are sufficient).
- Growth of `ai-service/logs/invariant_failures/` and `ai-service/logs/selfplay/failures/` to detect new invariant classes early.

---

## 6. Summary

- **Invariant semantics** are now TS-aligned: any ACTIVE state for `current_player` must offer placements, movements, captures, line/territory decisions, or forced elimination.
- **Soak harness** supports a light difficulty band and periodic cache/GC cleanup for long, memory-safe runs.
- **Self-play stability tests** exercise the strict invariant under a light AI band by default and can be scaled up via env vars.
- **Failure logs** under `logs/invariant_failures` and `logs/selfplay/failures` are the single source of truth for new invariant scenarios and should be regularly mined into explicit regression tests.
