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
  [--gc-interval 50]
```

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
