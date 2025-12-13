# Self-Play Data Hygiene (JSONL streams)

This doc describes how to keep self-play JSONL outputs usable for:

- last‑N‑hours analysis (to detect regressions fast), and
- canonical training pipelines (to avoid ingesting toxic / non‑canonical games).

The canonical rules specs remain the semantic source of truth; self-play outputs are only usable if they conform to those rules.

---

## 1. “Bad data” signatures (treat as toxic)

### 1.1 Timeout-dominated streams

If a `(boardType, numPlayers)` stream is dominated by `victory_type=timeout` (or any other synthetic terminal reason) it should be treated as **training-toxic** until root-caused.

Common causes:

- generator only implemented `ring_placement` and never reached real movement,
- move-limit policy too low for the board/player-count,
- engine bug causing dead loops or no-op spam.

### 1.2 Canonical config drift (rings/threshold mismatch)

Treat a stream as non-canonical if the initial state indicates:

- `ringsInHand` does not match canonical `ringsPerPlayer` for the board type, or
- `victoryThreshold` does not match RR‑CANON‑R061 for that board type + player count.

This usually indicates:

- stale processes still running older code,
- mismatched `board_size` ↔ `board_type` mapping (notably GPU hex embeddings),
- experimental `rulesOptions.ringsPerPlayer` overrides.

---

## 2. Fast “last 24h” audit (cluster + local)

From `ai-service/`:

- `PYTHONPATH=. python scripts/collect_last24h_selfplay_reports.py --scan-profile recent`

Then inspect:

- `ai-service/logs/selfplay/collected_last24h/<ts>/combined.md`
  - **Config Drift (ring supply)**
  - **Outcomes By ringsPerPlayer (mixed configs)**

Notes:

- File selection is “last 24h” by JSONL modification time, but the analyzer also filters **games** by per-game timestamps when present (`--game-max-age-hours`), which avoids counting older games in recently-copied/merged files.
- The `recent` scan profile excludes known stale/backfill buckets, including `data/selfplay/toxic_archives/**` and `data/selfplay/imported/**`.

---

## 3. Quarantine procedure (do not delete by default)

When a stream is determined to be toxic/non-canonical:

1. **Stop the generator** (see §4).
2. Move the output directory under:
   - `ai-service/data/selfplay/toxic_archives/<bucket>.<reason>.<timestamp>/`

This keeps the data available for forensic debugging while preventing it from polluting routine “recent” reports.

---

## 4. Stopping bad generators (cluster)

Typical offenders:

- GPU pipeline: `scripts/run_gpu_selfplay.py`
- CPU pipeline: `scripts/run_self_play_soak.py`

On the target host:

- find the process: `ps aux | grep -E 'run_gpu_selfplay.py|run_self_play_soak.py' | grep -v grep`
- stop it: `kill <pid>` (then `kill -9 <pid>` only if it doesn’t terminate)

After stopping, quarantine the output directory (see §3) and rerun the `recent` report to confirm the toxicity is gone from the aggregate.
