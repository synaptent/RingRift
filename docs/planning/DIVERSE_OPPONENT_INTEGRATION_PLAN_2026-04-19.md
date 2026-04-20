# Diverse Opponent Integration Plan — 2026-04-19

**Status:** DESIGN + Wire B EXECUTING
**Scope:** Wire existing diverse-selfplay machinery into the active minimal-loop trainers. No new infrastructure. Minimal disruption to the 7-node fleet.

**Motivation:** Three 2p configurations (hex8_2p v4, hex8_2p v5-heavy, square8_2p) are plateaued with the classical self-play-collapse signature — rejections clustered tightly near the promotion threshold (0.46, 0.465, 0.47 on square8_2p; exactly 0.475 on v5-heavy across 4 consecutive iters). Literature (Katago, AlphaStar, Player of Games) shows pure self-play is insufficient for complex non-2p-zero-sum games; diverse-opponent training breaks the distribution collapse.

**Key insight:** The codebase already has ~90% of the diversity machinery. This plan is about _wiring existing code_, not building new.

## References

- `ai-service/scripts/minimal_alphazero_loop.py` — trainer (DO NOT modify directly; use sidecars)
- `ai-service/app/training/diverse_ai_config.py` — `get_diverse_matchups()`, `GPU_OPTIMIZED_WEIGHTS`
- `ai-service/scripts/generate_gumbel_selfplay.py:177` — `GumbelSelfplayConfig.opponent_type` field
- `ai-service/scripts/policy_selfplay_worker.py` — batch + ingest loop
- `ai-service/scripts/ingest_policy_selfplay.py` — policy-entropy gate, NPZ output
- `ai-service/app/training/game_gauntlet.py:442` — `BaselineOpponent` enum, Elo ladder
- `ai-service/scripts/export_replay_dataset.py:826` — `_build_source_filters_from_args` + `RINGRIFT_INCLUDE_GAUNTLET` env hook

## 1. Pre-conditions (verified against live fleet)

### Wire B — gauntlet → supplemental shard (FIRST, ready now)

**Pre-condition:** `data/games/gauntlet_*.db` files exist with games tagged `source="gauntlet"`.

**Verified status (2026-04-19):** mac-studio (coordinator, 10.0.0.90) has **~100 GB of gauntlet data** dated Mar 24, 2026 (in-repo path) and Mar 1, 2026 (OWC archive). Every active trainer config has matching data:

| Active config          | Gauntlet DB (mac-studio in-repo) | Size   |
| ---------------------- | -------------------------------- | ------ |
| hex8_2p (v4, v5-heavy) | `gauntlet_hex8_2p.db`            | 3.0 GB |
| square8_2p             | `gauntlet_square8_2p.db`         | 3.7 GB |
| hex8_3p                | `gauntlet_hex8_3p.db`            | 1.0 GB |
| square8_3p             | `gauntlet_square8_3p.db`         | 1.7 GB |
| hex8_4p                | `gauntlet_hex8_4p.db`            | 4.4 GB |

Wire B runs **on mac-studio** (coordinator-centralized), produces NPZ shards via existing `export_replay_dataset.py --include-gauntlet --min-elo 1400`, rsyncs to trainer's `--supplemental-data-dir`. **Ready now, zero code.**

### Wire A — selfplay-worker node with diverse `opponent_type`

**Pre-condition:** A node running `policy_selfplay_worker.py` with diverse opponent sampling, writing NPZ shards to a dir the trainer reads.

**Verified status:** `gh200-14` (square19_2p) is idle — `systemctl is-active ringrift-training` returns `inactive`, `Result=success` (clean exit, not crash). Zero-risk slot for Wire A. Cost of repurposing: zero.

### Wire C — per-seat diverse AI in self-play

**Pre-condition:** Cannot break game-granular resume (commit 69fc25aa0).

**Verified status:** `_play_game()` in `minimal_alphazero_loop.py:109` already accepts `dict[int, AI]` with per-seat instances. The resume uses JSONL line count only, not AI identity. A **sidecar script** `scripts/experiments/diverse_selfplay_runner.py` can import `_play_game` + `_make_env` without modifying the loop. Zero risk to protected files.

## 2. Execution sequence (revised)

**Revised order with verified preconditions: B → A → C.**

### Phase 1 — Wire B (immediate, zero code, zero protected-file touches)

Target: hex8_2p on gh200-8 (v4 plateau). Single config at first — signal attribution requires isolation.

On mac-studio, for hex8_2p:

```
PYTHONPATH=. python scripts/export_replay_dataset.py \
  --db data/games/gauntlet_hex8_2p.db \
  --board-type hex8 --num-players 2 \
  --include-gauntlet \
  --min-elo 1400 \
  --encoder-version v3 \
  --output /tmp/gauntlet_hex8_2p_supplemental.npz
```

Then rsync to gh200-8:

```
rsync /tmp/gauntlet_hex8_2p_supplemental.npz \
  ubuntu@gh200-8:/home/ubuntu/ringrift/ai-service/data/minimal_loop_hex8_2p_v4/supplemental/
```

The trainer's sliding-window merge (`minimal_alphazero_loop.py:1161-1166`) picks it up automatically on the next iteration. No service restart required.

### Phase 2 — Wire A (low risk, pure config)

Re-activate `ringrift-selfplay-worker.service` on **gh200-14** (idle node). Set `--opponent-type` to use `GPU_OPTIMIZED_WEIGHTS` from `diverse_ai_config.py`. Point `--supplemental-output-dir` at a mac-studio-reachable path that rsyncs to one trainer's supplemental-data-dir.

### Phase 3 — Wire C (sidecar script, no loop changes)

Create `scripts/experiments/diverse_selfplay_runner.py` (~150 LOC). Imports `_play_game` from the loop, constructs a mixed AI dict via `diverse_ai_config.get_weighted_ai_type()`, writes JSONL, calls `ingest_policy_selfplay_files` for NPZ conversion. Produces 10-20% supplemental shards.

## 3. Rollback paths

**All three wires:**

```
rm /path/to/supplemental_dir/*.npz   # cancels within one iteration
```

No checkpoint state, `best.pth`, or `metrics.jsonl` row is ever touched by a sidecar.

**Wire A additional:** `systemctl stop ringrift-selfplay-worker` on gh200-14.

**Rollback trigger signals:**

- Promoted Elo rate drops below prior 5-iter average by >5 Elo/iter
- 3 consecutive staged-eval rejections while a control config (no supplemental) is still promoting
- Seat imbalance in `iter_*.jsonl` self-play exceeds 70/30 split — suggests value target distortion

## 4. Observability protocol

Tag shards with filename prefix: `gauntlet_*.npz`, `diverse_*.npz`, `worker_*.npz`. Trainer logs "combined N local + M supplemental NPZ files → K samples" at line 1176 — filenames are grep-able.

Per-iteration attribution signals:

- Supplemental fraction: `supplemental_samples / (selfplay_samples + supplemental_samples)`, target 5% to start
- Promoted Elo delta vs. prior 5-iter moving average
- `p1_wins`/`p2_wins` ratio in self-play JSONL (should stay 45-55% if diverse-opponent doesn't distort value targets)

## 5. Decision gates

Minimum observation window: **5 full iterations after supplemental dir populated**. Training window is `--train-window 10` default, so several iters are needed before supplemental data reaches meaningful fraction of training.

| Signal                                            | Interpretation             | Action                                |
| ------------------------------------------------- | -------------------------- | ------------------------------------- |
| Promoted Elo rate > +10/iter vs. prior 5-iter     | Wire accelerating progress | Keep, add square8_2p                  |
| Promoted Elo flat, win_rate stable                | Wire neutral               | Run 10 more iters                     |
| Promoted Elo negative or 3 consecutive rejections | Wire degrading             | Rollback, diagnose                    |
| Selfplay seat imbalance >70/30                    | Value target distortion    | Cut supplemental fraction to 2%       |
| More frequent stage-1 early promotions            | Strong positive            | Increase supplemental fraction to 15% |

## 6. Minimal file changes per wire

**Wire B:** zero new files. One-shot script in bash or `scripts/experiments/run_gauntlet_export.sh` (optional, ~10 LOC) that wraps `export_replay_dataset.py` + rsync.

**Wire A:** zero code changes (systemd config only). Optional `--diverse-opponent-sampling` flag on `policy_selfplay_worker.py` (~5 lines).

**Wire C:** one new file `scripts/experiments/diverse_selfplay_runner.py` (~150 LOC). **No modification to `minimal_alphazero_loop.py`.**

## 7. Resolved decisions (previously open)

| Question               | Resolution                                                   |
| ---------------------- | ------------------------------------------------------------ |
| gh200-14 status        | idle, systemd inactive, use for Wire A                       |
| Supplemental transport | coordinator-centralized on mac-studio, rsync to trainers     |
| Diverse fraction       | start 5%, scale to 10-15% if positive signal over 5 iters    |
| Wire C seat-2 pool     | same-config historical checkpoints only (action space match) |
| Priority config        | hex8_2p exclusively first (plateaued flagship)               |

## 8. Data caveat on gauntlet shards

Gauntlet DBs on mac-studio are from Mar 1–24 2026. The candidate models in those games were older canonical checkpoints. Training on them nudges the current model toward old-candidate style.

**Mitigation options:**

1. **Value-only training** — filter export to exclude policy targets (only game outcomes)
2. **Reanalysis** — `scripts/reanalyze_replay_dataset.py` exists; can re-run MCTS with current model over old positions to produce fresh policy labels + old diverse trajectories (Katago-style)
3. **Accept the drift** — old policies are still game-valid; some drift toward older style may not matter much

Start with option 3 (simplest) and observe whether value target distortion appears. If it does, pivot to option 1 or 2.

---

**Agent sources:** Explore pass by `feature-dev:code-explorer` (agent `ad3cf313bbdb20e5b`); Plan pass by `feature-dev:code-architect` (agent `ac53faf83af38e1b6`). Plan-agent factual errors (gauntlet DBs assumed missing; gh200-14 assumed active) corrected from live-fleet SSH verification.

---

## 2026-04-20 Addendum — Inert-shard correction + pivot

**Contract failure on v5-heavy experiment:** the shard deployed to `gh200-11:/home/ubuntu/ringrift/ai-service/data/minimal_loop_hex8_2p_v5_heavy/supplemental/` was **never consumed by training**. Root cause: gh200-11's live systemd `ExecStart` did not include `--supplemental-data-dir`, and `/etc/ringrift/training.conf` did not set `TRAINING_SUPPLEMENTAL_DATA_DIR`. All 6 completed metrics rows show `supplemental_data_dir=""`. The shard sat inert for ~15 hours. Lesson: **verify the consuming flag is wired, not just that the file exists on disk.**

**More fundamental issue on v5-heavy:** independent of the diversity experiment, every completed iter 1-6 logs `DEAD_VALUE_HEAD: value std=0.000000 across 7315 positions (mean=0.0000)`. Training loss changes across iters (398 → 283 → 463 → 124 → 130 → 92) so weights ARE updating, but the value-head output is identically zero every time. Combined with training losses 20-100× higher than every other config (normal range 3.4-6.3), this is strong evidence of a v5-heavy-specific training pathology (architectural or loss-scaling), NOT a distribution-collapse plateau. Diversity data cannot help a dead value head.

**Actions taken 2026-04-20:**

- Removed inert shard from gh200-11
- Wired `--supplemental-data-dir ${TRAINING_SUPPLEMENTAL_DATA_DIR}` into gh200-11 systemd ExecStart and `TRAINING_SUPPLEMENTAL_DATA_DIR=data/minimal_loop_hex8_2p_v5_heavy/supplemental` into `/etc/ringrift/training.conf`. `daemon-reload` executed. Service NOT restarted — deferred until (a) iter 7 eval completes cleanly and (b) the value-head pathology diagnosis lands
- Pivoted primary diversity experiment to **gh200-8 v4** (hex8_2p, 1 promo at 1534.9 Elo, 7-iter rejection streak at 0.44-0.47, healthy loss ~4.4). Deployed 2000-sample gauntlet shard (64ch, no heuristics, `--encoder-version v3`, `--canonical-model` from gh200-8's live best.pth). Shard landed at `/home/ubuntu/ringrift/ai-service/data/minimal_loop_hex8_2p_v4/supplemental/gauntlet_v4_v3noheur_2k.npz`. Next training step (iter 16) will be the first post-ingest data point
- Verified gh200-9 iter 46 mid-eval WR = 0.477 (above prior 3-iter mean of 0.453). Not rolling back the 1k shard there — will continue observation

**Open:** v5-heavy value-head pathology diagnosis (delegated to Codex, no cluster touch needed; deliverable is `docs/planning/V5_HEAVY_VALUE_HEAD_PATHOLOGY_2026-04-20.md`).

**Updated primary/secondary experiments:**

- Primary: gh200-8 v4 hex8_2p with 2000-sample gauntlet supplement (64ch, no heuristics)
- Secondary: gh200-9 square8_2p with 1000-sample gauntlet supplement (56ch, no heuristics)
- v5-heavy: frozen pending pathology diagnosis; service left running only to complete iter 7 eval as a control data point
