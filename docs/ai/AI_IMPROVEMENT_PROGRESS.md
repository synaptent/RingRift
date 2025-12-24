# AI Strength & Training Infrastructure Progress

> **Last Updated:** 2025-12-13

This document tracks concrete AI/training improvements as they land, so work
is not duplicated and future refactors stay canonical and debuggable.

**See also:** [`ai-service/docs/AI_TRAINING_PLAN.md`](../../ai-service/docs/AI_TRAINING_PLAN.md) for the primary training guide.

Status tags:

- **done**: merged/landed in main
- **next**: ready to implement
- **blocked**: needs a prerequisite

---

## NPZ Export Pipeline (2025-12-12)

- **done** `jsonl_to_npz.py` converts JSONL selfplay data to NPZ training format
  with proper 56-channel feature encoding (14 base × 4 history frames).
- **done** Checkpointing system added to prevent data loss on long-running exports:
  - `--checkpoint-dir` for incremental chunk saves
  - `--checkpoint-interval` configurable (default: 100 games)
  - `--resume` flag for resuming from checkpoint after interruption
  - Automatic chunk merge on completion
- **done** Trained `square19_2p_v1` model (8740 samples, early stopped epoch 21).
- **done** Trained `square8_2p_v5` model on H100 (120k samples, in progress).
- **in_progress** NPZ exports for hex_2p and sq19 quality datasets on GPU cluster.

## GPU Cluster Operations (2025-12-12)

- **done** Lambda H100 instance configured for training and large model inference.
- **done** Quality games synced across instances via rsync.
- **done** SSH config aliases for cluster access (`lambda-gpu`, `ringrift-staging`).
- **done** Documentation updated with cluster operations guide in `AI_TRAINING_PLAN.md`.

---

## Multiplayer Search (3P/4P)

- **done** Paranoid reduction for Minimax/MCTS/Descent (root vs coalition), with
  sign flips driven by `current_player`, not ply parity.
- **done** “Most‑threatening opponent” selection based on **victory progress**
  (territory, elimination, and LPS proximity) for multi-player scalarization.
- **done** NeuralNetAI globals framed against the most threatening opponent for
  3P/4P, keeping inference aligned with Paranoid semantics.
- **done** NN evaluation enabled for 3P/4P MCTS/Descent under Paranoid semantics.
- **next** Promote true multi-player NN checkpoints:
  - Stable encoder for multi-player (not current‑player scalar).
  - Per-player value / rank‑dist interpretation for MaxN‑style utilities.
- **next** Add optional multi-player search modes once vector utilities exist:
  - **MaxN** (vector-valued) as the principled long-term mode.
  - **BRS / best‑reply reduction** as a cost‑controlled intermediate.
  - Keep **Paranoid** as default until tournaments show clear gains.

## Swap (Pie Rule) in 2P

- **done** Search AIs (Minimax/MCTS/Descent) now handle `swap_sides` explicitly
  using the Opening Position Classifier to avoid mis-evaluating identity swaps.
- **next** Re-run 2P canonical self-play soaks to establish new swap usage
  baseline and update registry health summaries.

## Pipeline Orchestrator (2025-12-13)

- **done** SSH retry logic with exponential backoff (3 retries, 2-30s delays
  with jitter) for robust distributed execution.
- **done** Smart polling replaces fixed 45-minute waits for selfplay/CMA-ES/training
  phase completion detection.
- **done** Checkpointing and `--resume` flag for resuming interrupted iterations
  from last completed phase.
- **done** Elo rating system (K=32) with full history persistence for model
  comparison and leaderboard tracking.
- **done** Model registry for tracking trained models with lineage (parent_id),
  metrics, and deprecation status.
- **done** Tier gating integration (D2→D4→D6→D8) with automatic promotion
  based on 55% win rate threshold over 10+ matches.
- **done** Game deduplication via SHA256 hashing to prevent duplicate games
  in training data.
- **done** Resource monitoring (CPU/MEM/DISK/GPU) across all distributed workers.
- **done** Enhanced error logging with full stdout/stderr capture to daily log files.
- **done** Board-specific heuristic profiles for all 12 board×player configurations
  (square8/square19/hex8/hex × 2p/3p/4p).
- **done** 22 diverse selfplay job configurations (mixed, heuristic-only,
  minimax-only, mcts-only, descent-only, nn-only) across all boards.
- **done** Tournament games now saved and synced for training data augmentation.
- **done** CMA-ES matrix integrated with pipeline orchestrator for canonical configs (square8/square19/hex, with hex8 included when enabled).

## Canonical Data Pipeline

Status entries in this section reflect the state at the time of the update.
Do not treat `canonical_ok` claims here as current status. Always verify against
the latest gate summaries under `ai-service/data/games/` and the registry in
`ai-service/TRAINING_DATA_REGISTRY.md`.

- **done** Regenerated and re-gated `canonical_square8_2p.db` (2P) via distributed
  `generate_canonical_selfplay.py`; parity + canonical history pass and the
  registry is updated (`canonical_ok=true`). `canonical_square8.db` remains
  canonical but is smaller and primarily used for quick parity smoke runs.
- **done** `generate_canonical_selfplay.py` now streams gate progress to stderr
  (parity + pytest gates) and keeps stdout as clean JSON to avoid long silent runs.
- **done** `generate_canonical_selfplay.py` now supports local `--reset-db`
  archiving and runs sampled canonical config/history postchecks even when parity
  fails (to surface the first invalid move quickly).
- **done** Gated initial `canonical_square8_3p.db` (3P) (`canonical_ok=true`).
- **done** Gated initial `canonical_square8_4p.db` (4P) (`canonical_ok=true`);
  scale up before training.
- **done** Regenerated and gated `canonical_square19.db` (2P); parity + canonical
  history pass (`canonical_ok=true`). Scale up for training once large-board
  throughput is acceptable.
- **done** Regenerated and gated `canonical_hexagonal.db` (radius‑12); parity + canonical
  history + hex FE/territory fixtures pass (`canonical_ok=true`). Scale up for
  training once large-board throughput is acceptable.
- **blocked** Large-board scale-up remains low volume (`canonical_square19.db`, `canonical_hexagonal.db`).
  Scale up with canonical self-play and re-run gates before training use.
- **done** Canonical dataset export path fixed (`build_canonical_dataset.py`
  - programmatic `export_replay_dataset.py`) and neural tier training now
    defaults to `canonical_square8_2p.npz` for non-demo runs.
- **next** Make canonical-only enforcement the default in remaining training CLIs;
  keep `--allow-legacy` only for historical ablations.

## Training & Evaluation

- **done** Reset/reseed reused AI instances per game in `app.training.generate_data`
  to keep self-play runs reproducible under a base seed and reduce correlated RNG.
- **done** Promotion gating in `ai-service/scripts/run_improvement_loop.py` now uses
  a Wilson CI lower-bound check (configurable `--promotion-confidence` and `--eval-games`).
- **done** Tier evaluation runner now supports 3P/4P by rotating the candidate
  across all seats and instantiating one opponent AI per seat.
- **in_progress** Re-run distributed D1–D10 tournaments (10 games/matchup) on GPU
  hosts with `TORCH_COMPILE_DISABLE=1` for stability; collect reports from
  `ai-service/results/tournaments/report_*.json`.
- **done** ReAnalyze over replay DBs: `ai-service/scripts/reanalyze_replay_dataset.py`
  can regenerate policy targets from MCTS visit counts or Descent root softmax.
- **next** Wire ReAnalyze into the training pipeline and deprecate the
  legacy stub (`ai-service/app/training/reanalyze.py`).
- **next** Generalize tier tournaments to square19 + hex and 3P/4P pools
  (`ai-service/app/training/tournament.py`).
- **next** Re‑tune heuristic weight profiles for 3P/4P and large boards using
  canonical evaluation pools.
