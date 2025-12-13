# AI Strength & Training Infrastructure Progress

This document tracks concrete AI/training improvements as they land, so work
is not duplicated and future refactors stay canonical and debuggable.

Status tags:

- **done**: merged/landed in main
- **next**: ready to implement
- **blocked**: needs a prerequisite

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
- **done** Board-specific heuristic profiles for all 9 board×player configurations
  (square8/square19/hex × 2p/3p/4p).
- **done** 22 diverse selfplay job configurations (mixed, heuristic-only,
  minimax-only, mcts-only, descent-only, nn-only) across all boards.
- **done** Tournament games now saved and synced for training data augmentation.
- **done** CMA-ES matrix integrated with pipeline orchestrator for all 9 configs.

## Canonical Data Pipeline

- **done** Regenerated and re-gated `canonical_square8.db` (2P) via distributed
  `generate_canonical_selfplay.py`; parity + canonical history pass and the
  registry is updated (`canonical_ok=true`).
- **done** `generate_canonical_selfplay.py` now streams gate progress to stderr
  (parity + pytest gates) and keeps stdout as clean JSON to avoid long silent runs.
- **done** Gated initial `canonical_square8_3p.db` (3P) (`canonical_ok=true`).
- **done** Gated initial `canonical_square8_4p.db` (4P) (`canonical_ok=true`);
  scale up before training.
- **blocked** `canonical_square19.db` (2P) needs regeneration; latest parity-only
  recheck fails a phase/move invariant (`place_ring` applied during
  `territory_processing`). Re-run `generate_canonical_selfplay.py` to rebuild the DB
  and update `ai-service/TRAINING_DATA_REGISTRY.md` once `canonical_ok=true`.
- **next** Regenerate `canonical_hex.db` (radius‑12) and gate it, then update
  `ai-service/TRAINING_DATA_REGISTRY.md` with gate summaries.
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
