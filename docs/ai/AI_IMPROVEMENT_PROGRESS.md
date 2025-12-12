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

## Swap (Pie Rule) in 2P

- **done** Search AIs (Minimax/MCTS/Descent) now handle `swap_sides` explicitly
  using the Opening Position Classifier to avoid mis-evaluating identity swaps.
- **next** Re-run 2P canonical self-play soaks to establish new swap usage
  baseline and update registry health summaries.

## Canonical Data Pipeline

- **next** Regenerate canonical replay DBs via
  `ai-service/scripts/generate_canonical_selfplay.py` for:
  - `canonical_square8.db` (2P + 3P/4P variants)
  - `canonical_square19.db`
  - `canonical_hex.db` (radius‑12)
    then update `ai-service/TRAINING_DATA_REGISTRY.md` with gate summaries.
- **next** Make canonical-only enforcement the default in all training CLIs;
  keep `--allow-legacy` only for historical ablations.

## Training & Evaluation

- **next** Implement real ReAnalyze over DB‑stored games
  (`ai-service/app/training/reanalyze.py`) to refresh targets with newer models.
- **next** Generalize tier tournaments / gates to square19 + hex and 3P/4P pools
  (`ai-service/app/training/tournament.py`,
  `ai-service/app/training/tier_eval_config.py`).
- **next** Re‑tune heuristic weight profiles for 3P/4P and large boards using
  canonical evaluation pools.
