# NN Strengthening and Self-Improvement Loop Plan

Status: draft (2025-12)

Goal: produce neural-network (NN) models that reliably beat the current production baselines
and pass the promotion gates for ladder tiers (D6+), while establishing a repeatable,
closed-loop self-improvement pipeline (selfplay -> training -> evaluation -> promotion).

This plan assumes canonical data sources only (see `ai-service/TRAINING_DATA_REGISTRY.md`)
and keeps TS rules parity as the semantic SSoT.

---

## Success criteria (production readiness)

1. Strength:
   - > = +50 Elo vs current production baseline on square8 2P with stable confidence.
   - > = +25 Elo vs previous NN champion across 3 seeds and >= 500 games total.
2. Stability:
   - No regression on contract vectors or parity gates.
   - No increase in non-canonical history or parity divergence in training DBs.
3. Coverage:
   - Square8 2P is green; square19 and hexagonal have gated canonical DBs and
     pass the same evaluation pipeline.
4. Operational:
   - Training can run end-to-end unattended with reproducible configs, metrics,
     and automated promotion artifacts.

---

## Current system anchors (existing SSoT)

- Rules and parity: `RULES_CANONICAL_SPEC.md`, `src/shared/engine/**`.
- Canonical DB gate: `ai-service/scripts/generate_canonical_selfplay.py`.
- Training pipeline: `ai-service/app/training/**`.
- NN architectures: `ai-service/app/ai/neural_net/square_architectures.py`,
  `ai-service/app/ai/neural_net/hex_architectures.py`.
- Evaluation:
  - `ai-service/scripts/run_model_elo_tournament.py`
  - `ai-service/scripts/evaluate_ai_models.py`
  - `ai-service/scripts/run_tournament.py`

---

## Workstreams and opportunities

### A) Data quality and selfplay (primary leverage)

Objective: feed the NN with canonical, diverse, and current data.

- Scale canonical DB volume by board and player count using
  `generate_canonical_selfplay.py` and the registry targets.
- Use engine modes strategically (from `app/training/selfplay_config.py`):
  - `nnue-guided`, `policy-only`, `nn-descent`, `gumbel-mcts` for exploration.
  - Evaluate `ig-gmo` and `cage` as experimental exploration sources.
- Add curriculum and temperature schedules:
  - Early training: higher temperature + exploration noise.
  - Late training: lower temperature, focused on strong-play trajectories.
- Ensure move serialization includes `placement_count`, recovery/capture bookkeeping
  moves, and correct phase history (already required by RR-CANON-R075).
- Use the quality scoring pipeline (`QualityBridge`, `HotDataBuffer`,
  `StreamingDataPipeline`) to up-weight high-quality games and reduce noise.

Deliverables:

- Canonical DBs per board pass parity/history gates with sufficient volume.
- Selfplay configs for each board captured as versioned configs (YAML or JSON).

### B) Dataset generation and feature encoding

Objective: produce stable, informative training targets and consistent encodings.

- Validate that encoders in `app/training/encoding.py` match canonical rules
  and include FE/ANM/line/territory phases.
- Add board-aware data augmentation:
  - Square: rotations/reflections.
  - Hex: rotational symmetry (60-degree steps) and reflection if supported.
- Improve policy targets:
  - Blend MCTS policy targets with heuristic targets to prevent collapse.
  - Optional KL regularization to keep policy entropy in early training.
- Introduce dataset freshness windows (e.g., last N weeks) to prevent stale data.

Deliverables:

- A documented dataset schema and augmentation policy per board type.
- NPZ exporters aligned with canonical DBs and recorded metadata.

### C) Model architecture upgrades (square + hex)

Objective: increase capacity without destabilizing training.

- Baseline comparison sweep across existing architectures:
  - Square: `RingRiftCNN_v3`, `RingRiftCNN_v4`.
  - Hex: `HexNeuralNet_v3`, `HexNeuralNet_v3_Lite`.
- Add or evaluate:
  - Residual depth increases + SE blocks.
  - Lightweight attention in mid/late stages (if memory permits).
  - Policy head improvements: spatial logits + special move heads.
  - Auxiliary heads (rank distribution, phase prediction) for richer supervision.
- Tie architecture to board size via `policy_size` and memory scaling in
  `app/training/config.py` so the same code scales across boards.

Deliverables:

- Architecture benchmark table with ELO deltas and training stability metrics.
- Selected "vNext" architecture per board type for promotion.

### D) Training loop and optimization

Objective: improve convergence speed and avoid regression.

- Loss design:
  - Policy loss (cross-entropy), value loss (MSE or Huber), optional entropy
    bonus early training, label smoothing for policy.
- Optimizer and schedule:
  - AdamW + cosine decay + warmup; gradient clipping.
  - Mixed precision with stability checks.
- Sampling strategy:
  - Prioritized replay by quality score, with diversity constraints.
  - Balance across phases and game lengths to avoid phase bias.
- Regular evaluation checkpoints integrated into training loop
  (e.g., every N steps run a mini Elo gate).

Deliverables:

- Documented training config templates with versioned parameters.
- Training telemetry for loss, policy entropy, value calibration, and ELO.

### E) Evaluation and promotion gates

Objective: reliable go/no-go decisions for production.

- Standardize evaluation:
  - `run_model_elo_tournament.py` (recorded DB + metadata).
  - `evaluate_ai_models.py` for targeted tests.
  - `run_tournament.py` basic mode for quick sanity checks.
- Add board-specific gates:
  - Square8 2P primary gate.
  - Square19/hex parity gate before promotion.
- Track regressions:
  - Contract vectors, parity checks, scenario suites.
  - Maintain a fixed holdout DB for regression tests.

Deliverables:

- Promotion checklist aligned with ladder tiers.
- Standard summary JSON outputs for CI and dashboards.

### F) Closed-loop automation (self-improvement loop)

Objective: run the loop continuously with minimal manual intervention.

Pipeline loop (automated):

1. Selfplay (canonical) -> record DB and metadata.
2. Quality scoring -> manifest update.
3. Dataset export -> NPZ/streaming.
4. Training -> candidate model.
5. Evaluation -> Elo + parity + scenario checks.
6. Promotion -> update ladder config + model registry.
7. Repeat with new champion as selfplay seed.

Deliverables:

- One command / orchestrator entrypoint for the loop.
- Scheduled runs with status dashboards and alerts.

---

## Recommended execution order (phased plan)

Phase 0 (1-2 days): Baseline audit

- Identify current best NN model per board (ELO + config).
- Capture a minimal baseline evaluation bundle for comparison.

Phase 1 (1-2 weeks): Data and selfplay

- Scale canonical DBs to volume targets.
- Turn on quality-weighted sampling.
- Validate parity + canonical history on all new DBs.

Phase 2 (2-4 weeks): Architecture + training loop

- Run controlled architecture ablations (square8 first).
- Improve loss/schedule and sampling; pick the best configuration.

Phase 3 (2-3 weeks): Automation and promotion

- Wire the full loop (selfplay -> train -> evaluate -> promote).
- Add gating checks to CI / scheduled jobs.
- Promote the first NN that clears all gates.

---

## Metrics to track (minimum set)

- ELO vs baseline and vs previous champion.
- Win rate by phase category (capture-heavy, line/territory-heavy).
- Policy entropy and value calibration error.
- Parity gate pass/fail counts.
- Data freshness: percent of samples from last N days.

---

## Open questions

1. Which board type is the first production target for NN (square8 2P only or
   multi-board at once)?
2. What is the required margin vs the current heuristic baseline for promotion?
3. Which experimental engine modes (`ig-gmo`, `cage`) are acceptable for training
   data generation vs evaluation only?
