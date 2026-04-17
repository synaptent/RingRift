# Plan: AI Quality, Strength, Diversity, and Production Experience

**Status:** Active
**Created:** 2026-04-16
**Author:** Claude Opus 4.7 (1M context), synthesized from parallel investigation of AI architecture, training pipeline, production serving, and diversity mechanisms
**Scope:** Project quality, AI strength, AI diversity, and production website experience
**Related canonical docs:**

- [docs/RESULTS.md](../RESULTS.md) — current supported claims
- [docs/CODEBASE_QUALITY_PROGRAM.md](../CODEBASE_QUALITY_PROGRAM.md) — six-category quality framework
- [docs/ai/NN_AI_STRENGTH_REMEDIATION_PLAN.md](../ai/NN_AI_STRENGTH_REMEDIATION_PLAN.md) — earlier strength-focused plan (subset of this)
- [docs/architecture/TRAINING_INFRASTRUCTURE_STRATEGY.md](../architecture/TRAINING_INFRASTRUCTURE_STRATEGY.md)

---

## 1. Problem Statement

RingRift has built a credible AlphaZero-style training pipeline with two flagship results (`hex8_2p` 1979.8 Elo, `square8_2p` 1782.0 Elo). Investigation across the codebase surfaces a striking pattern: **substantial diversity, strength, and player-experience infrastructure already exists but is under-leveraged**. Four named heuristic personas, 23 AI types in the training distribution, an ensemble-inference module with five combiners, a CMA-ES diversity optimizer, multi-seat persona assignment, and a v4/v5-heavy architecture lineage all sit largely dormant behind a numeric D1–D10 ladder that exposes none of them to players.

At the same time, training is plateauing on hex8_2p and regressing on square8_3p, multiplayer evidence is weak, and the production AI serving path lacks telemetry, hot-reload, and observable fallback.

This plan prioritizes **wiring up existing dormant infrastructure** over building new systems, targets the **single highest-leverage bug** (seat fairness in 3p/4p evaluation), and sequences production-facing UX wins ahead of deeper training work so players see improvement quickly.

---

## 2. Key Findings (Condensed)

### AI architecture

- v2/v3/v4/v5-heavy/v5-heavy-large all in-tree. v2 in production. v4 experiment just unblocked today by the training-probe `model_version` fix.
- v5-heavy has FiLM heuristic conditioning, optional GNN, spatial policy heads — largest untapped architecture.
- GumbelMCTSAI with GPU tree mode is the production search; heuristic blend is optional but rarely wired in.

### Training

- hex8_2p plateau at 1979.8 is likely a 55% promote-threshold artifact + v2 capacity ceiling.
- square8_3p 20–30% WR is almost certainly **per-seat value-head imbalance**. Quality gate tracks opening diversity and value std but not seat-wise WR ([`model_quality_gate.py:264–299`](../../ai-service/scripts/lib/model_quality_gate.py)).
- Dirichlet root noise, hard-example mining, and EWC regularization all exist in code but are not enabled in the minimal loop.

### Diversity

- 4 personas × 51 heuristic weight params × 12 per-board profiles × CMA-ES tuning all present.
- Ensemble combiners (AVERAGE/WEIGHTED/VOTING/MAX/BAYESIAN) and dynamic online re-weighting in [`ensemble_inference.py`](../../ai-service/app/ai/ensemble_inference.py) — not used in production.
- 23 multi-seat matchup configurations in [`gpu_persona_mixin.py`](../../ai-service/app/ai/gpu_persona_mixin.py) — not exposed to players.

### Production

- Clean FastAPI `/ai/move` with per-game AI instance cache (1800s TTL, 512 max). No hot-reload, no model-version telemetry, silent fallback to local heuristic on Python service failure, 30s global timeout.
- D1–D10 ladder is board-aware via `heuristic_profile_id` but shows only a single numeric difficulty to players.
- No post-game analysis UI, no named opponents, no adaptive difficulty.

---

## 3. Tracks

Four parallel tracks. Within a track, tasks are ordered by execution sequence.

### Track A — Project Quality

| ID  | Task                                                             | Impact                                                  | Effort   | Files                                                                                                                                                                |
| --- | ---------------------------------------------------------------- | ------------------------------------------------------- | -------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| A1  | Add per-seat WR tracking to quality gate                         | **HIGH** — confirms/refutes 3p/4p structural hypothesis | ~1 day   | [`model_quality_gate.py:264`](../../ai-service/scripts/lib/model_quality_gate.py), [`minimal_alphazero_loop.py`](../../ai-service/scripts/minimal_alphazero_loop.py) |
| A2  | Plateau detector + optional auto-threshold relaxation            | HIGH                                                    | ~0.5 day | [`minimal_alphazero_loop.py`](../../ai-service/scripts/minimal_alphazero_loop.py)                                                                                    |
| A3  | Regression test for training-probe `model_version` propagation   | MEDIUM                                                  | ~2h      | `ai-service/tests/unit/scripts/test_training_probes_model_version.py` (new)                                                                                          |
| A4  | Checked-in schema for `distributed_hosts.yaml` + CI validation   | MEDIUM                                                  | ~1 day   | `ai-service/config/distributed_hosts.schema.yaml` (new)                                                                                                              |
| A5  | Playwright E2E test for VictoryModal stats path                  | MEDIUM                                                  | ~0.5 day | `tests/e2e/`                                                                                                                                                         |
| A6  | Begin `_neural_net_legacy.py` deprecation (tracked, not started) | LOW-MEDIUM                                              | ~3 days  | [`app/ai/_neural_net_legacy.py`](../../ai-service/app/ai/_neural_net_legacy.py)                                                                                      |

**A1 detail.** Extend `QualityGateTracker` with `seat_wins: dict[int, int]` and emit a `seat_wr_imbalance` warning when max/min per-seat WR ratio > 1.5. Record candidate seat per game in `staged_evaluate`. Expected output: quality-gate report `seat_wr: {1: 58%, 2: 22%, 3: 28%}` → proves whether the square8_3p issue is structural.

**A2 detail.** After every 10 iterations, if rejection rate ≥ 80% and last promotion ≥ 15 iterations ago, log `PLATEAU_DETECTED`. Optional flag to lower `promote_threshold` to 52% or bump `selfplay_randomness` for 3 iterations.

### Track B — AI Strength

| ID  | Task                                                  | Impact     | Effort               | Notes                                                                                                          |
| --- | ----------------------------------------------------- | ---------- | -------------------- | -------------------------------------------------------------------------------------------------------------- |
| B1  | Validate v4 experiment against 1980 baseline          | HIGH       | in flight on gh200-8 | checkpoint at iter 10                                                                                          |
| B2  | v5-heavy pilot on a third hex8_2p trainer             | HIGH       | ~2 days              | swap gh200-11 after 48h of v2 baseline data                                                                    |
| B3  | Seat-stratified value loss (if A1 confirms imbalance) | HIGH       | ~3–5 days            | [`neural_losses.py`](../../ai-service/app/ai/neural_losses.py) `multi_player_value_loss`                       |
| B4  | Per-tier search-budget calibration sweep              | MEDIUM     | ~1 day               | `ai-service/scripts/ladder_calibration.py` (new)                                                               |
| B5  | Dirichlet root noise in self-play                     | MEDIUM     | ~4h                  | [`gumbel_mcts_ai.py`](../../ai-service/app/ai/gumbel_mcts_ai.py)                                               |
| B6  | Wire hard-example mining into minimal loop            | MEDIUM     | ~1–2 days            | [`hard_example_mining.py`](../../ai-service/app/training/enhancements/hard_example_mining.py) (exists, unused) |
| B7  | LR warm-restarts + EWC for plateau-breaking           | LOW-MEDIUM | ~2h                  | [`ewc_regularization.py`](../../ai-service/app/training/enhancements/ewc_regularization.py) (exists, unused)   |

**B2 detail.** v5-heavy has strongest untapped potential: FiLM heuristic conditioning, optional GNN on adjacency graph, configurable 6 SE + 5 attention blocks, 160 filters. Initialize either from scratch or transfer-init from v2 conv blocks. Training config mirrors v4's shape with `TRAINING_MODEL_VERSION=v5-heavy`.

**B3 detail.** In `multi_player_value_loss`, compute separate MSE per seat, then average. Prevents the value head from learning "seat 1 usually wins." Alternative: seat-balanced batching.

### Track C — AI Diversity (highest ROI — mostly wiring)

| ID  | Task                                            | Impact                                         | Effort    | Notes                                                                                                                                                 |
| --- | ----------------------------------------------- | ---------------------------------------------- | --------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| C1  | Ensemble serving for D9–D10 tiers               | **HIGH** — expected +30–60 Elo                 | ~2 days   | [`ensemble_inference.py`](../../ai-service/app/ai/ensemble_inference.py) (exists, unused in prod)                                                     |
| C2  | Expose personas in production UI (4× opponents) | **HIGH** — biggest UX delta with zero training | ~1–2 days | [`aiQuickPlay.ts`](../../src/client/config/aiQuickPlay.ts), [`AIEngine.ts:82–143`](../../src/server/game/ai/AIEngine.ts)                              |
| C3  | Varied seat assignment for multiplayer AI games | MEDIUM                                         | ~1 day    | [`createGameRoute.ts`](../../src/server/routes/game/createGameRoute.ts), [`gpu_persona_mixin.py:14–39`](../../ai-service/app/ai/gpu_persona_mixin.py) |
| C4  | Adaptive difficulty / "Rival" mode              | MEDIUM-HIGH                                    | ~2–3 days | new client + server work                                                                                                                              |
| C5  | Opening book from promotion-winning games       | MEDIUM                                         | ~2 days   | `ai-service/scripts/build_opening_book.py` (new)                                                                                                      |
| C6  | Per-persona PUCT constant variants              | LOW-MEDIUM                                     | ~1 day    | [`gumbel_mcts_ai.py:230`](../../ai-service/app/ai/gumbel_mcts_ai.py), AIConfig                                                                        |
| C7  | Uncertainty / confidence display for engagement | LOW                                            | ~4h       | KL-divergence already computed in ensemble_inference                                                                                                  |

**C1 detail.** Load 3 checkpoints (canonical + last 2 promotions) for a given config. For D9–D10 use BAYESIAN ensemble; for D6–D8 keep single-model. Typical ensemble gain +30–60 Elo. Risk: ~3× inference latency for D10. Mitigate by batching NN forward passes.

**C2 detail.** Add a persona overlay to the existing D1–D10 grid. Names for the four heuristic personas (Balanced / Aggressive / Territorial / Defensive). Extend `AI_DIFFICULTY_PRESETS` to include `persona_id`; `/ai/move` already accepts persona-aware configs via [`heuristic_weights.py`](../../ai-service/app/ai/heuristic_weights.py). **Zero new training required.**

**C3 detail.** When a user creates a 3p/4p game with multiple AI opponents, default to a varied mix (e.g., `3p_mixed: [aggressive, defensive, territorial]`) instead of three identical bots.

### Track D — Production Website Experience

| ID  | Task                                                  | Impact                           | Effort         | Notes                                                                                                           |
| --- | ----------------------------------------------------- | -------------------------------- | -------------- | --------------------------------------------------------------------------------------------------------------- |
| D1  | Model-version telemetry on `/ai/move`                 | **HIGH** — unblocks all safe A/B | ~4h            | [`main.py`](../../ai-service/app/main.py), response headers, Prometheus                                         |
| D2  | Hot-reload endpoint / mtime watcher                   | MEDIUM                           | ~1 day         | [`main.py:399–494`](../../ai-service/app/main.py)                                                               |
| D3  | Calibrated AI Elo ladder visible to players           | MEDIUM                           | ~2 days        | `docs/data/ai_ladder_elo.json` (new), client UI                                                                 |
| D4  | Post-game analysis view                               | **HIGH**                         | ~3–5 days      | existing `evaluationHistory` + new client UI                                                                    |
| D5  | Silent-fallback observability + alerting              | MEDIUM                           | ~4h            | [`AIEngine.ts:299–315`](../../src/server/game/ai/AIEngine.ts), Prometheus                                       |
| D6  | Per-tier inference latency SLOs with graceful degrade | MEDIUM                           | ~1 day         | [`main.py`](../../ai-service/app/main.py), [`AIServiceClient.ts`](../../src/server/services/AIServiceClient.ts) |
| D7  | Named opponents / personality layer                   | LOW-MEDIUM                       | ~2–3 days (UX) | builds on C2                                                                                                    |

**D1 detail.** Add `X-RingRift-Model-Version` response header on `/ai/move`. Log `{game_id, player, tier, model_path, model_version, latency_ms}` per move. Prometheus counter `ai_moves_by_model_version_total`. Without this, every other production change flies blind.

**D4 detail.** Client already subscribes to `PositionEvaluationPayload`. Extend the game-end flow to show move-by-move eval swing from the AI's perspective, highlight "critical moves" where win probability swung >20%. Biggest single player-experience improvement in the plan.

**D5 detail.** When the Python service fails, `AIEngine.ts` silently falls back to local heuristic. Metric: `ai_fallback_moves_total` labeled by tier and reason. Alert when fallback rate > 5% over 5 min.

---

## 4. Execution Sequencing

### Week 1 — ship-ready, high-impact, low-risk

- **A1** per-seat WR tracking (confirms 3p/4p hypothesis)
- **A2** plateau detector
- **C2** personas in UI (4× opponents overnight)
- **D1** model-version telemetry
- **D5** fallback observability
- Monitor newly-spawned training: v4 on gh200-8, hex8_2p_b on gh200-11, hex8_4p on gh200-13

### Week 2–3 — contingent on earlier wins

- **B2** v5-heavy pilot (if v4 has not promoted)
- **C1** ensemble serving for D9–D10
- **C3** varied multiplayer seating
- **D2** hot reload
- **B3** seat-stratified loss (if A1 confirms imbalance)

### Week 4+

- **D3** calibrated Elo ladder
- **D4** post-game analysis view
- **B4/B5/B6** calibration / Dirichlet / hard-example mining
- **C4** adaptive difficulty
- **C5** opening book
- **A4** fleet inventory contract
- **A6** start legacy removal

### Longer horizon (2+ months)

- Multi-teacher distillation (train style-specific models from persona game logs)
- Large-board feasibility (`square19`, `hexagonal`) — v5-heavy-efficient on multi-node selfplay
- LLM-augmented post-game narrative

---

## 5. What to Do First (One-Day Budget)

1. **A1** per-seat WR tracking — answers the biggest open multiplayer-training question.
2. **C2** personas in UI — biggest player-experience unlock without training anything new.
3. **D1** model-version telemetry — without it, everything else in production is flying blind.

These three together in one day.

---

## 6. Risks and Mitigations

| Risk                                                    | Mitigation                                                                                   |
| ------------------------------------------------------- | -------------------------------------------------------------------------------------------- |
| Ensemble serving (C1) triples D10 latency               | Batch NN forward passes; gate behind feature flag                                            |
| Persona UI (C2) confuses players                        | Keep numeric D1–D10 as primary axis; persona as secondary selector with descriptive tooltips |
| Seat-stratified loss (B3) destabilizes training         | Feature-flagged; run in parallel with baseline loss on a dedicated node before promoting     |
| v5-heavy pilot (B2) consumes two weeks without gain     | Kill switch at iter 10 if no promotion above 1500 baseline                                   |
| Auto-threshold relaxation (A2) promotes weak candidates | Only lower to 52% not below; require 100+ games at the new threshold                         |
| Hot reload (D2) serves partial weights                  | Atomic swap: load new checkpoint into staging slot, then swap pointer                        |

---

## 7. Success Metrics

- **Project quality:** `CODEBASE_QUALITY_PROGRAM.md` score rises 8.2 → 8.7+; fewer than 5 skipped tests in full gate.
- **AI strength:** at least one config above 2000 Elo within 8 weeks; square8_3p above 1600 within 4 weeks.
- **AI diversity:** production serves ≥ 4 distinguishable personas; multiplayer games default to mixed-persona seating.
- **Production:** p95 inference latency within per-tier SLO; model-version telemetry on 100% of `/ai/move` calls; fallback rate < 1% under normal operation.

---

## 8. Tracking

- **GitHub issues** — one per Week 1 task (see [`gh issue list`](https://github.com/synaptent/RingRift/issues) filtered by `ai-plan-2026-04`)
- **This document** is the source of truth for scope and sequencing
- **[TODO.md](../../TODO.md)** references this plan under Active Priorities
- **Progress updates** land in a follow-up `AI_QUALITY_STRENGTH_DIVERSITY_PROGRESS.md` in this directory as tasks complete
