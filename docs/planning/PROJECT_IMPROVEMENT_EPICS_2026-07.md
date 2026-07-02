# Project Improvement Epics — July 2026

**Status**: Active roadmap
**Created**: 2026-07-01
**Source**: Full-repo evaluation (game rules engine, product surface, AI pipeline, docs)
**Tracking**: Each epic has a GitHub issue labeled `epic`. This document is the canonical scope
reference; issues track execution state.

## Purpose

RingRift's fundamentals are strong: a formally specified rules engine (TypeScript source of
truth, RR-CANON rule IDs), a live product at ringrift.ai, and replicated AlphaZero self-play
results (hex8_2p 2583.9 Elo on the fv3 lane, independent seed at 2193.4). The gap is
_consumability_: the novel parts are hard for outsiders — researchers, players, contributors —
to use. These 11 epics close that gap.

## Priority order

Execution order balances value against dependency and risk:

| Order | Epic                                                   | Issue                                                    | Why this position                                                      |
| ----- | ------------------------------------------------------ | -------------------------------------------------------- | ---------------------------------------------------------------------- |
| 1     | E2: Silent-failures research write-up                  | [#101](https://github.com/synaptent/RingRift/issues/101) | Highest transferable value, zero code risk, all source material exists |
| 2     | E1: `ringrift-env` RL environment package              | [#100](https://github.com/synaptent/RingRift/issues/100) | Biggest "useful to others" move; large but self-contained              |
| 3     | E5: Puzzle mode from self-play data                    | [#104](https://github.com/synaptent/RingRift/issues/104) | Cheapest onboarding win; data already exists                           |
| 4     | E11: Docs consolidation (three entry doors)            | [#110](https://github.com/synaptent/RingRift/issues/110) | Cheap, multiplies value of E1/E2                                       |
| 5     | E10: Green CI on main                                  | [#109](https://github.com/synaptent/RingRift/issues/109) | Credibility gate for external reviewers                                |
| 6     | E8: Shareable game notation/export                     | [#107](https://github.com/synaptent/RingRift/issues/107) | Enables community content; unblocks E5 sharing                         |
| 7     | E7: Client phase auto-advance + progressive disclosure | [#106](https://github.com/synaptent/RingRift/issues/106) | Onboarding cliff reduction                                             |
| 8     | E6: Post-game analysis / eval bar                      | [#105](https://github.com/synaptent/RingRift/issues/105) | Turns trained net into product feature                                 |
| 9     | E3: Consumer-GPU reproduction + published weights      | [#102](https://github.com/synaptent/RingRift/issues/102) | High value, needs release engineering care                             |
| 10    | E4: Multiplayer methodology write-up                   | [#103](https://github.com/synaptent/RingRift/issues/103) | Valuable but research is still moving                                  |
| 11    | E9: Legacy code quarantine                             | [#108](https://github.com/synaptent/RingRift/issues/108) | High churn; schedule when training lanes are quiet                     |

## Global constraints (apply to every epic)

- Do NOT modify `ai-service/scripts/minimal_alphazero_loop.py`, its support libs,
  `ai-service/config/distributed_hosts.yaml`, or databases under `ai-service/data/`.
- TypeScript engine (`src/shared/engine/`) is the rules source of truth; Python mirrors it.
- No weakening of canonical rules or parity gates to make anything "easier".
- Every epic ships with tests and updated docs.

---

## E1: `ringrift-env` — pip-installable RL environment package

**Goal**: Make RingRift usable as a multi-agent RL benchmark without adopting the whole
ai-service.

**Why**: Deterministic, perfect-information, 2–4 player games with mandatory chain captures and
heavy branching are exactly what multi-agent RL research lacks. A standard env API makes
RingRift citable and adoptable; today the only path is cloning 1.8M LOC of Python.

**Scope**:

- New package (working name `ringrift-env`) exposing the Python rules mirror + board encoder
  behind a PettizingZoo `AECEnv`-style API (`reset`, `step`, `observe`, `legal_moves` /
  action masking, `agents`, terminal rewards by rank).
- Wraps existing `ai-service/app` engine code as a dependency layer — no rules reimplementation.
- Supports all four board types × 2–4 players; observation = existing canonical encoding;
  action space = existing canonical action indexing.
- `pip install -e` path, minimal dependency footprint (numpy + the extracted engine modules).
- README with quickstart, random-agent example, and benchmark table (baseline Elo anchors).
- Unit tests: API contract, determinism (same seed → same trajectory), legal-mask correctness
  against the existing move generator, one full random-playout game per config.

**Acceptance criteria**:

- [ ] `pip install -e packages/ringrift-env && python -c "import ringrift_env"` works in a clean venv
- [ ] Random-vs-random full game completes on all 4 board types, 2/3/4 players
- [ ] Legal-action mask matches engine move enumeration on 100+ sampled states
- [ ] Deterministic replay: identical seeds produce identical trajectories
- [ ] No modification to protected files; parity story documented (env defers to Python mirror,
      which is parity-tested against TS)

**Follow-ups (separate issues later)**: PyPI publication, OpenSpiel/Pgx contribution, Gymnasium
single-agent wrapper vs. fixed opponents.

---

## E2: Silent-AlphaZero-failures research write-up

**Goal**: Publish the silent-failure catalog (`docs/research/SILENT_ALPHAZERO_FAILURES.md`) as a
standalone, publication-ready article: "Silent failure modes in AlphaZero pipelines, and the
gates that catch them."

**Why**: This is the most transferable knowledge in the repo — every failure mode (dead value
heads from FiLM init, device-placement reversion degrading evals to random play, encoding
mismatches causing silent heuristic fallback, silently no-op transfer scripts, …) applies to
anyone training AlphaZero on a custom game. It will draw more researchers to the project than
the game itself.

**Scope**:

- Standalone write-up under `docs/research/` structured for external readers: no repo-internal
  jargon, each failure mode = symptom → root cause → detection gate → fix, with commit refs.
- Generalized "detection gate checklist" section others can apply to their own pipelines.
- Publication-ready framing (abstract, motivation, related-work stub) suitable for a blog post
  or arXiv note; source-of-truth stays in-repo.

**Acceptance criteria**:

- [ ] Standalone article readable with zero RingRift context
- [ ] Every failure mode has: symptom, root cause, detection method, fix, evidence pointer
- [ ] Generalized checklist section (portable to other AlphaZero projects)
- [ ] Linked from docs index and README research section

---

## E3: Consumer-GPU reproduction path + published weights

**Goal**: Anyone can (a) _verify_ the headline Elo claim in minutes with released checkpoints,
and (b) _reproduce_ a rising Elo curve on one consumer GPU over a weekend.

**Scope**:

- Release model checkpoints (hex8_2p fv3 frontier + square8_2p) via GitHub Releases or S3 with
  documented hashes and provenance sidecars.
- One-command verification gauntlet: downloaded checkpoint vs. pinned baselines, reports Elo
  with CI.
- `--small` profile for the proven-experiment script targeting a single RTX-class GPU / Colab:
  reduced sims, smaller net, documented expected curve.
- REPRODUCIBILITY.md section covering both paths.

**Acceptance criteria**:

- [ ] Checkpoint download + verification gauntlet runs on a machine without cluster access
- [ ] Small-profile run shows Elo improvement over ≥3 promotions on consumer hardware
- [ ] Provenance (training config, data lineage, commit) shipped with weights

---

## E4: Multiplayer methodology write-up

**Goal**: Write up 3–4 player AlphaZero evaluation methodology (seat-fair evaluation, baseline
Elo pinning under kingmaking, rank-distribution targets) as a research note — including negative
results.

**Why**: Multiplayer AlphaZero is understudied; the measurement methodology is a contribution
even where model strength is unproven. "Standard value targets regress to the mean under
3-player kingmaking" is a publishable finding.

**Scope**:

- Research note under `docs/research/`: problem statement, seat-fairness protocol, baseline
  pinning rationale, observed failure modes (Elo inflation via kingmaking, false regressions),
  current open questions.
- Data pulled from existing eval DBs/docs only — no new training runs required.

**Acceptance criteria**:

- [ ] Note readable by an RL researcher with no RingRift context
- [ ] Every claim tied to existing evidence (docs/RESULTS.md, eval snapshots)
- [ ] Honest scoping: what is measured, what is unproven, what is next

---

## E5: Puzzle mode mined from self-play data

**Goal**: Daily-tactics-style puzzles mined from self-play games — positions with a single
strongly winning move (forced chain capture, territory seal), served in the client.

**Why**: Cheapest onboarding path. Puzzles teach chain captures and line collapse better than
any rulebook; they are shareable; and the mining pipeline is a script over existing DBs.

**Scope**:

- Phase 1 (mining): Python script scanning replay DBs (read-only) for candidate positions —
  large value swing between best and second-best move under the trained net or search; emits
  a JSON puzzle format (position FEN-equivalent, solution line, theme tag, difficulty).
- Phase 2 (product): client puzzle page rendering a position in the existing sandbox board,
  validating the solution move(s) locally, with reveal/hint.
- Puzzle JSON schema documented; puzzles stored as static assets (no new DB).

**Acceptance criteria**:

- [ ] Miner produces ≥50 validated puzzles from existing hex8/square8 DBs (read-only access)
- [ ] Each puzzle machine-validated: solution move is uniquely best by stated margin
- [ ] Client page: load puzzle, attempt move, correct/incorrect feedback, solution reveal
- [ ] Puzzle schema documented in docs/

---

## E6: Post-game analysis / eval bar

**Goal**: Use the trained value head for an eval bar and post-game "blunder review" (moves that
most dropped the network's evaluation).

**Scope**:

- Server or ai-service endpoint: batch-evaluate a finished game's positions with the canonical
  model; return per-move value trace.
- Client: eval graph on the replay/game-over view; top-3 largest value drops flagged with
  "what the network preferred".
- Ties into the existing UX explanation-model spec (`docs/UX_RULES_EXPLANATION_MODEL_SPEC.md`).

**Acceptance criteria**:

- [ ] Finished game shows a value-over-time graph per player
- [ ] Blunder list with preferred-move display
- [ ] Degrades gracefully when AI service is unavailable

---

## E7: Client phase auto-advance + progressive disclosure

**Goal**: Reduce perceived complexity of the 7-phase turn FSM without touching canonical rules.

**Scope**:

- Client-side auto-advance for phases with exactly one legal action (canonical move still
  recorded underneath — no engine changes).
- "Why is this forced?" tooltip on forced transitions (forced_elimination, mandatory chains).
- Beginner mode toggle on hex8 2p: recovery/pie-rule UI hidden until first relevant.

**Acceptance criteria**:

- [ ] Single-choice phases auto-resolve in UI with visible-but-unobtrusive move log entries
- [ ] Canonical move history byte-identical to manual play (replay parity test)
- [ ] Beginner mode reduces visible controls without changing legality

---

## E8: Shareable game notation + export/import

**Goal**: A compact human-readable notation and export/import so games can be shared, annotated,
and discussed (PGN-equivalent for RingRift).

**Scope**:

- Notation spec (docs/) derived from the canonical explicit move history — every move type,
  including no-op/skip bookkeeping moves (possibly elided in "display" form, exact in
  "canonical" form).
- TS serializer/parser in `src/shared/` with round-trip tests; export button in client replay
  view; import into sandbox.

**Acceptance criteria**:

- [ ] Round-trip: export → import reproduces identical GameState sequence
- [ ] Notation spec documented with examples
- [ ] Works for all board types and player counts

---

## E9: Legacy code quarantine

**Goal**: Make the filesystem match the supported/experimental/historical classification that
`docs/data/ai_surface_manifest.json` already records — dormant P2P/daemon/script code moved
under `ai-service/legacy/` (or archive repo), excluded from default test collection and search
space.

**Scope**:

- Mechanical moves guided by the surface manifest; import-shim layer where live code references
  historical modules; test collection updated; docs updated.
- Explicitly out of scope: deleting anything; touching protected files; changing behavior of
  the supported path.

**Acceptance criteria**:

- [ ] Supported-path tests green after moves
- [ ] Default pytest collection excludes quarantined tree
- [ ] Surface manifest and docs updated to reflect new paths

**Risk note**: highest-churn epic; execute when training lanes are quiet and in small, individually
verified moves.

---

## E10: Green CI on main

**Goal**: `main` is trustworthy at a glance: required workflows green, flaky/informational jobs
explicitly tiered.

**Scope**:

- Audit all 10 GitHub Actions workflows for current status and failure causes.
- Fix cheap breakages; tier the rest (required vs. informational) with documented rationale.
- Badge/docs updated to reflect the tiering.

**Acceptance criteria**:

- [ ] All _required_ workflows green on main (or PR opened fixing them)
- [ ] Each non-required workflow documented: what it covers, why informational
- [ ] No silently-red required checks

---

## E11: Docs consolidation — three entry doors

**Goal**: Restructure documentation entry around three audiences — **player** (rules, tutorial),
**researcher** (results, reproduction, env API), **contributor** (architecture, parity,
invariants) — with everything else reachable only from those doors.

**Scope**:

- Rework `DOCUMENTATION_INDEX.md` (and the README docs section) into the three-door structure.
- Each door: a short curated page linking the 5–10 documents that audience actually needs.
- No mass file moves — this is an index/navigation epic, not a migration.

**Acceptance criteria**:

- [ ] Three door pages exist and are linked from README + DOCUMENTATION_INDEX
- [ ] A cold visitor can reach rules, results, or architecture in ≤2 clicks from README
- [ ] Stale/duplicate index entries pruned
