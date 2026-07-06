# Execution Log — improvement epics run (2026-07-01)

Time budget: open return time; finite mode; scoped to Batches 0–3 then Final Completion.

## Batch plan

| #   | Name                           | Scope                                                       | Est. |
| --- | ------------------------------ | ----------------------------------------------------------- | ---- |
| 0   | Session setup + epic recording | Plan doc, session docs, `epic` label, issues E1–E11, PR     | S    |
| 1   | E2: silent-failures write-up   | Standalone publication-ready article in docs/research/      | M    |
| 2   | E1: ringrift-env MVP           | Package skeleton + AEC-style API over Python engine + tests | L    |
| 3   | E5: puzzle miner MVP           | Read-only DB miner + puzzle JSON schema + validation        | L    |
| —   | Scout / Final Completion       | TODO pass, report, cleanup                                  | S    |

## Entries

### Batch 0: Session setup + epic recording (in progress)

Started 2026-07-01.
**Contract** (trivial batch): plan doc + session docs committed; `epic` label exists; 11 issues
created, each linking the plan section; PR open with batch list. Acceptance: `gh issue list
--label epic` shows 11; PR number recorded in `.elves-session.json`.

- Wrote `docs/planning/PROJECT_IMPROVEMENT_EPICS_2026-07.md` (11 epics, priority order,
  global constraints).
- Wrote survival guide, learnings, this log.
- Next: label + issues + commit + push + PR.

### Batch 1: E2 silent-failures write-up (complete)

**Contract**: publication-ready article; evidence index; portable checklist; linked from README +
DOCUMENTATION_INDEX. All acceptance criteria met.
**Pre-implementation survey**: found existing SILENT_ALPHAZERO_FAILURES_BLOG_DRAFT.md (~2,500
words, all 8 bugs) — extended it instead of writing new (git mv → _ARTICLE.md). Catalog already
linked from README:145.
**Done**: abstract added; Appendix A portable detection-gate checklist (5 contract layers);
evidence index table (8 commits verified present via git cat-file, 4 test paths verified on
disk); README + DOCUMENTATION_INDEX links; an0mium→synaptent commit URL fixes.
**Validation**: all 7 referenced commits exist; all linked files exist; prettier clean.
**Regression attestation**: docs-only batch; no code or shared surfaces touched; test baseline
unaffected. Confidence HIGH — changes are additive markdown.
**Commit**: 4da67a06a. Rollback tag: elves/pre-batch-1.
**Also**: noted existing public-model-artifacts-2026-04-28 release on issue #102 (E3 partially
prefigured); progress comment on #101.

### Batch 2: E1 ringrift-env MVP (complete)

**Contract**: pip-installable PettingZoo AEC-style env over the canonical Python engine; all 4
boards × 2/3/4 players; exact action masks; deterministic replay; torch-free install.
**Pre-implementation survey**: found existing gym-like `RingRiftEnv` in app/training/env.py
(canonical bookkeeping synthesis, termination, rewards) — wrapped it instead of reimplementing.
Canonical action indexing via app/ai/canonical_move_encoding.py.
**Upstream enablers (behavior-preserving, verified)**:

- app/training/env.py: seed_all import made optional (guarded try/except; module attribute
  preserved for test patching; seeded reset raises informative ImportError without torch).
- app/ai/neural_net/**init**.py: PEP 562 lazy exports for torch-dependent symbols; torch-free
  encoding surface stays eager; canonical_move_encoding re-exports lazy (breaks a latent
  circular import). Repo precedent: lazy-imports-to-avoid-heavy-startup (ruff E402 note).
  **Finding**: canonical encoding is intentionally non-injective for choice moves (hex maps ALL
  special/choice moves to one sentinel; square collapses CHOOSE_LINE_OPTION variants). Env adds a
  deterministic per-state overflow block (default 256 slots) after canonical indices.
  **Validation**:
- Clean venv (numpy/pydantic/psutil, NO torch): install + import OK; fast suite 13 passed;
  slow suite (full-length square19/hexagonal 2/3/4 + hex8/square8 2p) 8 passed in 45s.
- With torch: tests/unit/ai 1699 passed 6 skipped; test_training_env.py 53 passed;
  test_neural_net_architectures.py 54 passed; test_minimal_alphazero_loop.py 30 passed.
  **Regression attestation**: shared surfaces modified: app/ai/neural_net/**init**.py (imported
  repo-wide; all **all** symbols verified resolvable with torch; full unit/ai slice green) and
  app/training/env.py (53/53 module tests + 30/30 minimal-loop tests green). Test totals only
  increased (21 new package tests). Confidence HIGH.

### Batch 3: E5 puzzle miner MVP (complete)

**Contract**: read-only chain-capture puzzle miner over replay DBs; JSON schema documented;

> =50 machine-validated puzzles; self-validation + CLI validate mode.
> **Done**: `ai-service/scripts/mine_chain_capture_puzzles.py` (structural chain-score metric,
> no NN required; --copy-to-temp so live DBs are never opened read-write); schema doc
> `docs/puzzles/PUZZLE_FORMAT.md`; asset `src/client/public/puzzles/hex8_2p_chain_capture.json`
> (60 puzzles, min-chain 3, min-margin 2, mined from a scratchpad copy of canonical_hex8_2p.db).
> **Validation**: 4/4 new unit tests pass (in-process random games, no DB dependency);
> `--validate` reports 60/60 valid; every emitted puzzle self-validated at mine time.
> **Regression attestation**: additive only (new script, new tests, new docs, new static asset);
> no shared surfaces touched; no databases modified (mined from a copy). Confidence HIGH.
> **Note**: client puzzle page is E5 Phase 2, tracked on #104.
