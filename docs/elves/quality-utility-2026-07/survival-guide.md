# READ THIS FILE FIRST AFTER ANY COMPACTION OR RESTART

# RingRift E10 CI Trust Survival Guide

## Mission

Deliver the small E10 prerequisite PR from exact `origin/main`: repair CI reporting, classify all
workflow policy, refresh the supported reviewer surface, and restore credible Node/Python
dependency audits. Leave the branch review-ready and unmerged. Do not begin PR #111 integration,
the documentation doors, or the puzzle UI in this run.

## Run Control

- **Run mode:** finite
- **Stop policy:** blocker-only after launch; staging handoff is a required stop boundary
- **User intent:** "PLEASE IMPLEMENT THIS PLAN" followed by "proceed"
- **Checkpoint due by:** none
- **Checkpoint semantics:** none
- **May continue after checkpoint:** yes
- **Actual stop conditions:** staging handoff before launch; after launch, all three E10 batches
  complete, explicit user stop, or a genuine blocker with no safe workaround
- **Workspace ownership:** dedicated worktree at
  `/Users/armand/.codex/worktrees/ringrift-e10-ci-trust/RingRift`
- **Branch tip at start (collision tripwire):**
  `a243b4e5ad1e7359052361a15a6be64c978d2746`
- **Merge policy:** user-merges; the agent never merges
- **Final-response policy:** allowed during staging handoff; disallowed after launch until the Stop
  Gate permits it
- **Batch completion rule:** update execution log, update this guide, commit, and push before the
  next batch
- **Re-read rule:** immediately after every commit and push, re-read this guide before any other
  action
- **Continuation rule:** after launch, if work remains and no actual stop condition is met,
  continue without waiting for acknowledgment

## Session Budget

- **Started:** 2026-07-17 10:36 CDT
- **User returns:** unknown
- **Checkpoint expectation:** no checkpoint; produce a review-ready E10 prerequisite PR
- **Time budget:** approximately 8 hours after launch; extend only through safe completion of the
  active atomic validation/fix cycle
- **Average batch time so far:** not started
- **Batches remaining:** 3 of 3

## Stop Gate

- **Planned batches remaining:** 3
- **Stop allowed right now:** yes
- **Why:** Elves requires staging and unattended execution to be separate calls; this call stops
  only after launch readiness and a paste-ready launch prompt
- **Next required action:** on a fresh launch call, set this gate to `no`, create
  `elves/pre-batch-1`, and start Batch 1

## Effort Standard

- Work through the full E10 run after launch; a green sub-check or clean commit is not completion.
- Keep changes small, reviewable, and evidence-backed.
- Continue from one completed batch directly into the next after the required commit/push/re-read.

## Forbidden Stop Reasons After Launch

- A commit or push succeeded.
- CI is green for only part of the branch.
- The draft PR exists.
- The user is silent.
- The current batch is complete while a later E10 batch remains.
- A dependency fix requires careful investigation but still has safe, bounded options.

## Memory Surfaces

- **Plan:** `docs/planning/RINGRIFT_THREE_WAVE_QUALITY_UTILITY_PLAN_2026-07.md`
- **Session state:** `.elves-session.json`
- **Survival guide:** `docs/elves/quality-utility-2026-07/survival-guide.md`
- **Learnings:** `docs/elves/quality-utility-2026-07/learnings.md`
- **Execution log:** `docs/elves/quality-utility-2026-07/execution-log.md`
- **Curated `.ai-docs`:** none on this base

## Non-Negotiables

- Preserve the dirty shared checkout at `/Users/armand/Development/RingRift`; never stage, stash,
  clean, or edit its dirty paths.
- Do not change canonical rules, parity contracts, protected training logic, live databases,
  public result claims, or unrelated CMA-ES work.
- Do not weaken tests, the 45-day reviewer-manifest rule, or audit severity thresholds.
- Never rebase, force-push, squash, or merge; use a regular PR for user review.
- Never alter a test merely to make it pass. Add focused tests for new policy and wrapper behavior.
- If the branch tip moves to a commit not created by this run, stop and report the collision.

## Launch Readiness

- [x] Plan cleaned and saved to disk
- [x] Survival guide written from the approved plan
- [x] Learnings file initialized
- [x] Execution log initialized with batch breakdown
- [x] Branch created from exact `origin/main`
- [x] Dedicated worktree ownership confirmed
- [ ] Draft PR opened and recorded
- [x] Preflight run and known baseline failures recorded
- [x] Run mode, stop behavior, merge policy, and non-negotiables recorded
- [x] Stop Gate explicitly permits the required staging handoff
- [x] Launch prompt prepared for the next call

## Current Phase

**Status:** Staging

**Active batch:** none; Batch 1 is queued

**What was just finished:** preflight captured the green build/tooling baseline and the exact
expected audit, freshness, coverage-threshold, and supported-doc-link failures

**Single next action:** finish preflight, commit/push Batch 0, open the draft PR, record its number,
and hand the user the launch prompt

## Active Compute

**No active paid or long-running compute.** A local `caffeinate` process is available to prevent
sleep but performs no project work by itself.

## Dirty Shared Checkout: No-Touch Inventory

The shared checkout was dirty before this run. Preserve these paths exactly as found:

- `.elves-session.json`
- `ai-service/archive/deprecated_ai/_game_engine_legacy.py`
- `ai-service/scripts/export_replay_dataset.py`
- `.security/`
- `ai-service/scripts/experiments/cmaes_nnue_shadow.py`
- `docs/planning/CMAES_ONLINE_NNUE_ADAPTATION_PLAN.md`

## Current Live Evidence

- `origin/main`: `a243b4e5ad1e7359052361a15a6be64c978d2746`
- PR #111: open at `f74e0cff0cd891ac08cf79ed921ada7477315939`, currently blocked
- Known PR #111 failures: stale reviewer manifest, broken lcov comment action, Node production
  audit, and Python dependency audit
- Baseline validation: lint passes with 11 existing warnings; typecheck and build pass; all 587
  executed Jest suites pass (11,777 tests, 200 skipped, 1 todo); standalone `test:coverage` exits 1
  on the main-branch 80% thresholds while the CI command explicitly overrides global thresholds
  to zero
- Python contracts baseline: 4,960 pass and one supported-doc-link test fails because
  `docs/RESULTS.md` points to missing `docs/research/QUALITY_GATE_RESUME_BUG.md`
- GitHub auth: active `scarmani` can read but push is denied with 403; configured secondary
  `an0mium` has `ADMIN`. Use a scoped account switch for push/PR operations and restore
  `scarmani` immediately afterward
- Required goal-cycle advisory conflicted with the approved E10-first plan by proposing direct
  PR #111 integration and no workflow edits; the user-approved plan controls
- Elves update doctor reported v2.6.0 available while the local skill is v1.12.0; do not update
  tooling during this scoped run

## Next Exact Batch

**Batch:** 1: Workflow and reviewer policy

**Scope:**

- Remove only the failing lcov comment action while preserving coverage and Codecov semantics.
- Refresh the reviewer manifest without changing the 45-day validator constant.
- Add the workflow-policy registry and fail-closed validator/tests.

**Acceptance criteria:**

- [ ] Every workflow YAML is classified exactly once as required, scheduled, or informational.
- [ ] Workflow and reviewer-surface validators pass.
- [ ] Coverage remains gating and Codecov remains non-gating.

**Risk:** confusing descriptive workflow policy with externally configured branch protection.

**Rollback tag:** `elves/pre-batch-1` (create before implementation starts)

## Post-Checkpoint Control Loop

After every completed batch: commit, push, re-read this guide, inspect new PR comments/checks,
update `.elves-session.json`, and immediately start the next required batch if the Stop Gate says
`no`.

## Documentation Triggers

- Workflow behavior changes require registry/README-level documentation if the registry is not
  self-explanatory.
- Audit behavior changes require a human-readable `docs/security/` policy beside the ledger.
- Reusable validation traps belong in `learnings.md`; stable repo conventions may be promoted to
  AGENTS or `.ai-docs` only if that is clearly warranted.

## Elves Report

- Generate a substantial finite-run report at
  `/tmp/elves-report-ringrift-e10-ci-trust-2026-07-17.html` before operational-artifact cleanup.
- Use the plan, session JSON, survival guide, learnings, execution log, and live PR/CI state.
- Do not commit the HTML report.

## Tool Configuration

```yaml
lint: npm run lint
typecheck: npx tsc --noEmit
build: npm run build
test: npm run test:coverage
python-focused: cd ai-service && PYTHONPATH=. python3 -m pytest tests/contracts -q
workflow-policy: python3 scripts/check_github_workflows.py
reviewer-surface: python3 scripts/check_reviewer_surface.py
supported-path: bash scripts/check_supported_path.sh
node-audit: npm audit --omit=dev --audit-level=high
review: github-pr-comments
notification: pr-comment
```

## Rollback and Safety Rules

1. Create and push `elves/pre-batch-N` before each implementation batch.
2. Never force-push or rebase.
3. Stage explicit files only; never use blanket `git add -A`.
4. Recover from the last good rollback tag on a new recovery branch rather than rewriting history.
5. The user merges; final readiness means review-ready, not merged.

## Plan and Log Paths

- **Plan:** `docs/planning/RINGRIFT_THREE_WAVE_QUALITY_UTILITY_PLAN_2026-07.md`
- **Learnings:** `docs/elves/quality-utility-2026-07/learnings.md`
- **Execution log:** `docs/elves/quality-utility-2026-07/execution-log.md`
- **Branch:** `codex/e10-ci-trust`
- **PR number:** not created yet
- **Plan hash at session start:** `1af49204f730bee8d233d935b6e2e887`

## After Any Compaction

Read, in order: this file, `.elves-session.json`, learnings, the plan, the execution log, live git
state, then PR comments/checks. Confirm the collision tripwire, Stop Gate, and single next action
before touching code.
