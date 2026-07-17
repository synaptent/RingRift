# READ THIS FILE FIRST AFTER ANY COMPACTION OR RESTART

# RingRift E10 CI Trust Survival Guide

## Mission

Deliver the small E10 prerequisite PR from exact `origin/main`: repair CI reporting, classify all
workflow policy, refresh the supported reviewer surface, and restore credible Node/Python
dependency audits. Leave the branch review-ready and unmerged. Do not begin PR #111 integration,
the documentation doors, or the puzzle UI in this run.

## Run Control

- **Run mode:** finite
- **Stop policy:** blocker-only
- **User intent:** "PLEASE IMPLEMENT THIS PLAN" followed by "proceed"
- **Checkpoint due by:** none
- **Checkpoint semantics:** none
- **May continue after checkpoint:** yes
- **Actual stop conditions:** all three E10 batches complete and readiness is clean, explicit user
  stop, or a genuine blocker with no safe workaround
- **Workspace ownership:** dedicated worktree at
  `/Users/armand/.codex/worktrees/ringrift-e10-ci-trust/RingRift`
- **Branch tip at start (collision tripwire):**
  `a243b4e5ad1e7359052361a15a6be64c978d2746`
- **Merge policy:** user-merges; the agent never merges
- **Final-response policy:** disallowed until the Stop Gate permits it
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
- **Average batch time so far:** approximately 32 minutes
- **Batches remaining:** 1 of 3

## Stop Gate

- **Planned batches remaining:** 1
- **Stop allowed right now:** no
- **Why:** Batch 2 implementation and review are complete, but it still must be committed and
  pushed before the final readiness batch begins
- **Next required action:** commit and push Batch 2, re-read this guide, inspect live PR state,
  then execute Batch 3 full validation, report, and run-scaffolding cleanup

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
- [x] Draft PR #112 opened and recorded
- [x] Preflight run and known baseline failures recorded
- [x] Run mode, stop behavior, merge policy, and non-negotiables recorded
- [x] Stop Gate explicitly permits the required staging handoff
- [x] Launch prompt prepared for the next call

## Current Phase

**Status:** In progress

**Active batch:** Batch 2: Dependency audit credibility (completion boundary)

**What was just finished:** compatible Node/Python upgrades, the fail-closed Python audit wrapper
and exception ledger, clean-environment runtime/contract proof, and clean final independent review

**Single next action:** commit and push Batch 2, then immediately re-read this guide

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
- E10 draft PR #112: `https://github.com/synaptent/RingRift/pull/112`; initial Batch 0 commit
  `0c2600030cbb865f42976500711d02f43b96916c` (always verify the live head before writing)
- PR #111: open at `f74e0cff0cd891ac08cf79ed921ada7477315939`, currently blocked
- Known PR #111 failures: stale reviewer manifest, broken lcov comment action, Node production
  audit, and Python dependency audit
- Baseline validation: lint passes with 11 existing warnings; typecheck and build pass; all 587
  executed Jest suites pass (11,777 tests, 200 skipped, 1 todo); standalone `test:coverage` exits 1
  on the main-branch 80% thresholds while the CI command explicitly overrides global thresholds
  to zero
- Python contracts baseline: 4,960 pass and one supported-doc-link test fails because
  `docs/RESULTS.md` points to missing `docs/research/QUALITY_GATE_RESUME_BUG.md`
- Batch 1 repaired the supported-doc link, re-attested reviewer and AI surface manifests, added a
  fail-closed workflow policy registry, and passed its final independent review.
- Batch 1 commit `72206400acdc083921ea49719315ac7368058cdb` is pushed and matches PR #112's
  live head; no PR comments or reviews were present at the Batch 2 boundary.
- The full supported-path command now fails only on the pre-existing May 12 public result snapshot
  being 66 days old against the 30-day evidence rule. No current local metrics are available, so
  the run must not redate the evidence or weaken the gate.
- Batch 2 now has a zero-finding Node production audit and a Python audit that reports one
  unfixable `ecdsa` finding covered by an exception approved 2026-07-17 and expiring 2026-08-31.
  Dedicated issue #113 tracks removal before expiry.
- Batch 2 clean-environment proof includes 5,004 Python contracts, 170 focused FastAPI/Torch tests,
  61 Python contract-vector tests, 43 parity-healthcheck cases with zero mismatches, lint,
  typecheck, build, focused Node runtime suites, and clean independent Node/Python/workflow
  re-reviews.
- GitHub auth: active `scarmani` can read but push is denied with 403; configured secondary
  `an0mium` has `ADMIN`. Use a scoped account switch for push/PR operations and restore
  `scarmani` immediately afterward
- Required goal-cycle advisory conflicted with the approved E10-first plan by proposing direct
  PR #111 integration and no workflow edits; the user-approved plan controls
- Elves update doctor reported v2.6.0 available while the local skill is v1.12.0; do not update
  tooling during this scoped run

## Next Exact Batch

**Batch:** 3: E10 final readiness

**Scope:**

- Run the complete Node, Python, parity, supported-path, clean-environment, puzzle-asset, and
  Docker-backed CI validation surface in proportion to repository risk.
- Inspect and remediate live PR checks and review feedback without weakening the truthful stale
  result-evidence gate.
- Generate the finite-run HTML report, remove `.elves-session.json` and `docs/elves/**` from the
  final PR surface, and leave PR #112 review-ready or documented at the genuine external blocker.

**Acceptance criteria:**

- [ ] All intended E10 checks pass except any independently verified external-evidence blocker.
- [ ] Final independent readiness review is clean and live PR feedback is resolved.
- [ ] Elves report exists outside the repository and run scaffolding is absent from the PR diff.

**Risk:** the required supported-path gate may remain red because current public result evidence is
not available locally; the run must report this honestly rather than fabricate freshness.

**Rollback tag:** `elves/e10-ci-trust/pre-batch-3` (create and push at the Batch 2 head after the
required commit/push/re-read boundary)

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

1. Create and push `elves/e10-ci-trust/pre-batch-N` before each implementation batch; generic
   `elves/pre-batch-N` tags already belong to older runs.
2. Never force-push or rebase.
3. Stage explicit files only; never use blanket `git add -A`.
4. Recover from the last good rollback tag on a new recovery branch rather than rewriting history.
5. The user merges; final readiness means review-ready, not merged.

## Plan and Log Paths

- **Plan:** `docs/planning/RINGRIFT_THREE_WAVE_QUALITY_UTILITY_PLAN_2026-07.md`
- **Learnings:** `docs/elves/quality-utility-2026-07/learnings.md`
- **Execution log:** `docs/elves/quality-utility-2026-07/execution-log.md`
- **Branch:** `codex/e10-ci-trust`
- **PR number:** #112
- **Plan hash after staging formatter:** `cc190eb242d71897caaec0717ddd1cc5`

## After Any Compaction

Read, in order: this file, `.elves-session.json`, learnings, the plan, the execution log, live git
state, then PR comments/checks. Confirm the collision tripwire, Stop Gate, and single next action
before touching code.
