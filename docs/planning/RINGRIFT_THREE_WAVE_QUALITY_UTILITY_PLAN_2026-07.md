# RingRift Three-Wave Quality and Utility Plan

## Mission

Improve repository trust and user utility in three ordered waves: make CI credible and PR #111
review-ready, give players/researchers/contributors concise entry paths, and expose the existing
puzzle dataset as a public playable experience. Completion means each wave is independently
reviewable, its specified gates pass, and no canonical rules, parity contracts, training-loop
logic, live databases, or unrelated CMA-ES work has changed.

## Delivery topology

This program crosses multiple merge gates and therefore must not be delivered as one branch or
one oversized pull request.

1. **Run A (current): E10 prerequisite PR.** Branch from the exact `origin/main` head in an
   isolated worktree. Deliver only CI policy, reviewer-manifest, and dependency-audit changes.
2. **Gate A:** the user reviews and merges the E10 prerequisite. The agent never merges by
   default.
3. **Run B: PR #111 integration.** In a new isolated worktree at the exact remote feature head,
   merge current `main` into `feat/improvement-epics` without rebasing, remove final-session
   scaffolding, re-attest the supported surface, and make #111 merge-ready.
4. **Gate B:** merge #111 only with explicit authority. After a verified merge, close completed
   issues #100 and #101 and mark Phase 1 complete on #104.
5. **Run C: E11 documentation doors.** Start from then-current `origin/main`, deliver the three
   audience entry pages and link-contract changes, and close #110 only after merge.
6. **Run D: E5 puzzle experience.** Start from then-current `origin/main`, deliver the public
   puzzle routes, strict typed loader, canonical grading, UI, and tests. Close #104 only after
   merge.

The current Elves session covers Run A only. Later runs are intentionally gated on prior merges so
they can start from live remote truth and avoid branch collisions or speculative integration.

## Scope

### In scope

- CI workflow reliability, workflow classification policy, reviewer-surface freshness, and
  supported-path validation.
- Compatible production dependency upgrades and an expiring, machine-readable Python audit
  exception mechanism.
- PR #111 integration and supported-surface attestation after the prerequisite lands.
- Three curated documentation entry pages and their link/discoverability contracts.
- A public, client-only puzzle browser/player over the checked-in puzzle bundle.
- Focused tests, cumulative validation, and issue updates after verified merges.

### Out of scope

- Canonical rule semantics, phase/move contracts, parity behavior, or capture-rule duplication.
- Protected training-loop logic, training data, model artifacts, live replay databases, or public
  result claims.
- Unrelated CMA-ES experiments or dirty shared-checkout files.
- E8 notation/export, deployment, authentication, or backend persistence for puzzles.
- Rebases, force pushes, squash merges, or agent-initiated merges without explicit authority.

## Run A batches: E10 prerequisite

### Batch 1: Workflow and reviewer policy

**Tasks:**

- [ ] Remove the failing `romeovs/lcov-reporter-action` coverage-comment step while retaining the
      full coverage test as gating and Codecov as non-gating reporting.
- [ ] Refresh `docs/data/reviewer_surface_manifest.json` from live supported surfaces without
      changing the 45-day freshness rule.
- [ ] Add a machine-readable workflow-policy registry classifying every workflow as `required`,
      `scheduled`, or `informational`.
- [ ] Extend `scripts/check_github_workflows.py` and focused tests so missing, extra, or invalid
      workflow classifications fail closed.

**Acceptance criteria:**

- [ ] `python3 scripts/check_github_workflows.py` passes and accounts for every workflow YAML file.
- [ ] Reviewer-surface validation passes with the unchanged 45-day maximum age.
- [ ] The CI workflow still runs full Jest coverage and retains a non-gating Codecov upload.
- [ ] Focused workflow-policy and supported-path tests pass.

**Docs likely touched:** workflow registry, reviewer manifest, this plan, run learnings if needed.

**Risk:** GitHub branch-protection semantics are external; the registry must describe intent
without pretending to configure required checks.

### Batch 2: Dependency audit credibility

**Tasks:**

- [ ] Apply compatible Node production dependency updates until
      `npm audit --omit=dev --audit-level=high` passes.
- [ ] Apply compatible Python patch/minor upgrades first and validate relevant Python contracts.
- [ ] Add a Python audit wrapper and machine-readable exception ledger under `docs/security/`.
- [ ] Require each exception to record advisory ID, package, rationale, tracking issue, approval
      date, and an expiry no later than 45 days; fail on unknown, malformed, mismatched, or expired
      exceptions.
- [ ] Wire the wrapper into CI and add fixtures covering clean, accepted, unknown, and expired
      findings.

**Acceptance criteria:**

- [ ] `npm audit --omit=dev --audit-level=high` exits zero.
- [ ] Python audit-wrapper fixtures pass, including unknown and expired failure cases.
- [ ] Direct `pip-audit` output is never suppressed inline; any unavoidable finding is represented
      only by a valid, expiring ledger entry.
- [ ] Compatible upgrades do not break focused Node/Python tests, lint, type checking, or build.

**Docs likely touched:** `docs/security/` policy and ledger, dependency manifests/locks.

**Risk:** Some Python advisories may have no compatible fix; exceptions must remain narrow,
traceable, and temporary rather than weakening the gate.

### Batch 3: E10 final readiness

**Tasks:**

- [ ] Run focused policy/audit tests, full Jest coverage, lint, TypeScript checks, build, Python
      unit/contracts, `ringrift-env` clean-environment checks where present, and puzzle asset
      validation where present on this base.
- [ ] Read every PR comment and check, resolve blocking findings, and rerun affected gates.
- [ ] Perform a fresh cumulative diff and regression review.
- [ ] Generate the Elves report, then remove `.elves-session.json` and disposable
      `docs/elves/**` session scaffolding from the final PR diff.

**Acceptance criteria:**

- [ ] Every intended required check is green; scheduled/informational failures remain visible and
      tracked rather than silently ignored.
- [ ] No files outside E10, its tests/docs, and the retained program plan are changed.
- [ ] The PR is review-ready and left unmerged for the user.

**Docs likely touched:** execution evidence during the run; only durable E10 docs remain at final
completion.

**Risk:** Full suites are broad and may expose unrelated baseline failures; classify them with
exact evidence and never hide a real regression.

## Later gated batches

### Batch 4: PR #111 integration and cleanup

- Merge current `main` into `feat/improvement-epics` without rebasing from an isolated exact-head
  worktree.
- Remove `.elves-session.json` and `docs/elves/**` from #111 as final-session scaffolding.
- Re-attest the reviewer manifest, including `packages/ringrift-env`, the publication article,
  and puzzle tooling.
- Run all PR checks and make #111 merge-ready; merge only with explicit authority.
- After a verified merge, close #100 and #101 and mark Phase 1 complete on #104.

### Batch 5: E11 three audience doors

- Add `docs/start-here/PLAYER.md`, `RESEARCHER.md`, and `CONTRIBUTOR.md` with 5–10 curated links
  per page.
- Reduce `DOCUMENTATION_INDEX.md` to those doors plus a maintainer/reference section without
  moving underlying documents.
- Link all doors from README and `docs/INDEX.md` and extend supported-doc link contracts and
  workflow path filters.
- Verify every audience reaches its primary destination in at most two clicks from README; close
  #110 after merge.

### Batch 6: E5 playable puzzle experience

- Add lazy-loaded public routes `/puzzles` and `/puzzles/:puzzleId` with no auth or backend.
- Add `PuzzleBundleV1`, `PuzzleV1`, and `PuzzleSolutionV1` plus a strict loader that rejects
  unsupported versions and malformed states.
- Render stored state through `BoardView` and `toBoardViewModel`; enumerate moves only through the
  canonical shared `getValidMoves`.
- Grade the normalized selected move against `solution.moves[0]`; support feedback, reset,
  reveal, next/previous, direct links, complete chain overlays, and graceful invalid states.
- Add parser/grader unit tests, component tests, and desktop/mobile Playwright smoke coverage.
- Link the experience from README and the player door; close #104 after merge. Make E8
  notation/export the next planned utility epic.

## Non-negotiables

- Preserve the dirty shared checkout; all writes occur in dedicated isolated worktrees from exact
  remote heads.
- Do not change canonical rules, parity contracts, protected training logic, live databases,
  public result claims, or unrelated CMA-ES work.
- Never weaken tests, the reviewer-manifest 45-day freshness rule, or an audit gate to obtain a
  green result.
- Never duplicate game/capture rules in UI code; puzzle legality comes from the canonical shared
  engine.
- The agent never merges by default. The user reviews and merges unless they later grant explicit
  merge-on-green authority; any authorized merge is a regular merge commit, never a squash.

## Test strategy

- **Run A policy gates:** `python3 scripts/check_github_workflows.py`, reviewer-surface validation,
  supported-path validation, Node audit, and Python audit-wrapper fixtures.
- **Run A regression gates:** full Jest coverage, lint, TypeScript/build, relevant Python
  unit/contracts, `ringrift-env` clean-environment tests, and puzzle asset validation where those
  surfaces exist on the branch.
- **Run C docs gates:** supported-doc link contracts covering README, both indexes, and all three
  doors; discoverability and 5–10-link budget assertions.
- **Run D puzzle gates:** strict loader/grader unit tests, component interaction tests, and a
  Playwright smoke solving a known puzzle at desktop and mobile widths.
- **Integrity:** test totals never decrease, skipped counts never increase without explicit
  evidence, and every batch ends with a cumulative diff/consumer review.

## Advisory disposition

The required bounded Fable goal cycle recommended integrating `main` directly into PR #111 and
avoiding workflow edits. That advice conflicts with the user-approved E10-first sequence and the
explicit workflow-policy requirements above. It is recorded as advisory input, not authority; the
approved plan and live gate failures control execution.
