# Contributing to RingRift

Thank you for your interest in contributing to RingRift! This document provides guidelines and priorities for development work.

**Related Documents (single source of truth):**

- [CURRENT_STATE_ASSESSMENT.md](./CURRENT_STATE_ASSESSMENT.md) - Factual, code-verified status snapshot
- [TODO.md](./TODO.md) - Task tracking and detailed implementation checklist
- [KNOWN_ISSUES.md](./KNOWN_ISSUES.md) - Specific bugs and issues
- [ARCHITECTURE_ASSESSMENT.md](./ARCHITECTURE_ASSESSMENT.md) - Architecture and refactoring axes
- [STRATEGIC_ROADMAP.md](./STRATEGIC_ROADMAP.md) - Phased strategic plan and milestones
- [archive/PLAYABLE_GAME_IMPLEMENTATION_PLAN.md](./archive/PLAYABLE_GAME_IMPLEMENTATION_PLAN.md) - Historical end-to-end playable lifecycle plan (lobby → game → AI → victory); for current, code-verified status and tasks, defer to `CURRENT_STATE_ASSESSMENT.md`, `ARCHITECTURE_ASSESSMENT.md`, `KNOWN_ISSUES.md`, and `TODO.md`.
- [archive/CODEBASE_EVALUATION.md](./archive/CODEBASE_EVALUATION.md) - Historical code-level evaluation (superseded by `ARCHITECTURE_ASSESSMENT.md` and `CURRENT_STATE_ASSESSMENT.md`)

---

## 🚦 Development Status

**Current State:** Stable beta with consolidated rules engine – **orchestrator at 100% rollout**
**Test Status:** 2621 tests passing, comprehensive coverage across all rule domains
**Priority Focus:** Production hardening, UX polish, and AI improvements

Before contributing, please review:

1. **[TODO.md](./TODO.md)** – Active task tracker (canonical priorities)
2. **[CURRENT_STATE_ASSESSMENT.md](./CURRENT_STATE_ASSESSMENT.md)** – Verified implementation status
3. **[ringrift_complete_rules.md](./ringrift_complete_rules.md)** – Comprehensive game rules
4. **[docs/ORCHESTRATOR_ROLLOUT_PLAN.md](./docs/ORCHESTRATOR_ROLLOUT_PLAN.md)** – Production rollout status

---

## 🎯 Development Priorities

**Current priorities are tracked in [TODO.md](./TODO.md).** Key areas include:

### Active Work (see TODO.md for details)

| Area | Status | Priority |
|------|--------|----------|
| **Orchestrator Rollout** | ✅ Complete (100% in all environments) | P0 |
| **Rules Parity** | ✅ All tests passing | P0 |
| **Backend/Sandbox Hosts** | ✅ Using TurnEngineAdapter/SandboxOrchestratorAdapter | P0 |
| **Legacy Cleanup** | In progress (Wave 5.4) | P1 |
| **UX Polish** | Ongoing | P1 |
| **AI Improvements** | Backlog | P2 |

### Architecture Overview

The rules engine is consolidated under a single canonical stack:

```
src/shared/engine/
├── orchestration/          # Turn orchestrator (processTurn, processTurnAsync)
├── aggregates/             # Domain aggregates (Movement, Capture, Line, Territory, Victory)
├── contracts/              # Contract schemas and serialization
└── [helpers]               # Shared logic (captureLogic, lineDetection, etc.)

Hosts:
├── src/server/game/turn/TurnEngineAdapter.ts      # Backend adapter
└── src/client/sandbox/SandboxOrchestratorAdapter.ts  # Sandbox adapter
```

For architectural details, see:
- `RULES_ENGINE_ARCHITECTURE.md` – Architecture overview
- `docs/RULES_SSOT_MAP.md` – SSOT boundaries and host integration
- `docs/CANONICAL_ENGINE_API.md` – Lifecycle and API reference

---

<details>
<summary>📜 Historical Development Phases (November 2025 - Completed)</summary>

### **PHASE 1: Core Game Logic** (COMPLETED)

**Goal:** Make the game engine correctly implement RingRift rules

**Status:** ✅ All core rules implemented in shared TypeScript engine

**Key completed tasks:**

#### 1.1 Fix BoardState Data Structure (CRITICAL)

**Estimated Time:** 1-2 days  
**Files:** `src/shared/types/game.ts`, `src/server/game/BoardManager.ts`

- [ ] Update `BoardState` interface to separately track stacks, markers, and collapsed spaces
- [ ] Add collapsed space representation (Map<string, number>)
- [ ] Update all methods in BoardManager to use new structure
- [ ] Ensure backward compatibility with existing code

**Why First:** All other fixes depend on proper state management

#### 1.2 Implement Marker System (CRITICAL)

**Estimated Time:** 2-3 days  
**Files:** `src/server/game/GameEngine.ts`, `src/server/game/BoardManager.ts`

- [ ] Add marker placement when rings move
- [ ] Implement marker flipping (opponent markers → your color)
- [ ] Implement marker collapsing (your markers → claimed Territory)
- [ ] Add same-color marker removal on landing
- [ ] Write tests for marker interactions

**Reference:** Section 8.3 of rules document

#### 1.3 Fix Movement Validation (CRITICAL)

**Estimated Time:** 2-3 days
**Files:** `src/server/game/RuleEngine.ts`

- [ ] Enforce minimum distance = stack height
- [ ] Implement landing on any valid space beyond markers
- [ ] Add path validation (no collapsed spaces, no rings)
- [ ] Validate marker interactions during movement
- [ ] Write tests for all movement scenarios

**Reference:** Section 8.2, FAQ Q2

#### 1.4 Fix Game Phase Transitions (HIGH)

**Estimated Time:** 1-2 days
**Files:** `src/shared/types/game.ts`, `src/server/game/GameEngine.ts`

- [ ] Remove `main_game` phase, add `line_processing` phase
- [ ] Implement correct phase flow
- [ ] Update `advanceGame()` method
- [ ] Ensure phases match rules exactly

**Reference:** Section 4, Section 15.2

#### 1.5 Complete Capture System (HIGH)

**Estimated Time:** 3-4 days
**Files:** `src/server/game/GameEngine.ts`, `src/server/game/RuleEngine.ts`

- [ ] Distinguish overtaking vs elimination captures
- [ ] Implement chain capture enforcement (mandatory once started)
- [ ] Add flexible landing during captures
- [ ] Fix cap height comparison logic
- [ ] Implement proper stack merging
- [ ] Write tests for capture sequences

**Reference:** Sections 9-10

#### 1.6 Implement Line Formation (HIGH)

**Estimated Time:** 3-4 days
**Files:** `src/server/game/GameEngine.ts`, `src/server/game/BoardManager.ts`

- [ ] Fix line detection (4+ for 8x8, 5+ for 19x19/hex)
- [ ] Implement graduated rewards (Option 1 vs Option 2)
- [ ] Add player choice mechanism for longer lines
- [ ] Implement ring elimination on line collapse
- [ ] Handle multiple line processing order
- [ ] Write tests for line scenarios

**Reference:** Section 11

#### 1.7 Implement Territory Disconnection (HIGH)

**Estimated Time:** 4-5 days
**Files:** `src/server/game/RuleEngine.ts`, `src/server/game/BoardManager.ts`

- [ ] Implement `findDisconnectedRegions()` with Von Neumann adjacency
- [ ] Add representation checking
- [ ] Implement self-elimination prerequisite validation
- [ ] Add border marker collapse
- [ ] Handle chain reactions
- [ ] Write tests for disconnection scenarios

**Reference:** Section 12, FAQ Q15

#### 1.8 Add Forced Elimination (MEDIUM)

**Estimated Time:** 1 day
**Files:** `src/server/game/GameEngine.ts`, `src/server/game/RuleEngine.ts`

- [ ] Detect when player has no valid moves but controls stacks
- [ ] Force cap elimination
- [ ] Update player state

**Reference:** Section 4.4, FAQ Q24

#### 1.9 Fix Player State Updates (MEDIUM)

**Estimated Time:** 1 day
**Files:** `src/server/game/GameEngine.ts`

- [ ] Update `ringsInHand` on placement
- [ ] Update `eliminatedRings` on elimination
- [ ] Update `territorySpaces` on collapse
- [ ] Ensure counts match actual game state

**Phase 1 Total Estimated Time:** 18-27 days (3-5 weeks)

---

### **PHASE 2: Testing & Validation** (After Phase 1)

**Goal:** Ensure correctness through comprehensive testing

**Tasks:**

#### 2.1 Unit Tests

**Estimated Time:** 1 week

- [ ] Test BoardManager position utilities
- [ ] Test BoardManager adjacency calculations
- [ ] Test BoardManager line detection
- [ ] Test RuleEngine movement validation
- [ ] Test RuleEngine capture validation
- [ ] Test RuleEngine line formation
- [ ] Test RuleEngine Territory disconnection
- [ ] Test GameEngine state transitions

#### 2.2 Integration Tests

**Estimated Time:** 3-5 days

- [ ] Test complete turn sequence
- [ ] Test ring placement → movement → capture flow
- [ ] Test line formation → ring elimination
- [ ] Test Territory disconnection → ring elimination
- [ ] Test chain capture sequences
- [ ] Test forced elimination scenarios

#### 2.3 Scenario Tests

**Estimated Time:** 3-5 days

- [ ] Test 180° reversal capture pattern (FAQ example)
- [ ] Test cyclic capture pattern (FAQ example)
- [ ] Test Territory disconnection example (Section 16.8.6)
- [ ] Test graduated line rewards
- [ ] Test victory conditions
- [ ] Test edge cases from FAQ (Q1-Q24)

**Phase 2 Total Estimated Time:** 2-3 weeks

---

### **PHASE 3: Frontend Implementation** (After Phase 2)

**Goal:** Build playable user interface

**Tasks:**

#### 3.1 Board Rendering

- [ ] Square board (8x8 and 19x19)
- [ ] Hexagonal board
- [ ] Cell/space components
- [ ] Coordinate overlay
- [ ] Responsive sizing

#### 3.2 Game Pieces

- [ ] Ring stack visualization
- [ ] Marker display
- [ ] Collapsed space display
- [ ] Player color coding
- [ ] Stack height indicators

#### 3.3 Controls & Interaction

- [ ] Ring placement interface
- [ ] Move selection (click source/destination)
- [ ] Valid move highlighting
- [ ] Move confirmation
- [ ] Undo/redo (if supported)

#### 3.4 Game State Display

- [ ] Current player indicator
- [ ] Ring counts display
- [ ] Territory statistics
- [ ] Move history
- [ ] Timer display

**Phase 3 Total Estimated Time:** 3-4 weeks

---

### **PHASE 4: Advanced Features** (After Phase 3)

**Tasks:**

- AI implementation (difficulty levels 1-10)
- WebSocket event completion
- Database integration
- Spectator mode
- Replay system

**Phase 4 Total Estimated Time:** 4-6 weeks

</details>

---

## 🛠️ Development Guidelines

### Code Style

**TypeScript:**

- Use explicit types (avoid `any`)
- Prefer interfaces over type aliases for objects
- Use const assertions where appropriate
- Follow existing naming conventions

**Naming Conventions:**

- Classes: PascalCase (`GameEngine`, `BoardManager`)
- Methods: camelCase (`validateMove`, `findAllLines`)
- Constants: UPPER_SNAKE_CASE (`BOARD_CONFIGS`)
- Interfaces: PascalCase with descriptive names

**Comments:**

- Add JSDoc comments for public methods
- Include rule references for game logic
- Explain non-obvious implementation choices

### Testing Requirements

**All new code must include tests:**

- Unit tests for individual functions
- Integration tests for complex workflows
- Scenario tests from rules document

**Test Structure:**

```typescript
describe('ComponentName', () => {
  describe('methodName', () => {
    it('should handle normal case', () => {
      // Test implementation
    });

    it('should handle edge case', () => {
      // Test implementation
    });

    it('should match rule X from section Y', () => {
      // Reference: ringrift_complete_rules.md Section Y
      // Test implementation
    });
  });
});
```

### Rule Implementation Process

When implementing game rules:

1. **Read the rule** in `ringrift_complete_rules.md`
2. **Check FAQ** for clarifications (Section 15.4)
3. **Write tests first** based on rule description
4. **Implement the logic** to pass tests
5. **Add code comments** referencing rule sections
6. **Test edge cases** from FAQ

**Example:**

```typescript
/**
 * Validates ring movement according to RingRift rules
 *
 * Rule Reference: Section 8.2 - Minimum Distance Requirements
 * Rules:
 * - Must move at least stack height spaces
 * - Can land on any valid space beyond markers meeting distance
 * - Cannot pass through collapsed spaces or other rings
 *
 * @param move - The move to validate
 * @param gameState - Current game state
 * @returns true if move is valid
 */
validateMovement(move: Move, gameState: GameState): boolean {
  // Implementation...
}
```

---

## 🔍 Code Quality

### Linting

This project uses ESLint to enforce code quality standards. **ESLint failures will block CI**, so it's important to check and fix lint issues locally before pushing.

```bash
# Check for lint errors
npm run lint

# Auto-fix lint errors
npm run lint:fix
```

**Important:** Run `npm run lint` before committing to catch and fix any issues locally. The `lint:fix` command will automatically fix many common issues, but some may require manual intervention.

### Pre-commit Hooks

This project uses Husky and lint-staged to automatically run linting on staged files before each commit. If linting fails, the commit will be blocked until the issues are resolved.

### Type Checking

TypeScript compilation errors also block CI. Ensure your code compiles cleanly:

```bash
npm run build
```

---

## 📋 Pull Request Process

### Before Submitting

1. **Run all tests:** `npm test`
2. **Check linting:** `npm run lint`
3. **Format code:** `npm run lint:fix`
4. **Build successfully:** `npm run build`
5. **Update documentation** if needed

### PR Title Format

Use conventional commits:

```
<type>(<scope>): <description>

Examples:
feat(game-engine): implement marker system
fix(rule-engine): correct movement distance validation
test(board-manager): add line detection tests
docs(readme): update installation instructions
```

**Types:**

- `feat`: New feature
- `fix`: Bug fix
- `test`: Adding tests
- `docs`: Documentation changes
- `refactor`: Code refactoring
- `perf`: Performance improvements
- `chore`: Build process, dependencies

### PR Description Template

```markdown
## Changes

Brief description of what this PR accomplishes

## Related Issues

Fixes #issue-number
Relates to #issue-number

## Rule References

- Section X.Y: [Rule name]
- FAQ QZ: [Question]

## Testing

- [ ] Unit tests added/updated
- [ ] Integration tests added/updated
- [ ] Manual testing completed
- [ ] All tests pass

## Checklist

- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Comments added to complex logic
- [ ] Documentation updated
- [ ] No new warnings introduced
```

---

### CI & PR policy (S-05.F.1)

> **Status (2025-11-27): Active (documentation/policy only, non-semantics)** \
> Implements S-05.F.1 from `docs/SUPPLY_CHAIN_AND_CI_SECURITY.md`. This section does **not** define game rules or lifecycle semantics. Those remain in:
>
> - `RULES_CANONICAL_SPEC.md`, `ringrift_complete_rules.md`, `ringrift_compact_rules.md` (rules semantics SSoT)
> - `docs/CANONICAL_ENGINE_API.md` + shared TS/WebSocket types and schemas (lifecycle/API SSoT)
> - Shared TS engine + contracts + contract vectors under `src/shared/engine/**` and `tests/fixtures/contract-vectors/v2/**`
>
> CI is a consumer of those SSoTs and of the test-layer taxonomy described in `tests/README.md`, `tests/TEST_LAYERS.md`, and `tests/TEST_SUITE_PARITY_PLAN.md`.

#### Branches and reviews

This policy applies to all pull requests targeting the protected branches:

- `main`
- `develop`

Expectations:

- Use PRs for all changes to `main` / `develop` (no direct pushes).
- Each PR targeting `main` or `develop` should have **at least one approval** from a non-author reviewer.
- For changes that touch any of the following, a **second review is strongly recommended**:
  - Shared TS engine orchestrator or aggregates under `src/shared/engine/**`
  - Contracts and vectors under `src/shared/engine/contracts/**` and `tests/fixtures/contract-vectors/v2/**`
  - Lifecycle/API surfaces in `docs/CANONICAL_ENGINE_API.md` or shared WebSocket types/schemas
  - Python rules/AI hosts that adapt the canonical engine (for example `ai-service/app/rules/*`, `ai-service/app/game_engine.py`)

#### Required CI checks (expected branch-protection gates)

CI is defined in [`.github/workflows/ci.yml`](./.github/workflows/ci.yml). For PRs into `main` and `develop`, the following jobs are expected to be configured as **required status checks** via branch protection:

- `lint-and-typecheck` – ESLint + TypeScript compilation for root, server, and client.
- `test` – `npm run test:coverage` across the Jest suite. See `tests/README.md`, `tests/TEST_LAYERS.md`, and `tests/TEST_SUITE_PARITY_PLAN.md` for the core vs diagnostics split and rules/trace/integration taxonomy.
- `ts-rules-engine` – `npm run test:ts-rules-engine`, the canonical TS rules semantics signal over the shared engine (helpers → aggregates → orchestrator → contracts).
- `build` – `npm run build` for server and client, plus archiving of the `dist/` artefacts.
- `security-scan` – Node dependency audits via `npm audit --production --audit-level=high` and a Snyk scan with `--severity-threshold=high`.
- `docker-build` – Docker Buildx build of the main `Dockerfile` (tagged `ringrift:test`, push disabled) to validate image build correctness in CI.
- `python-rules-parity` – TS→Python rules-parity fixture generation followed by `pytest ai-service/tests/parity/test_rules_parity_fixtures.py` (primary TS↔Python rules parity signal).
- `python-dependency-audit` – `pip-audit -r ai-service/requirements.txt --severity HIGH` over the AI-service dependency set.

The `e2e-tests` Playwright job is **CI-blocking**. Infrastructure (Postgres, Redis) is configured via GitHub Actions services, and Playwright has retry support (2x in CI) for flaky test resilience. Contributors should:

- Ensure E2E tests pass before merging PRs.
- Run E2E tests locally when making changes to auth/session flows, lobby/game lifecycle, or WebSocket transport behaviour.

As CI evolves (for example additional jobs for topology, SBOMs, or AI pipelines), new required checks should be documented in `docs/SUPPLY_CHAIN_AND_CI_SECURITY.md` and, where relevant, referenced here.

#### Handling flaky or failing checks

- Do **not** merge PRs into `main` or `develop` while any of the required checks listed above are failing, except in a clearly documented emergency fix.
- For known-flaky suites (typically long-running or infra-sensitive tests inside `test` or `e2e-tests`):
  - Prefer re-running the job in CI and/or fixing the underlying flakiness in a follow-up PR.
  - If you must temporarily quarantine a test, mark it clearly in code (for example with a comment referencing the tracking issue) and update the relevant test meta-doc (`tests/TEST_LAYERS.md` or `tests/TEST_SUITE_PARITY_PLAN.md`).
- Temporarily disabling a CI job (for example commenting out a job in `ci.yml`) should be rare, explicitly called out in the PR description, and paired with a tracking issue. When this happens for security- or CI-related jobs, also leave a short note in `docs/SUPPLY_CHAIN_AND_CI_SECURITY.md` so the S-05.F design stays in sync with reality.

#### Releases and deployments

- Production and staging deployments should originate from **tagged commits on `main`** (or dedicated release branches) that have passed all required CI checks above.
- Prefer Docker images built by CI (via the `docker-build` pipeline or a future push-enabled variant) over ad-hoc images built on developer machines.
- For incident response or security review, use this section together with:
  - `docs/SUPPLY_CHAIN_AND_CI_SECURITY.md` (S-05.F design and job map)
  - `docs/SECURITY_THREAT_MODEL.md` (threat scenarios and S-05 backlog)

## 🐛 Bug Reports

### Required Information

```markdown
**Bug Description:**
Clear description of the issue

**Expected Behavior:**
What should happen according to the rules

**Actual Behavior:**
What actually happens

**Rule Reference:**
Section/FAQ from ringrift_complete_rules.md

**Steps to Reproduce:**

1. Step one
2. Step two
3. Step three

**Code Location:**
File: src/server/game/FileName.ts
Method: methodName()
Line: 123

**Proposed Fix:** (optional)
Description of how to fix
```

---

## 🎓 Learning Resources

### Essential Reading

**Must Read (in order):**

1. `ringrift_complete_rules.md` - Complete game rules
2. `CURRENT_STATE_ASSESSMENT.md` - Current, code-verified state analysis
3. `KNOWN_ISSUES.md` - Specific bugs and gaps to fix

**Architecture & Planning Reference:**

1. `ARCHITECTURE_ASSESSMENT.md` - Current architecture, refactoring axes, and deep improvement plan
2. `STRATEGIC_ROADMAP.md` - Phased roadmap from core logic to UI, AI, and multiplayer
3. `archive/PLAYABLE_GAME_IMPLEMENTATION_PLAN.md` - Historical practical plan for a fully playable experience (lobby → game → AI → victory); for current, code-verified status and tasks, defer to `CURRENT_STATE_ASSESSMENT.md`, `ARCHITECTURE_ASSESSMENT.md`, `KNOWN_ISSUES.md`, and `TODO.md`.
4. `src/shared/types/game.ts` - Shared game type definitions
5. `archive/ringrift_architecture_plan.md` and `archive/TECHNICAL_ARCHITECTURE_ANALYSIS.md` - Historical design docs (preserved for context, but superseded by the files above)

### Understanding the Game

**Start Here:**

- Section 1.3: Quick Start Guide (rules doc)
- Section 2: Simplified 8×8 Version (rules doc)
- Section 15.4: FAQ (rules doc)

**Core Mechanics:**

- Section 8: Movement
- Section 9-10: Captures
- Section 11: Line Formation
- Section 12: Territory Disconnection

**Complex Scenarios:**

- Section 15.3: Common Capture Patterns
- Section 16.8.6: Territory Disconnection Example
- FAQ Q1-Q24: Edge cases and clarifications

---

## 💡 Development Tips

### Working on Game Logic

1. **Always reference the rules document**
2. **Start with tests** - Write what should happen
3. **Implement incrementally** - One rule at a time
4. **Test edge cases** - Check FAQ for tricky scenarios
5. **Document your code** - Include rule references

### Common Pitfalls

**Avoid:**

- ❌ Implementing without reading full rule description
- ❌ Mixing overtaking and elimination captures
- ❌ Forgetting marker interactions
- ❌ Not checking adjacency types (Moore vs Von Neumann)
- ❌ Missing mandatory vs optional actions

**Do:**

- ✅ Read entire rule section before coding
- ✅ Check FAQ for clarifications
- ✅ Test with examples from rules document
- ✅ Use correct adjacency for context (movement/lines/territory)
- ✅ Distinguish "must" from "may" in rules

### Debugging Game Logic

**When something doesn't work:**

1. **Find the rule:** Which section describes this behavior?
2. **Check the FAQ:** Is there a clarification?
3. **Compare types:** Moore (8-dir) vs Von Neumann (4-dir) vs Hexagonal (6-dir)
4. **Trace the flow:** Log each step of the process
5. **Test boundaries:** What happens at edges?

---

## 🤝 Getting Help

### Questions?

- **Game rules:** Check `ringrift_complete_rules.md` (especially the FAQ section)
- **Architecture & refactoring axes:** See `ARCHITECTURE_ASSESSMENT.md`
- **Roadmap & priorities:** See `STRATEGIC_ROADMAP.md` and `TODO.md`
- **Current bugs:** Check `KNOWN_ISSUES.md`
- **Current implementation state:** Review `CURRENT_STATE_ASSESSMENT.md`

### Discussion

- Open a GitHub Discussion for questions
- Create an issue for bugs
- Submit a PR for fixes

---

## 📅 Milestones

### Milestone 1: Core Logic Complete

**Target:** 4-6 weeks from start  
**Criteria:**

- All Phase 1 tasks completed
- Unit tests for all game rules
- No critical bugs (P0 issues resolved)

### Milestone 2: Tested & Validated

**Target:** 8-10 weeks from start  
**Criteria:**

- All Phase 2 tasks completed
- 80%+ test coverage
- All scenario tests passing

### Milestone 3: Playable Game

**Target:** 12-15 weeks from start  
**Criteria:**

- All Phase 3 tasks completed
- Functional UI for all board types
- Multiplayer working

### Milestone 4: Feature Complete

**Target:** 18-24 weeks from start  
**Criteria:**

- All phases completed
- AI opponents working
- Database integration complete
- Ready for beta testing

---

## 📄 License

By contributing to RingRift, you agree that your contributions will be licensed under the MIT License.

---

## 🙏 Thank You!

Your contributions help make RingRift a reality. Whether you fix a small bug or implement a major feature, every contribution matters!

**Happy coding!** 🎮

---

**Document Version:** 1.2
**Last Updated:** November 27, 2025
**Maintainer:** Development Team
