# Contributing to RingRift

Thank you for your interest in contributing to RingRift! This document provides guidelines and priorities for development work.

**Related Documents (single source of truth):**

- [CURRENT_STATE_ASSESSMENT.md](./CURRENT_STATE_ASSESSMENT.md) - Factual, code-verified status snapshot
- [TODO.md](./TODO.md) - Task tracking and detailed implementation checklist
- [KNOWN_ISSUES.md](./KNOWN_ISSUES.md) - Specific bugs and issues
- [RULES_ENGINE_ARCHITECTURE.md](./RULES_ENGINE_ARCHITECTURE.md) - Canonical engine architecture and SSOT boundaries
- [STRATEGIC_ROADMAP.md](./STRATEGIC_ROADMAP.md) - Phased strategic plan, milestones, and SLOs
- [docs/archive/plans/ARCHITECTURE_ASSESSMENT.md](./docs/archive/plans/ARCHITECTURE_ASSESSMENT.md) - Historical architecture analysis (archived)
- [docs/architecture/CANONICAL_ENGINE_API.md](./docs/architecture/CANONICAL_ENGINE_API.md) - Lifecycle and API reference

---

## 🚦 Development Status

**Last Updated:** 2025-12-07

**Current State:** Stable beta – **all 14 development waves complete**, orchestrator at 100%

### Key Metrics

| Metric                      | Value                      |
| --------------------------- | -------------------------- |
| TypeScript tests (CI-gated) | 2,987 passing              |
| Python tests                | 836 passing                |
| Contract vectors            | 54 (100% TS↔Python parity) |
| Line coverage               | ~69%                       |
| Canonical phases            | 8                          |
| Development waves complete  | 14/14                      |

### Priority Focus

1. **Client component test coverage** – Main gap (~0% React component coverage)
2. **Production validation** – Load testing at production scale
3. **Mobile UX polish** – Responsive design improvements
4. **AI improvements** – Neural network integration, weight optimization

Before contributing, please review:

1. **[TODO.md](./TODO.md)** – Active task tracker (canonical priorities)
2. **[CURRENT_STATE_ASSESSMENT.md](./CURRENT_STATE_ASSESSMENT.md)** – Verified implementation status
3. **[ringrift_complete_rules.md](./ringrift_complete_rules.md)** – Comprehensive game rules
4. **[RULES_ENGINE_ARCHITECTURE.md](./RULES_ENGINE_ARCHITECTURE.md)** – Engine architecture

---

## 🎯 Development Priorities

**Current priorities are tracked in [TODO.md](./TODO.md).** Key areas include:

### Completed Waves (All ✅)

| Wave | Name                            | Status      |
| ---- | ------------------------------- | ----------- |
| 1-4  | Core Engine & Architecture      | ✅ Complete |
| 5    | Orchestrator Production         | ✅ Complete |
| 6    | Observability & Monitoring      | ✅ Complete |
| 7    | Production Validation           | ✅ Complete |
| 8    | Player Experience & UX          | ✅ Complete |
| 9    | AI Strength & Optimization      | ✅ Complete |
| 10   | Game Records & Training Data    | ✅ Complete |
| 11   | Test Hardening & Golden Replays | ✅ Complete |
| 12   | Matchmaking & Ratings           | ✅ Complete |
| 13   | Multi-Player (3-4 Players)      | ✅ Complete |
| 14   | Accessibility & Code Quality    | ✅ Complete |

### Active Work (P0-P2)

| Area                          | Status                              | Priority |
| ----------------------------- | ----------------------------------- | -------- |
| **Client Test Coverage**      | Main gap (~0%)                      | P0       |
| **Production Scale Testing**  | Load test framework ready           | P0       |
| **Security Hardening Review** | Threat model exists, review pending | P1       |
| **Mobile UX**                 | Desktop-first, mobile needs polish  | P1       |
| **ML-backed AI**              | Scaffolding exists, not integrated  | P2       |

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

### Required tests for rules/AI/WebSocket changes

When a change touches any of:

- `src/shared/engine/**` (rules engine and orchestrator),
- `src/server/game/**` (backend GameEngine, GameSession, WebSocket/AI integration),
- `src/client/sandbox/**` (ClientSandboxEngine and sandbox AI/UX),
- WebSocket lifecycle or AI service boundaries (`WebSocketServer`, `AIServiceClient`, `AIInteractionHandler`),

you **must** run the P0 robustness profile before opening or merging a PR:

```bash
npm run test:p0-robustness
```

This script runs:

- `npm run test:ts-rules-engine` (shared-engine + orchestrator rules suites: RulesMatrix, FAQ, advanced turn/territory helpers),
- `npm run test:ts-integration` (backend/WebSocket/full game-flow integration),
- A focused parity/cancellation bundle:
  - `tests/contracts/contractVectorRunner.test.ts` (v2 contract vectors, including mixed territory/line sequences),
  - `tests/parity/Backend_vs_Sandbox.CaptureAndTerritoryParity.test.ts` (advanced capture + single-/multi-region line+territory backend↔sandbox parity),
  - `tests/unit/WebSocketServer.sessionTermination.test.ts` (WebSocket session termination + decision/AI cancellation for turn requests and AI-backed choices such as `region_order` and `line_reward_option`).

For UI-only or documentation-only changes, you can usually rely on the lighter `npm run test:core` lane; for anything rules/AI/WebSocket-adjacent, treat `npm run test:p0-robustness` as the default local gate.

---

<details>
<summary>📜 Historical Development Phases (November 2025 - All Completed)</summary>

All historical development phases have been completed. For detailed historical context, see the archived documents:

- `docs/archive/plans/ARCHITECTURE_ASSESSMENT.md` - Original architecture analysis
- `docs/archive/plans/ARCHITECTURE_REMEDIATION_PLAN.md` - Remediation plan (completed Nov 2025)
- `archive/PLAYABLE_GAME_IMPLEMENTATION_PLAN.md` - Original implementation plan

### Phase Summary

| Phase | Name                     | Completed |
| ----- | ------------------------ | --------- |
| 1     | Core Game Logic          | Nov 2025  |
| 2     | Testing & Validation     | Nov 2025  |
| 3     | Frontend Implementation  | Nov 2025  |
| 4     | Advanced Features        | Nov 2025  |
| 1.5   | Architecture Remediation | Nov 2025  |

### Key Milestones Achieved

- ✅ All core rules implemented in shared TypeScript engine
- ✅ Canonical turn orchestrator (`processTurn`, `processTurnAsync`)
- ✅ 6 domain aggregates (Placement, Movement, Capture, Line, Territory, Victory)
- ✅ Backend adapter (`TurnEngineAdapter.ts`, 326 lines)
- ✅ Sandbox adapter (`SandboxOrchestratorAdapter.ts`, 476 lines)
- ✅ Python contract tests with 100% parity on 54 vectors
- ✅ ~1,176 lines legacy code removed (PASS20)
- ✅ AI difficulty ladder (1-10) with service-backed choices
- ✅ 3 Grafana dashboards with 22 monitoring panels
- ✅ k6 load testing framework with 4 production scenarios
- ✅ ELO rating system and leaderboards
- ✅ 3-4 player multiplayer support
- ✅ Comprehensive accessibility features (WCAG compliance)

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

> **Status (2025-12-07): Active (documentation/policy only, non-semantics)** \
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
3. `TODO.md` - Active task tracking and priorities
4. `KNOWN_ISSUES.md` - Specific bugs and gaps to fix

**Architecture & Technical Reference:**

1. `RULES_ENGINE_ARCHITECTURE.md` - Canonical engine architecture, SSOT boundaries, and host integration
2. `docs/architecture/CANONICAL_ENGINE_API.md` - Lifecycle and API reference
3. `STRATEGIC_ROADMAP.md` - Phased roadmap, milestones, and SLOs
4. `AI_ARCHITECTURE.md` - AI system architecture, difficulty ladder, and training
5. `src/shared/types/game.ts` - Shared game type definitions

**Historical Reference (archived):**

- `docs/archive/plans/ARCHITECTURE_ASSESSMENT.md` - Original architecture analysis
- `archive/PLAYABLE_GAME_IMPLEMENTATION_PLAN.md` - Original implementation plan

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

## ♿ Accessibility Guidelines

RingRift is committed to being accessible to all players. When contributing UI components, follow these guidelines. For the full accessibility feature documentation, see [`docs/ACCESSIBILITY.md`](docs/ACCESSIBILITY.md).

### ARIA Requirements for New Components

**All interactive elements must include:**

1. **Appropriate ARIA roles:**
   - Use `role="button"` for clickable non-button elements
   - Use `role="grid"` and `role="gridcell"` for game boards
   - Use `role="status"` for live state indicators (phase, turn, timers)
   - Use `role="dialog"` with `aria-modal="true"` for modal dialogs

2. **Descriptive labels:**

   ```tsx
   // ✅ Good - descriptive label
   <button aria-label="Place ring at position A3">

   // ❌ Bad - no accessible name
   <button onClick={handleClick}>
   ```

3. **Live regions for dynamic content:**

   ```tsx
   // Use aria-live for announcements
   <div aria-live="polite">  // Non-urgent updates
   <div aria-live="assertive">  // Time-critical alerts (e.g., decision timer)
   ```

4. **Proper state attributes:**
   ```tsx
   aria-selected={isSelected}
   aria-disabled={isDisabled}
   aria-expanded={isOpen}
   aria-pressed={isToggled}
   ```

### Focus Management Patterns

**Focus must be visible and logical:**

1. **Tab order:** Ensure logical tab order follows visual layout
2. **Focus trapping:** Modal dialogs must trap focus within
3. **Focus restoration:** Return focus to trigger element when closing dialogs
4. **Skip links:** Provide "skip to main content" for navigation-heavy pages

**Example focus trap pattern:**

```tsx
useEffect(() => {
  if (!isOpen) return;

  const focusable = dialogRef.current?.querySelectorAll<HTMLElement>(
    'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
  );
  const first = focusable?.[0];
  const last = focusable?.[focusable.length - 1];

  first?.focus();

  const handleKeyDown = (e: KeyboardEvent) => {
    if (e.key === 'Tab' && e.shiftKey && document.activeElement === first) {
      e.preventDefault();
      last?.focus();
    } else if (e.key === 'Tab' && !e.shiftKey && document.activeElement === last) {
      e.preventDefault();
      first?.focus();
    }
  };

  document.addEventListener('keydown', handleKeyDown);
  return () => document.removeEventListener('keydown', handleKeyDown);
}, [isOpen]);
```

### Keyboard Navigation

**All functionality must be keyboard-accessible:**

1. **Game board navigation:** Arrow keys, Enter/Space for selection, Escape to cancel
2. **Dialog navigation:** Tab for focus, Enter to confirm, Escape to close
3. **Global shortcuts:** Use `useGlobalGameShortcuts` hook for consistent behavior

**Existing hooks to use:**

- `useKeyboardNavigation` - Board cell navigation with arrow keys
- `useGlobalGameShortcuts` - Game-wide shortcuts (?, R, M, F, Ctrl+Z)

```tsx
import { useKeyboardNavigation, useGlobalGameShortcuts } from '@/hooks';

const nav = useKeyboardNavigation({
  boardType: 'square8',
  size: 8,
  onSelect: handleCellClick,
  onClear: handleClear,
});

useGlobalGameShortcuts({
  onShowHelp: () => setHelpOpen(true),
  onResign: () => setResignConfirmOpen(true),
});
```

### Color and Visual Accessibility

**Don't rely on color alone:**

1. **Use `AccessibilityContext` for player colors:**

   ```tsx
   const { getPlayerColorClass, getPlayerColor } = useAccessibility();

   // Get Tailwind class for current color vision mode
   const bgClass = getPlayerColorClass(playerIndex, 'bg');

   // Get hex color for SVG/canvas
   const hexColor = getPlayerColor(playerIndex);
   ```

2. **Add secondary indicators:**
   - Icons or shapes alongside color
   - Patterns for colorblind modes (applied automatically via CSS)
   - Text labels for critical information

3. **Respect user preferences:**

   ```tsx
   const { effectiveReducedMotion, highContrastMode } = useAccessibility();

   // Skip animations if user prefers reduced motion
   const animation = effectiveReducedMotion ? 'none' : 'pulse';
   ```

### Testing Accessibility Changes

**Required tests for accessibility PRs:**

1. **Unit tests for ARIA attributes:**

   ```tsx
   it('has accessible role and label', () => {
     render(<MyComponent />);
     expect(screen.getByRole('button', { name: /place ring/i })).toBeInTheDocument();
   });
   ```

2. **Keyboard navigation tests:**

   ```tsx
   it('responds to keyboard navigation', async () => {
     render(<BoardView {...props} />);
     await userEvent.tab();
     expect(screen.getByTestId('cell-0-0')).toHaveFocus();
     await userEvent.keyboard('{ArrowRight}');
     expect(screen.getByTestId('cell-1-0')).toHaveFocus();
   });
   ```

3. **Manual testing checklist:**
   - [ ] Navigate entire component with keyboard only
   - [ ] Test with screen reader (VoiceOver/NVDA)
   - [ ] Verify focus indicators are visible
   - [ ] Check color contrast (minimum 4.5:1 for text)
   - [ ] Test at 200% zoom
   - [ ] Test with `prefers-reduced-motion: reduce`

### Accessibility Files Reference

| File                                              | Purpose                     |
| ------------------------------------------------- | --------------------------- |
| `src/client/contexts/AccessibilityContext.tsx`    | User preferences management |
| `src/client/hooks/useKeyboardNavigation.ts`       | Board keyboard navigation   |
| `src/client/components/ScreenReaderAnnouncer.tsx` | Live region announcements   |
| `src/client/components/KeyboardShortcutsHelp.tsx` | Shortcuts help dialog       |
| `src/client/styles/accessibility.css`             | Accessibility CSS utilities |
| `docs/ACCESSIBILITY.md`                           | Full accessibility guide    |

---

## 🤝 Getting Help

### Questions?

- **Game rules:** Check `ringrift_complete_rules.md` (especially the FAQ section)
- **Engine architecture:** See `RULES_ENGINE_ARCHITECTURE.md` and `docs/architecture/CANONICAL_ENGINE_API.md`
- **AI system:** See `AI_ARCHITECTURE.md` and `ai-service/README.md`
- **Roadmap & priorities:** See `TODO.md` and `STRATEGIC_ROADMAP.md`
- **Current bugs:** Check `KNOWN_ISSUES.md`
- **Current implementation state:** Review `CURRENT_STATE_ASSESSMENT.md`

### Discussion

- Open a GitHub Discussion for questions
- Create an issue for bugs
- Submit a PR for fixes

---

## 📅 Current Milestones

### ✅ Completed Milestones

| Milestone           | Status | Completed |
| ------------------- | ------ | --------- |
| Core Logic Complete | ✅     | Nov 2025  |
| Tested & Validated  | ✅     | Nov 2025  |
| Playable Game       | ✅     | Nov 2025  |
| Feature Complete    | ✅     | Dec 2025  |
| Waves 5-14          | ✅     | Dec 2025  |

### 🎯 v1.0 Release Criteria (In Progress)

**Target:** TBD

**Remaining Criteria:**

1. **Client Test Coverage** – Achieve meaningful React component test coverage (currently ~0%)
2. **Production Scale Testing** – Complete load testing at production scale
3. **Security Review** – Complete security hardening review per threat model
4. **Mobile UX** – Responsive design for mobile devices

**Quality Gates:**

- All 2,987+ TypeScript tests passing
- All 836+ Python tests passing
- 54 contract vectors at 100% parity
- No P0 bugs open
- Load test SLOs met with headroom

---

## 📄 License

By contributing to RingRift, you agree that your contributions will be licensed under the MIT License.

---

## 🙏 Thank You!

Your contributions help make RingRift a reality. Whether you fix a small bug or implement a major feature, every contribution matters!

**Happy coding!** 🎮

---

**Document Version:** 2.0
**Last Updated:** December 7, 2025
**Maintainer:** Development Team
