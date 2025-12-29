# Pass 16 – Comprehensive Assessment Report

> **⚠️ HISTORICAL DOCUMENT** – This is a point-in-time assessment from November 2025.
> For current project status, see:
>
> - [`../historical/CURRENT_STATE_ASSESSMENT.md`](../historical/CURRENT_STATE_ASSESSMENT.md) – Latest implementation status
> - `docs/PASS18A_ASSESSMENT_REPORT.md` – Most recent assessment pass

> **Assessment Date:** 2025-11-28
> **Assessment Pass:** 16 (Comprehensive Reassessment After Consolidation)
> **Assessor:** Architect mode – documentation, rules, AI, frontend, and SSOT review

> **Doc Status (2025-11-28): Historical**
> This report updates prior passes (PASS8–PASS15) using the consolidated shared engine and current coverage metrics from [`../historical/CURRENT_STATE_ASSESSMENT.md`](../historical/CURRENT_STATE_ASSESSMENT.md)(../historical/CURRENT_STATE_ASSESSMENT.md). It identifies the current weakest area, the hardest outstanding problem, and defines remediation tasks for subsequent passes.

---

## 1. Executive Summary

- **Weakest area (Pass 16): Frontend host architecture & UX ergonomics** – specifically the complexity and centralisation of [`GamePage.tsx`](../../../src/client/pages/GamePage.tsx) and related controllers. The client delivers a strong playable experience with good accessibility, but the main game host remains large and tightly coupled, which raises long-term maintainability and evolution risk.
- **Hardest outstanding problem: Orchestrator-first production rollout and legacy rules decommissioning** – now focused less on design and more on **operational rollout and cleanup**: completing the migration to the shared turn orchestrator across all hosts, tests, and operational environments while finally removing legacy paths in backend and sandbox engines, as outlined in [`WEAKNESS_ASSESSMENT_REPORT.md`](../plans/WEAKNESS_ASSESSMENT_REPORT.md) and refined in the updated rollout/runbook set.
- **Progress this pass (orchestrator/AI/runbooks):** Orchestrator and AI incident runbooks have been aligned with the shared-engine SSoT and rollout plan:
  - `docs/architecture/ORCHESTRATOR_ROLLOUT_PLAN.md` now documents flags (`ORCHESTRATOR_ADAPTER_ENABLED`, `ORCHESTRATOR_ROLLOUT_PERCENTAGE`, `ORCHESTRATOR_SHADOW_MODE_ENABLED`, `RINGRIFT_RULES_MODE`), “during incidents” posture, and CI profiles (orchestrator‑ON vs legacy/SHADOW diagnostics).
  - AI and game runbooks (`AI_ERRORS.md`, `AI_PERFORMANCE.md`, `AI_FALLBACK.md`, `AI_SERVICE_DOWN.md`, `GAME_HEALTH.md`, `GAME_PERFORMANCE.md`, `RULES_PARITY.md`) now consistently treat the shared TS engine + orchestrator as canonical, spell out when **not** to flip rules flags, and cross‑link to the rollout plan and `AI_ARCHITECTURE.md` §0 (AI incident overview).
  - Test meta-docs (`tests/README.md`, `tests/TEST_SUITE_PARITY_PLAN.md`) and archives (`archive/tests/**`, `ai-service/tests/archive/**`) clearly separate **canonical** `.shared`/contract/RulesMatrix suites from **diagnostic** trace/seed parity harnesses.
- **Documentation & tests:** Core rules, AI, and architecture docs remain accurate overall. Spot-checks found **minor drift** in test-count and helper-status descriptions (notably the placement helpers section in [`docs/architecture/CANONICAL_ENGINE_API.md`](../../architecture/CANONICAL_ENGINE_API.md) and historical counts in [`PROJECT_GOALS.md`](../../../PROJECT_GOALS.md)), but **no materially misleading semantics**. Coverage claims in [`../historical/CURRENT_STATE_ASSESSMENT.md`](../historical/CURRENT_STATE_ASSESSMENT.md)(../historical/CURRENT_STATE_ASSESSMENT.md) match the current Jest and pytest suite structure.
- **Frontend UX:** Board rendering and victory flows are now **highly accessible and polished**. [`BoardView.tsx`](../../../src/client/components/BoardView.tsx) and [`VictoryModal.tsx`](../../../src/client/components/VictoryModal.tsx) provide strong keyboard support, screen-reader hints, and motion/contrast safeguards via [`globals.css`](../../../src/client/styles/globals.css). Remaining sub-4 areas are **host-level code quality and touch/mobile ergonomics**, not basic usability.
- **Consolidation & SSOT:** Capture, movement, and placement semantics are **single-sourced in shared TS aggregates** – [`MovementAggregate`](../../../src/shared/engine/aggregates/MovementAggregate.ts), [`CaptureAggregate`](../../../src/shared/engine/aggregates/CaptureAggregate.ts), and [`PlacementAggregate`](../../../src/shared/engine/aggregates/PlacementAggregate.ts) – with closely aligned Python analogues in [`capture_chain.py`](../../../ai-service/app/rules/capture_chain.py) and [`placement.py`](../../../ai-service/app/rules/placement.py). Backend [`RuleEngine`](../../../src/server/game/RuleEngine.ts) and sandbox [`ClientSandboxEngine`](../../../src/client/sandbox/ClientSandboxEngine.ts) call these shared surfaces, and rules SSOT tooling such as [`rules-ssot-check.ts`](../../../scripts/ssot/rules-ssot-check.ts) is wired into CI.

---

## 2. Updated Component Scorecard (Pass 16)

### 2.1 Scoring model

Dimensions (1–5):

- **TC – Technical complexity:** How challenging the problem space and design are.
- **IC – Implementation completeness:** How fully requirements are implemented vs goals.
- **CQ – Code quality:** Structure, modularity, clarity, and evolution friendliness.
- **DC – Documentation coverage:** Design docs, runbooks, and discoverability.
- **Test – Test coverage:** Unit, integration, scenario, parity, and E2E coverage.
- **DR – Dependency risk:** External/systemic risk, brittleness, and blast radius.
- **AG – Alignment with goals:** Fit with [`PROJECT_GOALS.md`](../../../PROJECT_GOALS.md) and the strategic roadmap.

Composite scores use **DR** and **AG** at **2× weight**:

> composite = (TC + IC + CQ + DC + Test + 2·DR + 2·AG) / 9

Scores below **4.0** are highlighted as candidates for remediation focus.

### 2.2 Component scores (Pass 16)

| Component                                                                                      | TC  | IC  | CQ      | DC  | Test | DR  | AG  | **Composite** | Notes                                                                                                            |
| ---------------------------------------------------------------------------------------------- | --- | --- | ------- | --- | ---- | --- | --- | ------------- | ---------------------------------------------------------------------------------------------------------------- |
| **Shared TS rules engine & orchestrator** (`src/shared/engine/**`)                             | 5.0 | 5.0 | 5.0     | 5.0 | 5.0  | 4.5 | 5.0 | **4.9**       | Aggregates and orchestrator are the rules semantics SSoT; contracts and scenario suites are extensive.           |
| **Python rules engine & contracts** (`ai-service/app/rules/**`, `ai-service/app/game_engine/`) | 4.5 | 5.0 | 4.5     | 4.5 | 5.0  | 4.0 | 4.5 | **4.5**       | Mirrors TS semantics via contract vectors and parity tests; territory and forced elimination now aligned.        |
| **AI service & training stack** (`ai-service/app/**`, `ai-service/scripts/**`)                 | 5.0 | 4.0 | 4.0     | 4.5 | 4.5  | 3.5 | 4.0 | **4.2**       | Service and difficulty ladder are stable; training and NN pipelines are improving but still evolving.            |
| **Web/API backends** (HTTP routes, game/session services)                                      | 4.0 | 5.0 | 4.5     | 4.0 | 4.5  | 4.0 | 5.0 | **4.4**       | Endpoints, auth, session management, and AI integration are complete with strong tests.                          |
| **WebSocket & real-time layer** (`src/server/websocket/server.ts`)                             | 4.5 | 5.0 | 4.5     | 4.0 | 4.5  | 4.0 | 4.5 | **4.4**       | Robust auth, reconnection, choice handling, and metrics; scaling is planned and partially runbooked.             |
| **Database & persistence** (Prisma schema, lifecycle, retention)                               | 4.0 | 5.0 | 4.5     | 4.0 | 4.5  | 4.0 | 4.5 | **4.4**       | Schema and lifecycle services are complete (including S‑05.E), with solid integration tests.                     |
| **CI/CD & monitoring** (GitHub Actions, Prometheus, runbooks)                                  | 4.0 | 4.5 | 4.0     | 4.5 | 4.0  | 3.5 | 4.5 | **4.2**       | ESLint and E2E are now gating; Python cores are in CI; staging/prod rollout orchestration still needs hardening. |
| **Frontend client – architecture & pages** (`src/client/**`)                                   | 4.0 | 4.5 | **3.5** | 4.0 | 4.2  | 3.5 | 4.5 | **4.0**       | Strong UX and coverage, but `GamePage` and sandbox controllers remain large and tightly coupled.                 |
| **Documentation & runbooks** (`DOCUMENTATION_INDEX`, `docs/runbooks/**`)                       | 3.5 | 4.5 | 4.0     | 5.0 | 4.0  | 3.5 | 4.5 | **4.1**       | Docs are broad and indexed; a few sections show minor drift relative to current helpers and test counts.         |

**Sub‑4.0 dimensions:**

- **Frontend CQ = 3.5** – [`GamePage.tsx`](../../../src/client/pages/GamePage.tsx) and [`ClientSandboxEngine.ts`](../../../src/client/sandbox/ClientSandboxEngine.ts) are large, multi-responsibility files, even though they delegate rules semantics to shared helpers and the orchestrator. This concentrates UX and orchestration complexity into a small number of host components.
- **CI/CD DR = 3.5** – core CI gates are strong, but staged orchestrator rollout, hard SLO enforcement, and automated rollback approvals are still maturing.
- **Docs TC = 3.5 / DR = 3.5** – documentation is comprehensive but some sections lag behind the current code (e.g. placement-helper stub language), and the system still relies on SSOT banners and scripts to detect drift.

### 2.3 Weakest area selection

Given these scores and the **double weight on Dependency Risk and Alignment with Goals**, the **Frontend client – architecture & pages** component emerges as the weakest area despite an overall composite near 4.0:

- Most backend, rules, AI, and infra components now score in the **4.2–4.9** range with low structural risk.
- Frontend UX and accessibility are strong, but the **host architecture** around `GamePage` and sandbox controllers remains the **least modular and most difficult to evolve** area in the system.
- This weakness is primarily about **maintainability and evolvability** rather than user-visible instability, which is consistent with prior passes that shifted focus away from raw functionality to long-term sustainability.

---

## 3. Weakest Area Analysis – Frontend Host Architecture & UX Ergonomics

### 3.1 Scope

This weakest area covers the **host-level React code that orchestrates gameplay** on the client:

- [`GamePage.tsx`](../../../src/client/pages/GamePage.tsx) – primary host for backend games and local sandbox; owns layout, orchestration, error states, and many UX decisions.
- [`ClientSandboxEngine.ts`](../../../src/client/sandbox/ClientSandboxEngine.ts) – client-local engine harness that wires shared helpers, the orchestrator adapter [`SandboxOrchestratorAdapter`](../../../src/client/sandbox/SandboxOrchestratorAdapter.ts), and sandbox-specific UX semantics.
- Supporting sandbox utilities such as [`sandboxMovement.ts`](../../../src/client/sandbox/sandboxMovement.ts) and [`sandboxPlacement.ts`](../../../src/client/sandbox/sandboxPlacement.ts), which now mostly act as thin adapters over shared helpers.

### 3.2 Evidence of weakness

- **File size and responsibility concentration**
  - `GamePage` remains a very large component (~2k LOC) combining:
    - Backend game orchestration (WebSocket, reconnection, errors, AI diagnostics).
    - Local sandbox orchestration and AI stall tooling.
    - Layout and page-level UX decisions (HUD, log, chat, dialogs).
    - Selection & gesture semantics for movement, capture, and choices.
  - `ClientSandboxEngine` is large (~3.6k LOC) but conceptually cleaner; it delegates rules semantics to shared helpers such as [`applySimpleMovement()`](../../../src/shared/engine/aggregates/MovementAggregate.ts) and [`applyCaptureSegment()`](../../../src/shared/engine/aggregates/CaptureAggregate.ts), yet still carries substantial host- and trace-specific logic.
- **Coupling between UX and orchestration**
  - The main game host mixes **transport concerns** (WebSocket events, reconnection, AI errors) with **interaction semantics** (how placements and captures are chosen) and **layout**.
  - Sandbox and backend modes share much of this surface but diverge in details, making it harder to reason about or refactor only one mode.
- **Mobile and touch ergonomics**
  - Board interaction semantics are heavily optimised for **mouse and keyboard** (click-to-select, keyboard grid navigation, capture-direction choices via dialog), but advanced sandbox flows still lean on multi-step or less discoverable gestures.
  - While the underlying [`BoardView`](../../../src/client/components/BoardView.tsx) component has strong keyboard support and clear affordances, the host does not yet expose touch-friendly alternatives for all sandbox-specific flows (e.g. capture-direction choice paths) with equal clarity.
- **Testing vs behaviour surface**
  - Component, hooks, and E2E suites now cover many board and HUD interactions, but **host-level scenarios** for complex sandbox plus backend flows (AI stall diagnostics, advanced reconnection edge cases) are limited compared to the more thoroughly tested rules and AI layers.

### 3.3 Why this is weaker than other candidates

Other previously weak areas have improved markedly since earlier passes:

- **DevOps/CI enforcement** – [`WEAKNESS_ASSESSMENT_REPORT.md`](../plans/WEAKNESS_ASSESSMENT_REPORT.md) identified non-blocking ESLint and E2E jobs as the weakest area in Pass 7. These are now corrected; lint and Playwright suites are CI-blocking, and the Python core tests are included via dedicated jobs.
- **Shared rules helpers and parity** – The P0 movement/placement/capture helper gaps noted in mid-phase consolidation passes have been closed: movement, capture, and placement all share canonical helpers and aggregates, and contract-based TS↔Python parity suites are green.
- **Rules documentation and SSOT discipline** – [`CURRENT_RULES_STATE.md`](../../rules/CURRENT_RULES_STATE.md) and [`RULES_IMPLEMENTATION_MAPPING.md`](../../rules/RULES_IMPLEMENTATION_MAPPING.md) accurately track the canonical rules SSoT (rules spec + shared TS engine) and test harnesses, with SSOT checks like [`rules-ssot-check.ts`](../../../scripts/ssot/rules-ssot-check.ts) enforcing consistency.

By contrast, the **frontend host architecture** still carries:

- The **highest concentration of imperative orchestration logic** in the client.
- The **largest single React component** in the codebase (`GamePage`).
- A number of **future-facing responsibilities** (spectator UX, richer reconnection, sandbox AI diagnostics) that will be easier to evolve if the host surface is more modular.

### 3.4 Impact on project objectives

- **Maintainability & iteration speed** – The current structure makes it harder to introduce new UX features (e.g. improved spectator tools, new board topologies, or tutorial flows) without touching `GamePage` in many places.
- **Risk to onboarding contributors** – New developers face a steep learning curve to safely modify the main host components compared to more focused modules (rules aggregates, AI service, WebSocket server).
- **UX polish backlog** – Some multiplayer UX goals in [`../historical/CURRENT_STATE_ASSESSMENT.md`](../historical/CURRENT_STATE_ASSESSMENT.md)(../historical/CURRENT_STATE_ASSESSMENT.md) (spectator experience, reconnection UX, social features) are more expensive because of this concentration of responsibilities.

---

## 4. Hardest Outstanding Problem – Orchestrator-First Production Rollout & Legacy Decommissioning

### 4.1 Current state

The **canonical shared turn orchestrator** is implemented and widely integrated:

- Shared orchestrator in [`turnOrchestrator.ts`](../../../src/shared/engine/orchestration/turnOrchestrator.ts) with `processTurn` / `processTurnAsync` driving domain aggregates.
- Backend adapter [`TurnEngineAdapter`](../../../src/server/game/turn/TurnEngineAdapter.ts) and sandbox adapter [`SandboxOrchestratorAdapter`](../../../src/client/sandbox/SandboxOrchestratorAdapter.ts) wrap the orchestrator for respective hosts.
- Backend `RuleEngine` delegates core semantics to shared helpers and aggregates (placement, movement, capture) via calls such as [`applySimpleMovementAggregate()`](../../../src/server/game/RuleEngine.ts) and [`applyCaptureAggregate()`](../../../src/server/game/RuleEngine.ts).
- Sandbox `ClientSandboxEngine` increasingly routes canonical move application through orchestrator-based flows in [`applyCanonicalMoveInternal()`](../../../src/client/sandbox/ClientSandboxEngine.ts), gated by `useOrchestratorAdapter`.
- Contract-based parity tests and v2 test vectors in both TS and Python validate cross-language semantics, as summarised in [`../historical/CURRENT_STATE_ASSESSMENT.md`](../historical/CURRENT_STATE_ASSESSMENT.md)(../historical/CURRENT_STATE_ASSESSMENT.md) and [`AI_ARCHITECTURE.md`](../../architecture/AI_ARCHITECTURE.md).

However, **production rollout is not yet fully complete**:

- Feature flags and environment toggles (e.g. `ORCHESTRATOR_ADAPTER_ENABLED`, `ORCHESTRATOR_ROLLOUT_PERCENTAGE`, `ORCHESTRATOR_SHADOW_MODE_ENABLED`, `RINGRIFT_RULES_MODE`, `useOrchestratorAdapter`) remain the primary control surface for fallback to legacy paths; `docs/architecture/ORCHESTRATOR_ROLLOUT_PLAN.md` now documents how and when to use them, but they still permit regressing to legacy rules if misused.
- A substantial body of legacy rules orchestration and support code still exists (RuleEngine path-based post-processing, sandbox legacy helpers, Tier‑2 and Tier‑3 modules cited in [`WEAKNESS_ASSESSMENT_REPORT.md`](../plans/WEAKNESS_ASSESSMENT_REPORT.md)); these are now better fenced as **diagnostic** in docs and test plans but have not yet been fully removed.
- Staging/prod pipelines do not yet implement a full **SLO‑driven staged bake‑in + shadow mode + removal** sequence for orchestrator-only operation, even though the updated rollout/runbooks and parity plans now describe the desired profiles and metrics.

### 4.2 Why this remains hard

Even though much of the architecture design and implementation work is complete, the remaining rollout is still the hardest problem because it requires **multi-layer coordination under strict regression risk**:

- **Cross-host impact** – Changes affect backend game processing, client sandbox behaviour, and the Python rules parity story. A regression in the orchestrator path can surface as gameplay anomalies, AI evaluation divergence, or contract-test failures.
- **Test migration & invariants** – Existing tests that rely on legacy paths (especially around `RuleEngine.processMove` and sandbox legacy helpers) must be carefully migrated to orchestrator-first expectations without losing coverage or historical diagnostics.
- **Operational gating** – Production rollout requires coordinated staging experiments, shadow-mode validation, parity dashboards, and explicit rollback steps as outlined in prior rollout plans and pass reports.
- **Legacy code removal** – Removing thousands of lines of now-redundant logic is risky unless all call sites are proven to be on orchestrator-first paths and diagnostics-only helpers are clearly marked and covered.

### 4.3 Dependencies and risk

- **Dependencies**
  - Stable shared engine aggregates and orchestrator (already in place).
  - CI and monitoring strong enough to detect regressions (now improved but still being hardened for rollout scenarios).
  - Parity tests and contracts across TS and Python, especially around complex phases (chain capture, territory, last‑player‑standing).
- **Risk if left unsolved**
  - Continued duplication between orchestrator and legacy paths increases maintenance cost and the risk of rules drift over time.
  - Operational complexity around feature flags grows, especially if new features depend on orchestrator behaviour while older diagnostics still assume legacy flows.
  - Some architecture documents (including consolidation design and rollout runbooks) assume a future where orchestrator is the _only_ production path; leaving this incomplete undermines those assumptions.

In summary, **finishing the orchestrator-first rollout and safely decommissioning legacy rules paths** is still the most challenging open problem, even though the remaining work is now much more constrained than at the time of earlier passes.

---

## 5. Documentation & Test Consistency Findings

Spot-checks focused on documents either not deeply audited in earlier passes or likely to have drifted after consolidation and AI/engine improvements.

### 5.1 `PROJECT_GOALS.md`

- **Status:** _Minor drift_
- **Findings:**
  - High-level goals (stable rules SSoT, AI integration, strong documentation, production readiness) remain well aligned with the current architecture and implementation described in [`../historical/CURRENT_STATE_ASSESSMENT.md`](../historical/CURRENT_STATE_ASSESSMENT.md)(../historical/CURRENT_STATE_ASSESSMENT.md) and [`AI_ARCHITECTURE.md`](../../architecture/AI_ARCHITECTURE.md).
  - Earlier test-count references (e.g. "1195+" TS tests) are now **underestimates** compared to current metrics reported in [`../historical/CURRENT_STATE_ASSESSMENT.md`](../historical/CURRENT_STATE_ASSESSMENT.md)(../historical/CURRENT_STATE_ASSESSMENT.md), which lists **1629+** TS tests and **245** Python tests.
- **Recommendation:** Update numeric test-count examples to avoid confusion and point readers to `CURRENT_STATE_ASSESSMENT` as the live source of coverage counts.

### 5.2 `docs/architecture/STATE_MACHINES.md`

- **Status:** _Accurate_
- **Findings:**
  - Descriptions of `gameSession`, `aiRequest`, `choice`, and `connection` state machines match current implementations under `src/shared/stateMachines/**` and their usages in WebSocket flows and decision handling.
  - Associated tests (e.g. connection and reconnection suites) line up with the documented states and transitions.
- **Recommendation:** No changes required; keep referencing this document as the canonical lifecycle map alongside [`docs/architecture/CANONICAL_ENGINE_API.md`](../../architecture/CANONICAL_ENGINE_API.md).

### 5.3 `docs/architecture/CANONICAL_ENGINE_API.md`

- **Status:** _Major drift in specific subsections_
- **Findings:**
  - The document correctly positions the shared TS engine and orchestrator as the **lifecycle SSoT** and aligns with types under `src/shared/types/**` and `src/shared/engine/orchestration/types.ts`.
  - However, sections describing placement helpers still refer to them as **"design-time stubs"** and caution hosts not to depend on them until future phases complete.
  - In reality, [`placementHelpers.ts`](../../../src/shared/engine/placementHelpers.ts) and its aggregate wrapper [`PlacementAggregate`](../../../src/shared/engine/aggregates/PlacementAggregate.ts) are fully implemented and integrated, as confirmed by [`../historical/CURRENT_STATE_ASSESSMENT.md`](../historical/CURRENT_STATE_ASSESSMENT.md)(../historical/CURRENT_STATE_ASSESSMENT.md) and their dedicated test suites.
- **Recommendation:**
  - Update these sections to describe placement helpers as **canonical, production-backed helpers**, not stubs.
  - Cross-link directly to shared helper tests and contract vectors to reinforce SSOT status.

### 5.4 `docs/architecture/TOPOLOGY_MODES.md`, `docs/architecture/STATE_MACHINES.md`, runbooks

- **Status:** _Accurate and orchestrator-aligned_
- **Findings:**
  - [`docs/architecture/TOPOLOGY_MODES.md`](../../architecture/TOPOLOGY_MODES.md) correctly describes supported board topologies and references geometry helpers consistent with engine implementation and tests.
  - Runbooks such as [`docs/runbooks/RULES_PARITY.md`](../../runbooks/RULES_PARITY.md), [`docs/runbooks/AI_ERRORS.md`](../../runbooks/AI_ERRORS.md), [`docs/runbooks/AI_FALLBACK.md`](../../runbooks/AI_FALLBACK.md), [`docs/runbooks/AI_SERVICE_DOWN.md`](../../runbooks/AI_SERVICE_DOWN.md), [`docs/runbooks/GAME_HEALTH.md`](../../runbooks/GAME_HEALTH.md), and [`docs/runbooks/GAME_PERFORMANCE.md`](../../runbooks/GAME_PERFORMANCE.md) now explicitly:
    - Treat the canonical rules spec plus shared TS engine + orchestrator as the rules SSoT.
    - Document orchestrator flags (`ORCHESTRATOR_ADAPTER_ENABLED`, `ORCHESTRATOR_ROLLOUT_PERCENTAGE`, `ORCHESTRATOR_SHADOW_MODE_ENABLED`, `RINGRIFT_RULES_MODE`) and when **not** to change them in response to AI or infra-only incidents.
    - Cross-link to `AI_ARCHITECTURE.md` §0 (AI incident overview) and `docs/architecture/ORCHESTRATOR_ROLLOUT_PLAN.md` for Safe rollback.
- **Recommendation:** Keep these as **active operational references**; future changes to rules or rollout strategy should update `ORCHESTRATOR_ROLLOUT_PLAN` first and then keep these runbooks in sync.

### 5.5 `AI_ARCHITECTURE.md` and AI training docs

- **Status:** _Accurate and recently updated_
- **Findings:**
  - [`AI_ARCHITECTURE.md`](../../architecture/AI_ARCHITECTURE.md) now incorporates the current canonical rules SSoT (rules spec + shared TS engine) and orchestrator-first lifecycle, as well as detailed descriptions of the AI difficulty ladder, RNG determinism, and training pipelines.
  - New sections on pre-training preparation, `MemoryConfig`, and evaluation tooling match code in `ai-service/app/utils/memory_config.py`, training modules under `ai-service/app/training/**`, and tests such as [`test_multi_start_evaluation.py`](../../../ai-service/tests/test_multi_start_evaluation.py).
- **Recommendation:** Treat this as the SSoT for AI architecture, with `AI_IMPROVEMENT_BACKLOG` reserved for task-level planning.

### 5.6 Test suite and coverage claims

- [`../historical/CURRENT_STATE_ASSESSMENT.md`](../historical/CURRENT_STATE_ASSESSMENT.md)(../historical/CURRENT_STATE_ASSESSMENT.md) lists **1629+** TS tests and **245** Python tests across 230+ files, with detailed counts per suite type (component, hooks, contexts, helpers, contract tests).
- Workspace inspection confirms the presence of the referenced suites:
  - Shared engine and scenario tests under `tests/unit/**` and `tests/scenarios/**`.
  - Component and hooks tests for `BoardView`, view models, and state hooks.
  - Python rule, parity, and training tests under `ai-service/tests/**`.
- CI configuration runs these suites via standard test scripts and dedicated Python jobs; parity/vector tests are split so that **core correctness remains gating**, while fixture-specific parity failures are triaged separately. Diagnostic trace/seed parity suites have been moved into `archive/tests/**` and `ai-service/tests/archive/**` where appropriate and are clearly described as non‑canonical in `tests/README.md` and `tests/TEST_SUITE_PARITY_PLAN.md`.
- **Conclusion:** Coverage and classification claims in `CURRENT_STATE_ASSESSMENT` and recent PASS reports are **materially accurate**; remaining drift is limited to older numeric examples in `PROJECT_GOALS` and historical reports, not to current status documents.

---

## 6. Frontend UX Reassessment

### 6.1 UX dimension scores (Pass 16)

| Dimension                                                      | Score (1–5) | Notes                                                                                                                                                                                                                                   |
| -------------------------------------------------------------- | ----------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Accessibility (keyboard, ARIA, screen readers, reduced motion) | **4.6**     | `BoardView` uses grid semantics, focus management, and `aria-live` regions; `VictoryModal` is an accessible dialog with motion-safety via `prefers-reduced-motion` handling in [`globals.css`](../../../src/client/styles/globals.css). |
| Responsiveness & layout                                        | **4.2**     | Layout scales well from desktop to smaller viewports; some control panels and logs become cramped on very small screens.                                                                                                                |
| Visual clarity & animations                                    | **4.4**     | Piece movement, captures, valid-move pulses, and victory celebrations are visually clear and respect motion preferences.                                                                                                                |
| Error/loading/empty states                                     | **4.3**     | `GamePage` surfaces backend errors, reconnect banners, and loading states consistently; sandbox failure cases are handled but could be more discoverable.                                                                               |
| Host code quality & modularity                                 | **3.5**     | Main game host components (`GamePage`, `ClientSandboxEngine`) remain large, multi-responsibility units despite strong underlying architecture.                                                                                          |
| Mobile/touch ergonomics                                        | **3.7**     | Primary interactions work on touch devices, but advanced sandbox controls (capture direction, AI diagnostics) are less discoverable without keyboard/mouse.                                                                             |

### 6.2 Key observations

- **Accessibility:**
  - `BoardView` exposes cells as focusable grid cells with keyboard navigation and clearly labelled actions, satisfying core keyboard and screen-reader requirements.
  - `VictoryModal` uses a dialog pattern with focus trapping and proper labelling, and global CSS honours `prefers-reduced-motion` and high-contrast settings.
- **Visual feedback & animations:**
  - Movement and capture animations, marker highlights, and valid-move pulses provide strong state feedback without overwhelming the viewer.
  - Motion is suppressed or simplified when reduced-motion is requested, in line with best practices.
- **Host-level weaknesses:**
  - The complexity of `GamePage` and sandbox controllers makes it difficult to introduce small UX refinements (e.g. more granular spectator controls, per-device tweaks) without touching large sections of code.
  - Some sandbox-specific actions (advanced AI or territory scenarios) are not fully surfaced in the main UX and may feel opaque to non-developers.

### 6.3 Concrete UX improvement proposals

1. **P16.2-CODE – Decompose `GamePage` into focused containers**
   - **Goal:** Reduce `GamePage` complexity by extracting mode-specific controllers and layout shells, e.g. `BackendGameHost`, `SandboxGameHost`, and a minimal `GameShell` that wires them.
   - **Impact:** Improves readability and testability; lowers risk of regressions when adding new UX features or adjusting orchestrator integration.
   - **Effort:** Medium (multi-file refactor with existing tests as safety net).

2. **P16.3-CODE – Touch-friendly sandbox capture/placement UX**
   - **Goal:** Introduce explicit, touch-first controls for sandbox actions that are currently optimised for mouse/keyboard (e.g. capture-direction selection overlays, two-step placement confirmation on small screens).
   - **Impact:** Makes the sandbox a more approachable experimentation tool on tablets and phones, aligning with multiplayer and AI-observation goals.
   - **Effort:** Medium; mostly UI work building on existing shared-engine semantics.

3. **P16.4-CODE – Board interaction help & shortcut overlay**
   - **Goal:** Add a discoverable in-game help overlay (e.g. a "Board controls" button or hotkey) summarising keyboard shortcuts, selection rules, and sandbox-specific gestures.
   - **Impact:** Reduces cognitive load for new players and developers, and makes advanced interactions (e.g. chain-capture selection) clearer without reading external docs.
   - **Effort:** Small; largely presentational work combined with existing interaction metadata.

---

## 7. Consolidation & SSOT Verification

### 7.1 Movement

- **TS SSoT:** [`MovementAggregate`](../../../src/shared/engine/aggregates/MovementAggregate.ts) defines validation, enumeration, and mutation for non-capturing movement, delegating to shared helpers and `MovementBoardView`.
- **Python mirror:** Movement reachability and distance semantics are mirrored in Python via `BoardManager` and geometry helpers; capture-aware movement is handled in [`capture_chain.py`](../../../ai-service/app/rules/capture_chain.py) with consistent distance and obstruction checks.
- **Hosts:**
  - Backend [`RuleEngine`](../../../src/server/game/RuleEngine.ts) uses [`enumerateSimpleMoveTargetsFromStack()`](../../../src/server/game/RuleEngine.ts) and [`applySimpleMovementAggregate()`](../../../src/server/game/RuleEngine.ts) for movement validation and mutation.
  - Sandbox [`ClientSandboxEngine`](../../../src/client/sandbox/ClientSandboxEngine.ts) uses shared movement enumerators via [`enumerateSimpleMovementLandings`](../../../src/client/sandbox/sandboxMovement.ts) and applies movement through [`applySimpleMovement()`](../../../src/shared/engine/aggregates/MovementAggregate.ts).

### 7.2 Capture & chain capture

- **TS SSoT:** [`CaptureAggregate`](../../../src/shared/engine/aggregates/CaptureAggregate.ts) centralises capture validation, enumeration, mutation, and chain-capture state helpers. Validation defers to [`validateCaptureSegmentOnBoard()`](../../../src/shared/engine/core.ts), and enumeration uses shared movement directions and board adapters.
- **Python mirror:** [`capture_chain.py`](../../../ai-service/app/rules/capture_chain.py) implements Python analogues such as [`validate_capture_segment_on_board_py()`](../../../ai-service/app/rules/capture_chain.py), [`enumerate_capture_moves_py()`](../../../ai-service/app/rules/capture_chain.py), and [`apply_capture_py()`](../../../ai-service/app/rules/capture_chain.py), delegating final mutation to the Python [`GameEngine`](../../../ai-service/app/game_engine/__init__.py).
- **Hosts:**
  - Backend [`RuleEngine.processCapture()`](../../../src/server/game/RuleEngine.ts) is a thin adapter over [`applyCaptureAggregate()`](../../../src/server/game/RuleEngine.ts), with chain continuation determined via shared continuation helpers.
  - Sandbox `ClientSandboxEngine` uses shared capture helpers via [`applyCaptureSegmentAggregate()`](../../../src/client/sandbox/ClientSandboxEngine.ts) and chain-continuation checks via [`getChainCaptureContinuationInfoAggregate()`](../../../src/client/sandbox/ClientSandboxEngine.ts).

### 7.3 Placement & no-dead-placement

- **TS SSoT:** [`PlacementAggregate`](../../../src/shared/engine/aggregates/PlacementAggregate.ts) exposes [`validatePlacementOnBoard()`](../../../src/shared/engine/aggregates/PlacementAggregate.ts), [`enumeratePlacementPositions()`](../../../src/shared/engine/aggregates/PlacementAggregate.ts), and placement mutators, relying on shared helpers such as `hasAnyLegalMoveOrCaptureFromOnBoard` to enforce no-dead-placement.
- **Python mirror:** [`placement.py`](../../../ai-service/app/rules/placement.py) defines parallel structures – [`PlacementContextPy`](../../../ai-service/app/rules/placement.py), [`validate_placement_on_board_py()`](../../../ai-service/app/rules/placement.py), and [`apply_place_ring_py()`](../../../ai-service/app/rules/placement.py) – delegating low-level mutation and hypothetical placement checks to the Python `GameEngine`.
- **Hosts:**
  - Backend `RuleEngine` calls [`validatePlacementOnBoardAggregate()`](../../../src/server/game/RuleEngine.ts) and [`applyPlacementMoveAggregate()`](../../../src/server/game/RuleEngine.ts) for validation and mutation during `ring_placement`.
  - Sandbox `ClientSandboxEngine` routes both human and AI placements through shared helpers, notably [`validatePlacementAggregate()`](../../../src/client/sandbox/ClientSandboxEngine.ts), [`applyPlacementMoveAggregate()`](../../../src/client/sandbox/ClientSandboxEngine.ts), and enumerators such as [`enumeratePlacementPositions()`](../../../src/client/sandbox/ClientSandboxEngine.ts).

### 7.4 Residual duplication and risk

- Legacy sandbox helpers such as [`sandboxMovement.ts`](../../../src/client/sandbox/sandboxMovement.ts) and [`sandboxPlacement.ts`](../../../src/client/sandbox/sandboxPlacement.ts) now act as **thin adapters** over shared helpers and are used primarily for UX and hypothetical-board tooling, not as independent semantics sources.
- Backend `RuleEngine` retains some historical helpers (e.g. line and territory post-processing methods), but decision surfaces for lines, territory, and elimination now defer to shared helpers like [`enumerateProcessLineMoves()`](../../../src/shared/engine/lineDecisionHelpers.ts) and [`enumerateTerritoryEliminationMoves()`](../../../src/shared/engine/territoryDecisionHelpers.ts).
- SSOT tooling such as [`rules-ssot-check.ts`](../../../scripts/ssot/rules-ssot-check.ts) enforces mapping consistency between [`RULES_CANONICAL_SPEC.md`](../../../RULES_CANONICAL_SPEC.md) and [`RULES_IMPLEMENTATION_MAPPING.md`](../../rules/RULES_IMPLEMENTATION_MAPPING.md), reducing the risk of silent divergence.
- Remaining risk is concentrated in **legacy orchestration paths and diagnostics-only helpers** that still exist but are no longer canonical; these should be gradually marked and then removed as part of the orchestrator rollout tasks in Section 8.

---

## 8. Remediation Roadmap (P16.x Tasks)

The following tasks are intended for follow-up passes, each with a short ID, recommended mode, and clear acceptance criteria.

| ID               | Title                                                             | Mode      | Inputs                                                                                                                                                                                                                                                                                            | Acceptance Criteria                                                                                                                                                                                                                                                                                                                                               |
| ---------------- | ----------------------------------------------------------------- | --------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **P16.1-ARCH**   | Finalise orchestrator-first migration plan                        | Architect | This PASS16 report; [`WEAKNESS_ASSESSMENT_REPORT.md`](../plans/WEAKNESS_ASSESSMENT_REPORT.md); [`docs/drafts/RULES_ENGINE_CONSOLIDATION_DESIGN.md`](../RULES_ENGINE_CONSOLIDATION_DESIGN.md); [`docs/architecture/ORCHESTRATOR_ROLLOUT_PLAN.md`](../../architecture/ORCHESTRATOR_ROLLOUT_PLAN.md) | **(Largely complete in this pass.)** `docs/architecture/ORCHESTRATOR_ROLLOUT_PLAN.md` now defines phases, flags, CI profiles, metrics, and rollback levers for orchestrator-first rollout; remaining architectural decisions relate to concrete legacy deletions and any future SLO tweaks, to be driven by P16.6–P16.8.                                          |
| **P16.2-CODE**   | Decompose `GamePage` into backend and sandbox hosts               | Code      | [`GamePage.tsx`](../../../src/client/pages/GamePage.tsx); [`ClientSandboxEngine.ts`](../../../src/client/sandbox/ClientSandboxEngine.ts); PASS16 scorecard                                                                                                                                        | `GamePage.tsx` reduced to < 800 LOC with separate `BackendGameHost` and `SandboxGameHost` components; all existing Jest and Playwright suites pass; no behavioural regressions in manual smoke tests.                                                                                                                                                             |
| **P16.3-CODE**   | Touch-first sandbox interaction improvements                      | Code      | `BoardView`, `GamePage`, sandbox components; PASS16 UX findings                                                                                                                                                                                                                                   | New touch-friendly flows for capture-direction selection and sandbox-specific placement are available and documented in‑UI; keyboard and mouse flows remain intact; at least one new component/unit test and one new E2E scenario cover touch-focused paths.                                                                                                      |
| **P16.4-CODE**   | Board controls & shortcut overlay                                 | Code      | `BoardView`, HUD components; PASS16 UX findings                                                                                                                                                                                                                                                   | A contextual help overlay is accessible from the main game UI (button and keyboard shortcut) summarising keyboard, mouse, and sandbox interaction patterns; basic Jest tests ensure it renders and lists key bindings; E2E test confirms discoverability.                                                                                                         |
| **P16.5-DOCS**   | Update `CANONICAL_ENGINE_API` and goals/test-count docs           | Docs      | [`docs/architecture/CANONICAL_ENGINE_API.md`](../../architecture/CANONICAL_ENGINE_API.md); [`PROJECT_GOALS.md`](../../../PROJECT_GOALS.md); `CURRENT_STATE_ASSESSMENT`                                                                                                                            | Placement helper sections describe helpers as canonical (not stubs) and reference current tests; numeric test-count examples in `PROJECT_GOALS` are updated or replaced with references to `CURRENT_STATE_ASSESSMENT`; SSOT banners remain correct.                                                                                                               |
| **P16.6-CODE**   | Orchestrator-first backend & sandbox default with legacy shutdown | Code      | `GameEngine`, [`RuleEngine`](../../../src/server/game/RuleEngine.ts), [`ClientSandboxEngine`](../../../src/client/sandbox/ClientSandboxEngine.ts), orchestrator adapters                                                                                                                          | All production and sandbox paths execute turns via `processTurnAsync` through adapters; legacy turn-processing code paths are removed or clearly marked diagnostics-only with zero production call sites; full Jest, Playwright, and parity suites pass; rules/AI/runbook docs (including `ORCHESTRATOR_ROLLOUT_PLAN`) remain in sync with the implemented paths. |
| **P16.7-QA**     | Expanded orchestrator parity & soak testing                       | QA        | Orchestrator adapters; contract vectors; `ai-service/tests/parity/**`; `tests/scenarios/**`                                                                                                                                                                                                       | Additional contract vectors and scenario tests are added for complex multi-phase sequences (deep capture chains, late-game territory and LPS cases); CI includes a dedicated "orchestrator parity" job running in the orchestrator‑ON profile; soak tests run for representative seeds without divergence or shadow mismatches.                                   |
| **P16.8-DEVOPS** | Orchestrator rollout staging & SLO-driven gating                  | DevOps    | CI workflows; monitoring configs; runbooks (`ORCHESTRATOR_ROLLOUT_RUNBOOK`, `RULES_PARITY`, `GAME_HEALTH`, `GAME_PERFORMANCE`, `AI_*`)                                                                                                                                                            | Staging environment runs with orchestrator-only mode under a documented rules-mode flag; dashboards and alerts monitor orchestrator metrics (`ringrift_orchestrator_*`), parity metrics, and move latency using the updated runbooks; production rollout requires at least one green staging bake-in period with SLOs met and explicit approval gated in CI.      |
| **P16.9-ARCH**   | Frontend architectural guidelines & patterns                      | Architect | Client architecture docs; PASS16 frontend findings                                                                                                                                                                                                                                                | A short client-architecture guideline document defines patterns for host vs presentation components, interaction handlers, and orchestrator integration; `GamePage` and sandbox controllers are brought into compliance; guidelines are referenced from `src/client/ARCHITECTURE.md`.                                                                             |

These tasks collectively address the **identified weakest area** (frontend host architecture) and the **hardest outstanding problem** (orchestrator-first rollout and legacy shutdown), while tightening documentation SSOT discipline and UX polish for future passes.
