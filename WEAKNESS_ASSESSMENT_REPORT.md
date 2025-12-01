# RingRift Weakness Assessment & Hardest Problems Report

**Last Updated:** 2025-11-30 (Pass 18A)
**Status:** Active

This document tracks the project's **single weakest aspect** and **single hardest outstanding problem** over time. It serves as a high-level risk register to focus architectural and remediation efforts.

---

## 1. Current Assessment (Pass 18A)

### 1.1 Weakest Aspect: TS Rules/Host Integration & Parity

**Score: 3.5/5.0** (Lowest Component Score)

Second-pass review confirms that the weakest aspect is the **TypeScript rules and host stack for advanced phases and RNG parity**, decomposed into three tightly-related sub-areas that all touch RR‑CANON capture, line, territory, and Q23 elimination rules in [`RULES_CANONICAL_SPEC.md`](RULES_CANONICAL_SPEC.md:1) and invariants in [`docs/INVARIANTS_AND_PARITY_FRAMEWORK.md`](docs/INVARIANTS_AND_PARITY_FRAMEWORK.md:1):

1. **Capture and territory host-integration parity.**
   - The shared TS orchestrator and helper aggregates are strong, but wiring into backend host [`GameEngine.ts`](src/server/game/GameEngine.ts:1), host orchestration in [`GameSession.ts`](src/server/game/GameSession.ts:1), client sandbox [`ClientSandboxEngine.ts`](src/client/sandbox/ClientSandboxEngine.ts:1), and Python adapters [`RulesBackendFacade.ts`](src/server/game/RulesBackendFacade.ts:1) and [`PythonRulesClient.ts`](src/server/services/PythonRulesClient.ts:1) remains brittle for long multi-phase turns (capture → line → territory → elimination).
   - Failing suites such as `captureSequenceEnumeration.test.ts`, `GameEngine.chainCapture*.test.ts`, `GameEngine.territoryDisconnection.test.ts`, `ClientSandboxEngine.territoryDisconnection*.test.ts`, and `RulesMatrix.Territory.*.test.ts` show that backend and sandbox can still disagree on valid capture chains and territory outcomes in complex positions.

2. **AI RNG parity across hosts and engines.**
   - [`AIEngine.ts`](src/server/game/ai/AIEngine.ts:1) and [`AIServiceClient.ts`](src/server/services/AIServiceClient.ts:1) are designed to thread deterministic seeds from `gameState.rngSeed` into both the Python AI service and TS local heuristics, and there are strong unit tests (`AIEngine.fallback`, `AIServiceClient.concurrency`).
   - In real games, the primary AI paths now consistently derive their RNG from `gameState.rngSeed`: `GameSession.maybePerformAITurn` and `AIEngine` both construct seeded `LocalAIRng` instances, and the sandbox AI uses a per-game `SeededRNG`. The dedicated RNG parity suites (`Sandbox_vs_Backend.aiRngParity*.test.ts`, `Sandbox_vs_Backend.aiRngFullParity.test.ts`) and fallback/concurrency tests are currently green, but this remains a subtle area whenever new AI behaviours or hosts are added.

3. **Host edge semantics and decision lifecycles.**
   - Decision-phase timeouts, auto-resolutions, reconnection handling, and resignation or leave semantics route through [`WebSocketInteractionHandler.ts`](src/server/game/WebSocketInteractionHandler.ts:1), [`server.ts`](src/server/websocket/server.ts:1), [`GameSession.ts`](src/server/game/GameSession.ts:1), and HTTP routes such as [`game.ts`](src/server/routes/game.ts:1).
   - The design is robust and well-instrumented, but end-to-end coverage for rare edge cases (e.g. decision timeouts during reconnect, HTTP-level resignations for rated games) is thinner, and misalignment here can interact with late-game rules parity and perceived fairness.

**Why it is the weakest:**
Unlike ANM semantics (now heavily remediated) or frontend UX (mostly semantic/presentational), these integration issues strike at the **correctness and fairness of complex endgames**. They sit at the intersection of rules semantics, host orchestration, AI behaviour, and user-visible outcomes, and they are where the remaining red Jest clusters are most concentrated.

### 1.2 Hardest Outstanding Problem: Orchestrator-first Rollout & Deep Multi-engine Parity

**Difficulty: 5/5** (Operational Execution & Verification)

The architectural design for the shared turn orchestrator and parity framework is complete and implemented via [`TurnEngineAdapter.ts`](src/server/game/turn/TurnEngineAdapter.ts:1) and host wiring in [`GameEngine.ts`](src/server/game/GameEngine.ts:1). The **hardest outstanding problem** is executing an **orchestrator-first rollout under live load** while maintaining deep semantic parity across engines:

- **Rollout execution:** Driving the phases in [`docs/ORCHESTRATOR_ROLLOUT_PLAN.md`](docs/ORCHESTRATOR_ROLLOUT_PLAN.md:1) through staging and production, using the runbook in [`docs/runbooks/ORCHESTRATOR_ROLLOUT_RUNBOOK.md`](docs/runbooks/ORCHESTRATOR_ROLLOUT_RUNBOOK.md:1), with clear entry/exit criteria and rollbacks.
- **Deep multi-engine parity:** Achieving and sustaining parity across TS shared engine, TS hosts (backend + sandbox), and Python rules/AI for the long tail of complex scenarios (hex boards, multi-player games, combined line+territory events), using contract vectors, snapshot traces, and invariants.
- **Signal interpretation:** Reliably reading parity metrics (`ringrift_rules_parity_mismatches_total`), S-invariant counters, ACTIVE_NO_MOVES regression suites, and AI health checks to make safe go/no-go decisions during rollout.

**Why it is hard:**
1.  **Execution vs. design:** Moving from “we have a complete design” to “we are running orchestrator-first in staging/production with green SLOs” requires sustained operational discipline, careful coordination across services, and tight adherence to the rollout runbook.
2.  **Deep parity tail:** Achieving practical 100% parity for long-tail scenarios (e.g. hex territory disconnection with simultaneous line formation and forced elimination) requires both broader test vectors and long-running soaks, and parity must be maintained as rules evolve.
3.  **Legacy drag and dual paths:** Legacy engine paths and configuration flags must remain correct until final shutdown, increasing complexity and the risk of configuration drift.

---

## 2. Remediation Plan (Pass 18A)

The detailed, task-level remediation backlog is maintained in [`docs/PASS18A_ASSESSMENT_REPORT.md`](docs/PASS18A_ASSESSMENT_REPORT.md:1) §7. At a high level it is organised into three streams that directly address the weakest aspect and hardest problem identified above.

**Pass 18A Progress Note:** All tests are now green (223 suites, 2,621 tests passing). Recent fixes stabilized UI component tests (GameContext, GameHUD, SandboxGameHost) and corrected test expectations for line reward collapse semantics.

### P0 (Critical) – Stabilise host integration & parity (weakest aspect)

- **P18.1-CODE – Capture & chain-capture host parity**
  Stabilise capture sequence enumeration and chain-capture enforcement across backend host, client sandbox, and Python parity harness, focusing on the suites listed in §1.1 and the capture rules in [`RULES_CANONICAL_SPEC.md`](RULES_CANONICAL_SPEC.md:1).

- **P18.2-CODE – Territory & elimination integration**
  Correct ordering and completeness of line rewards, territory collapse, and Q23 self-elimination in host integration code so that TS hosts and Python rules agree on outcomes for complex long-turn scenarios.

- **P18.3-CODE – AI RNG parity & seeded behaviour**
  Ensure that AI move selection consistently threads `rngSeed` or equivalent RNG state through TS hosts and Python AI, eliminating reliance on `Math.random()` in host paths where deterministic behaviour is required for parity (see `Sandbox_vs_Backend.aiRngParity*.test.ts`).

- **P18.8-DEBUG – Decision lifecycle edge cases**
  Harden timeout, auto-resolution, reconnection, and resignation flows at the host and WebSocket layers so that late-game outcomes remain consistent with RR‑CANON rules and invariants even in degraded network conditions.

### P1 (Important) – Orchestrator rollout & deep multi-engine parity (hardest problem)

- **P18.4-DEVOPS – Execute Phase 1 in staging**
  Enable the orchestrator adapter in staging with SLO gating and parity monitoring, following the runbook, and keep it live for a defined soak period with no unacceptable parity or SLO regressions.

- **P18.5-ARCH – Expand contract vectors & parity artefacts**
  Add contract vectors and parity fixtures that specifically cover long-tail multi-phase interactions (hex territory disconnection, simultaneous line+territory, 3–4 player captures), and ensure both TS and Python runners, as well as SSOT parity checks, treat these artefacts as first-class.

- **P18.9-DEBUG – Orchestrator parity soaks & debugging harness**
  Run and monitor orchestrator-first soaks under representative load, with tooling to capture and triage any parity mismatches, invariant violations, or AI divergence discovered during rollout.

### P2 (Valuable) – Docs, SSOT, and UX alignment

- **P18.6-CODE – HUD & victory copy alignment**
  Update HUD and modal text for chain capture, victory thresholds, and decision phases to match RR‑CANON, reducing user confusion around the advanced semantics that are already hard to reason about on the backend.

- **P18.7-ARCH – Core docs clean-up & indexing alignment**
  Bring `CURRENT_STATE_ASSESSMENT.md`, `CURRENT_RULES_STATE.md`, `DOCUMENTATION_INDEX.md`, and `docs/INDEX.md` into alignment with `PROJECT_GOALS.md` and this report, clearly distinguishing historical highest-risk semantics (ANM/forced elimination) from the current weakest aspect and hardest problem.

- **P18.10-ARCH – SSOT & CI guardrails for parity and rollout**
  Extend SSoT checks (for example, `python-parity-ssot-check` and `lifecycle-ssot-check`) and CI wiring so that key parity suites and orchestrator rollout configs cannot silently drift out of the pipeline.

---

## 3. Historical Assessments

### Pass 18 (2025-11-30, superseded by 18A)

- **Weakest Area:** TS Rules/Host Integration & Parity (initial assessment).
- **Hardest Problem:** Orchestrator-first rollout & deep multi-engine parity.
- **Note:** PASS18A confirmed findings and added test stabilization progress.

### Pass 17 (2025-11-30)

- **Weakest Area:** Deep rules parity & invariants for territory / chain-capture / endgames.
- **Hardest Problem:** Operationalising orchestrator-first rollout with SLO gates.

### Pass 16 (2025-11-28)

- **Weakest Area:** Frontend host architecture & UX ergonomics.
- **Hardest Problem:** Orchestrator-first production rollout & legacy decommissioning.

### Pass 7 (2025-11-27)

- **Weakest Area:** DevOps/CI Enforcement (3.4/5.0).
- **Hardest Problem:** Orchestrator Production Rollout.
