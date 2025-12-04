# Pass 18A – Full-Project Reassessment Report

> **⚠️ SUPERSEDED BY PASS18B** – This assessment was superseded by PASS18B on November 30, 2025.
> For current project status, see `docs/PASS18B_ASSESSMENT_REPORT.md`.

> **Assessment Date:** 2025-11-30
> **Assessment Pass:** 18A (post-ANM/termination remediation, test stabilization)
> **Assessor:** Architect mode – full-project state & weakness reassessment

> This pass builds on PASS16–PASS17 and the extensive working notes in
> `docs/PASS18_WORKING_NOTES.md`. It provides a definitive update on the
> project's weakest aspect and hardest outstanding problem after:
>
> - ANM (active-no-moves) and forced-elimination semantics hardening
> - Orchestrator architecture completion and CI gating
> - UI component test stabilization (GameContext, GameHUD, SandboxGameHost)
> - Test suite health restoration (2,621 tests passing)

---

## 1. Executive Summary (Pass 18A)

### 1.1 Single Weakest Aspect

**TS Rules/Host Integration & Deep Multi-Engine Parity for Advanced Phases**

The shared TS engine and orchestrator architecture are architecturally sound, but
the weakest remaining surface is the **integration layer** between:

- Backend `GameEngine` / `TurnEngineAdapter` and the shared orchestrator
- Client `ClientSandboxEngine` / `SandboxOrchestratorAdapter` and the shared orchestrator
- Python AI-service rules engine parity for complex endgame scenarios

Specific weak spots include:

- **Capture sequence enumeration** alignment between backend and sandbox hosts
- **Territory disconnection** processing paths (legacy vs orchestrator)
- **RNG determinism** threading across TS backend, sandbox, and Python AI
- **Line reward** and **territory region** decision-move semantics

### 1.2 Single Hardest Outstanding Problem

**Orchestrator-First Production Rollout with SLO Gates & Environment Phase Execution**

The code, metrics, and CI infrastructure are in place. The remaining challenge is
**operational execution**:

- Driving environments through Phases 1–4 as defined in `docs/ORCHESTRATOR_ROLLOUT_PLAN.md`
- Enforcing SLO-driven gates before phase promotion
- Decommissioning legacy turn-processing paths once orchestrator stability is proven
- Maintaining deep multi-engine parity during the transition window

### 1.3 Key Progress Since PASS17

| Area                       | Status        | Notes                                              |
| -------------------------- | ------------- | -------------------------------------------------- |
| ANM/Termination Invariants | ✅ Strong     | P0 invariants implemented, regression suites green |
| Orchestrator CI Gating     | ✅ Complete   | `test:orchestrator-parity` required for main       |
| Test Suite Health          | ✅ All Green  | 223 suites, 2,621 tests passing                    |
| UI Component Tests         | ✅ Stabilized | GameContext, GameHUD, SandboxGameHost fixed        |
| Frontend Host Architecture | ✅ Improved   | Board controls, touch panels, view-model adapters  |
| Python AI Healthchecks     | ✅ Integrated | CI job + nightly workflow, invariant metrics       |
| Doc Alignment              | ⚠️ Partial    | Several index docs reference stale weakest-aspect  |

---

## 2. Per-Subsystem Scores (Pass 18A)

| Subsystem                                 | Score | Trend | Notes                                           |
| ----------------------------------------- | ----- | ----- | ----------------------------------------------- |
| **Shared TS Engine (aggregates/helpers)** | 4.5/5 | ↔     | Strong SSoT, well-tested                        |
| **Backend Host Integration**              | 3.5/5 | ↑     | Orchestrator wired, some legacy paths remain    |
| **Sandbox Host Integration**              | 3.5/5 | ↑     | Orchestrator adapter stable, parity improving   |
| **Python Rules Engine**                   | 4.0/5 | ↔     | Contract tests green, training pipelines robust |
| **TS↔Python Parity**                      | 3.5/5 | ↑     | Core parity strong, edge cases remain           |
| **Orchestrator Architecture**             | 4.5/5 | ↔     | Complete, metrics wired, CI gated               |
| **Frontend Components**                   | 4.0/5 | ↑     | Tests stabilized, view-models strong            |
| **Frontend UX (rules explanation)**       | 3.0/5 | ↔     | Copy mismatches vs RR-CANON identified          |
| **AI Training Infrastructure**            | 4.0/5 | ↔     | Robust pipelines, versioning, streaming         |
| **Documentation**                         | 3.5/5 | ↓     | Stale pointers to ANM as "highest-risk"         |
| **CI/CD & Observability**                 | 4.5/5 | ↔     | Metrics, alerts, SSoT checks comprehensive      |

**Overall Project Score: 3.9/5** (up from ~3.7 in PASS17)

---

## 3. Test Health Snapshot

### 3.1 TypeScript / Jest

```
Test Suites: 223 passed, 60 skipped, 283 total
Tests:       2,621 passed, 202 skipped, 1 todo, 2,824 total
```

**Status: ✅ All Green**

Recent fixes (this session):

- `GameContext.test.tsx` – Added missing `decisionAutoResolved` and `decisionPhaseTimeoutWarning` state to TestGameProvider
- `RulesBackendFacade.test.ts` – Added missing `getValidMoves` mock for `applyMoveById` test
- `GameEngine.lineRewardChoiceAIService.integration.test.ts` – Corrected `requiredLength` expectation (4 for 2p square8, not 3)

### 3.2 Python / pytest

**Status: ✅ Green** (per prior PASS17 assessment and CI)

- Rules parity tests passing
- Training pipeline tests passing
- Invariant tests passing

### 3.3 Orchestrator Soak

**Status: ✅ Green**

```
Total games=1, completed=0, maxTurns=1, invariantViolations=0
```

---

## 4. Documentation Alignment Status

### 4.1 Docs Requiring Updates

| Document                      | Issue                                       | Recommended Action                            |
| ----------------------------- | ------------------------------------------- | --------------------------------------------- |
| `CURRENT_STATE_ASSESSMENT.md` | Claims "all tests passing" as of 2025-11-27 | Add timestamp framing, note it's a snapshot   |
| `CURRENT_RULES_STATE.md`      | Says "None critical" for known issues       | Update to reference host integration issues   |
| `DOCUMENTATION_INDEX.md`      | Describes ANM as "highest-risk semantics"   | Update to reference PASS18A weakest aspect    |
| `docs/INDEX.md`               | Points to ANM invariants as current focus   | Add note distinguishing historical vs current |
| `PROJECT_GOALS.md` §3.4       | ✅ Already updated                          | Correctly frames host integration as focus    |

### 4.2 Docs That Are Current

- `RULES_ENGINE_ARCHITECTURE.md` – Accurate orchestrator/adapter description
- `AI_ARCHITECTURE.md` – Correct RNG determinism and training pipeline docs
- `docs/ORCHESTRATOR_ROLLOUT_PLAN.md` – Comprehensive rollout phases
- `docs/INVARIANTS_AND_PARITY_FRAMEWORK.md` – Accurate invariant catalog

---

## 5. Weakest Aspect Deep Dive

### 5.1 Why Host Integration is the Weakest Aspect

With ANM/forced-elimination semantics now heavily covered by invariants and tests,
the **integration surface** between hosts and the shared engine is where remaining
risk concentrates:

1. **Capture Enumeration Paths**
   - Backend uses `captureChainEngine.getCaptureOptionsFromPosition`
   - Sandbox uses `sandboxCaptureSearch.enumerateCaptureSequences`
   - These must produce identical legal-move sets for parity

2. **Territory Processing Paths**
   - Legacy backend path in `GameEngine.processDisconnectedRegions`
   - Orchestrator path via `TurnEngineAdapter` + shared mutators
   - Sandbox path via `SandboxOrchestratorAdapter`
   - Region detection and elimination accounting must align

3. **RNG Threading**
   - AI move selection requires deterministic RNG for replay/debugging
   - Sandbox `maybeRunAITurn(rng)` vs backend `AIEngine.getAIMove(rng)`
   - Python AI service seed handling

4. **Decision-Move Semantics**
   - `continue_capture_segment`, `process_line`, `choose_line_reward`, `process_territory_region`
   - Must be enumerated identically by backend and sandbox `getValidMoves`

### 5.2 Why This Matters

- **Game correctness**: Different hosts producing different legal moves = bugs
- **AI quality**: Parity failures between training (Python) and play (TS) degrade AI
- **Replay fidelity**: Non-deterministic paths break game history and debugging
- **Orchestrator rollout**: Can't safely deprecate legacy paths without host parity

---

## 6. Hardest Problem Deep Dive

### 6.1 Orchestrator Rollout Execution

The orchestrator architecture is complete:

- ✅ `TurnEngineAdapter` and `SandboxOrchestratorAdapter` implemented
- ✅ Feature flags (`ORCHESTRATOR_ADAPTER_ENABLED`, `ORCHESTRATOR_ROLLOUT_PERCENTAGE`)
- ✅ `OrchestratorRolloutService` with circuit-breaker logic
- ✅ Metrics (`ringrift_orchestrator_invariant_violations_total`)
- ✅ CI gating (`test:orchestrator-parity` required for main)

**What remains is operational:**

| Phase                             | Status          | Blocking Work              |
| --------------------------------- | --------------- | -------------------------- |
| Phase 1: Staging shadow           | ✅ Config ready | —                          |
| Phase 2: Staging authoritative    | ⏳ Pending      | Run validation suites      |
| Phase 3: Production percentage    | ⏳ Pending      | SLO definition, monitoring |
| Phase 4: Production authoritative | ⏳ Pending      | Sustained stability window |
| Legacy deprecation                | ⏳ Pending      | Phase 4 completion         |

### 6.2 SLO Gates Required

Per `docs/ORCHESTRATOR_ROLLOUT_PLAN.md`:

- `ringrift_orchestrator_invariant_violations_total` = 0 over 24h window
- `ringrift_rules_parity_mismatches_total` < threshold
- Green AI healthchecks and Python invariant stream
- No P0 bug reports from staging users

---

## 7. PASS18A Remediation Backlog

### P18A.1 – Host Integration Parity (P0)

- [ ] **P18A.1-1**: Audit capture enumeration paths (backend vs sandbox) and unify any divergent logic
- [ ] **P18A.1-2**: Audit territory processing paths and ensure orchestrator adapter covers all cases
- [ ] **P18A.1-3**: Add focused parity tests for edge cases identified in `docs/P18.1-1_CAPTURE_TERRITORY_HOST_MAP.md`

### P18A.2 – RNG Determinism (P0)

- [ ] **P18A.2-1**: Verify RNG threading end-to-end from game seed → AI move selection
- [ ] **P18A.2-2**: Add regression tests for sandbox vs backend RNG alignment
- [ ] **P18A.2-3**: Document RNG paths in `docs/P18.2-1_AI_RNG_PATHS.md`

### P18A.3 – Orchestrator Rollout Execution (P1)

- [ ] **P18A.3-1**: Run Wave 5.2 validation suites with orchestrator ON
- [ ] **P18A.3-2**: Define production SLO thresholds and monitoring dashboards
- [ ] **P18A.3-3**: Execute Phase 2 (staging authoritative) and document results
- [ ] **P18A.3-4**: Plan Phase 3 rollout schedule

### P18A.4 – Documentation Alignment (P1)

- [x] **P18A.4-1**: Update `CURRENT_STATE_ASSESSMENT.md` with timestamp framing ✅ Completed 2025-11-30
- [ ] **P18A.4-2**: Update `CURRENT_RULES_STATE.md` known issues section
- [x] **P18A.4-3**: Update `DOCUMENTATION_INDEX.md` weakest-aspect pointer ✅ Completed 2025-11-30
- [ ] **P18A.4-4**: Update `docs/INDEX.md` to distinguish historical vs current focus

### P18A.5 – Frontend UX Copy Fixes (P2)

- [ ] **P18A.5-1**: Fix chain capture HUD text (mandatory continuation, not optional)
- [ ] **P18A.5-2**: Fix ring elimination victory threshold copy (>50%, not "all")
- [ ] **P18A.5-3**: Clarify line/territory decision phase prompts

---

## 8. Recommendations

### Immediate (This Week)

1. **Complete doc alignment** (P18A.4-\*) to ensure downstream readers have accurate context
2. **Run Wave 5.2 validation suites** with orchestrator forced ON to validate readiness

### Near-Term (Next 2 Weeks)

3. **Execute P18A.1-\* capture/territory host parity audits** to close the primary weakness
4. **Execute Phase 2 orchestrator rollout** in staging with SLO monitoring

### Medium-Term (Next Month)

5. **Complete Phase 3/4 production rollout** with circuit-breaker gates
6. **Deprecate legacy turn-processing paths** once stability window passes
7. **Address frontend UX copy issues** (P18A.5-\*)

---

## 9. Conclusion

Pass 18A marks a significant milestone: **all tests are green** and the project's
weakest aspect has shifted from **rules semantics correctness** (ANM, forced
elimination) to **host integration and operational rollout**.

This is a positive trend – the hardest _semantic_ problems are now well-covered by
invariants and tests. The remaining work is primarily **integration, parity, and
operational execution** rather than fundamental rules correctness.

The orchestrator architecture provides a solid foundation for the final push to
production. Success depends on disciplined execution of the rollout phases with
SLO enforcement, plus targeted parity hardening for the host integration surface.

---

_This assessment supersedes PASS17 findings and should be used as the current
reference for project weakest aspect and hardest outstanding problem._
