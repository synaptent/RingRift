# PASS18 Working Notes – Reassessment & Planning

Last updated: 2025-11-30 (PASS18 in-progress).

## 1. Context and scope

- Task: PASS18-ARCH-REASSESS – full-project reassessment of weakest aspect, hardest outstanding problem, and doc/test freshness after ANM/termination remediation.
- Primary references so far:
  - [PROJECT_GOALS.md](PROJECT_GOALS.md:1)
  - [CURRENT_STATE_ASSESSMENT.md](../historical/CURRENT_STATE_ASSESSMENT.md:1)
  - [CURRENT_RULES_STATE.md](CURRENT_RULES_STATE.md:1)
  - [PASS16_ASSESSMENT_REPORT.md](docs/PASS16_ASSESSMENT_REPORT.md:1)
  - [PASS17_ASSESSMENT_REPORT.md](docs/PASS17_ASSESSMENT_REPORT.md:1)
  - [FINAL_ARCHITE_REPORT.md](archive/FINAL_ARCHITE_REPORT.md:1)
  - [AI_ARCHITECTURE.md](AI_ARCHITECTURE.md:1)
  - [RULES_ENGINE_ARCHITECTURE.md](RULES_ENGINE_ARCHITECTURE.md:1)
  - [DOCUMENTATION_INDEX.md](DOCUMENTATION_INDEX.md:1) and [docs/INDEX.md](docs/INDEX.md:1)
  - Jest test results from [jest-results.json](jest-results.json:1) (local run, many TS suites currently red).
  - Frontend code: [GameHUD.tsx](src/client/components/GameHUD.tsx:1), [VictoryModal.tsx](src/client/components/VictoryModal.tsx:1), [gameViewModels.ts](src/client/adapters/gameViewModels.ts:1), [ClientSandboxEngine.ts](src/client/sandbox/ClientSandboxEngine.ts:1).

## 2. Snapshot of prior passes (for comparison)

- **PASS16** ([docs/PASS16_ASSESSMENT_REPORT.md](docs/PASS16_ASSESSMENT_REPORT.md:1)):
  - Weakest area: **Frontend host architecture & UX ergonomics** – large, tightly coupled [GamePage.tsx](src/client/pages/GamePage.tsx:1) and [ClientSandboxEngine.ts](src/client/sandbox/ClientSandboxEngine.ts:1).
  - Hardest outstanding problem: **Orchestrator-first production rollout & legacy decommissioning**, focused on completing migration to shared turn orchestrator and removing legacy paths.

- **PASS17** ([docs/PASS17_ASSESSMENT_REPORT.md](docs/PASS17_ASSESSMENT_REPORT.md:1)):
  - Weakest area: **Deep rules parity & invariants for territory / chain-capture / endgames** across TS↔Python and hosts.
  - Hardest outstanding problem: **Operationalising orchestrator-first rollout with SLO gates and environment phases**, plus continued parity hardening for territory and combined-margin games.

- **FINAL_ARCHITE_REPORT** ([archive/FINAL_ARCHITE_REPORT.md](archive/FINAL_ARCHITE_REPORT.md:1)):
  - Confirms completion of major P0/P1 rules consolidation work (shared TS engine, unified Move model, contract tests) and establishes current canonical doc set.

## 3. Candidate weakest aspects (PASS18 – in-progress view)

### 3.1 Candidates considered

- **A. TS rules/host stack for advanced phases (capture/territory/chain-capture) + RNG parity**
  - Evidence: multiple failing Jest suites in current local run, including at least:
    - [captureSequenceEnumeration.test.ts](tests/unit/captureSequenceEnumeration.test.ts:1) – backend vs sandbox capture sequence parity.
    - [RefactoredEngine.test.ts](tests/unit/RefactoredEngine.test.ts:1) – capture, chain continuation, eliminate-stack behaviour.
    - [GameEngine.chainCapture.test.ts](tests/unit/GameEngine.chainCapture.test.ts:1) and [GameEngine.chainCaptureChoiceIntegration.test.ts](tests/unit/GameEngine.chainCaptureChoiceIntegration.test.ts:1) – complex chain-capture enforcement and choice enumeration.
    - [GameEngine.territoryDisconnection.test.ts](tests/unit/GameEngine.territoryDisconnection.test.ts:1) and [ClientSandboxEngine.territoryDisconnection.hex.test.ts](tests/unit/ClientSandboxEngine.territoryDisconnection.hex.test.ts:1) – territory disconnection and processing.
    - [Sandbox_vs_Backend.aiRngParity.test.ts](tests/unit/Sandbox_vs_Backend.aiRngParity.test.ts:1) and [Sandbox_vs_Backend.aiRngFullParity.test.ts](tests/unit/Sandbox_vs_Backend.aiRngFullParity.test.ts:1) – RNG-aligned sandbox vs backend AI move selection.
    - [RulesBackendFacade.test.ts](tests/unit/RulesBackendFacade.test.ts:1) – TS↔Python rules host behaviour in `python` and `shadow` modes.
  - Many of these tests previously passed in PASS16/17; clusters of failures now indicate instability or expectation drift around complex rules parity and host orchestration.

- **B. Orchestrator rollout & environment/SLO discipline**
  - Still hard operationally (per [docs/PASS17_ASSESSMENT_REPORT.md](docs/PASS17_ASSESSMENT_REPORT.md:1) and [docs/ORCHESTRATOR_ROLLOUT_PLAN.md](docs/ORCHESTRATOR_ROLLOUT_PLAN.md:1)), but architecture, metrics, and CI wiring are largely complete.

- **C. Frontend host architecture & UX**
  - PASS16 identified host code quality (especially [GamePage.tsx](src/client/pages/GamePage.tsx:1)) as weakest; PASS17 states host UX and component tests have **materially improved** and are no longer primary bottleneck.
  - Current audit of [GameHUD.tsx](src/client/components/GameHUD.tsx:1) and [gameViewModels.ts](src/client/adapters/gameViewModels.ts:1) confirms strong component structure but reveals **semantic mismatches** in player-facing text (chain capture optionality, victory thresholds).

- **D. AI training pipelines & dataset-level validation**
  - AI docs ([AI_ARCHITECTURE.md](AI_ARCHITECTURE.md:1), [docs/AI_TRAINING_AND_DATASETS.md](docs/AI_TRAINING_AND_DATASETS.md:1), [ai-service/AI_IMPROVEMENT_PLAN.md](ai-service/AI_IMPROVEMENT_PLAN.md:1)) describe remaining work (make/unmake pattern, hex training, dataset QA), but current tests under [ai-service/tests](ai-service/tests/test_engine_determinism.py:1) are broadly green.

- **E. ANM / forced-elimination semantics**
  - Previously top-risk semantics area ([PROJECT_GOALS.md](PROJECT_GOALS.md:1) §3.4, [docs/INVARIANTS_AND_PARITY_FRAMEWORK.md](docs/INVARIANTS_AND_PARITY_FRAMEWORK.md:1)), but now heavily covered by invariants, parity tests, and ANM-focused docs/tests.

### 3.2 Emerging weakest-aspect hypothesis

- With ANM semantics and orchestrator architecture substantially remediated, **A (TS rules/host stack for advanced phases + RNG parity)** currently looks like the **weakest aspect**:
  - It directly affects late-game correctness, fairness, and parity between TS backend, sandbox, and Python rules/AI.
  - Jest failures are concentrated in this area, suggesting either unstable expectations or real behavioural regressions.
  - Other candidate areas (B–E) are better documented, better tested, or primarily operational rather than semantic/behavioural.

## 4. Candidate hardest outstanding problems (PASS18 – in-progress view)

- **H1. Orchestrator-first rollout with SLO gates & environment phase execution**
  - From [docs/PASS17_ASSESSMENT_REPORT.md](docs/PASS17_ASSESSMENT_REPORT.md:1): design/metrics are in place, but executing Phases 1–4 across staging/production with SLO enforcement and legacy shutdown remains logistically hard.

- **H2. Deep multi-engine rules parity for territory / chain-capture / endgames (TS↔Python + hosts)**
  - Extends PASS17's weakest-area finding; still requires:
    - Expanded contract vectors and snapshot parity (especially for line+territory scenarios).
    - Stronger test coverage for edge geometries (hex, 3–4 player).
    - Keeping Python mutator-first evolution in lockstep with TS orchestrator semantics.

- **H3. AI training robustness & dataset-level validation**
  - Still non-trivial but largely decoupled from core game correctness; primary risk is AI quality, not rules correctness.

Current leaning: **H1+H2 together** describe the hardest outstanding problem: executing an orchestrator-first rollout while preserving deep multi-engine parity and invariants in complex endgames.

## 5. Documentation alignment notes (core docs audited so far)

- **[CURRENT_STATE_ASSESSMENT.md](../historical/CURRENT_STATE_ASSESSMENT.md:1)**
  - Status: **Snapshot accurate as of 2025-11-27, but now stale w.r.t. local test results.**
  - Claims: "1629+ TS tests, 245 Python tests" and "TypeScript tests: 1629+ tests passing" with all core suites green.
  - Current local Jest run shows multiple failing TS suites in advanced capture/territory/parity/AI RNG areas, so these "all passing" statements are no longer factually correct for the current workspace.
  - Recommendation (PASS18): treat this as a time-stamped snapshot; add language explicitly framing it as "status as of 2025-11-27" and avoid using it as a live indicator of test health. Cross-link to a lighter-weight "Current Test Health" note or CI dashboard if added.

- **[CURRENT_RULES_STATE.md](CURRENT_RULES_STATE.md:1)**
  - Quick status table reports **"Rules Engine: ✅ Fully implemented"** and **"Known Issues: None critical"**.
  - Given the current clusters of red Jest suites in rules/host parity (see §3.1), **"None critical"** is misleading; even if failures are partly due to expectation drift, they indicate active, unresolved semantics/test issues.
  - Recommendation (PASS18): update Known Issues summary to reference open advanced-phase/parity issues (e.g. capture sequence enumeration, some territory disconnection scenarios, RNG parity flakes), and cross-link to [KNOWN_ISSUES.md](KNOWN_ISSUES.md:1) and PASS18 report once written.

- **[AI_ARCHITECTURE.md](AI_ARCHITECTURE.md:1)**
  - Largely **current and internally consistent**: correctly positions the canonical rules SSoT (rules spec + shared TS implementation), AI difficulty ladder, RNG determinism, and training pipelines.
  - Section "Rules Completeness in AI Service" still describes some Python simplifications (auto-collapsing lines, simplified territory claim). Newer parity/equivalence tests in [ai-service/tests](ai-service/tests/test_engine_correctness.py:1) and [ai-service/AI_IMPROVEMENT_PLAN.md](ai-service/AI_IMPROVEMENT_PLAN.md:1) suggest some of these gaps have been reduced.
  - Recommendation: in PASS18 follow-up, re-audit Python rules completeness vs TS orchestrator using parity suites; if lines/territory are now fully parity-validated, soften or retire language that treats those simplifications as current behaviour.

- **[DOCUMENTATION_INDEX.md](DOCUMENTATION_INDEX.md:1)** and **[docs/INDEX.md](docs/INDEX.md:1)**
  - Both accurately describe the documentation map and SSoT layering.
  - They currently point to [PROJECT_GOALS.md](PROJECT_GOALS.md:1) and [docs/INVARIANTS_AND_PARITY_FRAMEWORK.md](docs/INVARIANTS_AND_PARITY_FRAMEWORK.md:1) as describing the **"current highest-risk rules semantics area"** (active-no-moves / forced elimination / territory & line disconnection).
  - Given ANM/termination work since those docs were written (new invariants, ANM regression suites, ACTIVE_NO_MOVES behaviour doc), and the emerging PASS18 view that advanced capture/territory parity in TS hosts is now weaker, this "current highest-risk" pointer is likely stale.
  - Recommendation: after PASS18 finalises a new weakest-aspect statement, update these index docs to reference that area instead (or to distinguish between "historical highest-risk semantics" vs "current weakest aspect").

- **[RULES_ENGINE_ARCHITECTURE.md](RULES_ENGINE_ARCHITECTURE.md:1)**
  - Appears **up-to-date** with shared TS engine, orchestrator, adapters, and Python parity mapping.
  - Explicitly frames Python as a host/adapter over the canonical rules SSoT (rules spec + shared TS implementation) and describes current contract/parity suites accurately.
  - No obvious contradictions with current code/tests; keep as-is, but reference PASS18 report for updated weakest-aspect and rollout-status commentary.

## 6. Test health snapshot (TS & Python – qualitative)

- **TypeScript / Jest (local run via [jest-results.json](jest-results.json:1))**
  - Many suites pass (core movement, basic placement, victory, components, contexts), but notable **failing clusters** include:
    - Advanced capture sequence enumeration and chain-capture enforcement (files listed in §3.1).
    - Territory disconnection and combined line+territory scenarios (e.g. [RulesMatrix.Territory.MiniRegion.test.ts](tests/scenarios/RulesMatrix.Territory.MiniRegion.test.ts:1), [GameEngine.territoryDisconnection.test.ts](tests/unit/GameEngine.territoryDisconnection.test.ts:1)).
    - Sandbox vs backend RNG parity and AI move-alignment (AI RNG smoke and full parity tests).
    - RulesBackendFacade python-mode semantics (extra `makeMove` calls vs expectations).
  - Interpretation: the underlying architecture and shared engine remain strong, but **TS host integration and advanced parity surfaces are currently unstable**.

- **Python / pytest (structure via [ai-service/tests](ai-service/tests/test_engine_determinism.py:1))**
  - Broad coverage of rules, parity, training, invariants, AI evaluation.
  - No large red clusters identified yet in this PASS18 view; prior AI docs ([ai-service/AI_ASSESSMENT_REPORT.md](ai-service/AI_ASSESSMENT_REPORT.md:1), [ai-service/AI_IMPROVEMENT_PLAN.md](ai-service/AI_IMPROVEMENT_PLAN.md:1)) mark state-copying, NN versioning, and hex training as main technical debts rather than correctness issues.

## 7. Frontend UX notes (preliminary PASS18 view)

- Source docs: [docs/supplementary/RULES_DOCS_UX_AUDIT.md](docs/supplementary/RULES_DOCS_UX_AUDIT.md:1), [src/client/adapters/gameViewModels.ts](src/client/adapters/gameViewModels.ts:1), [src/client/components/GameHUD.tsx](src/client/components/GameHUD.tsx:1), [src/client/components/VictoryModal.tsx](src/client/components/VictoryModal.tsx:1).
- Confirmed mismatches between RR-CANON and current HUD text include:
  - Chain capture text implying the player may "continue capturing or end your turn" when RR-CANON (and TS engine) require **mandatory continuation** if any capture exists.
  - Ring-elimination victory copy that still talks about "eliminating all opponent rings" while RR-CANON victory condition uses a **ringsPerPlayer eliminated rings** threshold.
  - Vague phrasing around line and territory decision phases that under-specifies when eliminations/self-eliminations are compulsory vs optional.
- UX quality and accessibility remain high at the component level (per PASS16 scores), but **rules-explanation UX** around advanced phases is still <4/5 and needs targeted copy/flow updates.

## 8. Next steps for PASS18 (for final report & remediation plan)

- Finalise **single weakest aspect** and **single hardest outstanding problem** selections, with explicit de-ranking of other candidates.
- Complete frontend UX reassessment (lobby/join/reconnect/spectator, in-game decision UX, error states) with updated 1–5 scores.
- Finish core-doc audit by adding notes for [PROJECT_GOALS.md](PROJECT_GOALS.md:1), [README.md](README.md:1), [RULES_CANONICAL_SPEC.md](RULES_CANONICAL_SPEC.md:1), and any SSOT-bannered rules docs not yet summarised here.
- Draft [docs/PASS18_ASSESSMENT_REPORT.md](docs/PASS18_ASSESSMENT_REPORT.md:1) with per-subsystem scores and updated weakest/hardest conclusions.
- Update [WEAKNESS_ASSESSMENT_REPORT.md](WEAKNESS_ASSESSMENT_REPORT.md:1) with a PASS18 section reflecting the new weakest aspect and hardest problem.
- Define a PASS18 remediation backlog (PASS18.\* tasks) across rules/host parity, orchestrator rollout execution, doc cleanup, test health, and frontend UX improvements.
- **P18.1-1 – Capture & Territory Host Path Diagnostic Map**
  - Created [`docs/P18.1-1_CAPTURE_TERRITORY_HOST_MAP.md`](docs/P18.1-1_CAPTURE_TERRITORY_HOST_MAP.md:1) as the local SSoT for host integration issues.
  - Identified key divergence in capture enumeration (sandbox vs backend) and legacy territory processing paths.
  - Mapped failing tests to specific host codepaths to guide P18.1 and P18.2 remediation.
- **P18.2-1 – AI RNG and Seed Handling Map**
  - Created [`docs/P18.2-1_AI_RNG_PATHS.md`](docs/P18.2-1_AI_RNG_PATHS.md:1) as the SSoT for RNG and seed-handling paths across TS hosts, sandbox, and Python AI/training.
  - Use this doc as the planning baseline for RNG-parity remediation tasks P18.2-2 and P18.2-3.

## 9. PASS18 second-pass doc alignment (additional notes)

- **[PROJECT_GOALS.md](PROJECT_GOALS.md:1)**
  - Confirms that the **current highest-risk semantics area** and **hardest outstanding problem** have already been updated to focus on **host integration & deep multi-engine parity** rather than ANM/forced-elimination alone (see §3.4).
  - This aligns cleanly with the PASS18 weakest-aspect/hardest-problem framing and effectively promotes the PASS18 view into the goals SSoT.
  - PASS18 implication: downstream index/overview docs that still describe `PROJECT_GOALS.md` as calling out ANM/forced elimination as the highest-risk semantics area are now **stale** relative to the goals SSoT and should be updated to match the new §3.4 framing.

- **[DOCUMENTATION_INDEX.md](DOCUMENTATION_INDEX.md:1)**
  - §2.1 still describes `PROJECT_GOALS.md` as including the “current highest-risk rules semantics area (active-no-moves / forced elimination / territory & line disconnection semantics)” (historical framing).
  - Given the updated §3.4 in `PROJECT_GOALS.md`, this pointer should be revised to either:
    - Refer to the **current** weakest aspect (“host integration & deep multi-engine parity”) in neutral language, or
    - Explicitly distinguish between **historical** highest-risk semantics (ANM/forced-elimination) and the **current** weakest aspect as tracked in [WEAKNESS_ASSESSMENT_REPORT.md](WEAKNESS_ASSESSMENT_REPORT.md:1) and [docs/PASS18_ASSESSMENT_REPORT.md](docs/PASS18_ASSESSMENT_REPORT.md:1).
  - PASS18 recommendation: treat this as a **doc alignment fix** rather than a re-framing of risk; `PROJECT_GOALS.md` remains the SSoT for direction, and index docs should simply describe what it now says.

- **[docs/INDEX.md](docs/INDEX.md:1)**
  - Quick links still highlight `docs/INVARIANTS_AND_PARITY_FRAMEWORK.md` as the place where “ruleset invariants & highest-risk semantics” are catalogued, emphasising **P0 ANM/forced-elimination invariants**.
  - This is still accurate for **invariants history**, but incomplete as a pointer to the **current weakest aspect**; readers now also need [WEAKNESS_ASSESSMENT_REPORT.md](WEAKNESS_ASSESSMENT_REPORT.md:1) and [docs/PASS18_ASSESSMENT_REPORT.md](docs/PASS18_ASSESSMENT_REPORT.md:1) for the updated host-integration & parity focus.
  - PASS18 recommendation: add a short note that ANM/forced-elimination invariants remain critical, but that the _current_ weakest aspect & hardest problem are tracked in the PASS18/weakness docs, to avoid confusion between “historically riskiest semantics” and “currently weakest aspect”.

## 10. Auth, lobby, and user flows (first-pass PASS18 notes)

Sources reviewed:

- Backend HTTP routes:
  - [src/server/routes/auth.ts](src/server/routes/auth.ts:1)
  - [src/server/routes/game.ts](src/server/routes/game.ts:1)
- Client connection/context:
  - [src/client/contexts/GameContext.tsx](src/client/contexts/GameContext.tsx:1)
- Existing WebSocket auth/connection tests (for reference):
  - [tests/unit/WebSocketServer.authRevocation.test.ts](tests/unit/WebSocketServer.authRevocation.test.ts:1)

### 10.1 Auth stack

- **Strengths:**
  - Password handling is modern and defensive: bcrypt-hashed passwords (`passwordHash` field), careful handling of missing/legacy hashes, and consistent use of `bcrypt.compare` with guardrails in [auth.ts](src/server/routes/auth.ts:1).
  - Refresh-token model is **rotating and family-scoped**:
    - All logins/registrations create a **token family** (`familyId`), with per-user single-active refresh token enforcement and hashed storage (`hashRefreshToken`) in Prisma’s `refreshToken` table.
    - `/auth/refresh` implements **reuse detection** and revokes the entire family on reuse, also incrementing per-user `tokenVersion` to invalidate access tokens.
  - Login lockout and rate limiting:
    - Per-email sliding-window lockout implemented over Redis (with in-memory fallback) plus route-level rate limiters for register/login/password reset.
    - Logging is careful to redact email via `redactEmail`.
  - Logout-all semantics:
    - `/auth/logout-all` increments `tokenVersion` and clears stored refresh tokens, providing a robust “sign out everywhere” operation consistent with the JWT contract.

- **Residual risks / complexity points (not primary PASS18 weaknesses):**
  - The refresh-token family model and mixed cookie/body handling add complexity that must be mirrored precisely in OpenAPI and client usage; however, this is **operational/security** complexity rather than rules/host integration risk.
  - Minimal direct coupling to rules/engine behaviour; auth incidents are unlikely to create semantic divergence between engines, so this area is **not a candidate** for “weakest aspect” or “hardest problem” despite its sophistication.

### 10.2 Game HTTP routes, lobby, and diagnostics

- **Game listing and details:** [game.ts](src/server/routes/game.ts:1)
  - `GET /games` and `GET /games/{gameId}` use Prisma includes to fetch player metadata and move history, with a shared `GameParticipantSnapshot` + `assertUserCanViewGame` invariant:
    - Enforces “participant or allowed spectator only” at the HTTP boundary, mirroring WebSocket access rules.
  - `GET /games/{gameId}/moves` and `/history` reuse the same participant/spectator invariant and shape move history with a compact `autoResolved` badge extracted from persisted `decisionAutoResolved` metadata, keeping HTTP history in sync with WebSocket decision auto-resolution semantics.

- **Game creation and lobby:**
  - `POST /games`:
    - Enforces **per-user and per-IP quotas** via `consumeRateLimit` and `adaptiveRateLimiter`, logging overruns with structured context.
    - Validates AI config and difficulty (1–10) and ensures **AI games are unrated**, aligning with `AI_ARCHITECTURE.md`’s expectation that AI-vs-human games are not used for ratings initially.
    - Persists `rngSeed` from the request into the DB (`game.rngSeed`), explicitly coercing `undefined` to `null`, which is important for AI RNG parity across TS and Python (seed as SSOT).
    - Broadcasts `lobby:game_created` via WebSocket server when a game is waiting for players, which is consistent with lobby expectations in client pages.
  - `POST /games/{gameId}/join`:
    - Enforces “waiting only” join semantics, detects already-joined players, and uses a simple next-slot heuristic (`player2Id` → `player3Id` → `player4Id`).
    - Broadcasts `lobby:game_joined` and, when threshold met, transitions to `status=active` and emits `lobby:game_started`, forming a clear server-side lobby state machine.
  - `GET /games/lobby/available`:
    - Returns joinable games filtered by `boardType` and `maxPlayers`, excluding any games where the current user is already a participant.

- **Leave / resign semantics:**
  - `POST /games/{gameId}/leave` uses `assertUserIsGameParticipant` to enforce seated-human-only mutations, but:
    - For **active** games, resignation is implemented as a simple DB status update (`status='completed'`) with a `TODO: Implement proper resignation logic` comment; rating changes and fine-grained LPS/victory semantics are not yet wired here.
    - For **waiting** games, behaviour is robust: players can leave; games with no players are marked `abandoned`; lobby broadcasts (`lobby:game_cancelled` or updated `playerCount`) are sent.
  - PASS18 implication:
    - HTTP-level resignation remains a **simplified path** relative to the full `GameEngine`/orchestrator semantics (no explicit host-integrated “resignation move” yet).
    - This is a **UX / fairness surface** to track (e.g. rated games, resign vs timeout semantics) but is not the current weakest aspect compared to deep rules/host integration for capture/territory & RNG parity.

- **Session diagnostics:**
  - `GET /games/{gameId}/diagnostics/session` exposes a concise view combining:
    - `GameSession` state-machine projections,
    - last AI-request state, and
    - current WebSocket connection state per player.
  - When the WebSocket server is not wired, returns a consistent “no in-memory session” shape instead of erroring, which is useful for runbooks and manual triage.

### 10.3 Client GameContext and network/error UX

- [GameContext.tsx](src/client/contexts/GameContext.tsx:1) acts as the **client-side host** for backend-driven games:
  - Maintains canonical `gameState`, `validMoves`, `pendingChoice`, `choiceDeadline`, `victoryState`, `chatMessages`, and `connectionStatus`.
  - Handles WebSocket messages:
    - `game_state` → hydrates `BoardState`/`GameState` from plain JSON and updates `validMoves`, `lastHeartbeatAt`, and `decisionAutoResolved` from `diffSummary.meta`.
    - `game_over` → finalizes `victoryState`, clears choices and decision metadata.
    - `choice_required` / `choice_canceled` → mirrors server `PlayerChoice` lifecycle with deadlines.
    - `decision_phase_timeout_warning` / `decision_phase_timed_out` → surfaces upcoming/actual auto-resolutions via `decisionPhaseTimeoutWarning` and resets it on timeout, ensuring HUD can reflect imminent AI/host decisions.
  - Error and disconnect handling:
    - `onError` displays toast errors and, when enabled, reports structured error payloads via `reportClientError`.
    - `onConnectionStatusChange` uses toast messaging for reconnect flows (“Reconnecting…” / “Reconnected!”) and maintains a `hasEverConnectedRef` flag to avoid false positives on initial connect.
    - `onDisconnect` clears heartbeat but does not automatically reset game state, allowing reconnect flows to restore state via new `game_state` payloads.

- PASS18 preliminary UX view (to be quantified later in the frontend UX scorecard):
  - **Strengths:** Good separation of concerns, explicit timeout/auto-resolution UX hooks, robust reconnection signalling hooks, and consistent mapping to WebSocket protocol.
  - **Gaps:** Some reconnect/spectator UX still depends on how pages and HUD components consume these signals; user-facing error copy and reconnect affordances likely remain **<4/5** and merit targeted UX tasks, but they are not comparable in risk to rules/host integration parity issues.

## 11. AI training and infra – PASS18 notes

Sources reviewed:

- Integration and component tests:
  - [ai-service/tests/integration/test_training_pipeline_e2e.py](ai-service/tests/integration/test_training_pipeline_e2e.py:1)
  - [ai-service/tests/test_engine_determinism.py](ai-service/tests/test_engine_determinism.py:1) and other `ai-service/tests/**` suites (previous passes).
- Training env and presets:
  - [ai-service/app/training/env.py](ai-service/app/training/env.py:1)

### 11.1 Training pipeline coverage and health

- `TestTrainingPipelineIntegration` and `TestComponentInteractions` provide **end-to-end and cross-component coverage**:
  - Square and hex training pipelines:
    - Use `StreamingDataLoader` to read synthetic NPZ datasets for square and hex boards, with configurable `policy_size` matching `RingRiftCNN` and `HexNeuralNet` expectations.
    - Train small test models (`SimpleSquareModel`, `SimpleHexModel`) for a few batches, then save versioned checkpoints via `ModelVersionManager` and register them in `AutoTournamentPipeline`.
  - Model versioning and registry:
    - `ModelVersionManager` tracks `architecture_version`, training metadata, and file checksums, with tests for:
      - Upgrade flows (`model_v1` → `model_v2`), parent checkpoint lineage, and checksum differences.
      - Error handling for checksum corruption (`ChecksumMismatchError`) and version mismatches (`VersionMismatchError`).
    - Tournament registry tests exercise Elo updates, challenger evaluation, and promotion logic, confirming **registry persistence** across pipeline instances.
  - Performance and memory:
    - Performance baseline tests validate minimum samples/sec throughput and streaming behaviour under larger synthetic datasets, with explicit tracemalloc-based peak-memory assertions.
    - Streaming vs dataset consistency tests confirm that `StreamingDataLoader` is semantically equivalent to `RingRiftDataset` for deterministic seeds.

- `TestErrorRecovery` and `TestPerformanceBaseline` demonstrate a **mature operational surface** for training:
  - Robust behaviour on missing/empty data sources (graceful handling of missing NPZ files, empty loader config).
  - Early-stopping behaviour and checkpoint resume capabilities (ensuring that training can be paused/resumed without losing best weights).

### 11.2 Training env and heuristic presets

- [env.py](ai-service/app/training/env.py:1) defines:
  - **DEFAULT_TRAINING_EVAL_CONFIG** and **TWO_PLAYER_TRAINING_PRESET**:
    - Multi-board, multi-start evaluation over `BoardType.SQUARE8`, `SQUARE19`, and `HEXAGONAL`.
    - `eval_mode="multi-start"`, `state_pool_id="v1"`, and controlled `eval_randomness` (0.0 baseline, 0.02 for the 2-player training preset) for symmetry breaking while preserving reproducibility.
    - These presets are the **SSoT for heuristic evaluation kwargs** (`games_per_eval`, `eval_randomness`, `seed`) used by CMA-ES/GA harnesses.
  - `TRAINING_HEURISTIC_EVAL_MODE_BY_BOARD` mapping:
    - Codifies a mixed “full” vs “light” evaluator policy per board type, with `SQUARE8` using full structural evaluation and `SQUARE19`/`HEXAGONAL` using lighter evaluators for throughput.
  - `RingRiftEnv` wrapper:
    - Provides an RL-style `reset`/`step`/`legal_moves` interface over the Python `GameEngine`, with board-type & player-count parameters.
    - `reset(seed=...)` threads the RNG seed through Python `random`, NumPy, and Torch, matching the broader RNG determinism strategy described in [AI_ARCHITECTURE.md](AI_ARCHITECTURE.md:1).
    - `step(...)` supports both terminal-only and shaped rewards via `calculate_outcome`, keeping **rules semantics deterministic** and confining randomness to sampling and training harnesses.

- PASS18 view:
  - Training and infra are **robust and well-tested**, with a strong emphasis on determinism, versioning, and error recovery.
  - Remaining risks are mostly:
    - **Scale/performance and operational cost**, and
    - Ensuring that training presets continue to match production rules/board configurations when rules evolve.
  - This area is **not a contender** for the single weakest aspect or hardest problem in PASS18; it is an important but well-contained domain with clear SSoT docs ([AI_ARCHITECTURE.md](AI_ARCHITECTURE.md:1), [docs/AI_TRAINING_AND_DATASETS.md](docs/AI_TRAINING_AND_DATASETS.md:1), [docs/AI_TRAINING_PREPARATION_GUIDE.md](docs/AI_TRAINING_PREPARATION_GUIDE.md:1)) and strong test coverage.

## 12. SSoT tooling – PASS18 second-pass notes

Sources reviewed:

- Aggregator: [scripts/ssot/ssot-check.ts](scripts/ssot/ssot-check.ts:1)
- Python parity SSoT check: [scripts/ssot/python-parity-ssot-check.ts](scripts/ssot/python-parity-ssot-check.ts:1)
- Docs banner SSoT check: [scripts/ssot/docs-banner-ssot-check.ts](scripts/ssot/docs-banner-ssot-check.ts:1)

### 12.1 Aggregated SSoT checks

- `ssot-check.ts` acts as a **single CLI entrypoint** that runs:
  - Rules SSoT checks (`runRulesSsotCheck`),
  - Lifecycle SSoT checks (`runLifecycleSsotCheck`),
  - Python parity SSoT checks (`runPythonParitySsotCheck`),
  - CI/config SSoT checks (`runCiAndConfigSsotCheck`),
  - Documentation/banner/link/env/secrets/API SSoT checks (`runDocsBannerSsotCheck`, `runEnvDocSsotCheck`, `runSecretsDocSsotCheck`, `runApiDocSsotCheck`, `runApiEndpointsSsotCheck`, `runDocsLinkSsotCheck`).
- The aggregator:
  - Catches thrown errors per-check and reports a structured `CheckResult` with `name`, `passed`, and `details`.
  - Summarises all checks as a simple PASS/FAIL table and exits non-zero if any check fails.
- PASS18 implication:
  - SSoT guardrails are **centralised and CI-friendly**; extending the SSoT surface for rules/host parity or orchestrator rollout is a matter of adding new checks to this list.

### 12.2 Python parity SSoT check (current state)

- [python-parity-ssot-check.ts](scripts/ssot/python-parity-ssot-check.ts:1) is intentionally **lightweight**:
  - Verifies the presence (via filesystem) of:
    - v2 contract vectors on the TS side (`tests/fixtures/contract-vectors/v2/*.vectors.json`),
    - TS contract runner (`tests/contracts/contractVectorRunner.test.ts`),
    - Python contract runner (`ai-service/tests/contracts/test_contract_vectors.py`), and
    - Parity requirements doc (`docs/PYTHON_PARITY_REQUIREMENTS.md`).
  - Returns a single `CheckResult` summarising whether all expected files exist.
- PASS18 view:
  - This check is valuable as a **drift guard** for critical parity artefacts (vectors, runners, doc) but does **not** yet verify:
    - That **new** parity artefacts (e.g. additional vectors, snapshot traces, or invariants) are reflected here, or
    - That the tests actually pass or are wired into CI.
  - Potential future work (P18.\* candidate):
    - Expand this check to:
      - Assert that key parity test suites are present in Jest/pytest configs, and
      - Optionally parse CI configs to ensure parity jobs cannot be silently removed.

### 12.3 Docs banner SSoT check

- [docs-banner-ssot-check.ts](scripts/ssot/docs-banner-ssot-check.ts:1) enforces:
  - Presence of an **SSoT banner** (by checking for `SSoT alignment`) in a curated list of core docs (e.g. [RULES_ENGINE_ARCHITECTURE.md](RULES_ENGINE_ARCHITECTURE.md:1), [RULES_IMPLEMENTATION_MAPPING.md](RULES_IMPLEMENTATION_MAPPING.md:1), [docs/RULES_ENGINE_SURFACE_AUDIT.md](docs/RULES_ENGINE_SURFACE_AUDIT.md:1), [docs/CANONICAL_ENGINE_API.md](docs/CANONICAL_ENGINE_API.md:1), [AI_ARCHITECTURE.md](AI_ARCHITECTURE.md:1), [docs/PYTHON_PARITY_REQUIREMENTS.md](docs/PYTHON_PARITY_REQUIREMENTS.md:1), [ARCHITECTURE_ASSESSMENT.md](ARCHITECTURE_ASSESSMENT.md:1), [ARCHITECTURE_REMEDIATION_PLAN.md](ARCHITECTURE_REMEDIATION_PLAN.md:1), [docs/MODULE_RESPONSIBILITIES.md](docs/MODULE_RESPONSIBILITIES.md:1)).
  - Presence of a **category-specific snippet** per doc (e.g. `Rules/invariants semantics SSoT`, `Lifecycle/API SSoT`, `Operational SSoT`, `rules semantics SSoT`, `Canonical TS rules surface`) as defined in [docs/SSOT_BANNER_GUIDE.md](docs/SSOT_BANNER_GUIDE.md:1).
- Behaviour:
  - If a monitored doc is missing or lacks the required snippet, the check fails with a descriptive message.
  - The implementation currently allows minor deviations in wording/structure as long as the key substrings are present, which matches the guidance in `SSOT_BANNER_GUIDE.md`.
- PASS18 view:
  - SSoT banners are **consistently present** in the reviewed docs; this reduces the risk that future edits silently demote canonical semantics or architecture docs to “just another markdown”.
  - For PASS18, the main alignment work is **semantic**: ensuring that the _content_ of these SSoT-aligned docs (especially `PROJECT_GOALS.md`, [`../historical/CURRENT_STATE_ASSESSMENT.md`](../historical/CURRENT_STATE_ASSESSMENT.md), `CURRENT_RULES_STATE.md`, and the indices) matches the latest weakest-aspect/hardest-problem framing and test reality, without weakening the SSoT structure itself.
