Original prompt: 1. Fix the production landing page (root should be public or go to /sandbox; fix CSP/font errors and repeated /api/games/sandbox/evaluate 404s). 2. Polish the sandbox game UI (clear AI difficulty labels, surface rules/help, improve mobile clarity, make Submit for Training feedback solid). 4. Run npm test and fix failures. Commit after each task. Do not touch training infrastructure or minimal_alphazero_loop.py.

2026-04-15

- Investigated frontend/server paths before editing.
- Confirmed public LandingPage already exists, but App.tsx defines two competing "/" routes.
- Confirmed sandbox evaluation endpoint is dev/test-only while client assumes availability.
- Confirmed client probes /api/games/sandbox/ai/health but server does not expose that route.
- Confirmed CSP errors come from Bunny Fonts + inline GA bootstrap in src/client/index.html.
- Confirmed developerToolsEnabled defaults true even though sandbox defaults to beginner mode, which can trigger evaluation requests for first-time visitors.

Next:

- Batch 1: route/CSP/sandbox-helper consistency fixes + focused tests.
- Batch 2: sandbox UX polish + browser verification.
- Batch 3: full npm test and any cleanup.

- Batch 1 edits applied: App root routing, CSP/bootstrap cleanup, sandbox AI helper gating, sandbox mode/devtools alignment.
- Next: focused tests + local browser verification before commit.

- Focused tests passed: App.authRouting, useSandboxEvaluation, securityHeaders.
- Browser check via local Playwright against Vite: root shows LandingPage; /sandbox opens directly with onboarding modal; no Bunny-font or inline-script CSP issues remained.
- Vite-only console still shows expected 500s for telemetry without the Express API; not treated as regressions for this batch.

- Batch 2 edits applied: visitor-facing AI labels, mobile preset layout, quick-rules help, guest launch copy, training submission availability note.
- Next: focused sandbox/UI tests and browser screenshots.

- Full npm test after landing/sandbox polish exposed 3 stale frontend expectations: difficultyUx, GameHUD snapshot, SandboxContext mode.
- SandboxContext mode expectations already updated to match beginner/debug developer-tools behavior.
- Next: patch difficultyUx + GameHUD snapshot, rerun focused slice, then rerun full npm test and commit the final test batch.

- Patched difficultyUx expectation to anchor D4 against the current recommended-audience wording.
- Updated the GameHUD snapshot to the new visitor-facing AI labels (Strong · D5 / Guided AI).

- GameHUD snapshot still had serialized whitespace drift; updating it with Jest to keep the snapshot authoritative.

- Full npm test rerun completed cleanly: 582 passed, 50 skipped, 11740 passed tests, 1 snapshot, no failing suites.
