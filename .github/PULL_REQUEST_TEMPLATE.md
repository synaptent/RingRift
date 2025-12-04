## Summary

<!-- Briefly describe what this PR changes and why. -->

## Type of change

- [ ] Rules / engine (`src/shared/engine/**`, RulesMatrix, FAQ scenarios)
- [ ] Backend host / WebSocket / GameSession (`src/server/game/**`, `src/server/websocket/**`)
- [ ] Client sandbox / AI (`src/client/sandbox/**`)
- [ ] AI service / Python
- [ ] UI / UX only
- [ ] Documentation / meta only

## Checklist

### For rules / AI / WebSocket changes

If you checked any of the first three boxes above, please run the P0 robustness profile locally before requesting review:

- [ ] `npm run test:p0-robustness`

This runs:

- `npm run test:ts-rules-engine` (shared-engine + orchestrator rules suites: RulesMatrix, FAQ, advanced turn/territory helpers)
- `npm run test:ts-integration` (backend/WebSocket/full game-flow integration)
- A focused parity/cancellation bundle:
  - `tests/contracts/contractVectorRunner.test.ts` (v2 contract vectors, including forced-elimination and territory-line endgames)
  - `tests/parity/Backend_vs_Sandbox.CaptureAndTerritoryParity.test.ts` (advanced capture + single-/multi-region line+territory backend↔sandbox parity)
  - `tests/unit/WebSocketServer.sessionTermination.test.ts` (WebSocket session termination + decision/AI cancellation, including AI-backed `region_order` and `line_reward_option` flows)

### General

- [ ] Tests pass locally for the areas I changed (or I have explained why not).
- [ ] I have updated relevant documentation (rules/architecture/test meta-docs) where needed.
