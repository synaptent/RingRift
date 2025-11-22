import { readFileSync } from 'fs';
import { join } from 'path';
import { GameState, Move } from '../../src/shared/types/game';
import { PythonRulesClient } from '../../src/server/services/PythonRulesClient';

/**
 * Optional live-integration test that exercises the real Python AI-service
 * /rules/evaluate_move endpoint via PythonRulesClient using a canonical
 * TS-generated rules-parity fixture.
 *
 * This test is intentionally:
 *   - SKIPPED by default unless RINGRIFT_PYTHON_RULES_HTTP_INTEGRATION is set
 *     to "1" or "true" AND AI_SERVICE_URL is provided.
 *   - Focused on end-to-end HTTP contract and invariant parity, not on
 *     detailed engine semantics (which are covered by the fixture-based
 *     parity suites in both TS and Python).
 *
 * To run locally against a live AI-service instance:
 *
 *   AI_SERVICE_URL=http://localhost:8001 \\
 *   RINGRIFT_PYTHON_RULES_HTTP_INTEGRATION=1 \\
 *   npm test -- PythonRulesClient.live.integration.test.ts
 */

const ENABLED =
  (process.env.RINGRIFT_PYTHON_RULES_HTTP_INTEGRATION === '1' ||
    process.env.RINGRIFT_PYTHON_RULES_HTTP_INTEGRATION === 'true' ||
    process.env.RINGRIFT_PYTHON_RULES_HTTP_INTEGRATION === 'TRUE') &&
  typeof process.env.AI_SERVICE_URL === 'string' &&
  process.env.AI_SERVICE_URL.length > 0;

const FIXTURE_PATH = join(
  __dirname,
  '..',
  'fixtures',
  'rules-parity',
  'v1',
  'state_action.square8_2p.place_ring_center.json'
);

interface OutcomeSnapshot {
  stateHash: string;
  S: number;
  gameStatus: string;
}

interface StateActionFixture {
  state: GameState;
  move: Move;
  expected: {
    tsValid: boolean;
    tsNext?: OutcomeSnapshot;
  };
}

(ENABLED ? describe : describe.skip)(
  'PythonRulesClient live HTTP parity against TS fixture',
  () => {
    const baseUrl = process.env.AI_SERVICE_URL as string;

    it('replays a TS-generated placement fixture through /rules/evaluate_move', async () => {
      const raw = readFileSync(FIXTURE_PATH, 'utf-8');
      const fixture = JSON.parse(raw) as StateActionFixture;

      const state = fixture.state as GameState;
      const move = fixture.move as Move;

      const client = new PythonRulesClient(baseUrl);
      const result = await client.evaluateMove(state, move);

      // The TS fixture encodes the shared-engine verdict as tsValid.
      expect(result.valid).toBe(fixture.expected.tsValid);

      if (!fixture.expected.tsValid) {
        // When TS considered the move invalid, the Python endpoint should also
        // treat it as invalid. In that case we do not expect a nextState.
        expect(result.nextState).toBeUndefined();
        return;
      }

      expect(result.nextState).toBeDefined();

      const tsNext = fixture.expected.tsNext;
      if (!tsNext) {
        // If the fixture did not encode a tsNext snapshot, we only assert
        // basic validity and the presence of nextState.
        return;
      }

      if (tsNext.stateHash) {
        expect(result.stateHash).toBe(tsNext.stateHash);
      }

      if (typeof tsNext.S === 'number') {
        expect(result.sInvariant).toBe(tsNext.S);
      }

      if (tsNext.gameStatus) {
        expect(result.gameStatus).toBe(tsNext.gameStatus);
      }
    });
  }
);
