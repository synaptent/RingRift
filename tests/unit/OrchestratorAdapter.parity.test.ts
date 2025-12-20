/**
 * Orchestrator adapter parity checks
 *
 * Ensures the backend TurnEngineAdapter and client SandboxOrchestratorAdapter
 * advance through identical phases for the same canonical move sequence.
 */

import { createInitialGameState } from '../../src/shared/engine/initialState';
import type { GameState, Move, Player, TimeControl } from '../../src/shared/types/game';
import {
  createAutoSelectDecisionHandler,
  createSimpleAdapter,
} from '../../src/server/game/turn/TurnEngineAdapter';
import { createSandboxAdapter } from '../../src/client/sandbox/SandboxOrchestratorAdapter';
import { createTestPlayer } from '../utils/fixtures';

describe('Orchestrator adapter parity', () => {
  const timeControl: TimeControl = { initialTime: 600, increment: 0, type: 'blitz' };

  function createBaseState(gameId: string, rngSeed: number): GameState {
    const players: Player[] = [createTestPlayer(1), createTestPlayer(2)];
    const state = createInitialGameState(gameId, 'square8', players, timeControl, true, rngSeed);
    state.gameStatus = 'active';
    return state;
  }

  it('keeps server and sandbox phase progression aligned for placement moves', async () => {
    const serverState = createBaseState('adapter-parity-server', 101);
    const sandboxState = createBaseState('adapter-parity-sandbox', 101);

    const { adapter: serverAdapter } = createSimpleAdapter(
      serverState,
      createAutoSelectDecisionHandler()
    );

    let sandboxCurrentState = sandboxState;
    const sandboxAdapter = createSandboxAdapter(
      () => sandboxCurrentState,
      (state) => {
        sandboxCurrentState = state;
      },
      {
        requestDecision: async (decision) => {
          if (decision.options.length === 0) {
            throw new Error(`No options for decision: ${decision.type}`);
          }
          return decision.options[0];
        },
      }
    );

    const move: Move = {
      id: 'place-1',
      type: 'place_ring',
      player: 1,
      to: { x: 0, y: 0 },
      timestamp: new Date(),
      thinkTime: 0,
      moveNumber: 1,
    };

    const [serverResult, sandboxResult] = await Promise.all([
      serverAdapter.processMove(move),
      sandboxAdapter.processMove(move),
    ]);

    expect(serverResult.success).toBe(true);
    expect(sandboxResult.success).toBe(true);

    const serverNext = serverResult.nextState;
    const sandboxNext = sandboxResult.nextState;

    expect(sandboxNext.currentPhase).toBe(serverNext.currentPhase);
    expect(sandboxNext.currentPlayer).toBe(serverNext.currentPlayer);
    expect(sandboxNext.moveHistory.length).toBe(serverNext.moveHistory.length);
    expect(sandboxNext.history.length).toBe(serverNext.history.length);
    expect(sandboxNext.moveHistory[0]?.type).toBe(serverNext.moveHistory[0]?.type);
  });
});
