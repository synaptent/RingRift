/**
 * ClientSandboxEngine orchestrator parity test (smoke)
 *
 * HISTORICAL NOTE: This test was originally designed to compare legacy sandbox
 * rules path vs orchestrator-delegated path. As of 2025-12-01 (Phase 3 migration),
 * the legacy path has been removed and the orchestrator is permanently enabled.
 *
 * This test now verifies that two independent sandbox engine instances produce
 * identical results when processing the same canonical Move sequence, confirming
 * determinism of the orchestrator-based rules processing.
 */

import {
  ClientSandboxEngine,
  SandboxConfig,
  SandboxInteractionHandler,
} from '../../src/client/sandbox/ClientSandboxEngine';
import type { BoardType, GameState, Move, Position } from '../../src/shared/types/game';
import { hashGameState } from '../../src/shared/engine/core';

const boardType: BoardType = 'square8';

function createSandboxConfig(): SandboxConfig {
  return {
    boardType,
    numPlayers: 2,
    playerKinds: ['human', 'human'],
  };
}

function createNoopInteractionHandler(): SandboxInteractionHandler {
  return {
    async requestChoice(_choice: any): Promise<any> {
      throw new Error('No choices expected in orchestrator parity smoke test');
    },
  };
}

function snapshot(state: GameState) {
  return {
    hash: hashGameState(state),
    currentPhase: state.currentPhase,
    currentPlayer: state.currentPlayer,
    gameStatus: state.gameStatus,
    moveCount: state.moveHistory.length,
  };
}

function canonicalEngineMove(move: Move): Omit<Move, 'id' | 'timestamp' | 'moveNumber'> {
  const { id, timestamp, moveNumber, ...rest } = move as any;
  return {
    ...rest,
    thinkTime: 0,
  };
}

describe('ClientSandboxEngine orchestrator determinism (short sequence)', () => {
  it('keeps two independent engines in lockstep for a short ring_placement + movement sequence on square8', async () => {
    const config = createSandboxConfig();
    const handler = createNoopInteractionHandler();

    // Both engines now use the orchestrator (permanently enabled)
    const engine1 = new ClientSandboxEngine({ config, interactionHandler: handler });
    const engine2 = new ClientSandboxEngine({ config, interactionHandler: handler });

    // Initial snapshots must match
    expect(snapshot(engine1.getGameState())).toEqual(snapshot(engine2.getGameState()));

    const maxSteps = 6;

    for (let i = 0; i < maxSteps; i++) {
      const state1Before = engine1.getGameState();
      const state2Before = engine2.getGameState();

      expect(snapshot(state2Before)).toEqual(snapshot(state1Before));

      if (state1Before.gameStatus !== 'active') {
        break;
      }

      const currentPlayer = state1Before.currentPlayer;
      const candidates = engine1.getValidMoves(currentPlayer);
      expect(candidates.length).toBeGreaterThan(0);

      // Deterministic choice: sort by type and coordinates and pick first
      const sorted = [...candidates].sort((a, b) => {
        if (a.type !== b.type) {
          return a.type < b.type ? -1 : 1;
        }
        const ax = a.to?.x ?? 0;
        const ay = a.to?.y ?? 0;
        const bx = b.to?.x ?? 0;
        const by = b.to?.y ?? 0;
        if (ay !== by) return ay - by;
        if (ax !== bx) return ax - bx;
        return 0;
      });

      const chosen = sorted[0];
      const engineMove = canonicalEngineMove(chosen);

      await engine1.applyCanonicalMove(engineMove as Move);
      await engine2.applyCanonicalMove(engineMove as Move);

      const state1After = engine1.getGameState();
      const state2After = engine2.getGameState();

      expect(snapshot(state2After)).toEqual(snapshot(state1After));

      if (state1After.gameStatus !== 'active') {
        break;
      }
    }
  });
});
