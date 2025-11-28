/**
 * ClientSandboxEngine orchestrator vs legacy parity (smoke)
 *
 * These tests run a short, deterministic sequence of canonical Moves through
 * two ClientSandboxEngine instances:
 *   - one using the legacy sandbox rules path, and
 *   - one delegating to SandboxOrchestratorAdapter (shared orchestrator).
 *
 * For the chosen scenario, the final GameState hashes and key fields should
 * match, providing a fast signal that the sandbox host stays in sync with the
 * shared orchestrator.
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

describe('ClientSandboxEngine orchestrator vs legacy parity (short sequence)', () => {
  it('keeps states in lockstep for a short ring_placement + movement sequence on square8', async () => {
    const config = createSandboxConfig();
    const handler = createNoopInteractionHandler();

    const legacy = new ClientSandboxEngine({ config, interactionHandler: handler });
    legacy.disableOrchestratorAdapter();

    const orchestrator = new ClientSandboxEngine({ config, interactionHandler: handler });
    orchestrator.enableOrchestratorAdapter();

    // Initial snapshots must match
    expect(snapshot(orchestrator.getGameState())).toEqual(snapshot(legacy.getGameState()));

    const maxSteps = 6;

    for (let i = 0; i < maxSteps; i++) {
      const legacyStateBefore = legacy.getGameState();
      const orchStateBefore = orchestrator.getGameState();

      expect(snapshot(orchStateBefore)).toEqual(snapshot(legacyStateBefore));

      if (legacyStateBefore.gameStatus !== 'active') {
        break;
      }

      const currentPlayer = legacyStateBefore.currentPlayer;
      const candidates = legacy.getValidMoves(currentPlayer);
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

      await legacy.applyCanonicalMove(engineMove as Move);
      await orchestrator.applyCanonicalMove(engineMove as Move);

      const legacyAfter = legacy.getGameState();
      const orchAfter = orchestrator.getGameState();

      expect(snapshot(orchAfter)).toEqual(snapshot(legacyAfter));

      if (legacyAfter.gameStatus !== 'active') {
        break;
      }
    }
  });
});
