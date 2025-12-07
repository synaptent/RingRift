import type { SandboxAIHooks } from '../../src/client/sandbox/sandboxAI';
import { maybeRunAITurnSandbox } from '../../src/client/sandbox/sandboxAI';
import type { GameState, Move } from '../../src/shared/types/game';
import { createTestGameState } from '../utils/fixtures';

describe('sandboxAI forced_elimination phase', () => {
  it('applies a forced_elimination move when available for the current AI player', async () => {
    const baseState: GameState = createTestGameState({ boardType: 'square8' });

    let currentState: GameState = {
      ...baseState,
      players: baseState.players.map((p) =>
        p.playerNumber === 1 ? { ...p, type: 'ai' } : { ...p, type: 'human' }
      ),
      currentPlayer: 1,
      currentPhase: 'forced_elimination',
      gameStatus: 'active',
    };

    const forcedEliminationCandidate: Move = {
      id: 'fe-1',
      type: 'forced_elimination',
      player: 1,
      moveNumber: 1,
      timestamp: new Date(),
      thinkTime: 0,
    } as Move;

    const applyCanonicalMove = jest.fn(async (_move: Move) => {
      // No-op stub: this test only asserts that maybeRunAITurnSandbox
      // selects and attempts to apply a forced_elimination move.
    });

    let lastAIMove: Move | null = null;

    const hooks: SandboxAIHooks = {
      getPlayerStacks: () => [],
      hasAnyLegalMoveOrCaptureFrom: () => false,
      enumerateLegalRingPlacements: () => [],
      getValidMovesForCurrentPlayer: () => [forcedEliminationCandidate],
      createHypotheticalBoardWithPlacement: (board) => board,
      tryPlaceRings: async () => false,
      enumerateCaptureSegmentsFrom: () => [],
      enumerateSimpleMovementLandings: () => [],
      maybeProcessForcedEliminationForCurrentPlayer: () => false,
      handleMovementClick: async () => {
        // no-op
      },
      appendHistoryEntry: () => {
        // no-op for this focused test
      },
      getGameState: () => currentState,
      setGameState: (state: GameState) => {
        currentState = state;
      },
      setLastAIMove: (move: Move | null) => {
        lastAIMove = move;
      },
      setSelectedStackKey: () => {
        // selection not relevant for this test
      },
      getMustMoveFromStackKey: () => undefined,
      applyCanonicalMove,
      hasPendingTerritorySelfElimination: () => false,
      hasPendingLineRewardElimination: () => false,
      canCurrentPlayerSwapSides: () => false,
      applySwapSidesForCurrentPlayer: () => false,
    };

    const rng = () => 0.5;

    await maybeRunAITurnSandbox(hooks, rng);

    expect(applyCanonicalMove).toHaveBeenCalledTimes(1);
    const appliedMove = applyCanonicalMove.mock.calls[0][0] as Move;

    expect(appliedMove.type).toBe('forced_elimination');
    expect(appliedMove.player).toBe(1);

    expect(lastAIMove).not.toBeNull();
    if (!lastAIMove) {
      throw new Error('Expected lastAIMove to be non-null after forced_elimination move');
    }
    const lastMove = lastAIMove as Move;
    expect(lastMove.type).toBe('forced_elimination');
    expect(lastMove.player).toBe(1);
  });
});
