import {
  ClientSandboxEngine,
  SandboxConfig,
  SandboxInteractionHandler,
} from '../../src/client/sandbox/ClientSandboxEngine';
import {
  BoardType,
  GameState,
  Position,
  RingStack,
  PlayerChoiceResponseFor,
  CaptureDirectionChoice,
  positionToString,
} from '../../src/shared/types/game';

/**
 * Sandbox placement + forced-elimination parity tests.
 *
 * These focus on:
 * - No-dead-placement: placement must leave at least one legal move/capture.
 * - Forced elimination: when a player is fully blocked with no rings in hand.
 *
 * NOTE: This file includes tests that throw errors from async methods.
 * Jest parallel workers sometimes crash on async rejections even when properly
 * caught. The unhandledRejection listener below prevents worker crashes.
 * If issues persist, run with: npx jest --runInBand <this-file>
 */

// Prevent Jest worker crashes from unhandled async rejections that ARE actually caught
let capturedUnhandledRejection: Error | null = null;
const unhandledRejectionHandler = (reason: unknown): void => {
  capturedUnhandledRejection = reason instanceof Error ? reason : new Error(String(reason));
};

beforeAll(() => {
  process.on('unhandledRejection', unhandledRejectionHandler);
});

afterAll(() => {
  process.removeListener('unhandledRejection', unhandledRejectionHandler);
});

afterEach(() => {
  capturedUnhandledRejection = null;
});

describe('ClientSandboxEngine placement + forced elimination', () => {
  const boardType: BoardType = 'square8';

  function createEngine(): ClientSandboxEngine {
    const config: SandboxConfig = {
      boardType,
      numPlayers: 2,
      playerKinds: ['human', 'human'],
    };

    const handler: SandboxInteractionHandler = {
      // For these tests we never actually trigger PlayerChoices, but we
      // provide a trivial handler to satisfy the constructor.
      async requestChoice<TChoice>(choice: TChoice): Promise<PlayerChoiceResponseFor<any>> {
        const anyChoice = choice as CaptureDirectionChoice;
        const selectedOption = (anyChoice as any).options
          ? (anyChoice as any).options[0]
          : undefined;

        return {
          choiceId: (choice as any).id,
          playerNumber: (choice as any).playerNumber,
          choiceType: (choice as any).type,
          selectedOption,
        } as PlayerChoiceResponseFor<any>;
      },
    };

    return new ClientSandboxEngine({ config, interactionHandler: handler });
  }

  test('no-dead-placement: sandbox rejects placements that leave no legal move/capture', async () => {
    const engine = createEngine();
    const engineAny = engine as any;
    const state: GameState = engineAny.gameState as GameState;

    // Ensure we are in ring_placement phase for player 1.
    state.currentPhase = 'ring_placement';
    state.currentPlayer = 1;

    const board = state.board;

    // Choose corner (0,0). For rays that stay in-bounds (east, north, northeast),
    // mark the immediate cells as collapsed so there is no legal movement path
    // or capture from a hypothetical placement at (0,0).
    const blockPositions: Position[] = [
      { x: 1, y: 0 },
      { x: 0, y: 1 },
      { x: 1, y: 1 },
    ];
    for (const pos of blockPositions) {
      board.collapsedSpaces.set(positionToString(pos), 0);
    }

    const player1 = state.players.find((p) => p.playerNumber === 1)!;
    const initialRingsInHand = player1.ringsInHand;

    // Attempt to place a ring at (0,0). The sandbox no-dead-placement check
    // should reject this with an error. The state remains unchanged.
    const placementPos: Position = { x: 0, y: 0 };

    // Use try/catch for robust Jest worker isolation (avoids parallel worker crashes)
    let caughtError: Error | null = null;
    try {
      await engine.handleHumanCellClick(placementPos);
    } catch (err) {
      caughtError = err as Error;
    }

    expect(caughtError).not.toBeNull();
    expect(caughtError!.message).toContain(
      'Placement would result in a stack with no legal moves or captures'
    );

    const finalState = engine.getGameState();
    const finalBoard = finalState.board;

    expect(finalBoard.stacks.size).toBe(0);
    const updatedPlayer1 = finalState.players.find((p) => p.playerNumber === 1)!;
    expect(updatedPlayer1.ringsInHand).toBe(initialRingsInHand);
  });

  test('forced elimination: blocked player with no rings in hand loses a cap and turn passes', () => {
    const engine = createEngine();
    const engineAny = engine as any;
    const state: GameState = engineAny.gameState as GameState;

    const board = state.board;

    // Make player 2 the current player, with a single stack at (0,0) that has
    // no legal moves or captures because all outward rays are blocked by
    // collapsed spaces as in the previous test.
    state.currentPlayer = 2;
    const player2 = state.players.find((p) => p.playerNumber === 2)!;
    player2.ringsInHand = 0;

    const stackPos: Position = { x: 0, y: 0 };
    const rings = [2, 2];
    const stack: RingStack = {
      position: stackPos,
      rings,
      stackHeight: rings.length,
      capHeight: rings.length,
      controllingPlayer: 2,
    };
    board.stacks.set(positionToString(stackPos), stack);

    const blockPositions: Position[] = [
      { x: 1, y: 0 },
      { x: 0, y: 1 },
      { x: 1, y: 1 },
    ];
    for (const pos of blockPositions) {
      board.collapsedSpaces.set(positionToString(pos), 0);
    }

    const initialTotalEliminated = state.totalRingsEliminated;

    // Directly invoke the sandbox forced-elimination helper.
    engineAny.maybeProcessForcedEliminationForCurrentPlayer();

    const finalState = engine.getGameState();
    const finalBoard = finalState.board;

    // Both rings from the (0,0) stack should have been eliminated.
    expect(finalBoard.stacks.get(positionToString(stackPos))).toBeUndefined();
    const finalPlayer2 = finalState.players.find((p) => p.playerNumber === 2)!;
    expect(finalPlayer2.eliminatedRings).toBeGreaterThanOrEqual(2);
    expect(finalState.totalRingsEliminated).toBeGreaterThanOrEqual(initialTotalEliminated + 2);

    // Turn should have passed to the next player (player 1 in this config).
    expect(finalState.currentPlayer).toBe(1);
  });
});
