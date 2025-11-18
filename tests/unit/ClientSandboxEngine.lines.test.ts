import {
  ClientSandboxEngine,
  SandboxConfig,
  SandboxInteractionHandler
} from '../../src/client/sandbox/ClientSandboxEngine';
import {
  BoardState,
  BoardType,
  GameState,
  Position,
  RingStack,
  PlayerChoiceResponseFor,
  CaptureDirectionChoice,
  positionToString,
  BOARD_CONFIGS
} from '../../src/shared/types/game';

/**
 * Sandbox line detection + reward tests.
 *
 * These exercise ClientSandboxEngine's line-processing behaviour, which
 * mirrors backend defaults when no interaction manager is wired:
 * - Exact-length line: collapse all markers in the line and eliminate a cap.
 * - Longer-than-required line: collapse only the minimum required markers,
 *   with no elimination.
 */

describe('ClientSandboxEngine line processing', () => {
  const boardType: BoardType = 'square8';
  const requiredLength = BOARD_CONFIGS[boardType].lineLength;

  function createEngine(): ClientSandboxEngine {
    const config: SandboxConfig = {
      boardType,
      numPlayers: 2,
      playerKinds: ['human', 'human']
    };

    const handler: SandboxInteractionHandler = {
      // For these tests we never actually trigger PlayerChoices, but we
      // provide a trivial handler to satisfy the constructor.
      async requestChoice<TChoice extends any>(choice: TChoice): Promise<PlayerChoiceResponseFor<any>> {
        const anyChoice = choice as CaptureDirectionChoice;
        const selectedOption = (anyChoice as any).options
          ? (anyChoice as any).options[0]
          : undefined;

        return {
          choiceId: (choice as any).id,
          playerNumber: (choice as any).playerNumber,
          choiceType: (choice as any).type,
          selectedOption
        } as PlayerChoiceResponseFor<any>;
      }
    };

    return new ClientSandboxEngine({ config, interactionHandler: handler });
  }

  function makeStack(playerNumber: number, height: number, position: Position, board: BoardState) {
    const rings = Array(height).fill(playerNumber);
    const stack: RingStack = {
      position,
      rings,
      stackHeight: rings.length,
      capHeight: rings.length,
      controllingPlayer: playerNumber
    };
    board.stacks.set(positionToString(position), stack);
  }

  test('Q7_exact_length_line_collapse_sandbox', () => {
    const engine = createEngine();
    const engineAny = engine as any;
    const state: GameState = engineAny.gameState as GameState;

    state.currentPlayer = 1;
    const board = state.board;

    // Clear any existing markers/stacks.
    board.markers.clear();
    board.stacks.clear();
    board.collapsedSpaces.clear();

    // Place an exact-length horizontal line of markers for player 1 at y=1.
    const linePositions: Position[] = [];
    for (let i = 0; i < requiredLength; i++) {
      const pos: Position = { x: i, y: 1 };
      linePositions.push(pos);
      board.markers.set(positionToString(pos), {
        player: 1,
        position: pos,
        type: 'regular'
      });
    }

    // Add a stack for player 1 so there is a cap to eliminate.
    const stackPos: Position = { x: 7, y: 7 };
    makeStack(1, 2, stackPos, board);

    const initialTotalEliminated = state.totalRingsEliminated;
    const initialTerritory = state.players.find(p => p.playerNumber === 1)!.territorySpaces;

    // Invoke line processing directly.
    engineAny.processLinesForCurrentPlayer();

    const finalState = engine.getGameState();
    const finalBoard = finalState.board;
    const player1 = finalState.players.find(p => p.playerNumber === 1)!;

    // All marker positions in the line should now be collapsed spaces for p1,
    // with no markers or stacks remaining at those positions.
    for (const pos of linePositions) {
      const key = positionToString(pos);
      expect(finalBoard.collapsedSpaces.get(key)).toBe(1);
      expect(finalBoard.markers.has(key)).toBe(false);
      expect(finalBoard.stacks.has(key)).toBe(false);
    }

    // The stack used for elimination should have been removed, and player1's
    // eliminatedRings / totalRingsEliminated increased by at least 1.
    expect(finalBoard.stacks.get(positionToString(stackPos))).toBeUndefined();
    expect(player1.eliminatedRings).toBeGreaterThan(0);
    expect(finalState.totalRingsEliminated).toBeGreaterThan(initialTotalEliminated);

    // Territory spaces should have increased by exactly the line length.
    expect(player1.territorySpaces).toBe(initialTerritory + requiredLength);
  });

  test('longer-than-required line collapses minimum markers without elimination', () => {
    const engine = createEngine();
    const engineAny = engine as any;
    const state: GameState = engineAny.gameState as GameState;

    state.currentPlayer = 1;
    const board = state.board;

    board.markers.clear();
    board.stacks.clear();
    board.collapsedSpaces.clear();

    // Place a line longer than required: requiredLength + 1 markers.
    const linePositions: Position[] = [];
    for (let i = 0; i < requiredLength + 1; i++) {
      const pos: Position = { x: i, y: 2 };
      linePositions.push(pos);
      board.markers.set(positionToString(pos), {
        player: 1,
        position: pos,
        type: 'regular'
      });
    }

    const detectedLines = engineAny.findAllLines(board) as any[];
    expect(detectedLines.length).toBeGreaterThanOrEqual(1);

    // Add a stack for player 1; since this is a longer line we expect no
    // elimination, so the stack should remain unchanged.
    const stackPos: Position = { x: 7, y: 7 };
    makeStack(1, 2, stackPos, board);

    const initialPlayer1 = state.players.find(p => p.playerNumber === 1)!;
    const initialEliminated = initialPlayer1.eliminatedRings;
    const initialTotalEliminated = state.totalRingsEliminated;
    const initialTerritory = initialPlayer1.territorySpaces;

    engineAny.processLinesForCurrentPlayer();

    const finalState = engine.getGameState();
    const finalBoard = finalState.board;
    const player1 = finalState.players.find(p => p.playerNumber === 1)!;

    // Exactly requiredLength markers should be collapsed; the remaining
    // marker should still exist and not be collapsed.
    const collapsedKeys = new Set<string>();
    for (const [key, owner] of finalBoard.collapsedSpaces) {
      if (owner === 1) collapsedKeys.add(key);
    }

    expect(collapsedKeys.size).toBe(requiredLength);

    const remainingPos = linePositions[requiredLength];
    const remainingKey = positionToString(remainingPos);
    expect(finalBoard.markers.has(remainingKey)).toBe(true);
    expect(finalBoard.collapsedSpaces.has(remainingKey)).toBe(false);

    // No elimination should have occurred.
    expect(player1.eliminatedRings).toBe(initialEliminated);
    expect(finalState.totalRingsEliminated).toBe(initialTotalEliminated);

    // Territory spaces should have increased by exactly requiredLength.
    expect(player1.territorySpaces).toBe(initialTerritory + requiredLength);

    // Stack should still exist at stackPos (no forced elimination in this test).
    expect(finalBoard.stacks.get(positionToString(stackPos))).toBeDefined();
  });
});
