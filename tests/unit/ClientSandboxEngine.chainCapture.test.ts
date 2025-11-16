import { ClientSandboxEngine, SandboxConfig, SandboxInteractionHandler } from '../../src/client/sandbox/ClientSandboxEngine';
import {
  BoardType,
  GameState,
  Position,
  RingStack,
  Player,
  PlayerChoiceResponseFor,
  CaptureDirectionChoice,
  positionToString
} from '../../src/shared/types/game';

/**
 * Sandbox chain capture parity tests.
 *
 * These tests mirror a core Rust chain-capture scenario and the
 * GameEngine.chainCapture tests, but run against the client-local
 * ClientSandboxEngine to ensure consistent overtaking behaviour
 * (top-ring-only), marker handling, and mandatory chain continuation.
 */

describe('ClientSandboxEngine chain capture parity', () => {
  const boardType: BoardType = 'square8';

  function createEngine(): ClientSandboxEngine {
    const config: SandboxConfig = {
      boardType,
      numPlayers: 3,
      playerKinds: ['human', 'human', 'human']
    };

    const handler: SandboxInteractionHandler = {
      async requestChoice<TChoice extends any>(choice: TChoice): Promise<PlayerChoiceResponseFor<any>> {
        // For capture_direction choices, just select the first option so
        // chains proceed deterministically.
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

  test('two-step chain capture mirrors backend/Rust behaviour in sandbox', async () => {
    // Scenario:
    // - Player 1 (Red) at (2,2) height 2
    // - Player 2 (Blue) at (2,3) height 1
    // - Player 3 (Green) at (2,5) height 1
    // Expected (matching GameEngine/Rust tests):
    // Red captures Blue and then Green in a chain, finishing at (2,7)
    // with height 4, and original target positions are empty.

    const engine = createEngine();
    const engineAny = engine as any;
    const state: GameState = engineAny.gameState as GameState;

    // Ensure we are in movement phase with player 1 to allow capture via
    // handleHumanCellClick.
    state.currentPhase = 'movement';
    state.currentPlayer = 1;

    const board = state.board;

    const makeStack = (playerNumber: number, height: number, position: Position) => {
      const rings = Array(height).fill(playerNumber);
      const stack: RingStack = {
        position,
        rings,
        stackHeight: rings.length,
        capHeight: rings.length,
        controllingPlayer: playerNumber
      };
      const key = positionToString(position);
      board.stacks.set(key, stack);
    };

    const redPos: Position = { x: 2, y: 2 };
    const bluePos: Position = { x: 2, y: 3 };
    const greenPos: Position = { x: 2, y: 5 };

    makeStack(1, 2, redPos);   // Red height 2 at (2,2)
    makeStack(2, 1, bluePos);  // Blue height 1 at (2,3)
    makeStack(3, 1, greenPos); // Green height 1 at (2,5)

    // Simulate user selecting the attacking stack, then choosing the first
    // capture landing at (2,4). The sandbox engine will drive the rest of
    // the chain internally using the interaction handler.
    engine.handleHumanCellClick(redPos);
    engine.handleHumanCellClick({ x: 2, y: 4 });

    // Allow any async chain-capture continuation to resolve (for the
    // capture_direction choice case) before asserting on final state.
    await Promise.resolve();

    const finalState = engine.getGameState();
    const finalBoard = finalState.board;

    const stackAtRed = finalBoard.stacks.get('2,2');
    const stackAtBlue = finalBoard.stacks.get('2,3');
    const stackAtGreen = finalBoard.stacks.get('2,5');
    const stackAtFinal = finalBoard.stacks.get('2,7');

    expect(stackAtRed).toBeUndefined();
    expect(stackAtBlue).toBeUndefined();
    expect(stackAtGreen).toBeUndefined();
    expect(stackAtFinal).toBeDefined();
    expect(stackAtFinal!.stackHeight).toBe(4);
    expect(stackAtFinal!.controllingPlayer).toBe(1);
  });
});
