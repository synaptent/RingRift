import {
  ClientSandboxEngine,
  SandboxConfig,
  SandboxInteractionHandler
} from '../../src/client/sandbox/ClientSandboxEngine';
import {
  BoardType,
  GameState,
  Position,
  RingStack,
  PlayerChoice,
  PlayerChoiceResponseFor,
  CaptureDirectionChoice,
  positionToString
} from '../../src/shared/types/game';

/**
 * Invariant tests for ClientSandboxEngine.assertBoardInvariants.
 *
 * These tests ensure that:
 * - Normal initial boards satisfy invariants.
 * - Deliberately constructed illegal states (e.g. stack on collapsed space)
 *   are detected and cause the helper to throw in NODE_ENV === 'test'.
 */

describe('ClientSandboxEngine BoardState invariants', () => {
  const boardType: BoardType = 'square8';

  function createEngine(numPlayers: number = 2): ClientSandboxEngine {
    const config: SandboxConfig = {
      boardType,
      numPlayers,
      playerKinds: Array.from({ length: numPlayers }, () => 'human')
    };

    const handler: SandboxInteractionHandler = {
      async requestChoice<TChoice extends PlayerChoice>(
        choice: TChoice
      ): Promise<PlayerChoiceResponseFor<TChoice>> {
        const anyChoice = choice as any;

        if (anyChoice.type === 'capture_direction') {
          const cd = anyChoice as CaptureDirectionChoice;
          const options = cd.options || [];
          if (options.length === 0) {
            throw new Error('Test SandboxInteractionHandler: no options for capture_direction');
          }

          // Deterministically pick the first option for reproducibility.
          const selected = options[0];
          return {
            choiceId: cd.id,
            playerNumber: cd.playerNumber,
            choiceType: cd.type,
            selectedOption: selected
          } as PlayerChoiceResponseFor<TChoice>;
        }

        const selectedOption = anyChoice.options ? anyChoice.options[0] : undefined;
        return {
          choiceId: anyChoice.id,
          playerNumber: anyChoice.playerNumber,
          choiceType: anyChoice.type,
          selectedOption
        } as PlayerChoiceResponseFor<TChoice>;
      }
    };

    return new ClientSandboxEngine({ config, interactionHandler: handler });
  }

  test('assertBoardInvariants passes on a fresh empty board', () => {
    const engine = createEngine();
    const engineAny = engine as any;

    expect(() => {
      engineAny.assertBoardInvariants('initial empty board');
    }).not.toThrow();
  });

  test('assertBoardInvariants throws when a stack exists on a collapsed space', () => {
    const engine = createEngine();
    const engineAny = engine as any;
    const state: GameState = engineAny.gameState as GameState;
    const board = state.board;

    const pos: Position = { x: 0, y: 0 };
    const key = positionToString(pos);

    const rings = [1, 1];
    const stack: RingStack = {
      position: pos,
      rings,
      stackHeight: rings.length,
      capHeight: rings.length,
      controllingPlayer: 1
    };

    board.stacks.set(key, stack);
    board.collapsedSpaces.set(key, 1);

    expect(() => {
      engineAny.assertBoardInvariants('stack on collapsed space');
    }).toThrow(/stack present on collapsed space/);
  });
});
