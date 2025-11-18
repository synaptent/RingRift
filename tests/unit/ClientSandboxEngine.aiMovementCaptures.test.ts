import { ClientSandboxEngine, SandboxConfig, SandboxInteractionHandler } from '../../src/client/sandbox/ClientSandboxEngine';
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
 * AI movement/capture behaviour tests for the client-local sandbox engine.
 *
 * These focus on ensuring that:
 * - maybeRunAITurn prefers capture chains when captures are available.
 * - maybeRunAITurn falls back to simple movement when only non-capturing
 *   moves exist.
 * - maybeRunAITurn does not stall in positions where captures are the only
 *   legal actions.
 */

describe('ClientSandboxEngine AI movement + capture behaviour', () => {
  const boardType: BoardType = 'square8';

  function createEngine(playerKinds: ('human' | 'ai')[] = ['ai', 'ai']): ClientSandboxEngine {
    const config: SandboxConfig = {
      boardType,
      numPlayers: playerKinds.length,
      playerKinds
    };

    const handler: SandboxInteractionHandler = {
      async requestChoice<TChoice extends PlayerChoice>(
        choice: TChoice
      ): Promise<PlayerChoiceResponseFor<TChoice>> {
        const anyChoice = choice as any;

        // For capture_direction choices, deterministically pick the option
        // with the lexicographically smallest landingPosition. This mirrors
        // chain-capture parity tests and keeps scenarios reproducible.
        if (anyChoice.type === 'capture_direction') {
          const cd = anyChoice as CaptureDirectionChoice;
          const options = cd.options || [];
          if (options.length === 0) {
            throw new Error('Test SandboxInteractionHandler: no options for capture_direction');
          }

          let selected = options[0];
          for (const opt of options) {
            if (
              opt.landingPosition.x < selected.landingPosition.x ||
              (opt.landingPosition.x === selected.landingPosition.x &&
                opt.landingPosition.y < selected.landingPosition.y)
            ) {
              selected = opt;
            }
          }

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

  test('AI in movement phase takes an available overtaking capture instead of stalling', async () => {
    const engine = createEngine(['ai', 'ai']);
    const engineAny = engine as any;
    const state: GameState = engineAny.gameState as GameState;

    // Configure a simple one-step overtaking capture for player 1.
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

    const attacker: Position = { x: 2, y: 2 };
    const target: Position = { x: 2, y: 3 };
    const landing: Position = { x: 2, y: 4 };

    makeStack(1, 2, attacker); // Player 1, height 2
    makeStack(2, 1, target);   // Player 2, height 1 directly in front

    // Sanity: no stack at the intended landing before the capture.
    expect(board.stacks.get(positionToString(landing))).toBeUndefined();

    // Run a single AI turn for player 1. With the new behaviour, this
    // should trigger an overtaking capture chain starting from attacker.
    await engine.maybeRunAITurn();

    const finalState = engine.getGameState();
    const finalBoard = finalState.board;

    const stackAtAttacker = finalBoard.stacks.get(positionToString(attacker));
    const stackAtTarget = finalBoard.stacks.get(positionToString(target));
    const stackAtLanding = finalBoard.stacks.get(positionToString(landing));

    // The capturing stack should have moved off the original attacker
    // square and removed the target, finishing on the landing square.
    expect(stackAtAttacker).toBeUndefined();
    expect(stackAtTarget).toBeUndefined();
    expect(stackAtLanding).toBeDefined();
    expect(stackAtLanding!.controllingPlayer).toBe(1);
    expect(stackAtLanding!.stackHeight).toBeGreaterThanOrEqual(2);
  });
});
