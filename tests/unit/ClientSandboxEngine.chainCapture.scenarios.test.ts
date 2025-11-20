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
  positionToString
} from '../../src/shared/types/game';

/**
 * Sandbox scenario tests: 180° reversal (FAQ 15.3.1) with ClientSandboxEngine.
 *
 * This mirrors the backend `FAQ_15_3_1_180_degree_reversal_basic` scenario in
 * `tests/scenarios/ComplexChainCaptures.test.ts`, but runs entirely through the
 * client-local sandbox engine and its `handleHumanCellClick` API.
 *
 * Rules/FAQ references:
 * - `ringrift_complete_rules.md` §10.3 (Chain Overtaking)
 * - `ringrift_complete_rules.md` §15.3.1 (180° Reversal Pattern)
 */

describe('ClientSandboxEngine chain capture scenarios (FAQ 15.3.1)', () => {
  const boardType: BoardType = 'square19';

  function createEngine(): ClientSandboxEngine {
    const config: SandboxConfig = {
      boardType,
      numPlayers: 2,
      playerKinds: ['human', 'human']
    };

    const handler: SandboxInteractionHandler = {
      async requestChoice<TChoice extends PlayerChoice>(
        choice: TChoice
      ): Promise<PlayerChoiceResponseFor<TChoice>> {
        // For this scenario we do not expect capture_direction choices, but
        // we still provide a generic fallback that selects the first option.
        const anyChoice = choice as any;
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

  test('FAQ_15_3_1_180_degree_reversal_basic_sandbox', async () => {
    const engine = createEngine();
    const engineAny = engine as any;
    const state: GameState = engineAny.gameState as GameState;

    // Allow a capture to be initiated from a human click.
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

    // Geometry shared with the backend scenario:
    // - Blue (player 1) at A with height 4
    // - Red (player 2) at B with height 3
    // - A straight horizontal line with enough empty spaces beyond.
    const A: Position = { x: 4, y: 4 }; // Blue start
    const B: Position = { x: 6, y: 4 }; // Red target stack
    const C: Position = { x: 8, y: 4 }; // First landing point beyond B

    makeStack(1, 4, A);
    makeStack(2, 3, B);

    // Human interaction:
    // 1. Select the attacking stack at A.
    // 2. Click a landing cell C beyond B to request an overtaking capture.
    //    ClientSandboxEngine uses the shared RuleEngine to validate and will
    //    drive any mandatory follow-up chain captures internally.
    await engine.handleHumanCellClick(A);
    await engine.handleHumanCellClick(C);

    // At this point both click handlers have fully resolved (including any
    // mandatory chain continuation), so we can safely inspect the final
    // sandbox state.

    const finalState = engine.getGameState();
    const finalBoard = finalState.board;

    const stacks = finalBoard.stacks as Map<string, RingStack>;
    const allStacks: RingStack[] = Array.from(stacks.values());

    const blueStacks: RingStack[] = allStacks.filter((s) => s.controllingPlayer === 1);
    const redStacksAtB = stacks.get('6,4');

    // Debug: log final board stacks for this FAQ 15.3.1 sandbox scenario.
    // This helps trace any off-by-one discrepancies in stack heights after
    // the full movement + post-movement pipeline has run.
    // eslint-disable-next-line no-console
    console.log(
      'FAQ_15_3_1_180_degree_reversal_basic_sandbox final stacks:',
      Array.from(stacks.entries()).map(([key, stack]) => ({
        key,
        controllingPlayer: stack.controllingPlayer,
        stackHeight: stack.stackHeight,
        rings: stack.rings,
      }))
    );

    // There should be exactly one Blue-controlled stack (the overtaker).
    expect(blueStacks.length).toBe(1);

    const finalBlue = blueStacks[0];

    // Aggregate expectations matching FAQ 15.3.1 and the backend scenario:
    // - Blue started with 4 rings and overtakes twice from the same target
    //   stack at B, ending with 6 rings in the overtaker stack.
    // - Red's original stack at B is reduced from height 3 down to 1.
    expect(finalBlue.stackHeight).toBe(6);
    expect(finalBlue.controllingPlayer).toBe(1);

    expect(redStacksAtB).toBeDefined();
    expect(redStacksAtB!.stackHeight).toBe(1);
  });
});
