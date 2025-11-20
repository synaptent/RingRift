import { ClientSandboxEngine, SandboxConfig, SandboxInteractionHandler } from '../../src/client/sandbox/ClientSandboxEngine';
import {
  BoardType,
  GameState,
  Position,
  RingStack,
  Player,
  PlayerChoice,
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
      async requestChoice<TChoice extends PlayerChoice>(
        choice: TChoice
      ): Promise<PlayerChoiceResponseFor<TChoice>> {
        const anyChoice = choice as any;

        // For capture_direction choices, select a deterministic option whose
        // landingPosition is lexicographically smallest (x, then y). This
        // mirrors the backend GameEngine chain‑capture tests and keeps
        // sandbox parity scenarios reproducible.
        if (anyChoice.type === 'capture_direction') {
          const cdChoice = anyChoice as CaptureDirectionChoice;
          const options = cdChoice.options || [];

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
            choiceId: cdChoice.id,
            playerNumber: cdChoice.playerNumber,
            choiceType: cdChoice.type,
            selectedOption: selected
          } as PlayerChoiceResponseFor<TChoice>;
        }

        // Fallback for other choice types used by the sandbox: pick the
        // first option if present.
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

  test('two-step chain capture mirrors backend/Rust behaviour in sandbox', async () => {
    // Q10 / Rust test_start_chain_capture + test_complete_chain_capture baseline.
    // This anchors the simplest straight two-step chain in the sandbox.
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
    await engine.handleHumanCellClick(redPos);
    await engine.handleHumanCellClick({ x: 2, y: 4 });

    // At this point, handleHumanCellClick has awaited the full chain
    // resolution (including any capture_direction choices), so we can
    // safely inspect the final state.

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

  test('orthogonal multi-branch chain capture uses capture_direction choices (Rust player-choice scenario parity)', async () => {
    // Mirrors the Rust `test_chain_capture_player_choice_simulation` and the
    // backend GameEngine chain‑capture choice integration tests:
    //
    // - Red at (3,3) h2 (attacker)
    // - Blue at (3,4) h1 (initial target)
    // - Green at (4,5) h1
    // - Yellow at (2,5) h1
    //
    // After Red captures Blue and lands at (3,5), there are multiple legal
    // follow‑up capture directions. The sandbox engine must:
    //   - detect those options using the same capture geometry as the backend
    //   - issue a capture_direction PlayerChoice
    //   - apply the selected chain branch deterministically (lexicographic
    //     landing selection) and finish the chain.

    const config: SandboxConfig = {
      boardType,
      numPlayers: 4,
      playerKinds: ['human', 'human', 'human', 'human']
    };

    const choices: CaptureDirectionChoice[] = [];

    const handler: SandboxInteractionHandler = {
      async requestChoice<TChoice extends PlayerChoice>(
        choice: TChoice
      ): Promise<PlayerChoiceResponseFor<TChoice>> {
        const anyChoice = choice as any;

        if (anyChoice.type === 'capture_direction') {
          const cd = anyChoice as CaptureDirectionChoice;
          choices.push(cd);
          const options = cd.options || [];
          expect(options.length).toBeGreaterThan(0);

          // Deterministically choose the option with the smallest
          // landingPosition (x, then y) to mirror backend tests.
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

    const engine = new ClientSandboxEngine({ config, interactionHandler: handler });
    const engineAny = engine as any;
    const state: GameState = engineAny.gameState as GameState;

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

    const redPos: Position = { x: 3, y: 3 };
    const bluePos: Position = { x: 3, y: 4 };
    const greenPos: Position = { x: 4, y: 5 };
    const yellowPos: Position = { x: 2, y: 5 };

    makeStack(1, 2, redPos); // Red attacker
    makeStack(2, 1, bluePos);
    makeStack(3, 1, greenPos);
    makeStack(4, 1, yellowPos);

    // Human performs the initial capture: select Red at (3,3), then click a
    // landing beyond Blue. We choose (3,5) to mirror the Rust/backend tests.
    await engine.handleHumanCellClick(redPos);
    await engine.handleHumanCellClick({ x: 3, y: 5 });

    // By awaiting both clicks, we ensure that any capture_direction choices
    // and mandatory chain continuation have fully resolved before we
    // inspect the final state.

    const finalState = engine.getGameState();
    const finalBoard = finalState.board;

    // At least one capture_direction choice should have been issued.
    expect(choices.length).toBeGreaterThan(0);

    const allPairs = choices.flatMap(ch =>
      (ch.options || []).map(o =>
        `${o.targetPosition.x},${o.targetPosition.y}->${o.landingPosition.x},${o.landingPosition.y}`
      )
    );

    // The core rule-faithful options from the first branching point should
    // appear somewhere in the accumulated choices, matching the backend
    // GameEngine.chainCaptureChoiceIntegration expectations.
    expect(allPairs).toEqual(
      expect.arrayContaining([
        '4,5->6,5',
        '4,5->7,5',
        '2,5->0,5'
      ])
    );

    // Use the final choice to locate the branch actually taken and assert on
    // the resulting board state.
    const lastChoice = choices[choices.length - 1];
    const options = lastChoice.options || [];
    const selected = options.reduce((prev, cur) =>
      cur.landingPosition.x < prev.landingPosition.x ||
      (cur.landingPosition.x === prev.landingPosition.x &&
        cur.landingPosition.y < prev.landingPosition.y)
        ? cur
        : prev
    );

    const startKey = '3,3';
    const blueKey = '3,4';
    const intermediateKey = '3,5';

    const stackAtStart = finalBoard.stacks.get(startKey);
    const stackAtBlue = finalBoard.stacks.get(blueKey);
    const stackAtIntermediate = finalBoard.stacks.get(intermediateKey);

    // The original attacker and first target must be gone; the chain
    // continues from (3,5) along one of the rule‑legal directions.
    expect(stackAtStart).toBeUndefined();
    expect(stackAtBlue).toBeUndefined();
    expect(stackAtIntermediate).toBeUndefined();

    // There should be exactly one Red-controlled stack on the board,
    // representing the final capturing stack after the chosen branch.
    const redStacks = Array.from(finalBoard.stacks.values()).filter(
      s => s.controllingPlayer === 1
    );
    expect(redStacks.length).toBe(1);
    expect(redStacks[0].stackHeight).toBeGreaterThanOrEqual(3);

    // Sandbox has no explicit chain state; successful completion is implied
    // by the movement phase having advanced after the chain resolves.
    expect(
      finalState.currentPhase === 'movement' ||
        finalState.currentPhase === 'ring_placement'
    ).toBe(true);
  });
});
