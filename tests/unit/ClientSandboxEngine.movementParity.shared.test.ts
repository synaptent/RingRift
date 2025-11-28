import {
  ClientSandboxEngine,
  SandboxConfig,
  SandboxInteractionHandler,
} from '../../src/client/sandbox/ClientSandboxEngine';
import type { GameState, BoardType, Position, RingStack } from '../../src/shared/engine';
import {
  BOARD_CONFIGS,
  positionToString,
  applySimpleMovement as applySimpleMovementAggregate,
  hashGameState,
} from '../../src/shared/engine';
import type { PlayerChoiceResponseFor, CaptureDirectionChoice } from '../../src/shared/types/game';

describe('ClientSandboxEngine movement parity with shared MovementAggregate', () => {
  const boardType: BoardType = 'square8';

  function createEngine(): ClientSandboxEngine {
    const config: SandboxConfig = {
      boardType,
      numPlayers: 2,
      playerKinds: ['human', 'human'],
    };

    const handler: SandboxInteractionHandler = {
      // For these tests we never actually trigger PlayerChoices in practice,
      // but we provide a trivial handler to satisfy the constructor and keep
      // types aligned with other sandbox tests.
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

  function makeStack(
    playerNumber: number,
    height: number,
    position: Position,
    state: GameState
  ): void {
    const rings = Array(height).fill(playerNumber);
    const stack: RingStack = {
      position,
      rings,
      stackHeight: rings.length,
      capHeight: rings.length,
      controllingPlayer: playerNumber,
    };
    state.board.stacks.set(positionToString(position), stack);
  }

  it('simple non-capture move via sandbox movement engine matches aggregate outcome', async () => {
    const engine = createEngine();
    engine.disableOrchestratorAdapter();

    const engineAny = engine as any;
    const internalState: GameState = engineAny.gameState as GameState;

    internalState.currentPlayer = 1;
    internalState.currentPhase = 'movement';

    const board = internalState.board;
    board.stacks.clear();
    board.markers.clear();
    board.collapsedSpaces.clear();

    // Place a single two-ring stack at (2,2) for player 1.
    const origin: Position = { x: 2, y: 2 };
    makeStack(1, 2, origin, internalState);

    // Destination two steps to the east with an empty path and landing cell.
    const dest: Position = { x: 4, y: 2 };

    const config = BOARD_CONFIGS[boardType];
    expect(dest.x).toBeGreaterThanOrEqual(0);
    expect(dest.x).toBeLessThan(config.size);
    expect(dest.y).toBeGreaterThanOrEqual(0);
    expect(dest.y).toBeLessThan(config.size);
    expect(board.stacks.has(positionToString(dest))).toBe(false);

    // Shared-core path: start from a defensive snapshot and apply the aggregate.
    const startingState: GameState = engine.getGameState();
    const coreOutcome = applySimpleMovementAggregate(startingState, {
      from: origin,
      to: dest,
      player: 1,
    });

    // Sandbox path: reset the engine's internal state to the same snapshot and
    // invoke the legacy movement click handler wired to sandboxMovementEngine.
    engineAny.gameState = startingState;

    const originalAdvanceAfterMovement = engineAny.advanceAfterMovement;
    engineAny.advanceAfterMovement = async () => {
      // No-op for this parity test: we only care about immediate movement effects.
    };

    const originKey = positionToString(origin);
    engineAny._movementInvocationContext = 'canonical';
    engineAny._selectedStackKey = originKey;

    await engineAny.handleMovementClick(dest);

    engineAny._movementInvocationContext = null;
    engineAny.advanceAfterMovement = originalAdvanceAfterMovement;

    const sandboxStateAfter: GameState = engine.getGameState();

    const coreHash = hashGameState(coreOutcome.nextState);
    const sandboxHash = hashGameState(sandboxStateAfter);

    expect(sandboxHash).toEqual(coreHash);
  });
});
