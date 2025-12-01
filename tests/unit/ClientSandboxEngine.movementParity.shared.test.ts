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
    // Now uses orchestrator-backed engine which delegates to shared MovementAggregate

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

    // Sandbox/orchestrator path: apply canonical move via applyCanonicalMove
    // which delegates to the orchestrator.
    engineAny.gameState = startingState;

    const moveStackMove = {
      id: 'move-1',
      type: 'move_stack' as const,
      player: 1,
      from: origin,
      to: dest,
      timestamp: new Date(),
      thinkTime: 0,
      moveNumber: 1,
    };

    await engine.applyCanonicalMove(moveStackMove);

    const sandboxStateAfter: GameState = engine.getGameState();

    // Compare board-level changes (stacks, markers) which is what the aggregate
    // transforms. The orchestrator also advances the turn, so we compare
    // specific board effects rather than full state hash.
    const coreBoard = coreOutcome.nextState.board;
    const sandboxBoard = sandboxStateAfter.board;

    // Origin should be empty in both
    expect(coreBoard.stacks.has(positionToString(origin))).toBe(false);
    expect(sandboxBoard.stacks.has(positionToString(origin))).toBe(false);

    // Destination should have the stack in both
    const coreDestStack = coreBoard.stacks.get(positionToString(dest));
    const sandboxDestStack = sandboxBoard.stacks.get(positionToString(dest));

    expect(coreDestStack).toBeDefined();
    expect(sandboxDestStack).toBeDefined();
    expect(sandboxDestStack?.controllingPlayer).toBe(coreDestStack?.controllingPlayer);
    expect(sandboxDestStack?.stackHeight).toBe(coreDestStack?.stackHeight);
    expect(sandboxDestStack?.capHeight).toBe(coreDestStack?.capHeight);

    // Markers left behind should match
    const coreMarkerAtOrigin = coreBoard.markers.get(positionToString(origin));
    const sandboxMarkerAtOrigin = sandboxBoard.markers.get(positionToString(origin));
    expect(sandboxMarkerAtOrigin).toEqual(coreMarkerAtOrigin);
  });
});
