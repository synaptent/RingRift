import {
  ClientSandboxEngine,
  SandboxConfig,
  SandboxInteractionHandler,
} from '../../src/client/sandbox/ClientSandboxEngine';
import type { GameState, BoardType, Position, RingStack, Move } from '../../src/shared/engine';
import {
  BOARD_CONFIGS,
  positionToString,
  hashGameState,
  validatePlacementAggregate,
  applyPlacementMoveAggregate,
  evaluateSkipPlacementEligibilityAggregate,
} from '../../src/shared/engine';
import type { PlayerChoiceResponseFor, CaptureDirectionChoice } from '../../src/shared/types/game';

describe('ClientSandboxEngine placement parity with shared PlacementAggregate', () => {
  const boardType: BoardType = 'square8';

  function createEngine(): ClientSandboxEngine {
    const config: SandboxConfig = {
      boardType,
      numPlayers: 2,
      playerKinds: ['human', 'human'],
    };

    const handler: SandboxInteractionHandler = {
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

    const engine = new ClientSandboxEngine({ config, interactionHandler: handler });
    engine.disableOrchestratorAdapter();
    return engine;
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

  it('applies legal placement via sandbox helpers with the same outcome as PlacementAggregate', async () => {
    const engine = createEngine();
    const engineAny = engine as any;
    const internalState: GameState = engineAny.gameState as GameState;

    internalState.currentPlayer = 1;
    internalState.currentPhase = 'ring_placement';

    const board = internalState.board;
    board.stacks.clear();
    board.markers.clear();
    board.collapsedSpaces.clear();

    const placementPos: Position = { x: 2, y: 2 };
    const config = BOARD_CONFIGS[boardType];
    expect(placementPos.x).toBeGreaterThanOrEqual(0);
    expect(placementPos.x).toBeLessThan(config.size);
    expect(placementPos.y).toBeGreaterThanOrEqual(0);
    expect(placementPos.y).toBeLessThan(config.size);

    const snapshotBefore: GameState = engine.getGameState();

    const validation = validatePlacementAggregate(snapshotBefore, {
      type: 'PLACE_RING',
      playerId: 1,
      position: placementPos,
      count: 1,
    } as any);
    expect(validation.valid).toBe(true);

    const aggregateOutcome = applyPlacementMoveAggregate(snapshotBefore, {
      id: 'place-1',
      type: 'place_ring',
      player: 1,
      from: undefined,
      to: placementPos,
      placementCount: 1,
      timestamp: new Date(),
      thinkTime: 0,
      moveNumber: 1,
    } as Move);

    engineAny.gameState = snapshotBefore;
    const placed = await engine.tryPlaceRings(placementPos, 1);
    expect(placed).toBe(true);

    const sandboxStateAfter: GameState = engine.getGameState();

    const normalisePhase = (state: GameState): GameState => ({
      ...state,
      currentPhase: 'ring_placement',
    });

    const coreHash = hashGameState(normalisePhase(aggregateOutcome.nextState));
    const sandboxHash = hashGameState(normalisePhase(sandboxStateAfter));

    expect(sandboxHash).toEqual(coreHash);
  });

  it('applies skip_placement in sandbox only when aggregate deems it eligible', async () => {
    const engine = createEngine();
    const engineAny = engine as any;
    const state: GameState = engineAny.gameState as GameState;

    state.currentPlayer = 1;
    state.currentPhase = 'ring_placement';

    const board = state.board;
    board.stacks.clear();
    board.markers.clear();
    board.collapsedSpaces.clear();

    const stackPos: Position = { x: 2, y: 2 };
    makeStack(1, 2, stackPos, state);

    const eligible = evaluateSkipPlacementEligibilityAggregate(state, 1);
    expect(eligible.eligible).toBe(true);

    const beforePhase = state.currentPhase;
    const skipMove: Move = {
      id: 'skip-1',
      type: 'skip_placement',
      player: 1,
      from: undefined,
      to: { x: 0, y: 0 },
      timestamp: new Date(),
      thinkTime: 0,
      moveNumber: 1,
    } as Move;

    await engine.applyCanonicalMove(skipMove);

    const afterState = engine.getGameState();
    expect(beforePhase).toBe('ring_placement');
    expect(afterState.currentPhase).toBe('movement');

    const stateBlocked: GameState = engineAny.gameState as GameState;
    stateBlocked.currentPlayer = 2;
    stateBlocked.currentPhase = 'ring_placement';
    const player2 = stateBlocked.players.find((p) => p.playerNumber === 2)!;
    player2.ringsInHand = 0;

    const blockedEligible = evaluateSkipPlacementEligibilityAggregate(stateBlocked, 2);
    expect(blockedEligible.eligible).toBe(false);

    const blockedPhaseBefore = stateBlocked.currentPhase;
    const blockedSkip: Move = {
      id: 'skip-2',
      type: 'skip_placement',
      player: 2,
      from: undefined,
      to: { x: 0, y: 0 },
      timestamp: new Date(),
      thinkTime: 0,
      moveNumber: 1,
    } as Move;

    await engine.applyCanonicalMove(blockedSkip);

    const blockedAfter = engine.getGameState();
    expect(blockedAfter.currentPhase).toBe(blockedPhaseBefore);
  });
});
