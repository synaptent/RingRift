import {
  ClientSandboxEngine,
  type SandboxConfig,
  type SandboxInteractionHandler,
} from '../../src/client/sandbox/ClientSandboxEngine';
import type {
  GameState,
  BoardType,
  PlayerChoice,
  PlayerChoiceResponseFor,
} from '../../src/shared/types/game';
import { deserializeGameState } from '../../src/shared/engine/contracts/serialization';
import {
  maybeRunAITurnSandbox,
  SANDBOX_STALL_WINDOW_STEPS,
} from '../../src/client/sandbox/sandboxAI';

// Minimal interaction handler: always picks the first option for any choice.
const interactionHandler: SandboxInteractionHandler = {
  async requestChoice<TChoice extends PlayerChoice>(
    choice: TChoice
  ): Promise<PlayerChoiceResponseFor<TChoice>> {
    const anyChoice = choice as any;
    const options = (anyChoice.options as any[]) ?? [];
    const selectedOption = options.length > 0 ? options[0] : undefined;
    return {
      choiceId: anyChoice.id,
      playerNumber: anyChoice.playerNumber,
      choiceType: anyChoice.type,
      selectedOption,
    } as PlayerChoiceResponseFor<TChoice>;
  },
};

describe('ClientSandboxEngine – AI stall completion normalization', () => {
  it('preserves active game status when stall detector runs but game has valid moves', async () => {
    // This fixture represents a sandbox AI game with players still having rings
    // and stacks on the board. Even if the mock hooks return no moves and the
    // stall window is exceeded, the stall detector should NOT force-complete
    // the game because:
    // 1. isANMState() checks the actual game state (not mock hooks)
    // 2. The state has valid moves according to the engine
    //
    // The stall detector is conservative: it only marks games as completed
    // when both isANMState() returns true AND evaluateVictory() confirms
    // the game is actually over. This prevents incorrectly ending games that
    // have valid moves but where the AI happens to not find them.
    const serializedState = {
      id: 'sandbox-local',
      boardType: 'square8' as BoardType,
      rngSeed: 173580855,
      board: {
        type: 'square8' as BoardType,
        size: 8,
        stacks: {
          '4,6': {
            position: { x: 4, y: 6 },
            rings: [2, 2, 2],
            stackHeight: 3,
            capHeight: 3,
            controllingPlayer: 2,
          },
          '7,0': {
            position: { x: 7, y: 0 },
            rings: [1, 1, 1, 1],
            stackHeight: 4,
            capHeight: 4,
            controllingPlayer: 1,
          },
          '2,6': {
            position: { x: 2, y: 6 },
            rings: [2, 2, 2],
            stackHeight: 3,
            capHeight: 3,
            controllingPlayer: 2,
          },
        },
        markers: {
          '2,3': { position: { x: 2, y: 3 }, player: 2, type: 'regular' },
          '1,6': { position: { x: 1, y: 6 }, player: 2, type: 'regular' },
          '7,5': { position: { x: 7, y: 5 }, player: 1, type: 'regular' },
          '2,1': { position: { x: 2, y: 1 }, player: 2, type: 'regular' },
        },
        collapsedSpaces: { '7,1': 1 },
        eliminatedRings: {},
        formedLines: [],
      },
      players: [
        {
          playerNumber: 1,
          ringsInHand: 14,
          eliminatedRings: 0,
          territorySpaces: 0,
          isActive: true,
        },
        {
          playerNumber: 2,
          ringsInHand: 12,
          eliminatedRings: 0,
          territorySpaces: 0,
          isActive: true,
        },
      ],
      currentPlayer: 2,
      currentPhase: 'chain_capture',
      gameStatus: 'active',
      moveHistory: [],
      history: [],
      timeControl: {
        initialTime: 600,
        increment: 0,
        type: 'rapid',
      },
      spectators: [],
      createdAt: new Date().toISOString(),
      lastMoveAt: new Date().toISOString(),
      isRated: false,
      maxPlayers: 2,
      totalRingsInPlay: 10,
      totalRingsEliminated: 0,
      victoryThreshold: 18, // RR-CANON-R061: ringsPerPlayer
      territoryVictoryThreshold: 33,
      chainCapturePosition: { x: 7, y: 5 },
    } as any;

    const gameState: GameState = deserializeGameState(serializedState);

    const config: SandboxConfig = {
      boardType: gameState.boardType,
      numPlayers: gameState.players.length,
      playerKinds: gameState.players.map(() => 'ai'),
    };

    const engine = new ClientSandboxEngine({
      config,
      interactionHandler,
      traceMode: false,
    });

    engine.initFromSerializedState(gameState as any, config.playerKinds, interactionHandler);

    const before = engine.getGameState();
    expect(before.gameStatus).toBe('active');
    expect(before.currentPhase).toBe('chain_capture');
    expect(before.chainCapturePosition).toEqual({ x: 7, y: 5 });

    // Mock hooks that return no moves. The stall detector will see no-op turns
    // but should NOT force-complete the game because isANMState and evaluateVictory
    // use the actual game state which has valid moves available.
    const hooks: any = {
      getGameState: () => engine.getGameState(),
      setGameState: (state: GameState) => {
        (engine as any).gameState = state;
      },
      setLastAIMove: () => undefined,
      setSelectedStackKey: () => undefined,
      getMustMoveFromStackKey: () => undefined,
      applyCanonicalMove: async () => {},
      maybeProcessForcedEliminationForCurrentPlayer: () => false,
      hasPendingTerritorySelfElimination: () => false,
      hasPendingLineRewardElimination: () => false,
      canCurrentPlayerSwapSides: () => false,
      applySwapSidesForCurrentPlayer: () => false,
      getPlayerStacks: () => [],
      hasAnyLegalMoveOrCaptureFrom: () => false,
      enumerateLegalRingPlacements: () => [],
      getValidMovesForCurrentPlayer: () => [],
      createHypotheticalBoardWithPlacement: (board: any) => board,
      tryPlaceRings: async () => false,
      enumerateCaptureSegmentsFrom: () => [],
      enumerateSimpleMovementLandings: () => [],
      handleMovementClick: async () => {},
      appendHistoryEntry: () => {},
    };

    for (let i = 0; i < SANDBOX_STALL_WINDOW_STEPS; i += 1) {
      await maybeRunAITurnSandbox(hooks, () => 0.5);
    }

    // Game should remain active because the actual state has valid moves
    // (players have rings, stacks exist). The stall detector is conservative.
    const after = engine.getGameState();
    expect(after.gameStatus).toBe('active');
    // Phase and capture position preserved since game wasn't force-completed
    expect(after.currentPhase).toBe('chain_capture');
    expect(after.chainCapturePosition).toEqual({ x: 7, y: 5 });
  });

  it('normalizes an imported stalled-completion fixture on initFromSerializedState', () => {
    // This mirrors a ringrift_sandbox_fixture_v1 snapshot where the inner
    // SerializedGameState has gameStatus === 'completed' but still reflects
    // a mid-capture phase/cursor. initFromSerializedState should normalise
    // the terminal state on load so sandbox HUD/hosts do not present an
    // already-completed game as if it were awaiting capture continuation.
    const serializedState = {
      id: 'sandbox-local',
      boardType: 'square8' as BoardType,
      rngSeed: 173580855,
      board: {
        type: 'square8' as BoardType,
        size: 8,
        stacks: {
          '4,6': {
            position: { x: 4, y: 6 },
            rings: [2, 2, 2],
            stackHeight: 3,
            capHeight: 3,
            controllingPlayer: 2,
          },
          '7,0': {
            position: { x: 7, y: 0 },
            rings: [1, 1, 1, 1],
            stackHeight: 4,
            capHeight: 4,
            controllingPlayer: 1,
          },
          '2,6': {
            position: { x: 2, y: 6 },
            rings: [2, 2, 2],
            stackHeight: 3,
            capHeight: 3,
            controllingPlayer: 2,
          },
        },
        markers: {
          '2,3': { position: { x: 2, y: 3 }, player: 2, type: 'regular' },
          '1,6': { position: { x: 1, y: 6 }, player: 2, type: 'regular' },
          '7,5': { position: { x: 7, y: 5 }, player: 1, type: 'regular' },
          '2,1': { position: { x: 2, y: 1 }, player: 2, type: 'regular' },
        },
        collapsedSpaces: { '7,1': 1 },
        eliminatedRings: {},
        formedLines: [],
      },
      players: [
        {
          playerNumber: 1,
          ringsInHand: 14,
          eliminatedRings: 0,
          territorySpaces: 0,
          isActive: true,
        },
        {
          playerNumber: 2,
          ringsInHand: 12,
          eliminatedRings: 0,
          territorySpaces: 0,
          isActive: true,
        },
      ],
      currentPlayer: 2,
      currentPhase: 'capture',
      gameStatus: 'completed',
      moveHistory: [],
      history: [],
      timeControl: {
        initialTime: 600,
        increment: 0,
        type: 'rapid',
      },
      spectators: [],
      createdAt: new Date().toISOString(),
      lastMoveAt: new Date().toISOString(),
      isRated: false,
      maxPlayers: 2,
      totalRingsInPlay: 10,
      totalRingsEliminated: 0,
      victoryThreshold: 18, // RR-CANON-R061: ringsPerPlayer
      territoryVictoryThreshold: 33,
      chainCapturePosition: { x: 7, y: 5 },
    } as any;

    const gameState: GameState = deserializeGameState(serializedState);

    const config: SandboxConfig = {
      boardType: gameState.boardType,
      numPlayers: gameState.players.length,
      playerKinds: gameState.players.map(() => 'ai'),
    };

    const engine = new ClientSandboxEngine({
      config,
      interactionHandler,
      traceMode: false,
    });

    engine.initFromSerializedState(gameState as any, config.playerKinds, interactionHandler);

    const after = engine.getGameState();
    expect(after.gameStatus).toBe('completed');
    expect(after.currentPhase).toBe('ring_placement');
    expect(after.chainCapturePosition).toBeUndefined();
    expect((after as any).mustMoveFromStackKey).toBeUndefined();
  });
});
