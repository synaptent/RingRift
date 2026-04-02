import {
  ClientSandboxEngine,
  type SandboxConfig,
  type SandboxInteractionHandler,
} from '../../src/client/sandbox/ClientSandboxEngine';
import type { BoardType, PlayerChoice, PlayerChoiceResponseFor } from '../../src/shared/types/game';
import { deserializeGameState } from '../../src/shared/engine/contracts/serialization';

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

describe('ClientSandboxEngine – traceMode no_*_action moves', () => {
  it('advances from line_processing with explicit no_line_action', async () => {
    const serializedState = {
      id: 'sandbox-local',
      boardType: 'square8' as BoardType,
      rngSeed: 1842977649,
      board: {
        type: 'square8' as BoardType,
        size: 8,
        stacks: {},
        markers: {},
        collapsedSpaces: {},
        eliminatedRings: {},
        formedLines: [],
      },
      players: [
        {
          playerNumber: 1,
          ringsInHand: 10,
          eliminatedRings: 0,
          territorySpaces: 0,
          isActive: true,
        },
        {
          playerNumber: 2,
          ringsInHand: 10,
          eliminatedRings: 0,
          territorySpaces: 0,
          isActive: true,
        },
      ],
      currentPlayer: 1,
      currentPhase: 'line_processing',
      gameStatus: 'active',
      moveHistory: [],
      history: [],
      timeControl: { initialTime: 600, increment: 0, type: 'rapid' },
      spectators: [],
      createdAt: new Date().toISOString(),
      lastMoveAt: new Date().toISOString(),
      isRated: false,
      maxPlayers: 2,
      totalRingsInPlay: 20,
      totalRingsEliminated: 0,
      victoryThreshold: 18,
      territoryVictoryThreshold: 33,
      rulesOptions: { swapRuleEnabled: true },
    } as any;

    const gameState = deserializeGameState(serializedState);
    const config: SandboxConfig = {
      boardType: gameState.boardType,
      numPlayers: gameState.players.length,
      playerKinds: gameState.players.map(() => 'ai'),
    };

    const engine = new ClientSandboxEngine({
      config,
      interactionHandler,
      traceMode: true,
    });
    engine.initFromSerializedState(gameState as any, config.playerKinds, interactionHandler);

    expect(engine.getGameState().currentPhase).toBe('line_processing');

    await engine.maybeRunAITurn(() => 0.5);

    expect(engine.getLastAIMoveForTesting()?.type).toBe('no_line_action');
    const after = engine.getGameState();
    expect(after.currentPhase).toBe('territory_processing');
    expect(after.currentPlayer).toBe(1);

    // Regression guard: adapter-driven moves should be recorded exactly once
    // in both moveHistory and structured history.
    expect(after.moveHistory).toHaveLength(1);
    expect(after.history).toHaveLength(1);
    expect(after.moveHistory[0]?.type).toBe('no_line_action');
    expect(after.moveHistory[0]?.moveNumber).toBe(1);
    expect(after.history[0]?.action.type).toBe('no_line_action');
    expect(after.history[0]?.moveNumber).toBe(1);
  });

  it('advances from a real-world square8 line_processing snapshot (no lines)', async () => {
    const serializedState = {
      gameId: 'sandbox-local',
      board: {
        type: 'square8' as BoardType,
        size: 8,
        stacks: {
          '5,0': {
            position: { x: 5, y: 0 },
            rings: [1, 1, 1, 2, 1, 1],
            stackHeight: 6,
            capHeight: 3,
            controllingPlayer: 1,
          },
          '3,0': {
            position: { x: 3, y: 0 },
            rings: [1],
            stackHeight: 1,
            capHeight: 1,
            controllingPlayer: 1,
          },
          '7,1': {
            position: { x: 7, y: 1 },
            rings: [1, 1, 1, 2],
            stackHeight: 4,
            capHeight: 3,
            controllingPlayer: 1,
          },
          '0,7': {
            position: { x: 0, y: 7 },
            rings: [2, 2, 2, 1, 2],
            stackHeight: 5,
            capHeight: 3,
            controllingPlayer: 2,
          },
          '5,7': {
            position: { x: 5, y: 7 },
            rings: [2],
            stackHeight: 1,
            capHeight: 1,
            controllingPlayer: 2,
          },
          '4,1': {
            position: { x: 4, y: 1 },
            rings: [1, 1],
            stackHeight: 2,
            capHeight: 2,
            controllingPlayer: 1,
          },
        },
        markers: {
          '2,5': { position: { x: 2, y: 5 }, player: 1, type: 'regular' },
          '7,2': { position: { x: 7, y: 2 }, player: 1, type: 'regular' },
          '4,5': { position: { x: 4, y: 5 }, player: 2, type: 'regular' },
          '0,0': { position: { x: 0, y: 0 }, player: 1, type: 'regular' },
          '5,1': { position: { x: 5, y: 1 }, player: 1, type: 'regular' },
          '4,2': { position: { x: 4, y: 2 }, player: 2, type: 'regular' },
          '0,6': { position: { x: 0, y: 6 }, player: 1, type: 'regular' },
          '0,1': { position: { x: 0, y: 1 }, player: 1, type: 'regular' },
          '1,4': { position: { x: 1, y: 4 }, player: 2, type: 'regular' },
          '2,4': { position: { x: 2, y: 4 }, player: 1, type: 'regular' },
          '0,4': { position: { x: 0, y: 4 }, player: 2, type: 'regular' },
          '6,7': { position: { x: 6, y: 7 }, player: 2, type: 'regular' },
          '2,3': { position: { x: 2, y: 3 }, player: 1, type: 'regular' },
        },
        collapsedSpaces: { '4,0': 1, '4,7': 2, '3,7': 2 },
        eliminatedRings: { '1': 1, '2': 1 },
        formedLines: [],
      },
      players: [
        { playerNumber: 1, ringsInHand: 5, eliminatedRings: 1, territorySpaces: 0, isActive: true },
        {
          playerNumber: 2,
          ringsInHand: 10,
          eliminatedRings: 1,
          territorySpaces: 0,
          isActive: true,
        },
      ],
      currentPlayer: 1,
      currentPhase: 'line_processing',
      turnNumber: 50,
      moveHistory: [],
      gameStatus: 'active',
      victoryThreshold: 18,
      territoryVictoryThreshold: 33,
      totalRingsEliminated: 2,
      rulesOptions: { swapRuleEnabled: true },
    } as any;

    const gameState = deserializeGameState(serializedState);
    const config: SandboxConfig = {
      boardType: gameState.boardType,
      numPlayers: gameState.players.length,
      playerKinds: gameState.players.map(() => 'ai'),
    };

    const engine = new ClientSandboxEngine({
      config,
      interactionHandler,
      traceMode: true,
    });
    engine.initFromSerializedState(gameState as any, config.playerKinds, interactionHandler);

    expect(engine.getGameState().currentPhase).toBe('line_processing');

    await engine.maybeRunAITurn(() => 0.25);

    expect(engine.getLastAIMoveForTesting()?.type).toBe('no_line_action');
    expect(engine.getGameState().currentPhase).toBe('territory_processing');
  });

  it('advances from territory_processing with explicit no_territory_action', async () => {
    const serializedState = {
      id: 'sandbox-local',
      boardType: 'square8' as BoardType,
      rngSeed: 1842977649,
      board: {
        type: 'square8' as BoardType,
        size: 8,
        stacks: {},
        markers: {},
        collapsedSpaces: {},
        eliminatedRings: {},
        formedLines: [],
      },
      players: [
        {
          playerNumber: 1,
          ringsInHand: 10,
          eliminatedRings: 0,
          territorySpaces: 0,
          isActive: true,
        },
        {
          playerNumber: 2,
          ringsInHand: 10,
          eliminatedRings: 0,
          territorySpaces: 0,
          isActive: true,
        },
      ],
      currentPlayer: 1,
      currentPhase: 'territory_processing',
      gameStatus: 'active',
      moveHistory: [],
      history: [],
      timeControl: { initialTime: 600, increment: 0, type: 'rapid' },
      spectators: [],
      createdAt: new Date().toISOString(),
      lastMoveAt: new Date().toISOString(),
      isRated: false,
      maxPlayers: 2,
      totalRingsInPlay: 20,
      totalRingsEliminated: 0,
      victoryThreshold: 18,
      territoryVictoryThreshold: 33,
      rulesOptions: { swapRuleEnabled: true },
    } as any;

    const gameState = deserializeGameState(serializedState);
    const config: SandboxConfig = {
      boardType: gameState.boardType,
      numPlayers: gameState.players.length,
      playerKinds: gameState.players.map(() => 'ai'),
    };

    const engine = new ClientSandboxEngine({
      config,
      interactionHandler,
      traceMode: true,
    });
    engine.initFromSerializedState(gameState as any, config.playerKinds, interactionHandler);

    expect(engine.getGameState().currentPhase).toBe('territory_processing');

    await engine.maybeRunAITurn(() => 0.5);

    expect(engine.getLastAIMoveForTesting()?.type).toBe('no_territory_action');
    expect(engine.getGameState().currentPhase).toBe('ring_placement');
    expect(engine.getGameState().currentPlayer).toBe(2);
  });
});
