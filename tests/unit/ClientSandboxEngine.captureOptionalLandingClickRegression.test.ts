import {
  ClientSandboxEngine,
  SandboxConfig,
  SandboxInteractionHandler,
} from '../../src/client/sandbox/ClientSandboxEngine';
import type {
  BoardType,
  GameState,
  PlayerChoice,
  PlayerChoiceResponseFor,
} from '../../src/shared/types/game';
import { deserializeGameState } from '../../src/shared/engine/contracts/serialization';
import { positionToString } from '../../src/shared/types/game';

describe('ClientSandboxEngine optional capture landing click regression (sandbox)', () => {
  function createEngineWithHandler(): {
    engine: ClientSandboxEngine;
    handler: SandboxInteractionHandler;
  } {
    const handler: SandboxInteractionHandler = {
      async requestChoice<TChoice extends PlayerChoice>(
        choice: TChoice
      ): Promise<PlayerChoiceResponseFor<TChoice>> {
        const anyChoice = choice as any;
        const options: any[] = (anyChoice.options as any[]) ?? [];
        const selectedOption = options.length > 0 ? options[0] : undefined;

        return {
          choiceId: anyChoice.id,
          playerNumber: anyChoice.playerNumber,
          choiceType: anyChoice.type,
          selectedOption,
        } as PlayerChoiceResponseFor<TChoice>;
      },
    };

    const config: SandboxConfig = {
      boardType: 'square8',
      numPlayers: 2,
      playerKinds: ['human', 'ai'],
    };

    const engine = new ClientSandboxEngine({ config, interactionHandler: handler });
    return { engine, handler };
  }

  test('clicking a highlighted capture landing without pre-selecting the stack applies overtaking_capture (seed 1546248154)', async () => {
    const { engine, handler } = createEngineWithHandler();
    const engineAny: any = engine;

    // Serialized state lifted from the user-provided ringrift_sandbox_fixture_v1
    // where Player 1 has just moved from d3→e3 and an optional capture is
    // available from the stack at e3.
    const serializedState = {
      id: 'sandbox-local',
      boardType: 'square8' as BoardType,
      rngSeed: 1546248154,
      board: {
        type: 'square8' as BoardType,
        size: 8,
        stacks: {
          '4,4': {
            position: { x: 4, y: 4 },
            rings: [1],
            stackHeight: 1,
            capHeight: 1,
            controllingPlayer: 1,
          },
          '2,7': {
            position: { x: 2, y: 7 },
            rings: [2, 2],
            stackHeight: 2,
            capHeight: 2,
            controllingPlayer: 2,
          },
          '4,3': {
            position: { x: 4, y: 3 },
            rings: [1],
            stackHeight: 1,
            capHeight: 1,
            controllingPlayer: 1,
          },
        },
        markers: {
          '3,4': {
            position: { x: 3, y: 4 },
            player: 1,
            type: 'regular',
          },
          '2,5': {
            position: { x: 2, y: 5 },
            player: 2,
            type: 'regular',
          },
          '3,3': {
            position: { x: 3, y: 3 },
            player: 1,
            type: 'regular',
          },
        },
        collapsedSpaces: {},
        eliminatedRings: {},
        formedLines: [],
      },
      players: [
        {
          playerNumber: 1,
          ringsInHand: 16,
          eliminatedRings: 0,
          territorySpaces: 0,
          isActive: true,
        },
        {
          playerNumber: 2,
          ringsInHand: 16,
          eliminatedRings: 0,
          territorySpaces: 0,
          isActive: true,
        },
      ],
      currentPlayer: 1,
      currentPhase: 'capture' as const,
      turnNumber: 5,
      moveHistory: [
        {
          id: 'move-3,4-4,4-1',
          type: 'move_stack',
          player: 1,
          from: { x: 3, y: 4 },
          to: { x: 4, y: 4 },
          moveNumber: 1,
          timestamp: '2025-12-05T09:20:23.634Z',
          thinkTime: 0,
        },
        {
          id: '',
          type: 'place_ring',
          player: 2,
          to: { x: 2, y: 5 },
          placementCount: 2,
          timestamp: '2025-12-05T09:20:23.641Z',
          thinkTime: 0,
          moveNumber: 2,
        },
        {
          id: '',
          type: 'move_stack',
          player: 2,
          from: { x: 2, y: 5 },
          to: { x: 2, y: 7 },
          moveNumber: 3,
          timestamp: '2025-12-05T09:20:23.747Z',
          thinkTime: 0,
        },
        {
          id: 'move-3,3-4,3-4',
          type: 'move_stack',
          player: 1,
          from: { x: 3, y: 3 },
          to: { x: 4, y: 3 },
          moveNumber: 4,
          timestamp: '2025-12-05T09:20:26.773Z',
          thinkTime: 0,
        },
      ],
      gameStatus: 'active' as const,
      victoryThreshold: 18, // RR-CANON-R061: ringsPerPlayer
      territoryVictoryThreshold: 33,
      totalRingsEliminated: 0,
    };

    const gameState: GameState = deserializeGameState(serializedState as any);

    const config: SandboxConfig = {
      boardType: gameState.boardType,
      numPlayers: gameState.players.length,
      playerKinds: ['human', 'ai'],
    };

    const seededEngine = new ClientSandboxEngine({
      config,
      interactionHandler: handler,
      traceMode: false,
    });

    seededEngine.initFromSerializedState(serializedState as any, config.playerKinds, handler);

    const before = seededEngine.getGameState();
    expect(before.currentPhase).toBe('capture');
    expect(before.currentPlayer).toBe(1);

    const engineSeededAny: any = seededEngine;

    const moves = seededEngine.getValidMoves(1);
    const captureMoves = moves.filter((m) => m.type === 'overtaking_capture');
    expect(captureMoves.length).toBeGreaterThan(0);
    const landing = captureMoves[0].to as Position;
    const landingKey = positionToString(landing);

    // No stack currently selected.
    expect(engineSeededAny._selectedStackKey).toBeUndefined();

    await engineSeededAny.handleMovementClick(landing);

    const after = seededEngine.getGameState();
    expect(after.moveHistory.length).toBeGreaterThan(before.moveHistory.length);
    const lastMove = after.moveHistory[after.moveHistory.length - 1];
    expect(lastMove.type).toBe('overtaking_capture');
    expect(lastMove.player).toBe(1);
    expect(lastMove.to && positionToString(lastMove.to)).toBe(landingKey);
  });

  test('clicking a highlighted capture landing without pre-selecting the stack applies overtaking_capture (seed 1365831209)', async () => {
    const { engine, handler } = createEngineWithHandler();
    const engineAny: any = engine;

    // Serialized state lifted from the later user-provided ringrift_sandbox_fixture_v1
    // where Player 1 has just moved from d3→e3 and an optional capture is
    // available from the stack at e3 against an adjacent opponent stack.
    const serializedState = {
      id: 'sandbox-local',
      boardType: 'square8' as BoardType,
      rngSeed: 1365831209,
      board: {
        type: 'square8' as BoardType,
        size: 8,
        stacks: {
          '4,4': {
            position: { x: 4, y: 4 },
            rings: [1],
            stackHeight: 1,
            capHeight: 1,
            controllingPlayer: 1,
          },
          '5,2': {
            position: { x: 5, y: 2 },
            rings: [2, 2],
            stackHeight: 2,
            capHeight: 2,
            controllingPlayer: 2,
          },
          '4,3': {
            position: { x: 4, y: 3 },
            rings: [1],
            stackHeight: 1,
            capHeight: 1,
            controllingPlayer: 1,
          },
        },
        markers: {
          '3,4': {
            position: { x: 3, y: 4 },
            player: 1,
            type: 'regular',
          },
          '5,6': {
            position: { x: 5, y: 6 },
            player: 2,
            type: 'regular',
          },
          '3,3': {
            position: { x: 3, y: 3 },
            player: 1,
            type: 'regular',
          },
        },
        collapsedSpaces: {},
        eliminatedRings: {},
        formedLines: [],
      },
      players: [
        {
          playerNumber: 1,
          ringsInHand: 16,
          eliminatedRings: 0,
          territorySpaces: 0,
          isActive: true,
        },
        {
          playerNumber: 2,
          ringsInHand: 16,
          eliminatedRings: 0,
          territorySpaces: 0,
          isActive: true,
        },
      ],
      currentPlayer: 1,
      currentPhase: 'capture' as const,
      turnNumber: 5,
      moveHistory: [
        {
          id: 'move-3,4-4,4-1',
          type: 'move_stack',
          player: 1,
          from: { x: 3, y: 4 },
          to: { x: 4, y: 4 },
          moveNumber: 1,
          timestamp: '2025-12-05T09:46:21.003Z',
          thinkTime: 0,
        },
        {
          id: '',
          type: 'place_ring',
          player: 2,
          to: { x: 5, y: 6 },
          placementCount: 2,
          timestamp: '2025-12-05T09:46:21.011Z',
          thinkTime: 0,
          moveNumber: 2,
        },
        {
          id: '',
          type: 'move_stack',
          player: 2,
          from: { x: 5, y: 6 },
          to: { x: 5, y: 2 },
          moveNumber: 3,
          timestamp: '2025-12-05T09:46:21.118Z',
          thinkTime: 0,
        },
        {
          id: 'move-3,3-4,3-4',
          type: 'move_stack',
          player: 1,
          from: { x: 3, y: 3 },
          to: { x: 4, y: 3 },
          moveNumber: 4,
          timestamp: '2025-12-05T09:46:23.229Z',
          thinkTime: 0,
        },
      ],
      gameStatus: 'active' as const,
      victoryThreshold: 18, // RR-CANON-R061: ringsPerPlayer
      territoryVictoryThreshold: 33,
      totalRingsEliminated: 0,
    };

    const gameState: GameState = deserializeGameState(serializedState as any);

    const config: SandboxConfig = {
      boardType: gameState.boardType,
      numPlayers: gameState.players.length,
      playerKinds: ['human', 'ai'],
    };

    const seededEngine = new ClientSandboxEngine({
      config,
      interactionHandler: handler,
      traceMode: false,
    });

    seededEngine.initFromSerializedState(serializedState as any, config.playerKinds, handler);

    const before = seededEngine.getGameState();
    expect(before.currentPhase).toBe('capture');
    expect(before.currentPlayer).toBe(1);

    const engineSeededAny: any = seededEngine;

    const moves = seededEngine.getValidMoves(1);
    const captureMoves = moves.filter((m) => m.type === 'overtaking_capture');
    expect(captureMoves.length).toBeGreaterThan(0);
    const landing = captureMoves[0].to as Position;
    const landingKey = positionToString(landing);

    // No stack currently selected.
    expect(engineSeededAny._selectedStackKey).toBeUndefined();

    await engineSeededAny.handleMovementClick(landing);

    const after = seededEngine.getGameState();
    expect(after.moveHistory.length).toBeGreaterThan(before.moveHistory.length);
    const lastMove = after.moveHistory[after.moveHistory.length - 1];
    expect(lastMove.type).toBe('overtaking_capture');
    expect(lastMove.player).toBe(1);
    expect(lastMove.to && positionToString(lastMove.to)).toBe(landingKey);
  });
});
