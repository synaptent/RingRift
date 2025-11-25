import { GameEngine } from '../../src/server/game/GameEngine';
import {
  BoardType,
  BoardState,
  GameState,
  Player,
  Position,
  TimeControl,
  positionToString,
} from '../../src/shared/types/game';
import {
  ClientSandboxEngine,
  SandboxConfig,
  SandboxInteractionHandler,
} from '../../src/client/sandbox/ClientSandboxEngine';
import {
  createTestBoard,
  createTestGameState,
  createTestPlayer,
  addStack,
  addMarker,
  pos,
} from '../utils/fixtures';

// Classification: backend ↔ sandbox territory-parity harness on a canonical 19x19
// disconnected-region scenario, validating shared territory detection and processing.

// NOTE: Diagnostic parity harness; currently skipped because backend processDisconnectedRegions
// does not yet mirror the sandbox territory loop from this synthetic setup.
describe('Territory parity – GameEngine vs ClientSandboxEngine (square19)', () => {
  const boardType: BoardType = 'square19';
  const timeControl: TimeControl = { initialTime: 600, increment: 0, type: 'blitz' };

  interface TerritoryFixture {
    baseState: GameState;
    interiorCoords: Position[];
    borderCoords: Position[];
  }

  function buildTerritoryFixture(): TerritoryFixture {
    const board: BoardState = createTestBoard(boardType);

    const players: Player[] = [createTestPlayer(1), createTestPlayer(2), createTestPlayer(3)];

    const interiorCoords: Position[] = [];
    for (let x = 5; x <= 7; x++) {
      for (let y = 5; y <= 7; y++) {
        const p = pos(x, y);
        interiorCoords.push(p);
        addStack(board, p, 2, 1);
      }
    }

    const borderCoords: Position[] = [];
    for (let x = 4; x <= 8; x++) {
      borderCoords.push(pos(x, 4));
      borderCoords.push(pos(x, 8));
    }
    for (let y = 5; y <= 7; y++) {
      borderCoords.push(pos(4, y));
      borderCoords.push(pos(8, y));
    }
    borderCoords.forEach((p) => addMarker(board, p, 1));

    const outsideP1 = pos(1, 1);
    addStack(board, outsideP1, 1, 1);

    const outsideP3 = pos(0, 0);
    addStack(board, outsideP3, 3, 3);

    const baseState: GameState = createTestGameState({
      boardType,
      board,
      players,
      currentPlayer: 1,
      currentPhase: 'territory_processing',
      totalRingsEliminated: 0,
    });

    return { baseState, interiorCoords, borderCoords };
  }

  function cloneGameState(state: GameState): GameState {
    const board = state.board;
    const clonedBoard: BoardState = {
      ...board,
      stacks: new Map(board.stacks),
      markers: new Map(board.markers),
      collapsedSpaces: new Map(board.collapsedSpaces),
      territories: new Map(board.territories),
      formedLines: [...board.formedLines],
      eliminatedRings: { ...board.eliminatedRings },
    };

    return {
      ...state,
      board: clonedBoard,
      players: state.players.map((p) => ({ ...p })),
      moveHistory: [...state.moveHistory],
      history: [...state.history],
    };
  }

  function mapByPlayerNumber(players: Player[]): Record<number, Player> {
    const out: Record<number, Player> = {};
    for (const p of players) {
      out[p.playerNumber] = p;
    }
    return out;
  }

  function keysOfMap(map: Map<string, unknown>): string[] {
    return Array.from(map.keys()).sort();
  }

  test('disconnected-region processing yields identical collapsed spaces and elimination for backend and sandbox', async () => {
    const { baseState, interiorCoords, borderCoords } = buildTerritoryFixture();

    const backendPlayersClone = baseState.players.map((p) => ({ ...p }));
    const backendEngine = new GameEngine(
      'territory-parity-square19',
      boardType,
      backendPlayersClone,
      timeControl,
      false
    );
    const backendAny: any = backendEngine;
    backendAny.gameState = cloneGameState(baseState);

    const sandboxConfig: SandboxConfig = {
      boardType,
      numPlayers: baseState.players.length,
      playerKinds: baseState.players.map((p) => p.type as 'human' | 'ai'),
    };
    const sandboxHandler: SandboxInteractionHandler = {
      async requestChoice(choice: any) {
        const options = ((choice as any).options as any[]) ?? [];
        const selectedOption = options.length > 0 ? options[0] : undefined;
        return {
          choiceId: (choice as any).id,
          playerNumber: (choice as any).playerNumber,
          choiceType: (choice as any).type,
          selectedOption,
        };
      },
    };
    const sandboxEngine = new ClientSandboxEngine({
      config: sandboxConfig,
      interactionHandler: sandboxHandler,
      traceMode: false,
    });
    const sandboxAny: any = sandboxEngine;
    sandboxAny.gameState = cloneGameState(baseState);

    const backendInitial = backendEngine.getGameState();
    const sandboxInitial = sandboxEngine.getGameState();
    expect(backendInitial.boardType).toBe(sandboxInitial.boardType);
    expect(backendInitial.currentPlayer).toBe(sandboxInitial.currentPlayer);
    expect(backendInitial.currentPhase).toBe('territory_processing');
    expect(sandboxInitial.currentPhase).toBe('territory_processing');

    await (backendAny as any).processDisconnectedRegions();
    await sandboxAny.processDisconnectedRegionsForCurrentPlayer();

    // Sandbox uses unified move model where elimination is a separate step.
    // Manually trigger elimination to match backend legacy behavior.
    if (sandboxAny._pendingTerritorySelfElimination) {
      sandboxAny.forceEliminateCap(1);
    }

    const backendFinal = backendEngine.getGameState();
    const sandboxFinal = sandboxEngine.getGameState();

    const backendCollapsed = backendFinal.board.collapsedSpaces;
    const sandboxCollapsed = sandboxFinal.board.collapsedSpaces;

    expect(keysOfMap(backendCollapsed)).toEqual(keysOfMap(sandboxCollapsed));
    for (const [key, owner] of backendCollapsed.entries()) {
      expect(sandboxCollapsed.get(key)).toBe(owner);
    }

    const backendStacks = backendFinal.board.stacks;
    const sandboxStacks = sandboxFinal.board.stacks;
    expect(keysOfMap(backendStacks as any)).toEqual(keysOfMap(sandboxStacks as any));
    for (const [key, stack] of backendStacks.entries()) {
      const other = sandboxStacks.get(key);
      if (!stack && !other) continue;
      expect(other).toBeDefined();
      if (other) {
        expect(other.controllingPlayer).toBe(stack.controllingPlayer);
        expect(other.stackHeight).toBe(stack.stackHeight);
        expect(other.capHeight).toBe(stack.capHeight);
      }
    }

    expect(backendFinal.board.eliminatedRings).toEqual(sandboxFinal.board.eliminatedRings);
    expect(backendFinal.totalRingsEliminated).toBe(sandboxFinal.totalRingsEliminated);

    const backendByPlayer = mapByPlayerNumber(backendFinal.players);
    const sandboxByPlayer = mapByPlayerNumber(sandboxFinal.players);

    for (const playerNumber of Object.keys(backendByPlayer)) {
      const n = Number(playerNumber);
      const b = backendByPlayer[n];
      const s = sandboxByPlayer[n];
      expect(s).toBeDefined();
      if (s) {
        expect(s.territorySpaces).toBe(b.territorySpaces);
        expect(s.eliminatedRings).toBe(b.eliminatedRings);
      }
    }

    const interiorKeys = new Set(
      interiorCoords.map((p) => (p.z !== undefined ? `${p.x},${p.y},${p.z}` : `${p.x},${p.y}`))
    );
    const borderKeys = new Set(
      borderCoords.map((p) => (p.z !== undefined ? `${p.x},${p.y},${p.z}` : `${p.x},${p.y}`))
    );

    interiorCoords.forEach((p) => {
      const key = p.z !== undefined ? `${p.x},${p.y},${p.z}` : `${p.x},${p.y}`;
      expect(backendCollapsed.get(key)).toBe(1);
      expect(sandboxCollapsed.get(key)).toBe(1);
    });
    borderCoords.forEach((p) => {
      const key = p.z !== undefined ? `${p.x},${p.y},${p.z}` : `${p.x},${p.y}`;
      expect(backendCollapsed.get(key)).toBe(1);
      expect(sandboxCollapsed.get(key)).toBe(1);
    });

    const expectedTerritory = interiorKeys.size + borderKeys.size;
    const backendTerritory =
      backendFinal.players.find((p) => p.playerNumber === 1)?.territorySpaces ?? 0;
    const sandboxTerritory =
      sandboxFinal.players.find((p) => p.playerNumber === 1)?.territorySpaces ?? 0;
    expect(backendTerritory).toBe(expectedTerritory);
    expect(sandboxTerritory).toBe(expectedTerritory);

    // Sanity: collapsed keys encode exactly the expected interior + border coordinates.
    const collapsedKeySet = new Set(Array.from(backendCollapsed.keys()));
    interiorKeys.forEach((k) => expect(collapsedKeySet.has(k)).toBe(true));
    borderKeys.forEach((k) => expect(collapsedKeySet.has(k)).toBe(true));
  });
});
