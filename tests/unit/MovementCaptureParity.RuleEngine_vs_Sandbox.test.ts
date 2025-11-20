import { GameEngine } from '../../src/server/game/GameEngine';
import { BoardManager } from '../../src/server/game/BoardManager';
import { RuleEngine } from '../../src/server/game/RuleEngine';
import {
  BoardType,
  BoardState,
  GameState,
  Move,
  Player,
  Position,
  positionToString,
} from '../../src/shared/types/game';
import {
  createTestBoard,
  createTestGameState,
  createTestPlayer,
  addStack,
} from '../utils/fixtures';
import { enumerateSimpleMovementLandings } from '../../src/client/sandbox/sandboxMovement';
import {
  enumerateCaptureSegmentsFromBoard,
  CaptureBoardAdapters,
  CaptureSegment,
} from '../../src/client/sandbox/sandboxCaptures';
import {
  ClientSandboxEngine,
  SandboxConfig,
  SandboxInteractionHandler,
} from '../../src/client/sandbox/ClientSandboxEngine';

describe('Movement/capture parity: stack at (4,7) over intermediate stack at (4,6)', () => {
  const boardType: BoardType = 'square8';

  function createBackendEngine(boardType: BoardType, players: Player[]): {
    engine: GameEngine;
    ruleEngine: RuleEngine;
    boardManager: BoardManager;
    gameState: GameState;
  } {
    const timeControl = { initialTime: 600, increment: 0, type: 'blitz' as const };
    const engine = new GameEngine('parity-test', boardType, players, timeControl, false);
    const anyEngine: any = engine;
    const gameState: GameState = anyEngine.gameState;
    const boardManager: BoardManager = anyEngine.boardManager;
    const ruleEngine = new RuleEngine(boardManager, boardType);

    return { engine, ruleEngine, boardManager, gameState };
  }

  function createSandboxEngine(boardType: BoardType, numPlayers: number): ClientSandboxEngine {
    const config: SandboxConfig = {
      boardType,
      numPlayers,
      playerKinds: Array.from({ length: numPlayers }, () => 'human'),
    };
    const handler: SandboxInteractionHandler = {
      async requestChoice<TChoice extends any>(choice: TChoice): Promise<any> {
        return {
          choiceId: (choice as any).id,
          playerNumber: (choice as any).playerNumber,
          choiceType: (choice as any).type,
          selectedOption: (choice as any).options?.[0],
        };
      },
    };
    return new ClientSandboxEngine({ config, interactionHandler: handler });
  }

  function createBackendState(): { state: GameState; manager: BoardManager; engine: RuleEngine } {
    const board = createTestBoard(boardType);

    const from: Position = { x: 4, y: 7 };
    const target: Position = { x: 4, y: 6 };

    // Attacker: player 1, height 2; Target: player 2, height 1.
    addStack(board, from, 1, 2);
    addStack(board, target, 2, 1);

    const players = [
      createTestPlayer(1, { type: 'human', ringsInHand: 0 }),
      createTestPlayer(2, { type: 'human', ringsInHand: 0 }),
    ];

    const state = createTestGameState({
      boardType,
      board,
      players,
      currentPlayer: 1,
      currentPhase: 'movement',
    });

    const manager = new BoardManager(boardType);
    const engine = new RuleEngine(manager, boardType as any);

    return { state, manager, engine };
  }

  it('backend only allows 4,7→4,4 as overtaking_capture and never as simple movement; sandbox simple movement excludes it', () => {
    const { state, manager, engine } = createBackendState();

    const from: Position = { x: 4, y: 7 };
    const target: Position = { x: 4, y: 6 };
    const landing: Position = { x: 4, y: 4 };

    const fromKey = positionToString(from);
    const landingKey = positionToString(landing);
    const targetKey = positionToString(target);

    const backendMoves: Move[] = engine.getValidMoves(state);

    const backendSimple = backendMoves.filter(
      (m) =>
        (m.type === 'move_stack' || m.type === 'move_ring') &&
        m.from &&
        m.to &&
        positionToString(m.from) === fromKey &&
        positionToString(m.to) === landingKey
    );

    expect(backendSimple).toHaveLength(0);

    const backendCaptures = backendMoves.filter(
      (m) =>
        m.type === 'overtaking_capture' &&
        m.from &&
        m.to &&
        m.captureTarget &&
        positionToString(m.from) === fromKey &&
        positionToString(m.to) === landingKey &&
        positionToString(m.captureTarget) === targetKey
    );

    expect(backendCaptures.length).toBeGreaterThan(0);

    const board = state.board;
    const sandboxLandings = enumerateSimpleMovementLandings(
      boardType,
      board,
      1,
      (pos) => manager.isValidPosition(pos)
    );

    const sandboxHasIllegalSimpleMove = sandboxLandings.some(
      (m) => m.fromKey === fromKey && positionToString(m.to) === landingKey
    );

    expect(sandboxHasIllegalSimpleMove).toBe(false);
  });

  it('sandbox vs backend parity for overtaking_capture 0,2→4,2 over 2,2 (seeded aiHeuristicCoverage case)', () => {
    const boardType: BoardType = 'square8';
    const players: Player[] = [1, 2].map((n) => ({
      id: `p${n}`,
      username: `P${n}`,
      type: 'human',
      playerNumber: n,
      isReady: true,
      timeRemaining: 60000,
      ringsInHand: 0,
      eliminatedRings: 0,
      territorySpaces: 0,
    })) as Player[];

    const { ruleEngine, boardManager, gameState } = createBackendEngine(boardType, players);

    const attackerPos: Position = { x: 0, y: 2 };
    const targetPos: Position = { x: 2, y: 2 };
    const landingPos: Position = { x: 4, y: 2 };

    gameState.currentPlayer = 1;
    gameState.currentPhase = 'movement';

    // Place attacker: Player 1. Height 3 (capHeight 3).
    boardManager.setStack(
      attackerPos,
      {
        position: attackerPos,
        rings: [1, 1, 1],
        stackHeight: 3,
        capHeight: 3,
        controllingPlayer: 1,
      },
      gameState.board
    );

    // Place target: Player 2. Height 2 (capHeight 2).
    boardManager.setStack(
      targetPos,
      {
        position: targetPos,
        rings: [2, 2],
        stackHeight: 2,
        capHeight: 2,
        controllingPlayer: 2,
      },
      gameState.board
    );

    // Verify backend moves
    const backendMoves = ruleEngine.getValidMoves({
      ...gameState,
      currentPlayer: 1,
      currentPhase: 'movement',
    } as GameState);

    const backendCapture = backendMoves.find(
      (m) =>
        m.type === 'overtaking_capture' &&
        m.from &&
        positionToString(m.from) === positionToString(attackerPos) &&
        m.captureTarget &&
        positionToString(m.captureTarget) === positionToString(targetPos) &&
        positionToString(m.to) === positionToString(landingPos)
    );

    expect(backendCapture).toBeDefined();

    // Now check Sandbox capture enumeration matches this segment.
    const sandbox = createSandboxEngine(boardType, 2);
    const sandboxState = sandbox.getGameState();
    const sandboxBoard = sandboxState.board as BoardState;

    // Mirror the board
    sandboxBoard.stacks.set(positionToString(attackerPos), {
      position: attackerPos,
      rings: [1, 1, 1],
      stackHeight: 3,
      capHeight: 3,
      controllingPlayer: 1,
    });
    sandboxBoard.stacks.set(positionToString(targetPos), {
      position: targetPos,
      rings: [2, 2],
      stackHeight: 2,
      capHeight: 2,
      controllingPlayer: 2,
    });

    const adapters: CaptureBoardAdapters = {
      isValidPosition: (pos: Position) => boardManager.isValidPosition(pos),
      isCollapsedSpace: (pos: Position, board: BoardState) =>
        boardManager.isCollapsedSpace(pos, board),
      getMarkerOwner: (_pos: Position, _board: BoardState) => undefined,
    };

    const captureSegments = enumerateCaptureSegmentsFromBoard(
      boardType,
      sandboxBoard,
      attackerPos,
      1,
      adapters
    );

    const sandboxHasMatchingCapture = captureSegments.some(
      (seg: CaptureSegment) =>
        positionToString(seg.from) === positionToString(attackerPos) &&
        positionToString(seg.target) === positionToString(targetPos) &&
        positionToString(seg.landing) === positionToString(landingPos)
    );

    expect(sandboxHasMatchingCapture).toBe(true);
  });
});
