import {
  BoardState,
  BoardType,
  GameState,
  Player,
  Position,
  RingStack,
  Move,
} from '../../src/shared/types/game';
import { BoardManager } from '../../src/server/game/BoardManager';
import { GameEngine } from '../../src/server/game/GameEngine';
import {
  ClientSandboxEngine,
  SandboxConfig,
  SandboxInteractionHandler,
} from '../../src/client/sandbox/ClientSandboxEngine';
import {
  snapshotFromBoardAndPlayers,
  snapshotsEqual,
  diffSnapshots,
} from '../utils/stateSnapshots';
import { calculateCapHeight } from '../../src/shared/engine/core';
import { applyCapture as applyCaptureAggregate } from '../../src/shared/engine';

/**
 * Focused parity test for capture + marker + elimination semantics:
 *
 *   - Backend: shared CaptureAggregate.applyCapture on a backend-style GameState.
 *   - Sandbox: ClientSandboxEngine.applyCaptureSegment (via internal helper).
 *
 * The fixture is constructed so that:
 *   - The capture path (from -> target -> landing) crosses both own and
 *     opponent markers.
 *   - The landing cell initially contains the mover's own marker, so
 *     "landing on your own marker eliminates your top ring" should apply.
 *   - No territory or line processing is involved; we compare only board
 *     stacks/markers/collapsedSpaces and per-player elimination counters.
 */
describe('Capture + marker + elimination semantics parity (backend vs sandbox)', () => {
  const boardType: BoardType = 'square8';

  function makeDummyPlayers(): Player[] {
    return [
      {
        id: 'p1',
        username: 'P1',
        type: 'ai',
        playerNumber: 1,
        isReady: true,
        timeRemaining: 0,
        ringsInHand: 0,
        eliminatedRings: 0,
        territorySpaces: 0,
      },
      {
        id: 'p2',
        username: 'P2',
        type: 'ai',
        playerNumber: 2,
        isReady: true,
        timeRemaining: 0,
        ringsInHand: 0,
        eliminatedRings: 0,
        territorySpaces: 0,
      },
    ];
  }

  function cloneBoard(board: BoardState): BoardState {
    return {
      ...board,
      stacks: new Map(board.stacks),
      markers: new Map(board.markers),
      collapsedSpaces: new Map(board.collapsedSpaces),
      territories: new Map(board.territories),
      formedLines: [...board.formedLines],
      eliminatedRings: { ...board.eliminatedRings },
    };
  }

  /**
   * Build a capture fixture with the following geometry:
   *
   *   from    = (2,1)   -- attacker (P1)
   *   target  = (2,3)   -- target stack (P2)
   *   landing = (2,5)   -- final landing cell
   *
   * Path from from -> target:  (2,1) -> (2,2) -> (2,3)
   * Path from target -> landing: (2,3) -> (2,4) -> (2,5)
   *
   * Markers:
   *   - (2,2): own marker (P1)       -- should collapse to territory.
   *   - (2,4): opponent marker (P2)  -- should flip or be removed as appropriate.
   *   - (2,5): own marker (P1)       -- landing on own marker triggers top-ring
   *                                    elimination after the capture.
   *
   * Stacks:
   *   - from   (2,1): P1 stack height 2.
   *   - target (2,3): P2 stack height 2.
   *
   * No collapsed spaces are present initially so we isolate
   * capture + marker + elimination behaviour.
   */
  function buildCaptureMarkerFixture() {
    const bm = new BoardManager(boardType);
    const board = bm.createBoard();

    const from: Position = { x: 2, y: 1 };
    const target: Position = { x: 2, y: 3 };
    const landing: Position = { x: 2, y: 5 };
    const movingPlayer = 1 as const;

    // Attacker stack at from (P1)
    const attackerRings = [movingPlayer, movingPlayer];
    const attackerStack: RingStack = {
      position: from,
      rings: attackerRings,
      stackHeight: attackerRings.length,
      capHeight: calculateCapHeight(attackerRings),
      controllingPlayer: movingPlayer,
    };
    bm.setStack(from, attackerStack, board);

    // Target stack at target (P2)
    const targetPlayer = 2;
    const targetRings = [targetPlayer, targetPlayer];
    const targetStack: RingStack = {
      position: target,
      rings: targetRings,
      stackHeight: targetRings.length,
      capHeight: calculateCapHeight(targetRings),
      controllingPlayer: targetPlayer,
    };
    bm.setStack(target, targetStack, board);

    // Markers along the capture legs
    // Own marker at (2,2) - should collapse to P1 territory along the first leg.
    bm.setMarker({ x: 2, y: 2 }, movingPlayer, board);

    // Opponent marker at (2,4) - should be flipped/removed appropriately
    // along the second leg.
    bm.setMarker({ x: 2, y: 4 }, targetPlayer, board);

    // Own marker at landing (2,5) - landing on this marker should trigger
    // top-ring elimination for the capturing stack.
    bm.setMarker(landing, movingPlayer, board);

    const players = makeDummyPlayers();

    return { board, players, from, target, landing, movingPlayer };
  }

  test('shared CaptureAggregate.applyCapture vs sandbox applyCaptureSegmentOnBoard', async () => {
    const { board, players, from, target, landing, movingPlayer } = buildCaptureMarkerFixture();

    // --- Backend-style path: shared CaptureAggregate.applyCapture on a GameState seeded with backend BoardManager geometry ---

    const timeControl = { initialTime: 0, increment: 0, type: 'rapid' as const };
    const backendEngine = new GameEngine(
      'capture-marker-test',
      boardType,
      players.map((p) => ({ ...p })),
      timeControl,
      false
    );

    const backendAny: any = backendEngine;
    const backendState0: GameState = backendEngine.getGameState();
    const backendBoard = cloneBoard(board);
    const backendPlayers = players.map((p) => ({ ...p }));

    backendAny.gameState = {
      ...backendState0,
      board: backendBoard,
      players: backendPlayers,
      currentPlayer: movingPlayer,
      currentPhase: 'movement',
    } as GameState;

    const captureMove: Move = {
      id: 'capture-marker-backend',
      type: 'overtaking_capture',
      player: movingPlayer,
      from,
      captureTarget: target,
      to: landing,
      timestamp: new Date(),
      thinkTime: 0,
      moveNumber: 1,
    };

    const captureResult = applyCaptureAggregate(backendAny.gameState as any, captureMove as any);

    if (!captureResult.success) {
      throw new Error(
        `CaptureAggregate.applyCapture failed in CaptureMarker.GameEngine_vs_Sandbox: ${captureResult.reason}`
      );
    }

    backendAny.gameState = captureResult.newState as GameState;

    const backendSnap = snapshotFromBoardAndPlayers(
      'backend-capture-marker',
      backendAny.gameState.board as BoardState,
      backendAny.gameState.players as Player[]
    );

    // --- Sandbox path: ClientSandboxEngine.applyCaptureSegment on a clone ---

    const config: SandboxConfig = {
      boardType,
      numPlayers: players.length,
      playerKinds: players.map((p) => p.type as 'human' | 'ai'),
    };

    const handler: SandboxInteractionHandler = {
      async requestChoice(choice: any) {
        const options = ((choice as any).options as any[]) ?? [];
        const selectedOption = options.length > 0 ? options[0] : undefined;
        return {
          choiceId: (choice as any).id,
          playerNumber: (choice as any).playerNumber,
          choiceType: (choice as any).type,
          selectedOption,
        } as any;
      },
    };

    const sandboxEngine = new ClientSandboxEngine({
      config,
      interactionHandler: handler,
      traceMode: true,
    });

    const sandboxAny: any = sandboxEngine;
    const sandboxState0: GameState = sandboxEngine.getGameState();
    const sandboxBoard = cloneBoard(board);
    const sandboxPlayers = players.map((p) => ({ ...p }));

    sandboxAny.gameState = {
      ...sandboxState0,
      board: sandboxBoard,
      players: sandboxPlayers,
      currentPlayer: movingPlayer,
      currentPhase: 'movement',
    } as GameState;

    // Use the sandbox engine's internal wrapper, which delegates to
    // applyCaptureSegmentOnBoard and landing-on-own-marker elimination logic.
    await sandboxAny.applyCaptureSegment(from, target, landing, movingPlayer);

    const sandboxSnap = snapshotFromBoardAndPlayers(
      'sandbox-capture-marker',
      sandboxAny.gameState.board as BoardState,
      sandboxAny.gameState.players as Player[]
    );

    if (!snapshotsEqual(backendSnap, sandboxSnap)) {
      // eslint-disable-next-line no-console
      console.error(
        '[CaptureMarker.GameEngine_vs_Sandbox] capture+marker mismatch',
        diffSnapshots(backendSnap, sandboxSnap)
      );
    }

    expect(snapshotsEqual(backendSnap, sandboxSnap)).toBe(true);
  });
  test('two-segment capture chain parity (backend vs sandbox)', async () => {
    const bm = new BoardManager(boardType);
    const board = bm.createBoard();

    const from1: Position = { x: 3, y: 3 };
    const target1: Position = { x: 5, y: 3 };
    const landing1: Position = { x: 7, y: 3 };
    const target2: Position = { x: 6, y: 2 };
    const landing2: Position = { x: 4, y: 0 };
    const movingPlayer = 1 as const;
    const targetPlayer = 2;

    const attackerRings = [movingPlayer, movingPlayer];
    const attackerStack: RingStack = {
      position: from1,
      rings: attackerRings,
      stackHeight: attackerRings.length,
      capHeight: calculateCapHeight(attackerRings),
      controllingPlayer: movingPlayer,
    };
    bm.setStack(from1, attackerStack, board);

    const target1Rings = [targetPlayer, targetPlayer];
    const target1Stack: RingStack = {
      position: target1,
      rings: target1Rings,
      stackHeight: target1Rings.length,
      capHeight: calculateCapHeight(target1Rings),
      controllingPlayer: targetPlayer,
    };
    bm.setStack(target1, target1Stack, board);

    const target2Rings = [targetPlayer, targetPlayer];
    const target2Stack: RingStack = {
      position: target2,
      rings: target2Rings,
      stackHeight: target2Rings.length,
      capHeight: calculateCapHeight(target2Rings),
      controllingPlayer: targetPlayer,
    };
    bm.setStack(target2, target2Stack, board);

    const players = makeDummyPlayers();

    const timeControl = { initialTime: 0, increment: 0, type: 'rapid' as const };
    const backendEngine = new GameEngine(
      'capture-chain-test',
      boardType,
      players.map((p) => ({ ...p })),
      timeControl,
      false
    );

    const backendAny: any = backendEngine;
    const backendState0: GameState = backendEngine.getGameState();
    const backendBoard = cloneBoard(board);
    const backendPlayers = players.map((p) => ({ ...p }));

    backendAny.gameState = {
      ...backendState0,
      board: backendBoard,
      players: backendPlayers,
      currentPlayer: movingPlayer,
      currentPhase: 'movement',
    } as GameState;

    const firstSegmentMove: Move = {
      id: 'capture-chain-backend-1',
      type: 'overtaking_capture',
      player: movingPlayer,
      from: from1,
      captureTarget: target1,
      to: landing1,
      timestamp: new Date(),
      thinkTime: 0,
      moveNumber: 1,
    };

    const firstResult = applyCaptureAggregate(backendAny.gameState as any, firstSegmentMove as any);
    if (!firstResult.success) {
      throw new Error(
        `CaptureAggregate.applyCapture failed for first segment in chain test: ${firstResult.reason}`
      );
    }
    backendAny.gameState = firstResult.newState as GameState;

    const secondSegmentMove: Move = {
      id: 'capture-chain-backend-2',
      type: 'continue_capture_segment',
      player: movingPlayer,
      from: landing1,
      captureTarget: target2,
      to: landing2,
      timestamp: new Date(),
      thinkTime: 0,
      moveNumber: 2,
    };

    const secondResult = applyCaptureAggregate(
      backendAny.gameState as any,
      secondSegmentMove as any
    );
    if (!secondResult.success) {
      throw new Error(
        `CaptureAggregate.applyCapture failed for second segment in chain test: ${secondResult.reason}`
      );
    }
    backendAny.gameState = secondResult.newState as GameState;

    const backendSnap = snapshotFromBoardAndPlayers(
      'backend-capture-chain',
      backendAny.gameState.board as BoardState,
      backendAny.gameState.players as Player[]
    );

    const config: SandboxConfig = {
      boardType,
      numPlayers: players.length,
      playerKinds: players.map((p) => p.type as 'human' | 'ai'),
    };

    const handler: SandboxInteractionHandler = {
      async requestChoice(choice: any) {
        const options = ((choice as any).options as any[]) ?? [];
        const selectedOption = options.length > 0 ? options[0] : undefined;
        return {
          choiceId: (choice as any).id,
          playerNumber: (choice as any).playerNumber,
          choiceType: (choice as any).type,
          selectedOption,
        } as any;
      },
    };

    const sandboxEngine = new ClientSandboxEngine({
      config,
      interactionHandler: handler,
      traceMode: true,
    });

    const sandboxAny: any = sandboxEngine;
    const sandboxState0: GameState = sandboxEngine.getGameState();
    const sandboxBoard = cloneBoard(board);
    const sandboxPlayers = players.map((p) => ({ ...p }));

    sandboxAny.gameState = {
      ...sandboxState0,
      board: sandboxBoard,
      players: sandboxPlayers,
      currentPlayer: movingPlayer,
      currentPhase: 'movement',
    } as GameState;

    await sandboxAny.applyCaptureSegment(from1, target1, landing1, movingPlayer);
    await sandboxAny.applyCaptureSegment(landing1, target2, landing2, movingPlayer);

    const sandboxSnap = snapshotFromBoardAndPlayers(
      'sandbox-capture-chain',
      sandboxAny.gameState.board as BoardState,
      sandboxAny.gameState.players as Player[]
    );

    if (!snapshotsEqual(backendSnap, sandboxSnap)) {
      // eslint-disable-next-line no-console
      console.error(
        '[CaptureMarker.GameEngine_vs_Sandbox] two-segment chain capture mismatch',
        diffSnapshots(backendSnap, sandboxSnap)
      );
    }

    expect(snapshotsEqual(backendSnap, sandboxSnap)).toBe(true);

    // In addition to board/parity checks, ensure that backend and sandbox
    // agree on the active player and phase after the same chain capture
    // sequence. This guards against subtle capture/phase tracking drift.
    expect(backendAny.gameState.currentPlayer).toBe(sandboxAny.gameState.currentPlayer);
    expect(backendAny.gameState.currentPhase).toBe(sandboxAny.gameState.currentPhase);
  });
});
