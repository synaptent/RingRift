import { GameEngine } from '../../src/server/game/GameEngine';
import {
  ClientSandboxEngine,
  SandboxConfig,
  SandboxInteractionHandler,
} from '../../src/client/sandbox/ClientSandboxEngine';
import { BoardType, GameState, Player, TimeControl, Move } from '../../src/shared/types/game';
import { summarizeBoard, computeProgressSnapshot } from '../../src/shared/engine/core';
import { addMarker, addStack, addCollapsedSpace, pos } from '../utils/fixtures';

/**
 * Targeted parity test for the earliest geometric mismatch in the
 * seed17 AI-vs-AI trace (square8 / 2 players), which occurs at
 * moveNumber 52: a move_stack from (0,0) -> (0,7) by player 2.
 *
 * We reconstruct the pre-step board + metadata from
 * logs/seed17_trace_debug2.log (boardBeforeSummary + players hash),
 * then apply the move and let each engine run its automatic
 * consequences. We assert that the resulting board summaries and
 * S-invariants match.
 */

describe('Seed17 move 52 parity: GameEngine vs ClientSandboxEngine', () => {
  const boardType: BoardType = 'square8';
  const timeControl: TimeControl = { initialTime: 600, increment: 0, type: 'blitz' };

  function createBackendPlayers(): Player[] {
    return [
      {
        id: 'p1-seed17',
        username: 'P1',
        type: 'human',
        playerNumber: 1,
        isReady: true,
        timeRemaining: timeControl.initialTime * 1000,
        ringsInHand: 2, // from stateHashBefore: 1:2:2:0
        eliminatedRings: 2,
        territorySpaces: 0,
      },
      {
        id: 'p2-seed17',
        username: 'P2',
        type: 'human',
        playerNumber: 2,
        isReady: true,
        timeRemaining: timeControl.initialTime * 1000,
        ringsInHand: 0, // from stateHashBefore: 2:0:5:6
        eliminatedRings: 5,
        territorySpaces: 6,
      },
    ];
  }

  function parsePos(key: string) {
    const [xStr, yStr] = key.split(',');
    return pos(parseInt(xStr, 10), parseInt(yStr, 10));
  }

  test('pre-step board + move_stack(0,0->0,7) yields identical geometry and S', async () => {
    // --- Shared pre-step geometry (boardBeforeSummary for move 52) ---
    const stackSpecs = [
      '0,0:2:1:1',
      '1,3:1:5:5',
      '1,6:2:3:3',
      '2,0:2:5:5',
      '3,0:1:5:2',
      '5,3:1:2:2',
      '5,6:1:1:1',
      '6,4:2:5:1',
    ];

    const markerSpecs = [
      '0,2:2',
      '0,3:1',
      '0,4:1',
      '0,6:2',
      '1,1:1',
      '1,7:1',
      '2,7:2',
      '3,6:1',
      '4,6:1',
      '4,7:1',
      '5,5:2',
      '7,4:1',
      '7,6:2',
    ];

    const collapsedSpecs = [
      '1,5:2',
      '2,3:2',
      '3,4:1',
      '3,5:1',
      '4,2:2',
      '4,3:2',
      '5,1:2',
      '5,2:2',
      '6,0:2',
      '7,0:2',
      '7,1:2',
    ];

    // --- Backend engine seeded to pre-step state ---
    const backendPlayers = createBackendPlayers();
    const backendEngine = new GameEngine(
      'seed17-move52-backend',
      boardType,
      backendPlayers,
      timeControl,
      false
    );
    backendEngine.startGame();

    const backendAny: any = backendEngine;
    const backendState: GameState = backendAny.gameState as GameState;
    const backendBoard = backendState.board;

    backendState.currentPlayer = 2; // actor 2 at move 52
    backendState.currentPhase = 'movement';
    backendState.gameStatus = 'active';
    backendState.totalRingsEliminated = 7; // from progressBefore.eliminated

    backendBoard.stacks.clear();
    backendBoard.markers.clear();
    backendBoard.collapsedSpaces.clear();
    backendBoard.eliminatedRings = { 1: 2, 2: 5 };

    for (const spec of stackSpecs) {
      const [posKey, playerStr, heightStr] = spec.split(':');
      const position = parsePos(posKey);
      const player = parseInt(playerStr, 10);
      const height = parseInt(heightStr, 10);
      addStack(backendBoard, position, player, height);
    }

    for (const spec of markerSpecs) {
      const [posKey, playerStr] = spec.split(':');
      const position = parsePos(posKey);
      const player = parseInt(playerStr, 10);
      addMarker(backendBoard, position, player);
    }

    for (const spec of collapsedSpecs) {
      const [posKey, ownerStr] = spec.split(':');
      const position = parsePos(posKey);
      const owner = parseInt(ownerStr, 10);
      addCollapsedSpace(backendBoard, position, owner);
    }

    // --- Sandbox engine seeded to the same pre-step state ---
    const sandboxConfig: SandboxConfig = {
      boardType,
      numPlayers: 2,
      playerKinds: ['human', 'human'],
    };

    const sandboxHandler: SandboxInteractionHandler = {
      async requestChoice<TChoice>(choice: TChoice): Promise<any> {
        const optionsArray = ((choice as any).options as any[]) ?? [];
        const selectedOption = optionsArray.length > 0 ? optionsArray[0] : undefined;

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
    });

    const sandboxAny: any = sandboxEngine;
    const sandboxState: GameState = sandboxAny.gameState as GameState;
    const sandboxBoard = sandboxState.board;

    sandboxState.currentPlayer = 2;
    sandboxState.currentPhase = 'movement';
    sandboxState.gameStatus = 'active';
    sandboxState.totalRingsEliminated = 7;

    // Sync player meta with backend players.
    const sP1 = sandboxState.players.find((p) => p.playerNumber === 1)!;
    sP1.ringsInHand = 2;
    sP1.eliminatedRings = 2;
    sP1.territorySpaces = 0;

    const sP2 = sandboxState.players.find((p) => p.playerNumber === 2)!;
    sP2.ringsInHand = 0;
    sP2.eliminatedRings = 5;
    sP2.territorySpaces = 6;

    sandboxBoard.stacks.clear();
    sandboxBoard.markers.clear();
    sandboxBoard.collapsedSpaces.clear();
    sandboxBoard.eliminatedRings = { 1: 2, 2: 5 };

    for (const spec of stackSpecs) {
      const [posKey, playerStr, heightStr] = spec.split(':');
      const position = parsePos(posKey);
      const player = parseInt(playerStr, 10);
      const height = parseInt(heightStr, 10);
      addStack(sandboxBoard, position, player, height);
    }

    for (const spec of markerSpecs) {
      const [posKey, playerStr] = spec.split(':');
      const position = parsePos(posKey);
      const player = parseInt(playerStr, 10);
      addMarker(sandboxBoard, position, player);
    }

    for (const spec of collapsedSpecs) {
      const [posKey, ownerStr] = spec.split(':');
      const position = parsePos(posKey);
      const owner = parseInt(ownerStr, 10);
      addCollapsedSpace(sandboxBoard, position, owner);
    }

    // Sanity: pre-move geometry and S-invariants match.
    expect(summarizeBoard(backendBoard)).toEqual(summarizeBoard(sandboxBoard));

    const backendSnapBefore = computeProgressSnapshot(backendState);
    const sandboxSnapBefore = computeProgressSnapshot(sandboxState);
    expect(backendSnapBefore).toEqual(sandboxSnapBefore);

    // --- Apply the canonical move 52 on both engines ---
    const move52: Move = {
      id: '',
      type: 'move_stack',
      player: 2,
      from: pos(0, 0),
      to: pos(0, 7),
      timestamp: new Date(),
      thinkTime: 0,
      moveNumber: 52,
    };

    const backendResult = await backendEngine.makeMove({
      type: move52.type,
      player: move52.player,
      from: move52.from,
      to: move52.to,
    } as any);

    expect(backendResult.success).toBe(true);

    await sandboxEngine.applyCanonicalMove(move52);

    const backendAfterState = backendEngine.getGameState();
    const sandboxAfterState = sandboxEngine.getGameState();

    const backendAfterBoard = backendAfterState.board;
    const sandboxAfterBoard = sandboxAfterState.board;

    // Final geometric parity: stacks/markers/collapsedSpaces.
    expect(summarizeBoard(backendAfterBoard)).toEqual(summarizeBoard(sandboxAfterBoard));

    // Final S-invariant parity.
    const backendSnapAfter = computeProgressSnapshot(backendAfterState);
    const sandboxSnapAfter = computeProgressSnapshot(sandboxAfterState);
    expect(backendSnapAfter).toEqual(sandboxSnapAfter);
  });
});
