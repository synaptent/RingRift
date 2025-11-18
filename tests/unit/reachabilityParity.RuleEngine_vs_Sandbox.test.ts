import { BoardState, BoardType, Position } from '../../src/shared/types/game';
import { BoardManager } from '../../src/server/game/BoardManager';
import { RuleEngine } from '../../src/server/game/RuleEngine';
import { ClientSandboxEngine } from '../../src/client/sandbox/ClientSandboxEngine';
import { createTestBoard, addStack, addCollapsedSpace, pos } from '../utils/fixtures';

function callRuleEngineReachability(
  boardType: BoardType,
  board: BoardState,
  from: Position,
  player: number
): boolean {
  const bm = new BoardManager(boardType);
  const engine = new RuleEngine(bm as any, boardType as any);
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  return (engine as any).hasAnyLegalMoveOrCaptureFrom(from, player, board);
}

function callSandboxReachability(
  boardType: BoardType,
  board: BoardState,
  from: Position,
  player: number
): boolean {
  const sandbox = new ClientSandboxEngine({
    config: {
      boardType,
      numPlayers: 2,
      playerKinds: ['human', 'human']
    },
    interactionHandler: {
      // Choices are not used in these tests
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      async requestChoice(choice: any): Promise<any> {
        throw new Error('SandboxInteractionHandler.requestChoice should not be called in reachability tests');
      }
    }
  });

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  return (sandbox as any).hasAnyLegalMoveOrCaptureFrom(from, player, board);
}

describe('RuleEngine vs ClientSandboxEngine reachability parity', () => {
  test('square8: open board with a single stack matches', () => {
    const boardType: BoardType = 'square8';
    const board = createTestBoard(boardType);
    const from = pos(3, 3);
    addStack(board, from, 1, 2); // height 2 stack

    const backend = callRuleEngineReachability(boardType, board, from, 1);
    const sandbox = callSandboxReachability(boardType, board, from, 1);
    expect(backend).toBe(sandbox);
  });

  test('square8: surrounded by collapsed spaces matches (no actions)', () => {
    const boardType: BoardType = 'square8';
    const board = createTestBoard(boardType);
    const from = pos(4, 4);
    addStack(board, from, 1, 1);

    const deltas = [
      [-1, -1], [-1, 0], [-1, 1],
      [0, -1],           [0, 1],
      [1, -1],  [1, 0],  [1, 1]
    ];
    for (const [dx, dy] of deltas) {
      const p = pos(from.x + dx, from.y + dy);
      addCollapsedSpace(board, p, 2);
    }

    const backend = callRuleEngineReachability(boardType, board, from, 1);
    const sandbox = callSandboxReachability(boardType, board, from, 1);
    expect(backend).toBe(false);
    expect(sandbox).toBe(false);
  });

  test('square8: simple capture opportunity matches', () => {
    const boardType: BoardType = 'square8';
    const board = createTestBoard(boardType);
    const from = pos(2, 2);
    const target = pos(4, 4); // diagonal target
    const landing = pos(6, 6); // diagonal landing beyond target

    // Attacker height 3, target cap height 2 ⇒ capturable
    addStack(board, from, 1, 3);
    addStack(board, target, 2, 2);

    const backend = callRuleEngineReachability(boardType, board, from, 1);
    const sandbox = callSandboxReachability(boardType, board, from, 1);

    // Both engines should agree that at least one action is available
    expect(backend).toBe(true);
    expect(sandbox).toBe(true);
  });

  test('hexagonal: simple open-board parity', () => {
    const boardType: BoardType = 'hexagonal';
    const board = createTestBoard(boardType);
    const from: Position = { x: 0, y: 0, z: 0 };
    addStack(board, from, 1, 1);

    const backend = callRuleEngineReachability(boardType, board, from, 1);
    const sandbox = callSandboxReachability(boardType, board, from, 1);
    expect(backend).toBe(sandbox);
  });
});
