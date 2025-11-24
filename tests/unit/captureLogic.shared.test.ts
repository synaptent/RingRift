import {
  BoardType,
  BoardState,
  GameState,
  Move,
  Player,
  Position,
  positionToString,
} from '../../src/shared/types/game';
import { enumerateCaptureMoves, CaptureBoardAdapters } from '../../src/shared/engine/captureLogic';
import {
  createTestBoard,
  createTestGameState,
  createTestPlayer,
  addStack,
  pos,
} from '../utils/fixtures';
import { BoardManager } from '../../src/server/game/BoardManager';
import { RuleEngine } from '../../src/server/game/RuleEngine';
import { getCaptureOptionsFromPosition as getBackendCaptureOptions } from '../../src/server/game/rules/captureChainEngine';

// Classification: canonical shared capture helper tests for enumerateCaptureMoves
// and backend parity in simple and multi-direction capture scenarios.

function isValidPositionForBoard(boardType: BoardType, board: BoardState, p: Position): boolean {
  if (boardType === 'hexagonal') {
    const radius = board.size - 1;
    const x = p.x;
    const y = p.y;
    const z = p.z !== undefined ? p.z : -x - y;
    const dist = Math.max(Math.abs(x), Math.abs(y), Math.abs(z));
    return dist <= radius;
  }

  return p.x >= 0 && p.x < board.size && p.y >= 0 && p.y < board.size;
}

function makeCaptureAdapters(boardType: BoardType, board: BoardState): CaptureBoardAdapters {
  return {
    isValidPosition: (p: Position) => isValidPositionForBoard(boardType, board, p),
    isCollapsedSpace: (p: Position) => {
      const key = positionToString(p);
      return board.collapsedSpaces.has(key);
    },
    getStackAt: (p: Position) => {
      const key = positionToString(p);
      const stack = board.stacks.get(key);
      if (!stack) return undefined;
      return {
        controllingPlayer: stack.controllingPlayer,
        capHeight: stack.capHeight,
        stackHeight: stack.stackHeight,
      };
    },
    getMarkerOwner: (p: Position) => {
      const key = positionToString(p);
      const marker = board.markers.get(key);
      return marker?.player;
    },
  };
}

describe('enumerateCaptureMoves shared helper', () => {
  const boardType: BoardType = 'square8';

  it('matches backend capture enumeration for a simple straight-line scenario', () => {
    const board = createTestBoard(boardType);
    const from = pos(2, 2);
    const target = pos(4, 2);
    const player = 1;

    // Attacker: height 3 so landings at distance >= 3 are legal.
    addStack(board, from, player, 3);

    // Target: opponent stack with smaller cap height.
    addStack(board, target, 2, 1);

    const adapters = makeCaptureAdapters(boardType, board);
    const sharedMoves = enumerateCaptureMoves(boardType, from, player, adapters, 1);

    const sharedSegments = sharedMoves.map((m: Move) => ({
      from: positionToString(m.from as Position),
      target: positionToString(m.captureTarget as Position),
      landing: positionToString(m.to as Position),
    }));

    expect(sharedSegments.length).toBeGreaterThan(0);

    // Backend path: captureChainEngine.getCaptureOptionsFromPosition via real RuleEngine.
    const players: Player[] = [createTestPlayer(1), createTestPlayer(2)];
    const state: GameState = createTestGameState({
      boardType,
      board,
      players,
      currentPlayer: player,
      currentPhase: 'capture',
    });

    const manager = new BoardManager(boardType);
    const engine = new RuleEngine(manager, boardType as any);
    const backendMoves = getBackendCaptureOptions(from, player, state, {
      boardManager: manager,
      ruleEngine: engine,
    });

    const backendSegments = backendMoves.map((m: Move) => ({
      from: positionToString(m.from as Position),
      target: positionToString(m.captureTarget as Position),
      landing: positionToString(m.to as Position),
    }));

    const sharedKeys = sharedSegments.map((s) => `${s.from}-${s.target}-${s.landing}`).sort();
    const backendKeys = backendSegments.map((s) => `${s.from}-${s.target}-${s.landing}`).sort();

    expect(sharedKeys).toEqual(backendKeys);
  });

  it('enumerates capture segments along multiple rays from a single attacker', () => {
    const board = createTestBoard(boardType);
    const from = pos(3, 3);
    const eastTarget = pos(5, 3);
    const northTarget = pos(3, 5);
    const player = 1;

    // Attacker: height 3. Targets: opponent stacks of height 1 on two rays.
    addStack(board, from, player, 3);
    addStack(board, eastTarget, 2, 1);
    addStack(board, northTarget, 2, 1);

    const adapters = makeCaptureAdapters(boardType, board);
    const sharedMoves = enumerateCaptureMoves(boardType, from, player, adapters, 1);

    expect(sharedMoves.length).toBeGreaterThan(0);

    const targetKeys = new Set(
      sharedMoves.map((m) => positionToString(m.captureTarget as Position))
    );

    const eastKey = positionToString(eastTarget);
    const northKey = positionToString(northTarget);

    expect(targetKeys.has(eastKey)).toBe(true);
    expect(targetKeys.has(northKey)).toBe(true);
  });
});
