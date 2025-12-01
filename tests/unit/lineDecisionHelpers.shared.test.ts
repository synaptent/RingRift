import {
  BoardType,
  Player,
  TimeControl,
  GameState,
  Position,
  BOARD_CONFIGS,
  positionToString,
} from '../../src/shared/types/game';
import { createInitialGameState } from '../../src/shared/engine/initialState';
import {
  enumerateProcessLineMoves,
  enumerateChooseLineRewardMoves,
  applyProcessLineDecision,
  applyChooseLineRewardDecision,
} from '../../src/shared/engine/lineDecisionHelpers';
import { getEffectiveLineLengthThreshold } from '../../src/shared/engine';

// Classification: canonical shared line decision helper tests. Backend and sandbox
// line scenario suites rely on these helpers for semantics; older engine-specific
// line-processing tests have been consolidated here where appropriate.

describe('lineDecisionHelpers – shared line decision enumeration and application', () => {
  const boardType: BoardType = 'square8';
  const timeControl: TimeControl = { initialTime: 600, increment: 0, type: 'blitz' };
  const players: Player[] = [
    {
      id: 'p1',
      username: 'Player 1',
      type: 'human',
      playerNumber: 1,
      isReady: true,
      timeRemaining: 600,
      ringsInHand: 0,
      eliminatedRings: 0,
      territorySpaces: 0,
    },
    {
      id: 'p2',
      username: 'Player 2',
      type: 'human',
      playerNumber: 2,
      isReady: true,
      timeRemaining: 600,
      ringsInHand: 0,
      eliminatedRings: 0,
      territorySpaces: 0,
    },
  ];
  const requiredLength = getEffectiveLineLengthThreshold(boardType, players.length);
  function createEmptyState(): GameState {
    const state = createInitialGameState(
      'line-helpers',
      boardType,
      players,
      timeControl
    ) as unknown as GameState;

    state.currentPlayer = 1;
    state.board.stacks.clear();
    state.board.markers.clear();
    state.board.collapsedSpaces.clear();
    state.board.formedLines = [];
    state.board.eliminatedRings = {};
    return state;
  }

  function seedExactLine(state: GameState, y: number): Position[] {
    const positions: Position[] = [];
    const board = state.board;

    for (let x = 0; x < requiredLength; x++) {
      const pos: Position = { x, y };
      positions.push(pos);
      const key = positionToString(pos);
      board.markers.set(key, {
        player: 1,
        position: pos,
        type: 'regular',
      } as any);
    }

    board.formedLines.push({
      player: 1,
      positions,
      length: positions.length,
      direction: { x: 1, y: 0 },
    } as any);

    return positions;
  }

  function seedOverlengthLine(state: GameState, y: number, extra: number): Position[] {
    const positions: Position[] = [];
    const board = state.board;

    for (let x = 0; x < requiredLength + extra; x++) {
      const pos: Position = { x, y };
      positions.push(pos);
      const key = positionToString(pos);
      board.markers.set(key, {
        player: 1,
        position: pos,
        type: 'regular',
      } as any);
    }

    board.formedLines.push({
      player: 1,
      positions,
      length: positions.length,
      direction: { x: 1, y: 0 },
    } as any);

    return positions;
  }

  function keyFrom(positions: Position[]): string {
    return positions
      .map((p) => positionToString(p))
      .sort()
      .join('|');
  }

  it('enumerateProcessLineMoves returns one process_line per player-owned line', () => {
    const state = createEmptyState();

    const line1 = seedExactLine(state, 0);
    const line2 = seedExactLine(state, 2);

    const moves = enumerateProcessLineMoves(state, 1, { detectionMode: 'use_board_cache' });

    expect(moves).toHaveLength(2);
    expect(moves.every((m) => m.type === 'process_line' && m.player === 1)).toBe(true);

    const moveKeys = moves.map((m) => keyFrom(m.formedLines![0].positions));
    expect(moveKeys).toEqual(expect.arrayContaining([keyFrom(line1), keyFrom(line2)]));

    // IDs should embed the canonical line keys.
    moves.forEach((m) => {
      const lineKey = keyFrom(m.formedLines![0].positions);
      expect(m.id).toContain(lineKey);
    });
  });

  it('enumerateProcessLineMoves detects lines directly from markers when board.formedLines is empty', () => {
    const state = createEmptyState();
    const board = state.board;

    const positions: Position[] = [];
    for (let x = 0; x < requiredLength; x++) {
      const pos: Position = { x, y: 4 };
      positions.push(pos);
      const key = positionToString(pos);
      board.markers.set(key, {
        player: 1,
        position: pos,
        type: 'regular',
      } as any);
    }

    // No cached formedLines: detection must come from shared lineDetection.
    board.formedLines = [];

    const moves = enumerateProcessLineMoves(state, 1, { detectionMode: 'detect_now' });

    expect(moves).toHaveLength(1);
    const move = moves[0];

    expect(move.type).toBe('process_line');
    expect(move.player).toBe(1);
    expect(keyFrom(move.formedLines![0].positions)).toBe(keyFrom(positions));
  });

  it('enumerateChooseLineRewardMoves yields a single collapse-all reward for an exact-length line', () => {
    const state = createEmptyState();
    const positions = seedExactLine(state, 0);

    const rewardMoves = enumerateChooseLineRewardMoves(state, 1, 0);

    expect(rewardMoves).toHaveLength(1);
    const move = rewardMoves[0];

    expect(move.type).toBe('choose_line_reward');
    expect(move.player).toBe(1);
    expect(move.collapsedMarkers).toBeUndefined();
    expect(keyFrom(move.formedLines![0].positions)).toBe(keyFrom(positions));
  });

  it('enumerateChooseLineRewardMoves exposes collapse-all plus all contiguous minimum segments for overlength lines', () => {
    const state = createEmptyState();
    const positions = seedOverlengthLine(state, 0, 2);
    const length = positions.length;
    const expectedMinSegments = length - requiredLength + 1;

    const rewardMoves = enumerateChooseLineRewardMoves(state, 1, 0);

    // One collapse-all + all contiguous minimum segments.
    expect(rewardMoves.length).toBe(1 + expectedMinSegments);

    const collapseAll = rewardMoves.filter((m) => {
      const collapsed = m.collapsedMarkers ?? [];
      return collapsed.length === 0 || collapsed.length >= length;
    });
    const minimumSegments = rewardMoves.filter(
      (m) => m.collapsedMarkers && m.collapsedMarkers.length === requiredLength
    );

    expect(collapseAll.length).toBe(1);
    expect(minimumSegments.length).toBe(expectedMinSegments);

    // All rewards should reference the same underlying line geometry.
    rewardMoves.forEach((m) => {
      expect(keyFrom(m.formedLines![0].positions)).toBe(keyFrom(positions));
    });
  });

  it('applyProcessLineDecision collapses exact-length line, returns rings from stacks, and flags pending reward', () => {
    const state = createEmptyState();
    const positions = seedExactLine(state, 0);
    const board = state.board;

    // Place a stack for player 1 on one of the line positions so its rings are
    // returned to hand rather than eliminated.
    const stackPos = positions[0];
    const stackKey = positionToString(stackPos);
    board.stacks.set(stackKey, {
      position: stackPos,
      rings: [1, 1],
      stackHeight: 2,
      capHeight: 2,
      controllingPlayer: 1,
    } as any);

    const playerBefore = state.players.find((p) => p.playerNumber === 1)!;
    const ringsInHandBefore = playerBefore.ringsInHand;

    const move = {
      id: 'process-line-test',
      type: 'process_line' as const,
      player: 1,
      to: positions[0],
      formedLines: [board.formedLines[0] as any],
      timestamp: new Date(),
      thinkTime: 0,
      moveNumber: 1,
    };

    const outcome = applyProcessLineDecision(state, move);
    const next = outcome.nextState;

    expect(outcome.pendingLineRewardElimination).toBe(true);

    const playerAfter = next.players.find((p) => p.playerNumber === 1)!;
    expect(playerAfter.ringsInHand).toBe(ringsInHandBefore + 2);

    // All positions in the line are now collapsed territory for player 1.
    positions.forEach((pos) => {
      const key = positionToString(pos);
      expect(next.board.collapsedSpaces.get(key)).toBe(1);
      expect(next.board.markers.has(key)).toBe(false);
      expect(next.board.stacks.has(key)).toBe(false);
    });

    // Any cached formedLines that intersect the collapse should be cleared.
    expect(next.board.formedLines.length).toBe(0);
  });

  it('applyProcessLineDecision is a no-op for shorter-than-required or overlength lines', () => {
    const state = createEmptyState();
    const board = state.board;

    // Shorter-than-required synthetic line.
    const shortPositions: Position[] = [
      { x: 0, y: 0 },
      { x: 1, y: 0 },
    ];
    board.formedLines.push({
      player: 1,
      positions: shortPositions,
      length: shortPositions.length,
      direction: { x: 1, y: 0 },
    } as any);

    const shortMove = {
      id: 'process-short',
      type: 'process_line' as const,
      player: 1,
      to: shortPositions[0],
      formedLines: [board.formedLines[0] as any],
      timestamp: new Date(),
      thinkTime: 0,
      moveNumber: 1,
    };

    const shortOutcome = applyProcessLineDecision(state, shortMove);
    expect(shortOutcome.pendingLineRewardElimination).toBe(false);
    expect(shortOutcome.nextState).toBe(state);

    // Overlength synthetic line.
    const state2 = createEmptyState();
    const overlengthPositions = seedOverlengthLine(state2, 0, 1);

    const overMove = {
      id: 'process-over',
      type: 'process_line' as const,
      player: 1,
      to: overlengthPositions[0],
      formedLines: [state2.board.formedLines[0] as any],
      timestamp: new Date(),
      thinkTime: 0,
      moveNumber: 1,
    };

    const overOutcome = applyProcessLineDecision(state2, overMove);
    expect(overOutcome.pendingLineRewardElimination).toBe(false);
    expect(overOutcome.nextState).toBe(state2);
  });

  it('applyChooseLineRewardDecision collapses entire exact-length or overlength line and flags pending reward', () => {
    // Exact-length collapse-all.
    const exactState = createEmptyState();
    const exactPositions = seedExactLine(exactState, 0);

    const exactMove = {
      id: 'choose-line-exact-all',
      type: 'choose_line_reward' as const,
      player: 1,
      to: exactPositions[0],
      formedLines: [exactState.board.formedLines[0] as any],
      timestamp: new Date(),
      thinkTime: 0,
      moveNumber: 1,
    };

    const exactOutcome = applyChooseLineRewardDecision(exactState, exactMove);
    const exactNext = exactOutcome.nextState;

    expect(exactOutcome.pendingLineRewardElimination).toBe(true);
    exactPositions.forEach((pos) => {
      const key = positionToString(pos);
      expect(exactNext.board.collapsedSpaces.get(key)).toBe(1);
      expect(exactNext.board.markers.has(key)).toBe(false);
    });

    // Overlength collapse-all.
    const overState = createEmptyState();
    const overPositions = seedOverlengthLine(overState, 0, 2);

    const overMoveAll = {
      id: 'choose-line-over-all',
      type: 'choose_line_reward' as const,
      player: 1,
      to: overPositions[0],
      formedLines: [overState.board.formedLines[0] as any],
      timestamp: new Date(),
      thinkTime: 0,
      moveNumber: 1,
    };

    const overAllOutcome = applyChooseLineRewardDecision(overState, overMoveAll);
    const overAllNext = overAllOutcome.nextState;

    expect(overAllOutcome.pendingLineRewardElimination).toBe(true);
    overPositions.forEach((pos) => {
      const key = positionToString(pos);
      expect(overAllNext.board.collapsedSpaces.get(key)).toBe(1);
    });
  });

  it('applyChooseLineRewardDecision collapses only minimum subset for MINIMUM_COLLAPSE and does not flag reward', () => {
    const state = createEmptyState();
    const positions = seedOverlengthLine(state, 0, 2);
    const line = state.board.formedLines[0] as any;
    const minSubset = positions.slice(1, 1 + requiredLength);

    const move = {
      id: 'choose-line-over-min',
      type: 'choose_line_reward' as const,
      player: 1,
      to: positions[0],
      formedLines: [line],
      collapsedMarkers: minSubset,
      timestamp: new Date(),
      thinkTime: 0,
      moveNumber: 1,
    };

    const outcome = applyChooseLineRewardDecision(state, move);
    const next = outcome.nextState;

    expect(outcome.pendingLineRewardElimination).toBe(false);

    // Only the subset positions should be collapsed.
    positions.forEach((pos) => {
      const key = positionToString(pos);
      if (
        minSubset.some((p) => p.x === pos.x && p.y === pos.y && (p as any).z === (pos as any).z)
      ) {
        expect(next.board.collapsedSpaces.get(key)).toBe(1);
      } else {
        expect(next.board.collapsedSpaces.has(key)).toBe(false);
        expect(next.board.markers.has(key)).toBe(true);
      }
    });
  });
});
