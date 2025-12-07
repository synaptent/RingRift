import {
  BoardType,
  BoardState,
  GameState,
  Position,
  positionToString,
} from '../../src/shared/types/game';
import {
  enumerateAllCaptureMoves,
  enumerateChainCaptureSegments,
  enumerateChainCaptures,
  getChainCaptureContinuationInfo,
  applyCaptureSegment,
  applyCapture,
  validateCapture,
  mutateCapture,
  type ChainCaptureStateSnapshot,
  type ChainCaptureEnumerationOptions,
} from '../../src/shared/engine/aggregates/CaptureAggregate';
import type { OvertakingCaptureAction } from '../../src/shared/engine/types';
import {
  createTestBoard,
  createTestGameState,
  createTestPlayer,
  addStack,
  pos,
} from '../utils/fixtures';

describe('CaptureAggregate – canonical capture-chain behaviour', () => {
  const boardType: BoardType = 'square8';

  function makeEmptyGameState(boardTypeOverride: BoardType = boardType): GameState {
    const board: BoardState = createTestBoard(boardTypeOverride);
    const players = [createTestPlayer(1), createTestPlayer(2)];
    return createTestGameState({
      boardType: boardTypeOverride,
      board,
      players,
      currentPlayer: 1,
      currentPhase: 'capture',
    });
  }

  it('enumerateAllCaptureMoves finds the same targets as per-position enumeration', () => {
    const state = makeEmptyGameState();
    const from = pos(2, 2);
    const target = pos(4, 2);

    addStack(state.board, from, 1, 3);
    addStack(state.board, target, 2, 1);

    const snapshot: ChainCaptureStateSnapshot = {
      player: 1,
      currentPosition: from,
      capturedThisChain: [],
    };

    const byPosition = enumerateChainCaptureSegments(state, snapshot, {
      kind: 'initial',
    });

    const all = enumerateAllCaptureMoves(state, 1);

    const byPositionKeys = new Set(
      byPosition.map((m) => `${m.from!.x},${m.from!.y}-${m.captureTarget!.x},${m.captureTarget!.y}`)
    );
    const allKeys = new Set(
      all.map((m) => `${m.from!.x},${m.from!.y}-${m.captureTarget!.x},${m.captureTarget!.y}`)
    );

    expect(allKeys).toEqual(byPositionKeys);
  });

  it('disallowRevisitedTargets filters out revisited capture targets', () => {
    const state = makeEmptyGameState();
    const from = pos(0, 0);
    const targetA = pos(2, 0);
    const targetB = pos(0, 2);

    addStack(state.board, from, 1, 2);
    addStack(state.board, targetA, 2, 1);
    addStack(state.board, targetB, 2, 1);

    const baseSnapshot: ChainCaptureStateSnapshot = {
      player: 1,
      currentPosition: from,
      capturedThisChain: [],
    };

    const options: ChainCaptureEnumerationOptions = {
      kind: 'continuation',
    };

    const allSegments = enumerateChainCaptureSegments(state, baseSnapshot, options);
    expect(allSegments.length).toBeGreaterThan(0);

    const visitedSnapshot: ChainCaptureStateSnapshot = {
      ...baseSnapshot,
      capturedThisChain: [targetA],
    };

    const filteredSegments = enumerateChainCaptureSegments(state, visitedSnapshot, {
      ...options,
      disallowRevisitedTargets: true,
    });

    const hasAWithoutFilter = allSegments.some(
      (m) => m.captureTarget!.x === targetA.x && m.captureTarget!.y === targetA.y
    );
    const hasAWithFilter = filteredSegments.some(
      (m) => m.captureTarget!.x === targetA.x && m.captureTarget!.y === targetA.y
    );

    expect(hasAWithoutFilter).toBe(true);
    expect(hasAWithFilter).toBe(false);
  });

  it('enumerateChainCaptures returns only landing positions', () => {
    const state = makeEmptyGameState();
    const from = pos(3, 3);
    const target = pos(5, 3);

    addStack(state.board, from, 1, 3);
    addStack(state.board, target, 2, 1);

    const landings = enumerateChainCaptures(state, from, 1);
    expect(landings.length).toBeGreaterThan(0);

    for (const landing of landings) {
      expect(landing.x).toBeGreaterThan(target.x);
      expect(landing.y).toBe(target.y);
    }
  });

  it('applyCaptureSegment reports continuation when further captures exist', () => {
    const state = makeEmptyGameState();
    const from = pos(2, 2);
    const firstTarget = pos(2, 3);
    const secondTarget = pos(2, 5);
    const firstLanding = pos(2, 4);

    addStack(state.board, from, 1, 2);
    addStack(state.board, firstTarget, 2, 1);
    addStack(state.board, secondTarget, 3, 1);

    const outcome = applyCaptureSegment(state, {
      from,
      target: firstTarget,
      landing: firstLanding,
      player: 1,
    });

    expect(outcome.nextState).not.toBe(state);
    expect(outcome.chainContinuationRequired).toBe(true);

    const info = getChainCaptureContinuationInfo(outcome.nextState, 1, firstLanding);
    expect(info.mustContinue).toBe(true);
    expect(info.availableContinuations.length).toBeGreaterThan(0);
  });

  it('applyCapture returns success and continuation landing positions when a chain exists', () => {
    const state = makeEmptyGameState();
    const from = pos(2, 2);
    const firstTarget = pos(2, 3);
    const secondTarget = pos(2, 5);
    const firstLanding = pos(2, 4);

    addStack(state.board, from, 1, 2);
    addStack(state.board, firstTarget, 2, 1);
    addStack(state.board, secondTarget, 3, 1);

    const move = {
      id: 'test-capture',
      type: 'overtaking_capture' as const,
      player: 1,
      from,
      captureTarget: firstTarget,
      to: firstLanding,
      timestamp: new Date(),
      thinkTime: 0,
      moveNumber: 1,
    };

    const result = applyCapture(state, move);
    expect(result.success).toBe(true);

    if (result.success) {
      expect(result.chainCaptures.length).toBeGreaterThan(0);
    }
  });

  it('enumerates simple chain captures on hexagonal board', () => {
    const hexType: BoardType = 'hexagonal';
    const state = makeEmptyGameState(hexType);
    const from: Position = { x: 0, y: 0, z: 0 };
    const target: Position = { x: 1, y: -1, z: 0 };

    (state.board as BoardState).type = hexType;
    (state.board as BoardState).size = 13; // radius=12

    addStack(state.board, from, 1, 2);
    addStack(state.board, target, 2, 1);

    const snapshot: ChainCaptureStateSnapshot = {
      player: 1,
      currentPosition: from,
      capturedThisChain: [],
    };

    const segments = enumerateChainCaptureSegments(state, snapshot, {
      kind: 'initial',
    });

    expect(segments.length).toBeGreaterThan(0);

    const capturesFromHelper = enumerateChainCaptures(state, from, 1);
    expect(capturesFromHelper.length).toBeGreaterThan(0);
  });

  it('validateCapture enforces phase, turn, position, and segment legality', () => {
    const state = makeEmptyGameState();
    const from = pos(2, 2);
    const target = pos(4, 2);
    const landing = pos(6, 2);

    addStack(state.board, from, 1, 3);
    addStack(state.board, target, 2, 1);

    const baseAction: OvertakingCaptureAction = {
      type: 'OVERTAKING_CAPTURE',
      playerId: 1,
      from,
      captureTarget: target,
      to: landing,
    };

    // Happy path
    state.currentPhase = 'movement';
    state.currentPlayer = 1;
    let result = validateCapture(state, baseAction);
    expect(result.valid).toBe(true);

    // Wrong phase
    const wrongPhaseState: GameState = { ...state, currentPhase: 'line_processing' };
    result = validateCapture(wrongPhaseState, baseAction);
    expect(result.valid).toBe(false);
    if (!result.valid) {
      expect(result.code).toBe('INVALID_PHASE');
    }

    // Wrong player
    const wrongPlayerState: GameState = { ...state, currentPlayer: 2 };
    result = validateCapture(wrongPlayerState, baseAction);
    expect(result.valid).toBe(false);
    if (!result.valid) {
      expect(result.code).toBe('NOT_YOUR_TURN');
    }

    // Off-board position
    const offBoardAction: OvertakingCaptureAction = {
      ...baseAction,
      from: { x: -1, y: -1 } as Position,
    };
    result = validateCapture(state, offBoardAction);
    expect(result.valid).toBe(false);
    if (!result.valid) {
      expect(result.code).toBe('INVALID_POSITION');
    }

    // Geometrically illegal segment: target not on a straight ray from `from`.
    const illegalTarget = pos(3, 4);
    addStack(state.board, illegalTarget, 2, 1);

    const illegalAction: OvertakingCaptureAction = {
      ...baseAction,
      captureTarget: illegalTarget,
      to: pos(5, 5),
    };

    result = validateCapture(state, illegalAction);
    expect(result.valid).toBe(false);
    if (!result.valid) {
      expect(result.code).toBe('INVALID_CAPTURE');
    }
  });

  it('mutateCapture applies landing-on-own-marker elimination semantics', () => {
    const state = makeEmptyGameState();
    const from = pos(2, 2);
    const target = pos(2, 4);
    const landing = pos(2, 6);

    addStack(state.board, from, 1, 2);
    addStack(state.board, target, 2, 1);

    const landingKey = positionToString(landing);
    const fromKey = positionToString(from);

    state.board.markers.set(landingKey, {
      player: 1,
      position: landing,
      type: 'regular',
    });

    const action: OvertakingCaptureAction = {
      type: 'OVERTAKING_CAPTURE',
      playerId: 1,
      from,
      captureTarget: target,
      to: landing,
    };

    const next = mutateCapture(state, action);

    // Departure marker placed
    expect(next.board.markers.get(fromKey)?.player).toBe(1);

    // Landing marker removed
    expect(next.board.markers.has(landingKey)).toBe(false);

    // One ring eliminated for player 1
    expect(next.board.eliminatedRings[1]).toBe(1);
    const player1 = next.players.find((p) => p.playerNumber === 1)!;
    expect(player1.eliminatedRings).toBe(1);
  });

  it('mutateCapture removes opponent landing marker and eliminates a ring per RR-CANON-R091/R092', () => {
    // Per canonical rules, landing on ANY marker (own or opponent) removes the
    // marker and eliminates a ring from the cap.
    const state = makeEmptyGameState();
    const from = pos(2, 2);
    const target = pos(2, 4);
    const landing = pos(2, 6);

    addStack(state.board, from, 1, 2);
    addStack(state.board, target, 2, 1);

    const landingKey = positionToString(landing);
    state.board.markers.set(landingKey, {
      player: 2,
      position: landing,
      type: 'regular',
    });

    const action: OvertakingCaptureAction = {
      type: 'OVERTAKING_CAPTURE',
      playerId: 1,
      from,
      captureTarget: target,
      to: landing,
    };

    const next = mutateCapture(state, action);

    // Marker should be removed
    expect(next.board.markers.has(landingKey)).toBe(false);
    // One ring eliminated per RR-CANON-R091/R092
    expect(next.board.eliminatedRings[1] || 0).toBe(1);
  });

  it('applyCapture handles non-capture moves and structural errors gracefully', () => {
    const state = makeEmptyGameState();
    const from = pos(0, 0);
    const target = pos(1, 0);
    const landing = pos(2, 0);

    // Non-capture move type
    const badTypeResult = applyCapture(state, {
      id: 'bad-type',
      type: 'movement' as any,
      player: 1,
      from,
      to: landing,
      timestamp: new Date(),
      thinkTime: 0,
      moveNumber: 1,
    });
    expect(badTypeResult.success).toBe(false);
    if (!badTypeResult.success) {
      expect(badTypeResult.reason).toMatch(/overtaking_capture/);
    }

    // Missing from / captureTarget
    const missingFieldsResult = applyCapture(state, {
      id: 'missing-fields',
      type: 'overtaking_capture',
      player: 1,
      // no from or captureTarget
      to: landing,
      timestamp: new Date(),
      thinkTime: 0,
      moveNumber: 1,
    } as any);
    expect(missingFieldsResult.success).toBe(false);
    if (!missingFieldsResult.success) {
      expect(missingFieldsResult.reason).toMatch(/Move.from and Move.captureTarget/);
    }

    // Structural error from mutateCapture (no stacks at from/target)
    const structuralResult = applyCapture(state, {
      id: 'structural',
      type: 'overtaking_capture',
      player: 1,
      from,
      captureTarget: target,
      to: landing,
      timestamp: new Date(),
      thinkTime: 0,
      moveNumber: 1,
    });
    expect(structuralResult.success).toBe(false);
    if (!structuralResult.success) {
      expect(structuralResult.reason).toMatch(/Missing attacker or target stack/);
    }
  });

  it('applyCapture can succeed without any required continuation', () => {
    const state = makeEmptyGameState();
    const from = pos(2, 2);
    const target = pos(4, 2);
    const landing = pos(6, 2);

    addStack(state.board, from, 1, 2);
    addStack(state.board, target, 2, 1);

    const move = {
      id: 'single-capture',
      type: 'overtaking_capture' as const,
      player: 1,
      from,
      captureTarget: target,
      to: landing,
      timestamp: new Date(),
      thinkTime: 0,
      moveNumber: 1,
    };

    const result = applyCapture(state, move);
    expect(result.success).toBe(true);
    if (result.success) {
      expect(result.chainCaptures).toEqual([]);
    }
  });
});
