import type {
  Move,
  MovementMove,
  BuildStackMove,
  CaptureMove,
  PlacementMove,
  TypedMove,
} from '../../../src/shared/types/game';
import {
  isMovementMove,
  isBuildStackMove,
  isCaptureMove,
  isPlacementMove,
  isSpatialMove,
} from '../../../src/shared/types/game';

function baseMove(overrides: Partial<Move>): Move {
  return {
    id: 'm1',
    player: 1,
    timestamp: new Date(),
    thinkTime: 0,
    moveNumber: 1,
    type: 'place_ring',
    to: { x: 0, y: 0 },
    ...overrides,
  } as Move;
}

describe('shared types – move type guards', () => {
  it('isMovementMove narrows move_stack and move_stack only', () => {
    const moveStack = baseMove({
      type: 'move_stack',
      from: { x: 0, y: 0 },
      to: { x: 1, y: 0 },
    }) as Move;
    const moveRing = baseMove({
      type: 'move_stack',
      from: { x: 0, y: 0 },
      to: { x: 0, y: 1 },
    }) as Move;
    const placement = baseMove({ type: 'place_ring', to: { x: 0, y: 0 } });

    expect(isMovementMove(moveStack)).toBe(true);
    expect(isMovementMove(moveRing)).toBe(true);
    expect(isMovementMove(placement)).toBe(false);

    if (isMovementMove(moveStack)) {
      const m: MovementMove = moveStack;
      expect(m.from).toEqual({ x: 0, y: 0 });
    }
  });

  it('isBuildStackMove matches only build_stack moves', () => {
    const build = baseMove({
      type: 'build_stack',
      from: { x: 0, y: 0 },
      to: { x: 0, y: 1 },
      buildAmount: 1,
    }) as Move;
    const movement = baseMove({
      type: 'move_stack',
      from: { x: 0, y: 0 },
      to: { x: 0, y: 1 },
    }) as Move;

    expect(isBuildStackMove(build)).toBe(true);
    expect(isBuildStackMove(movement)).toBe(false);

    if (isBuildStackMove(build)) {
      const b: BuildStackMove = build;
      expect(b.buildAmount).toBe(1);
    }
  });

  it('isCaptureMove matches overtaking_capture and continue_capture_segment only', () => {
    const capture = baseMove({
      type: 'overtaking_capture',
      from: { x: 0, y: 0 },
      to: { x: 1, y: 0 },
      captureTarget: { x: 2, y: 0 },
    }) as Move;
    const chain = baseMove({
      type: 'continue_capture_segment',
      from: { x: 0, y: 0 },
      to: { x: 1, y: 0 },
      captureTarget: { x: 2, y: 0 },
    }) as Move;
    const legacy = baseMove({ type: 'line_formation' }) as Move;

    expect(isCaptureMove(capture)).toBe(true);
    expect(isCaptureMove(chain)).toBe(true);
    expect(isCaptureMove(legacy)).toBe(false);

    if (isCaptureMove(capture)) {
      const c: CaptureMove = capture;
      expect(c.captureTarget).toEqual({ x: 2, y: 0 });
    }
  });

  it('isPlacementMove matches only place_ring moves', () => {
    const placement = baseMove({
      type: 'place_ring',
      to: { x: 2, y: 2 },
    }) as Move;
    const skip = baseMove({ type: 'skip_placement' }) as Move;

    expect(isPlacementMove(placement)).toBe(true);
    expect(isPlacementMove(skip)).toBe(false);

    if (isPlacementMove(placement)) {
      const p: PlacementMove = placement;
      expect(p.from).toBeUndefined();
      expect(p.to).toEqual({ x: 2, y: 2 });
    }
  });

  it('isSpatialMove matches movement, build, and capture moves', () => {
    const movement = baseMove({
      type: 'move_stack',
      from: { x: 0, y: 0 },
      to: { x: 1, y: 0 },
    }) as TypedMove;
    const build = baseMove({
      type: 'build_stack',
      from: { x: 0, y: 0 },
      to: { x: 0, y: 1 },
      buildAmount: 1,
    }) as TypedMove;
    const capture = baseMove({
      type: 'overtaking_capture',
      from: { x: 0, y: 0 },
      to: { x: 1, y: 0 },
      captureTarget: { x: 2, y: 0 },
    }) as TypedMove;
    const placement = baseMove({
      type: 'place_ring',
      to: { x: 3, y: 3 },
    }) as TypedMove;

    expect(isSpatialMove(movement)).toBe(true);
    expect(isSpatialMove(build)).toBe(true);
    expect(isSpatialMove(capture)).toBe(true);
    expect(isSpatialMove(placement)).toBe(false);
  });
});
