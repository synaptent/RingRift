import {
  BoardState,
  BoardType,
  BOARD_CONFIGS,
  Position,
  RingStack,
  positionToString,
  MarkerInfo,
} from '../../src/shared/types/game';
import {
  validatePlacementOnBoard,
  PlacementContext,
} from '../../src/shared/engine/validators/PlacementValidator';
import { applyPlacementOnBoard } from '../../src/shared/engine/mutators/PlacementMutator';

function createEmptyBoard(boardType: BoardType): BoardState {
  const config = BOARD_CONFIGS[boardType];
  return {
    stacks: new Map<string, RingStack>(),
    markers: new Map<string, MarkerInfo>(),
    collapsedSpaces: new Map<string, number>(),
    territories: new Map(),
    formedLines: [],
    eliminatedRings: {},
    size: config.size,
    type: boardType,
  };
}

describe('shared placement validator and mutator', () => {
  describe('capacity & multi-ring semantics on square8', () => {
    const boardType: BoardType = 'square8';

    test('empty cell: counts 1..3 valid, >3 rejected; maxPlacementCount reflects cap', () => {
      const board = createEmptyBoard(boardType);
      const ctx: PlacementContext = {
        boardType,
        player: 1,
        ringsInHand: 5,
        ringsPerPlayerCap: BOARD_CONFIGS[boardType].ringsPerPlayer,
      };
      const pos: Position = { x: 3, y: 3 };

      const res1 = validatePlacementOnBoard(board, pos, 1, ctx);
      const res2 = validatePlacementOnBoard(board, pos, 2, ctx);
      const res3 = validatePlacementOnBoard(board, pos, 3, ctx);
      const res4 = validatePlacementOnBoard(board, pos, 4, ctx);

      expect(res1.valid).toBe(true);
      expect(res2.valid).toBe(true);
      expect(res3.valid).toBe(true);
      expect(res1.maxPlacementCount).toBe(3);
      expect(res2.maxPlacementCount).toBe(3);
      expect(res3.maxPlacementCount).toBe(3);

      expect(res4.valid).toBe(false);
      expect(res4.maxPlacementCount).toBe(3);
      expect(res4.code).toBe('INVALID_COUNT');
    });

    test('per-player cap: global maxAvailableGlobal clamps placement', () => {
      const board = createEmptyBoard(boardType);
      const stackPos: Position = { x: 0, y: 0 };
      const stackKey = positionToString(stackPos);

      const ringsOnBoard = 17;
      const ringsInHand = 5;
      const perPlayerCap = BOARD_CONFIGS[boardType].ringsPerPlayer;

      const existingStack: RingStack = {
        position: stackPos,
        rings: new Array(ringsOnBoard).fill(1),
        stackHeight: ringsOnBoard,
        capHeight: ringsOnBoard,
        controllingPlayer: 1,
      };
      board.stacks.set(stackKey, existingStack);

      const ctx: PlacementContext = {
        boardType,
        player: 1,
        ringsInHand,
        ringsPerPlayerCap: perPlayerCap,
        ringsOnBoard,
        maxAvailableGlobal: Math.min(perPlayerCap - ringsOnBoard, ringsInHand),
      };

      const target: Position = { x: 7, y: 7 };

      const ok = validatePlacementOnBoard(board, target, 1, ctx);
      const tooMany = validatePlacementOnBoard(board, target, 2, ctx);

      expect(ok.valid).toBe(true);
      expect(ok.maxPlacementCount).toBe(1);

      expect(tooMany.valid).toBe(false);
      expect(tooMany.maxPlacementCount).toBe(1);
      // Either capacity or count error is acceptable; we primarily care that
      // the requested count is rejected once the global cap is effectively 1.
      expect(tooMany.code === 'INVALID_COUNT' || tooMany.code === 'NO_RINGS_AVAILABLE').toBe(true);
    });

    test('existing stack: only single-ring placements are accepted', () => {
      const board = createEmptyBoard(boardType);
      const pos: Position = { x: 2, y: 2 };
      const key = positionToString(pos);

      const baseStack: RingStack = {
        position: pos,
        rings: [1],
        stackHeight: 1,
        capHeight: 1,
        controllingPlayer: 1,
      };
      board.stacks.set(key, baseStack);

      const ctx: PlacementContext = {
        boardType,
        player: 1,
        ringsInHand: 5,
        ringsPerPlayerCap: BOARD_CONFIGS[boardType].ringsPerPlayer,
      };

      const oneRing = validatePlacementOnBoard(board, pos, 1, ctx);
      const twoRings = validatePlacementOnBoard(board, pos, 2, ctx);

      expect(oneRing.valid).toBe(true);
      expect(oneRing.maxPlacementCount).toBe(1);

      expect(twoRings.valid).toBe(false);
      expect(twoRings.maxPlacementCount).toBe(1);
      expect(twoRings.code).toBe('INVALID_COUNT');
    });
  });

  describe('marker/stack exclusivity and collapsed spaces', () => {
    const boardType: BoardType = 'square8';

    test('cannot place on markers or collapsed spaces', () => {
      const board = createEmptyBoard(boardType);
      const collapsedPos: Position = { x: 1, y: 1 };
      const markerPos: Position = { x: 2, y: 2 };

      const collapsedKey = positionToString(collapsedPos);
      const markerKey = positionToString(markerPos);

      board.collapsedSpaces.set(collapsedKey, 1);
      board.markers.set(markerKey, {
        player: 1,
        position: markerPos,
        type: 'regular',
      });

      const ctx: PlacementContext = {
        boardType,
        player: 1,
        ringsInHand: 5,
        ringsPerPlayerCap: BOARD_CONFIGS[boardType].ringsPerPlayer,
      };

      const onCollapsed = validatePlacementOnBoard(board, collapsedPos, 1, ctx);
      const onMarker = validatePlacementOnBoard(board, markerPos, 1, ctx);

      expect(onCollapsed.valid).toBe(false);
      expect(onCollapsed.code).toBe('COLLAPSED_SPACE');

      expect(onMarker.valid).toBe(false);
      expect(onMarker.code).toBe('MARKER_BLOCKED');
    });
  });

  describe('no-dead-placement semantics', () => {
    const boardType: BoardType = 'square8';

    test('placement rejected when resulting stack has no legal moves or captures', () => {
      const board = createEmptyBoard(boardType);
      const pos: Position = { x: 3, y: 3 };
      const ctx: PlacementContext = {
        boardType,
        player: 1,
        ringsInHand: 3,
        ringsPerPlayerCap: BOARD_CONFIGS[boardType].ringsPerPlayer,
      };

      // Surround pos with collapsed spaces so that a height-1 stack has no
      // legal non-capture moves or captures.
      for (let dx = -1; dx <= 1; dx++) {
        for (let dy = -1; dy <= 1; dy++) {
          if (dx === 0 && dy === 0) continue;
          const nx = pos.x + dx;
          const ny = pos.y + dy;
          if (nx < 0 || nx >= board.size || ny < 0 || ny >= board.size) continue;
          const nKey = positionToString({ x: nx, y: ny });
          board.collapsedSpaces.set(nKey, 1);
        }
      }

      const res = validatePlacementOnBoard(board, pos, 1, ctx);

      expect(res.valid).toBe(false);
      expect(res.code).toBe('NO_LEGAL_MOVES');
    });

    test('placement allowed when at least one legal move exists', () => {
      const board = createEmptyBoard(boardType);
      const pos: Position = { x: 3, y: 3 };
      const ctx: PlacementContext = {
        boardType,
        player: 1,
        ringsInHand: 3,
        ringsPerPlayerCap: BOARD_CONFIGS[boardType].ringsPerPlayer,
      };

      // Leave board otherwise empty: at least one adjacent non-collapsed
      // cell exists so the resulting stack has a legal move.
      const res = validatePlacementOnBoard(board, pos, 1, ctx);

      expect(res.valid).toBe(true);
      expect(res.maxPlacementCount).toBeGreaterThanOrEqual(1);
    });
  });

  describe('hex board geometry', () => {
    const boardType: BoardType = 'hexagonal';

    test('central hex accepts placement; off-board coordinates are rejected', () => {
      const board = createEmptyBoard(boardType);
      const ctx: PlacementContext = {
        boardType,
        player: 1,
        ringsInHand: 4,
        ringsPerPlayerCap: BOARD_CONFIGS[boardType].ringsPerPlayer,
      };

      const center: Position = { x: 0, y: 0, z: 0 };
      const centerRes = validatePlacementOnBoard(board, center, 1, ctx);
      expect(centerRes.valid).toBe(true);

      // Clearly off-board in hex coordinates (beyond radius).
      const offBoard: Position = {
        x: BOARD_CONFIGS[boardType].size,
        y: 0,
        z: -BOARD_CONFIGS[boardType].size,
      };
      const offBoardRes = validatePlacementOnBoard(board, offBoard, 1, ctx);
      expect(offBoardRes.valid).toBe(false);
      expect(offBoardRes.code).toBe('INVALID_POSITION');
    });
  });

  describe('applyPlacementOnBoard integration', () => {
    const boardType: BoardType = 'square8';

    test('applyPlacementOnBoard adds rings on top and clears markers', () => {
      const board = createEmptyBoard(boardType);
      const pos: Position = { x: 4, y: 4 };
      const key = positionToString(pos);

      // Start with a marker and an existing mixed stack.
      board.markers.set(key, {
        player: 2,
        position: pos,
        type: 'regular',
      });

      const baseStack: RingStack = {
        position: pos,
        rings: [2, 1], // top ring is player 2
        stackHeight: 2,
        capHeight: 1,
        controllingPlayer: 2,
      };
      board.stacks.set(key, baseStack);

      const updated = applyPlacementOnBoard(board, pos, 1, 2);

      const markerAfter = updated.markers.get(key);
      const stackAfter = updated.stacks.get(key);

      expect(markerAfter).toBeUndefined();
      expect(stackAfter).toBeDefined();
      expect(stackAfter!.rings.length).toBe(4);
      // New rings for player 1 are added on top.
      expect(stackAfter!.rings[0]).toBe(1);
    });
  });
});
