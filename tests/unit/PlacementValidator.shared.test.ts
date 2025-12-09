/**
 * PlacementValidator branch coverage tests
 * Tests for src/shared/engine/validators/PlacementValidator.ts
 *
 * Structural coverage for placement / skip-placement semantics aligned with:
 * - RulesMatrix placement cluster (parts of §§4, 8; FAQ 1–3).
 * - v2 placement contract vectors (`tests/fixtures/contract-vectors/v2/placement.vectors.json`).
 *
 * These tests exercise:
 * - Board geometry, collapsed-space and marker exclusivity.
 * - Per-player ring caps and ringsInHand supply.
 * - Per-cell caps (1 on existing stacks, up to 3 on empty cells).
 * - No-dead-placement invariant via hasAnyLegalMoveOrCaptureFromOnBoard.
 * - GameEngine-facing phase/turn checks and skip_placement gating.
 *
 * Scenario-level behaviour and turn transitions remain covered by:
 * - tests/contracts/contractVectorRunner.test.ts ("placement" suite)
 * - tests/unit/movement.shared.test.ts (for post-placement movement)
 * - tests/scenarios/RulesMatrix.Comprehensive.test.ts (placement rows)
 */

import {
  validatePlacementOnBoard,
  validatePlacement,
  validateSkipPlacement,
  PlacementContext,
} from '@shared/engine/validators/PlacementValidator';
import type { GameState, PlaceRingAction, SkipPlacementAction } from '@shared/engine/types';
import type { BoardState, BoardType } from '@shared/types/game';

// Helper to create minimal BoardState for placement tests
function createMinimalBoard(
  overrides: Partial<{
    type: BoardType;
    size: number;
    stacks: Map<
      string,
      {
        controllingPlayer: number;
        stackHeight: number;
        capHeight: number;
        rings: number[];
        position: { x: number; y: number };
      }
    >;
    markers: Map<string, { player: number }>;
    collapsedSpaces: Set<string>;
  }>
): BoardState {
  return {
    type: overrides.type ?? 'square8',
    size: overrides.size ?? 8,
    stacks: overrides.stacks ?? new Map(),
    markers: overrides.markers ?? new Map(),
    collapsedSpaces: overrides.collapsedSpaces ?? new Set(),
    rings: new Map(),
    territories: new Map(),
    geometry: { type: overrides.type ?? 'square8', size: overrides.size ?? 8 },
  } as BoardState;
}

// Helper to create minimal GameState for placement validation
function createMinimalState(
  overrides: Partial<{
    currentPhase: string;
    currentPlayer: number;
    boardType: BoardType;
    boardSize: number;
    stacks: Map<
      string,
      {
        controllingPlayer: number;
        stackHeight: number;
        capHeight: number;
        rings: number[];
        position: { x: number; y: number };
      }
    >;
    markers: Map<string, { player: number }>;
    collapsedSpaces: Set<string>;
    players: Array<{ playerNumber: number; ringsInHand: number; eliminated: boolean }>;
  }>
): GameState {
  const boardType = overrides.boardType ?? 'square8';
  const boardSize = overrides.boardSize ?? 8;

  const base = {
    board: createMinimalBoard({
      type: boardType,
      size: boardSize,
      stacks: overrides.stacks,
      markers: overrides.markers,
      collapsedSpaces: overrides.collapsedSpaces,
    }),
    currentPhase: overrides.currentPhase ?? 'ring_placement',
    currentPlayer: overrides.currentPlayer ?? 1,
    players: overrides.players ?? [
      {
        playerNumber: 1,
        ringsInHand: 10,
        eliminated: false,
        score: 0,
        reserveStacks: 0,
        reserveRings: 0,
      },
      {
        playerNumber: 2,
        ringsInHand: 10,
        eliminated: false,
        score: 0,
        reserveStacks: 0,
        reserveRings: 0,
      },
    ],
    turnNumber: 1,
    gameStatus: 'active' as const,
    moveHistory: [],
    pendingDecision: null,
    victoryCondition: null,
  };
  return base as unknown as GameState;
}

describe('PlacementValidator', () => {
  describe('validatePlacementOnBoard', () => {
    it('returns error when no rings in hand', () => {
      const board = createMinimalBoard({});
      const ctx: PlacementContext = {
        boardType: 'square8',
        player: 1,
        ringsInHand: 0,
        ringsPerPlayerCap: 12,
      };

      const result = validatePlacementOnBoard(board, { x: 2, y: 2 }, 1, ctx);

      expect(result.valid).toBe(false);
      expect(result.code).toBe('INSUFFICIENT_RINGS');
      expect(result.maxPlacementCount).toBe(0);
    });

    it('returns error when position is off board', () => {
      const board = createMinimalBoard({ size: 8 });
      const ctx: PlacementContext = {
        boardType: 'square8',
        player: 1,
        ringsInHand: 5,
        ringsPerPlayerCap: 12,
      };

      const result = validatePlacementOnBoard(board, { x: -1, y: 0 }, 1, ctx);

      expect(result.valid).toBe(false);
      expect(result.code).toBe('INVALID_POSITION');
      expect(result.maxPlacementCount).toBe(0);
    });

    it('returns error when placing on collapsed space', () => {
      const collapsedSpaces = new Set(['2,2']);
      const board = createMinimalBoard({ collapsedSpaces });
      const ctx: PlacementContext = {
        boardType: 'square8',
        player: 1,
        ringsInHand: 5,
        ringsPerPlayerCap: 12,
      };

      const result = validatePlacementOnBoard(board, { x: 2, y: 2 }, 1, ctx);

      expect(result.valid).toBe(false);
      expect(result.code).toBe('COLLAPSED_SPACE');
    });

    it('returns error when placing on marker', () => {
      const markers = new Map([['2,2', { player: 1 }]]);
      const board = createMinimalBoard({ markers });
      const ctx: PlacementContext = {
        boardType: 'square8',
        player: 1,
        ringsInHand: 5,
        ringsPerPlayerCap: 12,
      };

      const result = validatePlacementOnBoard(board, { x: 2, y: 2 }, 1, ctx);

      expect(result.valid).toBe(false);
      expect(result.code).toBe('MARKER_BLOCKED');
    });

    it('returns error when ring cap reached', () => {
      const board = createMinimalBoard({});
      const ctx: PlacementContext = {
        boardType: 'square8',
        player: 1,
        ringsInHand: 5,
        ringsPerPlayerCap: 12,
        ringsOnBoard: 12, // Already at cap
      };

      const result = validatePlacementOnBoard(board, { x: 2, y: 2 }, 1, ctx);

      expect(result.valid).toBe(false);
      expect(result.code).toBe('NO_RINGS_AVAILABLE');
    });

    it('returns error when count exceeds per-cell cap on existing stack', () => {
      const stacks = new Map([
        [
          '2,2',
          {
            controllingPlayer: 1,
            stackHeight: 2,
            capHeight: 2,
            rings: [1, 1],
            position: { x: 2, y: 2 },
          },
        ],
      ]);
      const board = createMinimalBoard({ stacks });
      const ctx: PlacementContext = {
        boardType: 'square8',
        player: 1,
        ringsInHand: 5,
        ringsPerPlayerCap: 12,
        ringsOnBoard: 2,
      };

      // Placing 2 rings on existing stack (max is 1)
      const result = validatePlacementOnBoard(board, { x: 2, y: 2 }, 2, ctx);

      expect(result.valid).toBe(false);
      expect(result.code).toBe('INVALID_COUNT');
      expect(result.maxPlacementCount).toBe(1);
    });

    it('returns error when count exceeds 3 on empty space', () => {
      const board = createMinimalBoard({});
      const ctx: PlacementContext = {
        boardType: 'square8',
        player: 1,
        ringsInHand: 10,
        ringsPerPlayerCap: 12,
        ringsOnBoard: 0,
      };

      // Placing 4 rings on empty space (max is 3)
      const result = validatePlacementOnBoard(board, { x: 2, y: 2 }, 4, ctx);

      expect(result.valid).toBe(false);
      expect(result.code).toBe('INVALID_COUNT');
      expect(result.maxPlacementCount).toBe(3);
    });

    it('returns error when count is less than 1', () => {
      const board = createMinimalBoard({});
      const ctx: PlacementContext = {
        boardType: 'square8',
        player: 1,
        ringsInHand: 5,
        ringsPerPlayerCap: 12,
        ringsOnBoard: 0,
      };

      const result = validatePlacementOnBoard(board, { x: 2, y: 2 }, 0, ctx);

      expect(result.valid).toBe(false);
      expect(result.code).toBe('INVALID_COUNT');
    });

    it('respects precomputed maxAvailableGlobal', () => {
      const board = createMinimalBoard({});
      const ctx: PlacementContext = {
        boardType: 'square8',
        player: 1,
        ringsInHand: 10,
        ringsPerPlayerCap: 12,
        maxAvailableGlobal: 0, // Pretend no rings available
      };

      const result = validatePlacementOnBoard(board, { x: 2, y: 2 }, 1, ctx);

      expect(result.valid).toBe(false);
      expect(result.code).toBe('NO_RINGS_AVAILABLE');
    });

    it('respects per-player cap and per-cell cap on square19 boards', () => {
      // Square19 uses the same per-cell caps, but has a higher ringsPerPlayerCap.
      const stacks = new Map([
        [
          '5,5',
          {
            controllingPlayer: 1,
            stackHeight: 2,
            capHeight: 2,
            rings: [1, 1],
            position: { x: 5, y: 5 },
          },
        ],
      ]);
      const board = createMinimalBoard({ type: 'square19', size: 19, stacks });
      const ctx: PlacementContext = {
        boardType: 'square19',
        player: 1,
        ringsInHand: 10,
        ringsPerPlayerCap: 36,
        ringsOnBoard: 30,
      };

      // Global remaining capacity is 6, but per-cell cap on existing stack is 1.
      const overCount = validatePlacementOnBoard(board, { x: 5, y: 5 }, 2, ctx);
      expect(overCount.valid).toBe(false);
      expect(overCount.code).toBe('INVALID_COUNT');
      expect(overCount.maxPlacementCount).toBe(1);

      const ok = validatePlacementOnBoard(board, { x: 5, y: 5 }, 1, ctx);
      expect(ok.valid).toBe(true);
      expect(ok.maxPlacementCount).toBe(1);
    });

    it('rejects off-board and collapsed placements consistently on hex boards', () => {
      const collapsedSpaces = new Set(['0,0,0']);
      const board = createMinimalBoard({
        type: 'hexagonal',
        size: 4,
        collapsedSpaces,
      });
      const ctx: PlacementContext = {
        boardType: 'hexagonal',
        player: 1,
        ringsInHand: 5,
        ringsPerPlayerCap: 36,
      };

      // Off-board axial coordinate
      const offBoard = validatePlacementOnBoard(board, { x: 5, y: -5, z: 0 } as any, 1, ctx);
      expect(offBoard.valid).toBe(false);
      expect(offBoard.code).toBe('INVALID_POSITION');

      // Collapsed center hex
      const collapsed = validatePlacementOnBoard(board, { x: 0, y: 0, z: 0 } as any, 1, ctx);
      expect(collapsed.valid).toBe(false);
      expect(collapsed.code).toBe('COLLAPSED_SPACE');
    });
  });

  describe('validatePlacement', () => {
    it('returns error when not in ring_placement phase', () => {
      const state = createMinimalState({ currentPhase: 'movement' });
      const action: PlaceRingAction = {
        type: 'placeRing',
        position: { x: 2, y: 2 },
        count: 1,
        playerId: 1,
      };

      const result = validatePlacement(state, action);

      expect(result.valid).toBe(false);
      expect(result.code).toBe('INVALID_PHASE');
    });

    it('returns error when not player turn', () => {
      const state = createMinimalState({
        currentPhase: 'ring_placement',
        currentPlayer: 2,
      });
      const action: PlaceRingAction = {
        type: 'placeRing',
        position: { x: 2, y: 2 },
        count: 1,
        playerId: 1,
      };

      const result = validatePlacement(state, action);

      expect(result.valid).toBe(false);
      expect(result.code).toBe('NOT_YOUR_TURN');
    });

    it('returns error when player not found', () => {
      const state = createMinimalState({
        currentPhase: 'ring_placement',
        currentPlayer: 99,
        players: [
          { playerNumber: 1, ringsInHand: 10, eliminated: false },
          { playerNumber: 2, ringsInHand: 10, eliminated: false },
        ],
      });
      const action: PlaceRingAction = {
        type: 'placeRing',
        position: { x: 2, y: 2 },
        count: 1,
        playerId: 99, // Non-existent player
      };

      const result = validatePlacement(state, action);

      expect(result.valid).toBe(false);
      expect(result.code).toBe('PLAYER_NOT_FOUND');
    });

    it('returns error when count is 0', () => {
      const state = createMinimalState({
        currentPhase: 'ring_placement',
        currentPlayer: 1,
      });
      const action: PlaceRingAction = {
        type: 'placeRing',
        position: { x: 2, y: 2 },
        count: 0,
        playerId: 1,
      };

      const result = validatePlacement(state, action);

      expect(result.valid).toBe(false);
      expect(result.code).toBe('INVALID_COUNT');
    });

    it('returns error when insufficient rings in hand', () => {
      const state = createMinimalState({
        currentPhase: 'ring_placement',
        currentPlayer: 1,
        players: [
          { playerNumber: 1, ringsInHand: 1, eliminated: false },
          { playerNumber: 2, ringsInHand: 10, eliminated: false },
        ],
      });
      const action: PlaceRingAction = {
        type: 'placeRing',
        position: { x: 2, y: 2 },
        count: 3, // More than ringsInHand
        playerId: 1,
      };

      const result = validatePlacement(state, action);

      expect(result.valid).toBe(false);
      expect(result.code).toBe('INSUFFICIENT_RINGS');
    });
  });

  describe('validateSkipPlacement', () => {
    it('returns error when not in ring_placement phase', () => {
      const state = createMinimalState({ currentPhase: 'movement' });
      const action: SkipPlacementAction = {
        type: 'skipPlacement',
        playerId: 1,
      };

      const result = validateSkipPlacement(state, action);

      expect(result.valid).toBe(false);
      expect(result.code).toBe('INVALID_PHASE');
    });

    it('returns error when not player turn', () => {
      const state = createMinimalState({
        currentPhase: 'ring_placement',
        currentPlayer: 2,
      });
      const action: SkipPlacementAction = {
        type: 'skipPlacement',
        playerId: 1,
      };

      const result = validateSkipPlacement(state, action);

      expect(result.valid).toBe(false);
      expect(result.code).toBe('NOT_YOUR_TURN');
    });

    it('returns error when player not found', () => {
      const state = createMinimalState({
        currentPhase: 'ring_placement',
        currentPlayer: 99,
        players: [
          { playerNumber: 1, ringsInHand: 10, eliminated: false },
          { playerNumber: 2, ringsInHand: 10, eliminated: false },
        ],
      });
      const action: SkipPlacementAction = {
        type: 'skipPlacement',
        playerId: 99,
      };

      const result = validateSkipPlacement(state, action);

      expect(result.valid).toBe(false);
      expect(result.code).toBe('PLAYER_NOT_FOUND');
    });

    it('returns error when player controls no stacks', () => {
      // FAQ 1–3 / placement cluster: skip_placement requires at least one
      // controlled stack so that the decision to skip is meaningful; otherwise
      // line/territory phases would be unreachable for that player.
      const state = createMinimalState({
        currentPhase: 'ring_placement',
        currentPlayer: 1,
        stacks: new Map(), // No stacks
      });
      const action: SkipPlacementAction = {
        type: 'skipPlacement',
        playerId: 1,
      };

      const result = validateSkipPlacement(state, action);

      expect(result.valid).toBe(false);
      expect(result.code).toBe('NO_CONTROLLED_STACKS');
    });

    it('returns error when player has stacks but no legal actions', () => {
      // FAQ 1–3: cannot skip if doing so would strand the player with no
      // legal follow-up actions; no-dead-placement / skip semantics ensure
      // turn progression still leads to a legal move or capture.
      // Create a stack in a cornered position with collapsed spaces blocking all moves
      const stacks = new Map([
        [
          '0,0',
          {
            controllingPlayer: 1,
            stackHeight: 1,
            capHeight: 1,
            rings: [1],
            position: { x: 0, y: 0 },
          },
        ],
      ]);
      // Block all adjacent spaces
      const collapsedSpaces = new Set(['0,1', '1,0', '1,1']);
      const state = createMinimalState({
        currentPhase: 'ring_placement',
        currentPlayer: 1,
        stacks,
        collapsedSpaces,
      });
      const action: SkipPlacementAction = {
        type: 'skipPlacement',
        playerId: 1,
      };

      const result = validateSkipPlacement(state, action);

      expect(result.valid).toBe(false);
      expect(result.code).toBe('NO_LEGAL_ACTIONS');
    });

    it('allows skip when player has controlled stack with legal moves', () => {
      // Create a stack with clear movement path
      const stacks = new Map([
        [
          '3,3',
          {
            controllingPlayer: 1,
            stackHeight: 1,
            capHeight: 1,
            rings: [1],
            position: { x: 3, y: 3 },
          },
        ],
      ]);
      const state = createMinimalState({
        currentPhase: 'ring_placement',
        currentPlayer: 1,
        stacks,
      });
      const action: SkipPlacementAction = {
        type: 'skipPlacement',
        playerId: 1,
      };

      const result = validateSkipPlacement(state, action);

      expect(result.valid).toBe(true);
    });

    it('handles player with stack controlled by other player', () => {
      // Player 1's turn but only player 2 has stacks
      const stacks = new Map([
        [
          '3,3',
          {
            controllingPlayer: 2,
            stackHeight: 1,
            capHeight: 1,
            rings: [2],
            position: { x: 3, y: 3 },
          },
        ],
      ]);
      const state = createMinimalState({
        currentPhase: 'ring_placement',
        currentPlayer: 1,
        stacks,
      });
      const action: SkipPlacementAction = {
        type: 'skipPlacement',
        playerId: 1,
      };

      const result = validateSkipPlacement(state, action);

      expect(result.valid).toBe(false);
      expect(result.code).toBe('NO_CONTROLLED_STACKS');
    });

    it('skips stacks with stackHeight 0', () => {
      // A stack with stackHeight 0 should be ignored
      const stacks = new Map([
        [
          '3,3',
          {
            controllingPlayer: 1,
            stackHeight: 0,
            capHeight: 0,
            rings: [],
            position: { x: 3, y: 3 },
          },
        ],
      ]);
      const state = createMinimalState({
        currentPhase: 'ring_placement',
        currentPlayer: 1,
        stacks,
        players: [
          { playerNumber: 1, ringsInHand: 5, eliminated: false },
          { playerNumber: 2, ringsInHand: 10, eliminated: false },
        ],
      });
      const action: SkipPlacementAction = {
        type: 'skipPlacement',
        playerId: 1,
      };

      const result = validateSkipPlacement(state, action);

      expect(result.valid).toBe(false);
      expect(result.code).toBe('NO_CONTROLLED_STACKS');
    });

    it('rejects skip when ringsInHand is 0 even if player has legal moves', () => {
      // Per canonical rules: skip_placement requires ringsInHand > 0. When ringsInHand = 0,
      // the player should use no_placement_action instead (placement is skipped automatically).
      const stacks = new Map([
        [
          '3,3',
          {
            controllingPlayer: 1,
            stackHeight: 1,
            capHeight: 1,
            rings: [1],
            position: { x: 3, y: 3 },
          },
        ],
      ]);
      const state = createMinimalState({
        currentPhase: 'ring_placement',
        currentPlayer: 1,
        stacks,
        players: [
          { playerNumber: 1, ringsInHand: 0, eliminated: false },
          { playerNumber: 2, ringsInHand: 10, eliminated: false },
        ],
      });
      const action: SkipPlacementAction = {
        type: 'skipPlacement',
        playerId: 1,
      };

      const result = validateSkipPlacement(state, action);

      // skip_placement is invalid when ringsInHand = 0; use no_placement_action instead
      expect(result.valid).toBe(false);
      expect(result.code).toBe('NO_RINGS_IN_HAND');
    });
  });

  describe('validatePlacementOnBoard - additional branch coverage', () => {
    it('computes ringsOnBoard from board when not precomputed', () => {
      // Don't provide ringsOnBoard in ctx, force computing from board
      const stacks = new Map([
        [
          '2,2',
          {
            controllingPlayer: 1,
            stackHeight: 2,
            capHeight: 2,
            rings: [1, 1],
            position: { x: 2, y: 2 },
          },
        ],
      ]);
      const board = createMinimalBoard({ stacks });
      const ctx: PlacementContext = {
        boardType: 'square8',
        player: 1,
        ringsInHand: 5,
        ringsPerPlayerCap: 12,
        // ringsOnBoard not provided - should be computed
      };

      // Place 1 ring on existing stack (valid)
      const result = validatePlacementOnBoard(board, { x: 2, y: 2 }, 1, ctx);

      // Should compute rings on board and allow placement
      expect(result.valid).toBe(true);
      expect(result.maxPlacementCount).toBe(1);
    });

    it('handles placement on existing stack controlled by different player', () => {
      // Stack controlled by player 2, player 1 is placing
      const stacks = new Map([
        [
          '2,2',
          {
            controllingPlayer: 2,
            stackHeight: 1,
            capHeight: 1,
            rings: [2],
            position: { x: 2, y: 2 },
          },
        ],
      ]);
      const board = createMinimalBoard({ stacks });
      const ctx: PlacementContext = {
        boardType: 'square8',
        player: 1,
        ringsInHand: 5,
        ringsPerPlayerCap: 12,
        ringsOnBoard: 0,
      };

      // The hypothetical stack changes control, so capHeight becomes count (not added)
      const result = validatePlacementOnBoard(board, { x: 2, y: 2 }, 1, ctx);

      // Should still be valid (1 ring on existing stack)
      expect(result.valid).toBe(true);
      expect(result.maxPlacementCount).toBe(1);
    });

    it('returns valid placement on empty cell with 3 rings', () => {
      const board = createMinimalBoard({});
      const ctx: PlacementContext = {
        boardType: 'square8',
        player: 1,
        ringsInHand: 10,
        ringsPerPlayerCap: 12,
        ringsOnBoard: 0,
      };

      const result = validatePlacementOnBoard(board, { x: 3, y: 3 }, 3, ctx);

      expect(result.valid).toBe(true);
      expect(result.maxPlacementCount).toBe(3);
    });

    it('handles cap limiting maxPlacementCount on empty cell', () => {
      const board = createMinimalBoard({});
      const ctx: PlacementContext = {
        boardType: 'square8',
        player: 1,
        ringsInHand: 2, // Only 2 rings available
        ringsPerPlayerCap: 12,
        ringsOnBoard: 0,
      };

      // Request 2 (which is valid since we only have 2)
      const result = validatePlacementOnBoard(board, { x: 3, y: 3 }, 2, ctx);

      expect(result.valid).toBe(true);
      expect(result.maxPlacementCount).toBe(2);
    });

    it('returns error when requesting more rings than supply on empty cell', () => {
      const board = createMinimalBoard({});
      const ctx: PlacementContext = {
        boardType: 'square8',
        player: 1,
        ringsInHand: 2, // Only 2 rings available
        ringsPerPlayerCap: 12,
        ringsOnBoard: 0,
      };

      // Request 3 but only 2 available
      const result = validatePlacementOnBoard(board, { x: 3, y: 3 }, 3, ctx);

      expect(result.valid).toBe(false);
      expect(result.code).toBe('INVALID_COUNT');
      expect(result.maxPlacementCount).toBe(2);
    });

    it('handles position off board (y < 0)', () => {
      const board = createMinimalBoard({ size: 8 });
      const ctx: PlacementContext = {
        boardType: 'square8',
        player: 1,
        ringsInHand: 5,
        ringsPerPlayerCap: 12,
      };

      const result = validatePlacementOnBoard(board, { x: 0, y: -1 }, 1, ctx);

      expect(result.valid).toBe(false);
      expect(result.code).toBe('INVALID_POSITION');
    });

    it('handles position off board (x >= size)', () => {
      const board = createMinimalBoard({ size: 8 });
      const ctx: PlacementContext = {
        boardType: 'square8',
        player: 1,
        ringsInHand: 5,
        ringsPerPlayerCap: 12,
      };

      const result = validatePlacementOnBoard(board, { x: 8, y: 0 }, 1, ctx);

      expect(result.valid).toBe(false);
      expect(result.code).toBe('INVALID_POSITION');
    });

    it('handles position off board (y >= size)', () => {
      const board = createMinimalBoard({ size: 8 });
      const ctx: PlacementContext = {
        boardType: 'square8',
        player: 1,
        ringsInHand: 5,
        ringsPerPlayerCap: 12,
      };

      const result = validatePlacementOnBoard(board, { x: 0, y: 8 }, 1, ctx);

      expect(result.valid).toBe(false);
      expect(result.code).toBe('INVALID_POSITION');
    });
  });

  describe('validatePlacement - additional branch coverage', () => {
    it('returns valid for correct placement', () => {
      // Need to create a state where placement is fully valid
      const stacks = new Map([
        [
          '3,3',
          {
            controllingPlayer: 1,
            stackHeight: 1,
            capHeight: 1,
            rings: [1],
            position: { x: 3, y: 3 },
          },
        ],
      ]);
      const state = createMinimalState({
        currentPhase: 'ring_placement',
        currentPlayer: 1,
        stacks,
        players: [
          { playerNumber: 1, ringsInHand: 10, eliminated: false },
          { playerNumber: 2, ringsInHand: 10, eliminated: false },
        ],
      });
      const action: PlaceRingAction = {
        type: 'placeRing',
        position: { x: 4, y: 4 }, // Empty cell that allows moves
        count: 1,
        playerId: 1,
      };

      const result = validatePlacement(state, action);

      expect(result.valid).toBe(true);
    });

    it('returns error with negative count', () => {
      const state = createMinimalState({
        currentPhase: 'ring_placement',
        currentPlayer: 1,
      });
      const action: PlaceRingAction = {
        type: 'placeRing',
        position: { x: 2, y: 2 },
        count: -1,
        playerId: 1,
      };

      const result = validatePlacement(state, action);

      expect(result.valid).toBe(false);
      expect(result.code).toBe('INVALID_COUNT');
    });

    it('returns error when ringsInHand is 0', () => {
      const state = createMinimalState({
        currentPhase: 'ring_placement',
        currentPlayer: 1,
        players: [
          { playerNumber: 1, ringsInHand: 0, eliminated: false },
          { playerNumber: 2, ringsInHand: 10, eliminated: false },
        ],
      });
      const action: PlaceRingAction = {
        type: 'placeRing',
        position: { x: 2, y: 2 },
        count: 1,
        playerId: 1,
      };

      const result = validatePlacement(state, action);

      expect(result.valid).toBe(false);
      expect(result.code).toBe('INSUFFICIENT_RINGS');
    });
  });
});
