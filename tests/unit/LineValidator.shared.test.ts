/**
 * LineValidator branch coverage tests
 * Tests for src/shared/engine/validators/LineValidator.ts
 *
 * Tests cover validateProcessLine and validateChooseLineReward
 * with focus on branch coverage for all validation paths.
 */

import {
  validateProcessLine,
  validateChooseLineReward,
} from '@shared/engine/validators/LineValidator';
import type { GameState, ProcessLineAction, ChooseLineRewardAction } from '@shared/engine/types';
import type { BoardState, BoardType } from '@shared/types/game';

// Helper to create minimal BoardState for line validation tests
function createMinimalBoard(
  overrides: Partial<{
    type: BoardType;
    size: number;
    formedLines: Array<{
      player: number;
      length: number;
      positions: Array<{ x: number; y: number }>;
    }>;
  }>
): BoardState {
  return {
    type: overrides.type ?? 'square8',
    size: overrides.size ?? 8,
    stacks: new Map(),
    markers: new Map(),
    collapsedSpaces: new Set(),
    rings: new Map(),
    territories: new Map(),
    formedLines: overrides.formedLines ?? [],
    geometry: { type: overrides.type ?? 'square8', size: overrides.size ?? 8 },
  } as BoardState;
}

// Helper to create minimal GameState for line validation
function createMinimalState(
  overrides: Partial<{
    currentPhase: string;
    currentPlayer: number;
    boardType: BoardType;
    boardSize: number;
    formedLines: Array<{
      player: number;
      length: number;
      positions: Array<{ x: number; y: number }>;
    }>;
    playerCount: number;
  }>
): GameState {
  const boardType = overrides.boardType ?? 'square8';
  const boardSize = overrides.boardSize ?? 8;
  const playerCount = overrides.playerCount ?? 2;

  const players = Array.from({ length: playerCount }, (_, i) => ({
    playerNumber: i + 1,
    ringsInHand: 10,
    eliminated: false,
    score: 0,
    reserveStacks: 0,
    reserveRings: 0,
  }));

  const base = {
    board: createMinimalBoard({
      type: boardType,
      size: boardSize,
      formedLines: overrides.formedLines,
    }),
    currentPhase: overrides.currentPhase ?? 'line_processing',
    currentPlayer: overrides.currentPlayer ?? 1,
    players,
    turnNumber: 1,
    gameStatus: 'active' as const,
    moveHistory: [],
    pendingDecision: null,
    victoryCondition: null,
  };
  return base as unknown as GameState;
}

describe('LineValidator', () => {
  describe('validateProcessLine', () => {
    it('returns error when not in line_processing phase', () => {
      const state = createMinimalState({
        currentPhase: 'movement',
        formedLines: [{ player: 1, length: 4, positions: [] }],
      });
      const action: ProcessLineAction = {
        type: 'processLine',
        lineIndex: 0,
        playerId: 1,
      };

      const result = validateProcessLine(state, action);

      expect(result.valid).toBe(false);
      expect(result.code).toBe('INVALID_PHASE');
    });

    it('returns error when not player turn', () => {
      const state = createMinimalState({
        currentPhase: 'line_processing',
        currentPlayer: 2,
        formedLines: [{ player: 1, length: 4, positions: [] }],
      });
      const action: ProcessLineAction = {
        type: 'processLine',
        lineIndex: 0,
        playerId: 1,
      };

      const result = validateProcessLine(state, action);

      expect(result.valid).toBe(false);
      expect(result.code).toBe('NOT_YOUR_TURN');
    });

    it('returns error when line index is negative', () => {
      const state = createMinimalState({
        currentPhase: 'line_processing',
        currentPlayer: 1,
        formedLines: [{ player: 1, length: 4, positions: [] }],
      });
      const action: ProcessLineAction = {
        type: 'processLine',
        lineIndex: -1,
        playerId: 1,
      };

      const result = validateProcessLine(state, action);

      expect(result.valid).toBe(false);
      expect(result.code).toBe('INVALID_LINE_INDEX');
    });

    it('returns error when line index exceeds available lines', () => {
      const state = createMinimalState({
        currentPhase: 'line_processing',
        currentPlayer: 1,
        formedLines: [{ player: 1, length: 4, positions: [] }],
      });
      const action: ProcessLineAction = {
        type: 'processLine',
        lineIndex: 1, // Only index 0 exists
        playerId: 1,
      };

      const result = validateProcessLine(state, action);

      expect(result.valid).toBe(false);
      expect(result.code).toBe('INVALID_LINE_INDEX');
    });

    it('returns error when line belongs to opponent', () => {
      const state = createMinimalState({
        currentPhase: 'line_processing',
        currentPlayer: 1,
        formedLines: [{ player: 2, length: 4, positions: [] }], // Player 2's line
      });
      const action: ProcessLineAction = {
        type: 'processLine',
        lineIndex: 0,
        playerId: 1,
      };

      const result = validateProcessLine(state, action);

      expect(result.valid).toBe(false);
      expect(result.code).toBe('NOT_YOUR_LINE');
    });

    it('allows valid process line action', () => {
      const state = createMinimalState({
        currentPhase: 'line_processing',
        currentPlayer: 1,
        formedLines: [{ player: 1, length: 4, positions: [] }],
      });
      const action: ProcessLineAction = {
        type: 'processLine',
        lineIndex: 0,
        playerId: 1,
      };

      const result = validateProcessLine(state, action);

      expect(result.valid).toBe(true);
    });
  });

  describe('validateChooseLineReward', () => {
    /**
     * These tests anchor the structural validation for line rewards:
     *
     * - RulesMatrix L1 – exact-length lines (forced full collapse).
     * - RulesMatrix L2 – overlength lines with graduated rewards (minimum collapse windows).
     *
     * Full behavioural flows for these scenarios live in:
     * - tests/unit/GameEngine.lines.scenarios.test.ts
     * - tests/scenarios/RulesMatrix.GameEngine.test.ts
     */
    const fourPositions = [
      { x: 0, y: 0 },
      { x: 1, y: 0 },
      { x: 2, y: 0 },
      { x: 3, y: 0 },
    ];

    const fivePositions = [
      { x: 0, y: 0 },
      { x: 1, y: 0 },
      { x: 2, y: 0 },
      { x: 3, y: 0 },
      { x: 4, y: 0 },
    ];

    const threePositions = [
      { x: 0, y: 0 },
      { x: 1, y: 0 },
      { x: 2, y: 0 },
    ];

    it('returns error when not in line_processing phase', () => {
      const state = createMinimalState({
        currentPhase: 'movement',
        formedLines: [{ player: 1, length: 5, positions: fivePositions }],
      });
      const action: ChooseLineRewardAction = {
        type: 'chooseLineReward',
        lineIndex: 0,
        playerId: 1,
        selection: 'FULL_COLLAPSE',
      };

      const result = validateChooseLineReward(state, action);

      expect(result.valid).toBe(false);
      expect(result.code).toBe('INVALID_PHASE');
    });

    it('returns error when not player turn', () => {
      const state = createMinimalState({
        currentPhase: 'line_processing',
        currentPlayer: 2,
        formedLines: [{ player: 1, length: 5, positions: fivePositions }],
      });
      const action: ChooseLineRewardAction = {
        type: 'chooseLineReward',
        lineIndex: 0,
        playerId: 1,
        selection: 'FULL_COLLAPSE',
      };

      const result = validateChooseLineReward(state, action);

      expect(result.valid).toBe(false);
      expect(result.code).toBe('NOT_YOUR_TURN');
    });

    it('returns error when line index is invalid', () => {
      const state = createMinimalState({
        currentPhase: 'line_processing',
        currentPlayer: 1,
        formedLines: [{ player: 1, length: 5, positions: fivePositions }],
      });
      const action: ChooseLineRewardAction = {
        type: 'chooseLineReward',
        lineIndex: 5,
        playerId: 1,
        selection: 'FULL_COLLAPSE',
      };

      const result = validateChooseLineReward(state, action);

      expect(result.valid).toBe(false);
      expect(result.code).toBe('INVALID_LINE_INDEX');
    });

    it('returns error when line belongs to opponent', () => {
      const state = createMinimalState({
        currentPhase: 'line_processing',
        currentPlayer: 1,
        formedLines: [{ player: 2, length: 5, positions: fivePositions }],
      });
      const action: ChooseLineRewardAction = {
        type: 'chooseLineReward',
        lineIndex: 0,
        playerId: 1,
        selection: 'FULL_COLLAPSE',
      };

      const result = validateChooseLineReward(state, action);

      expect(result.valid).toBe(false);
      expect(result.code).toBe('NOT_YOUR_LINE');
    });

    it('returns error when MINIMUM_COLLAPSE on exact-length line (3 in 3p)', () => {
      // Per RR-CANON-R120: square8 3-4p threshold is 3.
      // A 3-position line is exact-length for 3p, so MINIMUM_COLLAPSE is invalid.
      const state = createMinimalState({
        currentPhase: 'line_processing',
        currentPlayer: 1,
        formedLines: [{ player: 1, length: 3, positions: threePositions }],
        playerCount: 3, // Use 3 players so threshold is 3, making this exact-length
      });
      const action: ChooseLineRewardAction = {
        type: 'chooseLineReward',
        lineIndex: 0,
        playerId: 1,
        selection: 'MINIMUM_COLLAPSE',
        collapsedPositions: threePositions,
      };

      const result = validateChooseLineReward(state, action);

      expect(result.valid).toBe(false);
      expect(result.code).toBe('INVALID_SELECTION');
    });

    it('uses base threshold for 3p square8 (exact-length 3 forces full collapse, overlength 4 allows minimum segment)', () => {
      // RulesMatrix L1/L2: for 3‑player square8, the effective
      // requiredLength is the base BOARD_CONFIGS.square8.lineLength (3).
      // - Exact-length 3-line → MINIMUM_COLLAPSE is invalid.
      // - Overlength 4-line → MINIMUM_COLLAPSE on any consecutive 3-window is valid.

      // Exact-length 3 in 3p square8: MINIMUM_COLLAPSE is rejected.
      const exactState = createMinimalState({
        currentPhase: 'line_processing',
        currentPlayer: 1,
        formedLines: [{ player: 1, length: 3, positions: threePositions }],
        playerCount: 3,
      });
      const exactAction: ChooseLineRewardAction = {
        type: 'chooseLineReward',
        lineIndex: 0,
        playerId: 1,
        selection: 'MINIMUM_COLLAPSE',
        collapsedPositions: threePositions,
      };

      const exactResult = validateChooseLineReward(exactState, exactAction);

      expect(exactResult.valid).toBe(false);
      expect(exactResult.code).toBe('INVALID_SELECTION');

      // Overlength 4 in 3p square8: MINIMUM_COLLAPSE on any consecutive
      // window of 3 is accepted.
      const overlengthState = createMinimalState({
        currentPhase: 'line_processing',
        currentPlayer: 1,
        formedLines: [{ player: 1, length: 4, positions: fourPositions }],
        playerCount: 3,
      });
      const overlengthAction: ChooseLineRewardAction = {
        type: 'chooseLineReward',
        lineIndex: 0,
        playerId: 1,
        selection: 'MINIMUM_COLLAPSE',
        collapsedPositions: threePositions,
      };

      const overlengthResult = validateChooseLineReward(overlengthState, overlengthAction);

      expect(overlengthResult.valid).toBe(true);
    });

    it('respects square19 line-length threshold for exact vs overlength lines (L1/L2)', () => {
      // For square19, getEffectiveLineLengthThreshold always returns
      // BOARD_CONFIGS.square19.lineLength (4) regardless of player count.
      // This test ensures LineValidator honours that threshold when
      // validating MINIMUM_COLLAPSE vs FULL_COLLAPSE.

      // Exact-length 4 on square19: MINIMUM_COLLAPSE is invalid, FULL_COLLAPSE allowed.
      const exactState = createMinimalState({
        currentPhase: 'line_processing',
        currentPlayer: 1,
        boardType: 'square19',
        boardSize: 19,
        formedLines: [{ player: 1, length: 4, positions: fourPositions }],
        playerCount: 2,
      });
      const exactMinAction: ChooseLineRewardAction = {
        type: 'chooseLineReward',
        lineIndex: 0,
        playerId: 1,
        selection: 'MINIMUM_COLLAPSE',
        collapsedPositions: fourPositions,
      };
      const exactFullAction: ChooseLineRewardAction = {
        type: 'chooseLineReward',
        lineIndex: 0,
        playerId: 1,
        selection: 'FULL_COLLAPSE',
      };

      const exactMinResult = validateChooseLineReward(exactState, exactMinAction);
      const exactFullResult = validateChooseLineReward(exactState, exactFullAction);

      expect(exactMinResult.valid).toBe(false);
      expect(exactMinResult.code).toBe('INVALID_SELECTION');
      expect(exactFullResult.valid).toBe(true);

      // Overlength 5 on square19: MINIMUM_COLLAPSE on any consecutive
      // 4-window is accepted, mirroring the 2p square8 overlength case.
      const overlengthState = createMinimalState({
        currentPhase: 'line_processing',
        currentPlayer: 1,
        boardType: 'square19',
        boardSize: 19,
        formedLines: [{ player: 1, length: 5, positions: fivePositions }],
        playerCount: 2,
      });
      const overlengthAction: ChooseLineRewardAction = {
        type: 'chooseLineReward',
        lineIndex: 0,
        playerId: 1,
        selection: 'MINIMUM_COLLAPSE',
        collapsedPositions: fourPositions,
      };

      const overlengthResult = validateChooseLineReward(overlengthState, overlengthAction);

      expect(overlengthResult.valid).toBe(true);
    });

    it('returns error when MINIMUM_COLLAPSE without collapsedPositions', () => {
      const state = createMinimalState({
        currentPhase: 'line_processing',
        currentPlayer: 1,
        formedLines: [{ player: 1, length: 5, positions: fivePositions }],
        playerCount: 2,
      });
      const action: ChooseLineRewardAction = {
        type: 'chooseLineReward',
        lineIndex: 0,
        playerId: 1,
        selection: 'MINIMUM_COLLAPSE',
        // Missing collapsedPositions
      };

      const result = validateChooseLineReward(state, action);

      expect(result.valid).toBe(false);
      expect(result.code).toBe('MISSING_POSITIONS');
    });

    it('returns error when MINIMUM_COLLAPSE with wrong position count', () => {
      // For square8, threshold is 3. MINIMUM_COLLAPSE requires exactly 3 positions.
      const state = createMinimalState({
        currentPhase: 'line_processing',
        currentPlayer: 1,
        formedLines: [{ player: 1, length: 5, positions: fivePositions }],
        playerCount: 2,
      });
      const action: ChooseLineRewardAction = {
        type: 'chooseLineReward',
        lineIndex: 0,
        playerId: 1,
        selection: 'MINIMUM_COLLAPSE',
        collapsedPositions: [
          { x: 0, y: 0 },
          { x: 1, y: 0 },
        ], // Only 2, need 3
      };

      const result = validateChooseLineReward(state, action);

      expect(result.valid).toBe(false);
      expect(result.code).toBe('INVALID_POSITION_COUNT');
    });

    it('returns error when MINIMUM_COLLAPSE with position not in line', () => {
      // Per RR-CANON-R120: square8 3-4p threshold is 3.
      // MINIMUM_COLLAPSE requires exactly threshold positions, all in the line.
      const state = createMinimalState({
        currentPhase: 'line_processing',
        currentPlayer: 1,
        formedLines: [{ player: 1, length: 5, positions: fivePositions }],
        playerCount: 3, // Use 3 players so threshold is 3
      });
      const action: ChooseLineRewardAction = {
        type: 'chooseLineReward',
        lineIndex: 0,
        playerId: 1,
        selection: 'MINIMUM_COLLAPSE',
        collapsedPositions: [
          { x: 0, y: 0 },
          { x: 1, y: 0 },
          { x: 5, y: 5 }, // Not in the line
        ],
      };

      const result = validateChooseLineReward(state, action);

      expect(result.valid).toBe(false);
      expect(result.code).toBe('INVALID_POSITION');
    });

    it('returns error when MINIMUM_COLLAPSE with non-consecutive positions', () => {
      // Per RR-CANON-R120: square8 3-4p threshold is 3.
      // MINIMUM_COLLAPSE requires exactly threshold CONSECUTIVE positions.
      const state = createMinimalState({
        currentPhase: 'line_processing',
        currentPlayer: 1,
        formedLines: [{ player: 1, length: 5, positions: fivePositions }],
        playerCount: 3, // Use 3 players so threshold is 3
      });
      const action: ChooseLineRewardAction = {
        type: 'chooseLineReward',
        lineIndex: 0,
        playerId: 1,
        selection: 'MINIMUM_COLLAPSE',
        collapsedPositions: [
          { x: 0, y: 0 },
          { x: 1, y: 0 },
          // Skip x: 2, y: 0
          { x: 3, y: 0 },
        ],
      };

      const result = validateChooseLineReward(state, action);

      expect(result.valid).toBe(false);
      expect(result.code).toBe('NON_CONSECUTIVE');
    });

    it('allows FULL_COLLAPSE on longer line', () => {
      const state = createMinimalState({
        currentPhase: 'line_processing',
        currentPlayer: 1,
        formedLines: [{ player: 1, length: 5, positions: fivePositions }],
        playerCount: 2,
      });
      const action: ChooseLineRewardAction = {
        type: 'chooseLineReward',
        lineIndex: 0,
        playerId: 1,
        selection: 'FULL_COLLAPSE',
      };

      const result = validateChooseLineReward(state, action);

      expect(result.valid).toBe(true);
    });

    it('allows valid MINIMUM_COLLAPSE with consecutive positions', () => {
      // Per RR-CANON-R120: square8 3-4p threshold is 3.
      // MINIMUM_COLLAPSE requires exactly threshold consecutive positions.
      const state = createMinimalState({
        currentPhase: 'line_processing',
        currentPlayer: 1,
        formedLines: [{ player: 1, length: 5, positions: fivePositions }],
        playerCount: 3, // Use 3 players so threshold is 3
      });
      const action: ChooseLineRewardAction = {
        type: 'chooseLineReward',
        lineIndex: 0,
        playerId: 1,
        selection: 'MINIMUM_COLLAPSE',
        collapsedPositions: [
          { x: 1, y: 0 },
          { x: 2, y: 0 },
          { x: 3, y: 0 },
        ],
      };

      const result = validateChooseLineReward(state, action);

      expect(result.valid).toBe(true);
    });

    it('allows MINIMUM_COLLAPSE starting from beginning of line', () => {
      // Per RR-CANON-R120: square8 3-4p threshold is 3.
      // MINIMUM_COLLAPSE requires exactly threshold consecutive positions.
      const state = createMinimalState({
        currentPhase: 'line_processing',
        currentPlayer: 1,
        formedLines: [{ player: 1, length: 5, positions: fivePositions }],
        playerCount: 3, // Use 3 players so threshold is 3
      });
      const action: ChooseLineRewardAction = {
        type: 'chooseLineReward',
        lineIndex: 0,
        playerId: 1,
        selection: 'MINIMUM_COLLAPSE',
        collapsedPositions: [
          { x: 0, y: 0 },
          { x: 1, y: 0 },
          { x: 2, y: 0 },
        ],
      };

      const result = validateChooseLineReward(state, action);

      expect(result.valid).toBe(true);
    });

    it('handles exact-length lines with FULL_COLLAPSE (forced)', () => {
      const state = createMinimalState({
        currentPhase: 'line_processing',
        currentPlayer: 1,
        formedLines: [{ player: 1, length: 4, positions: fourPositions }],
        playerCount: 2,
      });
      const action: ChooseLineRewardAction = {
        type: 'chooseLineReward',
        lineIndex: 0,
        playerId: 1,
        selection: 'FULL_COLLAPSE',
      };

      const result = validateChooseLineReward(state, action);

      expect(result.valid).toBe(true);
    });

    it('handles multiple lines - select correct one', () => {
      const state = createMinimalState({
        currentPhase: 'line_processing',
        currentPlayer: 1,
        formedLines: [
          { player: 2, length: 4, positions: fourPositions },
          { player: 1, length: 5, positions: fivePositions },
        ],
      });
      const action: ChooseLineRewardAction = {
        type: 'chooseLineReward',
        lineIndex: 1,
        playerId: 1,
        selection: 'FULL_COLLAPSE',
      };

      const result = validateChooseLineReward(state, action);

      expect(result.valid).toBe(true);
    });
  });
});
