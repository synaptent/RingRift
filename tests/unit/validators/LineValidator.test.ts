/**
 * LineValidator Unit Tests
 *
 * Tests for line processing validation including:
 * - validateProcessLine: validates process line actions
 * - validateChooseLineReward: validates line reward selection
 */

import {
  validateProcessLine,
  validateChooseLineReward,
} from '../../../src/shared/engine/validators/LineValidator';
import type {
  GameState,
  ProcessLineAction,
  ChooseLineRewardAction,
} from '../../../src/shared/engine/types';
import type { FormedLine, Coord, GamePhase } from '../../../src/shared/types/game';

describe('LineValidator', () => {
  const createFormedLine = (overrides: Partial<FormedLine> = {}): FormedLine => ({
    player: 1,
    length: 5,
    positions: [
      { x: 0, y: 0 },
      { x: 1, y: 0 },
      { x: 2, y: 0 },
      { x: 3, y: 0 },
      { x: 4, y: 0 },
    ],
    direction: { x: 1, y: 0 },
    ...overrides,
  });

  const createMockState = (overrides: Partial<GameState> = {}): GameState =>
    ({
      currentPhase: 'line_processing' as GamePhase,
      currentPlayer: 1,
      board: {
        type: 'square8',
        size: 8,
        stacks: new Map(),
        markers: new Map(),
        collapsedSpaces: new Map(),
        territories: new Map(),
        formedLines: [createFormedLine()],
        eliminatedRings: {},
      },
      players: [{ playerNumber: 1 }, { playerNumber: 2 }],
      ...overrides,
    }) as unknown as GameState;

  describe('validateProcessLine', () => {
    it('should return valid for correct process line action', () => {
      const state = createMockState();
      const action: ProcessLineAction = {
        type: 'process_line',
        playerId: 1,
        lineIndex: 0,
      };

      const result = validateProcessLine(state, action);

      expect(result.valid).toBe(true);
    });

    it('should reject when not in line_processing phase', () => {
      const state = createMockState({ currentPhase: 'movement' as GamePhase });
      const action: ProcessLineAction = {
        type: 'process_line',
        playerId: 1,
        lineIndex: 0,
      };

      const result = validateProcessLine(state, action);

      expect(result.valid).toBe(false);
      expect(result.code).toBe('INVALID_PHASE');
      expect(result.reason).toBe('Not in line processing phase');
    });

    it('should reject when not players turn', () => {
      const state = createMockState({ currentPlayer: 2 });
      const action: ProcessLineAction = {
        type: 'process_line',
        playerId: 1,
        lineIndex: 0,
      };

      const result = validateProcessLine(state, action);

      expect(result.valid).toBe(false);
      expect(result.code).toBe('NOT_YOUR_TURN');
      expect(result.reason).toBe('Not your turn');
    });

    it('should reject negative line index', () => {
      const state = createMockState();
      const action: ProcessLineAction = {
        type: 'process_line',
        playerId: 1,
        lineIndex: -1,
      };

      const result = validateProcessLine(state, action);

      expect(result.valid).toBe(false);
      expect(result.code).toBe('INVALID_LINE_INDEX');
      expect(result.reason).toBe('Invalid line index');
    });

    it('should reject out of bounds line index', () => {
      const state = createMockState();
      const action: ProcessLineAction = {
        type: 'process_line',
        playerId: 1,
        lineIndex: 99,
      };

      const result = validateProcessLine(state, action);

      expect(result.valid).toBe(false);
      expect(result.code).toBe('INVALID_LINE_INDEX');
    });

    it('should reject processing opponent line', () => {
      const state = createMockState();
      state.board.formedLines = [createFormedLine({ player: 2 })];
      const action: ProcessLineAction = {
        type: 'process_line',
        playerId: 1,
        lineIndex: 0,
      };

      const result = validateProcessLine(state, action);

      expect(result.valid).toBe(false);
      expect(result.code).toBe('NOT_YOUR_LINE');
      expect(result.reason).toBe('Cannot process opponent line');
    });

    it('should handle multiple lines', () => {
      const state = createMockState();
      state.board.formedLines = [createFormedLine({ player: 2 }), createFormedLine({ player: 1 })];
      const action: ProcessLineAction = {
        type: 'process_line',
        playerId: 1,
        lineIndex: 1,
      };

      const result = validateProcessLine(state, action);

      expect(result.valid).toBe(true);
    });
  });

  describe('validateChooseLineReward', () => {
    it('should return valid for FULL_COLLAPSE selection', () => {
      const state = createMockState();
      const action: ChooseLineRewardAction = {
        type: 'choose_line_reward',
        playerId: 1,
        lineIndex: 0,
        selection: 'FULL_COLLAPSE',
      };

      const result = validateChooseLineReward(state, action);

      expect(result.valid).toBe(true);
    });

    it('should reject when not in line_processing phase', () => {
      const state = createMockState({ currentPhase: 'movement' as GamePhase });
      const action: ChooseLineRewardAction = {
        type: 'choose_line_reward',
        playerId: 1,
        lineIndex: 0,
        selection: 'FULL_COLLAPSE',
      };

      const result = validateChooseLineReward(state, action);

      expect(result.valid).toBe(false);
      expect(result.code).toBe('INVALID_PHASE');
    });

    it('should reject when not players turn', () => {
      const state = createMockState({ currentPlayer: 2 });
      const action: ChooseLineRewardAction = {
        type: 'choose_line_reward',
        playerId: 1,
        lineIndex: 0,
        selection: 'FULL_COLLAPSE',
      };

      const result = validateChooseLineReward(state, action);

      expect(result.valid).toBe(false);
      expect(result.code).toBe('NOT_YOUR_TURN');
    });

    it('should reject negative line index', () => {
      const state = createMockState();
      const action: ChooseLineRewardAction = {
        type: 'choose_line_reward',
        playerId: 1,
        lineIndex: -1,
        selection: 'FULL_COLLAPSE',
      };

      const result = validateChooseLineReward(state, action);

      expect(result.valid).toBe(false);
      expect(result.code).toBe('INVALID_LINE_INDEX');
    });

    it('should reject out of bounds line index', () => {
      const state = createMockState();
      const action: ChooseLineRewardAction = {
        type: 'choose_line_reward',
        playerId: 1,
        lineIndex: 10,
        selection: 'FULL_COLLAPSE',
      };

      const result = validateChooseLineReward(state, action);

      expect(result.valid).toBe(false);
      expect(result.code).toBe('INVALID_LINE_INDEX');
    });

    it('should reject processing opponent line', () => {
      const state = createMockState();
      state.board.formedLines = [createFormedLine({ player: 2 })];
      const action: ChooseLineRewardAction = {
        type: 'choose_line_reward',
        playerId: 1,
        lineIndex: 0,
        selection: 'FULL_COLLAPSE',
      };

      const result = validateChooseLineReward(state, action);

      expect(result.valid).toBe(false);
      expect(result.code).toBe('NOT_YOUR_LINE');
    });

    it('should reject MINIMUM_COLLAPSE for exact length line (3-in-a-row)', () => {
      // Per RR-CANON-R120: square8 3-4p has line length 3, 2p has line length 4.
      // Use 3 players to test exact-length rejection with 3-position lines.
      const state = createMockState({
        players: [{ playerNumber: 1 }, { playerNumber: 2 }, { playerNumber: 3 }],
      });
      // Create a line of exactly 3 (the threshold for 3-4 player square8)
      state.board.formedLines = [
        createFormedLine({
          player: 1,
          length: 3,
          positions: [
            { x: 0, y: 0 },
            { x: 1, y: 0 },
            { x: 2, y: 0 },
          ],
        }),
      ];
      const action: ChooseLineRewardAction = {
        type: 'choose_line_reward',
        playerId: 1,
        lineIndex: 0,
        selection: 'MINIMUM_COLLAPSE',
        collapsedPositions: [
          { x: 0, y: 0 },
          { x: 1, y: 0 },
          { x: 2, y: 0 },
        ],
      };

      const result = validateChooseLineReward(state, action);

      expect(result.valid).toBe(false);
      expect(result.code).toBe('INVALID_SELECTION');
      expect(result.reason).toBe('Cannot choose minimum collapse for exact length line');
    });

    it('should reject MINIMUM_COLLAPSE without collapsed positions', () => {
      const state = createMockState();
      const action: ChooseLineRewardAction = {
        type: 'choose_line_reward',
        playerId: 1,
        lineIndex: 0,
        selection: 'MINIMUM_COLLAPSE',
        // No collapsedPositions provided
      };

      const result = validateChooseLineReward(state, action);

      expect(result.valid).toBe(false);
      expect(result.code).toBe('MISSING_POSITIONS');
      expect(result.reason).toBe('Must provide collapsed positions for minimum collapse');
    });

    it('should reject MINIMUM_COLLAPSE with wrong number of positions', () => {
      const state = createMockState();
      const action: ChooseLineRewardAction = {
        type: 'choose_line_reward',
        playerId: 1,
        lineIndex: 0,
        selection: 'MINIMUM_COLLAPSE',
        collapsedPositions: [
          { x: 0, y: 0 },
          { x: 1, y: 0 },
        ], // Only 2 positions, need 3 (square8 threshold)
      };

      const result = validateChooseLineReward(state, action);

      expect(result.valid).toBe(false);
      expect(result.code).toBe('INVALID_POSITION_COUNT');
    });

    it('should reject MINIMUM_COLLAPSE with position not in line', () => {
      // Per RR-CANON-R120: square8 3-4p threshold is 3.
      // Use 3 players to test with 3-position MINIMUM_COLLAPSE.
      const state = createMockState({
        players: [{ playerNumber: 1 }, { playerNumber: 2 }, { playerNumber: 3 }],
      });
      const action: ChooseLineRewardAction = {
        type: 'choose_line_reward',
        playerId: 1,
        lineIndex: 0,
        selection: 'MINIMUM_COLLAPSE',
        collapsedPositions: [
          { x: 0, y: 0 },
          { x: 1, y: 0 },
          { x: 99, y: 99 }, // Not part of the line (need 3 positions for 3-4p square8)
        ],
      };

      const result = validateChooseLineReward(state, action);

      expect(result.valid).toBe(false);
      expect(result.code).toBe('INVALID_POSITION');
      expect(result.reason).toBe('Selected position is not part of the line');
    });

    it('should reject MINIMUM_COLLAPSE with non-consecutive positions', () => {
      // Per RR-CANON-R120: square8 3-4p threshold is 3.
      // Use 3 players to test with 3-position MINIMUM_COLLAPSE.
      const state = createMockState({
        players: [{ playerNumber: 1 }, { playerNumber: 2 }, { playerNumber: 3 }],
      });
      const action: ChooseLineRewardAction = {
        type: 'choose_line_reward',
        playerId: 1,
        lineIndex: 0,
        selection: 'MINIMUM_COLLAPSE',
        collapsedPositions: [
          { x: 0, y: 0 },
          // Skip { x: 1, y: 0 }
          { x: 2, y: 0 },
          { x: 3, y: 0 },
        ], // 3 positions for square8 3-4p threshold, but non-consecutive
      };

      const result = validateChooseLineReward(state, action);

      expect(result.valid).toBe(false);
      expect(result.code).toBe('NON_CONSECUTIVE');
      expect(result.reason).toBe('Selected positions must be consecutive');
    });

    it('should accept valid MINIMUM_COLLAPSE with consecutive positions', () => {
      // Per RR-CANON-R120: square8 3-4p threshold is 3.
      // Use 3 players to test MINIMUM_COLLAPSE with 3 positions.
      const state = createMockState({
        players: [{ playerNumber: 1 }, { playerNumber: 2 }, { playerNumber: 3 }],
      });
      const action: ChooseLineRewardAction = {
        type: 'choose_line_reward',
        playerId: 1,
        lineIndex: 0,
        selection: 'MINIMUM_COLLAPSE',
        collapsedPositions: [
          { x: 1, y: 0 },
          { x: 2, y: 0 },
          { x: 3, y: 0 },
        ], // 3 consecutive positions for square8 3-4p threshold
      };

      const result = validateChooseLineReward(state, action);

      expect(result.valid).toBe(true);
    });

    it('should accept MINIMUM_COLLAPSE at start of line', () => {
      // Per RR-CANON-R120: square8 3-4p threshold is 3.
      // Use 3 players to test MINIMUM_COLLAPSE with 3 positions.
      const state = createMockState({
        players: [{ playerNumber: 1 }, { playerNumber: 2 }, { playerNumber: 3 }],
      });
      const action: ChooseLineRewardAction = {
        type: 'choose_line_reward',
        playerId: 1,
        lineIndex: 0,
        selection: 'MINIMUM_COLLAPSE',
        collapsedPositions: [
          { x: 0, y: 0 },
          { x: 1, y: 0 },
          { x: 2, y: 0 },
        ], // 3 consecutive positions at start for square8 3-4p threshold
      };

      const result = validateChooseLineReward(state, action);

      expect(result.valid).toBe(true);
    });

    it('should handle positions in different order (sorts before checking)', () => {
      // Per RR-CANON-R120: square8 3-4p threshold is 3.
      // Use 3 players to test MINIMUM_COLLAPSE with 3 positions.
      const state = createMockState({
        players: [{ playerNumber: 1 }, { playerNumber: 2 }, { playerNumber: 3 }],
      });
      const action: ChooseLineRewardAction = {
        type: 'choose_line_reward',
        playerId: 1,
        lineIndex: 0,
        selection: 'MINIMUM_COLLAPSE',
        // Provide 3 positions in reverse order - should still work (square8 3-4p threshold)
        collapsedPositions: [
          { x: 3, y: 0 },
          { x: 1, y: 0 },
          { x: 2, y: 0 },
        ],
      };

      const result = validateChooseLineReward(state, action);

      expect(result.valid).toBe(true);
    });

    it('should handle 3-player game with different line threshold', () => {
      const state = createMockState({
        players: [{ playerNumber: 1 }, { playerNumber: 2 }, { playerNumber: 3 }],
      });
      // For 3+ players on square8, threshold is 3 (base lineLength)
      state.board.formedLines = [
        createFormedLine({
          player: 1,
          length: 5,
          positions: [
            { x: 0, y: 0 },
            { x: 1, y: 0 },
            { x: 2, y: 0 },
            { x: 3, y: 0 },
            { x: 4, y: 0 },
          ],
        }),
      ];

      const action: ChooseLineRewardAction = {
        type: 'choose_line_reward',
        playerId: 1,
        lineIndex: 0,
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
  });
});
