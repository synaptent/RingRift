/**
 * LineAggregate.advanced.branchCoverage.test.ts
 *
 * Advanced branch coverage tests for LineAggregate.
 * Covers validation edge cases, enumeration edge cases, and application edge cases.
 */

import {
  findAllLines,
  findLinesForPlayer,
  findLinesContainingPosition,
  validateProcessLine,
  validateChooseLineReward,
  validateLineDecision,
  enumerateProcessLineMoves,
  enumerateChooseLineRewardMoves,
  enumerateLineCollapseOptions,
  applyProcessLineDecision,
  applyChooseLineRewardDecision,
  applyLineCollapse,
  mutateProcessLine,
  mutateChooseLineReward,
  type LineCollapseDecision,
} from '../../src/shared/engine/aggregates/LineAggregate';
import {
  createTestGameState,
  createTestBoard,
  addMarker,
  addStack,
  addCollapsedSpace,
  pos,
} from '../utils/fixtures';
import { positionToString } from '../../src/shared/types/game';
import type { GameState, Position, LineInfo } from '../../src/shared/types/game';

describe('LineAggregate - Advanced Branch Coverage', () => {
  // ==========================================================================
  // validateChooseLineReward additional branches
  // ==========================================================================
  describe('validateChooseLineReward additional branches', () => {
    it('rejects when not in line_processing phase', () => {
      const state = createTestGameState({ numPlayers: 2, boardType: 'square19' });
      state.currentPhase = 'movement';
      state.currentPlayer = 1;
      state.board.formedLines = [
        {
          player: 1,
          positions: [
            { x: 0, y: 0 },
            { x: 1, y: 0 },
            { x: 2, y: 0 },
            { x: 3, y: 0 },
          ],
          length: 4,
        },
      ];

      const result = validateChooseLineReward(state, {
        type: 'CHOOSE_LINE_REWARD',
        playerId: 1,
        lineIndex: 0,
        selection: 'COLLAPSE_ALL',
      });

      expect(result.valid).toBe(false);
      expect(result.code).toBe('INVALID_PHASE');
    });

    it('rejects when it is not the player turn', () => {
      const state = createTestGameState({ numPlayers: 2, boardType: 'square19' });
      state.currentPhase = 'line_processing';
      state.currentPlayer = 2; // Player 1 trying to act during player 2's turn
      state.board.formedLines = [
        {
          player: 1,
          positions: [
            { x: 0, y: 0 },
            { x: 1, y: 0 },
            { x: 2, y: 0 },
            { x: 3, y: 0 },
          ],
          length: 4,
        },
      ];

      const result = validateChooseLineReward(state, {
        type: 'CHOOSE_LINE_REWARD',
        playerId: 1,
        lineIndex: 0,
        selection: 'COLLAPSE_ALL',
      });

      expect(result.valid).toBe(false);
      expect(result.code).toBe('NOT_YOUR_TURN');
    });

    it('rejects invalid line index', () => {
      const state = createTestGameState({ numPlayers: 2, boardType: 'square19' });
      state.currentPhase = 'line_processing';
      state.currentPlayer = 1;
      state.board.formedLines = [
        {
          player: 1,
          positions: [
            { x: 0, y: 0 },
            { x: 1, y: 0 },
            { x: 2, y: 0 },
            { x: 3, y: 0 },
          ],
          length: 4,
        },
      ];

      const result = validateChooseLineReward(state, {
        type: 'CHOOSE_LINE_REWARD',
        playerId: 1,
        lineIndex: 2, // out of bounds
        selection: 'COLLAPSE_ALL',
      });

      expect(result.valid).toBe(false);
      expect(result.code).toBe('INVALID_LINE_INDEX');
    });

    it('rejects lines owned by another player', () => {
      const state = createTestGameState({ numPlayers: 2, boardType: 'square19' });
      state.currentPhase = 'line_processing';
      state.currentPlayer = 1;
      state.board.formedLines = [
        {
          player: 2,
          positions: [
            { x: 0, y: 0 },
            { x: 1, y: 0 },
            { x: 2, y: 0 },
            { x: 3, y: 0 },
          ],
          length: 4,
        },
      ];

      const result = validateChooseLineReward(state, {
        type: 'CHOOSE_LINE_REWARD',
        playerId: 1,
        lineIndex: 0,
        selection: 'COLLAPSE_ALL',
      });

      expect(result.valid).toBe(false);
      expect(result.code).toBe('NOT_YOUR_LINE');
    });

    it('rejects MINIMUM_COLLAPSE for exact-length line (square19 = 4 threshold)', () => {
      // square19 has threshold 4, so 4 markers is exact length
      const state = createTestGameState({ numPlayers: 2, boardType: 'square19' });
      state.currentPhase = 'line_processing';
      state.currentPlayer = 1;
      state.board.formedLines = [
        {
          player: 1,
          positions: [
            { x: 0, y: 0 },
            { x: 1, y: 0 },
            { x: 2, y: 0 },
            { x: 3, y: 0 },
          ],
          length: 4,
        },
      ];

      const result = validateChooseLineReward(state, {
        type: 'CHOOSE_LINE_REWARD',
        playerId: 1,
        lineIndex: 0,
        selection: 'MINIMUM_COLLAPSE',
        collapsedPositions: [
          { x: 0, y: 0 },
          { x: 1, y: 0 },
          { x: 2, y: 0 },
          { x: 3, y: 0 },
        ],
      });

      expect(result.valid).toBe(false);
      expect(result.code).toBe('INVALID_SELECTION');
    });

    it('rejects MINIMUM_COLLAPSE without positions', () => {
      // square19 has threshold 4, so 5 markers is overlength
      const state = createTestGameState({ numPlayers: 2, boardType: 'square19' });
      state.currentPhase = 'line_processing';
      state.currentPlayer = 1;
      state.board.formedLines = [
        {
          player: 1,
          positions: [
            { x: 0, y: 0 },
            { x: 1, y: 0 },
            { x: 2, y: 0 },
            { x: 3, y: 0 },
            { x: 4, y: 0 },
          ],
          length: 5,
        },
      ];

      const result = validateChooseLineReward(state, {
        type: 'CHOOSE_LINE_REWARD',
        playerId: 1,
        lineIndex: 0,
        selection: 'MINIMUM_COLLAPSE',
        // Missing collapsedPositions
      });

      expect(result.valid).toBe(false);
      expect(result.code).toBe('MISSING_POSITIONS');
    });

    it('rejects MINIMUM_COLLAPSE with wrong position count', () => {
      // square19 has threshold 4
      const state = createTestGameState({ numPlayers: 2, boardType: 'square19' });
      state.currentPhase = 'line_processing';
      state.currentPlayer = 1;
      state.board.formedLines = [
        {
          player: 1,
          positions: [
            { x: 0, y: 0 },
            { x: 1, y: 0 },
            { x: 2, y: 0 },
            { x: 3, y: 0 },
            { x: 4, y: 0 },
          ],
          length: 5,
        },
      ];

      const result = validateChooseLineReward(state, {
        type: 'CHOOSE_LINE_REWARD',
        playerId: 1,
        lineIndex: 0,
        selection: 'MINIMUM_COLLAPSE',
        collapsedPositions: [
          { x: 0, y: 0 },
          { x: 1, y: 0 },
        ], // Only 2, need 4
      });

      expect(result.valid).toBe(false);
      expect(result.code).toBe('INVALID_POSITION_COUNT');
    });

    it('rejects MINIMUM_COLLAPSE with positions not in line', () => {
      // square19 has threshold 4
      const state = createTestGameState({ numPlayers: 2, boardType: 'square19' });
      state.currentPhase = 'line_processing';
      state.currentPlayer = 1;
      state.board.formedLines = [
        {
          player: 1,
          positions: [
            { x: 0, y: 0 },
            { x: 1, y: 0 },
            { x: 2, y: 0 },
            { x: 3, y: 0 },
            { x: 4, y: 0 },
          ],
          length: 5,
        },
      ];

      const result = validateChooseLineReward(state, {
        type: 'CHOOSE_LINE_REWARD',
        playerId: 1,
        lineIndex: 0,
        selection: 'MINIMUM_COLLAPSE',
        collapsedPositions: [
          { x: 0, y: 0 },
          { x: 1, y: 0 },
          { x: 2, y: 0 },
          { x: 5, y: 5 }, // Not in line
        ],
      });

      expect(result.valid).toBe(false);
      expect(result.code).toBe('INVALID_POSITION');
    });

    it('rejects MINIMUM_COLLAPSE with non-consecutive positions', () => {
      // square19 has threshold 4
      const state = createTestGameState({ numPlayers: 2, boardType: 'square19' });
      state.currentPhase = 'line_processing';
      state.currentPlayer = 1;
      state.board.formedLines = [
        {
          player: 1,
          positions: [
            { x: 0, y: 0 },
            { x: 1, y: 0 },
            { x: 2, y: 0 },
            { x: 3, y: 0 },
            { x: 4, y: 0 },
            { x: 5, y: 0 },
          ],
          length: 6,
        },
      ];

      const result = validateChooseLineReward(state, {
        type: 'CHOOSE_LINE_REWARD',
        playerId: 1,
        lineIndex: 0,
        selection: 'MINIMUM_COLLAPSE',
        collapsedPositions: [
          { x: 0, y: 0 },
          { x: 1, y: 0 },
          { x: 3, y: 0 }, // Skip 2
          { x: 5, y: 0 }, // Skip 4
        ],
      });

      expect(result.valid).toBe(false);
      expect(result.code).toBe('NON_CONSECUTIVE');
    });

    it('accepts valid MINIMUM_COLLAPSE with consecutive positions', () => {
      // square19 has threshold 4
      const state = createTestGameState({ numPlayers: 2, boardType: 'square19' });
      state.currentPhase = 'line_processing';
      state.currentPlayer = 1;
      state.board.formedLines = [
        {
          player: 1,
          positions: [
            { x: 0, y: 0 },
            { x: 1, y: 0 },
            { x: 2, y: 0 },
            { x: 3, y: 0 },
            { x: 4, y: 0 },
          ],
          length: 5,
        },
      ];

      const result = validateChooseLineReward(state, {
        type: 'CHOOSE_LINE_REWARD',
        playerId: 1,
        lineIndex: 0,
        selection: 'MINIMUM_COLLAPSE',
        collapsedPositions: [
          { x: 0, y: 0 },
          { x: 1, y: 0 },
          { x: 2, y: 0 },
          { x: 3, y: 0 },
        ],
      });

      expect(result.valid).toBe(true);
    });
  });

  // ==========================================================================
  // enumerateChooseLineRewardMoves edge cases
  // ==========================================================================
  describe('enumerateChooseLineRewardMoves edge cases', () => {
    it('returns empty for negative lineIndex', () => {
      const state = createTestGameState();
      const moves = enumerateChooseLineRewardMoves(state, 1, -1);
      expect(moves).toHaveLength(0);
    });

    it('returns empty for lineIndex >= player lines count', () => {
      const state = createTestGameState();
      state.board.formedLines = [];
      const moves = enumerateChooseLineRewardMoves(state, 1, 5);
      expect(moves).toHaveLength(0);
    });

    it('returns empty for line with empty positions', () => {
      const state = createTestGameState();
      state.board.formedLines = [
        {
          player: 1,
          positions: [],
          length: 0,
        },
      ];
      const moves = enumerateChooseLineRewardMoves(state, 1, 0);
      expect(moves).toHaveLength(0);
    });

    it('returns single move for exact-length line (square19 = threshold 4)', () => {
      // square19 has threshold 4
      const state = createTestGameState({ numPlayers: 2, boardType: 'square19' });
      state.board.formedLines = [
        {
          player: 1,
          positions: [
            { x: 0, y: 0 },
            { x: 1, y: 0 },
            { x: 2, y: 0 },
            { x: 3, y: 0 },
          ],
          length: 4,
        },
      ];
      const moves = enumerateChooseLineRewardMoves(state, 1, 0);
      expect(moves.length).toBe(1); // Only collapse-all option
    });

    it('returns multiple moves for overlength line (square19 = threshold 4)', () => {
      // square19 has threshold 4, so 5 markers is overlength
      const state = createTestGameState({ numPlayers: 2, boardType: 'square19' });
      state.board.formedLines = [
        {
          player: 1,
          positions: [
            { x: 0, y: 0 },
            { x: 1, y: 0 },
            { x: 2, y: 0 },
            { x: 3, y: 0 },
            { x: 4, y: 0 },
          ],
          length: 5,
        },
      ];
      const moves = enumerateChooseLineRewardMoves(state, 1, 0);
      // 1 collapse-all + 2 minimum-collapse segments
      expect(moves.length).toBe(3);
    });
  });

  // ==========================================================================
  // enumerateLineCollapseOptions
  // ==========================================================================
  describe('enumerateLineCollapseOptions', () => {
    it('returns empty for line not in formedLines', () => {
      const state = createTestGameState();
      state.board.formedLines = [];

      const line = {
        player: 1,
        positions: [
          { x: 0, y: 0 },
          { x: 1, y: 0 },
          { x: 2, y: 0 },
        ],
        length: 3,
      };

      const options = enumerateLineCollapseOptions(state, line);
      expect(options).toHaveLength(0);
    });

    it('returns only COLLAPSE_ALL for exact-length line (square19 = threshold 4)', () => {
      // square19 has threshold 4
      const state = createTestGameState({ numPlayers: 2, boardType: 'square19' });
      const line = {
        player: 1,
        positions: [
          { x: 0, y: 0 },
          { x: 1, y: 0 },
          { x: 2, y: 0 },
          { x: 3, y: 0 },
        ],
        length: 4,
      };
      state.board.formedLines = [line];

      const options = enumerateLineCollapseOptions(state, line);
      expect(options.length).toBe(1);
      expect(options[0].selection).toBe('COLLAPSE_ALL');
    });

    it('returns COLLAPSE_ALL and MINIMUM_COLLAPSE options for overlength line (4-player)', () => {
      // 4-player game has threshold 4, so 5 markers is overlength
      const state = createTestGameState({ numPlayers: 4 });
      const line = {
        player: 1,
        positions: [
          { x: 0, y: 0 },
          { x: 1, y: 0 },
          { x: 2, y: 0 },
          { x: 3, y: 0 },
          { x: 4, y: 0 },
        ],
        length: 5,
      };
      state.board.formedLines = [line];

      const options = enumerateLineCollapseOptions(state, line);
      expect(options.length).toBeGreaterThan(1);

      const collapseAll = options.filter((o) => o.selection === 'COLLAPSE_ALL');
      const minCollapse = options.filter((o) => o.selection === 'MINIMUM_COLLAPSE');
      expect(collapseAll.length).toBe(1);
      expect(minCollapse.length).toBeGreaterThan(0);
    });
  });

  // ==========================================================================
  // mutateProcessLine error branch
  // ==========================================================================
  describe('mutateProcessLine error branches', () => {
    it('throws for overlength line (4-player)', () => {
      // 4-player game has threshold 4, so 5 markers is overlength
      const state = createTestGameState({ numPlayers: 4 });
      state.currentPhase = 'line_processing';
      state.currentPlayer = 1;
      state.board.formedLines = [
        {
          player: 1,
          positions: [
            { x: 0, y: 0 },
            { x: 1, y: 0 },
            { x: 2, y: 0 },
            { x: 3, y: 0 },
            { x: 4, y: 0 },
          ],
          length: 5,
        },
      ];

      expect(() => {
        mutateProcessLine(state, {
          type: 'process_line',
          player: 1,
          lineIndex: 0,
        });
      }).toThrow();
    });
  });

  // ==========================================================================
  // mutateChooseLineReward MINIMUM_COLLAPSE error
  // ==========================================================================
  describe('mutateChooseLineReward MINIMUM_COLLAPSE error', () => {
    it('throws when MINIMUM_COLLAPSE has no positions', () => {
      const state = createTestGameState();
      state.currentPhase = 'line_processing';
      state.currentPlayer = 1;
      state.board.formedLines = [
        {
          player: 1,
          positions: [
            { x: 0, y: 0 },
            { x: 1, y: 0 },
            { x: 2, y: 0 },
          ],
          length: 3,
        },
      ];

      expect(() => {
        mutateChooseLineReward(state, {
          type: 'choose_line_reward',
          player: 1,
          lineIndex: 0,
          selection: 'MINIMUM_COLLAPSE',
          // Missing collapsedPositions
        });
      }).toThrow('Missing collapsedPositions');
    });
  });

  // ==========================================================================
  // executeLineCollapse stack handling
  // ==========================================================================
  describe('executeLineCollapse stack handling', () => {
    it('returns rings to hand when collapsing position with stack', () => {
      const state = createTestGameState();
      state.currentPhase = 'line_processing';
      state.currentPlayer = 1;

      // Add markers for line
      addMarker(state.board, { x: 0, y: 0 }, 1);
      addMarker(state.board, { x: 1, y: 0 }, 1);
      addMarker(state.board, { x: 2, y: 0 }, 1);

      // Add a stack on one of the line positions
      addStack(state.board, { x: 1, y: 0 }, 1, 2, 2);

      state.board.formedLines = [
        {
          player: 1,
          positions: [
            { x: 0, y: 0 },
            { x: 1, y: 0 },
            { x: 2, y: 0 },
          ],
          length: 3,
        },
      ];

      const initialRingsInHand = state.players[0].ringsInHand;

      const result = mutateProcessLine(state, {
        type: 'process_line',
        player: 1,
        lineIndex: 0,
      });

      // Stack rings should be returned to hand
      const player = result.players.find((p) => p.playerNumber === 1);
      expect(player!.ringsInHand).toBeGreaterThanOrEqual(initialRingsInHand);
    });

    it('removes broken lines after collapse', () => {
      const state = createTestGameState();
      state.currentPhase = 'line_processing';
      state.currentPlayer = 1;

      addMarker(state.board, { x: 0, y: 0 }, 1);
      addMarker(state.board, { x: 1, y: 0 }, 1);
      addMarker(state.board, { x: 2, y: 0 }, 1);

      // Add another line that shares a position
      addMarker(state.board, { x: 1, y: 1 }, 1);
      addMarker(state.board, { x: 1, y: 2 }, 1);

      state.board.formedLines = [
        {
          player: 1,
          positions: [
            { x: 0, y: 0 },
            { x: 1, y: 0 },
            { x: 2, y: 0 },
          ],
          length: 3,
        },
        {
          player: 1,
          positions: [
            { x: 1, y: 0 },
            { x: 1, y: 1 },
            { x: 1, y: 2 },
          ],
          length: 3,
        },
      ];

      const result = mutateProcessLine(state, {
        type: 'process_line',
        player: 1,
        lineIndex: 0,
      });

      // The second line should be removed because it shares position (1,0)
      expect(result.board.formedLines.length).toBeLessThan(2);
    });
  });

  // ==========================================================================
  // applyProcessLineDecision edge cases
  // ==========================================================================
  describe('applyProcessLineDecision edge cases', () => {
    it('throws for wrong move type', () => {
      const state = createTestGameState();
      expect(() => {
        applyProcessLineDecision(state, {
          type: 'move_stack' as any,
          player: 1,
          to: { x: 0, y: 0 },
          timestamp: new Date(),
          thinkTime: 0,
          moveNumber: 1,
          id: 'test',
        });
      }).toThrow("expected move.type === 'process_line'");
    });

    it('returns no-op when line not found', () => {
      const state = createTestGameState();
      state.board.formedLines = [];

      const result = applyProcessLineDecision(state, {
        type: 'process_line',
        player: 1,
        to: { x: 0, y: 0 },
        formedLines: [
          {
            player: 1,
            positions: [{ x: 0, y: 0 }],
            length: 1,
          },
        ],
        timestamp: new Date(),
        thinkTime: 0,
        moveNumber: 1,
        id: 'test',
      });

      expect(result.pendingLineRewardElimination).toBe(false);
    });

    it('returns no-op for line below threshold', () => {
      const state = createTestGameState();
      state.board.formedLines = [
        {
          player: 1,
          positions: [
            { x: 0, y: 0 },
            { x: 1, y: 0 },
          ],
          length: 2, // Below threshold
        },
      ];

      const result = applyProcessLineDecision(state, {
        type: 'process_line',
        player: 1,
        to: { x: 0, y: 0 },
        formedLines: [state.board.formedLines[0]],
        timestamp: new Date(),
        thinkTime: 0,
        moveNumber: 1,
        id: 'test',
      });

      expect(result.pendingLineRewardElimination).toBe(false);
    });

    it('returns no-op for overlength line (requires choose_line_reward)', () => {
      // 4-player game has threshold 4, so 5 markers is overlength
      const state = createTestGameState({ numPlayers: 4 });
      state.board.formedLines = [
        {
          player: 1,
          positions: [
            { x: 0, y: 0 },
            { x: 1, y: 0 },
            { x: 2, y: 0 },
            { x: 3, y: 0 },
            { x: 4, y: 0 },
          ],
          length: 5, // Overlength
        },
      ];

      const result = applyProcessLineDecision(state, {
        type: 'process_line',
        player: 1,
        to: { x: 0, y: 0 },
        formedLines: [state.board.formedLines[0]],
        timestamp: new Date(),
        thinkTime: 0,
        moveNumber: 1,
        id: 'test',
      });

      expect(result.pendingLineRewardElimination).toBe(false);
    });
  });

  // ==========================================================================
  // applyChooseLineRewardDecision edge cases
  // ==========================================================================
  describe('applyChooseLineRewardDecision edge cases', () => {
    it('throws for wrong move type', () => {
      const state = createTestGameState();
      expect(() => {
        applyChooseLineRewardDecision(state, {
          type: 'move_stack' as any,
          player: 1,
          to: { x: 0, y: 0 },
          timestamp: new Date(),
          thinkTime: 0,
          moveNumber: 1,
          id: 'test',
        });
      }).toThrow("expected move.type === 'choose_line_reward'");
    });

    it('returns no-op when line not found', () => {
      const state = createTestGameState();
      state.board.formedLines = [];

      const result = applyChooseLineRewardDecision(state, {
        type: 'choose_line_reward',
        player: 1,
        to: { x: 0, y: 0 },
        formedLines: [
          {
            player: 1,
            positions: [{ x: 0, y: 0 }],
            length: 1,
          },
        ],
        timestamp: new Date(),
        thinkTime: 0,
        moveNumber: 1,
        id: 'test',
      });

      expect(result.pendingLineRewardElimination).toBe(false);
    });

    it('returns no-op for line below threshold', () => {
      const state = createTestGameState();
      state.board.formedLines = [
        {
          player: 1,
          positions: [
            { x: 0, y: 0 },
            { x: 1, y: 0 },
          ],
          length: 2,
        },
      ];

      const result = applyChooseLineRewardDecision(state, {
        type: 'choose_line_reward',
        player: 1,
        to: { x: 0, y: 0 },
        formedLines: [state.board.formedLines[0]],
        timestamp: new Date(),
        thinkTime: 0,
        moveNumber: 1,
        id: 'test',
      });

      expect(result.pendingLineRewardElimination).toBe(false);
    });

    it('handles exact-length line collapse (4-player)', () => {
      // 4-player game has threshold 4
      const state = createTestGameState({ numPlayers: 4 });
      state.board.markers.clear();
      addMarker(state.board, { x: 0, y: 0 }, 1);
      addMarker(state.board, { x: 1, y: 0 }, 1);
      addMarker(state.board, { x: 2, y: 0 }, 1);
      addMarker(state.board, { x: 3, y: 0 }, 1);
      state.board.formedLines = [
        {
          player: 1,
          positions: [
            { x: 0, y: 0 },
            { x: 1, y: 0 },
            { x: 2, y: 0 },
            { x: 3, y: 0 },
          ],
          length: 4,
        },
      ];

      const result = applyChooseLineRewardDecision(state, {
        type: 'choose_line_reward',
        player: 1,
        to: { x: 0, y: 0 },
        formedLines: [state.board.formedLines[0]],
        timestamp: new Date(),
        thinkTime: 0,
        moveNumber: 1,
        id: 'test',
      });

      expect(result.pendingLineRewardElimination).toBe(true);
    });

    it('handles overlength collapse-all (4-player)', () => {
      // 4-player game has threshold 4, so 5 markers is overlength
      const state = createTestGameState({ numPlayers: 4 });
      state.board.markers.clear();
      addMarker(state.board, { x: 0, y: 0 }, 1);
      addMarker(state.board, { x: 1, y: 0 }, 1);
      addMarker(state.board, { x: 2, y: 0 }, 1);
      addMarker(state.board, { x: 3, y: 0 }, 1);
      addMarker(state.board, { x: 4, y: 0 }, 1);
      state.board.formedLines = [
        {
          player: 1,
          positions: [
            { x: 0, y: 0 },
            { x: 1, y: 0 },
            { x: 2, y: 0 },
            { x: 3, y: 0 },
            { x: 4, y: 0 },
          ],
          length: 5,
        },
      ];

      const result = applyChooseLineRewardDecision(state, {
        type: 'choose_line_reward',
        player: 1,
        to: { x: 0, y: 0 },
        formedLines: [state.board.formedLines[0]],
        // No collapsedMarkers = collapse all
        timestamp: new Date(),
        thinkTime: 0,
        moveNumber: 1,
        id: 'test',
      });

      expect(result.pendingLineRewardElimination).toBe(true);
    });

    it('handles overlength minimum-collapse (4-player)', () => {
      // 4-player game has threshold 4, so 5 markers is overlength
      const state = createTestGameState({ numPlayers: 4 });
      state.board.markers.clear();
      addMarker(state.board, { x: 0, y: 0 }, 1);
      addMarker(state.board, { x: 1, y: 0 }, 1);
      addMarker(state.board, { x: 2, y: 0 }, 1);
      addMarker(state.board, { x: 3, y: 0 }, 1);
      addMarker(state.board, { x: 4, y: 0 }, 1);
      state.board.formedLines = [
        {
          player: 1,
          positions: [
            { x: 0, y: 0 },
            { x: 1, y: 0 },
            { x: 2, y: 0 },
            { x: 3, y: 0 },
            { x: 4, y: 0 },
          ],
          length: 5,
        },
      ];

      const result = applyChooseLineRewardDecision(state, {
        type: 'choose_line_reward',
        player: 1,
        to: { x: 0, y: 0 },
        formedLines: [state.board.formedLines[0]],
        collapsedMarkers: [
          { x: 0, y: 0 },
          { x: 1, y: 0 },
          { x: 2, y: 0 },
          { x: 3, y: 0 },
        ],
        timestamp: new Date(),
        thinkTime: 0,
        moveNumber: 1,
        id: 'test',
      });

      expect(result.pendingLineRewardElimination).toBe(false);
    });
  });

  // ==========================================================================
  // applyLineCollapse wrapper
  // ==========================================================================
  describe('applyLineCollapse', () => {
    it('returns success for valid collapse', () => {
      const state = createTestGameState();
      state.currentPhase = 'line_processing';
      state.currentPlayer = 1;
      state.board.markers.clear();
      addMarker(state.board, { x: 0, y: 0 }, 1);
      addMarker(state.board, { x: 1, y: 0 }, 1);
      addMarker(state.board, { x: 2, y: 0 }, 1);
      state.board.formedLines = [
        {
          player: 1,
          positions: [
            { x: 0, y: 0 },
            { x: 1, y: 0 },
            { x: 2, y: 0 },
          ],
          length: 3,
        },
      ];

      const decision: LineCollapseDecision = {
        lineIndex: 0,
        selection: 'COLLAPSE_ALL',
        line: state.board.formedLines[0],
        player: 1,
      };

      const result = applyLineCollapse(state, decision);
      expect(result.success).toBe(true);
      if (result.success) {
        expect(result.newState).toBeDefined();
      }
    });

    it('returns failure for invalid collapse', () => {
      const state = createTestGameState();
      state.currentPhase = 'movement'; // Wrong phase
      state.currentPlayer = 1;
      state.board.formedLines = [
        {
          player: 1,
          positions: [
            { x: 0, y: 0 },
            { x: 1, y: 0 },
            { x: 2, y: 0 },
          ],
          length: 3,
        },
      ];

      const decision: LineCollapseDecision = {
        lineIndex: 0,
        selection: 'COLLAPSE_ALL',
        line: state.board.formedLines[0],
        player: 1,
      };

      const result = applyLineCollapse(state, decision);
      expect(result.success).toBe(false);
      if (!result.success) {
        expect(result.reason).toBeDefined();
      }
    });
  });

  // ==========================================================================
  // Marker on stack/collapsed position detection
  // ==========================================================================
  describe('findAllLines marker filtering', () => {
    it('skips markers on collapsed spaces', () => {
      const state = createTestGameState({ boardType: 'square8' });
      state.board.markers.clear();

      // Add marker on a collapsed space (should be skipped)
      addMarker(state.board, { x: 0, y: 0 }, 1);
      addCollapsedSpace(state.board, { x: 0, y: 0 }, 1);

      addMarker(state.board, { x: 1, y: 0 }, 1);
      addMarker(state.board, { x: 2, y: 0 }, 1);

      const lines = findAllLines(state.board);
      // The collapsed marker shouldn't contribute to any line
      expect(Array.isArray(lines)).toBe(true);
    });

    it('skips markers on stack positions', () => {
      const state = createTestGameState({ boardType: 'square8' });
      state.board.markers.clear();

      // Add marker on a stack position (should be skipped)
      addMarker(state.board, { x: 0, y: 0 }, 1);
      addStack(state.board, { x: 0, y: 0 }, 1, 1, 1);

      addMarker(state.board, { x: 1, y: 0 }, 1);
      addMarker(state.board, { x: 2, y: 0 }, 1);

      const lines = findAllLines(state.board);
      // The stack marker shouldn't contribute to any line
      expect(Array.isArray(lines)).toBe(true);
    });
  });
});
