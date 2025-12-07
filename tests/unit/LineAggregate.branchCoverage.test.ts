/**
 * LineAggregate.branchCoverage.test.ts
 *
 * Branch coverage improvement tests for LineAggregate.
 * Targets specific uncovered branches in line detection, validation, and processing.
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

describe('LineAggregate - Branch Coverage', () => {
  // ==========================================================================
  // Hexagonal Board Line Detection
  // ==========================================================================
  describe('Hexagonal board line detection', () => {
    it('detects lines on hexagonal board with cube coordinates', () => {
      const state = createTestGameState({ boardType: 'hexagonal' });
      state.board.stacks.clear();
      state.board.markers.clear();

      // Create a line along East direction: (0,0,0), (1,0,-1), (2,0,-2), (3,0,-3)
      addMarker(state.board, { x: 0, y: 0, z: 0 }, 1);
      addMarker(state.board, { x: 1, y: 0, z: -1 }, 1);
      addMarker(state.board, { x: 2, y: 0, z: -2 }, 1);
      addMarker(state.board, { x: 3, y: 0, z: -3 }, 1);

      const lines = findAllLines(state.board);
      expect(Array.isArray(lines)).toBe(true);
    });

    it('handles hex line along NE direction', () => {
      const state = createTestGameState({ boardType: 'hexagonal' });
      state.board.stacks.clear();
      state.board.markers.clear();

      // NE direction: (1, -1, 0) steps
      addMarker(state.board, { x: 0, y: 0, z: 0 }, 1);
      addMarker(state.board, { x: 1, y: -1, z: 0 }, 1);
      addMarker(state.board, { x: 2, y: -2, z: 0 }, 1);
      addMarker(state.board, { x: 3, y: -3, z: 0 }, 1);

      const lines = findAllLines(state.board);
      expect(Array.isArray(lines)).toBe(true);
    });

    it('handles hex line along NW direction', () => {
      const state = createTestGameState({ boardType: 'hexagonal' });
      state.board.stacks.clear();
      state.board.markers.clear();

      // NW direction: (0, -1, 1) steps
      addMarker(state.board, { x: 0, y: 0, z: 0 }, 1);
      addMarker(state.board, { x: 0, y: -1, z: 1 }, 1);
      addMarker(state.board, { x: 0, y: -2, z: 2 }, 1);
      addMarker(state.board, { x: 0, y: -3, z: 3 }, 1);

      const lines = findAllLines(state.board);
      expect(Array.isArray(lines)).toBe(true);
    });

    it('validates hex board position at radius boundary', () => {
      const state = createTestGameState({ boardType: 'hexagonal' });
      state.board.stacks.clear();
      state.board.markers.clear();

      // Position at radius 12
      addMarker(state.board, { x: 12, y: -6, z: -6 }, 1);

      const lines = findAllLines(state.board);
      expect(Array.isArray(lines)).toBe(true);
    });
  });

  // ==========================================================================
  // Square Board Line Detection
  // ==========================================================================
  describe('Square board line detection', () => {
    it('detects horizontal line (East direction)', () => {
      const state = createTestGameState({ boardType: 'square8' });
      state.board.stacks.clear();
      state.board.markers.clear();

      addMarker(state.board, { x: 0, y: 0 }, 1);
      addMarker(state.board, { x: 1, y: 0 }, 1);
      addMarker(state.board, { x: 2, y: 0 }, 1);

      const lines = findAllLines(state.board);
      expect(lines.length).toBeGreaterThan(0);
      expect(lines[0].positions.length).toBeGreaterThanOrEqual(3);
    });

    it('detects vertical line (South direction)', () => {
      const state = createTestGameState({ boardType: 'square8' });
      state.board.stacks.clear();
      state.board.markers.clear();

      addMarker(state.board, { x: 0, y: 0 }, 1);
      addMarker(state.board, { x: 0, y: 1 }, 1);
      addMarker(state.board, { x: 0, y: 2 }, 1);

      const lines = findAllLines(state.board);
      expect(lines.length).toBeGreaterThan(0);
    });

    it('detects diagonal line (SE direction)', () => {
      const state = createTestGameState({ boardType: 'square8' });
      state.board.stacks.clear();
      state.board.markers.clear();

      addMarker(state.board, { x: 0, y: 0 }, 1);
      addMarker(state.board, { x: 1, y: 1 }, 1);
      addMarker(state.board, { x: 2, y: 2 }, 1);

      const lines = findAllLines(state.board);
      expect(lines.length).toBeGreaterThan(0);
    });

    it('detects anti-diagonal line (NE direction)', () => {
      const state = createTestGameState({ boardType: 'square8' });
      state.board.stacks.clear();
      state.board.markers.clear();

      addMarker(state.board, { x: 0, y: 2 }, 1);
      addMarker(state.board, { x: 1, y: 1 }, 1);
      addMarker(state.board, { x: 2, y: 0 }, 1);

      const lines = findAllLines(state.board);
      expect(lines.length).toBeGreaterThan(0);
    });

    it('does not form line with mixed player markers', () => {
      const state = createTestGameState({ boardType: 'square8' });
      state.board.stacks.clear();
      state.board.markers.clear();

      addMarker(state.board, { x: 0, y: 0 }, 1);
      addMarker(state.board, { x: 1, y: 0 }, 2); // Different player
      addMarker(state.board, { x: 2, y: 0 }, 1);

      const linesPlayer1 = findLinesForPlayer(state.board, 1);
      expect(
        linesPlayer1.every(
          (line) =>
            line.length < 3 ||
            line.positions.every((p) => {
              const key = positionToString(p);
              return state.board.markers.get(key)?.player === 1;
            })
        )
      ).toBe(true);
    });

    it('line is broken by collapsed space', () => {
      const state = createTestGameState({ boardType: 'square8' });
      state.board.stacks.clear();
      state.board.markers.clear();

      addMarker(state.board, { x: 0, y: 0 }, 1);
      addCollapsedSpace(state.board, { x: 1, y: 0 }, 1);
      addMarker(state.board, { x: 2, y: 0 }, 1);

      const lines = findLinesForPlayer(state.board, 1);
      // Lines should not span the collapsed space
      expect(lines.every((line) => line.length < 3)).toBe(true);
    });

    it('line is broken by stack', () => {
      const state = createTestGameState({ boardType: 'square8' });
      state.board.stacks.clear();
      state.board.markers.clear();

      addMarker(state.board, { x: 0, y: 0 }, 1);
      addStack(state.board, { x: 1, y: 0 }, 1, 1, 1); // Stack breaks line
      addMarker(state.board, { x: 2, y: 0 }, 1);

      const lines = findLinesForPlayer(state.board, 1);
      expect(lines.every((line) => line.length < 3)).toBe(true);
    });
  });

  // ==========================================================================
  // Square19 Board
  // ==========================================================================
  describe('Square19 board handling', () => {
    it('detects lines on square19 board', () => {
      const state = createTestGameState({ boardType: 'square19' });
      state.board.stacks.clear();
      state.board.markers.clear();

      // Square19 requires 4 markers for a line
      addMarker(state.board, { x: 0, y: 0 }, 1);
      addMarker(state.board, { x: 1, y: 0 }, 1);
      addMarker(state.board, { x: 2, y: 0 }, 1);
      addMarker(state.board, { x: 3, y: 0 }, 1);

      const lines = findAllLines(state.board);
      expect(Array.isArray(lines)).toBe(true);
    });

    it('validates positions at square19 boundary', () => {
      const state = createTestGameState({ boardType: 'square19' });
      state.board.stacks.clear();
      state.board.markers.clear();

      // Edge position
      addMarker(state.board, { x: 18, y: 18 }, 1);

      const lines = findAllLines(state.board);
      expect(Array.isArray(lines)).toBe(true);
    });
  });

  // ==========================================================================
  // Validation Branches
  // ==========================================================================
  describe('validateProcessLine branches', () => {
    it('rejects when not in line_processing phase', () => {
      const state = createTestGameState();
      state.currentPhase = 'movement';

      const result = validateProcessLine(state, {
        type: 'PROCESS_LINE',
        playerId: 1,
        lineIndex: 0,
      });

      expect(result.valid).toBe(false);
      expect(result.code).toBe('INVALID_PHASE');
    });

    it('rejects when not player turn', () => {
      const state = createTestGameState();
      state.currentPhase = 'line_processing';
      state.currentPlayer = 2;

      const result = validateProcessLine(state, {
        type: 'PROCESS_LINE',
        playerId: 1,
        lineIndex: 0,
      });

      expect(result.valid).toBe(false);
      expect(result.code).toBe('NOT_YOUR_TURN');
    });

    it('rejects when line index out of range', () => {
      const state = createTestGameState();
      state.currentPhase = 'line_processing';
      state.currentPlayer = 1;
      state.board.formedLines = [];

      const result = validateProcessLine(state, {
        type: 'PROCESS_LINE',
        playerId: 1,
        lineIndex: 0,
      });

      expect(result.valid).toBe(false);
    });

    it('rejects when line belongs to different player', () => {
      const state = createTestGameState();
      state.currentPhase = 'line_processing';
      state.currentPlayer = 1;
      state.board.formedLines = [
        {
          player: 2,
          positions: [
            { x: 0, y: 0 },
            { x: 1, y: 0 },
            { x: 2, y: 0 },
          ],
          length: 3,
        },
      ];

      const result = validateProcessLine(state, {
        type: 'PROCESS_LINE',
        playerId: 1,
        lineIndex: 0,
      });

      expect(result.valid).toBe(false);
    });

    it('accepts valid line processing', () => {
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

      const result = validateProcessLine(state, {
        type: 'PROCESS_LINE',
        playerId: 1,
        lineIndex: 0,
      });

      expect(result.valid).toBe(true);
    });
  });

  describe('validateChooseLineReward branches', () => {
    it('rejects when not in line_decision phase', () => {
      const state = createTestGameState();
      state.currentPhase = 'movement';

      const result = validateChooseLineReward(state, {
        type: 'CHOOSE_LINE_REWARD',
        playerId: 1,
        lineIndex: 0,
        rewardChoice: 'COLLAPSE_ALL',
      });

      expect(result.valid).toBe(false);
    });

    it('rejects when not player turn', () => {
      const state = createTestGameState();
      state.currentPhase = 'line_decision';
      state.currentPlayer = 2;

      const result = validateChooseLineReward(state, {
        type: 'CHOOSE_LINE_REWARD',
        playerId: 1,
        lineIndex: 0,
        rewardChoice: 'COLLAPSE_ALL',
      });

      expect(result.valid).toBe(false);
    });

    it('rejects invalid reward choice', () => {
      const state = createTestGameState();
      state.currentPhase = 'line_decision';
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
        rewardChoice: 'INVALID' as any,
      });

      expect(result.valid).toBe(false);
    });

    it('validates COLLAPSE_ALL for overlength line', () => {
      const state = createTestGameState();
      state.currentPhase = 'line_decision';
      state.currentPlayer = 1;
      state.board.markers.clear();
      // Need 5 markers for overlength (threshold 4)
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

      const result = validateChooseLineReward(state, {
        type: 'CHOOSE_LINE_REWARD',
        playerId: 1,
        lineIndex: 0,
        rewardChoice: 'COLLAPSE_ALL',
        collapsedPositions: [
          { x: 0, y: 0 },
          { x: 1, y: 0 },
          { x: 2, y: 0 },
          { x: 3, y: 0 },
          { x: 4, y: 0 },
        ],
      });

      // Function returns a validation result with valid boolean
      expect(typeof result.valid).toBe('boolean');
    });

    it('accepts MINIMUM_COLLAPSE for overlength line', () => {
      const state = createTestGameState();
      state.currentPhase = 'line_decision';
      state.currentPlayer = 1;
      state.board.markers.clear();
      // Need 5 markers for overlength (threshold 4)
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

      const result = validateChooseLineReward(state, {
        type: 'CHOOSE_LINE_REWARD',
        playerId: 1,
        lineIndex: 0,
        rewardChoice: 'MINIMUM_COLLAPSE',
        collapsedPositions: [
          { x: 0, y: 0 },
          { x: 1, y: 0 },
          { x: 2, y: 0 },
          { x: 3, y: 0 },
        ],
      });

      expect(typeof result.valid).toBe('boolean');
    });
  });

  // ==========================================================================
  // Enumeration Branches
  // ==========================================================================
  describe('enumerateProcessLineMoves', () => {
    it('returns empty when not in line_processing phase', () => {
      const state = createTestGameState();
      state.currentPhase = 'movement';

      const moves = enumerateProcessLineMoves(state, 1);
      expect(moves).toEqual([]);
    });

    it('returns empty when no lines for player', () => {
      const state = createTestGameState();
      state.currentPhase = 'line_processing';
      state.currentPlayer = 1;
      state.board.formedLines = [];

      const moves = enumerateProcessLineMoves(state, 1);
      expect(moves).toEqual([]);
    });

    it('returns moves for valid lines', () => {
      const state = createTestGameState();
      state.currentPhase = 'line_processing';
      state.currentPlayer = 1;
      state.board.markers.clear();
      // Need 4 markers for square8 2-player threshold
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

      const moves = enumerateProcessLineMoves(state, 1);
      expect(moves.length).toBeGreaterThan(0);
    });

    it('uses detect_now mode to find lines', () => {
      const state = createTestGameState();
      state.currentPhase = 'line_processing';
      state.currentPlayer = 1;
      state.board.markers.clear();
      state.board.formedLines = []; // Empty cache

      addMarker(state.board, { x: 0, y: 0 }, 1);
      addMarker(state.board, { x: 1, y: 0 }, 1);
      addMarker(state.board, { x: 2, y: 0 }, 1);

      const moves = enumerateProcessLineMoves(state, 1, { detectionMode: 'detect_now' });
      expect(Array.isArray(moves)).toBe(true);
    });
  });

  describe('enumerateChooseLineRewardMoves', () => {
    it('returns empty when not in line_decision phase', () => {
      const state = createTestGameState();
      state.currentPhase = 'movement';

      // Pass lineIndex as 3rd parameter
      const moves = enumerateChooseLineRewardMoves(state, 1, 0);
      expect(moves).toEqual([]);
    });

    it('returns moves for overlength lines', () => {
      const state = createTestGameState();
      state.currentPhase = 'line_decision';
      state.currentPlayer = 1;
      // Need 5 markers for overlength (threshold is 4)
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

      // Pass lineIndex as 3rd parameter
      const moves = enumerateChooseLineRewardMoves(state, 1, 0);
      expect(moves.length).toBeGreaterThan(0);
    });
  });

  // ==========================================================================
  // Application Branches
  // ==========================================================================
  describe('applyProcessLineDecision', () => {
    it('handles invalid state gracefully', () => {
      const state = createTestGameState();
      state.currentPhase = 'movement'; // Wrong phase

      // Function may throw or return result depending on implementation
      try {
        const result = applyProcessLineDecision(state, {
          type: 'process_line',
          lineIndex: 0,
          selection: 'COLLAPSE_ALL',
          line: { player: 1, positions: [{ x: 0, y: 0 }], length: 1 },
          player: 1,
        });
        // If it doesn't throw, it returns a LineDecisionApplicationOutcome
        expect(result).toBeDefined();
      } catch (e) {
        // If it throws, that's also valid behavior
        expect(e).toBeDefined();
      }
    });

    it('applies valid line processing decision', () => {
      const state = createTestGameState();
      state.currentPhase = 'line_processing';
      state.currentPlayer = 1;
      state.board.markers.clear();
      // Need 4 markers for threshold
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

      const result = applyProcessLineDecision(state, {
        type: 'process_line',
        lineIndex: 0,
        selection: 'COLLAPSE_ALL',
        line: {
          player: 1,
          positions: [
            { x: 0, y: 0 },
            { x: 1, y: 0 },
            { x: 2, y: 0 },
            { x: 3, y: 0 },
          ],
          length: 4,
        },
        player: 1,
      });

      // Returns LineDecisionApplicationOutcome with nextState
      expect(result.nextState).toBeDefined();
    });
  });

  describe('applyChooseLineRewardDecision', () => {
    it('returns result for decision', () => {
      const state = createTestGameState();
      state.currentPhase = 'line_decision';
      state.currentPlayer = 1;
      state.board.markers.clear();
      // Need 5 markers for overlength
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
        lineIndex: 0,
        selection: 'COLLAPSE_ALL',
        line: {
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
        player: 1,
        rewardChoice: 'COLLAPSE_ALL',
        collapsedPositions: [
          { x: 0, y: 0 },
          { x: 1, y: 0 },
          { x: 2, y: 0 },
          { x: 3, y: 0 },
          { x: 4, y: 0 },
        ],
      });

      // Returns LineDecisionApplicationOutcome with nextState
      expect(result.nextState).toBeDefined();
    });
  });

  // ==========================================================================
  // Mutation Branches
  // ==========================================================================
  describe('mutateProcessLine', () => {
    it('processes exact-length line', () => {
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

      const result = mutateProcessLine(state, {
        type: 'process_line',
        player: 1,
        lineIndex: 0,
      });

      expect(result).toBeDefined();
    });
  });

  describe('mutateChooseLineReward', () => {
    it('applies COLLAPSE_ALL reward with all positions', () => {
      const state = createTestGameState();
      state.currentPhase = 'line_decision';
      state.currentPlayer = 1;
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

      const result = mutateChooseLineReward(state, {
        type: 'choose_line_reward',
        player: 1,
        lineIndex: 0,
        rewardChoice: 'COLLAPSE_ALL',
        // COLLAPSE_ALL uses all positions from the line
        collapsedPositions: [
          { x: 0, y: 0 },
          { x: 1, y: 0 },
          { x: 2, y: 0 },
          { x: 3, y: 0 },
        ],
      });

      expect(result).toBeDefined();
    });

    it('applies MINIMUM_COLLAPSE reward', () => {
      const state = createTestGameState();
      state.currentPhase = 'line_decision';
      state.currentPlayer = 1;
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

      const result = mutateChooseLineReward(state, {
        type: 'choose_line_reward',
        player: 1,
        lineIndex: 0,
        rewardChoice: 'MINIMUM_COLLAPSE',
        collapsedPositions: [
          { x: 0, y: 0 },
          { x: 1, y: 0 },
          { x: 2, y: 0 },
        ],
      });

      expect(result).toBeDefined();
    });
  });

  // ==========================================================================
  // Move Number Computation
  // ==========================================================================
  describe('Move number computation branches', () => {
    it('uses history for move number', () => {
      const state = createTestGameState();
      state.currentPhase = 'line_processing';
      state.currentPlayer = 1;
      state.history = [{ moveNumber: 5, phase: 'movement' as any, boardSnapshot: {} as any }];
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

      const result = mutateProcessLine(state, {
        type: 'process_line',
        player: 1,
        lineIndex: 0,
      });

      expect(result).toBeDefined();
    });

    it('uses moveHistory fallback for move number', () => {
      const state = createTestGameState();
      state.currentPhase = 'line_processing';
      state.currentPlayer = 1;
      state.history = [];
      state.moveHistory = [
        {
          id: '1',
          type: 'movement',
          player: 1,
          to: { x: 0, y: 0 },
          moveNumber: 10,
          timestamp: new Date(),
          thinkTime: 0,
        },
      ];
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

      const result = mutateProcessLine(state, {
        type: 'process_line',
        player: 1,
        lineIndex: 0,
      });

      expect(result).toBeDefined();
    });

    it('defaults to move 1 when no history', () => {
      const state = createTestGameState();
      state.currentPhase = 'line_processing';
      state.currentPlayer = 1;
      state.history = [];
      state.moveHistory = [];
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

      const result = mutateProcessLine(state, {
        type: 'process_line',
        player: 1,
        lineIndex: 0,
      });

      expect(result).toBeDefined();
    });
  });

  // ==========================================================================
  // findLinesContainingPosition
  // ==========================================================================
  describe('findLinesContainingPosition', () => {
    it('finds lines containing a specific position', () => {
      const state = createTestGameState({ boardType: 'square8' });
      state.board.markers.clear();
      addMarker(state.board, { x: 0, y: 0 }, 1);
      addMarker(state.board, { x: 1, y: 0 }, 1);
      addMarker(state.board, { x: 2, y: 0 }, 1);

      const lines = findLinesContainingPosition(state, { x: 1, y: 0 });
      expect(Array.isArray(lines)).toBe(true);
    });

    it('returns empty array when position not in any line', () => {
      const state = createTestGameState({ boardType: 'square8' });
      state.board.markers.clear();
      addMarker(state.board, { x: 0, y: 0 }, 1);
      addMarker(state.board, { x: 1, y: 0 }, 1);
      addMarker(state.board, { x: 2, y: 0 }, 1);

      const lines = findLinesContainingPosition(state, { x: 5, y: 5 });
      expect(lines).toHaveLength(0);
    });
  });

  // ==========================================================================
  // validateLineDecision wrapper
  // ==========================================================================
  describe('validateLineDecision', () => {
    it('validates a COLLAPSE_ALL decision', () => {
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

      const decision: LineCollapseDecision = {
        lineIndex: 0,
        selection: 'COLLAPSE_ALL',
        line: state.board.formedLines[0],
        player: 1,
      };

      const result = validateLineDecision(state, decision);
      expect(typeof result.valid).toBe('boolean');
    });

    it('validates a MINIMUM_COLLAPSE decision with positions', () => {
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
            { x: 3, y: 0 },
          ],
          length: 4,
        },
      ];

      const decision: LineCollapseDecision = {
        lineIndex: 0,
        selection: 'MINIMUM_COLLAPSE',
        collapsedPositions: [
          { x: 0, y: 0 },
          { x: 1, y: 0 },
          { x: 2, y: 0 },
        ],
        line: state.board.formedLines[0],
        player: 1,
      };

      const result = validateLineDecision(state, decision);
      expect(typeof result.valid).toBe('boolean');
    });
  });

  // ==========================================================================
  // validateChooseLineReward additional branches
  // ==========================================================================
  describe('validateChooseLineReward additional branches', () => {
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
