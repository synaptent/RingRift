/**
 * CaptureValidator.shared.test.ts
 *
 * Structural validation tests for overtaking captures, aligned with:
 * - RulesMatrix M3 – movement vs overtaking capture parity.
 * - RulesMatrix C1–C4 – basic and complex chain patterns (§9–10, FAQ 5–6, 9, 12, 15.3.x).
 *
 * These tests focus on phase/turn/position/height checks. Full multi-step chain
 * behaviour remains covered by:
 * - tests/scenarios/ComplexChainCaptures.test.ts
 * - tests/scenarios/RulesMatrix.ChainCapture.GameEngine.test.ts
 * - tests/scenarios/RulesMatrix.ChainCapture.ClientSandboxEngine.test.ts
 */

import { validateCapture } from '../../src/shared/engine/validators/CaptureValidator';
import type { GameState, OvertakingCaptureAction } from '../../src/shared/engine/types';
import type { BoardState, BoardType } from '../../src/shared/types/game';

// Helper to create minimal BoardState for capture tests
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
    formedLines: [],
    eliminatedRings: 0,
    geometry: { type: overrides.type ?? 'square8', size: overrides.size ?? 8 },
  } as unknown as BoardState;
}

// Helper to create minimal GameState for capture validation
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
    currentPhase: overrides.currentPhase ?? 'movement',
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

// Helper to create a capture action
function createCaptureAction(
  overrides: Partial<OvertakingCaptureAction> = {}
): OvertakingCaptureAction {
  return {
    type: 'OVERTAKING_CAPTURE',
    playerId: 1,
    from: { x: 0, y: 0 },
    to: { x: 2, y: 0 },
    captureTarget: { x: 1, y: 0 },
    ...overrides,
  };
}

describe('CaptureValidator', () => {
  describe('validateCapture', () => {
    describe('Phase Checks', () => {
      it('should reject capture in ring_placement phase', () => {
        const state = createMinimalState({ currentPhase: 'ring_placement' });
        const action = createCaptureAction();

        const result = validateCapture(state, action);

        expect(result.valid).toBe(false);
        if (!result.valid) {
          expect(result.code).toBe('INVALID_PHASE');
          expect(result.reason).toContain('Not in a phase allowing capture');
        }
      });

      it('should reject capture in line_processing phase', () => {
        const state = createMinimalState({ currentPhase: 'line_processing' });
        const action = createCaptureAction();

        const result = validateCapture(state, action);

        expect(result.valid).toBe(false);
        if (!result.valid) {
          expect(result.code).toBe('INVALID_PHASE');
        }
      });

      it('should reject capture in territory_processing phase', () => {
        const state = createMinimalState({ currentPhase: 'territory_processing' });
        const action = createCaptureAction();

        const result = validateCapture(state, action);

        expect(result.valid).toBe(false);
        if (!result.valid) {
          expect(result.code).toBe('INVALID_PHASE');
        }
      });

      it('should reject capture in game_over phase', () => {
        const state = createMinimalState({ currentPhase: 'game_over' });
        const action = createCaptureAction();

        const result = validateCapture(state, action);

        expect(result.valid).toBe(false);
        if (!result.valid) {
          expect(result.code).toBe('INVALID_PHASE');
        }
      });

      it('should allow capture in movement phase', () => {
        // Set up a valid capture scenario
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
          [
            '1,0',
            {
              controllingPlayer: 2,
              stackHeight: 1,
              capHeight: 1,
              rings: [2],
              position: { x: 1, y: 0 },
            },
          ],
        ]);
        const state = createMinimalState({ currentPhase: 'movement', stacks });

        const action = createCaptureAction();
        const result = validateCapture(state, action);

        // Phase check passes - may fail on core validation
        if (!result.valid) {
          expect(result.code).not.toBe('INVALID_PHASE');
        }
      });

      it('should allow capture in capture phase', () => {
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
          [
            '1,0',
            {
              controllingPlayer: 2,
              stackHeight: 1,
              capHeight: 1,
              rings: [2],
              position: { x: 1, y: 0 },
            },
          ],
        ]);
        const state = createMinimalState({ currentPhase: 'capture', stacks });

        const action = createCaptureAction();
        const result = validateCapture(state, action);

        // Phase check passes - may fail on core validation
        if (!result.valid) {
          expect(result.code).not.toBe('INVALID_PHASE');
        }
      });

      it('should allow capture in chain_capture phase', () => {
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
          [
            '1,0',
            {
              controllingPlayer: 2,
              stackHeight: 1,
              capHeight: 1,
              rings: [2],
              position: { x: 1, y: 0 },
            },
          ],
        ]);
        const state = createMinimalState({ currentPhase: 'chain_capture', stacks });

        const action = createCaptureAction();
        const result = validateCapture(state, action);

        // Phase check passes - may fail on core validation
        if (!result.valid) {
          expect(result.code).not.toBe('INVALID_PHASE');
        }
      });
    });

    describe('Turn Checks', () => {
      it('should reject capture when not player turn', () => {
        const state = createMinimalState({ currentPhase: 'movement', currentPlayer: 2 });
        const action = createCaptureAction({ playerId: 1 });

        const result = validateCapture(state, action);

        expect(result.valid).toBe(false);
        if (!result.valid) {
          expect(result.code).toBe('NOT_YOUR_TURN');
          expect(result.reason).toBe('Not your turn');
        }
      });

      it('should accept capture when it is player turn', () => {
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
          [
            '1,0',
            {
              controllingPlayer: 2,
              stackHeight: 1,
              capHeight: 1,
              rings: [2],
              position: { x: 1, y: 0 },
            },
          ],
        ]);
        const state = createMinimalState({ currentPhase: 'movement', currentPlayer: 1, stacks });

        const action = createCaptureAction({ playerId: 1 });
        const result = validateCapture(state, action);

        // Turn check passes - may fail on position or core validation
        if (!result.valid) {
          expect(result.code).not.toBe('NOT_YOUR_TURN');
        }
      });
    });

    describe('Position Validity Checks', () => {
      it('should reject capture with from position off board', () => {
        const state = createMinimalState({ currentPhase: 'movement', currentPlayer: 1 });
        const action = createCaptureAction({
          playerId: 1,
          from: { x: -1, y: 0 },
          to: { x: 2, y: 0 },
          captureTarget: { x: 1, y: 0 },
        });

        const result = validateCapture(state, action);

        expect(result.valid).toBe(false);
        if (!result.valid) {
          expect(result.code).toBe('INVALID_POSITION');
          expect(result.reason).toBe('Position off board');
        }
      });

      it('should reject capture with to position off board', () => {
        const state = createMinimalState({ currentPhase: 'movement', currentPlayer: 1 });
        const action = createCaptureAction({
          playerId: 1,
          from: { x: 0, y: 0 },
          to: { x: 10, y: 0 }, // Off an 8x8 board
          captureTarget: { x: 1, y: 0 },
        });

        const result = validateCapture(state, action);

        expect(result.valid).toBe(false);
        if (!result.valid) {
          expect(result.code).toBe('INVALID_POSITION');
        }
      });

      it('should reject capture with captureTarget position off board', () => {
        const state = createMinimalState({ currentPhase: 'movement', currentPlayer: 1 });
        const action = createCaptureAction({
          playerId: 1,
          from: { x: 0, y: 0 },
          to: { x: 2, y: 0 },
          captureTarget: { x: 1, y: -1 }, // Off board
        });

        const result = validateCapture(state, action);

        expect(result.valid).toBe(false);
        if (!result.valid) {
          expect(result.code).toBe('INVALID_POSITION');
        }
      });

      it('should reject capture with all positions off board', () => {
        const state = createMinimalState({ currentPhase: 'movement', currentPlayer: 1 });
        const action = createCaptureAction({
          playerId: 1,
          from: { x: -1, y: -1 },
          to: { x: 100, y: 100 },
          captureTarget: { x: 50, y: 50 },
        });

        const result = validateCapture(state, action);

        expect(result.valid).toBe(false);
        if (!result.valid) {
          expect(result.code).toBe('INVALID_POSITION');
        }
      });

      it('should allow diagonal capture attempts on square8 when valid', () => {
        // On square boards, diagonal movement is allowed as one of the valid
        // movement directions. A valid diagonal capture requires proper
        // stack height relationships.
        const stacks = new Map([
          [
            '0,0',
            {
              controllingPlayer: 1,
              stackHeight: 2,
              capHeight: 2,
              rings: [1, 1],
              position: { x: 0, y: 0 },
            },
          ],
          [
            '1,1',
            {
              controllingPlayer: 2,
              stackHeight: 1,
              capHeight: 1,
              rings: [2],
              position: { x: 1, y: 1 },
            },
          ],
        ]);
        const state = createMinimalState({
          currentPhase: 'movement',
          currentPlayer: 1,
          boardType: 'square8',
          boardSize: 8,
          stacks,
        });

        const action = createCaptureAction({
          playerId: 1,
          from: { x: 0, y: 0 },
          captureTarget: { x: 1, y: 1 },
          to: { x: 2, y: 2 },
        });

        const result = validateCapture(state, action);

        // Diagonal captures are structurally valid on square boards
        // Core rules determine if this specific capture is valid
        expect(result.valid).toBe(true);
      });
    });

    describe('Core Validation Integration', () => {
      it('should reject capture when no stack at from position', () => {
        const state = createMinimalState({ currentPhase: 'movement', currentPlayer: 1 });
        // No stacks on board - from position has no stack
        const action = createCaptureAction({
          playerId: 1,
          from: { x: 0, y: 0 },
          to: { x: 2, y: 0 },
          captureTarget: { x: 1, y: 0 },
        });

        const result = validateCapture(state, action);

        expect(result.valid).toBe(false);
        if (!result.valid) {
          expect(result.code).toBe('INVALID_CAPTURE');
        }
      });

      it('should reject capture when no target stack to capture', () => {
        // Only attacker stack, no target
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
        const state = createMinimalState({ currentPhase: 'movement', currentPlayer: 1, stacks });

        const action = createCaptureAction({
          playerId: 1,
          from: { x: 0, y: 0 },
          to: { x: 2, y: 0 },
          captureTarget: { x: 1, y: 0 },
        });

        const result = validateCapture(state, action);

        expect(result.valid).toBe(false);
        if (!result.valid) {
          expect(result.code).toBe('INVALID_CAPTURE');
        }
      });

      it('should reject capture when from stack not controlled by player', () => {
        // From stack controlled by player 2
        const stacks = new Map([
          [
            '0,0',
            {
              controllingPlayer: 2,
              stackHeight: 1,
              capHeight: 1,
              rings: [2],
              position: { x: 0, y: 0 },
            },
          ],
          [
            '1,0',
            {
              controllingPlayer: 2,
              stackHeight: 1,
              capHeight: 1,
              rings: [2],
              position: { x: 1, y: 0 },
            },
          ],
        ]);
        const state = createMinimalState({ currentPhase: 'movement', currentPlayer: 1, stacks });

        const action = createCaptureAction({
          playerId: 1,
          from: { x: 0, y: 0 },
          to: { x: 2, y: 0 },
          captureTarget: { x: 1, y: 0 },
        });

        const result = validateCapture(state, action);

        expect(result.valid).toBe(false);
        if (!result.valid) {
          expect(result.code).toBe('INVALID_CAPTURE');
        }
      });

      it('should allow capture when target is own stack (RR-CANON-R101: any owner)', () => {
        // RR-CANON-R101: "target contains a stack T with capHeight CH_T (any owner)"
        // Per rules, capturing own stacks is explicitly allowed
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
          [
            '1,0',
            {
              controllingPlayer: 1,
              stackHeight: 1,
              capHeight: 1,
              rings: [1],
              position: { x: 1, y: 0 },
            },
          ],
        ]);
        const state = createMinimalState({ currentPhase: 'movement', currentPlayer: 1, stacks });

        const action = createCaptureAction({
          playerId: 1,
          from: { x: 0, y: 0 },
          to: { x: 2, y: 0 },
          captureTarget: { x: 1, y: 0 },
        });

        const result = validateCapture(state, action);

        // Per RR-CANON-R101, capturing own stack is valid (passes phase/turn/position/core checks)
        expect(result.valid).toBe(true);
      });

      it('should reject capture when landing on collapsed space', () => {
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
          [
            '1,0',
            {
              controllingPlayer: 2,
              stackHeight: 1,
              capHeight: 1,
              rings: [2],
              position: { x: 1, y: 0 },
            },
          ],
        ]);
        const collapsedSpaces = new Set(['2,0']); // Landing position is collapsed
        const state = createMinimalState({
          currentPhase: 'movement',
          currentPlayer: 1,
          stacks,
          collapsedSpaces,
        });

        const action = createCaptureAction({
          playerId: 1,
          from: { x: 0, y: 0 },
          to: { x: 2, y: 0 },
          captureTarget: { x: 1, y: 0 },
        });

        const result = validateCapture(state, action);

        expect(result.valid).toBe(false);
        if (!result.valid) {
          expect(result.code).toBe('INVALID_CAPTURE');
        }
      });

      it('should reject capture when target stack height exceeds attacker', () => {
        // Attacker has height 1, target has height 2 (taller = cannot capture)
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
          [
            '1,0',
            {
              controllingPlayer: 2,
              stackHeight: 2,
              capHeight: 2,
              rings: [2, 2],
              position: { x: 1, y: 0 },
            },
          ],
        ]);
        const state = createMinimalState({ currentPhase: 'movement', currentPlayer: 1, stacks });

        const action = createCaptureAction({
          playerId: 1,
          from: { x: 0, y: 0 },
          to: { x: 2, y: 0 },
          captureTarget: { x: 1, y: 0 },
        });

        const result = validateCapture(state, action);

        expect(result.valid).toBe(false);
        if (!result.valid) {
          expect(result.code).toBe('INVALID_CAPTURE');
        }
      });

      it('should exercise getMarkerOwner path when markers exist in capture scenario', () => {
        // This test exercises lines 49-50 (getMarkerOwner boardView function)
        // by setting up a scenario with markers present on the board
        const stacks = new Map([
          [
            '0,0',
            {
              controllingPlayer: 1,
              stackHeight: 2,
              capHeight: 2,
              rings: [1, 1],
              position: { x: 0, y: 0 },
            },
          ],
          [
            '2,0',
            {
              controllingPlayer: 2,
              stackHeight: 1,
              capHeight: 1,
              rings: [2],
              position: { x: 2, y: 0 },
            },
          ],
        ]);
        // Place markers on the board - the core validator will call getMarkerOwner
        const markers = new Map([
          ['3,0', { player: 1 }], // Own marker beyond landing
          ['4,0', { player: 2 }], // Opponent marker
        ]);
        const state = createMinimalState({
          currentPhase: 'movement',
          currentPlayer: 1,
          stacks,
          markers,
        });

        const action = createCaptureAction({
          playerId: 1,
          from: { x: 0, y: 0 },
          to: { x: 4, y: 0 }, // Landing beyond the captured stack
          captureTarget: { x: 2, y: 0 },
        });

        const result = validateCapture(state, action);

        // The validation will reach getMarkerOwner to check marker ownership
        // Result depends on core rules, but we exercise the path
        // Even if invalid, code should not be INVALID_PHASE or INVALID_POSITION
        if (!result.valid) {
          expect(result.code).not.toBe('INVALID_PHASE');
          expect(result.code).not.toBe('NOT_YOUR_TURN');
          expect(result.code).not.toBe('INVALID_POSITION');
        }
      });

      it('should handle capture with marker at empty position (undefined path)', () => {
        // Test getMarkerOwner returning undefined for empty position
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
          [
            '1,0',
            {
              controllingPlayer: 1,
              stackHeight: 1,
              capHeight: 1,
              rings: [1],
              position: { x: 1, y: 0 },
            },
          ],
        ]);
        // No markers on the board
        const state = createMinimalState({
          currentPhase: 'movement',
          currentPlayer: 1,
          stacks,
          markers: new Map(),
        });

        const action = createCaptureAction({
          playerId: 1,
          from: { x: 0, y: 0 },
          to: { x: 2, y: 0 },
          captureTarget: { x: 1, y: 0 },
        });

        const result = validateCapture(state, action);

        // Self-capture is valid per RR-CANON-R101
        expect(result.valid).toBe(true);
      });
    });

    describe('Large Position Validation', () => {
      it('should reject capture with large off-board positions', () => {
        const state = createMinimalState({
          currentPhase: 'movement',
          currentPlayer: 1,
          boardType: 'square19',
          boardSize: 19,
        });

        // Very large positions well beyond any board
        const action = createCaptureAction({
          playerId: 1,
          from: { x: 0, y: 0 },
          to: { x: 100, y: 0 },
          captureTarget: { x: 50, y: 0 },
        });

        const result = validateCapture(state, action);

        expect(result.valid).toBe(false);
        if (!result.valid) {
          expect(result.code).toBe('INVALID_POSITION');
        }
      });

      it('should respect square19 board bounds for far captures', () => {
        const stacks = new Map([
          [
            '0,0',
            {
              controllingPlayer: 1,
              stackHeight: 2,
              capHeight: 2,
              rings: [1, 1],
              position: { x: 0, y: 0 },
            },
          ],
          [
            '9,0',
            {
              controllingPlayer: 2,
              stackHeight: 1,
              capHeight: 1,
              rings: [2],
              position: { x: 9, y: 0 },
            },
          ],
        ]);
        const state = createMinimalState({
          currentPhase: 'movement',
          currentPlayer: 1,
          boardType: 'square19',
          boardSize: 19,
          stacks,
        });

        // Landing well within a 19x19 board should be structurally allowed
        const inBoundsAction = createCaptureAction({
          playerId: 1,
          from: { x: 0, y: 0 },
          captureTarget: { x: 9, y: 0 },
          to: { x: 18, y: 0 },
        });
        const inBoundsResult = validateCapture(state, inBoundsAction);

        // Core distance/stack rules may still reject, but not as INVALID_POSITION.
        if (!inBoundsResult.valid) {
          expect(inBoundsResult.code).not.toBe('INVALID_POSITION');
        }

        // Landing beyond the 19x19 bounds must be rejected as INVALID_POSITION.
        const offBoardAction = createCaptureAction({
          playerId: 1,
          from: { x: 0, y: 0 },
          captureTarget: { x: 9, y: 0 },
          to: { x: 19, y: 0 },
        });
        const offBoardResult = validateCapture(state, offBoardAction);

        expect(offBoardResult.valid).toBe(false);
        if (!offBoardResult.valid) {
          expect(offBoardResult.code).toBe('INVALID_POSITION');
        }
      });

      it('should respect hexagonal bounds for capture geometry', () => {
        const stacks = new Map([
          [
            '0,0,0',
            {
              controllingPlayer: 1,
              stackHeight: 2,
              capHeight: 2,
              rings: [1, 1],
              position: { x: 0, y: 0, z: 0 },
            },
          ],
          [
            '1,-1,0',
            {
              controllingPlayer: 2,
              stackHeight: 1,
              capHeight: 1,
              rings: [2],
              position: { x: 1, y: -1, z: 0 },
            },
          ],
        ]);
        // Size = bounding box = 2*radius + 1. size=7 means radius=3.
        const state = createMinimalState({
          currentPhase: 'movement',
          currentPlayer: 1,
          boardType: 'hexagonal',
          boardSize: 7,
          stacks,
        });

        const inBoundsAction = createCaptureAction({
          playerId: 1,
          from: { x: 0, y: 0, z: 0 },
          captureTarget: { x: 1, y: -1, z: 0 },
          to: { x: 2, y: -2, z: 0 },
        });
        const inBoundsResult = validateCapture(state, inBoundsAction);

        if (!inBoundsResult.valid) {
          expect(inBoundsResult.code).not.toBe('INVALID_POSITION');
        }

        const offBoardAction = createCaptureAction({
          playerId: 1,
          from: { x: 0, y: 0, z: 0 },
          captureTarget: { x: 1, y: -1, z: 0 },
          to: { x: 5, y: -5, z: 0 },
        });
        const offBoardResult = validateCapture(state, offBoardAction);

        expect(offBoardResult.valid).toBe(false);
        if (!offBoardResult.valid) {
          expect(offBoardResult.code).toBe('INVALID_POSITION');
        }
      });
    });
  });
});
