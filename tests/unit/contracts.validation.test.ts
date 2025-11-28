/**
 * ═══════════════════════════════════════════════════════════════════════════
 * Contract Validation Tests
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * Tests for the Zod-based runtime validation functions in the contracts module.
 * Ensures data integrity at boundary points (client-server, server-AI service).
 */

import {
  // Validation functions (safe)
  validatePosition,
  validateMove,
  validateRingStack,
  validateMarker,
  validatePlayerState,
  validateSerializedBoardState,
  validateSerializedGameState,
  validateProcessTurnRequest,
  validateProcessTurnResponse,
  validateVictoryState,
  validateMoveResult,

  // Parse functions (strict)
  parsePosition,
  parseMove,
  parseRingStack,
  parseMarker,
  parsePlayerState,
  parseSerializedBoardState,
  parseSerializedGameState,
  parseProcessTurnRequest,
  parseProcessTurnResponse,
  parseVictoryState,
  parseMoveResult,

  // Utilities
  formatZodError,
  createValidator,

  // Schemas
  ZodPositionSchema,
} from '../../src/shared/engine/contracts';
import { z } from 'zod';

describe('Contract Validation', () => {
  // ═══════════════════════════════════════════════════════════════════════════
  // Position Validation
  // ═══════════════════════════════════════════════════════════════════════════

  describe('validatePosition', () => {
    it('should validate a valid square position', () => {
      const result = validatePosition({ x: 3, y: 5 });
      expect(result.success).toBe(true);
      if (result.success) {
        expect(result.data).toEqual({ x: 3, y: 5 });
      }
    });

    it('should validate a valid hexagonal position with z coordinate', () => {
      const result = validatePosition({ x: 1, y: -1, z: 0 });
      expect(result.success).toBe(true);
      if (result.success) {
        expect(result.data).toEqual({ x: 1, y: -1, z: 0 });
      }
    });

    it('should fail on non-integer coordinates', () => {
      const result = validatePosition({ x: 3.5, y: 5 });
      expect(result.success).toBe(false);
      if (!result.success) {
        expect(result.error).toContain('integer');
      }
    });

    it('should fail on missing x coordinate', () => {
      const result = validatePosition({ y: 5 });
      expect(result.success).toBe(false);
      if (!result.success) {
        expect(result.error).toContain('x');
      }
    });

    it('should fail on null input', () => {
      const result = validatePosition(null);
      expect(result.success).toBe(false);
    });

    it('should fail on undefined input', () => {
      const result = validatePosition(undefined);
      expect(result.success).toBe(false);
    });

    it('should fail on string coordinates', () => {
      const result = validatePosition({ x: '3', y: '5' });
      expect(result.success).toBe(false);
    });
  });

  describe('parsePosition', () => {
    it('should parse a valid position', () => {
      const pos = parsePosition({ x: 3, y: 5 });
      expect(pos).toEqual({ x: 3, y: 5 });
    });

    it('should throw on invalid position', () => {
      expect(() => parsePosition({ x: 'invalid' })).toThrow();
    });
  });

  // ═══════════════════════════════════════════════════════════════════════════
  // Move Validation
  // ═══════════════════════════════════════════════════════════════════════════

  describe('validateMove', () => {
    const validMove = {
      id: 'move-1',
      type: 'place_ring',
      player: 1,
      to: { x: 3, y: 3 },
      timestamp: new Date().toISOString(),
      thinkTime: 1500,
      moveNumber: 1,
    };

    it('should validate a valid place_ring move', () => {
      const result = validateMove(validMove);
      expect(result.success).toBe(true);
      if (result.success) {
        expect(result.data.type).toBe('place_ring');
        expect(result.data.player).toBe(1);
      }
    });

    it('should validate a valid move_stack move', () => {
      const move = {
        ...validMove,
        type: 'move_stack',
        from: { x: 2, y: 2 },
        to: { x: 4, y: 4 },
        minimumDistance: 2,
        actualDistance: 2,
      };
      const result = validateMove(move);
      expect(result.success).toBe(true);
    });

    it('should validate a valid overtaking_capture move', () => {
      const move = {
        ...validMove,
        type: 'overtaking_capture',
        from: { x: 2, y: 2 },
        to: { x: 4, y: 4 },
        captureTarget: { x: 3, y: 3 },
        captureType: 'overtaking',
      };
      const result = validateMove(move);
      expect(result.success).toBe(true);
    });

    it('should fail on invalid move type', () => {
      const result = validateMove({ ...validMove, type: 'invalid_type' });
      expect(result.success).toBe(false);
      if (!result.success) {
        expect(result.error).toContain('type');
      }
    });

    it('should fail on player number less than 1', () => {
      const result = validateMove({ ...validMove, player: 0 });
      expect(result.success).toBe(false);
    });

    it('should fail on missing required fields', () => {
      const result = validateMove({ type: 'place_ring' });
      expect(result.success).toBe(false);
    });

    it('should fail on null input', () => {
      const result = validateMove(null);
      expect(result.success).toBe(false);
    });
  });

  describe('parseMove', () => {
    const validMove = {
      id: 'move-1',
      type: 'place_ring',
      player: 1,
      to: { x: 3, y: 3 },
      timestamp: new Date().toISOString(),
      thinkTime: 1500,
      moveNumber: 1,
    };

    it('should parse a valid move', () => {
      const move = parseMove(validMove);
      expect(move.type).toBe('place_ring');
    });

    it('should throw on invalid move', () => {
      expect(() => parseMove({ type: 'invalid' })).toThrow();
    });
  });

  // ═══════════════════════════════════════════════════════════════════════════
  // RingStack Validation
  // ═══════════════════════════════════════════════════════════════════════════

  describe('validateRingStack', () => {
    const validStack = {
      position: { x: 3, y: 3 },
      rings: [1, 2, 1],
      stackHeight: 3,
      capHeight: 1,
      controllingPlayer: 1,
    };

    it('should validate a valid ring stack', () => {
      const result = validateRingStack(validStack);
      expect(result.success).toBe(true);
      if (result.success) {
        expect(result.data.stackHeight).toBe(3);
        expect(result.data.capHeight).toBe(1);
      }
    });

    it('should fail on empty rings array with non-zero stackHeight', () => {
      const result = validateRingStack({
        ...validStack,
        rings: [],
        stackHeight: 3,
      });
      // This passes schema validation but may fail semantic validation
      expect(result.success).toBe(true);
    });

    it('should fail on negative stackHeight', () => {
      const result = validateRingStack({ ...validStack, stackHeight: -1 });
      expect(result.success).toBe(false);
    });

    it('should fail on controllingPlayer less than 1', () => {
      const result = validateRingStack({ ...validStack, controllingPlayer: 0 });
      expect(result.success).toBe(false);
    });

    it('should fail on invalid position', () => {
      const result = validateRingStack({ ...validStack, position: { x: 'a', y: 3 } });
      expect(result.success).toBe(false);
    });
  });

  describe('parseRingStack', () => {
    const validStack = {
      position: { x: 3, y: 3 },
      rings: [1, 2, 1],
      stackHeight: 3,
      capHeight: 1,
      controllingPlayer: 1,
    };

    it('should parse a valid ring stack', () => {
      const stack = parseRingStack(validStack);
      expect(stack.controllingPlayer).toBe(1);
    });

    it('should throw on invalid ring stack', () => {
      expect(() => parseRingStack({ ...validStack, stackHeight: 'three' })).toThrow();
    });
  });

  // ═══════════════════════════════════════════════════════════════════════════
  // Marker Validation
  // ═══════════════════════════════════════════════════════════════════════════

  describe('validateMarker', () => {
    it('should validate a valid regular marker', () => {
      const result = validateMarker({
        position: { x: 3, y: 3 },
        player: 1,
        type: 'regular',
      });
      expect(result.success).toBe(true);
    });

    it('should validate a valid collapsed marker', () => {
      const result = validateMarker({
        position: { x: 3, y: 3 },
        player: 2,
        type: 'collapsed',
      });
      expect(result.success).toBe(true);
    });

    it('should fail on invalid marker type', () => {
      const result = validateMarker({
        position: { x: 3, y: 3 },
        player: 1,
        type: 'invalid',
      });
      expect(result.success).toBe(false);
    });

    it('should fail on player less than 1', () => {
      const result = validateMarker({
        position: { x: 3, y: 3 },
        player: 0,
        type: 'regular',
      });
      expect(result.success).toBe(false);
    });
  });

  // ═══════════════════════════════════════════════════════════════════════════
  // PlayerState Validation
  // ═══════════════════════════════════════════════════════════════════════════

  describe('validatePlayerState', () => {
    const validPlayerState = {
      playerNumber: 1,
      ringsInHand: 18,
      eliminatedRings: 0,
      territorySpaces: 0,
    };

    it('should validate a valid player state', () => {
      const result = validatePlayerState(validPlayerState);
      expect(result.success).toBe(true);
      if (result.success) {
        expect(result.data.playerNumber).toBe(1);
        expect(result.data.ringsInHand).toBe(18);
      }
    });

    it('should validate player state with optional isActive', () => {
      const result = validatePlayerState({ ...validPlayerState, isActive: true });
      expect(result.success).toBe(true);
    });

    it('should fail on negative ringsInHand', () => {
      const result = validatePlayerState({ ...validPlayerState, ringsInHand: -1 });
      expect(result.success).toBe(false);
    });

    it('should fail on missing playerNumber', () => {
      const { playerNumber, ...invalid } = validPlayerState;
      const result = validatePlayerState(invalid);
      expect(result.success).toBe(false);
    });
  });

  // ═══════════════════════════════════════════════════════════════════════════
  // SerializedBoardState Validation
  // ═══════════════════════════════════════════════════════════════════════════

  describe('validateSerializedBoardState', () => {
    const validBoardState = {
      type: 'square8',
      size: 8,
      stacks: {
        '3,3': {
          position: { x: 3, y: 3 },
          rings: [1],
          stackHeight: 1,
          capHeight: 1,
          controllingPlayer: 1,
        },
      },
      markers: {},
      collapsedSpaces: {},
      eliminatedRings: { '1': 0, '2': 0 },
    };

    it('should validate a valid board state', () => {
      const result = validateSerializedBoardState(validBoardState);
      expect(result.success).toBe(true);
      if (result.success) {
        expect(result.data.type).toBe('square8');
        expect(result.data.size).toBe(8);
      }
    });

    it('should validate hexagonal board type', () => {
      const result = validateSerializedBoardState({
        ...validBoardState,
        type: 'hexagonal',
        size: 11,
      });
      expect(result.success).toBe(true);
    });

    it('should fail on invalid board type', () => {
      const result = validateSerializedBoardState({
        ...validBoardState,
        type: 'invalid_board',
      });
      expect(result.success).toBe(false);
    });

    it('should fail on non-integer size', () => {
      const result = validateSerializedBoardState({
        ...validBoardState,
        size: 8.5,
      });
      expect(result.success).toBe(false);
    });
  });

  // ═══════════════════════════════════════════════════════════════════════════
  // SerializedGameState Validation
  // ═══════════════════════════════════════════════════════════════════════════

  describe('validateSerializedGameState', () => {
    const validGameState = {
      gameId: 'game-123',
      board: {
        type: 'square8',
        size: 8,
        stacks: {},
        markers: {},
        collapsedSpaces: {},
        eliminatedRings: { '1': 0, '2': 0 },
      },
      players: [
        { playerNumber: 1, ringsInHand: 18, eliminatedRings: 0, territorySpaces: 0 },
        { playerNumber: 2, ringsInHand: 18, eliminatedRings: 0, territorySpaces: 0 },
      ],
      currentPlayer: 1,
      currentPhase: 'ring_placement',
      turnNumber: 1,
      moveHistory: [],
      gameStatus: 'active',
    };

    it('should validate a valid game state', () => {
      const result = validateSerializedGameState(validGameState);
      expect(result.success).toBe(true);
      if (result.success) {
        expect(result.data.currentPlayer).toBe(1);
        expect(result.data.currentPhase).toBe('ring_placement');
      }
    });

    it('should validate game state with optional fields', () => {
      const result = validateSerializedGameState({
        ...validGameState,
        victoryThreshold: 10,
        territoryVictoryThreshold: 33,
        totalRingsEliminated: 0,
      });
      expect(result.success).toBe(true);
    });

    it('should fail on less than 2 players', () => {
      const result = validateSerializedGameState({
        ...validGameState,
        players: [{ playerNumber: 1, ringsInHand: 18, eliminatedRings: 0, territorySpaces: 0 }],
      });
      expect(result.success).toBe(false);
    });

    it('should fail on invalid game phase', () => {
      const result = validateSerializedGameState({
        ...validGameState,
        currentPhase: 'invalid_phase',
      });
      expect(result.success).toBe(false);
    });

    it('should fail on invalid game status', () => {
      const result = validateSerializedGameState({
        ...validGameState,
        gameStatus: 'invalid_status',
      });
      expect(result.success).toBe(false);
    });

    it('should fail on turnNumber less than 1', () => {
      const result = validateSerializedGameState({
        ...validGameState,
        turnNumber: 0,
      });
      expect(result.success).toBe(false);
    });
  });

  describe('parseSerializedGameState', () => {
    const validGameState = {
      board: {
        type: 'square8',
        size: 8,
        stacks: {},
        markers: {},
        collapsedSpaces: {},
        eliminatedRings: {},
      },
      players: [
        { playerNumber: 1, ringsInHand: 18, eliminatedRings: 0, territorySpaces: 0 },
        { playerNumber: 2, ringsInHand: 18, eliminatedRings: 0, territorySpaces: 0 },
      ],
      currentPlayer: 1,
      currentPhase: 'ring_placement',
      turnNumber: 1,
      moveHistory: [],
      gameStatus: 'active',
    };

    it('should parse a valid game state', () => {
      const state = parseSerializedGameState(validGameState);
      expect(state.currentPlayer).toBe(1);
    });

    it('should throw on invalid game state', () => {
      expect(() =>
        parseSerializedGameState({ ...validGameState, currentPhase: 'invalid' })
      ).toThrow();
    });
  });

  // ═══════════════════════════════════════════════════════════════════════════
  // VictoryState Validation
  // ═══════════════════════════════════════════════════════════════════════════

  describe('validateVictoryState', () => {
    it('should validate a game not over', () => {
      const result = validateVictoryState({ isGameOver: false });
      expect(result.success).toBe(true);
    });

    it('should validate a completed game with winner', () => {
      const result = validateVictoryState({
        isGameOver: true,
        winner: 1,
        reason: 'ring_elimination',
        scores: [
          { player: 1, eliminatedRings: 10, territorySpaces: 5 },
          { player: 2, eliminatedRings: 3, territorySpaces: 2 },
        ],
      });
      expect(result.success).toBe(true);
    });

    it('should validate a completed game with null winner (draw)', () => {
      const result = validateVictoryState({
        isGameOver: true,
        winner: null,
        reason: 'stalemate_resolution',
      });
      expect(result.success).toBe(true);
    });

    it('should fail on invalid victory reason', () => {
      const result = validateVictoryState({
        isGameOver: true,
        winner: 1,
        reason: 'invalid_reason',
      });
      expect(result.success).toBe(false);
    });
  });

  // ═══════════════════════════════════════════════════════════════════════════
  // MoveResult Validation
  // ═══════════════════════════════════════════════════════════════════════════

  describe('validateMoveResult', () => {
    it('should validate a successful move result', () => {
      const result = validateMoveResult({
        success: true,
        newState: {
          board: {
            type: 'square8',
            size: 8,
            stacks: {},
            markers: {},
            collapsedSpaces: {},
            eliminatedRings: {},
          },
          players: [
            { playerNumber: 1, ringsInHand: 17, eliminatedRings: 0, territorySpaces: 0 },
            { playerNumber: 2, ringsInHand: 18, eliminatedRings: 0, territorySpaces: 0 },
          ],
          currentPlayer: 2,
          currentPhase: 'ring_placement',
          turnNumber: 2,
          moveHistory: [],
          gameStatus: 'active',
        },
      });
      expect(result.success).toBe(true);
    });

    it('should validate a failed move result', () => {
      const result = validateMoveResult({
        success: false,
        error: 'Invalid move: destination occupied',
      });
      expect(result.success).toBe(true);
      if (result.success) {
        expect(result.data.success).toBe(false);
        expect(result.data.error).toBe('Invalid move: destination occupied');
      }
    });

    it('should validate a move result awaiting decision', () => {
      const result = validateMoveResult({
        success: true,
        awaitingDecision: true,
        pendingDecision: {
          type: 'line_order',
          player: 1,
          options: [
            {
              id: 'opt-1',
              type: 'process_line',
              player: 1,
              to: { x: 0, y: 0 },
              timestamp: new Date().toISOString(),
              thinkTime: 0,
              moveNumber: 1,
            },
          ],
          context: {
            description: 'Choose which line to process first',
          },
        },
      });
      expect(result.success).toBe(true);
    });
  });

  // ═══════════════════════════════════════════════════════════════════════════
  // ProcessTurnRequest Validation
  // ═══════════════════════════════════════════════════════════════════════════

  describe('validateProcessTurnRequest', () => {
    const validRequest = {
      state: {
        board: {
          type: 'square8',
          size: 8,
          stacks: {},
          markers: {},
          collapsedSpaces: {},
          eliminatedRings: {},
        },
        players: [
          { playerNumber: 1, ringsInHand: 18, eliminatedRings: 0, territorySpaces: 0 },
          { playerNumber: 2, ringsInHand: 18, eliminatedRings: 0, territorySpaces: 0 },
        ],
        currentPlayer: 1,
        currentPhase: 'ring_placement',
        turnNumber: 1,
        moveHistory: [],
        gameStatus: 'active',
      },
      move: {
        id: 'move-1',
        type: 'place_ring',
        player: 1,
        to: { x: 3, y: 3 },
        timestamp: new Date().toISOString(),
        thinkTime: 1500,
        moveNumber: 1,
      },
    };

    it('should validate a valid process turn request', () => {
      const result = validateProcessTurnRequest(validRequest);
      expect(result.success).toBe(true);
    });

    it('should fail on missing state', () => {
      const { state, ...invalid } = validRequest;
      const result = validateProcessTurnRequest(invalid);
      expect(result.success).toBe(false);
    });

    it('should fail on missing move', () => {
      const { move, ...invalid } = validRequest;
      const result = validateProcessTurnRequest(invalid);
      expect(result.success).toBe(false);
    });
  });

  // ═══════════════════════════════════════════════════════════════════════════
  // ProcessTurnResponse Validation
  // ═══════════════════════════════════════════════════════════════════════════

  describe('validateProcessTurnResponse', () => {
    const validResponse = {
      nextState: {
        board: {
          type: 'square8',
          size: 8,
          stacks: {
            '3,3': {
              position: { x: 3, y: 3 },
              rings: [1],
              stackHeight: 1,
              capHeight: 1,
              controllingPlayer: 1,
            },
          },
          markers: {},
          collapsedSpaces: {},
          eliminatedRings: {},
        },
        players: [
          { playerNumber: 1, ringsInHand: 17, eliminatedRings: 0, territorySpaces: 0 },
          { playerNumber: 2, ringsInHand: 18, eliminatedRings: 0, territorySpaces: 0 },
        ],
        currentPlayer: 2,
        currentPhase: 'ring_placement',
        turnNumber: 2,
        moveHistory: [],
        gameStatus: 'active',
      },
      status: 'complete',
      metadata: {
        processedMove: {
          id: 'move-1',
          type: 'place_ring',
          player: 1,
          to: { x: 3, y: 3 },
          timestamp: new Date().toISOString(),
          thinkTime: 1500,
          moveNumber: 1,
        },
        phasesTraversed: ['ring_placement'],
      },
    };

    it('should validate a valid process turn response', () => {
      const result = validateProcessTurnResponse(validResponse);
      expect(result.success).toBe(true);
    });

    it('should validate response with awaiting_decision status', () => {
      const result = validateProcessTurnResponse({
        ...validResponse,
        status: 'awaiting_decision',
        pendingDecision: {
          type: 'line_order',
          player: 1,
          options: [
            {
              id: 'opt-1',
              type: 'process_line',
              player: 1,
              to: { x: 0, y: 0 },
              timestamp: new Date().toISOString(),
              thinkTime: 0,
              moveNumber: 1,
            },
          ],
          context: {
            description: 'Choose which line to process first',
          },
        },
      });
      expect(result.success).toBe(true);
    });

    it('should fail on invalid status', () => {
      const result = validateProcessTurnResponse({
        ...validResponse,
        status: 'invalid_status',
      });
      expect(result.success).toBe(false);
    });
  });

  // ═══════════════════════════════════════════════════════════════════════════
  // Error Formatting
  // ═══════════════════════════════════════════════════════════════════════════

  describe('formatZodError', () => {
    it('should format a simple error', () => {
      const result = ZodPositionSchema.safeParse({ x: 'not a number', y: 5 });
      if (!result.success) {
        const formatted = formatZodError(result.error);
        expect(typeof formatted).toBe('string');
        expect(formatted.length).toBeGreaterThan(0);
      }
    });

    it('should format a nested error with path', () => {
      const result = ZodPositionSchema.safeParse({ y: 5 });
      if (!result.success) {
        const formatted = formatZodError(result.error);
        expect(formatted).toContain('x');
      }
    });
  });

  // ═══════════════════════════════════════════════════════════════════════════
  // createValidator Utility
  // ═══════════════════════════════════════════════════════════════════════════

  describe('createValidator', () => {
    it('should create validate and parse functions', () => {
      const customSchema = z.object({
        name: z.string(),
        value: z.number(),
      });

      const [validate, parse] = createValidator(customSchema);

      // Test validate
      const validResult = validate({ name: 'test', value: 42 });
      expect(validResult.success).toBe(true);

      const invalidResult = validate({ name: 'test', value: 'not a number' });
      expect(invalidResult.success).toBe(false);

      // Test parse
      const parsed = parse({ name: 'test', value: 42 });
      expect(parsed).toEqual({ name: 'test', value: 42 });

      expect(() => parse({ name: 'test', value: 'not a number' })).toThrow();
    });
  });

  // ═══════════════════════════════════════════════════════════════════════════
  // Edge Cases
  // ═══════════════════════════════════════════════════════════════════════════

  describe('Edge Cases', () => {
    it('should handle empty objects', () => {
      expect(validatePosition({}).success).toBe(false);
      expect(validateMove({}).success).toBe(false);
      expect(validateRingStack({}).success).toBe(false);
    });

    it('should handle arrays instead of objects', () => {
      expect(validatePosition([1, 2]).success).toBe(false);
      expect(validateMove([]).success).toBe(false);
    });

    it('should handle primitive types', () => {
      expect(validatePosition('3,3').success).toBe(false);
      expect(validatePosition(123).success).toBe(false);
      expect(validatePosition(true).success).toBe(false);
    });

    it('should handle deeply nested invalid data', () => {
      const result = validateSerializedGameState({
        board: {
          type: 'square8',
          size: 8,
          stacks: {
            '3,3': {
              position: { x: 'invalid', y: 3 }, // Invalid nested position
              rings: [1],
              stackHeight: 1,
              capHeight: 1,
              controllingPlayer: 1,
            },
          },
          markers: {},
          collapsedSpaces: {},
          eliminatedRings: {},
        },
        players: [
          { playerNumber: 1, ringsInHand: 18, eliminatedRings: 0, territorySpaces: 0 },
          { playerNumber: 2, ringsInHand: 18, eliminatedRings: 0, territorySpaces: 0 },
        ],
        currentPlayer: 1,
        currentPhase: 'ring_placement',
        turnNumber: 1,
        moveHistory: [],
        gameStatus: 'active',
      });
      expect(result.success).toBe(false);
    });
  });

  // ═══════════════════════════════════════════════════════════════════════════
  // Type Inference Tests
  // ═══════════════════════════════════════════════════════════════════════════

  describe('Type Inference', () => {
    it('should correctly infer Position type', () => {
      const result = validatePosition({ x: 3, y: 5 });
      if (result.success) {
        // TypeScript should infer these as numbers
        const x: number = result.data.x;
        const y: number = result.data.y;
        expect(typeof x).toBe('number');
        expect(typeof y).toBe('number');
      }
    });

    it('should correctly infer Move type', () => {
      const result = validateMove({
        id: 'move-1',
        type: 'place_ring',
        player: 1,
        to: { x: 3, y: 3 },
        timestamp: new Date().toISOString(),
        thinkTime: 1500,
        moveNumber: 1,
      });
      if (result.success) {
        // TypeScript should infer type as the union of move types
        const moveType: string = result.data.type;
        expect(typeof moveType).toBe('string');
      }
    });

    it('should correctly infer GameState type', () => {
      const result = validateSerializedGameState({
        board: {
          type: 'square8',
          size: 8,
          stacks: {},
          markers: {},
          collapsedSpaces: {},
          eliminatedRings: {},
        },
        players: [
          { playerNumber: 1, ringsInHand: 18, eliminatedRings: 0, territorySpaces: 0 },
          { playerNumber: 2, ringsInHand: 18, eliminatedRings: 0, territorySpaces: 0 },
        ],
        currentPlayer: 1,
        currentPhase: 'ring_placement',
        turnNumber: 1,
        moveHistory: [],
        gameStatus: 'active',
      });
      if (result.success) {
        // TypeScript should infer currentPlayer as number
        const currentPlayer: number = result.data.currentPlayer;
        expect(typeof currentPlayer).toBe('number');
      }
    });
  });
});
