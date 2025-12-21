/**
 * contracts-validators.branchCoverage.test.ts
 *
 * Branch coverage tests for src/shared/engine/contracts/validators.ts
 * Tests Zod-based runtime validation for engine contracts.
 */

import {
  validatePosition,
  parsePosition,
  validateRingStack,
  parseRingStack,
  validateMarker,
  parseMarker,
  validatePlayerState,
  parsePlayerState,
  validateSerializedBoardState,
  parseSerializedBoardState,
  validateSerializedGameState,
  parseSerializedGameState,
  validateProcessTurnRequest,
  parseProcessTurnRequest,
  validateVictoryState,
  parseVictoryState,
  validateProcessTurnResponse,
  parseProcessTurnResponse,
  validateMoveResult,
  parseMoveResult,
  validateMove,
  parseMove,
  formatZodError,
  createValidator,
  ZodSchemas,
  ZodPositionSchema,
  ZodMoveTypeSchema,
  ZodRingStackSchema,
  ZodMarkerSchema,
  ZodPlayerStateSchema,
  ZodBoardTypeSchema,
  ZodGamePhaseSchema,
  ZodGameStatusSchema,
  ZodDecisionTypeSchema,
  ZodVictoryReasonSchema,
} from '../../src/shared/engine/contracts/validators';
import { z } from 'zod';

describe('contracts/validators.ts - Branch Coverage', () => {
  // ==========================================================================
  // Position Validation
  // ==========================================================================
  describe('Position validation', () => {
    it('validates valid square position', () => {
      const result = validatePosition({ x: 0, y: 0 });
      expect(result.success).toBe(true);
      if (result.success) {
        expect(result.data).toEqual({ x: 0, y: 0 });
      }
    });

    it('validates valid hexagonal position with z', () => {
      const result = validatePosition({ x: 1, y: -1, z: 0 });
      expect(result.success).toBe(true);
      if (result.success) {
        expect(result.data).toEqual({ x: 1, y: -1, z: 0 });
      }
    });

    it('rejects position with non-integer x', () => {
      const result = validatePosition({ x: 1.5, y: 0 });
      expect(result.success).toBe(false);
      if (!result.success) {
        expect(result.error).toContain('x');
      }
    });

    it('rejects position with missing x', () => {
      const result = validatePosition({ y: 0 });
      expect(result.success).toBe(false);
    });

    it('rejects null input', () => {
      const result = validatePosition(null);
      expect(result.success).toBe(false);
    });

    it('parsePosition returns valid data', () => {
      const parsed = parsePosition({ x: 5, y: 3 });
      expect(parsed).toEqual({ x: 5, y: 3 });
    });

    it('parsePosition throws on invalid data', () => {
      expect(() => parsePosition({ x: 'invalid' })).toThrow();
    });
  });

  // ==========================================================================
  // RingStack Validation
  // ==========================================================================
  describe('RingStack validation', () => {
    const validStack = {
      position: { x: 0, y: 0 },
      rings: [1, 2],
      stackHeight: 2,
      capHeight: 1,
      controllingPlayer: 1,
    };

    it('validates valid ring stack', () => {
      const result = validateRingStack(validStack);
      expect(result.success).toBe(true);
    });

    it('rejects ring stack with negative stackHeight', () => {
      const result = validateRingStack({ ...validStack, stackHeight: -1 });
      expect(result.success).toBe(false);
    });

    it('rejects ring stack with invalid player', () => {
      const result = validateRingStack({ ...validStack, controllingPlayer: 0 });
      expect(result.success).toBe(false);
    });

    it('rejects ring stack with invalid rings array', () => {
      const result = validateRingStack({ ...validStack, rings: [0] });
      expect(result.success).toBe(false);
    });

    it('parseRingStack returns valid data', () => {
      const parsed = parseRingStack(validStack);
      expect(parsed.controllingPlayer).toBe(1);
    });

    it('parseRingStack throws on invalid data', () => {
      expect(() => parseRingStack({ rings: 'invalid' })).toThrow();
    });
  });

  // ==========================================================================
  // Marker Validation
  // ==========================================================================
  describe('Marker validation', () => {
    const validMarker = {
      position: { x: 3, y: 4 },
      player: 1,
      type: 'regular' as const,
    };

    it('validates valid marker', () => {
      const result = validateMarker(validMarker);
      expect(result.success).toBe(true);
    });

    it('validates collapsed marker', () => {
      const result = validateMarker({ ...validMarker, type: 'collapsed' });
      expect(result.success).toBe(true);
    });

    it('validates departure marker', () => {
      const result = validateMarker({ ...validMarker, type: 'departure' });
      expect(result.success).toBe(true);
    });

    it('rejects marker with invalid type', () => {
      const result = validateMarker({ ...validMarker, type: 'invalid' });
      expect(result.success).toBe(false);
    });

    it('rejects marker with player 0', () => {
      const result = validateMarker({ ...validMarker, player: 0 });
      expect(result.success).toBe(false);
    });

    it('parseMarker returns valid data', () => {
      const parsed = parseMarker(validMarker);
      expect(parsed.type).toBe('regular');
    });

    it('parseMarker throws on invalid data', () => {
      expect(() => parseMarker({ type: 'unknown' })).toThrow();
    });
  });

  // ==========================================================================
  // PlayerState Validation
  // ==========================================================================
  describe('PlayerState validation', () => {
    const validPlayer = {
      playerNumber: 1,
      ringsInHand: 18,
      eliminatedRings: 0,
      territorySpaces: 0,
    };

    it('validates valid player state', () => {
      const result = validatePlayerState(validPlayer);
      expect(result.success).toBe(true);
    });

    it('validates player state with isActive', () => {
      const result = validatePlayerState({ ...validPlayer, isActive: true });
      expect(result.success).toBe(true);
    });

    it('rejects player with negative ringsInHand', () => {
      const result = validatePlayerState({ ...validPlayer, ringsInHand: -1 });
      expect(result.success).toBe(false);
    });

    it('rejects player with playerNumber 0', () => {
      const result = validatePlayerState({ ...validPlayer, playerNumber: 0 });
      expect(result.success).toBe(false);
    });

    it('parsePlayerState returns valid data', () => {
      const parsed = parsePlayerState(validPlayer);
      expect(parsed.ringsInHand).toBe(18);
    });

    it('parsePlayerState throws on invalid data', () => {
      expect(() => parsePlayerState({ playerNumber: 'one' })).toThrow();
    });
  });

  // ==========================================================================
  // SerializedBoardState Validation
  // ==========================================================================
  describe('SerializedBoardState validation', () => {
    const validBoard = {
      type: 'square8' as const,
      size: 8,
      stacks: {},
      markers: {},
      collapsedSpaces: {},
      eliminatedRings: {},
    };

    it('validates valid board state', () => {
      const result = validateSerializedBoardState(validBoard);
      expect(result.success).toBe(true);
    });

    it('validates board with stacks', () => {
      const result = validateSerializedBoardState({
        ...validBoard,
        stacks: {
          '0,0': {
            position: { x: 0, y: 0 },
            rings: [1],
            stackHeight: 1,
            capHeight: 1,
            controllingPlayer: 1,
          },
        },
      });
      expect(result.success).toBe(true);
    });

    it('validates hexagonal board type', () => {
      const result = validateSerializedBoardState({ ...validBoard, type: 'hexagonal' });
      expect(result.success).toBe(true);
    });

    it('validates square19 board type', () => {
      const result = validateSerializedBoardState({ ...validBoard, type: 'square19', size: 19 });
      expect(result.success).toBe(true);
    });

    it('rejects board with invalid type', () => {
      const result = validateSerializedBoardState({ ...validBoard, type: 'invalid' });
      expect(result.success).toBe(false);
    });

    it('rejects board with size 0', () => {
      const result = validateSerializedBoardState({ ...validBoard, size: 0 });
      expect(result.success).toBe(false);
    });

    it('parseSerializedBoardState returns valid data', () => {
      const parsed = parseSerializedBoardState(validBoard);
      expect(parsed.type).toBe('square8');
    });

    it('parseSerializedBoardState throws on invalid data', () => {
      expect(() => parseSerializedBoardState({ type: 'circle' })).toThrow();
    });
  });

  // ==========================================================================
  // Move Validation
  // ==========================================================================
  describe('Move validation', () => {
    const validMove = {
      id: 'move-1',
      type: 'place_ring' as const,
      player: 1,
      to: { x: 0, y: 0 },
      timestamp: new Date().toISOString(),
      thinkTime: 100,
      moveNumber: 1,
    };

    it('validates valid place_ring move', () => {
      const result = validateMove(validMove);
      expect(result.success).toBe(true);
    });

    it('validates move_stack move', () => {
      const result = validateMove({
        ...validMove,
        type: 'move_stack',
        from: { x: 0, y: 0 },
        to: { x: 1, y: 0 },
      });
      expect(result.success).toBe(true);
    });

    it('validates recovery_slide move', () => {
      const result = validateMove({ ...validMove, type: 'recovery_slide' });
      expect(result.success).toBe(true);
    });

    it('validates move with Date timestamp', () => {
      const result = validateMove({ ...validMove, timestamp: new Date() });
      expect(result.success).toBe(true);
    });

    it('validates move with optional capture fields', () => {
      const result = validateMove({
        ...validMove,
        type: 'overtaking_capture',
        captureType: 'overtaking',
        captureTarget: { x: 1, y: 1 },
        capturedStacks: [],
      });
      expect(result.success).toBe(true);
    });

    it('validates move with line formation fields (canonical type)', () => {
      const result = validateMove({
        ...validMove,
        type: 'choose_line_option', // Canonical type (line_formation is legacy)
        formedLines: [
          {
            positions: [{ x: 0, y: 0 }],
            player: 1,
            length: 5,
            direction: { x: 1, y: 0 },
          },
        ],
      });
      expect(result.success).toBe(true);
    });

    it('validates move with territory fields (canonical type)', () => {
      const result = validateMove({
        ...validMove,
        type: 'choose_territory_option', // Canonical type (territory_claim is legacy)
        claimedTerritory: [
          {
            spaces: [{ x: 0, y: 0 }],
            controllingPlayer: 1,
            isDisconnected: false,
          },
        ],
      });
      expect(result.success).toBe(true);
    });

    it('validates move with eliminatedRings', () => {
      const result = validateMove({
        ...validMove,
        type: 'eliminate_rings_from_stack',
        eliminatedRings: [{ player: 1, count: 2 }],
      });
      expect(result.success).toBe(true);
    });

    it('rejects move with invalid type', () => {
      const result = validateMove({ ...validMove, type: 'invalid_type' });
      expect(result.success).toBe(false);
    });

    it('rejects move with player 0', () => {
      const result = validateMove({ ...validMove, player: 0 });
      expect(result.success).toBe(false);
    });

    it('rejects move with negative thinkTime', () => {
      const result = validateMove({ ...validMove, thinkTime: -1 });
      expect(result.success).toBe(false);
    });

    it('parseMove returns valid data', () => {
      const parsed = parseMove(validMove);
      expect(parsed.type).toBe('place_ring');
    });

    it('parseMove throws on invalid data', () => {
      expect(() => parseMove({ type: 'bad' })).toThrow();
    });
  });

  // ==========================================================================
  // SerializedGameState Validation
  // ==========================================================================
  describe('SerializedGameState validation', () => {
    const validGameState = {
      board: {
        type: 'square8' as const,
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
      currentPhase: 'ring_placement' as const,
      turnNumber: 1,
      moveHistory: [],
      gameStatus: 'active' as const,
    };

    it('validates valid game state', () => {
      const result = validateSerializedGameState(validGameState);
      expect(result.success).toBe(true);
    });

    it('validates game state with optional gameId', () => {
      const result = validateSerializedGameState({ ...validGameState, gameId: 'game-123' });
      expect(result.success).toBe(true);
    });

    it('validates game state with thresholds', () => {
      const result = validateSerializedGameState({
        ...validGameState,
        victoryThreshold: 12,
        territoryVictoryThreshold: 10,
        totalRingsEliminated: 5,
      });
      expect(result.success).toBe(true);
    });

    it('validates all game phases', () => {
      const phases = [
        'ring_placement',
        'movement',
        'capture',
        'chain_capture',
        'line_processing',
        'territory_processing',
        'forced_elimination',
      ] as const;
      for (const phase of phases) {
        const result = validateSerializedGameState({ ...validGameState, currentPhase: phase });
        expect(result.success).toBe(true);
      }
    });

    it('validates all game statuses', () => {
      const statuses = [
        'waiting',
        'active',
        'finished',
        'paused',
        'abandoned',
        'completed',
      ] as const;
      for (const status of statuses) {
        const result = validateSerializedGameState({ ...validGameState, gameStatus: status });
        expect(result.success).toBe(true);
      }
    });

    it('rejects game state with fewer than 2 players', () => {
      const result = validateSerializedGameState({
        ...validGameState,
        players: [validGameState.players[0]],
      });
      expect(result.success).toBe(false);
    });

    it('rejects game state with invalid phase', () => {
      const result = validateSerializedGameState({
        ...validGameState,
        currentPhase: 'invalid_phase',
      });
      expect(result.success).toBe(false);
    });

    it('parseSerializedGameState returns valid data', () => {
      const parsed = parseSerializedGameState(validGameState);
      expect(parsed.currentPhase).toBe('ring_placement');
    });

    it('parseSerializedGameState throws on invalid data', () => {
      expect(() => parseSerializedGameState({ board: 'invalid' })).toThrow();
    });
  });

  // ==========================================================================
  // ProcessTurnRequest Validation
  // ==========================================================================
  describe('ProcessTurnRequest validation', () => {
    const validRequest = {
      state: {
        board: {
          type: 'square8' as const,
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
        currentPhase: 'ring_placement' as const,
        turnNumber: 1,
        moveHistory: [],
        gameStatus: 'active' as const,
      },
      move: {
        id: 'move-1',
        type: 'place_ring' as const,
        player: 1,
        to: { x: 0, y: 0 },
        timestamp: new Date().toISOString(),
        thinkTime: 100,
        moveNumber: 1,
      },
    };

    it('validates valid request', () => {
      const result = validateProcessTurnRequest(validRequest);
      expect(result.success).toBe(true);
    });

    it('rejects request with invalid state', () => {
      const result = validateProcessTurnRequest({ ...validRequest, state: { invalid: true } });
      expect(result.success).toBe(false);
    });

    it('rejects request with invalid move', () => {
      const result = validateProcessTurnRequest({ ...validRequest, move: { invalid: true } });
      expect(result.success).toBe(false);
    });

    it('parseProcessTurnRequest returns valid data', () => {
      const parsed = parseProcessTurnRequest(validRequest);
      expect(parsed.move.type).toBe('place_ring');
    });

    it('parseProcessTurnRequest throws on invalid data', () => {
      expect(() => parseProcessTurnRequest({ state: null })).toThrow();
    });
  });

  // ==========================================================================
  // VictoryState Validation
  // ==========================================================================
  describe('VictoryState validation', () => {
    it('validates game not over', () => {
      const result = validateVictoryState({ isGameOver: false });
      expect(result.success).toBe(true);
    });

    it('validates game over with winner', () => {
      const result = validateVictoryState({
        isGameOver: true,
        winner: 1,
        reason: 'ring_elimination',
      });
      expect(result.success).toBe(true);
    });

    it('validates all victory reasons', () => {
      const reasons = [
        'ring_elimination',
        'territory_control',
        'last_player_standing',
        'stalemate_resolution',
        'resignation',
      ] as const;
      for (const reason of reasons) {
        const result = validateVictoryState({ isGameOver: true, winner: 1, reason });
        expect(result.success).toBe(true);
      }
    });

    it('validates victory state with scores', () => {
      const result = validateVictoryState({
        isGameOver: true,
        winner: 1,
        reason: 'ring_elimination',
        scores: [
          {
            player: 1,
            eliminatedRings: 12,
            territorySpaces: 5,
            ringsOnBoard: 6,
            ringsInHand: 0,
            markerCount: 10,
            isEliminated: false,
          },
          {
            player: 2,
            eliminatedRings: 3,
            territorySpaces: 2,
          },
        ],
      });
      expect(result.success).toBe(true);
    });

    it('validates victory with null winner (draw)', () => {
      const result = validateVictoryState({ isGameOver: true, winner: null });
      expect(result.success).toBe(true);
    });

    it('parseVictoryState returns valid data', () => {
      const parsed = parseVictoryState({ isGameOver: false });
      expect(parsed.isGameOver).toBe(false);
    });

    it('parseVictoryState throws on invalid data', () => {
      expect(() => parseVictoryState({ isGameOver: 'yes' })).toThrow();
    });
  });

  // ==========================================================================
  // ProcessTurnResponse Validation
  // ==========================================================================
  describe('ProcessTurnResponse validation', () => {
    const validResponse = {
      nextState: {
        board: {
          type: 'square8' as const,
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
        currentPhase: 'ring_placement' as const,
        turnNumber: 1,
        moveHistory: [],
        gameStatus: 'active' as const,
      },
      status: 'complete' as const,
      metadata: {
        processedMove: {
          id: 'move-1',
          type: 'place_ring' as const,
          player: 1,
          to: { x: 0, y: 0 },
          timestamp: new Date().toISOString(),
          thinkTime: 100,
          moveNumber: 1,
        },
        phasesTraversed: ['ring_placement'],
      },
    };

    it('validates valid complete response', () => {
      const result = validateProcessTurnResponse(validResponse);
      expect(result.success).toBe(true);
    });

    it('validates response with awaiting_decision status', () => {
      const result = validateProcessTurnResponse({
        ...validResponse,
        status: 'awaiting_decision',
        pendingDecision: {
          type: 'line_reward',
          player: 1,
          options: [validResponse.metadata.processedMove],
          context: {
            description: 'Choose line reward',
            relevantPositions: [{ x: 0, y: 0 }],
          },
        },
      });
      expect(result.success).toBe(true);
    });

    it('validates all decision types', () => {
      const types = [
        'line_order',
        'line_reward',
        'region_order',
        'elimination_target',
        'capture_direction',
        'chain_capture',
      ] as const;
      for (const type of types) {
        const result = validateProcessTurnResponse({
          ...validResponse,
          status: 'awaiting_decision',
          pendingDecision: {
            type,
            player: 1,
            options: [validResponse.metadata.processedMove],
            context: { description: 'Choose' },
          },
        });
        expect(result.success).toBe(true);
      }
    });

    it('validates response with victoryResult', () => {
      const result = validateProcessTurnResponse({
        ...validResponse,
        victoryResult: { isGameOver: true, winner: 1, reason: 'ring_elimination' },
      });
      expect(result.success).toBe(true);
    });

    it('validates response with optional metadata fields', () => {
      const result = validateProcessTurnResponse({
        ...validResponse,
        metadata: {
          ...validResponse.metadata,
          linesDetected: 2,
          regionsProcessed: 1,
          durationMs: 50,
          sInvariantBefore: 36,
          sInvariantAfter: 35,
        },
      });
      expect(result.success).toBe(true);
    });

    it('parseProcessTurnResponse returns valid data', () => {
      const parsed = parseProcessTurnResponse(validResponse);
      expect(parsed.status).toBe('complete');
    });

    it('parseProcessTurnResponse throws on invalid data', () => {
      expect(() => parseProcessTurnResponse({ status: 'invalid' })).toThrow();
    });
  });

  // ==========================================================================
  // MoveResult Validation
  // ==========================================================================
  describe('MoveResult validation', () => {
    it('validates successful move result', () => {
      const result = validateMoveResult({ success: true });
      expect(result.success).toBe(true);
    });

    it('validates failed move result with error', () => {
      const result = validateMoveResult({ success: false, error: 'Invalid move' });
      expect(result.success).toBe(true);
    });

    it('validates move result with new state', () => {
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
            { playerNumber: 1, ringsInHand: 18, eliminatedRings: 0, territorySpaces: 0 },
            { playerNumber: 2, ringsInHand: 18, eliminatedRings: 0, territorySpaces: 0 },
          ],
          currentPlayer: 1,
          currentPhase: 'ring_placement',
          turnNumber: 1,
          moveHistory: [],
          gameStatus: 'active',
        },
      });
      expect(result.success).toBe(true);
    });

    it('validates move result with awaiting decision', () => {
      const result = validateMoveResult({
        success: true,
        awaitingDecision: true,
        pendingDecision: {
          type: 'line_reward',
          player: 1,
          options: [
            {
              id: 'opt-1',
              type: 'choose_line_option',
              player: 1,
              to: { x: 0, y: 0 },
              timestamp: new Date().toISOString(),
              thinkTime: 0,
              moveNumber: 1,
            },
          ],
          context: { description: 'Choose reward' },
        },
      });
      expect(result.success).toBe(true);
    });

    it('parseMoveResult returns valid data', () => {
      const parsed = parseMoveResult({ success: true });
      expect(parsed.success).toBe(true);
    });

    it('parseMoveResult throws on invalid data', () => {
      expect(() => parseMoveResult({ success: 'yes' })).toThrow();
    });
  });

  // ==========================================================================
  // formatZodError
  // ==========================================================================
  describe('formatZodError', () => {
    it('returns "Invalid data" for null', () => {
      expect(formatZodError(null)).toBe('Invalid data');
    });

    it('returns "Invalid data" for undefined', () => {
      expect(formatZodError(undefined)).toBe('Invalid data');
    });

    it('returns message for Error with message', () => {
      const err = new Error('Custom error message');
      expect(formatZodError(err as unknown as z.ZodError)).toBe('Custom error message');
    });

    it('formats Zod error with path', () => {
      const result = ZodPositionSchema.safeParse({ x: 'invalid', y: 0 });
      if (!result.success) {
        const formatted = formatZodError(result.error);
        expect(formatted).toContain('x');
      }
    });

    it('formats Zod error with multiple issues', () => {
      const result = ZodPositionSchema.safeParse({ x: 'a', y: 'b' });
      if (!result.success) {
        const formatted = formatZodError(result.error);
        expect(formatted).toContain('x');
        expect(formatted).toContain('y');
      }
    });

    it('handles empty issues array', () => {
      const mockError = { issues: [] } as unknown as z.ZodError;
      expect(formatZodError(mockError)).toBe('Invalid data');
    });

    it('handles issues with numeric path segments', () => {
      const schema = z.array(z.number());
      const result = schema.safeParse(['invalid']);
      if (!result.success) {
        const formatted = formatZodError(result.error);
        expect(formatted).toContain('0');
      }
    });
  });

  // ==========================================================================
  // createValidator
  // ==========================================================================
  describe('createValidator', () => {
    it('creates working validate function', () => {
      const schema = z.object({ name: z.string() });
      const [validate] = createValidator(schema);

      const valid = validate({ name: 'test' });
      expect(valid.success).toBe(true);

      const invalid = validate({ name: 123 });
      expect(invalid.success).toBe(false);
    });

    it('creates working parse function', () => {
      const schema = z.object({ value: z.number() });
      const [, parse] = createValidator(schema);

      expect(parse({ value: 42 })).toEqual({ value: 42 });
      expect(() => parse({ value: 'not a number' })).toThrow();
    });
  });

  // ==========================================================================
  // ZodSchemas export
  // ==========================================================================
  describe('ZodSchemas export', () => {
    it('exports all expected schemas', () => {
      expect(ZodSchemas.Position).toBeDefined();
      expect(ZodSchemas.MoveType).toBeDefined();
      expect(ZodSchemas.Move).toBeDefined();
      expect(ZodSchemas.RingStack).toBeDefined();
      expect(ZodSchemas.Marker).toBeDefined();
      expect(ZodSchemas.LineInfo).toBeDefined();
      expect(ZodSchemas.Territory).toBeDefined();
      expect(ZodSchemas.PlayerState).toBeDefined();
      expect(ZodSchemas.BoardType).toBeDefined();
      expect(ZodSchemas.SerializedBoardState).toBeDefined();
      expect(ZodSchemas.GamePhase).toBeDefined();
      expect(ZodSchemas.GameStatus).toBeDefined();
      expect(ZodSchemas.SerializedGameState).toBeDefined();
      expect(ZodSchemas.ProcessTurnRequest).toBeDefined();
      expect(ZodSchemas.DecisionType).toBeDefined();
      expect(ZodSchemas.PendingDecision).toBeDefined();
      expect(ZodSchemas.VictoryReason).toBeDefined();
      expect(ZodSchemas.VictoryState).toBeDefined();
      expect(ZodSchemas.ProcessingMetadata).toBeDefined();
      expect(ZodSchemas.ProcessTurnResponse).toBeDefined();
      expect(ZodSchemas.MoveResult).toBeDefined();
    });
  });

  // ==========================================================================
  // Enum schemas
  // ==========================================================================
  describe('Enum schemas', () => {
    it('ZodMoveTypeSchema validates all canonical move types', () => {
      // Canonical move types only - legacy types (build_stack, line_formation, territory_claim)
      // are intentionally excluded as they are deprecated per RULES_CANONICAL_SPEC
      const canonicalMoveTypes = [
        'place_ring',
        'move_stack',
        'skip_placement',
        'no_placement_action',
        'no_movement_action',
        'overtaking_capture',
        'continue_capture_segment',
        'skip_capture',
        'recovery_slide',
        'skip_recovery',
        'process_line',
        'choose_line_option',
        'no_line_action',
        'choose_territory_option',
        'skip_territory_processing',
        'no_territory_action',
        'eliminate_rings_from_stack',
        'forced_elimination',
        'swap_sides',
      ];
      for (const type of canonicalMoveTypes) {
        expect(ZodMoveTypeSchema.safeParse(type).success).toBe(true);
      }
    });

    it('ZodBoardTypeSchema validates all board types', () => {
      expect(ZodBoardTypeSchema.safeParse('square8').success).toBe(true);
      expect(ZodBoardTypeSchema.safeParse('square19').success).toBe(true);
      expect(ZodBoardTypeSchema.safeParse('hexagonal').success).toBe(true);
      expect(ZodBoardTypeSchema.safeParse('invalid').success).toBe(false);
    });

    it('ZodGamePhaseSchema validates all phases', () => {
      const phases = [
        'ring_placement',
        'movement',
        'capture',
        'chain_capture',
        'line_processing',
        'territory_processing',
        'forced_elimination',
      ];
      for (const phase of phases) {
        expect(ZodGamePhaseSchema.safeParse(phase).success).toBe(true);
      }
    });

    it('ZodGameStatusSchema validates all statuses', () => {
      const statuses = ['waiting', 'active', 'finished', 'paused', 'abandoned', 'completed'];
      for (const status of statuses) {
        expect(ZodGameStatusSchema.safeParse(status).success).toBe(true);
      }
    });

    it('ZodDecisionTypeSchema validates all decision types', () => {
      const types = [
        'line_order',
        'line_reward',
        'region_order',
        'elimination_target',
        'capture_direction',
        'chain_capture',
      ];
      for (const type of types) {
        expect(ZodDecisionTypeSchema.safeParse(type).success).toBe(true);
      }
    });

    it('ZodVictoryReasonSchema validates all victory reasons', () => {
      const reasons = [
        'ring_elimination',
        'territory_control',
        'last_player_standing',
        'stalemate_resolution',
        'resignation',
      ];
      for (const reason of reasons) {
        expect(ZodVictoryReasonSchema.safeParse(reason).success).toBe(true);
      }
    });
  });
});
