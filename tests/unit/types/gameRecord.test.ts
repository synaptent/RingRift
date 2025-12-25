/**
 * Test suite for src/shared/types/gameRecord.ts
 *
 * Tests game record types, RRN notation, coordinate utilities,
 * and move record conversion functions.
 */

import {
  moveToMoveRecord,
  gameRecordToJsonlLine,
  jsonlLineToGameRecord,
  positionToRRN,
  rrnToPosition,
  moveRecordToRRN,
  parseRRNMove,
  gameRecordToRRN,
  rrnToMoves,
  CoordinateUtils,
  type GameRecord,
  type MoveRecord,
  type PlayerRecordInfo,
  type GameRecordMetadata,
  type FinalScore,
} from '../../../src/shared/types/gameRecord';
import type { Move, Position, BoardType, LineInfo } from '../../../src/shared/types/game';

describe('gameRecord', () => {
  // Helper to create a test Move
  function createTestMove(overrides: Partial<Move> = {}): Move {
    return {
      id: 'test-move-1',
      type: 'place_ring',
      player: 1,
      to: { x: 3, y: 3 },
      timestamp: new Date(),
      thinkTime: 1000,
      moveNumber: 1,
      ...overrides,
    };
  }

  // Helper to create a test GameRecord
  function createTestGameRecord(): GameRecord {
    const players: PlayerRecordInfo[] = [
      {
        playerNumber: 1,
        username: 'Player1',
        playerType: 'human',
        ratingBefore: 1500,
      },
      {
        playerNumber: 2,
        username: 'Player2',
        playerType: 'ai',
        aiDifficulty: 5,
        aiType: 'minimax',
      },
    ];

    const finalScore: FinalScore = {
      ringsEliminated: { 1: 10, 2: 18 },
      territorySpaces: { 1: 15, 2: 10 },
      ringsRemaining: { 1: 8, 2: 0 },
    };

    const metadata: GameRecordMetadata = {
      recordVersion: '1.0.0',
      createdAt: new Date().toISOString(),
      source: 'online_game',
      tags: ['test', 'square8'],
    };

    const moves: MoveRecord[] = [
      { moveNumber: 1, player: 1, type: 'place_ring', to: { x: 0, y: 0 }, thinkTimeMs: 500 },
      { moveNumber: 2, player: 2, type: 'place_ring', to: { x: 7, y: 7 }, thinkTimeMs: 300 },
    ];

    return {
      id: 'test-game-123',
      boardType: 'square8',
      numPlayers: 2,
      rngSeed: 12345,
      isRated: true,
      players,
      winner: 1,
      outcome: 'ring_elimination',
      finalScore,
      startedAt: new Date().toISOString(),
      endedAt: new Date().toISOString(),
      totalMoves: 50,
      totalDurationMs: 300000,
      moves,
      metadata,
    };
  }

  describe('moveToMoveRecord', () => {
    it('should convert basic Move to MoveRecord', () => {
      const move = createTestMove();
      const record = moveToMoveRecord(move);

      expect(record.moveNumber).toBe(1);
      expect(record.player).toBe(1);
      expect(record.type).toBe('place_ring');
      expect(record.thinkTimeMs).toBe(1000);
      expect(record.to).toEqual({ x: 3, y: 3 });
    });

    it('should include from position when present', () => {
      const move = createTestMove({
        type: 'move_stack',
        from: { x: 0, y: 0 },
        to: { x: 2, y: 2 },
      });
      const record = moveToMoveRecord(move);

      expect(record.from).toEqual({ x: 0, y: 0 });
      expect(record.to).toEqual({ x: 2, y: 2 });
    });

    it('should include capture target when present', () => {
      const move = createTestMove({
        type: 'overtaking_capture',
        from: { x: 0, y: 0 },
        captureTarget: { x: 1, y: 1 },
        to: { x: 2, y: 2 },
      });
      const record = moveToMoveRecord(move);

      expect(record.captureTarget).toEqual({ x: 1, y: 1 });
    });

    it('should include placement metadata when present', () => {
      const move = createTestMove({
        placementCount: 3,
        placedOnStack: true,
      });
      const record = moveToMoveRecord(move);

      expect(record.placementCount).toBe(3);
      expect(record.placedOnStack).toBe(true);
    });

    it('should include line metadata when present', () => {
      const lineInfo: LineInfo = {
        positions: [
          { x: 0, y: 0 },
          { x: 1, y: 0 },
          { x: 2, y: 0 },
          { x: 3, y: 0 },
        ],
        player: 1,
        length: 4,
        direction: { x: 1, y: 0 },
      };
      const move = createTestMove({
        type: 'process_line',
        formedLines: [lineInfo],
        collapsedMarkers: [
          { x: 0, y: 0 },
          { x: 1, y: 0 },
        ],
      });
      const record = moveToMoveRecord(move);

      expect(record.formedLines).toHaveLength(1);
      expect(record.collapsedMarkers).toHaveLength(2);
    });

    it('should include territory metadata when present', () => {
      const move = createTestMove({
        type: 'choose_territory_option',
        disconnectedRegions: [
          { spaces: [{ x: 0, y: 0 }], controllingPlayer: 1, isDisconnected: true },
        ],
        eliminatedRings: [{ player: 1, count: 2 }],
      });
      const record = moveToMoveRecord(move);

      expect(record.disconnectedRegions).toHaveLength(1);
      expect(record.eliminatedRings).toHaveLength(1);
    });
  });

  describe('JSONL serialization', () => {
    it('should serialize GameRecord to JSONL line', () => {
      const record = createTestGameRecord();
      const jsonl = gameRecordToJsonlLine(record);

      expect(typeof jsonl).toBe('string');
      expect(jsonl).not.toContain('\n');
      const parsed = JSON.parse(jsonl);
      expect(parsed.id).toBe('test-game-123');
    });

    it('should deserialize JSONL line to GameRecord', () => {
      const original = createTestGameRecord();
      const jsonl = gameRecordToJsonlLine(original);
      const deserialized = jsonlLineToGameRecord(jsonl);

      expect(deserialized.id).toBe(original.id);
      expect(deserialized.boardType).toBe(original.boardType);
      expect(deserialized.numPlayers).toBe(original.numPlayers);
      expect(deserialized.outcome).toBe(original.outcome);
    });
  });

  describe('positionToRRN', () => {
    it('should convert square8 positions to algebraic notation', () => {
      expect(positionToRRN({ x: 0, y: 0 }, 'square8')).toBe('a1');
      expect(positionToRRN({ x: 7, y: 7 }, 'square8')).toBe('h8');
      expect(positionToRRN({ x: 3, y: 4 }, 'square8')).toBe('d5');
    });

    it('should convert square19 positions to algebraic notation', () => {
      expect(positionToRRN({ x: 0, y: 0 }, 'square19')).toBe('a1');
      expect(positionToRRN({ x: 18, y: 18 }, 'square19')).toBe('s19');
    });

    it('should convert hexagonal positions with z coordinate', () => {
      expect(positionToRRN({ x: 1, y: 2, z: -3 }, 'hexagonal')).toBe('(1,2,-3)');
    });

    it('should convert hexagonal positions without z coordinate', () => {
      expect(positionToRRN({ x: 1, y: 2 }, 'hexagonal')).toBe('(1,2)');
    });
  });

  describe('rrnToPosition', () => {
    it('should parse square8 algebraic notation', () => {
      expect(rrnToPosition('a1', 'square8')).toEqual({ x: 0, y: 0 });
      expect(rrnToPosition('h8', 'square8')).toEqual({ x: 7, y: 7 });
      expect(rrnToPosition('d5', 'square8')).toEqual({ x: 3, y: 4 });
    });

    it('should parse square19 algebraic notation', () => {
      expect(rrnToPosition('a1', 'square19')).toEqual({ x: 0, y: 0 });
      expect(rrnToPosition('s19', 'square19')).toEqual({ x: 18, y: 18 });
    });

    it('should parse hexagonal coordinates with z', () => {
      expect(rrnToPosition('(1,2,-3)', 'hexagonal')).toEqual({ x: 1, y: 2, z: -3 });
    });

    it('should parse hexagonal coordinates without z', () => {
      expect(rrnToPosition('(1,2)', 'hexagonal')).toEqual({ x: 1, y: 2 });
    });

    it('should throw for invalid hex coordinate', () => {
      expect(() => rrnToPosition('invalid', 'hexagonal')).toThrow();
      expect(() => rrnToPosition('(1)', 'hexagonal')).toThrow();
    });

    it('should throw for invalid square coordinate', () => {
      expect(() => rrnToPosition('a', 'square8')).toThrow();
    });
  });

  describe('moveRecordToRRN', () => {
    it('should convert place_ring move', () => {
      const record: MoveRecord = {
        moveNumber: 1,
        player: 1,
        type: 'place_ring',
        to: { x: 0, y: 0 },
        thinkTimeMs: 500,
      };
      expect(moveRecordToRRN(record, 'square8')).toBe('Pa1');
    });

    it('should convert place_ring with count', () => {
      const record: MoveRecord = {
        moveNumber: 1,
        player: 1,
        type: 'place_ring',
        to: { x: 0, y: 0 },
        placementCount: 3,
        thinkTimeMs: 500,
      };
      expect(moveRecordToRRN(record, 'square8')).toBe('Pa1x3');
    });

    it('should convert skip_placement', () => {
      const record: MoveRecord = {
        moveNumber: 1,
        player: 1,
        type: 'skip_placement',
        thinkTimeMs: 500,
      };
      expect(moveRecordToRRN(record, 'square8')).toBe('-');
    });

    it('should convert swap_sides', () => {
      const record: MoveRecord = {
        moveNumber: 1,
        player: 1,
        type: 'swap_sides',
        thinkTimeMs: 500,
      };
      expect(moveRecordToRRN(record, 'square8')).toBe('S');
    });

    it('should convert move_stack', () => {
      const record: MoveRecord = {
        moveNumber: 1,
        player: 1,
        type: 'move_stack',
        from: { x: 4, y: 3 },
        to: { x: 4, y: 5 },
        thinkTimeMs: 500,
      };
      expect(moveRecordToRRN(record, 'square8')).toBe('e4-e6');
    });

    it('should convert overtaking_capture', () => {
      const record: MoveRecord = {
        moveNumber: 1,
        player: 1,
        type: 'overtaking_capture',
        from: { x: 3, y: 3 },
        captureTarget: { x: 3, y: 4 },
        to: { x: 3, y: 5 },
        thinkTimeMs: 500,
      };
      expect(moveRecordToRRN(record, 'square8')).toBe('d4xd5-d6');
    });

    it('should convert continue_capture_segment', () => {
      const record: MoveRecord = {
        moveNumber: 1,
        player: 1,
        type: 'continue_capture_segment',
        from: { x: 3, y: 3 },
        captureTarget: { x: 3, y: 4 },
        to: { x: 3, y: 5 },
        thinkTimeMs: 500,
      };
      expect(moveRecordToRRN(record, 'square8')).toBe('d4xd5-d6+');
    });

    it('should convert process_line', () => {
      const record: MoveRecord = {
        moveNumber: 1,
        player: 1,
        type: 'process_line',
        formedLines: [
          {
            positions: [
              { x: 0, y: 2 },
              { x: 1, y: 2 },
            ],
            player: 1,
            length: 2,
            direction: { x: 1, y: 0 },
          },
        ],
        thinkTimeMs: 500,
      };
      expect(moveRecordToRRN(record, 'square8')).toBe('La3');
    });

    it('should handle process_line without lines', () => {
      const record: MoveRecord = {
        moveNumber: 1,
        player: 1,
        type: 'process_line',
        thinkTimeMs: 500,
      };
      expect(moveRecordToRRN(record, 'square8')).toBe('L?');
    });

    it('should convert choose_line_option (option 1)', () => {
      const record: MoveRecord = {
        moveNumber: 1,
        player: 1,
        type: 'choose_line_option',
        formedLines: [
          {
            positions: [
              { x: 0, y: 0 },
              { x: 1, y: 0 },
              { x: 2, y: 0 },
              { x: 3, y: 0 },
            ],
            player: 1,
            length: 4,
            direction: { x: 1, y: 0 },
          },
        ],
        collapsedMarkers: [
          { x: 0, y: 0 },
          { x: 1, y: 0 },
          { x: 2, y: 0 },
          { x: 3, y: 0 },
        ],
        thinkTimeMs: 500,
      };
      expect(moveRecordToRRN(record, 'square8')).toBe('O1');
    });

    it('should convert choose_line_option (option 2)', () => {
      const record: MoveRecord = {
        moveNumber: 1,
        player: 1,
        type: 'choose_line_option',
        formedLines: [
          {
            positions: [
              { x: 0, y: 0 },
              { x: 1, y: 0 },
              { x: 2, y: 0 },
              { x: 3, y: 0 },
              { x: 4, y: 0 },
            ],
            player: 1,
            length: 5,
            direction: { x: 1, y: 0 },
          },
        ],
        collapsedMarkers: [
          { x: 0, y: 0 },
          { x: 1, y: 0 },
          { x: 2, y: 0 },
          { x: 3, y: 0 },
        ],
        thinkTimeMs: 500,
      };
      expect(moveRecordToRRN(record, 'square8')).toBe('O2');
    });

    it('should handle choose_line_option without data', () => {
      const record: MoveRecord = {
        moveNumber: 1,
        player: 1,
        type: 'choose_line_option',
        thinkTimeMs: 500,
      };
      expect(moveRecordToRRN(record, 'square8')).toBe('O?');
    });

    it('should convert choose_territory_option', () => {
      const record: MoveRecord = {
        moveNumber: 1,
        player: 1,
        type: 'choose_territory_option',
        disconnectedRegions: [
          {
            spaces: [{ x: 1, y: 1 }],
            controllingPlayer: 1,
            isDisconnected: true,
          },
        ],
        thinkTimeMs: 500,
      };
      expect(moveRecordToRRN(record, 'square8')).toBe('Tb2');
    });

    it('should handle choose_territory_option without regions', () => {
      const record: MoveRecord = {
        moveNumber: 1,
        player: 1,
        type: 'choose_territory_option',
        thinkTimeMs: 500,
      };
      expect(moveRecordToRRN(record, 'square8')).toBe('T?');
    });

    it('should convert eliminate_rings_from_stack', () => {
      const record: MoveRecord = {
        moveNumber: 1,
        player: 1,
        type: 'eliminate_rings_from_stack',
        to: { x: 2, y: 2 },
        thinkTimeMs: 500,
      };
      expect(moveRecordToRRN(record, 'square8')).toBe('Ec3');
    });

    it('should throw for legacy move types', () => {
      const record: MoveRecord = {
        moveNumber: 1,
        player: 1,
        type: 'move_ring',
        thinkTimeMs: 500,
      };
      expect(() => moveRecordToRRN(record, 'square8')).toThrow('Legacy move type');
    });

    it('should return fallback for unknown move types', () => {
      const record: MoveRecord = {
        moveNumber: 1,
        player: 1,
        type: 'no_movement_action',
        thinkTimeMs: 500,
      };
      expect(moveRecordToRRN(record, 'square8')).toBe('?no_movement_action');
    });
  });

  describe('parseRRNMove', () => {
    it('should parse skip notation', () => {
      expect(parseRRNMove('-', 'square8')).toEqual({ moveType: 'skip_placement' });
    });

    it('should parse swap notation', () => {
      expect(parseRRNMove('S', 'square8')).toEqual({ moveType: 'swap_sides' });
    });

    it('should parse placement notation', () => {
      const result = parseRRNMove('Pa1', 'square8');
      expect(result.moveType).toBe('place_ring');
      expect(result.to).toEqual({ x: 0, y: 0 });
    });

    it('should parse placement with count', () => {
      const result = parseRRNMove('Pd5x3', 'square8');
      expect(result.moveType).toBe('place_ring');
      expect(result.to).toEqual({ x: 3, y: 4 });
    });

    it('should parse line notation', () => {
      const result = parseRRNMove('La3', 'square8');
      expect(result.moveType).toBe('process_line');
      expect(result.to).toEqual({ x: 0, y: 2 });
    });

    it('should parse territory notation', () => {
      const result = parseRRNMove('Tb2', 'square8');
      expect(result.moveType).toBe('choose_territory_option');
      expect(result.to).toEqual({ x: 1, y: 1 });
    });

    it('should parse elimination notation', () => {
      const result = parseRRNMove('Ec3', 'square8');
      expect(result.moveType).toBe('eliminate_rings_from_stack');
      expect(result.to).toEqual({ x: 2, y: 2 });
    });

    it('should parse O1 and O2 notation', () => {
      expect(parseRRNMove('O1', 'square8')).toEqual({ moveType: 'choose_line_option' });
      expect(parseRRNMove('O2', 'square8')).toEqual({ moveType: 'choose_line_option' });
    });

    it('should parse movement notation', () => {
      const result = parseRRNMove('e4-e6', 'square8');
      expect(result.moveType).toBe('move_stack');
      expect(result.from).toEqual({ x: 4, y: 3 });
      expect(result.to).toEqual({ x: 4, y: 5 });
    });

    it('should parse capture notation', () => {
      const result = parseRRNMove('d4xd5-d6', 'square8');
      expect(result.moveType).toBe('overtaking_capture');
      expect(result.from).toEqual({ x: 3, y: 3 });
      expect(result.to).toEqual({ x: 3, y: 5 });
    });

    it('should parse chain capture notation', () => {
      const result = parseRRNMove('d4xd5-d6+', 'square8');
      expect(result.moveType).toBe('continue_capture_segment');
    });

    it('should throw for legacy notation', () => {
      expect(() => parseRRNMove('a1>b2', 'square8')).toThrow('Legacy RRN notation');
    });

    it('should throw for unparseable notation', () => {
      expect(() => parseRRNMove('invalid', 'square8')).toThrow('Unable to parse');
    });
  });

  describe('gameRecordToRRN', () => {
    it('should convert full game record to RRN string', () => {
      const record = createTestGameRecord();
      const rrn = gameRecordToRRN(record);

      expect(rrn).toMatch(/^square8:2:12345:/);
      expect(rrn).toContain('Pa1');
      expect(rrn).toContain('Ph8');
    });

    it('should use underscore for missing seed', () => {
      const record = createTestGameRecord();
      delete record.rngSeed;
      const rrn = gameRecordToRRN(record);

      expect(rrn).toMatch(/^square8:2:_:/);
    });
  });

  describe('rrnToMoves', () => {
    it('should parse RRN string to moves', () => {
      const rrn = 'square8:2:12345:Pa1 Ph8 a1-a3';
      const result = rrnToMoves(rrn);

      expect(result.boardType).toBe('square8');
      expect(result.numPlayers).toBe(2);
      expect(result.rngSeed).toBe(12345);
      expect(result.moves).toHaveLength(3);
    });

    it('should handle missing seed (underscore)', () => {
      const rrn = 'square8:2:_:Pa1';
      const result = rrnToMoves(rrn);

      expect(result.rngSeed).toBeUndefined();
    });

    it('should throw for invalid format', () => {
      expect(() => rrnToMoves('invalid')).toThrow('Invalid RRN format');
    });
  });

  describe('CoordinateUtils', () => {
    describe('toAlgebraic and fromAlgebraic', () => {
      it('should be symmetrical for square boards', () => {
        const pos: Position = { x: 4, y: 5 };
        const algebraic = CoordinateUtils.toAlgebraic(pos, 'square8');
        const back = CoordinateUtils.fromAlgebraic(algebraic, 'square8');
        expect(back).toEqual(pos);
      });

      it('should be symmetrical for hex boards', () => {
        const pos: Position = { x: 1, y: 2, z: -3 };
        const algebraic = CoordinateUtils.toAlgebraic(pos, 'hexagonal');
        const back = CoordinateUtils.fromAlgebraic(algebraic, 'hexagonal');
        expect(back).toEqual(pos);
      });
    });

    describe('toKey and fromKey', () => {
      it('should convert position to key string', () => {
        expect(CoordinateUtils.toKey({ x: 3, y: 4 })).toBe('3,4');
        expect(CoordinateUtils.toKey({ x: 1, y: 2, z: -3 })).toBe('1,2,-3');
      });

      it('should parse key string back to position', () => {
        expect(CoordinateUtils.fromKey('3,4')).toEqual({ x: 3, y: 4 });
        expect(CoordinateUtils.fromKey('1,2,-3')).toEqual({ x: 1, y: 2, z: -3 });
      });
    });

    describe('getAllPositions', () => {
      it('should return 64 positions for square8', () => {
        const positions = CoordinateUtils.getAllPositions('square8');
        expect(positions).toHaveLength(64);
      });

      it('should return 361 positions for square19', () => {
        const positions = CoordinateUtils.getAllPositions('square19');
        expect(positions).toHaveLength(361);
      });

      it('should return positions for hexagonal board', () => {
        const positions = CoordinateUtils.getAllPositions('hexagonal');
        expect(positions.length).toBeGreaterThan(0);
        // All hex positions should have z coordinate
        positions.forEach((pos) => {
          expect(pos.z).toBeDefined();
        });
      });
    });

    describe('isValid', () => {
      it('should validate square8 positions', () => {
        expect(CoordinateUtils.isValid({ x: 0, y: 0 }, 'square8')).toBe(true);
        expect(CoordinateUtils.isValid({ x: 7, y: 7 }, 'square8')).toBe(true);
        expect(CoordinateUtils.isValid({ x: 8, y: 0 }, 'square8')).toBe(false);
        expect(CoordinateUtils.isValid({ x: -1, y: 0 }, 'square8')).toBe(false);
      });

      it('should validate square19 positions', () => {
        expect(CoordinateUtils.isValid({ x: 0, y: 0 }, 'square19')).toBe(true);
        expect(CoordinateUtils.isValid({ x: 18, y: 18 }, 'square19')).toBe(true);
        expect(CoordinateUtils.isValid({ x: 19, y: 0 }, 'square19')).toBe(false);
      });

      it('should validate hexagonal positions', () => {
        expect(CoordinateUtils.isValid({ x: 0, y: 0, z: 0 }, 'hexagonal')).toBe(true);
        expect(CoordinateUtils.isValid({ x: 1, y: -1, z: 0 }, 'hexagonal')).toBe(true);
        // Invalid: q + r + s != 0
        expect(CoordinateUtils.isValid({ x: 1, y: 1, z: 1 }, 'hexagonal')).toBe(false);
        // Out of radius
        expect(CoordinateUtils.isValid({ x: 20, y: 0, z: -20 }, 'hexagonal')).toBe(false);
      });

      it('should derive z for hexagonal validation', () => {
        // When z is not provided, it should be calculated as -x - y
        expect(CoordinateUtils.isValid({ x: 0, y: 0 }, 'hexagonal')).toBe(true);
        expect(CoordinateUtils.isValid({ x: 1, y: -1 }, 'hexagonal')).toBe(true);
      });
    });

    describe('distance', () => {
      it('should calculate Chebyshev distance for square boards', () => {
        expect(CoordinateUtils.distance({ x: 0, y: 0 }, { x: 3, y: 4 }, 'square8')).toBe(4);
        expect(CoordinateUtils.distance({ x: 2, y: 2 }, { x: 5, y: 6 }, 'square8')).toBe(4);
      });

      it('should calculate hex distance for hexagonal boards', () => {
        expect(
          CoordinateUtils.distance({ x: 0, y: 0, z: 0 }, { x: 2, y: -1, z: -1 }, 'hexagonal')
        ).toBe(2);
      });

      it('should derive z for hex distance calculation', () => {
        // z should be calculated as -x - y
        expect(CoordinateUtils.distance({ x: 0, y: 0 }, { x: 2, y: -1 }, 'hexagonal')).toBe(2);
      });
    });

    describe('getAdjacent', () => {
      it('should return 8 adjacent for center of square8', () => {
        const adjacent = CoordinateUtils.getAdjacent({ x: 4, y: 4 }, 'square8');
        expect(adjacent).toHaveLength(8);
      });

      it('should return 3 adjacent for corner of square8', () => {
        const adjacent = CoordinateUtils.getAdjacent({ x: 0, y: 0 }, 'square8');
        expect(adjacent).toHaveLength(3);
      });

      it('should return 5 adjacent for edge of square8', () => {
        const adjacent = CoordinateUtils.getAdjacent({ x: 4, y: 0 }, 'square8');
        expect(adjacent).toHaveLength(5);
      });

      it('should return 6 adjacent for center of hexagonal', () => {
        const adjacent = CoordinateUtils.getAdjacent({ x: 0, y: 0, z: 0 }, 'hexagonal');
        expect(adjacent).toHaveLength(6);
      });

      it('should filter out invalid hex positions', () => {
        // Edge position should have fewer neighbors
        const adjacent = CoordinateUtils.getAdjacent({ x: 11, y: -11, z: 0 }, 'hexagonal');
        expect(adjacent.length).toBeLessThan(6);
      });
    });
  });
});
