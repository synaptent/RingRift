import { z } from 'zod';
import { CreateGameSchema } from '../../../src/shared/validation/schemas';

describe('CreateGameSchema', () => {
  it('accepts a minimal valid payload and applies defaults', () => {
    const input = {
      boardType: 'square8',
      timeControl: {
        type: 'blitz',
        initialTime: 600,
        increment: 5,
      },
      isRated: true,
      isPrivate: false,
      maxPlayers: 2,
    };

    const parsed = CreateGameSchema.parse(input);

    expect(parsed.boardType).toBe('square8');
    expect(parsed.timeControl).toEqual({ type: 'blitz', initialTime: 600, increment: 5 });
    expect(parsed.isRated).toBe(true);
    expect(parsed.isPrivate).toBe(false);
    expect(parsed.maxPlayers).toBe(2);
    expect(parsed.aiOpponents).toBeUndefined();
  });

  it('applies defaults when optional fields are omitted', () => {
    const input = {
      boardType: 'square19',
      timeControl: {
        type: 'rapid',
        initialTime: 300,
        increment: 0,
      },
    } as any;

    const parsed = CreateGameSchema.parse(input);

    expect(parsed.boardType).toBe('square19');
    expect(parsed.isRated).toBe(true);
    expect(parsed.isPrivate).toBe(false);
    expect(parsed.maxPlayers).toBe(2);
  });

  it('validates aiOpponents structure and values', () => {
    const input = {
      boardType: 'hexagonal',
      timeControl: {
        type: 'classical',
        initialTime: 900,
        increment: 10,
      },
      isRated: false,
      isPrivate: true,
      maxPlayers: 4,
      aiOpponents: {
        count: 2,
        difficulty: [5, 7],
        mode: 'service',
        aiType: 'heuristic',
      },
    };

    const parsed = CreateGameSchema.parse(input);

    expect(parsed.aiOpponents).toEqual({
      count: 2,
      difficulty: [5, 7],
      mode: 'service',
      aiType: 'heuristic',
    });
  });

  it('rejects invalid payloads with clear errors', () => {
    const badInput = {
      boardType: 'invalid-board',
      timeControl: {
        type: 'invalid-type',
        initialTime: 10,
        increment: -1,
      },
      maxPlayers: 10,
      aiOpponents: {
        count: -1,
        difficulty: [0],
      },
    } as any;

    const result = CreateGameSchema.safeParse(badInput);
    expect(result.success).toBe(false);
    if (!result.success) {
      const issueCodes = result.error.issues.map((i) => i.code);
      // Zod uses 'invalid_value' for enum validation failures (not 'invalid_enum_value')
      expect(issueCodes).toContain('invalid_value');
      expect(issueCodes).toContain('too_small');
      expect(issueCodes).toContain('too_big');
    }
  });
});
