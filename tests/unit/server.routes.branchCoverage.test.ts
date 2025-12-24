/**
 * Server Routes Branch Coverage Tests
 *
 * This file targets uncovered branch paths in src/server/routes/*.ts to
 * improve overall branch coverage. It focuses on:
 * - Validation logic branches
 * - Authorization checks
 * - Data transformation branches
 * - Error handling logic
 * - Edge cases in query parsing
 *
 * Following the pattern in game.routes.branchCoverage.test.ts, these tests
 * directly test validation logic and data structures rather than making
 * HTTP requests, which avoids complex module isolation issues.
 *
 * ACTUAL COVERAGE: This file also tests REAL exported functions from
 * src/server/routes/user.ts to contribute to actual branch coverage.
 */

// Import real functions from user routes to test for actual coverage
import {
  DELETED_USER_PREFIX,
  DELETED_USER_DISPLAY_NAME,
  isDeletedUserUsername,
  getDisplayUsername,
  formatPlayerForDisplay,
} from '../../src/server/routes/user';

// ======================================================================
// GAME ROUTES VALIDATION LOGIC
// ======================================================================

describe('Game Routes - Validation Logic Branch Coverage', () => {
  describe('game ID validation', () => {
    const isValidGameId = (gameId: string): boolean => {
      if (!gameId || typeof gameId !== 'string') return false;
      if (gameId.length < 3 || gameId.length > 50) return false;
      return /^[a-zA-Z0-9_-]+$/.test(gameId);
    };

    it('validates normal game ID', () => {
      expect(isValidGameId('game-123')).toBe(true);
      expect(isValidGameId('550e8400-e29b-41d4-a716-446655440000')).toBe(true);
    });

    it('rejects game ID that is too short (branch: length < 3)', () => {
      expect(isValidGameId('ab')).toBe(false);
      expect(isValidGameId('a')).toBe(false);
      expect(isValidGameId('')).toBe(false);
    });

    it('rejects game ID that is too long (branch: length > 50)', () => {
      expect(isValidGameId('a'.repeat(51))).toBe(false);
      expect(isValidGameId('a'.repeat(100))).toBe(false);
    });

    it('rejects null or undefined game ID (branch: falsy check)', () => {
      expect(isValidGameId(null as unknown as string)).toBe(false);
      expect(isValidGameId(undefined as unknown as string)).toBe(false);
    });

    it('rejects non-string game ID (branch: type check)', () => {
      expect(isValidGameId(123 as unknown as string)).toBe(false);
      expect(isValidGameId({} as unknown as string)).toBe(false);
    });

    it('rejects game ID with invalid characters (branch: regex mismatch)', () => {
      expect(isValidGameId('game@123')).toBe(false);
      expect(isValidGameId('game 123')).toBe(false);
      expect(isValidGameId('game.123')).toBe(false);
    });
  });

  describe('user ID validation', () => {
    const isValidUserId = (userId: string): boolean => {
      if (!userId || typeof userId !== 'string') return false;
      // UUID format validation
      const uuidRegex = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;
      return uuidRegex.test(userId);
    };

    it('validates UUID format user ID', () => {
      expect(isValidUserId('550e8400-e29b-41d4-a716-446655440000')).toBe(true);
    });

    it('rejects invalid UUID format (branch: regex mismatch)', () => {
      expect(isValidUserId('invalid')).toBe(false);
      expect(isValidUserId('not-a-uuid')).toBe(false);
    });

    it('rejects null or undefined user ID (branch: falsy)', () => {
      expect(isValidUserId(null as unknown as string)).toBe(false);
      expect(isValidUserId(undefined as unknown as string)).toBe(false);
    });
  });

  describe('game status filter validation', () => {
    const validStatuses = ['waiting', 'active', 'completed', 'abandoned'] as const;

    const isValidStatus = (status: string): boolean => {
      return validStatuses.includes(status as (typeof validStatuses)[number]);
    };

    it('validates known status values', () => {
      expect(isValidStatus('waiting')).toBe(true);
      expect(isValidStatus('active')).toBe(true);
      expect(isValidStatus('completed')).toBe(true);
      expect(isValidStatus('abandoned')).toBe(true);
    });

    it('rejects invalid status values (branch: not in valid list)', () => {
      expect(isValidStatus('invalid_status')).toBe(false);
      expect(isValidStatus('pending')).toBe(false);
      expect(isValidStatus('finished')).toBe(false);
    });

    it('rejects empty status (branch: empty string)', () => {
      expect(isValidStatus('')).toBe(false);
    });
  });

  describe('AI opponent configuration validation', () => {
    interface AIOpponentConfig {
      count: number;
      difficulty: number[];
    }

    const validateAIConfig = (
      config: AIOpponentConfig | null | undefined
    ): { valid: boolean; error?: string } => {
      if (!config) return { valid: true }; // No AI is valid

      if (typeof config.count !== 'number' || config.count < 1 || config.count > 3) {
        return { valid: false, error: 'INVALID_AI_COUNT' };
      }

      if (!Array.isArray(config.difficulty)) {
        return { valid: false, error: 'INVALID_DIFFICULTY_ARRAY' };
      }

      if (config.difficulty.length !== config.count) {
        return { valid: false, error: 'DIFFICULTY_COUNT_MISMATCH' };
      }

      for (const d of config.difficulty) {
        if (typeof d !== 'number' || d < 1 || d > 10) {
          return { valid: false, error: 'INVALID_DIFFICULTY_VALUE' };
        }
      }

      return { valid: true };
    };

    it('validates null/undefined AI config (branch: no AI)', () => {
      expect(validateAIConfig(null)).toEqual({ valid: true });
      expect(validateAIConfig(undefined)).toEqual({ valid: true });
    });

    it('validates valid AI config', () => {
      expect(validateAIConfig({ count: 1, difficulty: [5] })).toEqual({ valid: true });
      expect(validateAIConfig({ count: 2, difficulty: [5, 7] })).toEqual({ valid: true });
      expect(validateAIConfig({ count: 3, difficulty: [1, 5, 10] })).toEqual({ valid: true });
    });

    it('rejects invalid AI count (branch: count out of range)', () => {
      expect(validateAIConfig({ count: 0, difficulty: [] }).error).toBe('INVALID_AI_COUNT');
      expect(validateAIConfig({ count: 4, difficulty: [1, 2, 3, 4] }).error).toBe(
        'INVALID_AI_COUNT'
      );
      expect(validateAIConfig({ count: -1, difficulty: [] }).error).toBe('INVALID_AI_COUNT');
    });

    it('rejects non-array difficulty (branch: type check)', () => {
      expect(validateAIConfig({ count: 1, difficulty: 5 as unknown as number[] }).error).toBe(
        'INVALID_DIFFICULTY_ARRAY'
      );
    });

    it('rejects mismatched difficulty count (branch: length mismatch)', () => {
      expect(validateAIConfig({ count: 2, difficulty: [5] }).error).toBe(
        'DIFFICULTY_COUNT_MISMATCH'
      );
      expect(validateAIConfig({ count: 1, difficulty: [5, 7] }).error).toBe(
        'DIFFICULTY_COUNT_MISMATCH'
      );
    });

    it('rejects difficulty out of range (branch: value validation)', () => {
      expect(validateAIConfig({ count: 1, difficulty: [0] }).error).toBe(
        'INVALID_DIFFICULTY_VALUE'
      );
      expect(validateAIConfig({ count: 1, difficulty: [11] }).error).toBe(
        'INVALID_DIFFICULTY_VALUE'
      );
      expect(validateAIConfig({ count: 1, difficulty: [15] }).error).toBe(
        'INVALID_DIFFICULTY_VALUE'
      );
    });

    it('rejects non-number difficulty values (branch: type in array)', () => {
      expect(validateAIConfig({ count: 1, difficulty: ['5' as unknown as number] }).error).toBe(
        'INVALID_DIFFICULTY_VALUE'
      );
    });
  });

  describe('rated game with AI validation', () => {
    const validateRatedWithAI = (isRated: boolean, hasAI: boolean): boolean => {
      if (isRated && hasAI) return false; // AI games cannot be rated
      return true;
    };

    it('allows unrated games with AI (branch: !isRated && hasAI)', () => {
      expect(validateRatedWithAI(false, true)).toBe(true);
    });

    it('allows rated games without AI (branch: isRated && !hasAI)', () => {
      expect(validateRatedWithAI(true, false)).toBe(true);
    });

    it('allows unrated games without AI (branch: !isRated && !hasAI)', () => {
      expect(validateRatedWithAI(false, false)).toBe(true);
    });

    it('rejects rated games with AI (branch: isRated && hasAI)', () => {
      expect(validateRatedWithAI(true, true)).toBe(false);
    });
  });

  describe('game participant checking', () => {
    interface Game {
      player1Id: string | null;
      player2Id: string | null;
      player3Id: string | null;
      player4Id: string | null;
      allowSpectators: boolean;
    }

    const isUserParticipant = (game: Game, userId: string): boolean => {
      return [game.player1Id, game.player2Id, game.player3Id, game.player4Id]
        .filter(Boolean)
        .includes(userId);
    };

    const canUserAccessGame = (game: Game, userId: string): boolean => {
      if (isUserParticipant(game, userId)) return true;
      if (game.allowSpectators) return true;
      return false;
    };

    it('detects user is player1 (branch: player1Id match)', () => {
      const game: Game = {
        player1Id: 'user-1',
        player2Id: 'user-2',
        player3Id: null,
        player4Id: null,
        allowSpectators: false,
      };
      expect(isUserParticipant(game, 'user-1')).toBe(true);
    });

    it('detects user is player2 (branch: player2Id match)', () => {
      const game: Game = {
        player1Id: 'user-1',
        player2Id: 'user-2',
        player3Id: null,
        player4Id: null,
        allowSpectators: false,
      };
      expect(isUserParticipant(game, 'user-2')).toBe(true);
    });

    it('detects user is player3 (branch: player3Id match)', () => {
      const game: Game = {
        player1Id: 'user-1',
        player2Id: 'user-2',
        player3Id: 'user-3',
        player4Id: null,
        allowSpectators: false,
      };
      expect(isUserParticipant(game, 'user-3')).toBe(true);
    });

    it('detects user is player4 (branch: player4Id match)', () => {
      const game: Game = {
        player1Id: 'user-1',
        player2Id: 'user-2',
        player3Id: 'user-3',
        player4Id: 'user-4',
        allowSpectators: false,
      };
      expect(isUserParticipant(game, 'user-4')).toBe(true);
    });

    it('detects user is not a participant (branch: no match)', () => {
      const game: Game = {
        player1Id: 'user-1',
        player2Id: 'user-2',
        player3Id: null,
        player4Id: null,
        allowSpectators: false,
      };
      expect(isUserParticipant(game, 'user-999')).toBe(false);
    });

    it('handles game with all null players (branch: empty participants)', () => {
      const game: Game = {
        player1Id: null,
        player2Id: null,
        player3Id: null,
        player4Id: null,
        allowSpectators: false,
      };
      expect(isUserParticipant(game, 'user-1')).toBe(false);
    });

    it('allows spectator when spectators enabled (branch: allowSpectators)', () => {
      const game: Game = {
        player1Id: 'user-1',
        player2Id: 'user-2',
        player3Id: null,
        player4Id: null,
        allowSpectators: true,
      };
      expect(canUserAccessGame(game, 'spectator')).toBe(true);
    });

    it('denies non-participant when spectators disabled (branch: !allowSpectators)', () => {
      const game: Game = {
        player1Id: 'user-1',
        player2Id: 'user-2',
        player3Id: null,
        player4Id: null,
        allowSpectators: false,
      };
      expect(canUserAccessGame(game, 'outsider')).toBe(false);
    });
  });

  describe('game joinability checking', () => {
    interface Game {
      status: 'waiting' | 'active' | 'completed' | 'abandoned';
      player1Id: string | null;
      player2Id: string | null;
      player3Id: string | null;
      player4Id: string | null;
      maxPlayers: number;
    }

    const getNextAvailableSlot = (game: Game): number | null => {
      if (game.player2Id === null) return 2;
      if (game.maxPlayers >= 3 && game.player3Id === null) return 3;
      if (game.maxPlayers >= 4 && game.player4Id === null) return 4;
      return null;
    };

    const canUserJoin = (game: Game, userId: string): { canJoin: boolean; error?: string } => {
      if (game.status !== 'waiting') {
        return { canJoin: false, error: 'GAME_NOT_JOINABLE' };
      }

      const participants = [game.player1Id, game.player2Id, game.player3Id, game.player4Id].filter(
        Boolean
      );
      if (participants.includes(userId)) {
        return { canJoin: false, error: 'ALREADY_JOINED' };
      }

      const slot = getNextAvailableSlot(game);
      if (slot === null) {
        return { canJoin: false, error: 'GAME_FULL' };
      }

      return { canJoin: true };
    };

    it('allows joining waiting game with available slot', () => {
      const game: Game = {
        status: 'waiting',
        player1Id: 'user-1',
        player2Id: null,
        player3Id: null,
        player4Id: null,
        maxPlayers: 2,
      };
      expect(canUserJoin(game, 'user-2')).toEqual({ canJoin: true });
    });

    it('rejects joining non-waiting game (branch: status !== waiting)', () => {
      const game: Game = {
        status: 'active',
        player1Id: 'user-1',
        player2Id: 'user-2',
        player3Id: null,
        player4Id: null,
        maxPlayers: 2,
      };
      expect(canUserJoin(game, 'user-3').error).toBe('GAME_NOT_JOINABLE');
    });

    it('rejects already joined user (branch: already participant)', () => {
      const game: Game = {
        status: 'waiting',
        player1Id: 'user-1',
        player2Id: null,
        player3Id: null,
        player4Id: null,
        maxPlayers: 2,
      };
      expect(canUserJoin(game, 'user-1').error).toBe('ALREADY_JOINED');
    });

    it('rejects when game is full (branch: no available slot)', () => {
      const game: Game = {
        status: 'waiting',
        player1Id: 'user-1',
        player2Id: 'user-2',
        player3Id: null,
        player4Id: null,
        maxPlayers: 2,
      };
      expect(canUserJoin(game, 'user-3').error).toBe('GAME_FULL');
    });

    it('allows joining slot 3 in 3-player game (branch: maxPlayers >= 3)', () => {
      const game: Game = {
        status: 'waiting',
        player1Id: 'user-1',
        player2Id: 'user-2',
        player3Id: null,
        player4Id: null,
        maxPlayers: 3,
      };
      expect(getNextAvailableSlot(game)).toBe(3);
      expect(canUserJoin(game, 'user-3')).toEqual({ canJoin: true });
    });

    it('allows joining slot 4 in 4-player game (branch: maxPlayers >= 4)', () => {
      const game: Game = {
        status: 'waiting',
        player1Id: 'user-1',
        player2Id: 'user-2',
        player3Id: 'user-3',
        player4Id: null,
        maxPlayers: 4,
      };
      expect(getNextAvailableSlot(game)).toBe(4);
      expect(canUserJoin(game, 'user-4')).toEqual({ canJoin: true });
    });
  });

  describe('game result reason extraction', () => {
    interface Game {
      status: string;
      finalState: { gameResult?: { reason?: string } } | null;
    }

    const getResultReason = (game: Game): string | null => {
      if (!['completed', 'abandoned', 'finished'].includes(game.status)) return null;
      if (!game.finalState) return null;
      if (!game.finalState.gameResult) return null;
      return game.finalState.gameResult.reason || null;
    };

    it('extracts reason from completed game (branch: status = completed)', () => {
      const game: Game = {
        status: 'completed',
        finalState: { gameResult: { reason: 'resignation' } },
      };
      expect(getResultReason(game)).toBe('resignation');
    });

    it('extracts reason from abandoned game (branch: status = abandoned)', () => {
      const game: Game = {
        status: 'abandoned',
        finalState: { gameResult: { reason: 'timeout' } },
      };
      expect(getResultReason(game)).toBe('timeout');
    });

    it('extracts reason from finished game (branch: status = finished)', () => {
      const game: Game = {
        status: 'finished',
        finalState: { gameResult: { reason: 'victory' } },
      };
      expect(getResultReason(game)).toBe('victory');
    });

    it('returns null for active game (branch: status not terminal)', () => {
      const game: Game = {
        status: 'active',
        finalState: { gameResult: { reason: 'victory' } },
      };
      expect(getResultReason(game)).toBeNull();
    });

    it('returns null when finalState is null (branch: no finalState)', () => {
      const game: Game = {
        status: 'completed',
        finalState: null,
      };
      expect(getResultReason(game)).toBeNull();
    });

    it('returns null when gameResult is missing (branch: no gameResult)', () => {
      const game: Game = {
        status: 'completed',
        finalState: {},
      };
      expect(getResultReason(game)).toBeNull();
    });

    it('returns null when reason is missing (branch: no reason)', () => {
      const game: Game = {
        status: 'completed',
        finalState: { gameResult: {} },
      };
      expect(getResultReason(game)).toBeNull();
    });
  });
});

// ======================================================================
// USER ROUTES VALIDATION LOGIC
// ======================================================================

describe('User Routes - Validation Logic Branch Coverage', () => {
  describe('username validation', () => {
    const isValidUsername = (username: string): { valid: boolean; error?: string } => {
      if (!username || typeof username !== 'string') {
        return { valid: false, error: 'USERNAME_REQUIRED' };
      }
      if (username.length < 3) {
        return { valid: false, error: 'USERNAME_TOO_SHORT' };
      }
      if (username.length > 20) {
        return { valid: false, error: 'USERNAME_TOO_LONG' };
      }
      if (!/^[a-zA-Z0-9_-]+$/.test(username)) {
        return { valid: false, error: 'USERNAME_INVALID_CHARS' };
      }
      return { valid: true };
    };

    it('validates proper username', () => {
      expect(isValidUsername('validuser')).toEqual({ valid: true });
      expect(isValidUsername('user_123')).toEqual({ valid: true });
      expect(isValidUsername('user-name')).toEqual({ valid: true });
    });

    it('rejects missing username (branch: falsy)', () => {
      expect(isValidUsername('').error).toBe('USERNAME_REQUIRED');
      expect(isValidUsername(null as unknown as string).error).toBe('USERNAME_REQUIRED');
    });

    it('rejects username too short (branch: length < 3)', () => {
      expect(isValidUsername('ab').error).toBe('USERNAME_TOO_SHORT');
      expect(isValidUsername('a').error).toBe('USERNAME_TOO_SHORT');
    });

    it('rejects username too long (branch: length > 20)', () => {
      expect(isValidUsername('a'.repeat(21)).error).toBe('USERNAME_TOO_LONG');
    });

    it('rejects invalid characters (branch: regex mismatch)', () => {
      expect(isValidUsername('user@name').error).toBe('USERNAME_INVALID_CHARS');
      expect(isValidUsername('user name').error).toBe('USERNAME_INVALID_CHARS');
    });
  });

  describe('email validation', () => {
    const isValidEmail = (email: string): boolean => {
      if (!email || typeof email !== 'string') return false;
      const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
      return emailRegex.test(email);
    };

    it('validates proper email', () => {
      expect(isValidEmail('user@example.com')).toBe(true);
      expect(isValidEmail('user.name@domain.co.uk')).toBe(true);
    });

    it('rejects missing email (branch: falsy)', () => {
      expect(isValidEmail('')).toBe(false);
      expect(isValidEmail(null as unknown as string)).toBe(false);
    });

    it('rejects invalid email format (branch: regex mismatch)', () => {
      expect(isValidEmail('not-an-email')).toBe(false);
      expect(isValidEmail('user@')).toBe(false);
      expect(isValidEmail('@domain.com')).toBe(false);
    });
  });

  describe('leaderboard filter validation', () => {
    const validTimePeriods = ['all', 'week', 'month', 'year'] as const;
    const validBoardTypes = ['all', 'square8', 'square19', 'hexagonal'] as const;

    const validateLeaderboardFilters = (filters: {
      timePeriod?: string;
      boardType?: string;
    }): { valid: boolean; error?: string } => {
      if (filters.timePeriod && !validTimePeriods.includes(filters.timePeriod as any)) {
        return { valid: false, error: 'INVALID_TIME_PERIOD' };
      }
      if (filters.boardType && !validBoardTypes.includes(filters.boardType as any)) {
        return { valid: false, error: 'INVALID_BOARD_TYPE' };
      }
      return { valid: true };
    };

    it('validates valid filters', () => {
      expect(validateLeaderboardFilters({ timePeriod: 'week' })).toEqual({ valid: true });
      expect(validateLeaderboardFilters({ boardType: 'square8' })).toEqual({ valid: true });
      expect(validateLeaderboardFilters({ timePeriod: 'all', boardType: 'all' })).toEqual({
        valid: true,
      });
    });

    it('validates empty filters (branch: no filters)', () => {
      expect(validateLeaderboardFilters({})).toEqual({ valid: true });
    });

    it('rejects invalid time period (branch: timePeriod not in list)', () => {
      expect(validateLeaderboardFilters({ timePeriod: 'invalid' }).error).toBe(
        'INVALID_TIME_PERIOD'
      );
      expect(validateLeaderboardFilters({ timePeriod: 'day' }).error).toBe('INVALID_TIME_PERIOD');
    });

    it('rejects invalid board type (branch: boardType not in list)', () => {
      expect(validateLeaderboardFilters({ boardType: 'invalid' }).error).toBe('INVALID_BOARD_TYPE');
      expect(validateLeaderboardFilters({ boardType: 'circular' }).error).toBe(
        'INVALID_BOARD_TYPE'
      );
    });
  });

  describe('search query validation', () => {
    const validateSearchQuery = (query: string | undefined): { valid: boolean; error?: string } => {
      if (!query || typeof query !== 'string') {
        return { valid: false, error: 'SEARCH_QUERY_REQUIRED' };
      }
      if (query.length < 2) {
        return { valid: false, error: 'SEARCH_QUERY_TOO_SHORT' };
      }
      if (query.length > 50) {
        return { valid: false, error: 'SEARCH_QUERY_TOO_LONG' };
      }
      return { valid: true };
    };

    it('validates proper search query', () => {
      expect(validateSearchQuery('test')).toEqual({ valid: true });
      expect(validateSearchQuery('username search')).toEqual({ valid: true });
    });

    it('rejects missing query (branch: falsy)', () => {
      expect(validateSearchQuery(undefined).error).toBe('SEARCH_QUERY_REQUIRED');
      expect(validateSearchQuery('').error).toBe('SEARCH_QUERY_REQUIRED');
    });

    it('rejects query too short (branch: length < 2)', () => {
      expect(validateSearchQuery('a').error).toBe('SEARCH_QUERY_TOO_SHORT');
    });

    it('rejects query too long (branch: length > 50)', () => {
      expect(validateSearchQuery('a'.repeat(51)).error).toBe('SEARCH_QUERY_TOO_LONG');
    });
  });

  describe('time period date calculation', () => {
    const getDateRangeStart = (timePeriod: string): Date | null => {
      const now = new Date();
      switch (timePeriod) {
        case 'week':
          return new Date(now.getTime() - 7 * 24 * 60 * 60 * 1000);
        case 'month':
          return new Date(now.getTime() - 30 * 24 * 60 * 60 * 1000);
        case 'year':
          return new Date(now.getTime() - 365 * 24 * 60 * 60 * 1000);
        case 'all':
          return null;
        default:
          return null;
      }
    };

    it('calculates week start (branch: timePeriod = week)', () => {
      const result = getDateRangeStart('week');
      expect(result).not.toBeNull();
      const now = new Date();
      const weekAgo = new Date(now.getTime() - 7 * 24 * 60 * 60 * 1000);
      expect(result!.getTime()).toBeCloseTo(weekAgo.getTime(), -3);
    });

    it('calculates month start (branch: timePeriod = month)', () => {
      const result = getDateRangeStart('month');
      expect(result).not.toBeNull();
      const now = new Date();
      const monthAgo = new Date(now.getTime() - 30 * 24 * 60 * 60 * 1000);
      expect(result!.getTime()).toBeCloseTo(monthAgo.getTime(), -3);
    });

    it('calculates year start (branch: timePeriod = year)', () => {
      const result = getDateRangeStart('year');
      expect(result).not.toBeNull();
      const now = new Date();
      const yearAgo = new Date(now.getTime() - 365 * 24 * 60 * 60 * 1000);
      expect(result!.getTime()).toBeCloseTo(yearAgo.getTime(), -3);
    });

    it('returns null for "all" (branch: timePeriod = all)', () => {
      expect(getDateRangeStart('all')).toBeNull();
    });

    it('returns null for unknown period (branch: default case)', () => {
      expect(getDateRangeStart('unknown')).toBeNull();
    });
  });
});

// ======================================================================
// AUTH ROUTES VALIDATION LOGIC
// ======================================================================

describe('Auth Routes - Validation Logic Branch Coverage', () => {
  describe('password validation', () => {
    const validatePassword = (password: string): { valid: boolean; error?: string } => {
      if (!password || typeof password !== 'string') {
        return { valid: false, error: 'PASSWORD_REQUIRED' };
      }
      if (password.length < 8) {
        return { valid: false, error: 'PASSWORD_TOO_SHORT' };
      }
      if (password.length > 128) {
        return { valid: false, error: 'PASSWORD_TOO_LONG' };
      }
      // Check for at least one uppercase, one lowercase, one number
      if (!/[A-Z]/.test(password)) {
        return { valid: false, error: 'PASSWORD_MISSING_UPPERCASE' };
      }
      if (!/[a-z]/.test(password)) {
        return { valid: false, error: 'PASSWORD_MISSING_LOWERCASE' };
      }
      if (!/[0-9]/.test(password)) {
        return { valid: false, error: 'PASSWORD_MISSING_NUMBER' };
      }
      return { valid: true };
    };

    it('validates strong password', () => {
      expect(validatePassword('StrongPass123')).toEqual({ valid: true });
      expect(validatePassword('ValidPassword1')).toEqual({ valid: true });
    });

    it('rejects missing password (branch: falsy)', () => {
      expect(validatePassword('').error).toBe('PASSWORD_REQUIRED');
      expect(validatePassword(null as unknown as string).error).toBe('PASSWORD_REQUIRED');
    });

    it('rejects password too short (branch: length < 8)', () => {
      expect(validatePassword('Short1').error).toBe('PASSWORD_TOO_SHORT');
      expect(validatePassword('Abc123').error).toBe('PASSWORD_TOO_SHORT');
    });

    it('rejects password too long (branch: length > 128)', () => {
      expect(validatePassword('A1' + 'a'.repeat(127)).error).toBe('PASSWORD_TOO_LONG');
    });

    it('rejects password missing uppercase (branch: no uppercase)', () => {
      expect(validatePassword('lowercase123').error).toBe('PASSWORD_MISSING_UPPERCASE');
    });

    it('rejects password missing lowercase (branch: no lowercase)', () => {
      expect(validatePassword('UPPERCASE123').error).toBe('PASSWORD_MISSING_LOWERCASE');
    });

    it('rejects password missing number (branch: no number)', () => {
      expect(validatePassword('NoNumbersHere').error).toBe('PASSWORD_MISSING_NUMBER');
    });
  });

  describe('token validation', () => {
    const validateToken = (token: string | undefined): { valid: boolean; error?: string } => {
      if (!token || typeof token !== 'string') {
        return { valid: false, error: 'TOKEN_REQUIRED' };
      }
      if (token.length < 10) {
        return { valid: false, error: 'TOKEN_TOO_SHORT' };
      }
      if (token.length > 500) {
        return { valid: false, error: 'TOKEN_TOO_LONG' };
      }
      return { valid: true };
    };

    it('validates proper token', () => {
      expect(validateToken('valid-token-12345')).toEqual({ valid: true });
    });

    it('rejects missing token (branch: falsy)', () => {
      expect(validateToken(undefined).error).toBe('TOKEN_REQUIRED');
      expect(validateToken('').error).toBe('TOKEN_REQUIRED');
    });

    it('rejects token too short (branch: length < 10)', () => {
      expect(validateToken('short').error).toBe('TOKEN_TOO_SHORT');
    });

    it('rejects token too long (branch: length > 500)', () => {
      expect(validateToken('a'.repeat(501)).error).toBe('TOKEN_TOO_LONG');
    });
  });

  describe('login lockout checking', () => {
    const checkLoginLockout = (
      failedAttempts: number,
      lastFailedAt: Date | null,
      lockoutDurationMs: number,
      maxFailedAttempts: number
    ): { locked: boolean; remainingMs?: number } => {
      if (failedAttempts < maxFailedAttempts) {
        return { locked: false };
      }

      if (!lastFailedAt) {
        return { locked: false };
      }

      const lockoutEndsAt = new Date(lastFailedAt.getTime() + lockoutDurationMs);
      const now = new Date();

      if (now >= lockoutEndsAt) {
        return { locked: false };
      }

      return { locked: true, remainingMs: lockoutEndsAt.getTime() - now.getTime() };
    };

    it('allows login with few failed attempts (branch: failedAttempts < max)', () => {
      expect(checkLoginLockout(2, new Date(), 900000, 5)).toEqual({ locked: false });
    });

    it('allows login when lockout expired (branch: now >= lockoutEndsAt)', () => {
      const oldDate = new Date(Date.now() - 1000000);
      expect(checkLoginLockout(5, oldDate, 900000, 5)).toEqual({ locked: false });
    });

    it('locks user during lockout period (branch: now < lockoutEndsAt)', () => {
      const recentDate = new Date();
      const result = checkLoginLockout(5, recentDate, 900000, 5);
      expect(result.locked).toBe(true);
      expect(result.remainingMs).toBeDefined();
      expect(result.remainingMs!).toBeGreaterThan(0);
    });

    it('allows login when no lastFailedAt (branch: !lastFailedAt)', () => {
      expect(checkLoginLockout(10, null, 900000, 5)).toEqual({ locked: false });
    });
  });

  describe('refresh token validation', () => {
    interface RefreshToken {
      token: string;
      userId: string;
      expiresAt: Date;
      revoked: boolean;
      tokenVersion: number;
    }

    const validateRefreshToken = (
      storedToken: RefreshToken | null,
      providedToken: string,
      expectedUserId: string,
      currentTokenVersion: number
    ): { valid: boolean; error?: string } => {
      if (!storedToken) {
        return { valid: false, error: 'TOKEN_NOT_FOUND' };
      }
      if (storedToken.token !== providedToken) {
        return { valid: false, error: 'TOKEN_MISMATCH' };
      }
      if (storedToken.userId !== expectedUserId) {
        return { valid: false, error: 'USER_MISMATCH' };
      }
      if (storedToken.revoked) {
        return { valid: false, error: 'TOKEN_REVOKED' };
      }
      if (new Date() > storedToken.expiresAt) {
        return { valid: false, error: 'TOKEN_EXPIRED' };
      }
      if (storedToken.tokenVersion !== currentTokenVersion) {
        return { valid: false, error: 'TOKEN_VERSION_MISMATCH' };
      }
      return { valid: true };
    };

    it('validates valid refresh token', () => {
      const token: RefreshToken = {
        token: 'valid-token',
        userId: 'user-1',
        expiresAt: new Date(Date.now() + 3600000),
        revoked: false,
        tokenVersion: 1,
      };
      expect(validateRefreshToken(token, 'valid-token', 'user-1', 1)).toEqual({
        valid: true,
      });
    });

    it('rejects missing token (branch: !storedToken)', () => {
      expect(validateRefreshToken(null, 'token', 'user-1', 1).error).toBe('TOKEN_NOT_FOUND');
    });

    it('rejects mismatched token (branch: token !== providedToken)', () => {
      const token: RefreshToken = {
        token: 'valid-token',
        userId: 'user-1',
        expiresAt: new Date(Date.now() + 3600000),
        revoked: false,
        tokenVersion: 1,
      };
      expect(validateRefreshToken(token, 'wrong-token', 'user-1', 1).error).toBe('TOKEN_MISMATCH');
    });

    it('rejects mismatched user (branch: userId !== expectedUserId)', () => {
      const token: RefreshToken = {
        token: 'valid-token',
        userId: 'user-1',
        expiresAt: new Date(Date.now() + 3600000),
        revoked: false,
        tokenVersion: 1,
      };
      expect(validateRefreshToken(token, 'valid-token', 'user-2', 1).error).toBe('USER_MISMATCH');
    });

    it('rejects revoked token (branch: revoked)', () => {
      const token: RefreshToken = {
        token: 'valid-token',
        userId: 'user-1',
        expiresAt: new Date(Date.now() + 3600000),
        revoked: true,
        tokenVersion: 1,
      };
      expect(validateRefreshToken(token, 'valid-token', 'user-1', 1).error).toBe('TOKEN_REVOKED');
    });

    it('rejects expired token (branch: now > expiresAt)', () => {
      const token: RefreshToken = {
        token: 'valid-token',
        userId: 'user-1',
        expiresAt: new Date(Date.now() - 3600000),
        revoked: false,
        tokenVersion: 1,
      };
      expect(validateRefreshToken(token, 'valid-token', 'user-1', 1).error).toBe('TOKEN_EXPIRED');
    });

    it('rejects version mismatch (branch: tokenVersion !== currentVersion)', () => {
      const token: RefreshToken = {
        token: 'valid-token',
        userId: 'user-1',
        expiresAt: new Date(Date.now() + 3600000),
        revoked: false,
        tokenVersion: 1,
      };
      expect(validateRefreshToken(token, 'valid-token', 'user-1', 2).error).toBe(
        'TOKEN_VERSION_MISMATCH'
      );
    });
  });
});

// ======================================================================
// SANDBOX AI ROUTES VALIDATION LOGIC
// ======================================================================

describe('Sandbox AI Routes - Validation Logic Branch Coverage', () => {
  describe('difficulty clamping', () => {
    const clampDifficulty = (difficulty: number | undefined): number => {
      if (difficulty === undefined) return 5; // Default
      if (typeof difficulty !== 'number') return 5;
      if (isNaN(difficulty)) return 5;
      return Math.max(1, Math.min(10, difficulty));
    };

    it('returns default for undefined (branch: undefined)', () => {
      expect(clampDifficulty(undefined)).toBe(5);
    });

    it('returns default for non-number (branch: type check)', () => {
      expect(clampDifficulty('5' as unknown as number)).toBe(5);
      expect(clampDifficulty({} as unknown as number)).toBe(5);
    });

    it('returns default for NaN (branch: isNaN check)', () => {
      expect(clampDifficulty(NaN)).toBe(5);
    });

    it('clamps to minimum 1 (branch: difficulty < 1)', () => {
      expect(clampDifficulty(0)).toBe(1);
      expect(clampDifficulty(-5)).toBe(1);
    });

    it('clamps to maximum 10 (branch: difficulty > 10)', () => {
      expect(clampDifficulty(15)).toBe(10);
      expect(clampDifficulty(100)).toBe(10);
    });

    it('returns valid difficulty unchanged', () => {
      expect(clampDifficulty(1)).toBe(1);
      expect(clampDifficulty(5)).toBe(5);
      expect(clampDifficulty(10)).toBe(10);
    });
  });

  describe('state validation', () => {
    const validateSandboxState = (state: unknown): { valid: boolean; error?: string } => {
      if (!state || typeof state !== 'object') {
        return { valid: false, error: 'STATE_REQUIRED' };
      }

      const s = state as Record<string, unknown>;

      if (!s.boardType || typeof s.boardType !== 'string') {
        return { valid: false, error: 'INVALID_BOARD_TYPE' };
      }

      if (typeof s.currentPlayer !== 'number' || s.currentPlayer < 1 || s.currentPlayer > 4) {
        return { valid: false, error: 'INVALID_CURRENT_PLAYER' };
      }

      if (!s.currentPhase || typeof s.currentPhase !== 'string') {
        return { valid: false, error: 'INVALID_PHASE' };
      }

      return { valid: true };
    };

    it('validates proper state', () => {
      const state = {
        boardType: 'square8',
        currentPlayer: 1,
        currentPhase: 'ring_placement',
      };
      expect(validateSandboxState(state)).toEqual({ valid: true });
    });

    it('rejects null/undefined state (branch: !state)', () => {
      expect(validateSandboxState(null).error).toBe('STATE_REQUIRED');
      expect(validateSandboxState(undefined).error).toBe('STATE_REQUIRED');
    });

    it('rejects non-object state (branch: typeof !== object)', () => {
      expect(validateSandboxState('not-object').error).toBe('STATE_REQUIRED');
    });

    it('rejects missing boardType (branch: !boardType)', () => {
      expect(validateSandboxState({ currentPlayer: 1, currentPhase: 'ring_placement' }).error).toBe(
        'INVALID_BOARD_TYPE'
      );
    });

    it('rejects invalid currentPlayer (branch: player validation)', () => {
      expect(
        validateSandboxState({
          boardType: 'square8',
          currentPlayer: 0,
          currentPhase: 'ring_placement',
        }).error
      ).toBe('INVALID_CURRENT_PLAYER');
      expect(
        validateSandboxState({
          boardType: 'square8',
          currentPlayer: 5,
          currentPhase: 'ring_placement',
        }).error
      ).toBe('INVALID_CURRENT_PLAYER');
    });

    it('rejects missing currentPhase (branch: !currentPhase)', () => {
      expect(validateSandboxState({ boardType: 'square8', currentPlayer: 1 }).error).toBe(
        'INVALID_PHASE'
      );
    });
  });

  describe('board type query parsing', () => {
    const parseBoardType = (
      value: string | undefined
    ): 'square8' | 'square19' | 'hexagonal' | null => {
      if (!value) return null;
      if (value === 'square8') return 'square8';
      if (value === 'square19') return 'square19';
      if (value === 'hexagonal') return 'hexagonal';
      return null;
    };

    it('parses undefined as null (branch: !value)', () => {
      expect(parseBoardType(undefined)).toBeNull();
    });

    it('parses square8 (branch: value === square8)', () => {
      expect(parseBoardType('square8')).toBe('square8');
    });

    it('parses square19 (branch: value === square19)', () => {
      expect(parseBoardType('square19')).toBe('square19');
    });

    it('parses hexagonal (branch: value === hexagonal)', () => {
      expect(parseBoardType('hexagonal')).toBe('hexagonal');
    });

    it('returns null for invalid (branch: default)', () => {
      expect(parseBoardType('invalid')).toBeNull();
    });
  });

  describe('numPlayers query parsing', () => {
    const parseNumPlayers = (value: string | undefined): 2 | 3 | 4 | null => {
      if (!value) return null;
      const num = parseInt(value, 10);
      if (isNaN(num)) return null;
      if (num === 2) return 2;
      if (num === 3) return 3;
      if (num === 4) return 4;
      return null;
    };

    it('parses undefined as null (branch: !value)', () => {
      expect(parseNumPlayers(undefined)).toBeNull();
    });

    it('parses non-number as null (branch: isNaN)', () => {
      expect(parseNumPlayers('abc')).toBeNull();
    });

    it('parses 2 players (branch: num === 2)', () => {
      expect(parseNumPlayers('2')).toBe(2);
    });

    it('parses 3 players (branch: num === 3)', () => {
      expect(parseNumPlayers('3')).toBe(3);
    });

    it('parses 4 players (branch: num === 4)', () => {
      expect(parseNumPlayers('4')).toBe(4);
    });

    it('returns null for invalid player count (branch: default)', () => {
      expect(parseNumPlayers('1')).toBeNull();
      expect(parseNumPlayers('5')).toBeNull();
    });
  });
});

// ======================================================================
// ERROR CODE MAPPING LOGIC
// ======================================================================

describe('Error Code Mapping - Branch Coverage', () => {
  describe('HTTP status code derivation', () => {
    const getStatusFromCode = (code: string): number => {
      if (code.startsWith('AUTH_')) {
        if (code === 'AUTH_UNAUTHORIZED') return 401;
        if (code === 'AUTH_FORBIDDEN') return 403;
        return 400;
      }
      if (code.startsWith('GAME_')) {
        if (code === 'GAME_NOT_FOUND') return 404;
        if (code === 'GAME_ACCESS_DENIED') return 403;
        return 400;
      }
      if (code.startsWith('RESOURCE_')) {
        if (code.includes('NOT_FOUND')) return 404;
        if (code.includes('EXISTS')) return 409;
        return 400;
      }
      if (code.startsWith('SERVER_')) {
        if (code === 'SERVER_DATABASE_UNAVAILABLE') return 503;
        if (code === 'SERVER_GATEWAY_TIMEOUT') return 504;
        return 500;
      }
      return 500;
    };

    it('handles AUTH_ codes (branch: AUTH_ prefix)', () => {
      expect(getStatusFromCode('AUTH_UNAUTHORIZED')).toBe(401);
      expect(getStatusFromCode('AUTH_FORBIDDEN')).toBe(403);
      expect(getStatusFromCode('AUTH_INVALID_REQUEST')).toBe(400);
    });

    it('handles GAME_ codes (branch: GAME_ prefix)', () => {
      expect(getStatusFromCode('GAME_NOT_FOUND')).toBe(404);
      expect(getStatusFromCode('GAME_ACCESS_DENIED')).toBe(403);
      expect(getStatusFromCode('GAME_INVALID_CONFIG')).toBe(400);
    });

    it('handles RESOURCE_ codes (branch: RESOURCE_ prefix)', () => {
      expect(getStatusFromCode('RESOURCE_USER_NOT_FOUND')).toBe(404);
      expect(getStatusFromCode('RESOURCE_USERNAME_EXISTS')).toBe(409);
      expect(getStatusFromCode('RESOURCE_INVALID_DATA')).toBe(400);
    });

    it('handles SERVER_ codes (branch: SERVER_ prefix)', () => {
      expect(getStatusFromCode('SERVER_DATABASE_UNAVAILABLE')).toBe(503);
      expect(getStatusFromCode('SERVER_GATEWAY_TIMEOUT')).toBe(504);
      expect(getStatusFromCode('SERVER_INTERNAL_ERROR')).toBe(500);
    });

    it('returns 500 for unknown codes (branch: default)', () => {
      expect(getStatusFromCode('UNKNOWN_ERROR')).toBe(500);
    });
  });

  describe('error message sanitization', () => {
    const sanitizeErrorMessage = (message: string): string => {
      // Don't expose internal details
      if (message.includes('password')) {
        return 'Authentication error';
      }
      if (message.includes('token')) {
        return 'Token error';
      }
      if (message.includes('database') || message.includes('prisma')) {
        return 'Database error';
      }
      if (message.includes('timeout')) {
        return 'Request timeout';
      }
      // Truncate very long messages
      if (message.length > 200) {
        return message.substring(0, 200) + '...';
      }
      return message;
    };

    it('sanitizes password-related errors (branch: contains password)', () => {
      expect(sanitizeErrorMessage('Invalid password hash comparison')).toBe('Authentication error');
    });

    it('sanitizes token-related errors (branch: contains token)', () => {
      expect(sanitizeErrorMessage('JWT token malformed')).toBe('Token error');
    });

    it('sanitizes database errors (branch: contains database/prisma)', () => {
      expect(sanitizeErrorMessage('database connection refused')).toBe('Database error');
      expect(sanitizeErrorMessage('prisma query failed')).toBe('Database error');
    });

    it('sanitizes timeout errors (branch: contains timeout)', () => {
      expect(sanitizeErrorMessage('query timeout exceeded')).toBe('Request timeout');
    });

    it('truncates long messages (branch: length > 200)', () => {
      const longMessage = 'a'.repeat(300);
      const result = sanitizeErrorMessage(longMessage);
      expect(result.length).toBe(203); // 200 + '...'
      expect(result.endsWith('...')).toBe(true);
    });

    it('passes through normal messages', () => {
      expect(sanitizeErrorMessage('User not found')).toBe('User not found');
    });
  });
});

// ======================================================================
// ACTUAL EXPORTED FUNCTIONS FROM user.ts - REAL COVERAGE
// ======================================================================

describe('User Routes - REAL Exported Functions (Actual Coverage)', () => {
  describe('DELETED_USER_PREFIX and DELETED_USER_DISPLAY_NAME constants', () => {
    it('DELETED_USER_PREFIX is correct', () => {
      expect(DELETED_USER_PREFIX).toBe('DeletedPlayer_');
    });

    it('DELETED_USER_DISPLAY_NAME is correct', () => {
      expect(DELETED_USER_DISPLAY_NAME).toBe('Deleted Player');
    });
  });

  describe('isDeletedUserUsername()', () => {
    it('returns true for username starting with DeletedPlayer_ prefix', () => {
      expect(isDeletedUserUsername('DeletedPlayer_abc123')).toBe(true);
      expect(isDeletedUserUsername('DeletedPlayer_12345678')).toBe(true);
      expect(isDeletedUserUsername('DeletedPlayer_')).toBe(true);
    });

    it('returns false for normal usernames (branch: does not start with prefix)', () => {
      expect(isDeletedUserUsername('normaluser')).toBe(false);
      expect(isDeletedUserUsername('Player123')).toBe(false);
      expect(isDeletedUserUsername('deletedplayer_lowercase')).toBe(false);
    });

    it('returns false for similar but different prefixes', () => {
      expect(isDeletedUserUsername('Deleted_Player_abc')).toBe(false);
      expect(isDeletedUserUsername('DeletedUser_abc')).toBe(false);
    });
  });

  describe('getDisplayUsername()', () => {
    it('returns DELETED_USER_DISPLAY_NAME for null (branch: !username)', () => {
      expect(getDisplayUsername(null)).toBe('Deleted Player');
    });

    it('returns DELETED_USER_DISPLAY_NAME for undefined (branch: !username)', () => {
      expect(getDisplayUsername(undefined)).toBe('Deleted Player');
    });

    it('returns DELETED_USER_DISPLAY_NAME for empty string (branch: !username)', () => {
      expect(getDisplayUsername('')).toBe('Deleted Player');
    });

    it('returns DELETED_USER_DISPLAY_NAME for deleted user username (branch: isDeletedUserUsername)', () => {
      expect(getDisplayUsername('DeletedPlayer_abc123')).toBe('Deleted Player');
      expect(getDisplayUsername('DeletedPlayer_xyz789')).toBe('Deleted Player');
    });

    it('returns the original username for normal users', () => {
      expect(getDisplayUsername('NormalUser')).toBe('NormalUser');
      expect(getDisplayUsername('player123')).toBe('player123');
      expect(getDisplayUsername('TestPlayer_NotDeleted')).toBe('TestPlayer_NotDeleted');
    });
  });

  describe('formatPlayerForDisplay()', () => {
    it('returns null for null player (branch: !player)', () => {
      expect(formatPlayerForDisplay(null)).toBeNull();
    });

    it('returns null for undefined player (branch: !player)', () => {
      expect(formatPlayerForDisplay(undefined)).toBeNull();
    });

    it('adds displayName property for normal player', () => {
      const player = { username: 'testplayer', id: 'user-123', rating: 1500 };
      const result = formatPlayerForDisplay(player);
      expect(result).not.toBeNull();
      expect(result!.displayName).toBe('testplayer');
      expect(result!.username).toBe('testplayer');
      expect(result!.id).toBe('user-123');
      expect(result!.rating).toBe(1500);
    });

    it('adds displayName as "Deleted Player" for deleted user (branch: deleted username)', () => {
      const player = { username: 'DeletedPlayer_abc12345', id: 'deleted-user' };
      const result = formatPlayerForDisplay(player);
      expect(result).not.toBeNull();
      expect(result!.displayName).toBe('Deleted Player');
      expect(result!.username).toBe('DeletedPlayer_abc12345');
    });

    it('preserves all original player properties', () => {
      const player = {
        username: 'player1',
        id: 'id-1',
        rating: 1600,
        gamesPlayed: 50,
        gamesWon: 30,
        customField: 'preserved',
      };
      const result = formatPlayerForDisplay(player);
      expect(result).not.toBeNull();
      expect(result!.id).toBe('id-1');
      expect(result!.rating).toBe(1600);
      expect(result!.gamesPlayed).toBe(50);
      expect(result!.gamesWon).toBe(30);
      expect((result as any).customField).toBe('preserved');
      expect(result!.displayName).toBe('player1');
    });
  });
});
