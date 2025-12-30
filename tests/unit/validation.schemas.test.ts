/**
 * Unit tests for input validation schemas and sanitization utilities.
 * These tests verify that validation constraints are enforced correctly
 * and that sanitization functions properly handle XSS vectors.
 */

import {
  RegisterSchema,
  LoginSchema,
  UpdateProfileSchema,
  CreateGameSchema,
  ChatMessageSchema,
  UUIDSchema,
  GameIdParamSchema,
  GameListingQuerySchema,
  UserSearchQuerySchema,
  LeaderboardQuerySchema,
  MatchmakingPreferencesSchema,
  CreateTournamentSchema,
  sanitizeString,
  sanitizeHtmlContent,
  createSanitizedStringSchema,
  MAX_USERNAME_LENGTH,
  GameStateSchema,
  SafeStringSchema,
  PositionSchema,
  MoveSchema,
  PaginationSchema,
  SearchSchema,
  FileUploadSchema,
  PaginationQuerySchema,
  SocketEventSchema,
  ChangePasswordSchema,
  RefreshTokenSchema,
  VerifyEmailSchema,
  ForgotPasswordSchema,
  ResetPasswordSchema,
  createSuccessResponse,
  createErrorResponse,
  ValidationError,
  AuthenticationError,
  AuthorizationError,
  type MoveInput,
} from '../../src/shared/validation/schemas';
import type { MoveType } from '../../src/shared/types/game';
import {
  DecisionAutoResolvedMetaSchema,
  GameStateUpdateMetaSchema,
  type DecisionAutoResolvedMetaPayload,
} from '../../src/shared/validation/websocketSchemas';

function assertMoveSchemaMoveTypeIsSubsetOfMoveType<T extends MoveType>(): void {}

void assertMoveSchemaMoveTypeIsSubsetOfMoveType<MoveInput['moveType']>();

describe('Validation Schemas', () => {
  describe('UUIDSchema', () => {
    it('accepts valid UUID v4', () => {
      const validUUID = '123e4567-e89b-12d3-a456-426614174000';
      expect(UUIDSchema.safeParse(validUUID).success).toBe(true);
    });

    it('rejects invalid UUID formats', () => {
      const invalidUUIDs = ['not-a-uuid', '123e4567-e89b-12d3-a456', '', '   '];
      for (const uuid of invalidUUIDs) {
        expect(UUIDSchema.safeParse(uuid).success).toBe(false);
      }
    });
  });

  describe('GameIdParamSchema', () => {
    it('accepts valid UUID and CUID gameId values', () => {
      const uuidResult = GameIdParamSchema.safeParse({
        gameId: '123e4567-e89b-12d3-a456-426614174000',
      });
      const cuidResult = GameIdParamSchema.safeParse({
        gameId: 'c0123456789abcdefghijklmn',
      });

      expect(uuidResult.success).toBe(true);
      expect(cuidResult.success).toBe(true);
    });

    it('rejects missing or invalid gameId', () => {
      expect(GameIdParamSchema.safeParse({}).success).toBe(false);
      expect(GameIdParamSchema.safeParse({ gameId: 'not-valid' }).success).toBe(false);
      expect(GameIdParamSchema.safeParse({ gameId: 'c123' }).success).toBe(false);
    });
  });

  describe('PositionSchema', () => {
    it('accepts valid non-negative integer coordinates', () => {
      const result = PositionSchema.safeParse({ x: 0, y: 5 });
      expect(result.success).toBe(true);
    });

    it('rejects negative coordinates or non-integers', () => {
      expect(PositionSchema.safeParse({ x: -1, y: 0 }).success).toBe(false);
      expect(PositionSchema.safeParse({ x: 0.5, y: 1 }).success).toBe(false);
      expect(PositionSchema.safeParse({ x: 0, y: 0, z: -1 }).success).toBe(false);
      expect(PositionSchema.safeParse({ x: 0, y: 0, z: 1.2 }).success).toBe(false);
    });
  });

  describe('GameListingQuerySchema', () => {
    it('accepts valid query parameters and provides defaults', () => {
      const result = GameListingQuerySchema.safeParse({ status: 'active' });
      expect(result.success).toBe(true);
      if (result.success) {
        expect(result.data.status).toBe('active');
        expect(result.data.limit).toBe(20);
        expect(result.data.offset).toBe(0);
      }
    });

    it('coerces string numbers to integers', () => {
      const result = GameListingQuerySchema.safeParse({ limit: '50', offset: '100' });
      expect(result.success).toBe(true);
      if (result.success) {
        expect(result.data.limit).toBe(50);
        expect(result.data.offset).toBe(100);
      }
    });

    it('accepts supported boardType filters and rejects invalid values', () => {
      const boardTypes = ['square8', 'square19', 'hexagonal'] as const;

      for (const boardType of boardTypes) {
        const result = GameListingQuerySchema.safeParse({ boardType });
        expect(result.success).toBe(true);
        if (result.success) {
          expect(result.data.boardType).toBe(boardType);
        }
      }

      expect(GameListingQuerySchema.safeParse({ boardType: 'circle' as any }).success).toBe(false);
    });

    it('rejects invalid status or out-of-range values', () => {
      expect(GameListingQuerySchema.safeParse({ status: 'invalid' }).success).toBe(false);
      expect(GameListingQuerySchema.safeParse({ limit: '1000' }).success).toBe(false);
      expect(GameListingQuerySchema.safeParse({ offset: '-1' }).success).toBe(false);
    });
  });

  describe('LeaderboardQuerySchema', () => {
    it('accepts empty payload and applies defaults', () => {
      const result = LeaderboardQuerySchema.safeParse({});
      expect(result.success).toBe(true);
      if (result.success) {
        expect(result.data.limit).toBe(50);
        expect(result.data.offset).toBe(0);
      }
    });

    it('coerces string numbers and enforces bounds', () => {
      const ok = LeaderboardQuerySchema.safeParse({ limit: '25', offset: '10' });
      expect(ok.success).toBe(true);
      if (ok.success) {
        expect(ok.data.limit).toBe(25);
        expect(ok.data.offset).toBe(10);
      }

      expect(LeaderboardQuerySchema.safeParse({ limit: '0' }).success).toBe(false);
      expect(LeaderboardQuerySchema.safeParse({ limit: '101' }).success).toBe(false);
      expect(LeaderboardQuerySchema.safeParse({ offset: '-1' }).success).toBe(false);
    });
  });

  describe('UserSearchQuerySchema', () => {
    it('accepts valid search query and sanitizes', () => {
      const result = UserSearchQuerySchema.safeParse({ q: 'test\x00query' });
      expect(result.success).toBe(true);
      if (result.success) {
        expect(result.data.q).toBe('testquery');
        expect(result.data.limit).toBe(10);
      }
    });

    it('rejects empty or too-long queries', () => {
      expect(UserSearchQuerySchema.safeParse({ q: '' }).success).toBe(false);
      expect(UserSearchQuerySchema.safeParse({ q: 'a'.repeat(101) }).success).toBe(false);
    });
  });

  describe('RegisterSchema', () => {
    const validData = {
      username: 'testuser',
      email: 'test@example.com',
      password: 'Password123',
      confirmPassword: 'Password123',
    };

    it('accepts valid registration data', () => {
      expect(RegisterSchema.safeParse(validData).success).toBe(true);
    });

    it('rejects invalid username (too short, invalid chars)', () => {
      expect(RegisterSchema.safeParse({ ...validData, username: 'ab' }).success).toBe(false);
      expect(
        RegisterSchema.safeParse({
          ...validData,
          username: 'a'.repeat(MAX_USERNAME_LENGTH + 1),
        }).success
      ).toBe(false);
      expect(RegisterSchema.safeParse({ ...validData, username: 'test user' }).success).toBe(false);
    });

    it('rejects weak passwords', () => {
      expect(
        RegisterSchema.safeParse({
          ...validData,
          password: 'password',
          confirmPassword: 'password',
        }).success
      ).toBe(false);
      expect(
        RegisterSchema.safeParse({ ...validData, password: 'Pass1', confirmPassword: 'Pass1' })
          .success
      ).toBe(false);
    });

    it('rejects mismatched passwords', () => {
      expect(
        RegisterSchema.safeParse({ ...validData, confirmPassword: 'Different123' }).success
      ).toBe(false);
    });
  });

  describe('LoginSchema', () => {
    it('accepts valid login payload', () => {
      const result = LoginSchema.safeParse({
        email: 'user@example.com',
        password: 'secret',
      });
      expect(result.success).toBe(true);
    });

    it('rejects invalid email or empty password', () => {
      expect(
        LoginSchema.safeParse({
          email: 'not-an-email',
          password: 'secret',
        }).success
      ).toBe(false);

      expect(
        LoginSchema.safeParse({
          email: 'user@example.com',
          password: '',
        }).success
      ).toBe(false);
    });
  });

  describe('CreateGameSchema', () => {
    const validGame = {
      boardType: 'square8',
      timeControl: { type: 'rapid', initialTime: 600, increment: 10 },
    };

    it('accepts valid game creation request', () => {
      expect(CreateGameSchema.safeParse(validGame).success).toBe(true);
    });

    it('rejects invalid board type or time control', () => {
      expect(CreateGameSchema.safeParse({ ...validGame, boardType: 'invalid' }).success).toBe(
        false
      );
      expect(
        CreateGameSchema.safeParse({
          ...validGame,
          timeControl: { type: 'rapid', initialTime: 30, increment: 5 },
        }).success
      ).toBe(false);

      expect(
        CreateGameSchema.safeParse({
          ...validGame,
          timeControl: { type: 'invalid', initialTime: 600, increment: 10 },
        }).success
      ).toBe(false);
    });

    it('accepts aiOpponents config and optional rulesOptions', () => {
      const result = CreateGameSchema.safeParse({
        ...validGame,
        aiOpponents: {
          count: 2,
          difficulty: [3, 5],
          mode: 'local_heuristic',
          aiType: 'heuristic',
        },
        rulesOptions: {
          swapRuleEnabled: true,
        },
      });

      expect(result.success).toBe(true);
      if (result.success) {
        expect(result.data.aiOpponents?.count).toBe(2);
        expect(result.data.aiOpponents?.mode).toBe('local_heuristic');
        expect(result.data.aiOpponents?.aiType).toBe('heuristic');
        expect(result.data.rulesOptions?.swapRuleEnabled).toBe(true);
      }
    });

    it('accepts all supported aiOpponents mode and aiType options', () => {
      const modes = ['local_heuristic', 'service'] as const;
      const aiTypes = [
        'random',
        'heuristic',
        'minimax',
        'mcts',
        'descent',
        'policy_only',
        'gumbel_mcts',
        'ig_gmo',
      ] as const;

      for (const mode of modes) {
        const result = CreateGameSchema.safeParse({
          ...validGame,
          aiOpponents: {
            count: 1,
            difficulty: [5],
            mode,
            aiType: 'random',
          },
        });

        expect(result.success).toBe(true);
        if (result.success) {
          expect(result.data.aiOpponents?.mode).toBe(mode);
        }
      }

      for (const aiType of aiTypes) {
        const result = CreateGameSchema.safeParse({
          ...validGame,
          aiOpponents: {
            count: 1,
            difficulty: [5],
            mode: 'service',
            aiType,
          },
        });

        expect(result.success).toBe(true);
        if (result.success) {
          expect(result.data.aiOpponents?.aiType).toBe(aiType);
        }
      }
    });
  });

  describe('MoveSchema', () => {
    // Structural validation for wire-level MovePayload as described in
    // docs/architecture/CANONICAL_ENGINE_API.md (player_move) and mapped under the
    // schemas/payload row in RULES_SCENARIO_MATRIX.md.
    it('accepts a valid place_ring move with string position', () => {
      const result = MoveSchema.safeParse({
        moveType: 'place_ring',
        position: '{"x":0,"y":0}',
        moveNumber: 1,
      });
      expect(result.success).toBe(true);
    });

    it('accepts a valid move_stack move with structured from/to', () => {
      const result = MoveSchema.safeParse({
        moveType: 'move_stack',
        position: {
          from: { x: 0, y: 0 },
          to: { x: 0, y: 1 },
        },
        moveNumber: 5,
      });
      expect(result.success).toBe(true);
      if (result.success) {
        expect(result.data.position).toEqual({
          from: { x: 0, y: 0 },
          to: { x: 0, y: 1 },
        });
      }
    });

    it('rejects moves with missing position', () => {
      const result = MoveSchema.safeParse({
        moveType: 'place_ring',
      } as any);
      expect(result.success).toBe(false);
    });

    it('rejects moves with invalid moveType', () => {
      const result = MoveSchema.safeParse({
        moveType: 'invalid_type',
        position: '{"x":0,"y":0}',
      } as any);
      expect(result.success).toBe(false);
    });

    it('rejects moves with non-integer moveNumber', () => {
      const result = MoveSchema.safeParse({
        moveType: 'place_ring',
        position: '{"x":0,"y":0}',
        moveNumber: 1.5,
      });
      expect(result.success).toBe(false);
    });

    it('rejects moves with malformed position payloads', () => {
      const numberPositionResult = MoveSchema.safeParse({
        moveType: 'place_ring',
        position: 123 as any,
      });
      expect(numberPositionResult.success).toBe(false);

      const missingToResult = MoveSchema.safeParse({
        moveType: 'move_stack',
        position: {
          from: { x: 0, y: 0 },
        } as any,
      });
      expect(missingToResult.success).toBe(false);
    });
  });

  describe('PaginationSchema', () => {
    it('provides sensible defaults when fields are omitted', () => {
      const result = PaginationSchema.safeParse({});
      expect(result.success).toBe(true);
      if (result.success) {
        expect(result.data.page).toBe(1);
        expect(result.data.limit).toBe(20);
        expect(result.data.sortOrder).toBe('desc');
      }
    });

    it('rejects invalid page or limit values', () => {
      expect(PaginationSchema.safeParse({ page: 0 }).success).toBe(false);
      expect(PaginationSchema.safeParse({ limit: 0 }).success).toBe(false);
      expect(PaginationSchema.safeParse({ limit: 101 }).success).toBe(false);
    });
  });

  describe('SearchSchema', () => {
    it('accepts valid search payload', () => {
      const result = SearchSchema.safeParse({ query: 'hello', type: 'users' });
      expect(result.success).toBe(true);
      if (result.success) {
        expect(result.data.query).toBe('hello');
        expect(result.data.type).toBe('users');
      }
    });

    it('rejects too-short or too-long queries', () => {
      expect(SearchSchema.safeParse({ query: '' }).success).toBe(false);
      expect(SearchSchema.safeParse({ query: 'a'.repeat(101) }).success).toBe(false);
    });
  });

  describe('PaginationQuerySchema', () => {
    it('applies defaults for missing fields', () => {
      const result = PaginationQuerySchema.safeParse({});
      expect(result.success).toBe(true);
      if (result.success) {
        expect(result.data.page).toBe(1);
        expect(result.data.limit).toBe(20);
        expect(result.data.offset).toBe(0);
        expect(result.data.sortOrder).toBe('desc');
      }
    });

    it('coerces query string values to numbers', () => {
      const result = PaginationQuerySchema.safeParse({
        page: '2',
        limit: '10',
        offset: '30',
      });
      expect(result.success).toBe(true);
      if (result.success) {
        expect(result.data.page).toBe(2);
        expect(result.data.limit).toBe(10);
        expect(result.data.offset).toBe(30);
      }
    });

    it('rejects out-of-range values', () => {
      expect(PaginationQuerySchema.safeParse({ page: 0 }).success).toBe(false);
      expect(PaginationQuerySchema.safeParse({ limit: 0 }).success).toBe(false);
      expect(PaginationQuerySchema.safeParse({ limit: 101 }).success).toBe(false);
      expect(PaginationQuerySchema.safeParse({ offset: -1 }).success).toBe(false);
    });
  });

  describe('SocketEventSchema', () => {
    // Structural validation for generic WebSocket event envelopes used by
    // docs/architecture/CANONICAL_ENGINE_API.md and referenced from RULES_SCENARIO_MATRIX.md.
    it('accepts minimal valid event payload', () => {
      const result = SocketEventSchema.safeParse({
        event: 'player_move',
        data: { some: 'payload' },
      });
      expect(result.success).toBe(true);
      if (result.success) {
        expect(result.data.event).toBe('player_move');
        expect(result.data.data).toEqual({ some: 'payload' });
      }
    });

    it('rejects empty event names', () => {
      expect(
        SocketEventSchema.safeParse({
          event: '',
          data: {},
        }).success
      ).toBe(false);
    });

    it('accepts an optional Date timestamp when present', () => {
      const now = new Date();
      const result = SocketEventSchema.safeParse({
        event: 'player_move',
        data: {},
        timestamp: now,
      });

      expect(result.success).toBe(true);
      if (result.success) {
        expect(result.data.timestamp).toEqual(now);
      }
    });

    it('rejects non-Date timestamps', () => {
      const result = SocketEventSchema.safeParse({
        event: 'player_move',
        data: {},
        timestamp: 'not-a-date' as any,
      });
      expect(result.success).toBe(false);
    });
  });

  describe('FileUploadSchema', () => {
    it('accepts a valid image upload payload', () => {
      const result = FileUploadSchema.safeParse({
        filename: 'avatar.png',
        mimetype: 'image/png',
        size: 1024 * 1024, // 1MB
      });
      expect(result.success).toBe(true);
      if (result.success) {
        expect(result.data.filename).toBe('avatar.png');
      }
    });

    it('rejects non-image mimetypes or oversized files', () => {
      expect(
        FileUploadSchema.safeParse({
          filename: 'avatar.txt',
          mimetype: 'text/plain',
          size: 1024,
        }).success
      ).toBe(false);

      expect(
        FileUploadSchema.safeParse({
          filename: 'avatar.png',
          mimetype: 'image/png',
          size: 6 * 1024 * 1024, // 6MB, over 5MB limit
        }).success
      ).toBe(false);
    });
  });

  describe('ChatMessageSchema', () => {
    it('accepts valid chat message and trims content', () => {
      const result = ChatMessageSchema.safeParse({
        gameId: '123e4567-e89b-12d3-a456-426614174000',
        content: '  Hello  ',
        type: 'game',
      });
      expect(result.success).toBe(true);
      if (result.success) {
        expect(result.data.content).toBe('Hello');
      }
    });

    it('rejects empty or too-long content', () => {
      const base = { gameId: '123e4567-e89b-12d3-a456-426614174000', type: 'game' };
      expect(ChatMessageSchema.safeParse({ ...base, content: '' }).success).toBe(false);
      expect(ChatMessageSchema.safeParse({ ...base, content: 'a'.repeat(501) }).success).toBe(
        false
      );
    });

    it('defaults type to \"game\" when omitted', () => {
      const result = ChatMessageSchema.safeParse({
        gameId: '123e4567-e89b-12d3-a456-426614174000',
        content: 'Hello world',
      });
      expect(result.success).toBe(true);
      if (result.success) {
        expect(result.data.type).toBe('game');
        expect(result.data.recipientId).toBeUndefined();
      }
    });

    it('accepts spectator and private message types', () => {
      const base = {
        gameId: '123e4567-e89b-12d3-a456-426614174000',
        content: 'Hello from spectator',
      };

      const spectator = ChatMessageSchema.safeParse({
        ...base,
        type: 'spectator',
      });
      expect(spectator.success).toBe(true);
      if (spectator.success) {
        expect(spectator.data.type).toBe('spectator');
        expect(spectator.data.recipientId).toBeUndefined();
      }

      const recipientId = '123e4567-e89b-12d3-a456-426614174000';
      const privateMessage = ChatMessageSchema.safeParse({
        ...base,
        type: 'private',
        recipientId,
      });
      expect(privateMessage.success).toBe(true);
      if (privateMessage.success) {
        expect(privateMessage.data.type).toBe('private');
        expect(privateMessage.data.recipientId).toBe(recipientId);
      }
    });
  });

  describe('MatchmakingPreferencesSchema', () => {
    it('accepts valid preferences and enforces min/max invariants', () => {
      const result = MatchmakingPreferencesSchema.safeParse({
        boardType: 'square8',
        timeControl: { min: 300, max: 900 },
        ratingRange: { min: 800, max: 1800 },
        allowAI: true,
      });
      expect(result.success).toBe(true);
      if (result.success) {
        expect(result.data.boardType).toBe('square8');
        expect(result.data.timeControl.min).toBe(300);
        expect(result.data.timeControl.max).toBe(900);
        expect(result.data.ratingRange.min).toBe(800);
        expect(result.data.ratingRange.max).toBe(1800);
        expect(result.data.allowAI).toBe(true);
      }
    });

    it('rejects inverted time or rating ranges', () => {
      const badTime = MatchmakingPreferencesSchema.safeParse({
        boardType: 'square8',
        timeControl: { min: 1200, max: 300 },
        ratingRange: { min: 800, max: 1800 },
      });
      expect(badTime.success).toBe(false);

      const badRating = MatchmakingPreferencesSchema.safeParse({
        boardType: 'square8',
        timeControl: { min: 300, max: 900 },
        ratingRange: { min: 2000, max: 1500 },
      });
      expect(badRating.success).toBe(false);
    });
  });

  describe('CreateTournamentSchema', () => {
    it('accepts a valid tournament payload', () => {
      const now = Date.now();
      const startsAt = new Date(now + 2 * 60 * 60 * 1000);
      const registrationDeadline = new Date(now + 60 * 60 * 1000);

      const result = CreateTournamentSchema.safeParse({
        name: 'Test Tournament',
        format: 'single_elimination',
        boardType: 'square8',
        maxParticipants: 16,
        timeControl: {
          initialTime: 600,
          increment: 5,
        },
        isRated: true,
        entryFee: 0,
        prizePool: 0,
        startsAt,
        registrationDeadline,
      });

      expect(result.success).toBe(true);
      if (result.success) {
        expect(result.data.name).toBe('Test Tournament');
        expect(result.data.maxParticipants).toBe(16);
        expect(result.data.timeControl.initialTime).toBe(600);
        expect(result.data.startsAt).toEqual(startsAt);
        expect(result.data.registrationDeadline).toEqual(registrationDeadline);
      }
    });

    it('rejects tournaments with registration deadline after start time', () => {
      const now = Date.now();
      const startsAt = new Date(now + 60 * 60 * 1000);
      const registrationDeadline = new Date(now + 2 * 60 * 60 * 1000);

      const result = CreateTournamentSchema.safeParse({
        name: 'Bad Tournament',
        format: 'single_elimination',
        boardType: 'square8',
        maxParticipants: 8,
        timeControl: {
          initialTime: 600,
          increment: 5,
        },
        startsAt,
        registrationDeadline,
      });

      expect(result.success).toBe(false);
    });

    it('accepts tournament without registrationDeadline', () => {
      const now = Date.now();
      const startsAt = new Date(now + 2 * 60 * 60 * 1000);

      const result = CreateTournamentSchema.safeParse({
        name: 'Open Registration Tournament',
        format: 'swiss',
        boardType: 'hexagonal',
        maxParticipants: 32,
        timeControl: {
          initialTime: 300,
          increment: 3,
        },
        isRated: false,
        startsAt,
        // registrationDeadline intentionally omitted
      });

      expect(result.success).toBe(true);
      if (result.success) {
        expect(result.data.name).toBe('Open Registration Tournament');
        expect(result.data.registrationDeadline).toBeUndefined();
      }
    });
  });

  describe('UpdateProfileSchema', () => {
    it('accepts partial updates and enforces username rules when present', () => {
      const result = UpdateProfileSchema.safeParse({
        username: 'updated_user',
        email: 'updated@example.com',
      });
      expect(result.success).toBe(true);

      expect(
        UpdateProfileSchema.safeParse({
          username: 'no spaces allowed',
        }).success
      ).toBe(false);
    });
  });

  describe('ChangePasswordSchema', () => {
    const base = {
      currentPassword: 'OldPassword123',
      newPassword: 'NewPassword123',
      confirmPassword: 'NewPassword123',
    };

    it('accepts valid change-password payload', () => {
      expect(ChangePasswordSchema.safeParse(base).success).toBe(true);
    });

    it('rejects weak or mismatched new passwords', () => {
      expect(
        ChangePasswordSchema.safeParse({
          ...base,
          newPassword: 'weak',
          confirmPassword: 'weak',
        }).success
      ).toBe(false);

      expect(
        ChangePasswordSchema.safeParse({
          ...base,
          confirmPassword: 'Different123',
        }).success
      ).toBe(false);
    });
  });

  describe('Auth token & email flows', () => {
    it('validates refresh token payload', () => {
      expect(RefreshTokenSchema.safeParse({ refreshToken: 'token-123' }).success).toBe(true);
      expect(RefreshTokenSchema.safeParse({}).success).toBe(false);
    });

    it('validates verify-email payload', () => {
      expect(VerifyEmailSchema.safeParse({ token: 'abc' }).success).toBe(true);
      expect(VerifyEmailSchema.safeParse({ token: '' }).success).toBe(false);
    });

    it('validates forgot-password payload', () => {
      expect(ForgotPasswordSchema.safeParse({ email: 'user@example.com' }).success).toBe(true);
      expect(ForgotPasswordSchema.safeParse({ email: 'not-an-email' }).success).toBe(false);
    });

    it('validates reset-password payload', () => {
      expect(
        ResetPasswordSchema.safeParse({
          token: 'reset-token',
          newPassword: 'NewPassword123',
        }).success
      ).toBe(true);

      expect(
        ResetPasswordSchema.safeParse({
          token: '',
          newPassword: 'short',
        }).success
      ).toBe(false);
    });
  });

  describe('API response helpers', () => {
    it('createSuccessResponse wraps data with success flag and timestamp', () => {
      const payload = { foo: 'bar' };
      const response = createSuccessResponse(payload);
      expect(response.success).toBe(true);
      expect(response.data).toBe(payload);
      expect(response.timestamp instanceof Date).toBe(true);
    });

    it('createErrorResponse wraps error details with timestamp', () => {
      const response = createErrorResponse('Something went wrong', 'E_TEST', { extra: true });
      expect(response.success).toBe(false);
      expect(response.error.message).toBe('Something went wrong');
      expect(response.error.code).toBe('E_TEST');
      expect(response.error.details).toEqual({ extra: true });
      expect(response.error.timestamp instanceof Date).toBe(true);
    });
  });

  describe('GameStateSchema', () => {
    const basePlayer = {
      id: '123e4567-e89b-12d3-a456-426614174000',
      username: 'Player 1',
      type: 'human' as const,
      playerNumber: 1,
      isReady: true,
      timeRemaining: 600000,
    };

    it('accepts a minimal valid game state payload', () => {
      const result = GameStateSchema.safeParse({
        id: '123e4567-e89b-12d3-a456-426614174000',
        boardType: 'square8',
        players: [basePlayer],
        currentPhase: 'ring_placement',
        currentPlayer: 1,
        gameStatus: 'waiting',
        isRated: true,
        maxPlayers: 2,
      });
      expect(result.success).toBe(true);
      if (result.success) {
        expect(result.data.boardType).toBe('square8');
        expect(result.data.players[0].username).toBe('Player 1');
      }
    });

    it('rejects invalid phase or status values', () => {
      const base = {
        id: '123e4567-e89b-12d3-a456-426614174000',
        boardType: 'square8',
        players: [basePlayer],
        currentPhase: 'ring_placement',
        currentPlayer: 1,
        gameStatus: 'waiting',
        isRated: true,
        maxPlayers: 2,
      };

      expect(
        GameStateSchema.safeParse({
          ...base,
          currentPhase: 'invalid_phase',
        }).success
      ).toBe(false);

      expect(
        GameStateSchema.safeParse({
          ...base,
          gameStatus: 'invalid_status',
        }).success
      ).toBe(false);
    });

    it('accepts other board types and terminal statuses', () => {
      const base = {
        id: '123e4567-e89b-12d3-a456-426614174000',
        players: [basePlayer],
        currentPlayer: 1,
        isRated: false,
        maxPlayers: 4,
      };

      const variants = [
        {
          boardType: 'square19' as const,
          currentPhase: 'movement' as const,
          gameStatus: 'active' as const,
        },
        {
          boardType: 'hexagonal' as const,
          currentPhase: 'territory_processing' as const,
          gameStatus: 'completed' as const,
        },
      ];

      for (const variant of variants) {
        const result = GameStateSchema.safeParse({
          ...base,
          boardType: variant.boardType,
          currentPhase: variant.currentPhase,
          currentPlayer: 2,
          gameStatus: variant.gameStatus,
        });

        expect(result.success).toBe(true);
        if (result.success) {
          expect(result.data.boardType).toBe(variant.boardType);
          expect(result.data.currentPhase).toBe(variant.currentPhase);
          expect(result.data.gameStatus).toBe(variant.gameStatus);
        }
      }
    });
  });

  describe('DecisionAutoResolvedMetaSchema', () => {
    const base: DecisionAutoResolvedMetaPayload = {
      choiceType: 'line_order',
      choiceKind: 'line_order',
      actingPlayerNumber: 1,
      resolvedMoveId: 'move-123',
      resolvedOptionIndex: 0,
      resolvedOptionKey: 'option_1_collapse_all_and_eliminate',
      reason: 'timeout',
    };

    it('accepts a structurally valid auto-resolve meta payload', () => {
      const result = DecisionAutoResolvedMetaSchema.safeParse(base);
      expect(result.success).toBe(true);
      if (result.success) {
        expect(result.data.choiceType).toBe('line_order');
        expect(result.data.choiceKind).toBe('line_order');
        expect(result.data.actingPlayerNumber).toBe(1);
        expect(result.data.reason).toBe('timeout');
      }
    });

    it('rejects invalid reason discriminator', () => {
      const result = DecisionAutoResolvedMetaSchema.safeParse({
        ...base,
        reason: 'not_a_reason',
      } as any);
      expect(result.success).toBe(false);
    });

    it('rejects invalid choiceType', () => {
      const result = DecisionAutoResolvedMetaSchema.safeParse({
        ...base,
        choiceType: 'invalid_type',
      } as any);
      expect(result.success).toBe(false);
    });
  });

  describe('GameStateUpdateMetaSchema', () => {
    it('accepts meta with a valid decisionAutoResolved diff summary', () => {
      const meta = {
        diffSummary: {
          decisionAutoResolved: {
            choiceType: 'ring_elimination',
            choiceKind: 'ring_elimination',
            actingPlayerNumber: 2,
            resolvedMoveId: 'move-456',
            reason: 'timeout',
          },
        },
      };

      const result = GameStateUpdateMetaSchema.safeParse(meta);
      expect(result.success).toBe(true);
      if (result.success) {
        expect(result.data.diffSummary?.decisionAutoResolved?.actingPlayerNumber).toBe(2);
        expect(result.data.diffSummary?.decisionAutoResolved?.choiceType).toBe('ring_elimination');
      }
    });

    it('accepts meta without diffSummary or decisionAutoResolved', () => {
      expect(GameStateUpdateMetaSchema.safeParse({}).success).toBe(true);
      expect(
        GameStateUpdateMetaSchema.safeParse({
          diffSummary: {},
        }).success
      ).toBe(true);
    });

    it('rejects meta with invalid decisionAutoResolved shape', () => {
      const result = GameStateUpdateMetaSchema.safeParse({
        diffSummary: {
          decisionAutoResolved: {
            choiceType: 'line_order',
            choiceKind: 'line_order',
            actingPlayerNumber: 0, // must be >= 1
            reason: 'timeout',
          },
        },
      });

      expect(result.success).toBe(false);
    });
  });
});

describe('Sanitization Utilities', () => {
  describe('sanitizeString', () => {
    it('removes null bytes and trims whitespace', () => {
      expect(sanitizeString('hello\x00world')).toBe('helloworld');
      expect(sanitizeString('  hello  ')).toBe('hello');
    });

    it('handles non-string input gracefully', () => {
      expect(sanitizeString(null as any)).toBe('');
      expect(sanitizeString(undefined as any)).toBe('');
      expect(sanitizeString(42 as any)).toBe('');
      expect(sanitizeString({} as any)).toBe('');
    });
  });

  describe('sanitizeHtmlContent', () => {
    it('escapes HTML special characters', () => {
      expect(sanitizeHtmlContent('<script>alert("xss")</script>')).toBe(
        '&lt;script&gt;alert(&quot;xss&quot;)&lt;&#x2F;script&gt;'
      );
      expect(sanitizeHtmlContent('Tom & Jerry')).toBe('Tom &amp; Jerry');
    });

    it('handles empty strings', () => {
      expect(sanitizeHtmlContent('')).toBe('');
    });

    it('handles non-string inputs gracefully', () => {
      expect(sanitizeHtmlContent(null as any)).toBe('');
      expect(sanitizeHtmlContent(undefined as any)).toBe('');
      expect(sanitizeHtmlContent(42 as any)).toBe('');
      expect(sanitizeHtmlContent({} as any)).toBe('');
    });
  });

  describe('createSanitizedStringSchema', () => {
    it('creates schema with custom limits and applies sanitization', () => {
      const schema = createSanitizedStringSchema(50, 5);
      expect(schema.safeParse('a'.repeat(51)).success).toBe(false);
      expect(schema.safeParse('abc').success).toBe(false);

      const result = schema.safeParse('  hello\x00world  ');
      expect(result.success).toBe(true);
      if (result.success) {
        expect(result.data).toBe('helloworld');
      }
    });

    it('uses default maxLength and minLength when not provided', () => {
      const schemaDefaults = createSanitizedStringSchema();
      // Default minLength is 1, so empty string fails
      expect(schemaDefaults.safeParse('').success).toBe(false);
      // Single char should pass with default minLength of 1
      const singleChar = schemaDefaults.safeParse('x');
      expect(singleChar.success).toBe(true);
      if (singleChar.success) {
        expect(singleChar.data).toBe('x');
      }
    });
  });

  describe('SafeStringSchema', () => {
    it('applies sanitizeString transformation', () => {
      const result = SafeStringSchema.safeParse('  hello\x00world  ');
      expect(result.success).toBe(true);
      if (result.success) {
        expect(result.data).toBe('helloworld');
      }
    });
  });
});

describe('Custom Error Classes', () => {
  describe('ValidationError', () => {
    it('creates error with message only', () => {
      const error = new ValidationError('Field is required');
      expect(error).toBeInstanceOf(Error);
      expect(error.name).toBe('ValidationError');
      expect(error.message).toBe('Field is required');
      expect(error.field).toBeUndefined();
      expect(error.code).toBeUndefined();
    });

    it('creates error with message and field', () => {
      const error = new ValidationError('Invalid format', 'email');
      expect(error.message).toBe('Invalid format');
      expect(error.field).toBe('email');
      expect(error.code).toBeUndefined();
    });

    it('creates error with message, field, and code', () => {
      const error = new ValidationError('Too short', 'password', 'E_MIN_LENGTH');
      expect(error.message).toBe('Too short');
      expect(error.field).toBe('password');
      expect(error.code).toBe('E_MIN_LENGTH');
    });
  });

  describe('AuthenticationError', () => {
    it('creates error with default message', () => {
      const error = new AuthenticationError();
      expect(error).toBeInstanceOf(Error);
      expect(error.name).toBe('AuthenticationError');
      expect(error.message).toBe('Authentication required');
    });

    it('creates error with custom message', () => {
      const error = new AuthenticationError('Token expired');
      expect(error.message).toBe('Token expired');
    });
  });

  describe('AuthorizationError', () => {
    it('creates error with default message', () => {
      const error = new AuthorizationError();
      expect(error).toBeInstanceOf(Error);
      expect(error.name).toBe('AuthorizationError');
      expect(error.message).toBe('Insufficient permissions');
    });

    it('creates error with custom message', () => {
      const error = new AuthorizationError('Admin access required');
      expect(error.message).toBe('Admin access required');
    });
  });
});

describe('Security Validation', () => {
  it('rejects SQL injection attempts in UUIDs', () => {
    const attacks = ["'; DROP TABLE users; --", "1' OR '1'='1"];
    for (const attack of attacks) {
      expect(UUIDSchema.safeParse(attack).success).toBe(false);
    }
  });

  it('enforces maximum limits to prevent DoS', () => {
    expect(UserSearchQuerySchema.safeParse({ q: 'a'.repeat(200) }).success).toBe(false);
    expect(GameListingQuerySchema.safeParse({ limit: '999999' }).success).toBe(false);
  });

  it('enforces password complexity requirements', () => {
    const base = { username: 'test', email: 'test@test.com' };
    const weakPasswords = ['password', 'PASSWORD', '12345678'];
    for (const pwd of weakPasswords) {
      expect(
        RegisterSchema.safeParse({ ...base, password: pwd, confirmPassword: pwd }).success
      ).toBe(false);
    }
  });
});
