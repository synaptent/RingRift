/**
 * OpenAPI/Swagger configuration for RingRift REST API.
 *
 * This module generates an OpenAPI 3.0 specification from JSDoc annotations
 * in the route files. The specification is used to serve interactive
 * documentation via Swagger UI at /api/docs.
 */

import swaggerJsdoc from 'swagger-jsdoc';

const options: swaggerJsdoc.Options = {
  definition: {
    openapi: '3.0.0',
    info: {
      title: 'RingRift API',
      version: '1.0.0',
      description: `
REST API for RingRift game server.

## Authentication

Most endpoints require authentication via Bearer token. Include the access token
in the Authorization header:

\`\`\`
Authorization: Bearer <access_token>
\`\`\`

Tokens are obtained through the /auth/login or /auth/register endpoints.

## Error Handling

All errors follow a standardized format:

\`\`\`json
{
  "success": false,
  "error": {
    "code": "ERROR_CODE",
    "message": "Human-readable message",
    "details": {}
  }
}
\`\`\`

Error codes are prefixed by category:
- \`AUTH_*\` - Authentication/authorization errors
- \`VALIDATION_*\` - Input validation errors
- \`RESOURCE_*\` - Resource-related errors (not found, already exists)
- \`GAME_*\` - Game-specific errors
- \`RATE_LIMIT_*\` - Rate limiting errors
- \`SERVER_*\` - Internal server errors

## Rate Limiting

Rate limits are applied to prevent abuse. Limits vary by endpoint:
- Auth endpoints: 10 requests/minute for login/register
- Game creation: 5 games/hour per user
- General API: 100 requests/minute per authenticated user

Rate limit headers are included in responses:
- \`X-RateLimit-Limit\`: Request limit
- \`X-RateLimit-Remaining\`: Remaining requests
- \`X-RateLimit-Reset\`: Reset timestamp (Unix epoch)
      `,
      contact: {
        name: 'RingRift Team',
      },
      license: {
        name: 'Proprietary',
      },
    },
    servers: [
      {
        url: '/api',
        description: 'API Base URL',
      },
    ],
    tags: [
      {
        name: 'Authentication',
        description: 'User registration, login, token management, and password reset',
      },
      {
        name: 'Users',
        description: 'User profile, statistics, and account management',
      },
      {
        name: 'Games',
        description: 'Game creation, joining, and game state management',
      },
    ],
    components: {
      securitySchemes: {
        bearerAuth: {
          type: 'http',
          scheme: 'bearer',
          bearerFormat: 'JWT',
          description: 'JWT access token obtained from /auth/login or /auth/register',
        },
      },
      schemas: {
        // ========================================
        // Common Response Schemas
        // ========================================
        Error: {
          type: 'object',
          required: ['success', 'error'],
          properties: {
            success: {
              type: 'boolean',
              example: false,
            },
            error: {
              type: 'object',
              required: ['code', 'message'],
              properties: {
                code: {
                  type: 'string',
                  description: 'Standardized error code',
                  example: 'AUTH_INVALID_CREDENTIALS',
                },
                message: {
                  type: 'string',
                  description: 'Human-readable error message',
                  example: 'Invalid credentials',
                },
                details: {
                  type: 'object',
                  description: 'Additional error context (optional)',
                  additionalProperties: true,
                },
              },
            },
          },
        },
        SuccessResponse: {
          type: 'object',
          required: ['success'],
          properties: {
            success: {
              type: 'boolean',
              example: true,
            },
            message: {
              type: 'string',
              description: 'Optional success message',
            },
          },
        },
        Pagination: {
          type: 'object',
          properties: {
            total: {
              type: 'integer',
              description: 'Total number of items',
              example: 100,
            },
            limit: {
              type: 'integer',
              description: 'Number of items per page',
              example: 20,
            },
            offset: {
              type: 'integer',
              description: 'Current offset',
              example: 0,
            },
            hasMore: {
              type: 'boolean',
              description: 'Whether more items exist',
              example: true,
            },
          },
        },

        // ========================================
        // User Schemas
        // ========================================
        User: {
          type: 'object',
          properties: {
            id: {
              type: 'string',
              format: 'uuid',
              description: 'Unique user identifier',
            },
            email: {
              type: 'string',
              format: 'email',
              description: 'User email address',
            },
            username: {
              type: 'string',
              minLength: 3,
              maxLength: 20,
              description: 'Display name',
              example: 'player123',
            },
            role: {
              type: 'string',
              enum: ['USER', 'ADMIN'],
              description: 'User role',
            },
            rating: {
              type: 'integer',
              description: 'Elo rating',
              example: 1500,
            },
            gamesPlayed: {
              type: 'integer',
              description: 'Total games played',
              example: 42,
            },
            gamesWon: {
              type: 'integer',
              description: 'Total games won',
              example: 25,
            },
            createdAt: {
              type: 'string',
              format: 'date-time',
              description: 'Account creation timestamp',
            },
            lastLoginAt: {
              type: 'string',
              format: 'date-time',
              description: 'Last login timestamp',
            },
            emailVerified: {
              type: 'boolean',
              description: 'Whether email is verified',
            },
            isActive: {
              type: 'boolean',
              description: 'Whether account is active',
            },
          },
        },
        UserPublic: {
          type: 'object',
          description: 'Public user information (no sensitive data)',
          properties: {
            id: {
              type: 'string',
              format: 'uuid',
            },
            username: {
              type: 'string',
            },
            rating: {
              type: 'integer',
            },
            gamesPlayed: {
              type: 'integer',
            },
            gamesWon: {
              type: 'integer',
            },
          },
        },
        UserStats: {
          type: 'object',
          properties: {
            rating: {
              type: 'integer',
              example: 1500,
            },
            gamesPlayed: {
              type: 'integer',
              example: 42,
            },
            gamesWon: {
              type: 'integer',
              example: 25,
            },
            gamesLost: {
              type: 'integer',
              example: 17,
            },
            winRate: {
              type: 'number',
              format: 'float',
              example: 59.52,
            },
            recentGames: {
              type: 'array',
              items: {
                $ref: '#/components/schemas/GameSummary',
              },
            },
            ratingHistory: {
              type: 'array',
              items: {
                type: 'object',
                properties: {
                  date: {
                    type: 'string',
                    format: 'date-time',
                  },
                  rating: {
                    type: 'integer',
                  },
                },
              },
            },
          },
        },

        // ========================================
        // Auth Schemas
        // ========================================
        RegisterRequest: {
          type: 'object',
          required: ['email', 'username', 'password', 'confirmPassword'],
          properties: {
            email: {
              type: 'string',
              format: 'email',
              maxLength: 255,
              description: 'Valid email address',
              example: 'player@example.com',
            },
            username: {
              type: 'string',
              minLength: 3,
              maxLength: 20,
              pattern: '^[a-zA-Z0-9_-]+$',
              description: 'Username (letters, numbers, underscores, hyphens only)',
              example: 'player123',
            },
            password: {
              type: 'string',
              minLength: 8,
              maxLength: 128,
              description: 'Password (must contain lowercase, uppercase, and number)',
              example: 'SecurePass123',
            },
            confirmPassword: {
              type: 'string',
              description: 'Must match password',
              example: 'SecurePass123',
            },
          },
        },
        LoginRequest: {
          type: 'object',
          required: ['email', 'password'],
          properties: {
            email: {
              type: 'string',
              format: 'email',
              example: 'player@example.com',
            },
            password: {
              type: 'string',
              example: 'SecurePass123',
            },
            rememberMe: {
              type: 'boolean',
              description: 'Extend session duration',
            },
          },
        },
        AuthResponse: {
          type: 'object',
          properties: {
            success: {
              type: 'boolean',
              example: true,
            },
            data: {
              type: 'object',
              properties: {
                user: {
                  $ref: '#/components/schemas/User',
                },
                accessToken: {
                  type: 'string',
                  description: 'JWT access token (short-lived)',
                },
                refreshToken: {
                  type: 'string',
                  description: 'Refresh token (long-lived)',
                },
              },
            },
            message: {
              type: 'string',
            },
          },
        },
        RefreshTokenRequest: {
          type: 'object',
          required: ['refreshToken'],
          properties: {
            refreshToken: {
              type: 'string',
              description: 'Refresh token from login/register response',
            },
          },
        },
        TokenRefreshResponse: {
          type: 'object',
          properties: {
            success: {
              type: 'boolean',
              example: true,
            },
            data: {
              type: 'object',
              properties: {
                accessToken: {
                  type: 'string',
                  description: 'New JWT access token',
                },
                refreshToken: {
                  type: 'string',
                  description: 'New refresh token (rotated)',
                },
              },
            },
            message: {
              type: 'string',
            },
          },
        },
        VerifyEmailRequest: {
          type: 'object',
          required: ['token'],
          properties: {
            token: {
              type: 'string',
              description: 'Email verification token',
            },
          },
        },
        ForgotPasswordRequest: {
          type: 'object',
          required: ['email'],
          properties: {
            email: {
              type: 'string',
              format: 'email',
            },
          },
        },
        ResetPasswordRequest: {
          type: 'object',
          required: ['token', 'newPassword'],
          properties: {
            token: {
              type: 'string',
              description: 'Password reset token from email',
            },
            newPassword: {
              type: 'string',
              minLength: 8,
              maxLength: 128,
              description: 'New password',
            },
          },
        },

        // ========================================
        // Game Schemas
        // ========================================
        TimeControl: {
          type: 'object',
          required: ['type', 'initialTime', 'increment'],
          properties: {
            type: {
              type: 'string',
              enum: ['blitz', 'rapid', 'classical'],
              description: 'Time control category',
              example: 'rapid',
            },
            initialTime: {
              type: 'integer',
              minimum: 60,
              maximum: 7200,
              description: 'Initial time in seconds (1 min to 2 hours)',
              example: 600,
            },
            increment: {
              type: 'integer',
              minimum: 0,
              maximum: 60,
              description: 'Increment per move in seconds',
              example: 10,
            },
          },
        },
        AIOpponentsConfig: {
          type: 'object',
          properties: {
            count: {
              type: 'integer',
              minimum: 0,
              maximum: 3,
              description: 'Number of AI opponents',
            },
            difficulty: {
              type: 'array',
              items: {
                type: 'integer',
                minimum: 1,
                maximum: 10,
              },
              description: 'Difficulty level for each AI (1-10)',
            },
            mode: {
              type: 'string',
              enum: ['local_heuristic', 'service'],
              description: 'AI computation mode',
            },
            aiType: {
              type: 'string',
              enum: [
                'random',
                'heuristic',
                'minimax',
                'mcts',
                'descent',
                'policy_only',
                'gumbel_mcts',
                'ig_gmo',
              ],
              description:
                'Single AI algorithm type (legacy; policy_only/ig_gmo are override-only)',
            },
            aiTypes: {
              type: 'array',
              items: {
                type: 'string',
                enum: [
                  'random',
                  'heuristic',
                  'minimax',
                  'mcts',
                  'descent',
                  'policy_only',
                  'gumbel_mcts',
                  'ig_gmo',
                ],
              },
              description:
                'Per-opponent AI types (overrides aiType when provided; policy_only/ig_gmo are override-only)',
            },
          },
          example: {
            count: 3,
            difficulty: [3, 7, 9],
            mode: 'service',
            aiTypes: ['policy_only', 'gumbel_mcts', 'ig_gmo'],
          },
        },
        RulesOptions: {
          type: 'object',
          properties: {
            swapRuleEnabled: {
              type: 'boolean',
              description: 'Enable pie rule (swap_sides) for 2-player games',
            },
            ringsPerPlayer: {
              type: 'integer',
              minimum: 1,
              maximum: 200,
              description: 'Override rings per player',
            },
            lpsRoundsRequired: {
              type: 'integer',
              minimum: 1,
              maximum: 10,
              description: 'Override LPS victory rounds',
            },
          },
        },
        CreateGameRequest: {
          type: 'object',
          required: ['boardType', 'timeControl'],
          properties: {
            boardType: {
              type: 'string',
              enum: ['square8', 'square19', 'hex8', 'hexagonal'],
              description: 'Board type',
              example: 'square8',
            },
            timeControl: {
              $ref: '#/components/schemas/TimeControl',
            },
            isRated: {
              type: 'boolean',
              default: true,
              description: 'Whether game affects ratings',
            },
            isPrivate: {
              type: 'boolean',
              default: false,
              description: 'Whether game is hidden from lobby',
            },
            maxPlayers: {
              type: 'integer',
              minimum: 2,
              maximum: 4,
              default: 2,
              description: 'Maximum number of players',
            },
            aiOpponents: {
              $ref: '#/components/schemas/AIOpponentsConfig',
            },
            rulesOptions: {
              $ref: '#/components/schemas/RulesOptions',
            },
            seed: {
              type: 'integer',
              minimum: 0,
              maximum: 2147483647,
              description: 'RNG seed for deterministic games',
            },
            isCalibrationGame: {
              type: 'boolean',
              description: 'Marks this game as part of an AI calibration run',
            },
            calibrationDifficulty: {
              type: 'integer',
              minimum: 1,
              maximum: 10,
              description: 'Primary AI difficulty tier being calibrated (1-10)',
            },
          },
        },
        Game: {
          type: 'object',
          properties: {
            id: {
              type: 'string',
              format: 'uuid',
            },
            boardType: {
              type: 'string',
              enum: ['square8', 'square19', 'hex8', 'hexagonal'],
            },
            maxPlayers: {
              type: 'integer',
            },
            timeControl: {
              $ref: '#/components/schemas/TimeControl',
            },
            isRated: {
              type: 'boolean',
            },
            allowSpectators: {
              type: 'boolean',
            },
            status: {
              type: 'string',
              enum: ['waiting', 'active', 'completed', 'abandoned', 'paused'],
            },
            player1: {
              $ref: '#/components/schemas/UserPublic',
            },
            player2: {
              $ref: '#/components/schemas/UserPublic',
            },
            player3: {
              $ref: '#/components/schemas/UserPublic',
            },
            player4: {
              $ref: '#/components/schemas/UserPublic',
            },
            winnerId: {
              type: 'string',
              format: 'uuid',
              nullable: true,
            },
            createdAt: {
              type: 'string',
              format: 'date-time',
            },
            startedAt: {
              type: 'string',
              format: 'date-time',
              nullable: true,
            },
            endedAt: {
              type: 'string',
              format: 'date-time',
              nullable: true,
            },
          },
        },
        GameSummary: {
          type: 'object',
          description: 'Abbreviated game info for listings',
          properties: {
            id: {
              type: 'string',
              format: 'uuid',
            },
            boardType: {
              type: 'string',
            },
            status: {
              type: 'string',
            },
            winnerId: {
              type: 'string',
              format: 'uuid',
              nullable: true,
            },
            endedAt: {
              type: 'string',
              format: 'date-time',
              nullable: true,
            },
            player1Id: {
              type: 'string',
              format: 'uuid',
            },
            player2Id: {
              type: 'string',
              format: 'uuid',
              nullable: true,
            },
          },
        },
        Move: {
          type: 'object',
          properties: {
            id: {
              type: 'string',
              format: 'uuid',
            },
            gameId: {
              type: 'string',
              format: 'uuid',
            },
            moveNumber: {
              type: 'integer',
            },
            player: {
              type: 'object',
              properties: {
                id: {
                  type: 'string',
                  format: 'uuid',
                },
                username: {
                  type: 'string',
                },
              },
            },
            moveData: {
              type: 'object',
              description: 'Move-specific data',
            },
            createdAt: {
              type: 'string',
              format: 'date-time',
            },
          },
        },
        UpdateProfileRequest: {
          type: 'object',
          properties: {
            username: {
              type: 'string',
              minLength: 3,
              maxLength: 20,
              pattern: '^[a-zA-Z0-9_-]+$',
            },
            email: {
              type: 'string',
              format: 'email',
              maxLength: 255,
            },
            preferences: {
              type: 'object',
              properties: {
                boardTheme: {
                  type: 'string',
                },
                pieceStyle: {
                  type: 'string',
                },
                soundEnabled: {
                  type: 'boolean',
                },
                animationsEnabled: {
                  type: 'boolean',
                },
                highlightLastMove: {
                  type: 'boolean',
                },
                confirmMoves: {
                  type: 'boolean',
                },
                timeZone: {
                  type: 'string',
                },
                language: {
                  type: 'string',
                },
              },
            },
          },
        },
        LeaderboardEntry: {
          type: 'object',
          properties: {
            id: {
              type: 'string',
              format: 'uuid',
            },
            username: {
              type: 'string',
            },
            rating: {
              type: 'integer',
            },
            gamesPlayed: {
              type: 'integer',
            },
            gamesWon: {
              type: 'integer',
            },
            rank: {
              type: 'integer',
            },
            winRate: {
              type: 'number',
              format: 'float',
            },
          },
        },
      },
      responses: {
        BadRequest: {
          description: 'Invalid request parameters',
          content: {
            'application/json': {
              schema: {
                $ref: '#/components/schemas/Error',
              },
              example: {
                success: false,
                error: {
                  code: 'VALIDATION_INVALID_REQUEST',
                  message: 'Invalid request',
                },
              },
            },
          },
        },
        Unauthorized: {
          description: 'Authentication required or invalid',
          content: {
            'application/json': {
              schema: {
                $ref: '#/components/schemas/Error',
              },
              example: {
                success: false,
                error: {
                  code: 'AUTH_TOKEN_INVALID',
                  message: 'Invalid authentication token',
                },
              },
            },
          },
        },
        Forbidden: {
          description: 'Access denied',
          content: {
            'application/json': {
              schema: {
                $ref: '#/components/schemas/Error',
              },
              example: {
                success: false,
                error: {
                  code: 'RESOURCE_ACCESS_DENIED',
                  message: 'Access denied',
                },
              },
            },
          },
        },
        NotFound: {
          description: 'Resource not found',
          content: {
            'application/json': {
              schema: {
                $ref: '#/components/schemas/Error',
              },
              example: {
                success: false,
                error: {
                  code: 'RESOURCE_NOT_FOUND',
                  message: 'Resource not found',
                },
              },
            },
          },
        },
        Conflict: {
          description: 'Resource conflict (e.g., duplicate)',
          content: {
            'application/json': {
              schema: {
                $ref: '#/components/schemas/Error',
              },
              example: {
                success: false,
                error: {
                  code: 'RESOURCE_ALREADY_EXISTS',
                  message: 'Resource already exists',
                },
              },
            },
          },
        },
        TooManyRequests: {
          description: 'Rate limit exceeded',
          headers: {
            'X-RateLimit-Limit': {
              schema: {
                type: 'integer',
              },
              description: 'Request limit',
            },
            'X-RateLimit-Remaining': {
              schema: {
                type: 'integer',
              },
              description: 'Remaining requests',
            },
            'X-RateLimit-Reset': {
              schema: {
                type: 'integer',
              },
              description: 'Reset time (Unix epoch)',
            },
          },
          content: {
            'application/json': {
              schema: {
                $ref: '#/components/schemas/Error',
              },
              example: {
                success: false,
                error: {
                  code: 'RATE_LIMIT_EXCEEDED',
                  message: 'Rate limit exceeded. Please try again later.',
                },
              },
            },
          },
        },
        InternalError: {
          description: 'Internal server error',
          content: {
            'application/json': {
              schema: {
                $ref: '#/components/schemas/Error',
              },
              example: {
                success: false,
                error: {
                  code: 'SERVER_INTERNAL_ERROR',
                  message: 'Internal server error',
                },
              },
            },
          },
        },
        ServiceUnavailable: {
          description: 'Service temporarily unavailable',
          content: {
            'application/json': {
              schema: {
                $ref: '#/components/schemas/Error',
              },
              example: {
                success: false,
                error: {
                  code: 'SERVER_DATABASE_UNAVAILABLE',
                  message: 'Database not available',
                },
              },
            },
          },
        },
      },
    },
    security: [
      {
        bearerAuth: [],
      },
    ],
  },
  // Path to route files with JSDoc annotations
  apis: ['./src/server/routes/*.ts'],
};

/**
 * Generated OpenAPI specification.
 * Use this with swagger-ui-express to serve interactive documentation.
 */
export const swaggerSpec = swaggerJsdoc(options) as object;

export default swaggerSpec;
