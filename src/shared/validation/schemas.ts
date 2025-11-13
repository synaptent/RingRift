import { z } from 'zod';
import { Request } from 'express';

// Position validation
export const PositionSchema = z.object({
  x: z.number().int().min(0),
  y: z.number().int().min(0),
  z: z.number().int().min(0).optional()
});

// Move validation
export const MoveSchema = z.object({
  type: z.enum(['place_ring', 'move_ring', 'place_marker', 'remove_row']),
  player: z.number().int().min(1).max(4),
  from: PositionSchema.optional(),
  to: PositionSchema,
  removedRow: z.array(PositionSchema).optional(),
  thinkTime: z.number().min(0).max(300000) // Max 5 minutes think time
});

// Game creation validation
export const CreateGameSchema = z.object({
  boardType: z.enum(['square8', 'square19', 'hexagonal']),
  timeControl: z.object({
    initialTime: z.number().min(60).max(7200), // 1 minute to 2 hours
    increment: z.number().min(0).max(60) // Max 60 seconds increment
  }),
  isRated: z.boolean().default(true),
  isPrivate: z.boolean().default(false),
  maxPlayers: z.number().min(2).max(4).default(2),
  aiOpponents: z.object({
    count: z.number().min(0).max(3),
    difficulty: z.array(z.number().min(1).max(10))
  }).optional()
});

// User registration validation
export const RegisterSchema = z.object({
  username: z.string()
    .min(3, 'Username must be at least 3 characters')
    .max(20, 'Username must be at most 20 characters')
    .regex(/^[a-zA-Z0-9_-]+$/, 'Username can only contain letters, numbers, underscores, and hyphens'),
  email: z.string()
    .email('Invalid email address')
    .max(255, 'Email must be at most 255 characters'),
  password: z.string()
    .min(8, 'Password must be at least 8 characters')
    .max(128, 'Password must be at most 128 characters')
    .regex(/^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)/, 'Password must contain at least one lowercase letter, one uppercase letter, and one number'),
  confirmPassword: z.string()
}).refine((data: any) => data.password === data.confirmPassword, {
  message: "Passwords don't match",
  path: ["confirmPassword"]
});

// User login validation
export const LoginSchema = z.object({
  email: z.string().email('Invalid email address'),
  password: z.string().min(1, 'Password is required'),
  rememberMe: z.boolean().optional()
});

// Profile update validation
export const UpdateProfileSchema = z.object({
  username: z.string()
    .min(3, 'Username must be at least 3 characters')
    .max(20, 'Username must be at most 20 characters')
    .regex(/^[a-zA-Z0-9_-]+$/, 'Username can only contain letters, numbers, underscores, and hyphens')
    .optional(),
  email: z.string()
    .email('Invalid email address')
    .max(255, 'Email must be at most 255 characters')
    .optional(),
  preferences: z.object({
    boardTheme: z.string().optional(),
    pieceStyle: z.string().optional(),
    soundEnabled: z.boolean().optional(),
    animationsEnabled: z.boolean().optional(),
    autoPromoteQueen: z.boolean().optional(),
    showCoordinates: z.boolean().optional(),
    highlightLastMove: z.boolean().optional(),
    confirmMoves: z.boolean().optional(),
    timeZone: z.string().optional(),
    language: z.string().optional()
  }).optional()
});

// Password change validation
export const ChangePasswordSchema = z.object({
  currentPassword: z.string().min(1, 'Current password is required'),
  newPassword: z.string()
    .min(8, 'Password must be at least 8 characters')
    .max(128, 'Password must be at most 128 characters')
    .regex(/^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)/, 'Password must contain at least one lowercase letter, one uppercase letter, and one number'),
  confirmPassword: z.string()
}).refine((data: any) => data.newPassword === data.confirmPassword, {
  message: "Passwords don't match",
  path: ["confirmPassword"]
});

// Chat message validation
export const ChatMessageSchema = z.object({
  gameId: z.string().uuid('Invalid game ID'),
  content: z.string()
    .min(1, 'Message cannot be empty')
    .max(500, 'Message must be at most 500 characters')
    .trim(),
  type: z.enum(['game', 'spectator', 'private']).default('game'),
  recipientId: z.string().uuid().optional()
});

// Matchmaking preferences validation
export const MatchmakingPreferencesSchema = z.object({
  boardType: z.enum(['square8', 'square19', 'hexagonal']),
  timeControl: z.object({
    min: z.number().min(60).max(7200),
    max: z.number().min(60).max(7200)
  }).refine((data: any) => data.min <= data.max, {
    message: "Minimum time must be less than or equal to maximum time"
  }),
  ratingRange: z.object({
    min: z.number().min(100).max(3000),
    max: z.number().min(100).max(3000)
  }).refine((data: any) => data.min <= data.max, {
    message: "Minimum rating must be less than or equal to maximum rating"
  }),
  allowAI: z.boolean().default(false)
});

// Tournament creation validation
export const CreateTournamentSchema = z.object({
  name: z.string()
    .min(3, 'Tournament name must be at least 3 characters')
    .max(100, 'Tournament name must be at most 100 characters'),
  format: z.enum(['single_elimination', 'double_elimination', 'round_robin', 'swiss']),
  boardType: z.enum(['square8', 'square19', 'hexagonal']),
  maxParticipants: z.number().min(4).max(256),
  timeControl: z.object({
    initialTime: z.number().min(60).max(7200),
    increment: z.number().min(0).max(60)
  }),
  isRated: z.boolean().default(true),
  entryFee: z.number().min(0).optional(),
  prizePool: z.number().min(0).optional(),
  startsAt: z.date().min(new Date(), 'Tournament start time must be in the future'),
  registrationDeadline: z.date().optional()
}).refine((data: any) => {
  if (data.registrationDeadline) {
    return data.registrationDeadline <= data.startsAt;
  }
  return true;
}, {
  message: "Registration deadline must be before tournament start time",
  path: ["registrationDeadline"]
});

// WebSocket event validation
export const SocketEventSchema = z.object({
  event: z.string().min(1),
  data: z.any(),
  timestamp: z.date().optional()
});

// Game state validation (for API responses)
export const GameStateSchema = z.object({
  id: z.string().uuid(),
  boardType: z.enum(['square8', 'square19', 'hexagonal']),
  players: z.array(z.object({
    id: z.string().uuid(),
    username: z.string(),
    type: z.enum(['human', 'ai']),
    playerNumber: z.number().int().min(1).max(4),
    rating: z.number().optional(),
    isReady: z.boolean(),
    timeRemaining: z.number(),
    aiDifficulty: z.number().min(1).max(10).optional()
  })),
  currentPhase: z.enum(['setup', 'placement', 'movement']),
  currentPlayer: z.number().int().min(1).max(4),
  gameStatus: z.enum(['waiting', 'active', 'finished', 'paused', 'abandoned']),
  winner: z.number().int().min(1).max(4).optional(),
  isRated: z.boolean(),
  maxPlayers: z.number().min(2).max(4)
});

// Pagination validation
export const PaginationSchema = z.object({
  page: z.number().int().min(1).default(1),
  limit: z.number().int().min(1).max(100).default(20),
  sortBy: z.string().optional(),
  sortOrder: z.enum(['asc', 'desc']).default('desc')
});

// Search validation
export const SearchSchema = z.object({
  query: z.string().min(1).max(100),
  type: z.enum(['users', 'games', 'tournaments']).optional(),
  filters: z.record(z.any()).optional()
});

// File upload validation
export const FileUploadSchema = z.object({
  filename: z.string().min(1).max(255),
  mimetype: z.string().regex(/^image\/(jpeg|png|gif|webp)$/, 'Only image files are allowed'),
  size: z.number().max(5 * 1024 * 1024, 'File size must be less than 5MB')
});

// API response validation helpers
export const createSuccessResponse = <T>(data: T) => ({
  success: true,
  data,
  timestamp: new Date()
});

export const createErrorResponse = (message: string, code?: string, details?: any) => ({
  success: false,
  error: {
    message,
    code,
    details,
    timestamp: new Date()
  }
});

// Validation middleware helper types
export type ValidationSchema = z.ZodSchema<any>;
export type ValidatedRequest<T> = Request & { validatedData: T };

// Common validation patterns
export const UUID_REGEX = /^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;
export const USERNAME_REGEX = /^[a-zA-Z0-9_-]+$/;
export const EMAIL_REGEX = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
export const PASSWORD_REGEX = /^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)/;

// Validation error types
export class ValidationError extends Error {
  constructor(
    message: string,
    public field?: string,
    public code?: string
  ) {
    super(message);
    this.name = 'ValidationError';
  }
}

export class AuthenticationError extends Error {
  constructor(message: string = 'Authentication required') {
    super(message);
    this.name = 'AuthenticationError';
  }
}

export class AuthorizationError extends Error {
  constructor(message: string = 'Insufficient permissions') {
    super(message);
    this.name = 'AuthorizationError';
  }
}