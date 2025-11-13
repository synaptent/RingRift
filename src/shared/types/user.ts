export type UserRole = 'player' | 'moderator' | 'admin';
export type UserStatus = 'online' | 'offline' | 'in_game' | 'away';

export interface User {
  id: string;
  username: string;
  email: string;
  role: UserRole;
  rating: number;
  gamesPlayed: number;
  gamesWon: number;
  createdAt: Date;
  lastActive: Date;
  status: UserStatus;
  avatar?: string;
  preferences: UserPreferences;
}

export interface UserPreferences {
  boardTheme: string;
  pieceStyle: string;
  soundEnabled: boolean;
  animationsEnabled: boolean;
  autoPromoteQueen: boolean;
  showCoordinates: boolean;
  highlightLastMove: boolean;
  confirmMoves: boolean;
  timeZone: string;
  language: string;
}

export interface UserStats {
  userId: string;
  totalGames: number;
  wins: number;
  losses: number;
  draws: number;
  winRate: number;
  averageGameLength: number;
  favoriteBoard: string;
  currentStreak: number;
  longestStreak: number;
  ratingHistory: RatingPoint[];
  achievements: Achievement[];
}

export interface RatingPoint {
  rating: number;
  timestamp: Date;
  gameId: string;
  change: number;
}

export interface Achievement {
  id: string;
  name: string;
  description: string;
  icon: string;
  unlockedAt: Date;
  rarity: 'common' | 'rare' | 'epic' | 'legendary';
}

export interface AuthTokens {
  accessToken: string;
  refreshToken: string;
  expiresIn: number;
}

export interface JWTPayload {
  userId: string;
  username: string;
  role: UserRole;
  iat: number;
  exp: number;
}

export interface LoginRequest {
  email: string;
  password: string;
  rememberMe?: boolean;
}

export interface RegisterRequest {
  username: string;
  email: string;
  password: string;
  confirmPassword: string;
}

export interface UpdateProfileRequest {
  username?: string;
  email?: string;
  preferences?: Partial<UserPreferences>;
}

export interface ChangePasswordRequest {
  currentPassword: string;
  newPassword: string;
  confirmPassword: string;
}

// Default user preferences
export const DEFAULT_USER_PREFERENCES: UserPreferences = {
  boardTheme: 'classic',
  pieceStyle: 'traditional',
  soundEnabled: true,
  animationsEnabled: true,
  autoPromoteQueen: true,
  showCoordinates: true,
  highlightLastMove: true,
  confirmMoves: false,
  timeZone: 'UTC',
  language: 'en'
};

// Rating system constants
export const RATING_CONSTANTS = {
  INITIAL_RATING: 1200,
  MIN_RATING: 100,
  MAX_RATING: 3000,
  K_FACTOR: 32, // For Elo rating calculation
  PROVISIONAL_GAMES: 20 // Games before rating stabilizes
} as const;