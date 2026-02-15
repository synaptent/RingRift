import axios from 'axios';
import { User } from '../../shared/types/user';
import {
  Game,
  CreateGameRequest,
  BoardType,
  GameStatus,
  GameResult,
  Move,
} from '../../shared/types/game';
import { readEnv } from '../../shared/utils/envFlags';
import { AUTH_UNAUTHORIZED_EVENT } from '../utils/authEvents';

/**
 * Response format for game history endpoint
 */
export interface GameHistoryMove {
  moveNumber: number;
  playerId: string;
  playerName: string;
  moveType: string;
  moveData: Record<string, unknown>;
  timestamp: string;
  /**
   * Present when the move was auto-resolved by the server (for example due to
   * decision-phase timeout). This is a compact projection of the
   * decisionAutoResolved metadata persisted in moveData.decisionAutoResolved.
   */
  autoResolved?: {
    reason: 'timeout' | 'disconnected' | 'fallback';
    choiceKind?: string;
    choiceType?: string;
  };
}

export interface GameHistoryResponse {
  gameId: string;
  moves: GameHistoryMove[];
  totalMoves: number;
  /**
   * Optional terminal GameResult projection when a finished game's history
   * is requested. This allows history consumers to distinguish timeout,
   * resignation, abandonment, and other victory conditions without making
   * a separate call to the game-details endpoint.
   */
  result?: {
    reason: GameResult['reason'];
    /**
     * Winner seat index (1-based) when known. May be null for draws or
     * abandonment without a winner, and omitted entirely when the game has
     * not yet finished.
     */
    winner?: number | null;
  };
}

/**
 * Response format for game details endpoint
 */
export interface GameDetailsResponse {
  id: string;
  status: string;
  boardType: string;
  maxPlayers: number;
  isRated: boolean;
  allowSpectators: boolean;
  players: {
    id: string;
    username: string;
    rating: number;
  }[];
  winner?: {
    id: string;
    username: string;
  } | null;
  createdAt: string;
  updatedAt: string;
  startedAt?: string | null;
  endedAt?: string | null;
  moveCount: number;
}

/**
 * Summary of a game for listing purposes
 *
 * This shape is used by:
 * - Profile/recent-games views
 * - User games API (gameApi.getUserGames)
 * - Backend replay browser as the source of per-game metadata.
 *
 * New optional fields should be added conservatively and only when they can be
 * derived from existing columns on the Game row so that the backend route
 * remains cheap.
 */
export interface GameSummary {
  id: string;
  boardType: BoardType;
  status: GameStatus;
  /** Number of seated players (historical name kept for compatibility). */
  playerCount: number;
  maxPlayers: number;
  winnerId?: string | null;
  winnerName?: string | null;
  createdAt: string;
  endedAt?: string | null;
  moveCount: number;

  /**
   * Alias for playerCount used by replay tooling. When present, this should
   * always equal playerCount.
   */
  numPlayers?: number;

  /** Whether the game was rated. */
  isRated?: boolean;

  /**
   * High-level source of the record, when available:
   * - 'online_game'  – standard backend game
   * - 'self_play'    – imported self-play record
   * - other strings for future extensions (tournament, soak tests, etc.)
   */
  source?: string;

  /**
   * Canonical outcome for the game when available (timeout, resignation,
   * ring_elimination, etc.). Mirrors GameResult['reason'] on the backend.
   */
  outcome?: GameResult['reason'] | string;

  /**
   * Lightweight terminal result reason projected from finalState.gameResult for
   * compatibility with older callers. Prefer `outcome` when present.
   */
  resultReason?: GameResult['reason'] | string;
}

/**
 * Response format for user games endpoint
 */
export interface UserGamesResponse {
  games: GameSummary[];
  pagination: {
    total: number;
    limit: number;
    offset: number;
    hasMore: boolean;
  };
}

/**
 * API base URL:
 * - In typical dev/prod setups, we rely on same-origin `/api` (Vite dev proxy or nginx).
 * - VITE_API_URL is available for advanced setups (e.g. pointing the client at a remote API).
 *   For browser builds, this is injected into the client bundle via `process.env.VITE_API_URL`
 *   in [`vite.config.ts`](vite.config.ts:29).
 */
export function getApiBaseUrl(): string {
  const explicit = readEnv('VITE_API_URL');
  if (explicit && typeof explicit === 'string') {
    return explicit.replace(/\/$/, '');
  }

  // In the browser, prefer same-origin `/api` when no explicit base URL is configured.
  if (typeof window !== 'undefined' && window.location?.origin) {
    const origin = window.location.origin.replace(/\/$/, '');
    return `${origin}/api`;
  }

  // Jest / Node or other non-browser environments fall back to relative `/api`.
  return '/api';
}

const API_BASE_URL = getApiBaseUrl();

// Create axios instance with default config
const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
  // Include cookies in requests (for refresh token httpOnly cookie)
  withCredentials: true,
});

// Track refresh state to avoid concurrent refresh attempts
let isRefreshing = false;
let refreshSubscribers: Array<(token: string) => void> = [];

function subscribeTokenRefresh(cb: (token: string) => void) {
  refreshSubscribers.push(cb);
}

function onTokenRefreshed(token: string) {
  refreshSubscribers.forEach((cb) => cb(token));
  refreshSubscribers = [];
}

// Add auth token to requests
api.interceptors.request.use((config) => {
  const token = localStorage.getItem('token');
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

// Handle auth errors with automatic token refresh
api.interceptors.response.use(
  (response) => response,
  async (error) => {
    const originalRequest = error.config;
    const url: string = originalRequest?.url || '';

    // Don't try to refresh for auth endpoints (login, register, refresh itself)
    const isAuthEndpoint = url.startsWith('/auth');

    if (error.response?.status === 401 && !isAuthEndpoint && !originalRequest._retry) {
      // Try to refresh the token
      if (!isRefreshing) {
        isRefreshing = true;
        originalRequest._retry = true;

        try {
          // Call refresh endpoint - the httpOnly cookie is sent automatically
          const response = await axios.post(
            `${API_BASE_URL}/auth/refresh`,
            {},
            { withCredentials: true }
          );

          const newToken = response.data?.data?.accessToken;
          if (newToken) {
            localStorage.setItem('token', newToken);
            onTokenRefreshed(newToken);
            isRefreshing = false;

            // Retry the original request with new token
            originalRequest.headers.Authorization = `Bearer ${newToken}`;
            return api(originalRequest);
          }
        } catch (_refreshError) {
          isRefreshing = false;
          refreshSubscribers = [];
          // Refresh failed - proceed with 401 handling below
        }
      } else {
        // Another request is already refreshing, queue this one
        return new Promise((resolve) => {
          subscribeTokenRefresh((token: string) => {
            originalRequest.headers.Authorization = `Bearer ${token}`;
            resolve(api(originalRequest));
          });
        });
      }
    }

    // If we get here with a 401, refresh failed or wasn't attempted
    if (error.response?.status === 401) {
      localStorage.removeItem('token');

      // For most 401s (profile, protected game APIs, etc.) we redirect back
      // to /login so that expired/invalid tokens bounce the user cleanly
      // into the auth flow.
      //
      // However, there are a few important exceptions where the caller
      // needs to handle the 401 explicitly without a hard navigation:
      // - Auth endpoints (/auth/login, /auth/register, /auth/refresh):
      //   the login/register pages interpret INVALID_CREDENTIALS and other
      //   error codes to either show inline errors or redirect to the
      //   register flow. A global window.location redirect here would race
      //   with that logic and break E2E flows.
      // - Game creation endpoints under /games (used by the public /sandbox
      //   flow when no user is logged in): callers fall back to a pure
      //   local sandbox rather than being forced to /login.
      const isGameEndpoint = url.startsWith('/games');
      const isAuthEndpoint = url.startsWith('/auth');
      const isRulesUxTelemetryEndpoint = url.startsWith('/telemetry/rules-ux');

      if (!isGameEndpoint && !isAuthEndpoint && !isRulesUxTelemetryEndpoint) {
        try {
          window.dispatchEvent(
            new CustomEvent(AUTH_UNAUTHORIZED_EVENT, {
              detail: { url },
            })
          );
        } catch {
          window.location.href = '/login';
        }
      }
    }
    return Promise.reject(error);
  }
);

// Auth API
export const authApi = {
  async login(
    email: string,
    password: string,
    rememberMe?: boolean
  ): Promise<{ user: User; token: string }> {
    const response = await api.post('/auth/login', { email, password, rememberMe });
    const { data } = response.data;
    return {
      user: data.user as User,
      token: data.accessToken as string,
    };
  },

  async register(
    email: string,
    username: string,
    password: string,
    confirmPassword: string
  ): Promise<{ user: User; token: string }> {
    const response = await api.post('/auth/register', {
      email,
      username,
      password,
      confirmPassword,
    });
    const { data } = response.data;
    return {
      user: data.user as User,
      token: data.accessToken as string,
    };
  },

  async getProfile(): Promise<User> {
    // Backend exposes the authenticated user profile under /api/users/profile
    // and wraps it in the standard { success, data: { user } } envelope.
    const response = await api.get('/users/profile');
    const { data } = response.data;
    return data.user as User;
  },

  async updateProfile(userData: Partial<User>): Promise<User> {
    const response = await api.put('/users/profile', userData);
    const { data } = response.data;
    return data.user as User;
  },
};

// Game API
export const gameApi = {
  /**
   * Create a new backend game using the shared CreateGameRequest
   * shape. The server responds with a standard envelope of the
   * form { success, data: { game }, message }.
   */
  async createGame(gameData: CreateGameRequest): Promise<Game> {
    const response = await api.post('/games', gameData);
    return response.data.data.game as Game;
  },

  async getGame(gameId: string): Promise<Game> {
    const response = await api.get(`/games/${gameId}`);
    return response.data;
  },

  async getGames(params?: {
    status?: string;
    boardType?: string;
    page?: number;
    limit?: number;
  }): Promise<{ games: Game[]; total: number; page: number; totalPages: number }> {
    const response = await api.get('/games', { params });
    return response.data;
  },

  async getAvailableGames(params?: {
    boardType?: string;
    maxPlayers?: number;
  }): Promise<{ games: Game[] }> {
    const response = await api.get('/games/lobby/available', { params });
    const { data } = response.data;
    return {
      games: (data?.games || []) as Game[],
    };
  },

  async joinGame(gameId: string): Promise<Game> {
    const response = await api.post(`/games/${gameId}/join`);
    return response.data;
  },

  async getGameByInvite(inviteCode: string): Promise<{
    id: string;
    inviteCode: string;
    boardType: string;
    maxPlayers: number;
    status: string;
    isRated: boolean;
    playerCount: number;
    players: { id: string; username: string; rating: number }[];
    createdAt: string;
  }> {
    const response = await api.get(`/games/invite/${inviteCode}`);
    return response.data.data.game;
  },

  async joinByInvite(inviteCode: string): Promise<{ id: string }> {
    const response = await api.post(`/games/invite/${inviteCode}/join`);
    return response.data.data.game;
  },

  async leaveGame(gameId: string): Promise<void> {
    await api.post(`/games/${gameId}/leave`);
  },

  async makeMove(gameId: string, move: Move): Promise<void> {
    await api.post(`/games/${gameId}/moves`, { move });
  },

  /**
   * Get the complete move history for a game
   * @param gameId - The ID of the game
   * @returns GameHistoryResponse with all moves
   */
  async getGameHistory(gameId: string): Promise<GameHistoryResponse> {
    const response = await api.get(`/games/${gameId}/history`);
    return response.data.data as GameHistoryResponse;
  },

  /**
   * Get detailed information about a specific game
   * @param gameId - The ID of the game
   * @returns GameDetailsResponse with full game details
   */
  async getGameDetails(gameId: string): Promise<GameDetailsResponse> {
    const response = await api.get(`/games/${gameId}`);
    const game = response.data.data.game;

    // Transform the response to match our typed format
    const players = [game.player1, game.player2, game.player3, game.player4].filter(Boolean);

    return {
      id: game.id,
      status: game.status,
      boardType: game.boardType,
      maxPlayers: game.maxPlayers,
      isRated: game.isRated,
      allowSpectators: game.allowSpectators,
      players,
      winner: game.winner || null,
      createdAt: game.createdAt,
      updatedAt: game.updatedAt,
      startedAt: game.startedAt || null,
      endedAt: game.endedAt || null,
      moveCount: game.moves?.length || 0,
    };
  },

  /**
   * Get list of games for a specific user
   * @param userId - The ID of the user
   * @param options - Optional pagination and filtering options
   * @returns UserGamesResponse with games list and pagination info
   */
  async getUserGames(
    userId: string,
    options?: {
      limit?: number;
      offset?: number;
      status?: string;
    }
  ): Promise<UserGamesResponse> {
    const params = new URLSearchParams();
    if (options?.limit) params.set('limit', options.limit.toString());
    if (options?.offset) params.set('offset', options.offset.toString());
    if (options?.status) params.set('status', options.status);

    const url = `/games/user/${userId}${params.toString() ? `?${params.toString()}` : ''}`;
    const response = await api.get(url);
    return response.data.data as UserGamesResponse;
  },
};

// User API
export const userApi = {
  async getUsers(params?: {
    search?: string;
    page?: number;
    limit?: number;
  }): Promise<{ users: User[]; total: number; page: number; totalPages: number }> {
    const response = await api.get('/users', { params });
    return response.data;
  },

  async getUser(userId: string): Promise<User> {
    const response = await api.get(`/users/${userId}`);
    return response.data;
  },

  async getLeaderboard(params?: {
    page?: number;
    limit?: number;
  }): Promise<{ users: User[]; total: number; page: number; totalPages: number }> {
    const response = await api.get('/users/leaderboard', { params });
    return response.data;
  },

  async getStats(): Promise<{
    ratingHistory: Array<{
      date: string;
      rating: number;
      change: number;
      gameId: string | null;
    }>;
  }> {
    const response = await api.get('/users/stats');
    return response.data;
  },

  async searchUsers(
    q: string,
    limit?: number
  ): Promise<{
    users: Array<{
      id: string;
      username: string;
      rating: number;
      gamesPlayed: number;
      gamesWon: number;
    }>;
  }> {
    const params: Record<string, string | number> = { q };
    if (limit) params.limit = limit;
    const response = await api.get('/users/search', { params });
    return response.data.data;
  },

  async getPublicProfile(userId: string): Promise<{
    user: {
      id: string;
      username: string;
      rating: number;
      gamesPlayed: number;
      gamesWon: number;
      winRate: number;
      isProvisional: boolean;
      memberSince: string;
    };
    recentGames: Array<{
      id: string;
      boardType: string;
      winnerId: string | null;
      endedAt: string | null;
      maxPlayers: number;
      player1: { id: string; username: string } | null;
      player2: { id: string; username: string } | null;
      player3: { id: string; username: string } | null;
      player4: { id: string; username: string } | null;
    }>;
    ratingHistory: Array<{
      date: string;
      rating: number;
      change: number;
    }>;
  }> {
    const response = await api.get(`/users/${userId}/public-profile`);
    return response.data.data;
  },
};

export default api;
