import axios from 'axios';
import { User } from '../../shared/types/user';
import { Game, CreateGameRequest, BoardType, GameStatus } from '../../shared/types/game';

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
 */
export interface GameSummary {
  id: string;
  boardType: BoardType;
  status: GameStatus;
  playerCount: number;
  maxPlayers: number;
  winnerId?: string | null;
  winnerName?: string | null;
  createdAt: string;
  endedAt?: string | null;
  moveCount: number;
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
function getApiBaseUrl(): string {
  const env =
    typeof process !== 'undefined' && (process as any).env
      ? ((process as any).env as Record<string, string | undefined>)
      : {};

  const explicit = env.VITE_API_URL;
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
});

// Add auth token to requests
api.interceptors.request.use((config) => {
  const token = localStorage.getItem('token');
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

// Handle auth errors
api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      localStorage.removeItem('token');

      const url: string = error.config?.url || '';

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

      if (!isGameEndpoint && !isAuthEndpoint) {
        window.location.href = '/login';
      }
    }
    return Promise.reject(error);
  }
);

// Auth API
export const authApi = {
  async login(email: string, password: string): Promise<{ user: User; token: string }> {
    const response = await api.post('/auth/login', { email, password });
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

  async leaveGame(gameId: string): Promise<void> {
    await api.post(`/games/${gameId}/leave`);
  },

  async makeMove(gameId: string, move: any): Promise<void> {
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
};

export default api;
