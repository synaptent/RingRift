import axios from 'axios';
import { User } from '../../shared/types/user';
import { Game, CreateGameRequest } from '../../shared/types/game';

/**
 * API base URL:
 * - In typical dev/prod setups, we rely on same-origin `/api` (Vite dev proxy or nginx).
 * - VITE_API_URL is available for advanced setups (e.g. pointing the client at a remote API).
 */
const API_BASE_URL =
  (import.meta as any).env?.VITE_API_URL || '/api';

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

      // For most 401s (auth/profile/etc.) we redirect back to /login.
      // However, for game creation endpoints (used by the public /sandbox
      // flow when no user is logged in), we do NOT redirect; callers
      // are expected to catch the 401 and fall back to a pure local
      // sandbox instead of being bounced back to /login.
      if (!url.startsWith('/games')) {
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
