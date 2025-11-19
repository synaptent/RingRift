import axios from 'axios';
import { User } from '../../shared/types/user';
import { Game, CreateGameRequest } from '../../shared/types/game';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:5000/api';

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
      window.location.href = '/login';
    }
    return Promise.reject(error);
  }
);

// Auth API
export const authApi = {
  async login(email: string, password: string) {
    const response = await api.post('/auth/login', { email, password });
    return response.data;
  },

  async register(email: string, username: string, password: string) {
    const response = await api.post('/auth/register', { email, username, password });
    return response.data;
  },

  async getProfile(): Promise<User> {
    const response = await api.get('/auth/profile');
    return response.data;
  },

  async updateProfile(userData: Partial<User>): Promise<User> {
    const response = await api.put('/auth/profile', userData);
    return response.data;
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
    return response.data;
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
