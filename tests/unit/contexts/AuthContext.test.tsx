/**
 * AuthContext Unit Tests
 *
 * Tests the AuthContext provider and useAuth hook for:
 * - Initial state management
 * - Login flow (success and failure)
 * - Register flow (success and failure)
 * - Logout flow
 * - Token persistence and restoration
 * - User profile updates
 * - Error handling
 */

import React from 'react';
import { render, screen, waitFor, act, renderHook } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { AuthProvider, useAuth } from '../../../src/client/contexts/AuthContext';
import { authApi } from '../../../src/client/services/api';
import type { User } from '../../../src/shared/types/user';

// Mock the API module
jest.mock('../../../src/client/services/api', () => ({
  authApi: {
    login: jest.fn(),
    register: jest.fn(),
    getProfile: jest.fn(),
    updateProfile: jest.fn(),
  },
}));

// Create mock user factory
function createMockUser(overrides: Partial<User> = {}): User {
  return {
    id: 'user-123',
    username: 'testuser',
    email: 'test@example.com',
    role: 'player',
    rating: 1200,
    gamesPlayed: 10,
    gamesWon: 5,
    createdAt: new Date('2024-01-01'),
    lastActive: new Date('2024-01-15'),
    status: 'online',
    preferences: {
      boardTheme: 'classic',
      pieceStyle: 'traditional',
      soundEnabled: true,
      animationsEnabled: true,
      autoPromoteQueen: true,
      showCoordinates: true,
      highlightLastMove: true,
      confirmMoves: false,
      timeZone: 'UTC',
      language: 'en',
    },
    ...overrides,
  };
}

// Test component that exposes AuthContext values
function TestConsumer() {
  const { user, isLoading, login, register, logout, updateUser } = useAuth();

  return (
    <div>
      <span data-testid="loading">{isLoading ? 'loading' : 'ready'}</span>
      <span data-testid="user">{user ? user.username : 'anonymous'}</span>
      <span data-testid="email">{user ? user.email : 'no-email'}</span>
      <button data-testid="login-btn" onClick={() => login('test@example.com', 'password123')}>
        Login
      </button>
      <button
        data-testid="register-btn"
        onClick={() => register('test@example.com', 'testuser', 'password123', 'password123')}
      >
        Register
      </button>
      <button data-testid="logout-btn" onClick={logout}>
        Logout
      </button>
      <button data-testid="update-btn" onClick={() => updateUser({ username: 'updateduser' })}>
        Update
      </button>
    </div>
  );
}

describe('AuthContext', () => {
  const mockAuthApi = authApi as jest.Mocked<typeof authApi>;
  const localStorageMock: Record<string, string> = {};

  beforeEach(() => {
    jest.clearAllMocks();

    // Reset localStorage mock
    Object.keys(localStorageMock).forEach((key) => delete localStorageMock[key]);

    // Mock localStorage
    jest.spyOn(Storage.prototype, 'getItem').mockImplementation((key) => {
      return localStorageMock[key] || null;
    });
    jest.spyOn(Storage.prototype, 'setItem').mockImplementation((key, value) => {
      localStorageMock[key] = value;
    });
    jest.spyOn(Storage.prototype, 'removeItem').mockImplementation((key) => {
      delete localStorageMock[key];
    });
  });

  afterEach(() => {
    jest.restoreAllMocks();
  });

  describe('Initial State', () => {
    it('provides initial null user state when no token exists', async () => {
      render(
        <AuthProvider>
          <TestConsumer />
        </AuthProvider>
      );

      // Wait for loading to complete
      await waitFor(() => {
        expect(screen.getByTestId('loading')).toHaveTextContent('ready');
      });

      expect(screen.getByTestId('user')).toHaveTextContent('anonymous');
    });

    it('starts loading and then becomes ready', async () => {
      // Since the loading state transition is synchronous when no token exists,
      // we verify the final state rather than the initial loading state
      render(
        <AuthProvider>
          <TestConsumer />
        </AuthProvider>
      );

      // Should eventually be ready (not loading)
      await waitFor(() => {
        expect(screen.getByTestId('loading')).toHaveTextContent('ready');
      });
    });

    it('restores user from token on mount when token exists', async () => {
      const mockUser = createMockUser();
      localStorageMock['token'] = 'valid-jwt-token';
      mockAuthApi.getProfile.mockResolvedValue(mockUser);

      render(
        <AuthProvider>
          <TestConsumer />
        </AuthProvider>
      );

      await waitFor(() => {
        expect(screen.getByTestId('loading')).toHaveTextContent('ready');
      });

      expect(screen.getByTestId('user')).toHaveTextContent('testuser');
      expect(mockAuthApi.getProfile).toHaveBeenCalledTimes(1);
    });

    it('clears invalid token when profile fetch fails', async () => {
      localStorageMock['token'] = 'invalid-token';
      mockAuthApi.getProfile.mockRejectedValue(new Error('Unauthorized'));

      render(
        <AuthProvider>
          <TestConsumer />
        </AuthProvider>
      );

      await waitFor(() => {
        expect(screen.getByTestId('loading')).toHaveTextContent('ready');
      });

      expect(screen.getByTestId('user')).toHaveTextContent('anonymous');
      expect(localStorageMock['token']).toBeUndefined();
    });
  });

  describe('Login Flow', () => {
    it('successfully logs in and updates user state', async () => {
      const mockUser = createMockUser();
      const mockToken = 'jwt-token-123';
      mockAuthApi.login.mockResolvedValue({ user: mockUser, token: mockToken });

      render(
        <AuthProvider>
          <TestConsumer />
        </AuthProvider>
      );

      await waitFor(() => {
        expect(screen.getByTestId('loading')).toHaveTextContent('ready');
      });

      const user = userEvent.setup();
      await user.click(screen.getByTestId('login-btn'));

      await waitFor(() => {
        expect(screen.getByTestId('user')).toHaveTextContent('testuser');
      });

      expect(mockAuthApi.login).toHaveBeenCalledWith('test@example.com', 'password123');
      expect(localStorageMock['token']).toBe(mockToken);
    });

    it('handles login failure by throwing error', async () => {
      const loginError = new Error('Invalid credentials');
      mockAuthApi.login.mockRejectedValue(loginError);

      // Use renderHook to test error handling
      const wrapper = ({ children }: { children: React.ReactNode }) => (
        <AuthProvider>{children}</AuthProvider>
      );

      const { result } = renderHook(() => useAuth(), { wrapper });

      // Wait for initial loading to complete
      await waitFor(() => {
        expect(result.current.isLoading).toBe(false);
      });

      // Login should reject
      await expect(result.current.login('test@example.com', 'wrongpassword')).rejects.toThrow(
        'Invalid credentials'
      );

      // User should remain null
      expect(result.current.user).toBeNull();
    });

    it('persists token to localStorage on successful login', async () => {
      const mockUser = createMockUser();
      const mockToken = 'persisted-token';
      mockAuthApi.login.mockResolvedValue({ user: mockUser, token: mockToken });

      const wrapper = ({ children }: { children: React.ReactNode }) => (
        <AuthProvider>{children}</AuthProvider>
      );

      const { result } = renderHook(() => useAuth(), { wrapper });

      await waitFor(() => {
        expect(result.current.isLoading).toBe(false);
      });

      await act(async () => {
        await result.current.login('test@example.com', 'password123');
      });

      expect(localStorage.setItem).toHaveBeenCalledWith('token', mockToken);
    });
  });

  describe('Register Flow', () => {
    it('successfully registers and updates user state', async () => {
      const mockUser = createMockUser({ username: 'newuser' });
      const mockToken = 'new-user-token';
      mockAuthApi.register.mockResolvedValue({ user: mockUser, token: mockToken });

      render(
        <AuthProvider>
          <TestConsumer />
        </AuthProvider>
      );

      await waitFor(() => {
        expect(screen.getByTestId('loading')).toHaveTextContent('ready');
      });

      const user = userEvent.setup();
      await user.click(screen.getByTestId('register-btn'));

      await waitFor(() => {
        expect(screen.getByTestId('user')).toHaveTextContent('newuser');
      });

      expect(mockAuthApi.register).toHaveBeenCalledWith(
        'test@example.com',
        'testuser',
        'password123',
        'password123'
      );
      expect(localStorageMock['token']).toBe(mockToken);
    });

    it('handles registration failure by throwing error', async () => {
      const registerError = new Error('Email already exists');
      mockAuthApi.register.mockRejectedValue(registerError);

      const wrapper = ({ children }: { children: React.ReactNode }) => (
        <AuthProvider>{children}</AuthProvider>
      );

      const { result } = renderHook(() => useAuth(), { wrapper });

      await waitFor(() => {
        expect(result.current.isLoading).toBe(false);
      });

      await expect(
        result.current.register('existing@example.com', 'user', 'pass', 'pass')
      ).rejects.toThrow('Email already exists');

      expect(result.current.user).toBeNull();
    });
  });

  describe('Logout Flow', () => {
    it('successfully logs out and clears user state', async () => {
      const mockUser = createMockUser();
      localStorageMock['token'] = 'existing-token';
      mockAuthApi.getProfile.mockResolvedValue(mockUser);

      render(
        <AuthProvider>
          <TestConsumer />
        </AuthProvider>
      );

      // Wait for user to be restored
      await waitFor(() => {
        expect(screen.getByTestId('user')).toHaveTextContent('testuser');
      });

      const user = userEvent.setup();
      await user.click(screen.getByTestId('logout-btn'));

      expect(screen.getByTestId('user')).toHaveTextContent('anonymous');
      expect(localStorageMock['token']).toBeUndefined();
    });

    it('removes token from localStorage on logout', async () => {
      const mockUser = createMockUser();
      localStorageMock['token'] = 'token-to-remove';
      mockAuthApi.getProfile.mockResolvedValue(mockUser);

      const wrapper = ({ children }: { children: React.ReactNode }) => (
        <AuthProvider>{children}</AuthProvider>
      );

      const { result } = renderHook(() => useAuth(), { wrapper });

      await waitFor(() => {
        expect(result.current.user).not.toBeNull();
      });

      act(() => {
        result.current.logout();
      });

      expect(localStorage.removeItem).toHaveBeenCalledWith('token');
      expect(result.current.user).toBeNull();
    });
  });

  describe('User Updates', () => {
    it('updates user state with partial data', async () => {
      const mockUser = createMockUser();
      localStorageMock['token'] = 'valid-token';
      mockAuthApi.getProfile.mockResolvedValue(mockUser);

      render(
        <AuthProvider>
          <TestConsumer />
        </AuthProvider>
      );

      await waitFor(() => {
        expect(screen.getByTestId('user')).toHaveTextContent('testuser');
      });

      const user = userEvent.setup();
      await user.click(screen.getByTestId('update-btn'));

      expect(screen.getByTestId('user')).toHaveTextContent('updateduser');
      // Email should remain unchanged
      expect(screen.getByTestId('email')).toHaveTextContent('test@example.com');
    });

    it('does not update when no user is logged in', async () => {
      const wrapper = ({ children }: { children: React.ReactNode }) => (
        <AuthProvider>{children}</AuthProvider>
      );

      const { result } = renderHook(() => useAuth(), { wrapper });

      await waitFor(() => {
        expect(result.current.isLoading).toBe(false);
      });

      // Should not throw when updating with no user
      act(() => {
        result.current.updateUser({ username: 'newname' });
      });

      expect(result.current.user).toBeNull();
    });

    it('merges update data with existing user data', async () => {
      const mockUser = createMockUser({ rating: 1500, gamesPlayed: 20 });
      localStorageMock['token'] = 'valid-token';
      mockAuthApi.getProfile.mockResolvedValue(mockUser);

      const wrapper = ({ children }: { children: React.ReactNode }) => (
        <AuthProvider>{children}</AuthProvider>
      );

      const { result } = renderHook(() => useAuth(), { wrapper });

      await waitFor(() => {
        expect(result.current.user).not.toBeNull();
      });

      act(() => {
        result.current.updateUser({ rating: 1600 });
      });

      expect(result.current.user?.rating).toBe(1600);
      expect(result.current.user?.gamesPlayed).toBe(20); // Unchanged
      expect(result.current.user?.username).toBe('testuser'); // Unchanged
    });
  });

  describe('Hook Usage', () => {
    it('throws error when useAuth is used outside AuthProvider', () => {
      // Suppress console.error for this test
      const consoleSpy = jest.spyOn(console, 'error').mockImplementation(() => {});

      expect(() => {
        renderHook(() => useAuth());
      }).toThrow('useAuth must be used within an AuthProvider');

      consoleSpy.mockRestore();
    });

    it('provides all required context methods', async () => {
      const wrapper = ({ children }: { children: React.ReactNode }) => (
        <AuthProvider>{children}</AuthProvider>
      );

      const { result } = renderHook(() => useAuth(), { wrapper });

      await waitFor(() => {
        expect(result.current.isLoading).toBe(false);
      });

      // Verify all methods are available and are functions
      expect(typeof result.current.login).toBe('function');
      expect(typeof result.current.logout).toBe('function');
      expect(typeof result.current.register).toBe('function');
      expect(typeof result.current.updateUser).toBe('function');
    });
  });

  describe('Edge Cases', () => {
    it('handles concurrent login attempts gracefully', async () => {
      const mockUser = createMockUser();
      const mockToken = 'token';
      mockAuthApi.login.mockResolvedValue({ user: mockUser, token: mockToken });

      const wrapper = ({ children }: { children: React.ReactNode }) => (
        <AuthProvider>{children}</AuthProvider>
      );

      const { result } = renderHook(() => useAuth(), { wrapper });

      await waitFor(() => {
        expect(result.current.isLoading).toBe(false);
      });

      // Fire multiple login attempts concurrently
      await act(async () => {
        await Promise.all([
          result.current.login('test@example.com', 'password'),
          result.current.login('test@example.com', 'password'),
        ]);
      });

      // Should still be in valid state
      expect(result.current.user).toEqual(mockUser);
    });

    it('handles login after logout correctly', async () => {
      const mockUser1 = createMockUser({ id: 'user-1', username: 'user1' });
      const mockUser2 = createMockUser({ id: 'user-2', username: 'user2' });

      mockAuthApi.login
        .mockResolvedValueOnce({ user: mockUser1, token: 'token1' })
        .mockResolvedValueOnce({ user: mockUser2, token: 'token2' });

      const wrapper = ({ children }: { children: React.ReactNode }) => (
        <AuthProvider>{children}</AuthProvider>
      );

      const { result } = renderHook(() => useAuth(), { wrapper });

      await waitFor(() => {
        expect(result.current.isLoading).toBe(false);
      });

      // Login as first user
      await act(async () => {
        await result.current.login('user1@example.com', 'password');
      });
      expect(result.current.user?.username).toBe('user1');

      // Logout
      act(() => {
        result.current.logout();
      });
      expect(result.current.user).toBeNull();

      // Login as second user
      await act(async () => {
        await result.current.login('user2@example.com', 'password');
      });
      expect(result.current.user?.username).toBe('user2');
    });

    it('handles empty token in localStorage', async () => {
      localStorageMock['token'] = '';

      render(
        <AuthProvider>
          <TestConsumer />
        </AuthProvider>
      );

      await waitFor(() => {
        expect(screen.getByTestId('loading')).toHaveTextContent('ready');
      });

      // Empty string is falsy, so should not call getProfile
      expect(mockAuthApi.getProfile).not.toHaveBeenCalled();
      expect(screen.getByTestId('user')).toHaveTextContent('anonymous');
    });
  });
});
