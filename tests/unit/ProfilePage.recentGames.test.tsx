import React from 'react';
import { render, screen, waitFor } from '@testing-library/react';
import '@testing-library/jest-dom';
import { BrowserRouter } from 'react-router-dom';
import ProfilePage from '../../src/client/pages/ProfilePage';
import { authApi, gameApi, userApi } from '../../src/client/services/api';
import type { User } from '../../src/shared/types/user';
import type { GameResult } from '../../src/shared/types/game';

jest.mock('../../src/client/services/api');
jest.mock('../../src/client/hooks/useDocumentTitle', () => ({
  useDocumentTitle: jest.fn(),
}));

const mockUseAuth = jest.fn();
jest.mock('../../src/client/contexts/AuthContext', () => ({
  useAuth: () => mockUseAuth(),
}));

function createMockUser(overrides: Partial<User> = {}): User {
  return {
    id: 'user-1',
    username: 'TestUser',
    email: 'test@example.com',
    createdAt: new Date().toISOString(),
    updatedAt: new Date().toISOString(),
    rating: 1200,
    gamesPlayed: 10,
    gamesWon: 6,
    ...overrides,
  } as User;
}

type ResultReason = GameResult['reason'];

function createMockGame(overrides: Partial<any> = {}, resultReason?: ResultReason) {
  return {
    id: 'game-1',
    boardType: 'square8',
    status: 'completed',
    winnerId: 'user-1',
    createdAt: new Date().toISOString(),
    endedAt: new Date().toISOString(),
    maxPlayers: 2,
    player1Id: 'user-1',
    player2Id: 'user-2',
    resultReason,
    ...overrides,
  };
}

describe('ProfilePage recent games reason labels', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    mockUseAuth.mockReturnValue({ user: createMockUser() });
  });

  it.each([
    { reason: 'ring_elimination', expected: 'Ring Elimination' },
    { reason: 'territory_control', expected: 'Territory Control' },
    { reason: 'timeout', expected: 'Timeout' },
    { reason: 'resignation', expected: 'Resignation' },
    { reason: 'abandonment', expected: 'Abandonment' },
  ] as { reason: ResultReason; expected: string }[])(
    'renders "$expected" label for resultReason=$reason in recent games',
    async ({ reason, expected }) => {
      const user = createMockUser();
      const game = createMockGame({}, reason);

      (authApi.getProfile as jest.Mock).mockResolvedValue(user);
      (gameApi.getGames as jest.Mock).mockResolvedValue({
        games: [game],
        total: 1,
        page: 1,
        totalPages: 1,
      });
      (gameApi.getUserGames as jest.Mock).mockResolvedValue({
        games: [],
        pagination: { total: 0, limit: 100, offset: 0, hasMore: false },
      });
      (userApi.getStats as jest.Mock).mockResolvedValue({ ratingHistory: [] });

      render(
        <BrowserRouter>
          <ProfilePage />
        </BrowserRouter>
      );

      await waitFor(() => {
        expect(screen.getByText(/Recent Games/i)).toBeInTheDocument();
      });

      // Victory/Defeat label still present (may also match "First Victory" achievement)
      expect(screen.getAllByText(/Victory/i).length).toBeGreaterThanOrEqual(1);

      // Reason-specific label appears alongside board type
      expect(screen.getByText(expected)).toBeInTheDocument();
    }
  );
});
