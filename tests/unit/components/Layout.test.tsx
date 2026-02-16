import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import '@testing-library/jest-dom';
import Layout from '../../../src/client/components/Layout';
import { AuthProvider, useAuth } from '../../../src/client/contexts/AuthContext';

// Mock the auth context
const mockLogout = jest.fn();
const mockLogin = jest.fn();
const mockRegister = jest.fn();
const mockUpdateUser = jest.fn();

jest.mock('../../../src/client/contexts/AuthContext', () => {
  const actual = jest.requireActual('../../../src/client/contexts/AuthContext');
  return {
    ...actual,
    useAuth: jest.fn(),
  };
});

const mockedUseAuth = useAuth as jest.MockedFunction<typeof useAuth>;

// Helper to render Layout with router
const renderWithRouter = (initialEntries: string[] = ['/']) => {
  return render(
    <MemoryRouter initialEntries={initialEntries}>
      <Layout />
    </MemoryRouter>
  );
};

describe('Layout', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  describe('when user is logged out', () => {
    beforeEach(() => {
      mockedUseAuth.mockReturnValue({
        user: null,
        isLoading: false,
        login: mockLogin,
        register: mockRegister,
        logout: mockLogout,
        updateUser: mockUpdateUser,
      });
    });

    it('renders without crashing', () => {
      renderWithRouter();
      expect(screen.getByText('RingRift')).toBeInTheDocument();
    });

    it('renders login link when not authenticated', () => {
      renderWithRouter();
      expect(screen.getByRole('link', { name: /login/i })).toBeInTheDocument();
    });

    it('does not show logout button when not authenticated', () => {
      renderWithRouter();
      expect(screen.queryByRole('button', { name: /logout/i })).not.toBeInTheDocument();
    });
  });

  describe('when user is logged in', () => {
    beforeEach(() => {
      mockedUseAuth.mockReturnValue({
        user: {
          id: 'user-1',
          email: 'test@example.com',
          username: 'TestUser',
          role: 'player' as const,
          rating: 1500,
          gamesPlayed: 17,
          gamesWon: 10,
          createdAt: new Date(),
          lastActive: new Date(),
          status: 'online' as const,
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
        },
        isLoading: false,
        login: mockLogin,
        register: mockRegister,
        logout: mockLogout,
        updateUser: mockUpdateUser,
      });
    });

    it('displays username when authenticated', () => {
      renderWithRouter();
      expect(screen.getByText('TestUser')).toBeInTheDocument();
    });

    it('displays user rating when available', () => {
      renderWithRouter();
      expect(screen.getByText(/Rating 1500/)).toBeInTheDocument();
    });

    it('renders logout button when authenticated', () => {
      renderWithRouter();
      expect(screen.getByRole('button', { name: /logout/i })).toBeInTheDocument();
    });

    it('calls logout when logout button is clicked', () => {
      renderWithRouter();
      const logoutButton = screen.getByRole('button', { name: /logout/i });
      fireEvent.click(logoutButton);
      expect(mockLogout).toHaveBeenCalledTimes(1);
    });

    it('does not show login link when authenticated', () => {
      renderWithRouter();
      expect(screen.queryByRole('link', { name: /^login$/i })).not.toBeInTheDocument();
    });
  });

  describe('navigation', () => {
    beforeEach(() => {
      mockedUseAuth.mockReturnValue({
        user: null,
        isLoading: false,
        login: mockLogin,
        register: mockRegister,
        logout: mockLogout,
        updateUser: mockUpdateUser,
      });
    });

    it('renders Home navigation link', () => {
      renderWithRouter();
      expect(screen.getByRole('link', { name: /home/i })).toBeInTheDocument();
    });

    it('renders Lobby navigation link', () => {
      renderWithRouter();
      expect(screen.getByRole('link', { name: /lobby/i })).toBeInTheDocument();
    });

    it('renders Leaderboard navigation link', () => {
      renderWithRouter();
      expect(screen.getByRole('link', { name: /leaderboard/i })).toBeInTheDocument();
    });

    it('renders Practice navigation link', () => {
      renderWithRouter();
      expect(screen.getByRole('link', { name: /practice/i })).toBeInTheDocument();
    });

    it('renders RingRift brand link to home', () => {
      renderWithRouter();
      const brandLink = screen.getByRole('link', { name: /ringrift/i });
      expect(brandLink).toBeInTheDocument();
      expect(brandLink).toHaveAttribute('href', '/');
    });
  });

  describe('accessibility', () => {
    beforeEach(() => {
      mockedUseAuth.mockReturnValue({
        user: null,
        isLoading: false,
        login: mockLogin,
        register: mockRegister,
        logout: mockLogout,
        updateUser: mockUpdateUser,
      });
    });

    it('has skip to main content link', () => {
      renderWithRouter();
      const skipLink = screen.getByText(/skip to main content/i);
      expect(skipLink).toBeInTheDocument();
      expect(skipLink).toHaveAttribute('href', '#main-content');
    });

    it('has main landmark with correct id', () => {
      renderWithRouter();
      const main = screen.getByRole('main');
      expect(main).toHaveAttribute('id', 'main-content');
    });

    it('has navigation landmark', () => {
      renderWithRouter();
      expect(screen.getByRole('navigation')).toBeInTheDocument();
    });
  });

  describe('styling', () => {
    beforeEach(() => {
      mockedUseAuth.mockReturnValue({
        user: null,
        isLoading: false,
        login: mockLogin,
        register: mockRegister,
        logout: mockLogout,
        updateUser: mockUpdateUser,
      });
    });

    it('has dark background styling', () => {
      const { container } = renderWithRouter();
      const wrapper = container.firstChild as HTMLElement;
      expect(wrapper).toHaveClass('bg-slate-950');
    });

    it('has minimum screen height', () => {
      const { container } = renderWithRouter();
      const wrapper = container.firstChild as HTMLElement;
      expect(wrapper).toHaveClass('min-h-screen');
    });
  });
});
