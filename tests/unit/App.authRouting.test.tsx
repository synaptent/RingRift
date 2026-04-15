import React from 'react';
import { MemoryRouter } from 'react-router-dom';
import { render, screen } from '@testing-library/react';
import '@testing-library/jest-dom';
import App from '../../src/client/App';
import { useAuth } from '../../src/client/contexts/AuthContext';

jest.mock('../../src/client/contexts/AuthContext', () => ({
  __esModule: true,
  useAuth: jest.fn(),
}));

// Lightweight mock components for routed pages to make assertions easier.
function mockLoginPage() {
  return <div data-testid="login-page">login</div>;
}

function mockHomePage() {
  return <div data-testid="home-page">home</div>;
}

function mockLandingPage() {
  return <div data-testid="landing-page">landing</div>;
}

function mockLobbyPage() {
  return <div data-testid="lobby-page">lobby</div>;
}

function mockSandboxGameHost() {
  return <div data-testid="sandbox-host">sandbox</div>;
}

function mockBackendGameHost(props: { gameId: string }) {
  return <div data-testid="backend-host">backend-{props.gameId}</div>;
}

jest.mock('../../src/client/pages/LoginPage', () => ({
  __esModule: true,
  default: mockLoginPage,
}));

jest.mock('../../src/client/pages/HomePage', () => ({
  __esModule: true,
  default: mockHomePage,
}));

jest.mock('../../src/client/pages/LandingPage', () => ({
  __esModule: true,
  default: mockLandingPage,
}));

jest.mock('../../src/client/pages/LobbyPage', () => ({
  __esModule: true,
  default: mockLobbyPage,
}));

jest.mock('../../src/client/pages/SandboxGameHost', () => ({
  __esModule: true,
  SandboxGameHost: mockSandboxGameHost,
}));

jest.mock('../../src/client/pages/BackendGameHost', () => ({
  __esModule: true,
  BackendGameHost: mockBackendGameHost,
}));

const mockedUseAuth = useAuth as jest.MockedFunction<typeof useAuth>;

describe('App auth routing', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('redirects unauthenticated users to login for protected lobby route', () => {
    mockedUseAuth.mockReturnValue({ user: null, isLoading: false } as any);

    render(
      <MemoryRouter initialEntries={['/lobby']}>
        <App />
      </MemoryRouter>
    );

    expect(screen.getByTestId('login-page')).toBeInTheDocument();
    expect(screen.queryByTestId('layout')).not.toBeInTheDocument();
  });

  it('redirects unauthenticated users to login for protected game route', () => {
    mockedUseAuth.mockReturnValue({ user: null, isLoading: false } as any);

    render(
      <MemoryRouter initialEntries={['/game/abc123']}>
        <App />
      </MemoryRouter>
    );

    expect(screen.getByTestId('login-page')).toBeInTheDocument();
    expect(screen.queryByTestId('layout')).not.toBeInTheDocument();
  });

  it('allows unauthenticated users to access /sandbox without redirect', async () => {
    mockedUseAuth.mockReturnValue({ user: null, isLoading: false } as any);

    render(
      <MemoryRouter initialEntries={['/sandbox']}>
        <App />
      </MemoryRouter>
    );

    expect(await screen.findByTestId('sandbox-host')).toBeInTheDocument();
    expect(screen.queryByTestId('login-page')).not.toBeInTheDocument();
  });

  it('shows the landing page for unauthenticated visitors at root', () => {
    mockedUseAuth.mockReturnValue({ user: null, isLoading: false } as any);

    render(
      <MemoryRouter initialEntries={['/']}>
        <App />
      </MemoryRouter>
    );

    expect(screen.getByTestId('landing-page')).toBeInTheDocument();
    expect(screen.queryByTestId('login-page')).not.toBeInTheDocument();
  });

  it('shows the authenticated home page at root for signed-in users', async () => {
    mockedUseAuth.mockReturnValue({ user: { id: 'u1' }, isLoading: false } as any);

    render(
      <MemoryRouter initialEntries={['/']}>
        <App />
      </MemoryRouter>
    );

    expect(await screen.findByTestId('home-page')).toBeInTheDocument();
    expect(screen.queryByTestId('landing-page')).not.toBeInTheDocument();
  });

  it('allows authenticated users to access protected routes', async () => {
    mockedUseAuth.mockReturnValue({ user: { id: 'u1' }, isLoading: false } as any);

    render(
      <MemoryRouter initialEntries={['/lobby']}>
        <App />
      </MemoryRouter>
    );

    expect(await screen.findByTestId('lobby-page')).toBeInTheDocument();
    expect(screen.queryByTestId('login-page')).not.toBeInTheDocument();
  });
});
