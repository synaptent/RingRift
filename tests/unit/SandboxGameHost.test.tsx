import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import '@testing-library/jest-dom';
import { MemoryRouter } from 'react-router-dom';
import { SandboxGameHost } from '../../src/client/pages/SandboxGameHost';
import { useSandbox } from '../../src/client/contexts/SandboxContext';
import { useAuth } from '../../src/client/contexts/AuthContext';
import { useSandboxInteractions } from '../../src/client/hooks/useSandboxInteractions';
import { gameApi } from '../../src/client/services/api';

jest.mock('../../src/client/contexts/SandboxContext', () => ({
  useSandbox: jest.fn(),
}));

jest.mock('../../src/client/contexts/AuthContext', () => ({
  useAuth: jest.fn(),
}));

jest.mock('../../src/client/hooks/useSandboxInteractions', () => ({
  useSandboxInteractions: jest.fn(),
}));

jest.mock('../../src/client/services/api', () => ({
  gameApi: {
    createGame: jest.fn(),
  },
}));

const mockedUseSandbox = useSandbox as jest.MockedFunction<typeof useSandbox>;
const mockedUseAuth = useAuth as jest.MockedFunction<typeof useAuth>;
const mockedUseSandboxInteractions = useSandboxInteractions as jest.MockedFunction<
  typeof useSandboxInteractions
>;

let sandboxValue: any;

describe('SandboxGameHost', () => {
  beforeEach(() => {
    jest.clearAllMocks();

    mockedUseAuth.mockReturnValue({
      user: { id: 'user-1' },
      isLoading: false,
      login: jest.fn(),
      register: jest.fn(),
      logout: jest.fn(),
      updateUser: jest.fn(),
    } as any);

    sandboxValue = {
      config: {
        numPlayers: 2,
        boardType: 'square8',
        playerTypes: ['human', 'ai', 'ai', 'ai'],
      },
      setConfig: jest.fn(),
      isConfigured: false,
      setIsConfigured: jest.fn(),
      backendSandboxError: null,
      setBackendSandboxError: jest.fn(),
      sandboxEngine: null,
      sandboxPendingChoice: null,
      setSandboxPendingChoice: jest.fn(),
      sandboxCaptureChoice: null,
      setSandboxCaptureChoice: jest.fn(),
      sandboxCaptureTargets: [],
      setSandboxCaptureTargets: jest.fn(),
      sandboxLastProgressAt: null,
      setSandboxLastProgressAt: jest.fn(),
      sandboxStallWarning: null,
      setSandboxStallWarning: jest.fn(),
      sandboxStateVersion: 0,
      setSandboxStateVersion: jest.fn(),
      sandboxDiagnosticsEnabled: false,
      initLocalSandboxEngine: jest.fn(),
      getSandboxGameState: jest.fn().mockReturnValue(null),
      resetSandboxEngine: jest.fn(),
    } as any;

    mockedUseSandbox.mockReturnValue(sandboxValue);

    mockedUseSandboxInteractions.mockReturnValue({
      handleCellClick: jest.fn(),
      handleCellDoubleClick: jest.fn(),
      handleCellContextMenu: jest.fn(),
      maybeRunSandboxAiIfNeeded: jest.fn(),
    } as any);
  });

  it('renders sandbox setup view when not configured', () => {
    render(
      <MemoryRouter initialEntries={['/sandbox']}>
        <SandboxGameHost />
      </MemoryRouter>
    );

    expect(screen.getByText(/Start a RingRift Game \(Local Sandbox\)/i)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /Launch Game/i })).toBeInTheDocument();
  });

  it('attempts to create backend game when Launch Game is clicked', async () => {
    const createGameMock = gameApi.createGame as jest.Mock;
    createGameMock.mockResolvedValue({ id: 'game-123' } as any);

    render(
      <MemoryRouter initialEntries={['/sandbox']}>
        <SandboxGameHost />
      </MemoryRouter>
    );

    fireEvent.click(screen.getByRole('button', { name: /Launch Game/i }));

    await waitFor(() => {
      expect(createGameMock).toHaveBeenCalledTimes(1);
    });

    expect(createGameMock).toHaveBeenCalledWith(
      expect.objectContaining({
        boardType: 'square8',
        maxPlayers: 2,
        isRated: false,
        isPrivate: true,
      })
    );
  });

  it('falls back to local sandbox engine when backend game creation fails', async () => {
    const createGameMock = gameApi.createGame as jest.Mock;
    createGameMock.mockRejectedValue(new Error('backend failure'));

    const initLocalSandboxEngine = sandboxValue.initLocalSandboxEngine as jest.Mock;
    const setBackendSandboxError = sandboxValue.setBackendSandboxError as jest.Mock;

    render(
      <MemoryRouter initialEntries={['/sandbox']}>
        <SandboxGameHost />
      </MemoryRouter>
    );

    // Click Launch Game to trigger backend attempt + fallback.
    fireEvent.click(screen.getByRole('button', { name: /Launch Game/i }));

    await waitFor(() => {
      expect(createGameMock).toHaveBeenCalledTimes(1);
    });

    // Backend failure should trigger a local sandbox engine initialisation.
    expect(initLocalSandboxEngine).toHaveBeenCalledTimes(1);
    expect(initLocalSandboxEngine).toHaveBeenCalledWith(
      expect.objectContaining({
        boardType: 'square8',
        numPlayers: 2,
      })
    );

    // And a user-visible backendSandboxError should be recorded.
    expect(setBackendSandboxError).toHaveBeenCalledWith(
      'Backend sandbox game could not be created; falling back to local-only board only.'
    );
  });

  it('skips backend game creation and starts a local sandbox when unauthenticated', async () => {
    const createGameMock = gameApi.createGame as jest.Mock;
    createGameMock.mockResolvedValue({ id: 'game-should-not-be-used' } as any);

    const initLocalSandboxEngine = sandboxValue.initLocalSandboxEngine as jest.Mock;

    mockedUseAuth.mockReturnValue({
      user: null,
      isLoading: false,
      login: jest.fn(),
      register: jest.fn(),
      logout: jest.fn(),
      updateUser: jest.fn(),
    } as any);

    render(
      <MemoryRouter initialEntries={['/sandbox']}>
        <SandboxGameHost />
      </MemoryRouter>
    );

    fireEvent.click(screen.getByRole('button', { name: /Launch Game/i }));

    await waitFor(() => {
      expect(initLocalSandboxEngine).toHaveBeenCalledTimes(1);
    });

    expect(createGameMock).not.toHaveBeenCalled();
  });
});
