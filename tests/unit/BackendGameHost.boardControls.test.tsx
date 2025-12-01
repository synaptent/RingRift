import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import '@testing-library/jest-dom';
import { MemoryRouter } from 'react-router-dom';
import { BackendGameHost } from '../../src/client/pages/BackendGameHost';
import { useGame } from '../../src/client/contexts/GameContext';
import { useAuth } from '../../src/client/contexts/AuthContext';
import type { GameState, BoardState, Player } from '../../src/shared/types/game';

jest.mock('../../src/client/contexts/GameContext', () => ({
  useGame: jest.fn(),
}));

jest.mock('../../src/client/contexts/AuthContext', () => ({
  useAuth: jest.fn(),
}));

// Stub out heavy presentational components that are not relevant to overlay wiring.
// Important: avoid JSX or imports inside the mock factory to satisfy Jest's
// "module factory must not reference out-of-scope variables" rule.
jest.mock('../../src/client/components/BoardView', () => ({
  __esModule: true,
  BoardView: () => 'BoardView',
}));

jest.mock('../../src/client/components/GameEventLog', () => ({
  __esModule: true,
  GameEventLog: () => 'GameEventLog',
}));

jest.mock('../../src/client/components/ChoiceDialog', () => ({
  __esModule: true,
  ChoiceDialog: () => null,
}));

jest.mock('../../src/client/components/VictoryModal', () => ({
  __esModule: true,
  VictoryModal: () => null,
}));

// Mock GameHUD so we can drive the onShowBoardControls callback directly via
// a small test button instead of relying on internal layout details.
jest.mock('../../src/client/components/GameHUD', () => {
  // Use require inside the factory so we do not reference the top-level React
  // import, which Jest forbids in mock factories.
  // eslint-disable-next-line @typescript-eslint/no-var-requires
  const React = require('react');

  const GameHUDMock = ({ onShowBoardControls }: { onShowBoardControls?: () => void }) =>
    React.createElement(
      'div',
      { 'data-testid': 'game-hud-mock' },
      React.createElement(
        'button',
        {
          type: 'button',
          'data-testid': 'board-controls-button',
          onClick: onShowBoardControls,
        },
        'Help',
      ),
    );

  return {
    __esModule: true,
    GameHUD: GameHUDMock,
  };
});

// Keep the real BoardControlsOverlay API but simplify its rendering so we can
// assert mode wiring without depending on copy/layout.
jest.mock('../../src/client/components/BoardControlsOverlay', () => ({
  __esModule: true,
  BoardControlsOverlay: ({ mode }: { mode: 'backend' | 'sandbox' | 'spectator' }) =>
    `BoardControlsOverlay-${mode}`,
}));

const mockedUseGame = useGame as jest.MockedFunction<typeof useGame>;
const mockedUseAuth = useAuth as jest.MockedFunction<typeof useAuth>;

function createTestGameState(): GameState {
  const board: BoardState = {
    stacks: new Map(),
    markers: new Map(),
    collapsedSpaces: new Map(),
    territories: new Map(),
    formedLines: [],
    eliminatedRings: {},
    size: 8,
    type: 'square8',
  };

  const players: Player[] = [
    {
      id: 'user-1',
      username: 'Alice',
      playerNumber: 1,
      type: 'human',
      isReady: true,
      timeRemaining: 600_000,
      ringsInHand: 18,
      eliminatedRings: 0,
      territorySpaces: 0,
    },
    {
      id: 'user-2',
      username: 'Bob',
      playerNumber: 2,
      type: 'human',
      isReady: true,
      timeRemaining: 600_000,
      ringsInHand: 18,
      eliminatedRings: 0,
      territorySpaces: 0,
    },
  ];

  return {
    id: 'game-123',
    boardType: 'square8',
    board,
    players,
    currentPhase: 'ring_placement',
    currentPlayer: 1,
    moveHistory: [],
    history: [],
    timeControl: { type: 'rapid', initialTime: 600, increment: 0 },
    spectators: [],
    gameStatus: 'active',
    createdAt: new Date(),
    lastMoveAt: new Date(),
    isRated: false,
    maxPlayers: 2,
    totalRingsInPlay: 0,
    totalRingsEliminated: 0,
    victoryThreshold: 10,
    territoryVictoryThreshold: 32,
  };
}

describe('BackendGameHost board controls overlay wiring', () => {
  beforeEach(() => {
    jest.clearAllMocks();

    mockedUseAuth.mockReturnValue({
      user: { id: 'user-1' } as any,
      isLoading: false,
      login: jest.fn(),
      register: jest.fn(),
      logout: jest.fn(),
      updateUser: jest.fn(),
    } as any);

    const gameState = createTestGameState();

    mockedUseGame.mockReturnValue({
      gameId: gameState.id,
      gameState,
      validMoves: [],
      isConnecting: false,
      error: null,
      victoryState: null,
      connectToGame: jest.fn(),
      disconnect: jest.fn(),
      pendingChoice: null,
      choiceDeadline: null,
      respondToChoice: jest.fn(),
      submitMove: jest.fn(),
      sendChatMessage: jest.fn(),
      chatMessages: [],
      connectionStatus: 'connected',
      lastHeartbeatAt: Date.now(),
    } as any);
  });

  function renderHost() {
    render(
      <MemoryRouter initialEntries={['/game/game-123']}>
        <BackendGameHost gameId="game-123" />
      </MemoryRouter>,
    );
  }

  it('opens the board controls overlay in backend mode when the help button is clicked', () => {
    renderHost();

    expect(
      screen.queryByTestId('board-controls-overlay'),
    ).not.toBeInTheDocument();

    fireEvent.click(screen.getByTestId('board-controls-button'));

    // With the simplified mock, we only assert that some overlay marker is present.
    // The specific text content ("BoardControlsOverlay-backend") comes from the mock.
    expect(screen.getByText('BoardControlsOverlay-backend')).toBeInTheDocument();
  });

  it('toggles the overlay with the "?" keyboard shortcut', () => {
    renderHost();

    expect(
      screen.queryByText('BoardControlsOverlay-backend'),
    ).not.toBeInTheDocument();

    fireEvent.keyDown(window, { key: '?', shiftKey: true });

    expect(
      screen.getByText('BoardControlsOverlay-backend'),
    ).toBeInTheDocument();

    fireEvent.keyDown(window, { key: '?', shiftKey: true });

    expect(
      screen.queryByText('BoardControlsOverlay-backend'),
    ).not.toBeInTheDocument();
  });

  it('closes the overlay with Escape when it is open', () => {
    renderHost();

    fireEvent.keyDown(window, { key: '?', shiftKey: true });
    expect(
      screen.getByText('BoardControlsOverlay-backend'),
    ).toBeInTheDocument();

    fireEvent.keyDown(window, { key: 'Escape' });

    expect(
      screen.queryByText('BoardControlsOverlay-backend'),
    ).not.toBeInTheDocument();
  });
});