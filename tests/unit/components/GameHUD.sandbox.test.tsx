import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import '@testing-library/jest-dom';
import { GameHUD } from '../../../src/client/components/GameHUD';
import type { Player, TimeControl } from '../../../src/shared/types/game';
import type { HUDViewModel } from '../../../src/client/adapters/gameViewModels';

function createPlayers(): Player[] {
  return [
    {
      id: 'p1',
      username: 'Alice',
      playerNumber: 1,
      type: 'human',
      isReady: true,
      timeRemaining: 120_000,
      ringsInHand: 5,
      eliminatedRings: 1,
      territorySpaces: 2,
    },
    {
      id: 'p2',
      username: 'Bob',
      playerNumber: 2,
      type: 'human',
      isReady: true,
      timeRemaining: 90_000,
      ringsInHand: 4,
      eliminatedRings: 2,
      territorySpaces: 0,
    },
  ];
}

function baseHUDViewModel(overrides: Partial<HUDViewModel> = {}): HUDViewModel {
  const players = createPlayers();
  return {
    phase: {
      phaseKey: 'movement',
      label: 'Movement Phase',
      description: 'Move a stack',
      icon: '⚡',
      colorClass: 'bg-green-500',
      actionHint: 'Select a stack',
      spectatorHint: 'Watching',
    },
    players: [
      {
        id: 'p1',
        username: 'Alice',
        playerNumber: 1,
        colorClass: 'bg-blue-500',
        isCurrentPlayer: true,
        isUserPlayer: true,
        timeRemaining: 120_000,
        ringStats: { inHand: 5, onBoard: 3, eliminated: 1, total: 9 },
        territorySpaces: 2,
        aiInfo: {
          isAI: false,
          difficulty: 0,
          difficultyLabel: '',
          difficultyColor: '',
          difficultyBgColor: '',
          aiTypeLabel: '',
        },
      },
      {
        id: 'p2',
        username: 'Bob',
        playerNumber: 2,
        colorClass: 'bg-red-500',
        isCurrentPlayer: false,
        isUserPlayer: false,
        timeRemaining: 90_000,
        ringStats: { inHand: 4, onBoard: 4, eliminated: 2, total: 10 },
        territorySpaces: 0,
        aiInfo: {
          isAI: false,
          difficulty: 0,
          difficultyLabel: '',
          difficultyColor: '',
          difficultyBgColor: '',
          aiTypeLabel: '',
        },
      },
    ],
    turnNumber: 3,
    moveNumber: 7,
    connectionStatus: 'connected',
    isConnectionStale: false,
    isSpectator: false,
    spectatorCount: 0,
    ...overrides,
  };
}

describe('GameHUD – sandbox/connection surfaces', () => {
  it('shows sandbox banner and clock fallback when no time control', () => {
    const viewModel = baseHUDViewModel({ spectatorCount: 0 });

    render(<GameHUD viewModel={viewModel} isLocalSandboxOnly />);

    expect(screen.getByTestId('sandbox-local-only-banner')).toBeInTheDocument();
    expect(screen.getByText(/Clock: No clock \(local sandbox\)/i)).toBeInTheDocument();
  });

  it('renders time control summary when clocks are enabled', () => {
    const timeControl: TimeControl = { type: 'rapid', initialTime: 600, increment: 5 };
    const viewModel = baseHUDViewModel();

    render(<GameHUD viewModel={viewModel} timeControl={timeControl} />);

    expect(screen.getByTestId('hud-time-control-summary')).toHaveTextContent('Rapid • 10+5');
  });

  it('surfaces connection state, spectator count, and board controls entrypoint', () => {
    const onShowBoardControls = jest.fn();
    const viewModel = baseHUDViewModel({
      connectionStatus: 'connecting',
      isConnectionStale: true,
      spectatorCount: 3,
    });

    render(<GameHUD viewModel={viewModel} onShowBoardControls={onShowBoardControls} />);

    expect(screen.getByText(/Connection: Connecting…/i)).toBeInTheDocument();
    expect(screen.getByText(/\(no recent updates from server\)/i)).toBeInTheDocument();
    expect(screen.getByText('3 watching')).toBeInTheDocument();

    fireEvent.click(screen.getByTestId('board-controls-button'));
    expect(onShowBoardControls).toHaveBeenCalledTimes(1);
  });
});
