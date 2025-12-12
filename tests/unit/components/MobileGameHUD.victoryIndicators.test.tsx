/**
 * Tests for LPS and Victory Progress indicators in MobileGameHUD.
 *
 * Tests compliance with:
 * - RR-CANON-R172: LPS requires 2 consecutive rounds where only 1 player has real actions
 * - RR-CANON-R061: Ring elimination victory threshold (victoryThreshold)
 * - RR-CANON-R062: Territory threshold = floor(totalSpaces/2)+1
 */
import React from 'react';
import { render, screen } from '@testing-library/react';
import '@testing-library/jest-dom';
import { MobileGameHUD } from '../../../src/client/components/MobileGameHUD';
import type { HUDViewModel, PlayerViewModel } from '../../../src/client/adapters/gameViewModels';

function basePlayers(): PlayerViewModel[] {
  return [
    {
      id: 'p1',
      username: 'Alice',
      playerNumber: 1,
      colorClass: 'bg-blue-500',
      isCurrentPlayer: true,
      isUserPlayer: true,
      timeRemaining: 60_000,
      ringStats: { inHand: 5, onBoard: 3, eliminated: 10, total: 18 },
      territorySpaces: 15,
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
      timeRemaining: 60_000,
      ringStats: { inHand: 4, onBoard: 4, eliminated: 5, total: 13 },
      territorySpaces: 8,
      aiInfo: {
        isAI: false,
        difficulty: 0,
        difficultyLabel: '',
        difficultyColor: '',
        difficultyBgColor: '',
        aiTypeLabel: '',
      },
    },
  ];
}

function baseHudViewModel(overrides?: Partial<HUDViewModel>): HUDViewModel {
  return {
    phase: {
      phaseKey: 'movement',
      label: 'Movement Phase',
      description: 'Move a stack',
      icon: '⚡',
      colorClass: 'bg-green-500',
      actionHint: 'Select your stack',
      spectatorHint: 'Player is moving',
    },
    players: basePlayers(),
    turnNumber: 1,
    moveNumber: 0,
    connectionStatus: 'connected',
    isConnectionStale: false,
    isSpectator: false,
    spectatorCount: 0,
    decisions: [],
    ...overrides,
  };
}

describe('MobileLpsIndicator', () => {
  it('does not render when lpsTracking is undefined', () => {
    const viewModel = baseHudViewModel({ lpsTracking: undefined });
    render(<MobileGameHUD viewModel={viewModel} />);

    expect(screen.queryByTestId('mobile-lps-indicator')).not.toBeInTheDocument();
  });

  it('does not render when consecutiveExclusiveRounds is 0', () => {
    const viewModel = baseHudViewModel({
      lpsTracking: {
        roundIndex: 5,
        consecutiveExclusiveRounds: 0,
        consecutiveExclusivePlayer: null,
      },
    });
    render(<MobileGameHUD viewModel={viewModel} />);

    expect(screen.queryByTestId('mobile-lps-indicator')).not.toBeInTheDocument();
  });

  it('renders when consecutiveExclusiveRounds is 1', () => {
    const viewModel = baseHudViewModel({
      lpsTracking: {
        roundIndex: 6,
        consecutiveExclusiveRounds: 1,
        consecutiveExclusivePlayer: 1,
      },
    });
    render(<MobileGameHUD viewModel={viewModel} />);

    const indicator = screen.getByTestId('mobile-lps-indicator');
    expect(indicator).toBeInTheDocument();
    // Mobile shows username + "exclusive" and progress dots with aria-label
    expect(indicator).toHaveTextContent('Alice exclusive');
    expect(screen.getByLabelText('1 of 2 rounds')).toBeInTheDocument();
  });

  it('shows LPS at 2 rounds (per RR-CANON-R172)', () => {
    const viewModel = baseHudViewModel({
      lpsTracking: {
        roundIndex: 7,
        consecutiveExclusiveRounds: 2,
        consecutiveExclusivePlayer: 2,
      },
    });
    render(<MobileGameHUD viewModel={viewModel} />);

    const indicator = screen.getByTestId('mobile-lps-indicator');
    expect(indicator).toHaveTextContent('Bob exclusive');
    expect(screen.getByLabelText('2 of 2 rounds')).toBeInTheDocument();
  });
});

describe('MobileVictoryProgress', () => {
  it('does not render when victoryProgress is undefined', () => {
    const viewModel = baseHudViewModel({ victoryProgress: undefined });
    render(<MobileGameHUD viewModel={viewModel} />);

    expect(screen.queryByTestId('mobile-victory-progress')).not.toBeInTheDocument();
  });

  it('does not render when both leaders are below 25% threshold (mobile uses higher threshold)', () => {
    const viewModel = baseHudViewModel({
      victoryProgress: {
        ringElimination: {
          threshold: 18,
          leader: { playerNumber: 1, eliminated: 4, percentage: 21 }, // 21% < 25%
        },
        territory: {
          threshold: 33,
          leader: { playerNumber: 2, spaces: 6, percentage: 18 },
        },
      },
    });
    render(<MobileGameHUD viewModel={viewModel} />);

    expect(screen.queryByTestId('mobile-victory-progress')).not.toBeInTheDocument();
  });

  it('renders ring progress when leader has >= 25%', () => {
    const viewModel = baseHudViewModel({
      victoryProgress: {
        ringElimination: {
          threshold: 18,
          leader: { playerNumber: 1, eliminated: 5, percentage: 26 },
        },
        territory: {
          threshold: 33,
          leader: { playerNumber: 2, spaces: 3, percentage: 9 },
        },
      },
    });
    render(<MobileGameHUD viewModel={viewModel} />);

    const indicator = screen.getByTestId('mobile-victory-progress');
    expect(indicator).toBeInTheDocument();
    expect(indicator).toHaveTextContent('5/18');
    // Should show only rings, not territory (below threshold)
  });

  it('renders territory progress when leader has >= 25%', () => {
    const viewModel = baseHudViewModel({
      victoryProgress: {
        ringElimination: {
          threshold: 18,
          leader: { playerNumber: 1, eliminated: 2, percentage: 10 },
        },
        territory: {
          threshold: 33,
          leader: { playerNumber: 2, spaces: 10, percentage: 30 },
        },
      },
    });
    render(<MobileGameHUD viewModel={viewModel} />);

    const indicator = screen.getByTestId('mobile-victory-progress');
    expect(indicator).toBeInTheDocument();
    expect(indicator).toHaveTextContent('10/33');
  });

  it('renders both when both are >= 25%', () => {
    const viewModel = baseHudViewModel({
      victoryProgress: {
        ringElimination: {
          threshold: 18,
          leader: { playerNumber: 1, eliminated: 8, percentage: 42 },
        },
        territory: {
          threshold: 33,
          leader: { playerNumber: 1, spaces: 12, percentage: 36 },
        },
      },
    });
    render(<MobileGameHUD viewModel={viewModel} />);

    const indicator = screen.getByTestId('mobile-victory-progress');
    expect(indicator).toHaveTextContent('8/18');
    expect(indicator).toHaveTextContent('12/33');
  });

  it('handles null leader gracefully', () => {
    const viewModel = baseHudViewModel({
      victoryProgress: {
        ringElimination: {
          threshold: 18,
          leader: null,
        },
        territory: {
          threshold: 33,
          leader: null,
        },
      },
    });
    render(<MobileGameHUD viewModel={viewModel} />);

    expect(screen.queryByTestId('mobile-victory-progress')).not.toBeInTheDocument();
  });
});
