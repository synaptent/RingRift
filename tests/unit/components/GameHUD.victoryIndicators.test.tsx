/**
 * Tests for LPS and Victory Progress indicators in GameHUD.
 *
 * Tests compliance with:
 * - RR-CANON-R172: LPS requires 2 consecutive rounds where only 1 player has real actions
 * - RR-CANON-R061: Ring elimination victory threshold (victoryThreshold)
 * - RR-CANON-R062: Territory threshold = floor(totalSpaces/2)+1
 */
import React from 'react';
import { render, screen } from '@testing-library/react';
import '@testing-library/jest-dom';
import { GameHUD } from '../../../src/client/components/GameHUD';
import type { HUDViewModel, PlayerViewModel } from '../../../src/client/adapters/gameViewModels';
import type { TimeControl } from '../../../src/shared/types/game';

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

const baseTimeControl: TimeControl = { type: 'rapid', initialTime: 600, increment: 0 };

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
    decisions: [],
    timers: {
      serverTimeOffsetMs: 0,
      decisionDeadlineMs: null,
      reconciledDecisionTimeRemainingMs: null,
      isServerCapped: false,
    },
    connectionStatus: 'connected',
    isSpectator: false,
    isLocalSandboxOnly: false,
    ...overrides,
  };
}

describe('LpsTrackingIndicator', () => {
  it('does not render when lpsTracking is undefined', () => {
    const viewModel = baseHudViewModel({ lpsTracking: undefined });
    render(<GameHUD viewModel={viewModel} timeControl={baseTimeControl} />);

    expect(screen.queryByTestId('hud-lps-indicator')).not.toBeInTheDocument();
  });

  it('does not render when consecutiveExclusiveRounds is 0', () => {
    const viewModel = baseHudViewModel({
      lpsTracking: {
        roundIndex: 5,
        consecutiveExclusiveRounds: 0,
        consecutiveExclusivePlayer: null,
      },
    });
    render(<GameHUD viewModel={viewModel} timeControl={baseTimeControl} />);

    expect(screen.queryByTestId('hud-lps-indicator')).not.toBeInTheDocument();
  });

  it('renders when consecutiveExclusiveRounds is 1', () => {
    const viewModel = baseHudViewModel({
      lpsTracking: {
        roundIndex: 6,
        consecutiveExclusiveRounds: 1,
        consecutiveExclusivePlayer: 1,
      },
    });
    render(<GameHUD viewModel={viewModel} timeControl={baseTimeControl} />);

    const indicator = screen.getByTestId('hud-lps-indicator');
    expect(indicator).toBeInTheDocument();
    expect(indicator).toHaveTextContent('Alice has exclusive actions');
    expect(indicator).toHaveTextContent('1 round until LPS');
    expect(indicator).toHaveTextContent('Round 1/2');
  });

  it('shows LPS Victory at 2 rounds (per RR-CANON-R172)', () => {
    const viewModel = baseHudViewModel({
      lpsTracking: {
        roundIndex: 7,
        consecutiveExclusiveRounds: 2,
        consecutiveExclusivePlayer: 2,
      },
    });
    render(<GameHUD viewModel={viewModel} timeControl={baseTimeControl} />);

    const indicator = screen.getByTestId('hud-lps-indicator');
    expect(indicator).toHaveTextContent('Bob has exclusive actions');
    expect(indicator).toHaveTextContent('LPS Victory!');
    expect(indicator).toHaveTextContent('Round 2/2');
  });

  it('falls back to Player N when player not found', () => {
    const viewModel = baseHudViewModel({
      lpsTracking: {
        roundIndex: 6,
        consecutiveExclusiveRounds: 1,
        consecutiveExclusivePlayer: 99, // Unknown player
      },
    });
    render(<GameHUD viewModel={viewModel} timeControl={baseTimeControl} />);

    const indicator = screen.getByTestId('hud-lps-indicator');
    expect(indicator).toHaveTextContent('Player 99 has exclusive actions');
  });
});

describe('VictoryProgressIndicator', () => {
  it('does not render when victoryProgress is undefined', () => {
    const viewModel = baseHudViewModel({ victoryProgress: undefined });
    render(<GameHUD viewModel={viewModel} timeControl={baseTimeControl} />);

    expect(screen.queryByTestId('hud-victory-progress')).not.toBeInTheDocument();
  });

  it('does not render when both leaders are below 20% threshold', () => {
    const viewModel = baseHudViewModel({
      victoryProgress: {
        ringElimination: {
          threshold: 18, // Per RR-CANON-R061 for 2P square8
          leader: { playerNumber: 1, eliminated: 3, percentage: 15 },
        },
        territory: {
          threshold: 33, // Per RR-CANON-R062 for square8
          leader: { playerNumber: 2, spaces: 5, percentage: 15 },
        },
      },
    });
    render(<GameHUD viewModel={viewModel} timeControl={baseTimeControl} />);

    expect(screen.queryByTestId('hud-victory-progress')).not.toBeInTheDocument();
  });

  it('renders ring progress when leader has >= 20%', () => {
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
    render(<GameHUD viewModel={viewModel} timeControl={baseTimeControl} />);

    const indicator = screen.getByTestId('hud-victory-progress');
    expect(indicator).toBeInTheDocument();
    expect(indicator).toHaveTextContent('Rings:');
    expect(indicator).toHaveTextContent('Alice: 5/18');
    // Territory should not show (below threshold)
    expect(indicator).not.toHaveTextContent('Territory:');
  });

  it('renders territory progress when leader has >= 20%', () => {
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
    render(<GameHUD viewModel={viewModel} timeControl={baseTimeControl} />);

    const indicator = screen.getByTestId('hud-victory-progress');
    expect(indicator).toBeInTheDocument();
    expect(indicator).toHaveTextContent('Territory:');
    expect(indicator).toHaveTextContent('Bob: 10/33');
    // Rings should not show (below threshold)
    expect(indicator).not.toHaveTextContent('Rings:');
  });

  it('renders both ring and territory when both are >= 20%', () => {
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
    render(<GameHUD viewModel={viewModel} timeControl={baseTimeControl} />);

    const indicator = screen.getByTestId('hud-victory-progress');
    expect(indicator).toHaveTextContent('Rings:');
    expect(indicator).toHaveTextContent('Alice: 8/18');
    expect(indicator).toHaveTextContent('Territory:');
    expect(indicator).toHaveTextContent('Alice: 12/33');
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
    render(<GameHUD viewModel={viewModel} timeControl={baseTimeControl} />);

    expect(screen.queryByTestId('hud-victory-progress')).not.toBeInTheDocument();
  });

  it('uses player number fallback when player not found', () => {
    const viewModel = baseHudViewModel({
      victoryProgress: {
        ringElimination: {
          threshold: 18,
          leader: { playerNumber: 99, eliminated: 5, percentage: 26 },
        },
        territory: {
          threshold: 33,
          leader: null,
        },
      },
    });
    render(<GameHUD viewModel={viewModel} timeControl={baseTimeControl} />);

    const indicator = screen.getByTestId('hud-victory-progress');
    expect(indicator).toHaveTextContent('P99: 5/18');
  });
});
