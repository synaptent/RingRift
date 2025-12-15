import React from 'react';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import '@testing-library/jest-dom';
import { MobileGameHUD } from '../../../src/client/components/MobileGameHUD';
import type { HUDViewModel } from '../../../src/client/adapters/gameViewModels';

function createBaseViewModel(overrides: Partial<HUDViewModel> = {}): HUDViewModel {
  return {
    phase: {
      phaseKey: 'movement' as any,
      label: 'Movement',
      description: 'Move a stack',
      icon: '⚡',
      colorClass: 'bg-green-500',
      actionHint: 'Select a stack then a destination',
      spectatorHint: 'Player is choosing a move',
    },
    players: [],
    turnNumber: 1,
    moveNumber: 0,
    connectionStatus: 'connected' as any,
    isConnectionStale: false,
    isSpectator: false,
    spectatorCount: 0,
    subPhaseDetail: undefined,
    decisionPhase: undefined,
    weirdState: undefined,
    ...overrides,
  };
}

describe('MobileGameHUD', () => {
  it('toggles player expansion to show detailed ring stats', async () => {
    const user = userEvent.setup();
    const vm = createBaseViewModel({
      players: [
        {
          id: 'p1',
          username: 'Alice',
          playerNumber: 1,
          colorClass: 'bg-blue-500',
          isCurrentPlayer: true,
          isUserPlayer: true,
          timeRemaining: 12_000,
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
      ] as any,
    });

    render(<MobileGameHUD viewModel={vm} />);

    // Collapsed by default: expanded stats labels are hidden.
    expect(screen.queryByText('In Hand')).not.toBeInTheDocument();

    // Expand the player row to reveal detailed stats.
    await user.click(screen.getByRole('button', { name: /You|Alice/ }));
    expect(screen.getByText('In Hand')).toBeInTheDocument();
    expect(screen.getByText('On Board')).toBeInTheDocument();
    expect(screen.getByText('Eliminated')).toBeInTheDocument();
  });

  it('renders spectator badge with viewer count when spectating', () => {
    const vm = createBaseViewModel({
      isSpectator: true,
      spectatorCount: 3,
    });

    render(<MobileGameHUD viewModel={vm} />);

    const badge = screen.getByTestId('mobile-spectator-badge');
    expect(badge).toBeInTheDocument();
    expect(badge).toHaveTextContent('Spectating');
    expect(badge).toHaveTextContent('3 viewer');
  });

  it('renders a phase help button during line_processing', () => {
    const vm = createBaseViewModel({
      phase: {
        phaseKey: 'line_processing' as any,
        label: 'Line Formation',
        description: 'Resolve completed lines and rewards.',
        icon: '═',
        colorClass: 'bg-amber-500',
        actionHint: 'Choose how to resolve lines',
        spectatorHint: 'Player is resolving lines',
      },
    });

    render(<MobileGameHUD viewModel={vm} />);

    const helpButton = screen.getByTestId('mobile-phase-help-line_processing');
    expect(helpButton).toBeInTheDocument();
    expect(helpButton).toHaveTextContent('Phase rules');
  });

  it('renders a territory help button during territory_processing', () => {
    const vm = createBaseViewModel({
      phase: {
        phaseKey: 'territory_processing' as any,
        label: 'Territory Processing',
        description: 'Resolve disconnected regions and territory.',
        icon: '▣',
        colorClass: 'bg-emerald-500',
        actionHint: 'Choose a region to process',
        spectatorHint: 'Player is resolving territory',
      },
      decisionPhase: {
        isActive: true,
        actingPlayerNumber: 1,
        actingPlayerName: 'Alice',
        isLocalActor: true,
        label: 'Your decision: Choose region order',
        description: 'Choose which disconnected region to process first.',
        shortLabel: 'Territory region order',
        timeRemainingMs: null,
        showCountdown: false,
        warningThresholdMs: undefined,
        isServerCapped: undefined,
        spectatorLabel: 'Waiting for Alice to choose a region to process first',
      } as any,
    });

    render(<MobileGameHUD viewModel={vm} />);

    const helpButton = screen.getByTestId('mobile-territory-help');
    expect(helpButton).toBeInTheDocument();
    expect(helpButton).toHaveTextContent('Territory rules');
  });

  it('renders decision status chip and skip hint when decisionPhase provides them', () => {
    const vm = createBaseViewModel({
      phase: {
        phaseKey: 'line_processing' as any,
        label: 'Line Formation',
        description: 'Resolve completed lines and rewards.',
        icon: '═',
        colorClass: 'bg-amber-500',
        actionHint: 'Choose how to resolve lines',
        spectatorHint: 'Player is resolving lines',
      },
      decisionPhase: {
        isActive: true,
        actingPlayerNumber: 1,
        actingPlayerName: 'Alice',
        isLocalActor: true,
        label: 'Your decision: Ring elimination',
        description: 'Choose which stack to eliminate from.',
        shortLabel: 'Ring elimination',
        timeRemainingMs: null,
        showCountdown: false,
        warningThresholdMs: undefined,
        isServerCapped: undefined,
        spectatorLabel: 'Waiting for Alice to choose a stack for ring elimination',
        statusChip: {
          text: 'Select stack cap to eliminate',
          tone: 'attention',
        },
        canSkip: true,
      } as any,
    });

    render(<MobileGameHUD viewModel={vm} />);

    const chip = screen.getByTestId('mobile-decision-status-chip');
    expect(chip).toBeInTheDocument();
    expect(chip).toHaveTextContent('Select stack cap to eliminate');

    const skipHint = screen.getByTestId('mobile-decision-skip-hint');
    expect(skipHint).toBeInTheDocument();
    expect(skipHint).toHaveTextContent(/Skip available/i);
  });

  it('surfaces connection status and board controls button in footer', () => {
    const onShowBoardControls = jest.fn();
    const vm = createBaseViewModel({
      connectionStatus: 'reconnecting' as any,
      isConnectionStale: true,
    });

    render(<MobileGameHUD viewModel={vm} onShowBoardControls={onShowBoardControls} />);

    expect(screen.getByText(/reconnecting/i)).toBeInTheDocument();

    const controlsBtn = screen.getByTestId('mobile-controls-button');
    controlsBtn && controlsBtn.click();
    expect(onShowBoardControls).toHaveBeenCalledTimes(1);
  });

  it('invokes onPlayerTap when a player row is tapped', async () => {
    const user = userEvent.setup();
    const onPlayerTap = jest.fn();
    const vm = createBaseViewModel({
      players: [
        {
          id: 'p1',
          username: 'Alice',
          playerNumber: 1,
          colorClass: 'bg-blue-500',
          isCurrentPlayer: true,
          isUserPlayer: true,
          timeRemaining: 12_000,
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
      ] as any,
    });

    render(<MobileGameHUD viewModel={vm} onPlayerTap={onPlayerTap} />);

    await user.click(screen.getByRole('button', { name: /You|Alice/ }));

    expect(onPlayerTap).toHaveBeenCalledWith(
      expect.objectContaining({
        id: 'p1',
        username: 'Alice',
        playerNumber: 1,
      })
    );
  });
});
