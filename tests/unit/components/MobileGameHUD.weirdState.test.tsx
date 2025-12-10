import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import '@testing-library/jest-dom';
import { MobileGameHUD } from '../../../src/client/components/MobileGameHUD';
import type { HUDViewModel } from '../../../src/client/adapters/gameViewModels';
import { sendRulesUxEvent } from '../../../src/client/utils/rulesUxTelemetry';

jest.mock('../../../src/client/utils/rulesUxTelemetry', () => {
  const actual = jest.requireActual('../../../src/client/utils/rulesUxTelemetry');
  return {
    ...actual,
    sendRulesUxEvent: jest.fn().mockResolvedValue(undefined),
  };
});

function baseViewModel(): HUDViewModel {
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
      {
        id: 'p2',
        username: 'Bob',
        playerNumber: 2,
        colorClass: 'bg-red-500',
        isCurrentPlayer: false,
        isUserPlayer: false,
        timeRemaining: 9_000,
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
  };
}

describe('MobileGameHUD – decision timer and weird-state help', () => {
  afterEach(() => {
    jest.clearAllMocks();
  });

  it('shows decision timer severity and server-cap indicator', () => {
    const vm: HUDViewModel = {
      ...baseViewModel(),
      decisionPhase: {
        isActive: true,
        actingPlayerNumber: 1,
        actingPlayerName: 'Alice',
        isLocalActor: true,
        label: 'Choose Line Reward',
        description: 'Select a reward for your formed line',
        shortLabel: 'Line reward',
        timeRemainingMs: 4_000,
        showCountdown: true,
        isServerCapped: true,
        spectatorLabel: 'Waiting for Alice',
      },
    } as HUDViewModel;

    render(<MobileGameHUD viewModel={vm} />);

    const timer = screen.getByTestId('mobile-decision-timer');
    expect(timer).toHaveAttribute('data-severity', 'warning');
    expect(timer).toHaveTextContent('0:04');
    // Server cap indicator shows an asterisk
    expect(timer).toHaveTextContent('*');
  });

  it('emits weird-state help telemetry and opens teaching overlay', () => {
    const vm: HUDViewModel = {
      ...baseViewModel(),
      weirdState: {
        type: 'forced-elimination',
        title: 'Forced Elimination',
        body: 'You have no legal moves; forced eliminations apply.',
        tone: 'warning',
      },
    } as HUDViewModel;

    render(
      <MobileGameHUD
        viewModel={vm}
        rulesUxContext={{
          boardType: 'square8',
          numPlayers: 2,
          rulesConcept: 'anm_forced_elimination',
        }}
      />
    );

    fireEvent.click(screen.getByTestId('mobile-weird-state-help'));

    const overlayHeading = screen.getByRole('heading', { name: /Forced Elimination/i });
    expect(overlayHeading).toBeInTheDocument();

    expect(sendRulesUxEvent).toHaveBeenCalledWith(
      expect.objectContaining({
        type: 'rules_weird_state_help',
        boardType: 'square8',
        numPlayers: 2,
        weirdStateType: 'forced-elimination',
        topic: 'forced_elimination',
      })
    );
  });
});
