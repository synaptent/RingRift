import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import '@testing-library/jest-dom';
import { GameHUD } from '../../../src/client/components/GameHUD';
import type { HUDViewModel } from '../../../src/client/adapters/gameViewModels';
import type { Player } from '../../../src/shared/types/game';
import {
  logRulesUxEvent,
  sendRulesUxEvent,
  logHelpOpenEvent,
} from '../../../src/client/utils/rulesUxTelemetry';

jest.mock('../../../src/client/utils/rulesUxTelemetry', () => {
  const actual = jest.requireActual('../../../src/client/utils/rulesUxTelemetry');
  return {
    ...actual,
    logRulesUxEvent: jest.fn().mockResolvedValue(undefined),
    sendRulesUxEvent: jest.fn().mockResolvedValue(undefined),
    logHelpOpenEvent: jest.fn().mockResolvedValue(undefined),
    newOverlaySessionId: jest.fn(() => 'overlay-session-test'),
    newHelpSessionId: jest.fn(() => 'help-session-test'),
  };
});

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

function createWeirdStateHUDViewModel(): HUDViewModel {
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
        aiInfo: { isAI: false, difficulty: 0, difficultyLabel: '', difficultyColor: '', difficultyBgColor: '', aiTypeLabel: '' },
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
        aiInfo: { isAI: false, difficulty: 0, difficultyLabel: '', difficultyColor: '', difficultyBgColor: '', aiTypeLabel: '' },
      },
    ],
    turnNumber: 5,
    moveNumber: 12,
    connectionStatus: 'connected',
    isConnectionStale: false,
    isSpectator: false,
    spectatorCount: 0,
    weirdState: {
      type: 'forced-elimination',
      title: 'Forced Elimination in progress',
      body: 'You have no legal moves; the rules will remove caps from your stacks.',
      tone: 'critical',
    },
  };
}

describe('GameHUD – weird-state help & telemetry', () => {
  afterEach(() => {
    jest.clearAllMocks();
  });

  it('opens forced-elimination teaching overlay and emits telemetry', async () => {
    const viewModel = createWeirdStateHUDViewModel();

    render(
      <GameHUD
        viewModel={viewModel}
        rulesUxContext={{ boardType: 'square8', numPlayers: 2, rulesConcept: 'anm_forced_elimination' }}
      />
    );

    // Banner impression should be logged when weird state is shown.
    await waitFor(() => {
      expect(logRulesUxEvent).toHaveBeenCalledWith(
        expect.objectContaining({
          type: 'weird_state_banner_impression',
          boardType: 'square8',
          numPlayers: 2,
          weirdStateType: 'forced-elimination',
          overlaySessionId: 'overlay-session-test',
        })
      );
    });

    // Open the help chip to surface TeachingOverlay.
    fireEvent.click(screen.getByTestId('hud-weird-state-help'));

    await waitFor(() => {
      expect(screen.getByRole('dialog')).toBeInTheDocument();
      expect(screen.getByRole('heading', { name: /Forced Elimination/i })).toBeInTheDocument();
    });

    // Telemetry for the help open should be emitted.
    await waitFor(() => {
      expect(logRulesUxEvent).toHaveBeenCalledWith(
        expect.objectContaining({
          type: 'weird_state_details_open',
          boardType: 'square8',
          numPlayers: 2,
          weirdStateType: 'forced-elimination',
          topic: 'forced_elimination',
          overlaySessionId: 'overlay-session-test',
        })
      );
      expect(sendRulesUxEvent).toHaveBeenCalledWith(
        expect.objectContaining({
          type: 'rules_weird_state_help',
          boardType: 'square8',
          numPlayers: 2,
          weirdStateType: 'forced-elimination',
          topic: 'forced_elimination',
        })
      );
      expect(logHelpOpenEvent).toHaveBeenCalledWith(
        expect.objectContaining({
          entrypoint: 'hud_help_chip',
          boardType: 'square8',
          numPlayers: 2,
        })
      );
    });
  });
});
