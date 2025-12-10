import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import '@testing-library/jest-dom';
import { GameHUD } from '../../../src/client/components/GameHUD';
import type { Player } from '../../../src/shared/types/game';
import type { HUDViewModel } from '../../../src/client/adapters/gameViewModels';
import {
  logHelpOpenEvent,
  sendRulesUxEvent,
} from '../../../src/client/utils/rulesUxTelemetry';

jest.mock('../../../src/client/utils/rulesUxTelemetry', () => {
  const actual = jest.requireActual('../../../src/client/utils/rulesUxTelemetry');
  return {
    ...actual,
    logRulesUxEvent: jest.fn().mockResolvedValue(undefined),
    logHelpOpenEvent: jest.fn().mockResolvedValue(undefined),
    sendRulesUxEvent: jest.fn().mockResolvedValue(undefined),
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

function baseHUDViewModel(): HUDViewModel {
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
    turnNumber: 5,
    moveNumber: 12,
    connectionStatus: 'connected',
    isConnectionStale: false,
    isSpectator: false,
    spectatorCount: 0,
  };
}

describe('GameHUD – contextual help open flows', () => {
  afterEach(() => {
    jest.clearAllMocks();
  });

  it('opens phase help overlay and emits help telemetry', async () => {
    const viewModel = baseHUDViewModel();

    render(
      <GameHUD
        viewModel={viewModel}
        rulesUxContext={{ boardType: 'square8', numPlayers: 2, rulesConcept: 'movement_basic' }}
      />
    );

    fireEvent.click(screen.getByTestId('hud-phase-help-movement'));

    await waitFor(() => {
      expect(screen.getByRole('heading', { name: /Stack Movement/i })).toBeInTheDocument();
    });

    await waitFor(() => {
      expect(logHelpOpenEvent).toHaveBeenCalledWith(
        expect.objectContaining({
          entrypoint: 'hud_help_chip',
          boardType: 'square8',
          numPlayers: 2,
        })
      );
      expect(sendRulesUxEvent).toHaveBeenCalledWith(
        expect.objectContaining({
          type: 'rules_help_open',
          boardType: 'square8',
          numPlayers: 2,
          topic: 'stack_movement',
        })
      );
    });
  });

  it('auto-opens scenario help when configured for the active phase', async () => {
    const viewModel = {
      ...baseHUDViewModel(),
      phase: {
        ...baseHUDViewModel().phase,
        phaseKey: 'territory_processing',
        label: 'Territory Processing',
      },
    };

    render(
      <GameHUD
        viewModel={viewModel}
        rulesUxContext={{
          boardType: 'square8',
          numPlayers: 2,
          rulesConcept: 'territory_mini_region_q23',
          scenarioId: 'scenario-123',
        }}
      />
    );

    await waitFor(() => {
      expect(screen.getByRole('heading', { name: /Territory/i })).toBeInTheDocument();
    });

    await waitFor(() => {
      expect(logHelpOpenEvent).toHaveBeenCalledWith(
        expect.objectContaining({
          entrypoint: 'hud_help_chip',
          boardType: 'square8',
          numPlayers: 2,
          scenarioId: 'scenario-123',
        })
      );
      expect(sendRulesUxEvent).toHaveBeenCalledWith(
        expect.objectContaining({
          type: 'rules_help_open',
          boardType: 'square8',
          numPlayers: 2,
          topic: 'territory',
          scenarioId: 'scenario-123',
        })
      );
    });
  });
});
