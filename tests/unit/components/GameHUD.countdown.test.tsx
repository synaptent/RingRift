import React from 'react';
import { render, screen } from '@testing-library/react';
import '@testing-library/jest-dom';
import { GameHUD } from '../../../src/client/components/GameHUD';
import type { HUDViewModel } from '../../../src/client/adapters/gameViewModels';
import type { GameState, Player } from '../../../src/shared/types/game';

function basePlayers(): Player[] {
  return [
    {
      id: 'p1',
      username: 'Alice',
      playerNumber: 1,
      type: 'human',
      isReady: true,
      timeRemaining: 12_000,
      ringsInHand: 5,
      eliminatedRings: 1,
      territorySpaces: 2,
    },
    {
      id: 'p2',
      username: 'Bot',
      playerNumber: 2,
      type: 'ai',
      isReady: true,
      timeRemaining: 9_000,
      ringsInHand: 4,
      eliminatedRings: 2,
      territorySpaces: 0,
      aiDifficulty: 3,
      aiProfile: { difficulty: 3, aiType: 'minimax' },
    },
  ];
}

function baseGameState(): GameState {
  const players = basePlayers();
  return {
    id: 'g1',
    boardType: 'square8',
    board: {
      stacks: new Map(),
      markers: new Map(),
      collapsedSpaces: new Map(),
      territories: new Map(),
      formedLines: [],
      eliminatedRings: {},
      size: 8,
      type: 'square8',
    },
    players,
    currentPhase: 'movement',
    currentPlayer: 1,
    moveHistory: [],
    history: [],
    timeControl: { type: 'rapid', initialTime: 600, increment: 0 },
    spectators: [],
    gameStatus: 'active',
    createdAt: new Date(),
    lastMoveAt: new Date(),
    isRated: false,
    maxPlayers: players.length,
    totalRingsInPlay: 0,
    totalRingsEliminated: 0,
    victoryThreshold: 10,
    territoryVictoryThreshold: 32,
  };
}

function baseHudViewModel(): HUDViewModel {
  return {
    phase: {
      phaseKey: 'movement',
      label: 'Movement Phase',
      description: 'Move a stack or capture opponent pieces',
      icon: '⚡',
      colorClass: 'bg-green-500',
      actionHint: 'Select your stack, then click a destination to move',
      spectatorHint: 'Player is choosing a move',
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
        username: 'Bot',
        playerNumber: 2,
        colorClass: 'bg-red-500',
        isCurrentPlayer: false,
        isUserPlayer: false,
        timeRemaining: 9_000,
        ringStats: { inHand: 4, onBoard: 4, eliminated: 2, total: 10 },
        territorySpaces: 0,
        aiInfo: {
          isAI: true,
          difficulty: 3,
          difficultyLabel: 'Advanced · Minimax',
          difficultyColor: 'text-blue-300',
          difficultyBgColor: 'bg-blue-900/40',
          aiTypeLabel: 'Minimax',
        },
      },
    ],
    decisions: [],
    timers: {
      serverTimeOffsetMs: 0,
      decisionDeadlineMs: Date.now() + 4_000,
      reconciledDecisionTimeRemainingMs: 4_000,
      isServerCapped: true,
    },
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
    connectionStatus: 'connected',
    isSpectator: false,
    isLocalSandboxOnly: false,
  };
}

describe('GameHUD decision countdown', () => {
  it('shows decision timer with server-cap styling and countdown seconds', () => {
    const hud = baseHudViewModel();
    const state = baseGameState();

    render(<GameHUD viewModel={hud} timeControl={state.timeControl} />);

    const timer = screen.getByTestId('decision-phase-countdown');
    expect(timer).toBeInTheDocument();
    expect(timer).toHaveAttribute('data-severity', 'warning');
    expect(timer).toHaveAttribute('data-server-capped', 'true');
    expect(screen.getByText(/Server deadline/i)).toBeInTheDocument();
    // The countdown shows 0:04 - check within the timer element
    expect(timer).toHaveTextContent('0:04');
  });

  it('shows critical severity when under 3 seconds', () => {
    const hud = baseHudViewModel();
    hud.decisionPhase = {
      ...hud.decisionPhase!,
      timeRemainingMs: 2_000,
      isServerCapped: false,
    };

    render(<GameHUD viewModel={hud} timeControl={baseGameState().timeControl} />);

    const timer = screen.getByTestId('decision-phase-countdown');
    expect(timer).toHaveAttribute('data-severity', 'critical');
    expect(timer).not.toHaveAttribute('data-server-capped');
    expect(timer).toHaveTextContent('0:02');
  });

  it('hides countdown when showCountdown is false', () => {
    const hud = baseHudViewModel();
    hud.decisionPhase = {
      ...hud.decisionPhase!,
      showCountdown: false,
      timeRemainingMs: 4_000,
    };

    render(<GameHUD viewModel={hud} timeControl={baseGameState().timeControl} />);

    expect(screen.queryByTestId('decision-phase-countdown')).toBeNull();
  });

  it('shows normal severity and remote-actor copy for spectators', () => {
    const hud = baseHudViewModel();
    hud.decisionPhase = {
      ...hud.decisionPhase!,
      actingPlayerNumber: 2,
      actingPlayerName: 'Bot',
      isLocalActor: false,
      timeRemainingMs: 15_000,
      showCountdown: true,
      isServerCapped: false,
    };

    render(<GameHUD viewModel={hud} timeControl={baseGameState().timeControl} />);

    const timer = screen.getByTestId('decision-phase-countdown');
    expect(timer).toHaveAttribute('data-severity', 'normal');
    expect(timer).not.toHaveAttribute('data-server-capped');
    expect(timer).toHaveTextContent('0:15');

    const badge = screen.getByTestId('hud-decision-time-pressure');
    expect(badge).toHaveAttribute('data-severity', 'normal');
    expect(badge).toHaveTextContent("Time left for Bot's decision: 0:15");
  });
});
