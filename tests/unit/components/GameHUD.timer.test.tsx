import React from 'react';
import { render, screen, act } from '@testing-library/react';
import '@testing-library/jest-dom';
import { GameHUD } from '../../../src/client/components/GameHUD';
import type { HUDViewModel } from '../../../src/client/adapters/gameViewModels';
import type { GameState, Player } from '../../../src/shared/types/game';

jest.useFakeTimers();

function basePlayers(): Player[] {
  return [
    {
      id: 'p1',
      username: 'Alice',
      playerNumber: 1,
      type: 'human',
      isReady: true,
      timeRemaining: 70_000,
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
      timeRemaining: 50_000,
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
        timeRemaining: 70_000,
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
        timeRemaining: 50_000,
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
      decisionDeadlineMs: null,
      reconciledDecisionTimeRemainingMs: null,
      isServerCapped: false,
    },
    connectionStatus: 'connected',
    isSpectator: false,
    isLocalSandboxOnly: false,
  };
}

describe('GameHUD player timers', () => {
  it('ticks down active player timer each second', () => {
    const hud = baseHudViewModel();
    const state = baseGameState();

    render(<GameHUD viewModel={hud} timeControl={state.timeControl} />);

    // Initial time for active player (p1) should render minutes:seconds.
    expect(screen.getByText('1:10')).toBeInTheDocument();

    act(() => {
      jest.advanceTimersByTime(2_000);
    });

    // After 2 seconds, timer should decrease.
    expect(screen.getByText('1:08')).toBeInTheDocument();
  });

  it('highlights low time in red when under one minute', () => {
    const hud = baseHudViewModel();
    // Simulate low time for active player.
    hud.players[0].timeRemaining = 50_000;
    const state = baseGameState();
    state.players[0].timeRemaining = 50_000;

    render(<GameHUD viewModel={hud} timeControl={state.timeControl} />);

    const timer = screen.getByText('0:50');
    expect(timer).toHaveClass('text-red-600');
  });
});
