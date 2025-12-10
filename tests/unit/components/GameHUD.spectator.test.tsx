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
      timeRemaining: 60_000,
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
      timeRemaining: 60_000,
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
        isUserPlayer: false,
        timeRemaining: 60_000,
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
        timeRemaining: 60_000,
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
    isSpectator: true,
    isLocalSandboxOnly: false,
  };
}

describe('GameHUD spectator banner', () => {
  it('shows spectator banner when isSpectator is true and hides sandbox banner', () => {
    const hud = baseHudViewModel();
    const state = baseGameState();

    render(<GameHUD viewModel={hud} timeControl={state.timeControl} />);

    expect(
      screen.getByLabelText('Spectator Mode - You are watching this game')
    ).toBeInTheDocument();
    expect(screen.queryByTestId('sandbox-local-only-banner')).toBeNull();
  });
});
