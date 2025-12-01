import React from 'react';
import { render, screen } from '@testing-library/react';
import '@testing-library/jest-dom';
import { GameHUD } from '../../../src/client/components/GameHUD';
import type { GameState, Player } from '../../../src/shared/types/game';
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
      username: 'Bot',
      playerNumber: 2,
      type: 'ai',
      isReady: true,
      timeRemaining: 90_000,
      ringsInHand: 4,
      eliminatedRings: 2,
      territorySpaces: 0,
      aiDifficulty: 3,
      aiProfile: { difficulty: 3, aiType: 'minimax' },
    },
  ];
}

function createGameState(): GameState {
  const players = createPlayers();
  return {
    id: 'game-1',
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

describe('GameHUD – legacy props', () => {
  it('renders connection status, spectator badge, ring and territory stats', () => {
    const gameState = createGameState();
    const currentPlayer = gameState.players[0];

    render(
      <GameHUD
        gameState={gameState}
        currentPlayer={currentPlayer}
        instruction="Select a stack to move."
        connectionStatus="connected"
        isSpectator={true}
        currentUserId="p1"
      />
    );

    // Connection line and instruction banner
    expect(screen.getByText(/Connection: Connected/i)).toBeInTheDocument();
    expect(screen.getByText('Select a stack to move.')).toBeInTheDocument();

    // Spectator badge
    expect(screen.getByText(/Spectator/i)).toBeInTheDocument();

    // Ring stats labels
    expect(screen.getAllByText('In Hand').length).toBeGreaterThan(0);
    expect(screen.getAllByText('On Board').length).toBeGreaterThan(0);
    expect(screen.getAllByText('Captured').length).toBeGreaterThan(0);

    // Territory stats: only first player has non-zero spaces
    // The text is split across elements (<span>2</span> territory spaces)
    // so we use a custom matcher that checks the full text content
    expect(
      screen.getByText((_, element) => element?.textContent === '2 territory spaces')
    ).toBeInTheDocument();
  });
});

describe('GameHUD – view-model props', () => {
  function createHUDViewModel(): HUDViewModel {
    return {
      phase: {
        label: 'Movement Phase',
        description: 'Move a stack or capture opponent pieces',
        icon: '⚡',
        colorClass: 'bg-green-500',
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
          username: 'Bot',
          playerNumber: 2,
          colorClass: 'bg-red-500',
          isCurrentPlayer: false,
          isUserPlayer: false,
          timeRemaining: 90_000,
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
      turnNumber: 3,
      moveNumber: 7,
      instruction: 'Select a stack to move.',
      connectionStatus: 'connected',
      isConnectionStale: true,
      isSpectator: false,
      spectatorCount: 1,
      subPhaseDetail: 'Sub-phase detail',
      decisionPhase: undefined,
    };
  }

  it('renders connection status, spectator count, phase, players, and timers from view model', () => {
    jest.useFakeTimers();
    const viewModel = createHUDViewModel();

    render(
      <GameHUD
        viewModel={viewModel}
        timeControl={{ type: 'rapid', initialTime: 600, increment: 0 }}
      />
    );

    // Connection label with stale hint
    expect(screen.getByText(/Connection: Connected/i)).toBeInTheDocument();
    expect(screen.getByText(/\(awaiting update…\)/i)).toBeInTheDocument();

    // Spectator count badge (there may be multiple '1's like timer "1:30")
    // Just verify at least one '1' exists; SVG eye icon accompanies spectator count
    expect(screen.getAllByText('1').length).toBeGreaterThan(0);

    // Phase label and description
    expect(screen.getByText(/Movement Phase/i)).toBeInTheDocument();
    expect(screen.getByText(/Move a stack or capture opponent pieces/i)).toBeInTheDocument();

    // Instruction banner
    expect(screen.getByText('Select a stack to move.')).toBeInTheDocument();

    // Player cards: current player + AI badge + territory spaces
    expect(screen.getByText('Alice')).toBeInTheDocument();
    expect(screen.getByText('Bot')).toBeInTheDocument();
    // Territory spaces text is split across elements, verify at least one matches
    expect(
      screen.getAllByText(
        (_, element) => element?.textContent?.includes('territory space') ?? false
      ).length
    ).toBeGreaterThan(0);
    expect(screen.getByText(/AI/)).toBeInTheDocument();
    expect(screen.getByText(/Advanced · Minimax Lv3/)).toBeInTheDocument();
    // "Minimax" appears in both difficulty badge and AI type label
    expect(screen.getAllByText(/Minimax/).length).toBeGreaterThan(0);

    // Timers render in mm:ss format
    expect(screen.getAllByText(/2:00|1:30/).length).toBeGreaterThan(0);

    jest.useRealTimers();
  });
});
