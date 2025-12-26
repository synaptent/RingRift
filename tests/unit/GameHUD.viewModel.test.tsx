import React from 'react';
import { render, screen } from '@testing-library/react';
import '@testing-library/jest-dom';
import { GameHUD } from '../../src/client/components/GameHUD';
import { toHUDViewModel } from '../../src/client/adapters/gameViewModels';
import type { GameState, Player, BoardState } from '../../src/shared/types/game';

function createTestGameState(overrides: Partial<GameState> = {}): GameState {
  const board: BoardState = {
    stacks: new Map(),
    markers: new Map(),
    collapsedSpaces: new Map(),
    territories: new Map(),
    formedLines: [],
    eliminatedRings: {},
    size: 8,
    type: 'square8',
  };

  const players: Player[] = [
    {
      id: 'p1',
      username: 'Alice',
      playerNumber: 1,
      type: 'human',
      isReady: true,
      timeRemaining: 5 * 60 * 1000,
      ringsInHand: 18,
      eliminatedRings: 0,
      territorySpaces: 0,
    },
    {
      id: 'p2',
      username: 'Bob',
      playerNumber: 2,
      type: 'ai',
      isReady: true,
      timeRemaining: 4 * 60 * 1000,
      ringsInHand: 18,
      eliminatedRings: 0,
      territorySpaces: 0,
      aiDifficulty: 5,
      aiProfile: { difficulty: 5, aiType: 'heuristic' },
    },
  ];

  const base: GameState = {
    id: 'game-1',
    boardType: 'square8',
    board,
    players,
    currentPhase: 'movement',
    currentPlayer: 1,
    moveHistory: [],
    history: [],
    timeControl: { type: 'rapid', initialTime: 600, increment: 0 },
    spectators: ['s1'],
    gameStatus: 'active',
    createdAt: new Date(),
    lastMoveAt: new Date(),
    isRated: false,
    maxPlayers: 2,
    totalRingsInPlay: 0,
    totalRingsEliminated: 0,
    victoryThreshold: 10,
    territoryVictoryThreshold: 32,
  };

  return { ...base, ...overrides };
}

describe('GameHUD (view model path)', () => {
  it('renders phase, connection status, and players from HUD view model', () => {
    const gameState = createTestGameState();
    const hud = toHUDViewModel(gameState, {
      instruction: 'Select a stack to move.',
      connectionStatus: 'connected',
      lastHeartbeatAt: Date.now(),
      isSpectator: false,
      currentUserId: 'p1',
    });

    render(<GameHUD viewModel={hud} timeControl={gameState.timeControl} />);

    expect(screen.getByTestId('game-hud')).toBeInTheDocument();

    // Assert stable, view-model-driven semantics (phase indicator + current-turn indicator)
    expect(screen.getByTestId('phase-indicator')).toBeInTheDocument();
    expect(screen.getByText(hud.phase.label)).toBeInTheDocument();
    expect(screen.getByText(/Connection: Connected/)).toBeInTheDocument();

    // Action hint should surface for the active local player
    expect(screen.getByTestId('phase-action-hint')).toHaveTextContent(hud.phase.actionHint);

    // Instruction banner is passed through from the view model
    expect(screen.getByText('Select a stack to move.')).toBeInTheDocument();

    // Player names appear in both player cards and score summary, so use getAllByText
    expect(screen.getAllByText('Alice').length).toBeGreaterThanOrEqual(1);
    expect(screen.getAllByText('Bob').length).toBeGreaterThanOrEqual(1);
  });

  it('shows spectator badge and victory conditions helper when spectating', () => {
    const gameState = createTestGameState();
    const hud = toHUDViewModel(gameState, {
      instruction: undefined,
      connectionStatus: 'reconnecting',
      lastHeartbeatAt: Date.now() - 10_000,
      isSpectator: true,
      currentUserId: 'observer',
    });

    render(<GameHUD viewModel={hud} timeControl={gameState.timeControl} />);

    expect(screen.getByText('Spectator Mode')).toBeInTheDocument();
    expect(screen.getByTestId('victory-conditions-help')).toBeInTheDocument();
  });
});
