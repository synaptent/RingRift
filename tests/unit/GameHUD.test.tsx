import React from 'react';
import { render, screen } from '@testing-library/react';
import { GameHUD } from '../../src/client/components/GameHUD';
import { GameState, Player, BoardState } from '../../shared/types/game';

// Helper to create a minimal test GameState
function createTestGameState(overrides: Partial<GameState> = {}): GameState {
  const defaultBoard: BoardState = {
    stacks: new Map(),
    markers: new Map(),
    collapsedSpaces: new Map(),
    territories: new Map(),
    formedLines: [],
    eliminatedRings: {},
    size: 8,
    type: 'square8',
  };

  const defaultPlayers: Player[] = [
    {
      id: 'p1',
      username: 'Player 1',
      playerNumber: 1,
      type: 'human',
      isReady: true,
      timeRemaining: 600000,
      ringsInHand: 18,
      eliminatedRings: 0,
      territorySpaces: 0,
    },
    {
      id: 'p2',
      username: 'Player 2',
      playerNumber: 2,
      type: 'human',
      isReady: true,
      timeRemaining: 600000,
      ringsInHand: 18,
      eliminatedRings: 0,
      territorySpaces: 0,
    },
  ];

  return {
    id: 'test-game',
    boardType: 'square8',
    board: defaultBoard,
    players: defaultPlayers,
    currentPhase: 'ring_placement',
    currentPlayer: 1,
    moveHistory: [],
    history: [],
    timeControl: { type: 'rapid', initialTime: 600000, increment: 0 },
    spectators: [],
    gameStatus: 'active',
    createdAt: new Date(),
    lastMoveAt: new Date(),
    isRated: false,
    maxPlayers: 2,
    totalRingsInPlay: 0,
    totalRingsEliminated: 0,
    victoryThreshold: 10,
    territoryVictoryThreshold: 32,
    ...overrides,
  };
}

describe('GameHUD', () => {
  it('should display phase indicator', () => {
    const gameState = createTestGameState({ currentPhase: 'movement' });
    const currentPlayer = gameState.players[0];

    render(<GameHUD gameState={gameState} currentPlayer={currentPlayer} />);

    expect(screen.getByText('Movement Phase')).toBeInTheDocument();
    expect(screen.getByText('Move a stack or capture opponent pieces')).toBeInTheDocument();
  });

  it('should highlight current player', () => {
    const gameState = createTestGameState({ currentPlayer: 1 });
    const currentPlayer = gameState.players[0];

    render(<GameHUD gameState={gameState} currentPlayer={currentPlayer} currentUserId="p1" />);

    expect(screen.getByText('Current Turn')).toBeInTheDocument();
  });

  it('should display ring statistics', () => {
    const gameState = createTestGameState();
    const currentPlayer = gameState.players[0];

    render(<GameHUD gameState={gameState} currentPlayer={currentPlayer} />);

    // There is a ring stats block for each player, so these labels may
    // appear multiple times. We only assert that at least one is present.
    expect(screen.getAllByText('In Hand').length).toBeGreaterThan(0);
    expect(screen.getAllByText('On Board').length).toBeGreaterThan(0);
    expect(screen.getAllByText('Lost').length).toBeGreaterThan(0);
  });

  it('should show timer when time controls active', () => {
    const gameState = createTestGameState({
      timeControl: { type: 'rapid', initialTime: 300000, increment: 5000 },
    });
    const currentPlayer = gameState.players[0];

    render(<GameHUD gameState={gameState} currentPlayer={currentPlayer} />);

    // Both players have timers; assert that at least one mm:ss time is shown.
    expect(screen.getAllByText(/\d+:\d{2}/).length).toBeGreaterThan(0);
  });

  it('should display game progress turn counter', () => {
    const gameState = createTestGameState();
    const currentPlayer = gameState.players[0];

    render(<GameHUD gameState={gameState} currentPlayer={currentPlayer} />);

    expect(screen.getByText('Turn')).toBeInTheDocument();
  });

  it('should show AI badge for AI players', () => {
    const gameState = createTestGameState({
      players: [
        {
          id: 'p1',
          username: 'Player 1',
          playerNumber: 1,
          type: 'human',
          isReady: true,
          timeRemaining: 600000,
          ringsInHand: 18,
          eliminatedRings: 0,
          territorySpaces: 0,
        },
        {
          id: 'ai-1',
          username: 'AI Level 5',
          playerNumber: 2,
          type: 'ai',
          isReady: true,
          timeRemaining: 600000,
          ringsInHand: 18,
          eliminatedRings: 0,
          territorySpaces: 0,
          aiDifficulty: 5,
          aiProfile: { difficulty: 5, aiType: 'heuristic' },
        },
      ],
    });
    const currentPlayer = gameState.players[0];

    render(<GameHUD gameState={gameState} currentPlayer={currentPlayer} />);

    expect(screen.getByText('🤖 AI')).toBeInTheDocument();
  });

  it('should display connection status', () => {
    const gameState = createTestGameState();
    const currentPlayer = gameState.players[0];

    render(
      <GameHUD gameState={gameState} currentPlayer={currentPlayer} connectionStatus="connected" />
    );

    expect(screen.getByText(/Connected/)).toBeInTheDocument();
  });

  it('should show spectator badge when spectating', () => {
    const gameState = createTestGameState();
    const currentPlayer = gameState.players[0];

    render(<GameHUD gameState={gameState} currentPlayer={currentPlayer} isSpectator={true} />);

    expect(screen.getByText('Spectator')).toBeInTheDocument();
  });

  it('should display instruction when provided', () => {
    const gameState = createTestGameState();
    const currentPlayer = gameState.players[0];

    render(
      <GameHUD
        gameState={gameState}
        currentPlayer={currentPlayer}
        instruction="Place a ring on an empty edge space."
      />
    );

    expect(screen.getByText('Place a ring on an empty edge space.')).toBeInTheDocument();
  });

  it('should show territory count when player has territory', () => {
    const gameState = createTestGameState({
      players: [
        {
          id: 'p1',
          username: 'Player 1',
          playerNumber: 1,
          type: 'human',
          isReady: true,
          timeRemaining: 600000,
          ringsInHand: 18,
          eliminatedRings: 0,
          territorySpaces: 5,
        },
        {
          id: 'p2',
          username: 'Player 2',
          playerNumber: 2,
          type: 'human',
          isReady: true,
          timeRemaining: 600000,
          ringsInHand: 18,
          eliminatedRings: 0,
          territorySpaces: 0,
        },
      ],
    });
    const currentPlayer = gameState.players[0];

    render(<GameHUD gameState={gameState} currentPlayer={currentPlayer} />);

    // Territory summary text should be rendered for the player with territory.
    expect(screen.getByText(/territory space/)).toBeInTheDocument();
  });

  it('should display different phase colors and icons', () => {
    const phases: Array<GameState['currentPhase']> = [
      'ring_placement',
      'movement',
      'chain_capture',
      'line_processing',
      'territory_processing',
    ];

    phases.forEach((phase) => {
      const gameState = createTestGameState({ currentPhase: phase });
      const currentPlayer = gameState.players[0];

      const { container } = render(<GameHUD gameState={gameState} currentPlayer={currentPlayer} />);

      // Should have phase-specific styling
      const phaseIndicator = container.querySelector('[class*="bg-"]');
      expect(phaseIndicator).toBeInTheDocument();
    });
  });
});
