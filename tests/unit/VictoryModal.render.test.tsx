import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import { VictoryModal } from '../../src/client/components/VictoryModal';
import type { GameResult, GameState, Player } from '../../src/shared/types/game';

jest.mock('../../src/client/utils/rulesUxTelemetry', () => ({
  logRulesUxEvent: jest.fn(),
  newOverlaySessionId: jest.fn(() => 'overlay-session'),
}));

jest.mock('../../src/client/components/TeachingOverlay', () => {
  const React = require('react');
  return {
    TeachingOverlay: ({ children }: { children: React.ReactNode }) =>
      React.createElement('div', { 'data-testid': 'teaching-overlay' }, children),
    useTeachingOverlay: () => ({
      currentTopic: null,
      isOpen: false,
      showTopic: jest.fn(),
      hideTopic: jest.fn(),
    }),
  };
});

function buildPlayers(): Player[] {
  return [
    {
      id: 'p1',
      username: 'Alice',
      playerNumber: 1,
      type: 'human',
      isReady: true,
      timeRemaining: 0,
      ringsInHand: 3,
      eliminatedRings: 15,
      territorySpaces: 20,
    },
    {
      id: 'p2',
      username: 'Bob',
      playerNumber: 2,
      type: 'human',
      isReady: true,
      timeRemaining: 0,
      ringsInHand: 10,
      eliminatedRings: 8,
      territorySpaces: 10,
    },
  ];
}

function buildGameResult(): GameResult {
  return {
    winner: 1,
    reason: 'ring_elimination',
    finalScore: {
      ringsEliminated: { 1: 15, 2: 8 },
      territorySpaces: { 1: 20, 2: 10 },
      ringsRemaining: { 1: 3, 2: 10 },
    },
  };
}

function buildGameState(players: Player[]): GameState {
  const now = new Date();
  return {
    id: 'g1',
    boardType: 'square8',
    board: {
      stacks: new Map(),
      markers: new Map(),
      collapsedSpaces: new Map(),
      territories: new Map(),
      formedLines: [],
      eliminatedRings: { 1: 15, 2: 8 },
      size: 8,
      type: 'square8',
    },
    players,
    currentPhase: 'game_over',
    currentPlayer: 1,
    moveHistory: [],
    history: [],
    timeControl: { type: 'rapid', initialTime: 600, increment: 0 },
    spectators: [],
    gameStatus: 'finished',
    createdAt: now,
    lastMoveAt: now,
    isRated: false,
    maxPlayers: 2,
    totalRingsInPlay: 36,
    totalRingsEliminated: 23,
    victoryThreshold: 18, // RR-CANON-R061: ringsPerPlayer
    territoryVictoryThreshold: 33,
  };
}

describe('VictoryModal render', () => {
  it('renders winner summary and stats, and closes on backdrop click', () => {
    const players = buildPlayers();
    const gameResult = buildGameResult();
    const gameState = buildGameState(players);
    const onClose = jest.fn();
    const onReturnToLobby = jest.fn();

    render(
      <VictoryModal
        isOpen
        gameResult={gameResult}
        players={players}
        gameState={gameState}
        onClose={onClose}
        onReturnToLobby={onReturnToLobby}
      />
    );

    expect(screen.getAllByText(/Alice/i).length).toBeGreaterThan(0);
    expect(screen.getAllByText(/Bob/i).length).toBeGreaterThan(0);
    expect(screen.getByText(/Rings Eliminated/i)).toBeInTheDocument();
    expect(screen.getByText(/Victory/i)).toBeInTheDocument();

    // Backdrop click closes modal
    fireEvent.click(screen.getByRole('dialog'));
    expect(onClose).toHaveBeenCalled();
  });

  it('emits weird_state_banner_impression once per open', () => {
    const { logRulesUxEvent } = require('../../src/client/utils/rulesUxTelemetry');
    (logRulesUxEvent as jest.Mock).mockClear();

    const players = buildPlayers();
    const gameState = buildGameState(players);
    const gameResult: GameResult = {
      ...buildGameResult(),
      reason: 'game_completed', // maps to structural-stalemate weird state
    };

    const { rerender } = render(
      <VictoryModal
        isOpen
        gameResult={gameResult}
        players={players}
        gameState={gameState}
        onClose={jest.fn()}
        onReturnToLobby={jest.fn()}
      />
    );

    expect(logRulesUxEvent).toHaveBeenCalledTimes(1);

    // Rerender with the same references should not emit again
    rerender(
      <VictoryModal
        isOpen
        gameResult={gameResult}
        players={players}
        gameState={gameState}
        onClose={jest.fn()}
        onReturnToLobby={jest.fn()}
      />
    );

    expect(logRulesUxEvent).toHaveBeenCalledTimes(1);
  });

  it('allows weird_state_banner_impression to fire again after close/reopen', () => {
    const { logRulesUxEvent } = require('../../src/client/utils/rulesUxTelemetry');
    (logRulesUxEvent as jest.Mock).mockClear();

    const players = buildPlayers();
    const gameState = buildGameState(players);
    const gameResult: GameResult = {
      ...buildGameResult(),
      reason: 'game_completed',
    };

    const { rerender } = render(
      <VictoryModal
        isOpen
        gameResult={gameResult}
        players={players}
        gameState={gameState}
        onClose={jest.fn()}
        onReturnToLobby={jest.fn()}
      />
    );

    expect(logRulesUxEvent).toHaveBeenCalledTimes(1);

    // Close the modal
    rerender(
      <VictoryModal
        isOpen={false}
        gameResult={gameResult}
        players={players}
        gameState={gameState}
        onClose={jest.fn()}
        onReturnToLobby={jest.fn()}
      />
    );

    // Reopen should log again
    rerender(
      <VictoryModal
        isOpen
        gameResult={gameResult}
        players={players}
        gameState={gameState}
        onClose={jest.fn()}
        onReturnToLobby={jest.fn()}
      />
    );

    expect(logRulesUxEvent).toHaveBeenCalledTimes(2);
  });
});
