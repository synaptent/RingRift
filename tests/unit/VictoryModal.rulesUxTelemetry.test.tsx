import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import '@testing-library/jest-dom';
import { VictoryModal } from '../../src/client/components/VictoryModal';
import type { GameResult, Player, GameState, BoardState } from '../../src/shared/types/game';
import * as rulesUxTelemetry from '../../src/client/utils/rulesUxTelemetry';

jest.mock('../../src/client/utils/rulesUxTelemetry', () => {
  const actual = jest.requireActual('../../src/client/utils/rulesUxTelemetry');
  return {
    __esModule: true,
    ...actual,
    logRulesUxEvent: jest.fn(),
    // Use a stable overlay session id so we can assert linkage between
    // banner impressions, detail opens, and TeachingOverlay lifecycle.
    newOverlaySessionId: jest.fn(() => 'overlay-session-victory-1'),
  };
});

const mockLogRulesUxEvent = rulesUxTelemetry.logRulesUxEvent as jest.MockedFunction<
  typeof rulesUxTelemetry.logRulesUxEvent
>;

function createGameResult(reason: GameResult['reason']): GameResult {
  return {
    winner: undefined,
    reason,
    finalScore: {
      ringsEliminated: { 1: 10, 2: 10 },
      territorySpaces: { 1: 20, 2: 20 },
      ringsRemaining: { 1: 0, 2: 0 },
    },
  };
}

function createPlayers(): Player[] {
  return [
    {
      id: 'user1',
      username: 'Alice',
      playerNumber: 1,
      type: 'human',
      isReady: true,
      timeRemaining: 0,
      ringsInHand: 0,
      eliminatedRings: 10,
      territorySpaces: 20,
    },
    {
      id: 'user2',
      username: 'Bob',
      playerNumber: 2,
      type: 'human',
      isReady: true,
      timeRemaining: 0,
      ringsInHand: 0,
      eliminatedRings: 10,
      territorySpaces: 20,
    },
  ];
}

function createGameState(players: Player[]): GameState {
  const board: BoardState = {
    stacks: new Map(),
    markers: new Map(),
    collapsedSpaces: new Map(),
    territories: new Map(),
    formedLines: [],
    eliminatedRings: { 1: 10, 2: 10 },
    size: 8,
    type: 'square8',
  };

  return {
    id: 'game-structural-stalemate',
    boardType: 'square8',
    board,
    players,
    currentPhase: 'movement',
    currentPlayer: 1,
    moveHistory: [],
    history: [],
    timeControl: { type: 'rapid', initialTime: 600, increment: 0 },
    spectators: [],
    gameStatus: 'finished',
    createdAt: new Date(),
    lastMoveAt: new Date(),
    isRated: true,
    maxPlayers: players.length,
    totalRingsInPlay: 40,
    totalRingsEliminated: 20,
    victoryThreshold: 19,
    territoryVictoryThreshold: 33,
  };
}

describe('VictoryModal rules-UX telemetry for weird states', () => {
  const noop = () => {};

  beforeEach(() => {
    mockLogRulesUxEvent.mockReset();
  });

  it('emits weird_state_banner_impression for structural-stalemate (game_completed) results', async () => {
    const players = createPlayers();
    const gameResult = createGameResult('game_completed');
    const gameState = createGameState(players);

    render(
      <VictoryModal
        isOpen
        gameResult={gameResult}
        players={players}
        gameState={gameState}
        onClose={noop}
        onReturnToLobby={noop}
        isSandbox={false}
      />
    );

    await waitFor(() => {
      expect(mockLogRulesUxEvent).toHaveBeenCalled();
    });

    const events = mockLogRulesUxEvent.mock.calls.map(([arg]) => arg as any);
    const impression = events.find((event) => event.type === 'weird_state_banner_impression');
    expect(impression).toBeDefined();

    expect(impression).toMatchObject({
      type: 'weird_state_banner_impression',
      source: 'victory_modal',
      boardType: 'square8',
      numPlayers: players.length,
      rulesContext: 'structural_stalemate',
      weirdStateType: 'structural-stalemate',
      reasonCode: 'STRUCTURAL_STALEMATE_TIEBREAK',
      isSandbox: false,
      isRanked: true,
      overlaySessionId: 'overlay-session-victory-1',
    });
  });

  it('emits weird_state_details_open with the same overlaySessionId when "What happened?" is clicked', async () => {
    const players = createPlayers();
    const gameResult = createGameResult('game_completed');
    const gameState = createGameState(players);

    render(
      <VictoryModal
        isOpen
        gameResult={gameResult}
        players={players}
        gameState={gameState}
        onClose={noop}
        onReturnToLobby={noop}
        isSandbox
      />
    );

    const detailsButton = await screen.findByRole('button', { name: /what happened\?/i });
    fireEvent.click(detailsButton);

    await waitFor(() => {
      const events = mockLogRulesUxEvent.mock.calls.map(([arg]) => arg as any);
      expect(events.some((event) => event.type === 'weird_state_details_open')).toBeTruthy();
    });

    const events = mockLogRulesUxEvent.mock.calls.map(([arg]) => arg as any);
    const impression = events.find((event) => event.type === 'weird_state_banner_impression');
    const details = events.find((event) => event.type === 'weird_state_details_open');

    expect(impression).toBeDefined();
    expect(details).toBeDefined();

    expect(details).toMatchObject({
      type: 'weird_state_details_open',
      source: 'victory_modal',
      boardType: 'square8',
      numPlayers: players.length,
      rulesContext: 'structural_stalemate',
      weirdStateType: 'structural-stalemate',
      reasonCode: 'STRUCTURAL_STALEMATE_TIEBREAK',
      isSandbox: true,
      isRanked: true,
      overlaySessionId: 'overlay-session-victory-1',
      topic: 'victory_stalemate',
    });

    // Banner impression and details open should share the same overlay session id.
    expect(details.overlaySessionId).toBe(impression.overlaySessionId);
  });
});
