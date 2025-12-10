import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import '@testing-library/jest-dom';
import { VictoryModal } from '../../src/client/components/VictoryModal';
import type { GameResult, Player, GameState, BoardState } from '../../src/shared/types/game';
import type { GameEndExplanation } from '../../src/shared/engine/gameEndExplanation';
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
    victoryThreshold: 18, // RR-CANON-R061: ringsPerPlayer
    territoryVictoryThreshold: 33,
  };
}

describe('VictoryModal rules-UX telemetry for weird states', () => {
  const noop = () => {};

  beforeEach(() => {
    mockLogRulesUxEvent.mockReset();
  });

  it('uses GameEndExplanation weird-state context (ANM/FE) for telemetry and emits once', async () => {
    const players = createPlayers();
    const gameResult: GameResult = {
      reason: 'last_player_standing',
      winner: 1,
      finalScore: {
        ringsEliminated: { 1: 12, 2: 12 },
        territorySpaces: { 1: 16, 2: 16 },
        ringsRemaining: { 1: 0, 2: 0 },
      },
    };
    const gameState = createGameState(players);
    const explanation: GameEndExplanation = {
      boardType: 'square8',
      numPlayers: players.length,
      winnerPlayerId: 'user1',
      outcomeType: 'last_player_standing',
      victoryReasonCode: 'victory_last_player_standing',
      uxCopy: { shortSummaryKey: 'game_end.lps.with_anm_fe' },
      weirdStateContext: {
        // Prefer forced-elimination tagging over the fallback LPS mapping
        reasonCodes: ['FE_SEQUENCE_CURRENT_PLAYER', 'ANM_MOVEMENT_FE_BLOCKED'],
        primaryReasonCode: 'FE_SEQUENCE_CURRENT_PLAYER',
        rulesContextTags: ['anm_forced_elimination'],
      },
    };

    render(
      <VictoryModal
        isOpen
        gameResult={gameResult}
        players={players}
        gameState={gameState}
        gameEndExplanation={explanation}
        onClose={noop}
        onReturnToLobby={noop}
        isSandbox={false}
      />
    );

    await waitFor(() => {
      expect(mockLogRulesUxEvent).toHaveBeenCalledTimes(1);
    });

    const [event] = mockLogRulesUxEvent.mock.calls[0] as any[];
    expect(event).toMatchObject({
      type: 'weird_state_banner_impression',
      rulesContext: 'anm_forced_elimination',
      reasonCode: 'FE_SEQUENCE_CURRENT_PLAYER',
      weirdStateType: 'active-no-moves-movement',
      boardType: 'square8',
      numPlayers: players.length,
      isSandbox: false,
      isRanked: true,
    });
  });

  it('does not emit weird-state telemetry for standard ring elimination victories', () => {
    const players = createPlayers();
    const gameResult = createGameResult('ring_elimination');
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

    expect(mockLogRulesUxEvent).not.toHaveBeenCalled();
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
