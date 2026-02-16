import React from 'react';
import { render, screen } from '@testing-library/react';
import '@testing-library/jest-dom';
import { VictoryModal } from '../../src/client/components/VictoryModal';
import { toVictoryViewModel } from '../../src/client/adapters/gameViewModels';
import { getGameOverBannerText } from '../../src/client/utils/gameCopy';
import type { GameResult, Player, GameState, BoardState } from '../../src/shared/types/game';
import { RulesUxPhrases } from './rulesUxExpectations.testutil';
import * as rulesUxTelemetry from '../../src/client/utils/rulesUxTelemetry';

// Stub rules-UX telemetry so VictoryModal tests do not perform real network calls.
// We keep the rest of the module's behaviour intact.
jest.mock('../../src/client/utils/rulesUxTelemetry', () => ({
  __esModule: true,
  // Preserve actual exports so other helpers (e.g. id generators) remain usable.
  ...jest.requireActual('../../src/client/utils/rulesUxTelemetry'),
  // Replace logRulesUxEvent with a Jest mock to avoid HTTP requests and keep tests deterministic.
  logRulesUxEvent: jest.fn(),
}));

// Explicitly reference the mocked function so TypeScript understands the mock shape.
const _mockLogRulesUxEvent = rulesUxTelemetry.logRulesUxEvent as jest.MockedFunction<
  typeof rulesUxTelemetry.logRulesUxEvent
>;

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

function createPlayers(): Player[] {
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
      territorySpaces: 25,
    } as Player,
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
    } as Player,
  ];
}

function createBoardState(): BoardState {
  return {
    stacks: new Map(),
    markers: new Map(),
    collapsedSpaces: new Map(),
    territories: new Map(),
    formedLines: [],
    eliminatedRings: { 1: 15, 2: 8 },
    size: 8,
    type: 'square8',
  } as BoardState;
}

function createGameState(players: Player[]): GameState {
  const board = createBoardState();
  return {
    id: 'test-game',
    boardType: 'square8',
    board,
    players,
    currentPhase: 'ring_placement',
    currentPlayer: 1,
    moveHistory: [],
    history: [],
    timeControl: { type: 'rapid', initialTime: 600, increment: 0 },
    spectators: [],
    gameStatus: 'finished',
    createdAt: new Date(),
    lastMoveAt: new Date(),
    isRated: false,
    maxPlayers: players.length,
    totalRingsInPlay: 36,
    totalRingsEliminated: 23,
    victoryThreshold: 18, // RR-CANON-R061: ringsPerPlayer
    territoryVictoryThreshold: 33,
  } as GameState;
}

function createGameResult(
  reason: GameResult['reason'],
  winner: number | undefined = 1
): GameResult {
  return {
    winner,
    reason,
    finalScore: {
      ringsEliminated: { 1: 15, 2: 8 },
      territorySpaces: { 1: 25, 2: 10 },
      ringsRemaining: { 1: 3, 2: 10 },
    },
  };
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

describe('GameEnd UX regression – VictoryModal & banners', () => {
  it('renders canonical Ring Elimination phrasing in VictoryModal', () => {
    const players = createPlayers();
    const gameState = createGameState(players);
    const gameResult = createGameResult('ring_elimination', 1);

    const viewModel = toVictoryViewModel(gameResult, players, gameState, {
      currentUserId: players[0].id,
      isDismissed: false,
    });

    expect(viewModel).not.toBeNull();
    if (!viewModel) return;

    render(
      <VictoryModal
        isOpen={true}
        viewModel={viewModel}
        onClose={() => {}}
        onReturnToLobby={() => {}}
      />
    );

    const dialog = screen.getByRole('dialog');

    // For Ring Elimination victories we assert the core global-threshold phrasing
    // without requiring every elimination-related phrase used in other surfaces.
    const [primarySnippet] = RulesUxPhrases.victory.ringElimination;
    expect(dialog).toHaveTextContent(new RegExp(primarySnippet, 'i'));
  });

  it('distinguishes Territory Control victories from Ring Elimination in VictoryModal copy', () => {
    const players = createPlayers();
    const gameState = createGameState(players);
    const territoryResult = createGameResult('territory_control', 1);

    const viewModel = toVictoryViewModel(territoryResult, players, gameState, {
      currentUserId: players[0].id,
      isDismissed: false,
    });

    expect(viewModel).not.toBeNull();
    if (!viewModel) return;

    render(
      <VictoryModal
        isOpen={true}
        viewModel={viewModel}
        onClose={() => {}}
        onReturnToLobby={() => {}}
      />
    );

    const dialog = screen.getByRole('dialog');
    const text = dialog.textContent || '';

    // Territory victories must clearly mention territory-based winning,
    // not re-use Ring Elimination wording.
    expect(text).toMatch(/territory/i);
    expect(text).not.toMatch(/eliminating more than half of all rings in play/i);
  });

  it('uses canonical structural-stalemate phrasing in both VictoryModal and game-over banner', () => {
    const players = createPlayers();
    const gameState = createGameState(players);
    const stalemateResult = createGameResult('game_completed', undefined);

    const viewModel = toVictoryViewModel(stalemateResult, players, gameState, {
      currentUserId: players[0].id,
      isDismissed: false,
    });

    expect(viewModel).not.toBeNull();
    if (!viewModel) return;

    render(
      <VictoryModal
        isOpen={true}
        viewModel={viewModel}
        onClose={() => {}}
        onReturnToLobby={() => {}}
      />
    );

    const dialog = screen.getByRole('dialog');
    const bannerText = getGameOverBannerText('game_completed');

    // The VictoryModal description uses the full canonical phrasing including
    // "structural stalemate" and the detailed tiebreak ladder.
    RulesUxPhrases.victory.structuralStalemate.forEach((snippet) => {
      const pattern = new RegExp(snippet, 'i');
      expect(dialog).toHaveTextContent(pattern);
    });

    // The game-over banner uses a concise summary without "structural stalemate"
    // phrasing; verify it describes the score-based resolution instead.
    expect(bannerText).toMatch(/no moves remain/i);
    expect(bannerText).toMatch(/territory/i);
    expect(bannerText).toMatch(/rings eliminated/i);

    const modalText = dialog.textContent || '';

    // Structural-stalemate explanation should enumerate the four-step tiebreak ladder:
    // 1) Territory spaces, 2) Eliminated rings (including rings in hand),
    // 3) Markers, 4) Who made the last real action.
    expect(modalText).toMatch(/Territory spaces/i);
    expect(modalText).toMatch(/Eliminated rings \(including rings in hand\)/i);
    expect(modalText).toMatch(/Markers/i);
    expect(modalText).toMatch(/last real action/i);
  });

  it('explains Last Player Standing using real-move vs forced-elimination semantics', () => {
    const players = createPlayers();
    const gameState = createGameState(players);
    const lpsResult = createGameResult('last_player_standing', 1);

    const viewModel = toVictoryViewModel(lpsResult, players, gameState, {
      currentUserId: players[0].id,
      isDismissed: false,
    });

    expect(viewModel).not.toBeNull();
    if (!viewModel) return;

    render(
      <VictoryModal
        isOpen={true}
        viewModel={viewModel}
        onClose={() => {}}
        onReturnToLobby={() => {}}
      />
    );

    const dialog = screen.getByRole('dialog');
    const text = dialog.textContent || '';

    // LPS UX copy explains the concept without full canonical detail (per gameViewModels.ts)
    expect(text).toMatch(
      /only player able to make real moves \(placements, movements, or captures\) for a full round of turns/i
    );
    expect(text).toMatch(
      /forced eliminations?, which do not count as real moves for Last Player Standing even though they still remove caps and permanently eliminate rings/i
    );
  });
});
