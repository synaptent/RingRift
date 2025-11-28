/**
 * Game View Model Adapters Unit Tests
 *
 * Comprehensive tests for the view model transformation functions
 * that convert game state into presentation-ready data structures.
 */

import {
  PLAYER_COLORS,
  getPlayerColors,
  toHUDViewModel,
  toEventLogViewModel,
  toBoardViewModel,
  toVictoryViewModel,
  type HUDViewModel,
  type EventLogViewModel,
  type BoardViewModel,
  type VictoryViewModel,
  type PhaseViewModel,
  type PlayerViewModel,
  type CellViewModel,
  type ToHUDViewModelOptions,
  type ToEventLogViewModelOptions,
  type ToBoardViewModelOptions,
  type ToVictoryViewModelOptions,
} from '../../../src/client/adapters/gameViewModels';
import type {
  GameState,
  BoardState,
  Player,
  GameResult,
  GameHistoryEntry,
  Move,
  RingStack,
  Position,
  GamePhase,
  LineInfo,
} from '../../../src/shared/types/game';
import {
  createTestBoard,
  createTestPlayer,
  createTestGameState,
  addStack,
  addMarker,
  addCollapsedSpace,
  pos,
  posStr,
} from '../../utils/fixtures';

// ═══════════════════════════════════════════════════════════════════════════
// Helper Functions
// ═══════════════════════════════════════════════════════════════════════════

function createDefaultHUDOptions(
  overrides: Partial<ToHUDViewModelOptions> = {}
): ToHUDViewModelOptions {
  return {
    connectionStatus: 'connected',
    lastHeartbeatAt: Date.now(),
    isSpectator: false,
    ...overrides,
  };
}

function createGameResult(winner: number | undefined, reason: GameResult['reason']): GameResult {
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

function createTestHistoryEntry(
  moveNumber: number,
  player: number,
  type: Move['type'] = 'place_ring'
): GameHistoryEntry {
  return {
    moveNumber,
    actor: player,
    action: {
      id: `move-${moveNumber}`,
      type,
      player,
      to: { x: 0, y: 0 },
      timestamp: new Date(),
      thinkTime: 0,
      moveNumber,
    },
    phaseBefore: 'ring_placement',
    phaseAfter: 'ring_placement',
    statusBefore: 'active',
    statusAfter: 'active',
    progressBefore: { markers: 0, collapsed: 0, eliminated: 0, S: 0 },
    progressAfter: { markers: 1, collapsed: 0, eliminated: 0, S: 1 },
  };
}

function createAIPlayer(playerNumber: number, difficulty: number = 5): Player {
  return {
    ...createTestPlayer(playerNumber),
    type: 'ai',
    aiDifficulty: difficulty,
    aiProfile: {
      difficulty,
      aiType: 'heuristic',
      mode: 'service',
    },
  };
}

// ═══════════════════════════════════════════════════════════════════════════
// PLAYER_COLORS Constant Tests
// ═══════════════════════════════════════════════════════════════════════════

describe('PLAYER_COLORS constant', () => {
  it('should define colors for players 1-4', () => {
    expect(PLAYER_COLORS[1]).toBeDefined();
    expect(PLAYER_COLORS[2]).toBeDefined();
    expect(PLAYER_COLORS[3]).toBeDefined();
    expect(PLAYER_COLORS[4]).toBeDefined();
  });

  it('should have all required color properties for each player', () => {
    const requiredProps = ['ring', 'ringBorder', 'marker', 'territory', 'card', 'hex'];

    for (let i = 1; i <= 4; i++) {
      const playerKey = i as keyof typeof PLAYER_COLORS;
      const colors = PLAYER_COLORS[playerKey];

      requiredProps.forEach((prop) => {
        expect(colors).toHaveProperty(prop);
        expect(typeof colors[prop as keyof typeof colors]).toBe('string');
      });
    }
  });

  it('should have valid hex color codes', () => {
    for (let i = 1; i <= 4; i++) {
      const playerKey = i as keyof typeof PLAYER_COLORS;
      const hexColor = PLAYER_COLORS[playerKey].hex;
      expect(hexColor).toMatch(/^#[0-9a-fA-F]{6}$/);
    }
  });

  it('should use Tailwind CSS class names', () => {
    expect(PLAYER_COLORS[1].ring).toMatch(/^bg-/);
    expect(PLAYER_COLORS[1].ringBorder).toMatch(/^border-/);
    expect(PLAYER_COLORS[1].marker).toMatch(/^border-/);
    expect(PLAYER_COLORS[1].territory).toMatch(/^bg-/);
    expect(PLAYER_COLORS[1].card).toMatch(/^bg-/);
  });
});

// ═══════════════════════════════════════════════════════════════════════════
// getPlayerColors Function Tests
// ═══════════════════════════════════════════════════════════════════════════

describe('getPlayerColors', () => {
  describe('valid player numbers', () => {
    it('should return colors for player 1', () => {
      const colors = getPlayerColors(1);
      expect(colors.ring).toBe(PLAYER_COLORS[1].ring);
      expect(colors.hex).toBe(PLAYER_COLORS[1].hex);
    });

    it('should return colors for player 2', () => {
      const colors = getPlayerColors(2);
      expect(colors.ring).toBe(PLAYER_COLORS[2].ring);
      expect(colors.hex).toBe(PLAYER_COLORS[2].hex);
    });

    it('should return colors for player 3', () => {
      const colors = getPlayerColors(3);
      expect(colors.ring).toBe(PLAYER_COLORS[3].ring);
      expect(colors.hex).toBe(PLAYER_COLORS[3].hex);
    });

    it('should return colors for player 4', () => {
      const colors = getPlayerColors(4);
      expect(colors.ring).toBe(PLAYER_COLORS[4].ring);
      expect(colors.hex).toBe(PLAYER_COLORS[4].hex);
    });
  });

  describe('edge cases', () => {
    it('should return default colors for undefined player', () => {
      const colors = getPlayerColors(undefined);
      expect(colors.ring).toBe('bg-slate-300');
      expect(colors.hex).toBe('#64748b');
    });

    it('should return default colors for player 0', () => {
      const colors = getPlayerColors(0);
      expect(colors.ring).toBe('bg-slate-300');
    });

    it('should return default colors for invalid player number (5)', () => {
      const colors = getPlayerColors(5);
      expect(colors.ring).toBe('bg-slate-300');
    });

    it('should return default colors for negative player number', () => {
      const colors = getPlayerColors(-1);
      expect(colors.ring).toBe('bg-slate-300');
    });
  });
});

// ═══════════════════════════════════════════════════════════════════════════
// toHUDViewModel Function Tests
// ═══════════════════════════════════════════════════════════════════════════

describe('toHUDViewModel', () => {
  describe('basic transformation', () => {
    it('should transform initial game state into HUD view model', () => {
      const gameState = createTestGameState();
      const options = createDefaultHUDOptions();

      const hud = toHUDViewModel(gameState, options);

      expect(hud).toBeDefined();
      expect(hud.phase).toBeDefined();
      expect(hud.players).toHaveLength(2);
      expect(hud.connectionStatus).toBe('connected');
      expect(hud.isSpectator).toBe(false);
    });

    it('should correctly map phase information', () => {
      const gameState = createTestGameState({ currentPhase: 'movement' });
      const options = createDefaultHUDOptions();

      const hud = toHUDViewModel(gameState, options);

      expect(hud.phase.phaseKey).toBe('movement');
      expect(hud.phase.label).toBe('Movement Phase');
      expect(hud.phase.icon).toBe('⚡');
      expect(hud.phase.colorClass).toMatch(/^bg-/);
    });

    it('should include instruction when provided', () => {
      const gameState = createTestGameState();
      const options = createDefaultHUDOptions({ instruction: 'Place your rings' });

      const hud = toHUDViewModel(gameState, options);

      expect(hud.instruction).toBe('Place your rings');
    });
  });

  describe('phase view model mapping', () => {
    const phaseTests: Array<{
      phase: GamePhase;
      expectedLabel: string;
      expectedIcon: string;
    }> = [
      { phase: 'ring_placement', expectedLabel: 'Ring Placement', expectedIcon: '🎯' },
      { phase: 'movement', expectedLabel: 'Movement Phase', expectedIcon: '⚡' },
      { phase: 'capture', expectedLabel: 'Capture Phase', expectedIcon: '⚔️' },
      { phase: 'chain_capture', expectedLabel: 'Chain Capture', expectedIcon: '🔗' },
      { phase: 'line_processing', expectedLabel: 'Line Reward', expectedIcon: '📏' },
      { phase: 'territory_processing', expectedLabel: 'Territory Claim', expectedIcon: '🏰' },
    ];

    phaseTests.forEach(({ phase, expectedLabel, expectedIcon }) => {
      it(`should correctly map ${phase} phase`, () => {
        const gameState = createTestGameState({ currentPhase: phase });
        const options = createDefaultHUDOptions();

        const hud = toHUDViewModel(gameState, options);

        expect(hud.phase.phaseKey).toBe(phase);
        expect(hud.phase.label).toBe(expectedLabel);
        expect(hud.phase.icon).toBe(expectedIcon);
      });
    });
  });

  describe('player view model mapping', () => {
    it('should correctly identify current player', () => {
      const gameState = createTestGameState({ currentPlayer: 1 });
      const options = createDefaultHUDOptions();

      const hud = toHUDViewModel(gameState, options);

      expect(hud.players[0].isCurrentPlayer).toBe(true);
      expect(hud.players[1].isCurrentPlayer).toBe(false);
    });

    it('should correctly identify user player', () => {
      const player1 = createTestPlayer(1, { id: 'current-user-id' });
      const player2 = createTestPlayer(2);
      const gameState = createTestGameState({ players: [player1, player2] });
      const options = createDefaultHUDOptions({ currentUserId: 'current-user-id' });

      const hud = toHUDViewModel(gameState, options);

      expect(hud.players[0].isUserPlayer).toBe(true);
      expect(hud.players[1].isUserPlayer).toBe(false);
    });

    it('should calculate ring statistics correctly', () => {
      const board = createTestBoard('square8');
      addStack(board, pos(0, 0), 1, 3); // 3 rings for player 1
      addStack(board, pos(1, 1), 2, 2); // 2 rings for player 2

      const player1 = createTestPlayer(1, { ringsInHand: 10, eliminatedRings: 5 });
      const player2 = createTestPlayer(2, { ringsInHand: 12, eliminatedRings: 4 });

      const gameState = createTestGameState({
        board,
        players: [player1, player2],
      });
      const options = createDefaultHUDOptions();

      const hud = toHUDViewModel(gameState, options);

      expect(hud.players[0].ringStats.onBoard).toBe(3);
      expect(hud.players[0].ringStats.inHand).toBe(10);
      expect(hud.players[0].ringStats.eliminated).toBe(5);

      expect(hud.players[1].ringStats.onBoard).toBe(2);
      expect(hud.players[1].ringStats.inHand).toBe(12);
      expect(hud.players[1].ringStats.eliminated).toBe(4);
    });

    it('should include territory spaces', () => {
      const player1 = createTestPlayer(1, { territorySpaces: 15 });
      const player2 = createTestPlayer(2, { territorySpaces: 8 });
      const gameState = createTestGameState({ players: [player1, player2] });
      const options = createDefaultHUDOptions();

      const hud = toHUDViewModel(gameState, options);

      expect(hud.players[0].territorySpaces).toBe(15);
      expect(hud.players[1].territorySpaces).toBe(8);
    });
  });

  describe('AI player handling', () => {
    it('should identify AI player correctly', () => {
      const player1 = createTestPlayer(1);
      const player2 = createAIPlayer(2, 5);
      const gameState = createTestGameState({ players: [player1, player2] });
      const options = createDefaultHUDOptions();

      const hud = toHUDViewModel(gameState, options);

      expect(hud.players[0].aiInfo.isAI).toBe(false);
      expect(hud.players[1].aiInfo.isAI).toBe(true);
    });

    it('should include AI difficulty information', () => {
      const player1 = createTestPlayer(1);
      const player2 = createAIPlayer(2, 7);
      const gameState = createTestGameState({ players: [player1, player2] });
      const options = createDefaultHUDOptions();

      const hud = toHUDViewModel(gameState, options);

      expect(hud.players[1].aiInfo.difficulty).toBe(7);
      expect(hud.players[1].aiInfo.difficultyLabel).toBeDefined();
    });

    it('should correctly label different AI difficulty levels', () => {
      const difficulties = [
        { level: 1, expectedPattern: /Beginner|Random/i },
        { level: 2, expectedPattern: /Easy|Heuristic/i },
        { level: 5, expectedPattern: /Advanced|Minimax/i },
        { level: 7, expectedPattern: /Expert|MCTS/i },
        { level: 9, expectedPattern: /Grandmaster|Descent/i },
      ];

      difficulties.forEach(({ level, expectedPattern }) => {
        const aiPlayer = createAIPlayer(2, level);
        const gameState = createTestGameState({ players: [createTestPlayer(1), aiPlayer] });
        const options = createDefaultHUDOptions();

        const hud = toHUDViewModel(gameState, options);

        expect(hud.players[1].aiInfo.difficultyLabel).toMatch(expectedPattern);
      });
    });
  });

  describe('connection status', () => {
    it('should mark connection as stale when heartbeat is old', () => {
      const gameState = createTestGameState();
      const options = createDefaultHUDOptions({
        connectionStatus: 'connected',
        lastHeartbeatAt: Date.now() - 10000, // 10 seconds ago
      });

      const hud = toHUDViewModel(gameState, options);

      expect(hud.isConnectionStale).toBe(true);
    });

    it('should not mark connection as stale when heartbeat is recent', () => {
      const gameState = createTestGameState();
      const options = createDefaultHUDOptions({
        connectionStatus: 'connected',
        lastHeartbeatAt: Date.now() - 1000, // 1 second ago
      });

      const hud = toHUDViewModel(gameState, options);

      expect(hud.isConnectionStale).toBe(false);
    });

    it('should not mark as stale when disconnected', () => {
      const gameState = createTestGameState();
      const options = createDefaultHUDOptions({
        connectionStatus: 'disconnected',
        lastHeartbeatAt: Date.now() - 10000,
      });

      const hud = toHUDViewModel(gameState, options);

      expect(hud.isConnectionStale).toBe(false);
    });
  });

  describe('sub-phase details', () => {
    it('should include line count during line_processing phase', () => {
      const board = createTestBoard('square8');
      board.formedLines = [
        { positions: [], player: 1, length: 3, direction: { x: 1, y: 0 } },
        { positions: [], player: 1, length: 3, direction: { x: 0, y: 1 } },
      ];

      const gameState = createTestGameState({
        board,
        currentPhase: 'line_processing',
      });
      const options = createDefaultHUDOptions();

      const hud = toHUDViewModel(gameState, options);

      expect(hud.subPhaseDetail).toBe('Processing 2 lines');
    });

    it('should include singular line message', () => {
      const board = createTestBoard('square8');
      board.formedLines = [{ positions: [], player: 1, length: 3, direction: { x: 1, y: 0 } }];

      const gameState = createTestGameState({
        board,
        currentPhase: 'line_processing',
      });
      const options = createDefaultHUDOptions();

      const hud = toHUDViewModel(gameState, options);

      expect(hud.subPhaseDetail).toBe('Processing 1 line');
    });

    it('should include territory processing message', () => {
      const gameState = createTestGameState({
        currentPhase: 'territory_processing',
      });
      const options = createDefaultHUDOptions();

      const hud = toHUDViewModel(gameState, options);

      expect(hud.subPhaseDetail).toBe('Processing disconnected regions');
    });
  });

  describe('spectator handling', () => {
    it('should include spectator status', () => {
      const gameState = createTestGameState({ spectators: ['user1', 'user2', 'user3'] });
      const options = createDefaultHUDOptions({ isSpectator: true });

      const hud = toHUDViewModel(gameState, options);

      expect(hud.isSpectator).toBe(true);
      expect(hud.spectatorCount).toBe(3);
    });
  });

  describe('turn and move numbers', () => {
    it('should include turn and move numbers', () => {
      const gameState = createTestGameState();
      gameState.moveHistory = [
        { id: '1', player: 1 } as Move,
        { id: '2', player: 2 } as Move,
        { id: '3', player: 1 } as Move,
      ];
      gameState.history = [
        createTestHistoryEntry(1, 1),
        createTestHistoryEntry(2, 2),
        createTestHistoryEntry(3, 1),
      ];

      const options = createDefaultHUDOptions();

      const hud = toHUDViewModel(gameState, options);

      expect(hud.turnNumber).toBe(3);
      expect(hud.moveNumber).toBe(3);
    });
  });
});

// ═══════════════════════════════════════════════════════════════════════════
// toEventLogViewModel Function Tests
// ═══════════════════════════════════════════════════════════════════════════

describe('toEventLogViewModel', () => {
  describe('basic transformation', () => {
    it('should return empty view model for empty history', () => {
      const result = toEventLogViewModel([], [], null);

      expect(result.entries).toHaveLength(0);
      expect(result.hasContent).toBe(false);
      expect(result.victoryMessage).toBeUndefined();
    });

    it('should transform history entries into event log items', () => {
      const history: GameHistoryEntry[] = [
        createTestHistoryEntry(1, 1, 'place_ring'),
        createTestHistoryEntry(2, 2, 'place_ring'),
      ];

      const result = toEventLogViewModel(history, [], null);

      expect(result.entries).toHaveLength(2);
      expect(result.hasContent).toBe(true);
    });

    it('should include system events', () => {
      const systemEvents = ['Game started', 'Player 1 connected'];

      const result = toEventLogViewModel([], systemEvents, null);

      expect(result.entries).toHaveLength(2);
      expect(result.entries[0].type).toBe('system');
      expect(result.entries[1].type).toBe('system');
    });
  });

  describe('history entry formatting', () => {
    it('should format place_ring action', () => {
      const history: GameHistoryEntry[] = [
        {
          ...createTestHistoryEntry(1, 1, 'place_ring'),
          action: {
            ...createTestHistoryEntry(1, 1, 'place_ring').action,
            to: { x: 3, y: 4 },
            placementCount: 2,
          },
        },
      ];

      const result = toEventLogViewModel(history, [], null);

      expect(result.entries[0].text).toContain('P1');
      expect(result.entries[0].text).toContain('placed');
      expect(result.entries[0].text).toContain('2 rings');
      expect(result.entries[0].text).toContain('(3, 4)');
    });

    it('should format move_stack action', () => {
      const history: GameHistoryEntry[] = [
        {
          ...createTestHistoryEntry(1, 1, 'move_stack'),
          action: {
            ...createTestHistoryEntry(1, 1, 'move_stack').action,
            type: 'move_stack',
            from: { x: 0, y: 0 },
            to: { x: 3, y: 3 },
          },
        },
      ];

      const result = toEventLogViewModel(history, [], null);

      expect(result.entries[0].text).toContain('moved from');
      expect(result.entries[0].text).toContain('(0, 0)');
      expect(result.entries[0].text).toContain('(3, 3)');
    });

    it('should format overtaking_capture action', () => {
      const history: GameHistoryEntry[] = [
        {
          ...createTestHistoryEntry(1, 1, 'overtaking_capture'),
          action: {
            ...createTestHistoryEntry(1, 1, 'overtaking_capture').action,
            type: 'overtaking_capture',
            from: { x: 0, y: 0 },
            captureTarget: { x: 1, y: 1 },
            to: { x: 2, y: 2 },
            overtakenRings: [2, 2],
          },
        },
      ];

      const result = toEventLogViewModel(history, [], null);

      expect(result.entries[0].text).toContain('capture');
      expect(result.entries[0].text).toContain('x2');
    });

    it('should format skip_placement action', () => {
      const history: GameHistoryEntry[] = [
        {
          ...createTestHistoryEntry(1, 1, 'skip_placement'),
          action: {
            ...createTestHistoryEntry(1, 1, 'skip_placement').action,
            type: 'skip_placement',
          },
        },
      ];

      const result = toEventLogViewModel(history, [], null);

      expect(result.entries[0].text).toContain('skipped placement');
    });
  });

  describe('victory message', () => {
    it('should include victory message when game won', () => {
      const victoryState: GameResult = {
        winner: 1,
        reason: 'ring_elimination',
        finalScore: {
          ringsEliminated: { 1: 20, 2: 10 },
          territorySpaces: { 1: 15, 2: 10 },
          ringsRemaining: { 1: 5, 2: 0 },
        },
      };

      const result = toEventLogViewModel([], [], victoryState);

      expect(result.victoryMessage).toBeDefined();
      expect(result.victoryMessage).toContain('P1');
      expect(result.victoryMessage).toContain('wins');
    });

    it('should include draw message', () => {
      const victoryState: GameResult = {
        winner: undefined,
        reason: 'draw',
        finalScore: {
          ringsEliminated: { 1: 10, 2: 10 },
          territorySpaces: { 1: 15, 2: 15 },
          ringsRemaining: { 1: 5, 2: 5 },
        },
      };

      const result = toEventLogViewModel([], [], victoryState);

      expect(result.victoryMessage).toBeDefined();
      expect(result.victoryMessage?.toLowerCase()).toContain('draw');
    });

    it('should format different victory reasons', () => {
      const reasons: GameResult['reason'][] = [
        'ring_elimination',
        'territory_control',
        'last_player_standing',
        'timeout',
        'resignation',
      ];

      reasons.forEach((reason) => {
        const victoryState: GameResult = {
          winner: 1,
          reason,
          finalScore: {
            ringsEliminated: { 1: 20 },
            territorySpaces: { 1: 20 },
            ringsRemaining: { 1: 5 },
          },
        };

        const result = toEventLogViewModel([], [], victoryState);

        expect(result.victoryMessage).toBeDefined();
      });
    });
  });

  describe('maxEntries option', () => {
    it('should respect maxEntries option', () => {
      const history: GameHistoryEntry[] = [];
      for (let i = 1; i <= 50; i++) {
        history.push(createTestHistoryEntry(i, (i % 2) + 1));
      }

      const result = toEventLogViewModel(history, [], null, { maxEntries: 10 });

      expect(result.entries.length).toBeLessThanOrEqual(10);
    });

    it('should show most recent entries first', () => {
      const history: GameHistoryEntry[] = [
        createTestHistoryEntry(1, 1),
        createTestHistoryEntry(2, 2),
        createTestHistoryEntry(3, 1),
      ];

      const result = toEventLogViewModel(history, [], null);

      // Most recent should be first
      expect(result.entries[0].moveNumber).toBe(3);
      expect(result.entries[1].moveNumber).toBe(2);
      expect(result.entries[2].moveNumber).toBe(1);
    });
  });

  describe('entry types', () => {
    it('should correctly tag move entries', () => {
      const history: GameHistoryEntry[] = [createTestHistoryEntry(1, 1)];

      const result = toEventLogViewModel(history, [], null);

      expect(result.entries[0].type).toBe('move');
    });

    it('should correctly tag system entries', () => {
      const result = toEventLogViewModel([], ['Game started'], null);

      expect(result.entries[0].type).toBe('system');
    });

    it('should correctly tag victory entries', () => {
      const victoryState = createGameResult(1, 'ring_elimination');

      const result = toEventLogViewModel([], [], victoryState);

      expect(result.entries[0].type).toBe('victory');
    });
  });
});

// ═══════════════════════════════════════════════════════════════════════════
// toBoardViewModel Function Tests
// ═══════════════════════════════════════════════════════════════════════════

describe('toBoardViewModel', () => {
  describe('square board transformation', () => {
    it('should transform empty square8 board', () => {
      const board = createTestBoard('square8');

      const result = toBoardViewModel(board);

      expect(result.boardType).toBe('square8');
      expect(result.size).toBe(8);
      expect(result.cells).toHaveLength(64);
      expect(result.rows).toHaveLength(8);
      expect(result.rows![0]).toHaveLength(8);
    });

    it('should transform empty square19 board', () => {
      const board = createTestBoard('square19');

      const result = toBoardViewModel(board);

      expect(result.boardType).toBe('square19');
      expect(result.size).toBe(19);
      expect(result.cells).toHaveLength(361);
      expect(result.rows).toHaveLength(19);
    });

    it('should correctly organize cells into rows', () => {
      const board = createTestBoard('square8');

      const result = toBoardViewModel(board);

      // Check that row 0 contains y=0 cells
      result.rows![0].forEach((cell) => {
        expect(cell.position.y).toBe(0);
      });

      // Check that row 7 contains y=7 cells
      result.rows![7].forEach((cell) => {
        expect(cell.position.y).toBe(7);
      });
    });
  });

  describe('hexagonal board transformation', () => {
    it('should transform hexagonal board with stacks', () => {
      const board = createTestBoard('hexagonal');
      addStack(board, pos(0, 0, 0), 1, 2);
      addStack(board, pos(1, -1, 0), 2, 1);

      const result = toBoardViewModel(board);

      expect(result.boardType).toBe('hexagonal');
      expect(result.cells.length).toBeGreaterThan(0);
      expect(result.rows).toBeUndefined(); // Hex boards don't use row organization
    });

    it('should include z coordinate in position key for hex cells', () => {
      const board = createTestBoard('hexagonal');
      addStack(board, pos(0, 0, 0), 1, 1);

      const result = toBoardViewModel(board);

      const centerCell = result.cells.find((c) => c.position.x === 0 && c.position.y === 0);
      expect(centerCell).toBeDefined();
      expect(centerCell!.positionKey).toBe('0,0,0');
    });
  });

  describe('cell view model properties', () => {
    it('should include stack view model when cell has stack', () => {
      const board = createTestBoard('square8');
      addStack(board, pos(2, 3), 1, 3);

      const result = toBoardViewModel(board);

      const cell = result.cells.find((c) => c.position.x === 2 && c.position.y === 3);
      expect(cell?.stack).toBeDefined();
      expect(cell?.stack?.stackHeight).toBe(3);
      expect(cell?.stack?.controllingPlayer).toBe(1);
    });

    it('should include marker view model when cell has marker', () => {
      const board = createTestBoard('square8');
      addMarker(board, pos(4, 5), 2);

      const result = toBoardViewModel(board);

      const cell = result.cells.find((c) => c.position.x === 4 && c.position.y === 5);
      expect(cell?.marker).toBeDefined();
      expect(cell?.marker?.playerNumber).toBe(2);
    });

    it('should include collapsed space view model when cell is collapsed', () => {
      const board = createTestBoard('square8');
      addCollapsedSpace(board, pos(1, 1), 1);

      const result = toBoardViewModel(board);

      const cell = result.cells.find((c) => c.position.x === 1 && c.position.y === 1);
      expect(cell?.collapsedSpace).toBeDefined();
      expect(cell?.collapsedSpace?.ownerPlayerNumber).toBe(1);
    });

    it('should correctly identify dark squares', () => {
      const board = createTestBoard('square8');

      const result = toBoardViewModel(board);

      const cell00 = result.cells.find((c) => c.position.x === 0 && c.position.y === 0);
      const cell01 = result.cells.find((c) => c.position.x === 0 && c.position.y === 1);

      // (0+0) % 2 = 0 → dark
      expect(cell00?.isDarkSquare).toBe(true);
      // (0+1) % 2 = 1 → light
      expect(cell01?.isDarkSquare).toBe(false);
    });
  });

  describe('selection and valid targets', () => {
    it('should mark selected cell', () => {
      const board = createTestBoard('square8');
      const options: ToBoardViewModelOptions = { selectedPosition: pos(3, 3) };

      const result = toBoardViewModel(board, options);

      const selectedCell = result.cells.find((c) => c.position.x === 3 && c.position.y === 3);
      expect(selectedCell?.isSelected).toBe(true);

      const otherCell = result.cells.find((c) => c.position.x === 0 && c.position.y === 0);
      expect(otherCell?.isSelected).toBe(false);
    });

    it('should mark valid target cells', () => {
      const board = createTestBoard('square8');
      const options: ToBoardViewModelOptions = {
        validTargets: [pos(1, 1), pos(2, 2), pos(3, 3)],
      };

      const result = toBoardViewModel(board, options);

      const targetCell = result.cells.find((c) => c.position.x === 2 && c.position.y === 2);
      expect(targetCell?.isValidTarget).toBe(true);

      const nonTargetCell = result.cells.find((c) => c.position.x === 5 && c.position.y === 5);
      expect(nonTargetCell?.isValidTarget).toBe(false);
    });
  });

  describe('stack view model details', () => {
    it('should correctly map ring view models', () => {
      const board = createTestBoard('square8');
      const stackKey = posStr(2, 2);
      board.stacks.set(stackKey, {
        position: pos(2, 2),
        rings: [1, 2, 1, 2], // Mixed stack
        stackHeight: 4,
        capHeight: 1, // Only top ring is cap
        controllingPlayer: 1,
      });

      const result = toBoardViewModel(board);

      const cell = result.cells.find((c) => c.position.x === 2 && c.position.y === 2);
      expect(cell?.stack?.rings).toHaveLength(4);
      expect(cell?.stack?.rings[0].isTop).toBe(true);
      expect(cell?.stack?.rings[0].playerNumber).toBe(1);
    });

    it('should correctly identify rings in cap', () => {
      const board = createTestBoard('square8');
      const stackKey = posStr(2, 2);
      board.stacks.set(stackKey, {
        position: pos(2, 2),
        rings: [1, 1, 2, 2], // Player 1 has 2 rings on top (cap)
        stackHeight: 4,
        capHeight: 2,
        controllingPlayer: 1,
      });

      const result = toBoardViewModel(board);

      const cell = result.cells.find((c) => c.position.x === 2 && c.position.y === 2);
      expect(cell?.stack?.rings[0].isInCap).toBe(true);
      expect(cell?.stack?.rings[1].isInCap).toBe(true);
      expect(cell?.stack?.rings[2].isInCap).toBe(false);
    });
  });

  describe('color assignments', () => {
    it('should assign correct colors to stacks', () => {
      const board = createTestBoard('square8');
      addStack(board, pos(0, 0), 1, 1);
      addStack(board, pos(1, 1), 2, 1);

      const result = toBoardViewModel(board);

      const cell1 = result.cells.find((c) => c.position.x === 0 && c.position.y === 0);
      const cell2 = result.cells.find((c) => c.position.x === 1 && c.position.y === 1);

      expect(cell1?.stack?.rings[0].colorClass).toBe(PLAYER_COLORS[1].ring);
      expect(cell2?.stack?.rings[0].colorClass).toBe(PLAYER_COLORS[2].ring);
    });

    it('should assign correct colors to markers', () => {
      const board = createTestBoard('square8');
      addMarker(board, pos(3, 3), 1);
      addMarker(board, pos(4, 4), 2);

      const result = toBoardViewModel(board);

      const cell1 = result.cells.find((c) => c.position.x === 3 && c.position.y === 3);
      const cell2 = result.cells.find((c) => c.position.x === 4 && c.position.y === 4);

      expect(cell1?.marker?.colorClass).toBe(PLAYER_COLORS[1].marker);
      expect(cell2?.marker?.colorClass).toBe(PLAYER_COLORS[2].marker);
    });

    it('should assign correct colors to collapsed spaces', () => {
      const board = createTestBoard('square8');
      addCollapsedSpace(board, pos(5, 5), 1);
      addCollapsedSpace(board, pos(6, 6), 2);

      const result = toBoardViewModel(board);

      const cell1 = result.cells.find((c) => c.position.x === 5 && c.position.y === 5);
      const cell2 = result.cells.find((c) => c.position.x === 6 && c.position.y === 6);

      expect(cell1?.collapsedSpace?.territoryColorClass).toBe(PLAYER_COLORS[1].territory);
      expect(cell2?.collapsedSpace?.territoryColorClass).toBe(PLAYER_COLORS[2].territory);
    });
  });
});

// ═══════════════════════════════════════════════════════════════════════════
// toVictoryViewModel Function Tests
// ═══════════════════════════════════════════════════════════════════════════

describe('toVictoryViewModel', () => {
  describe('basic transformation', () => {
    it('should return null when no game result', () => {
      const result = toVictoryViewModel(null, [], undefined);
      expect(result).toBeNull();
    });

    it('should return null when dismissed', () => {
      const gameResult = createGameResult(1, 'ring_elimination');
      const players = [createTestPlayer(1), createTestPlayer(2)];

      const result = toVictoryViewModel(gameResult, players, undefined, { isDismissed: true });

      expect(result).toBeNull();
    });

    it('should return view model when game has result', () => {
      const gameResult = createGameResult(1, 'ring_elimination');
      const players = [createTestPlayer(1), createTestPlayer(2)];
      const gameState = createTestGameState({ players });

      const result = toVictoryViewModel(gameResult, players, gameState);

      expect(result).not.toBeNull();
      expect(result?.isVisible).toBe(true);
    });
  });

  describe('winner identification', () => {
    it('should identify winner correctly', () => {
      const gameResult = createGameResult(1, 'ring_elimination');
      const players = [createTestPlayer(1), createTestPlayer(2)];
      const gameState = createTestGameState({ players });

      const result = toVictoryViewModel(gameResult, players, gameState);

      expect(result?.winner).toBeDefined();
      expect(result?.winner?.playerNumber).toBe(1);
    });

    it('should identify user as winner', () => {
      const player1 = createTestPlayer(1, { id: 'current-user' });
      const player2 = createTestPlayer(2);
      const gameResult = createGameResult(1, 'ring_elimination');
      const gameState = createTestGameState({ players: [player1, player2] });

      const result = toVictoryViewModel(gameResult, [player1, player2], gameState, {
        currentUserId: 'current-user',
      });

      expect(result?.userWon).toBe(true);
      expect(result?.userLost).toBe(false);
    });

    it('should identify user as loser', () => {
      const player1 = createTestPlayer(1);
      const player2 = createTestPlayer(2, { id: 'current-user' });
      const gameResult = createGameResult(1, 'ring_elimination');
      const gameState = createTestGameState({ players: [player1, player2] });

      const result = toVictoryViewModel(gameResult, [player1, player2], gameState, {
        currentUserId: 'current-user',
      });

      expect(result?.userWon).toBe(false);
      expect(result?.userLost).toBe(true);
    });
  });

  describe('draw handling', () => {
    it('should identify draw correctly', () => {
      const gameResult = createGameResult(undefined, 'draw');
      const players = [createTestPlayer(1), createTestPlayer(2)];
      const gameState = createTestGameState({ players });

      const result = toVictoryViewModel(gameResult, players, gameState);

      expect(result?.isDraw).toBe(true);
      expect(result?.winner).toBeUndefined();
      expect(result?.userWon).toBe(false);
      expect(result?.userLost).toBe(false);
    });

    it('should include draw in title', () => {
      const gameResult = createGameResult(undefined, 'draw');
      const players = [createTestPlayer(1), createTestPlayer(2)];
      const gameState = createTestGameState({ players });

      const result = toVictoryViewModel(gameResult, players, gameState);

      expect(result?.title.toLowerCase()).toContain('draw');
    });
  });

  describe('victory reasons', () => {
    const victoryReasonTests: Array<{
      reason: GameResult['reason'];
      expectedTitlePattern: RegExp;
      expectedDescPattern: RegExp;
    }> = [
      {
        reason: 'ring_elimination',
        expectedTitlePattern: /Wins!/,
        expectedDescPattern: /eliminating.*(opponent|rings)/i,
      },
      {
        reason: 'territory_control',
        expectedTitlePattern: /Wins!/,
        expectedDescPattern: /controlling.*majority/i,
      },
      {
        reason: 'last_player_standing',
        expectedTitlePattern: /Last Player Standing/i,
        expectedDescPattern: /only player/i,
      },
      {
        reason: 'timeout',
        expectedTitlePattern: /Wins!/,
        expectedDescPattern: /timeout/i,
      },
      {
        reason: 'resignation',
        expectedTitlePattern: /Wins!/,
        expectedDescPattern: /resignation/i,
      },
      {
        reason: 'abandonment',
        expectedTitlePattern: /Abandoned/i,
        expectedDescPattern: /unresolved/i,
      },
    ];

    victoryReasonTests.forEach(({ reason, expectedTitlePattern, expectedDescPattern }) => {
      it(`should format ${reason} reason correctly`, () => {
        const gameResult = createGameResult(1, reason);
        const players = [createTestPlayer(1), createTestPlayer(2)];
        const gameState = createTestGameState({ players });

        const result = toVictoryViewModel(gameResult, players, gameState);

        expect(result?.title).toMatch(expectedTitlePattern);
        expect(result?.description).toMatch(expectedDescPattern);
      });
    });
  });

  describe('final stats', () => {
    it('should include final stats for all players', () => {
      const player1 = createTestPlayer(1, { eliminatedRings: 15, territorySpaces: 20 });
      const player2 = createTestPlayer(2, { eliminatedRings: 8, territorySpaces: 10 });
      const gameResult = createGameResult(1, 'ring_elimination');
      const gameState = createTestGameState({ players: [player1, player2] });

      const result = toVictoryViewModel(gameResult, [player1, player2], gameState);

      expect(result?.finalStats).toHaveLength(2);
    });

    it('should mark winner in final stats', () => {
      const gameResult = createGameResult(1, 'ring_elimination');
      const players = [createTestPlayer(1), createTestPlayer(2)];
      const gameState = createTestGameState({ players });

      const result = toVictoryViewModel(gameResult, players, gameState);

      const winnerStats = result?.finalStats.find((s) => s.player.playerNumber === 1);
      const loserStats = result?.finalStats.find((s) => s.player.playerNumber === 2);

      expect(winnerStats?.isWinner).toBe(true);
      expect(loserStats?.isWinner).toBe(false);
    });

    it('should sort winner first', () => {
      const gameResult = createGameResult(2, 'ring_elimination');
      const players = [createTestPlayer(1), createTestPlayer(2)];
      const gameState = createTestGameState({ players });

      const result = toVictoryViewModel(gameResult, players, gameState);

      expect(result?.finalStats[0].player.playerNumber).toBe(2);
    });

    it('should include ring counts from board', () => {
      const board = createTestBoard('square8');
      addStack(board, pos(0, 0), 1, 5);
      addStack(board, pos(1, 1), 2, 3);

      const gameResult = createGameResult(1, 'ring_elimination');
      const players = [createTestPlayer(1), createTestPlayer(2)];
      const gameState = createTestGameState({ board, players });

      const result = toVictoryViewModel(gameResult, players, gameState);

      const p1Stats = result?.finalStats.find((s) => s.player.playerNumber === 1);
      expect(p1Stats?.ringsOnBoard).toBe(5);
    });
  });

  describe('game summary', () => {
    it('should include game summary', () => {
      const gameResult = createGameResult(1, 'ring_elimination');
      const players = [createTestPlayer(1), createTestPlayer(2)];
      const gameState = createTestGameState({
        players,
        boardType: 'square8',
        isRated: true,
      });

      const result = toVictoryViewModel(gameResult, players, gameState);

      expect(result?.gameSummary.boardType).toBe('square8');
      expect(result?.gameSummary.playerCount).toBe(2);
      expect(result?.gameSummary.isRated).toBe(true);
    });

    it('should include total turns from history', () => {
      const gameResult = createGameResult(1, 'ring_elimination');
      const players = [createTestPlayer(1), createTestPlayer(2)];
      const gameState = createTestGameState({ players });
      gameState.history = [
        createTestHistoryEntry(1, 1),
        createTestHistoryEntry(2, 2),
        createTestHistoryEntry(3, 1),
      ];

      const result = toVictoryViewModel(gameResult, players, gameState);

      expect(result?.gameSummary.totalTurns).toBe(3);
    });
  });

  describe('title color', () => {
    it('should use green for user win', () => {
      const player1 = createTestPlayer(1, { id: 'current-user' });
      const player2 = createTestPlayer(2);
      const gameResult = createGameResult(1, 'ring_elimination');
      const gameState = createTestGameState({ players: [player1, player2] });

      const result = toVictoryViewModel(gameResult, [player1, player2], gameState, {
        currentUserId: 'current-user',
      });

      expect(result?.titleColorClass).toContain('green');
    });

    it('should use red for user loss', () => {
      const player1 = createTestPlayer(1);
      const player2 = createTestPlayer(2, { id: 'current-user' });
      const gameResult = createGameResult(1, 'ring_elimination');
      const gameState = createTestGameState({ players: [player1, player2] });

      const result = toVictoryViewModel(gameResult, [player1, player2], gameState, {
        currentUserId: 'current-user',
      });

      expect(result?.titleColorClass).toContain('red');
    });

    it('should use neutral color for spectator view (no currentUserId)', () => {
      const gameResult = createGameResult(1, 'ring_elimination');
      const players = [createTestPlayer(1), createTestPlayer(2)];
      const gameState = createTestGameState({ players });

      // When no currentUserId is provided, user is neither winner nor loser
      const result = toVictoryViewModel(gameResult, players, gameState);

      // With no currentUserId, both userWon and userLost are false
      expect(result?.userWon).toBe(false);
      expect(result?.userLost).toBe(false);
      expect(result?.titleColorClass).toContain('slate');
    });
  });

  describe('multi-player games', () => {
    it('should handle 3-player game', () => {
      const players = [createTestPlayer(1), createTestPlayer(2), createTestPlayer(3)];
      const gameResult: GameResult = {
        winner: 1,
        reason: 'ring_elimination',
        finalScore: {
          ringsEliminated: { 1: 28, 2: 8, 3: 12 },
          territorySpaces: { 1: 25, 2: 10, 3: 15 },
          ringsRemaining: { 1: 3, 2: 10, 3: 5 },
        },
      };
      const gameState = createTestGameState({ players, maxPlayers: 3 });

      const result = toVictoryViewModel(gameResult, players, gameState);

      expect(result?.finalStats).toHaveLength(3);
      expect(result?.gameSummary.playerCount).toBe(3);
    });

    it('should handle 4-player game', () => {
      const players = [
        createTestPlayer(1),
        createTestPlayer(2),
        createTestPlayer(3),
        createTestPlayer(4),
      ];
      const gameResult: GameResult = {
        winner: 1,
        reason: 'ring_elimination',
        finalScore: {
          ringsEliminated: { 1: 40, 2: 8, 3: 12, 4: 10 },
          territorySpaces: { 1: 25, 2: 10, 3: 15, 4: 20 },
          ringsRemaining: { 1: 3, 2: 10, 3: 5, 4: 2 },
        },
      };
      const gameState = createTestGameState({ players, maxPlayers: 4 });

      const result = toVictoryViewModel(gameResult, players, gameState);

      expect(result?.finalStats).toHaveLength(4);
      expect(result?.gameSummary.playerCount).toBe(4);
    });
  });
});

// ═══════════════════════════════════════════════════════════════════════════
// Edge Case Tests
// ═══════════════════════════════════════════════════════════════════════════

describe('Edge Cases', () => {
  describe('null/undefined handling', () => {
    it('toHUDViewModel should handle player with undefined username', () => {
      const player1 = createTestPlayer(1, { username: undefined as unknown as string });
      const player2 = createTestPlayer(2);
      const gameState = createTestGameState({ players: [player1, player2] });
      const options = createDefaultHUDOptions();

      const hud = toHUDViewModel(gameState, options);

      expect(hud.players[0].username).toBe('Player 1');
    });

    it('toHUDViewModel should handle empty player username', () => {
      const player1 = createTestPlayer(1, { username: '' });
      const player2 = createTestPlayer(2);
      const gameState = createTestGameState({ players: [player1, player2] });
      const options = createDefaultHUDOptions();

      const hud = toHUDViewModel(gameState, options);

      expect(hud.players[0].username).toBe('Player 1');
    });

    it('toEventLogViewModel should handle null history', () => {
      const result = toEventLogViewModel(null as unknown as GameHistoryEntry[], [], null);

      expect(result.entries).toHaveLength(0);
      expect(result.hasContent).toBe(false);
    });

    it('toBoardViewModel should handle empty board', () => {
      const board = createTestBoard('square8');

      const result = toBoardViewModel(board);

      expect(result.cells.every((c) => !c.stack && !c.marker && !c.collapsedSpace)).toBe(true);
    });
  });

  describe('board type handling', () => {
    it('should correctly handle square8 board type', () => {
      const board = createTestBoard('square8');
      const result = toBoardViewModel(board);

      expect(result.boardType).toBe('square8');
      expect(result.cells).toHaveLength(64);
    });

    it('should correctly handle square19 board type', () => {
      const board = createTestBoard('square19');
      const result = toBoardViewModel(board);

      expect(result.boardType).toBe('square19');
      expect(result.cells).toHaveLength(361);
    });

    it('should correctly handle hexagonal board type', () => {
      const board = createTestBoard('hexagonal');
      addStack(board, pos(0, 0, 0), 1, 1);

      const result = toBoardViewModel(board);

      expect(result.boardType).toBe('hexagonal');
      expect(result.rows).toBeUndefined();
    });
  });

  describe('position comparison with z coordinate', () => {
    it('should correctly compare hex positions with z coordinate', () => {
      const board = createTestBoard('hexagonal');
      addStack(board, pos(1, 0, -1), 1, 1);

      const options: ToBoardViewModelOptions = {
        selectedPosition: pos(1, 0, -1),
      };

      const result = toBoardViewModel(board, options);

      const cell = result.cells.find((c) => c.positionKey === '1,0,-1');
      expect(cell?.isSelected).toBe(true);
    });

    it('should correctly compare hex positions in valid targets', () => {
      const board = createTestBoard('hexagonal');
      // For hex boards, cells are only generated for positions with content
      // So we need to add content at the valid target positions
      addStack(board, pos(0, 0, 0), 1, 1);
      addStack(board, pos(1, -1, 0), 2, 1); // Target 1
      addMarker(board, pos(-1, 1, 0), 1); // Target 2

      const options: ToBoardViewModelOptions = {
        validTargets: [pos(1, -1, 0), pos(-1, 1, 0)],
      };

      const result = toBoardViewModel(board, options);

      const targetCell1 = result.cells.find((c) => c.positionKey === '1,-1,0');
      const targetCell2 = result.cells.find((c) => c.positionKey === '-1,1,0');

      expect(targetCell1).toBeDefined();
      expect(targetCell2).toBeDefined();
      expect(targetCell1?.isValidTarget).toBe(true);
      expect(targetCell2?.isValidTarget).toBe(true);
    });
  });

  describe('game end states', () => {
    it('should handle finished game status', () => {
      const gameState = createTestGameState({ gameStatus: 'finished' });
      const options = createDefaultHUDOptions();

      const hud = toHUDViewModel(gameState, options);

      expect(hud).toBeDefined();
    });

    it('should handle abandoned game', () => {
      const gameResult = createGameResult(undefined, 'abandonment');
      const players = [createTestPlayer(1), createTestPlayer(2)];
      const gameState = createTestGameState({ players, gameStatus: 'abandoned' });

      const result = toVictoryViewModel(gameResult, players, gameState);

      expect(result?.title).toContain('Abandoned');
    });
  });

  describe('large board performance', () => {
    it('should handle square19 board without timeout', () => {
      const board = createTestBoard('square19');

      // Add some content to process
      for (let x = 0; x < 10; x++) {
        for (let y = 0; y < 10; y++) {
          if ((x + y) % 3 === 0) {
            addStack(board, pos(x, y), (x % 2) + 1, 2);
          }
        }
      }

      const start = performance.now();
      const result = toBoardViewModel(board);
      const elapsed = performance.now() - start;

      expect(result.cells).toHaveLength(361);
      expect(elapsed).toBeLessThan(100); // Should complete in under 100ms
    });
  });

  describe('ring stats edge cases', () => {
    it('should handle player with all rings in hand', () => {
      const player1 = createTestPlayer(1, { ringsInHand: 18, eliminatedRings: 0 });
      const player2 = createTestPlayer(2, { ringsInHand: 18, eliminatedRings: 0 });
      const gameState = createTestGameState({ players: [player1, player2] });
      const options = createDefaultHUDOptions();

      const hud = toHUDViewModel(gameState, options);

      expect(hud.players[0].ringStats.inHand).toBe(18);
      expect(hud.players[0].ringStats.onBoard).toBe(0);
      expect(hud.players[0].ringStats.eliminated).toBe(0);
    });

    it('should handle player with all rings eliminated', () => {
      const player1 = createTestPlayer(1, { ringsInHand: 0, eliminatedRings: 18 });
      const player2 = createTestPlayer(2, { ringsInHand: 10, eliminatedRings: 8 });
      const gameState = createTestGameState({ players: [player1, player2] });
      const options = createDefaultHUDOptions();

      const hud = toHUDViewModel(gameState, options);

      expect(hud.players[0].ringStats.eliminated).toBe(18);
      expect(hud.players[0].ringStats.inHand).toBe(0);
    });
  });
});

// ═══════════════════════════════════════════════════════════════════════════
// Integration-style Tests
// ═══════════════════════════════════════════════════════════════════════════

describe('Full Transformation Pipeline', () => {
  it('should correctly transform a complete game state through all view models', () => {
    // Create a realistic game state
    const board = createTestBoard('square8');
    addStack(board, pos(0, 0), 1, 3);
    addStack(board, pos(7, 7), 2, 2);
    addMarker(board, pos(3, 3), 1);
    addMarker(board, pos(4, 4), 2);
    addCollapsedSpace(board, pos(5, 5), 1);

    const player1 = createTestPlayer(1, {
      id: 'user-1',
      ringsInHand: 10,
      eliminatedRings: 5,
      territorySpaces: 3,
    });
    const player2 = createTestPlayer(2, {
      ringsInHand: 12,
      eliminatedRings: 4,
      territorySpaces: 1,
    });

    const gameState = createTestGameState({
      board,
      players: [player1, player2],
      currentPlayer: 1,
      currentPhase: 'movement',
      spectators: ['spectator-1', 'spectator-2'],
    });

    // Test HUD transformation
    const hudOptions = createDefaultHUDOptions({ currentUserId: 'user-1' });
    const hud = toHUDViewModel(gameState, hudOptions);

    expect(hud.phase.phaseKey).toBe('movement');
    expect(hud.players[0].isCurrentPlayer).toBe(true);
    expect(hud.players[0].isUserPlayer).toBe(true);
    expect(hud.spectatorCount).toBe(2);

    // Test Board transformation
    const boardOptions: ToBoardViewModelOptions = {
      selectedPosition: pos(0, 0),
      validTargets: [pos(1, 1), pos(2, 2)],
    };
    const boardVM = toBoardViewModel(board, boardOptions);

    expect(boardVM.cells).toHaveLength(64);

    const selectedCell = boardVM.cells.find((c) => c.position.x === 0 && c.position.y === 0);
    expect(selectedCell?.isSelected).toBe(true);
    expect(selectedCell?.stack).toBeDefined();

    const targetCell = boardVM.cells.find((c) => c.position.x === 1 && c.position.y === 1);
    expect(targetCell?.isValidTarget).toBe(true);

    // Test Event Log transformation
    const history: GameHistoryEntry[] = [
      createTestHistoryEntry(1, 1, 'place_ring'),
      createTestHistoryEntry(2, 2, 'place_ring'),
    ];
    const eventLog = toEventLogViewModel(history, ['Game started'], null);

    expect(eventLog.entries.length).toBe(3); // 2 moves + 1 system event
    expect(eventLog.hasContent).toBe(true);

    // Test Victory transformation (when game ends)
    const gameResult: GameResult = {
      winner: 1,
      reason: 'ring_elimination',
      finalScore: {
        ringsEliminated: { 1: 20, 2: 15 },
        territorySpaces: { 1: 3, 2: 1 },
        ringsRemaining: { 1: 5, 2: 0 },
      },
    };

    const victoryVM = toVictoryViewModel(gameResult, [player1, player2], gameState, {
      currentUserId: 'user-1',
    });

    expect(victoryVM?.userWon).toBe(true);
    expect(victoryVM?.winner?.playerNumber).toBe(1);
    expect(victoryVM?.finalStats).toHaveLength(2);
  });

  it('should handle game state with lines and territories', () => {
    const board = createTestBoard('square8');

    // Add formed lines
    board.formedLines = [
      {
        positions: [pos(0, 0), pos(1, 1), pos(2, 2)],
        player: 1,
        length: 3,
        direction: { x: 1, y: 1 },
      },
    ];

    const gameState = createTestGameState({
      board,
      currentPhase: 'line_processing',
    });

    const hud = toHUDViewModel(gameState, createDefaultHUDOptions());

    expect(hud.phase.phaseKey).toBe('line_processing');
    expect(hud.subPhaseDetail).toBe('Processing 1 line');
  });
});
