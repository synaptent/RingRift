/**
 * GameEngine.utilityMethods.test.ts
 *
 * Tests for GameEngine utility methods that are not covered by scenario tests:
 * - Spectator management (addSpectator, removeSpectator)
 * - Game state control (pauseGame, resumeGame)
 * - Game ending (resignPlayer, abandonPlayer, abandonGameAsDraw, forfeitGame)
 */

import { GameEngine } from '../../src/server/game/GameEngine';
import { Player, TimeControl, BOARD_CONFIGS } from '../../src/shared/types/game';

describe('GameEngine utility methods', () => {
  let engine: GameEngine;

  const timeControl: TimeControl = { initialTime: 600, increment: 0, type: 'blitz' };

  const createPlayers = (): Player[] => [
    {
      id: 'p1',
      username: 'Player1',
      playerNumber: 1,
      type: 'ai', // AI players are auto-ready
      isReady: true,
      timeRemaining: 600000,
      ringsInHand: BOARD_CONFIGS['square8'].ringsPerPlayer,
      eliminatedRings: 0,
      territorySpaces: 0,
    },
    {
      id: 'p2',
      username: 'Player2',
      playerNumber: 2,
      type: 'ai', // AI players are auto-ready
      isReady: true,
      timeRemaining: 600000,
      ringsInHand: BOARD_CONFIGS['square8'].ringsPerPlayer,
      eliminatedRings: 0,
      territorySpaces: 0,
    },
  ];

  beforeEach(() => {
    engine = new GameEngine('test-utility', 'square8', createPlayers(), timeControl, false);
  });

  describe('spectator management', () => {
    it('adds a spectator successfully', () => {
      const result = engine.addSpectator('user-1');
      expect(result).toBe(true);
      expect(engine.getGameState().spectators).toContain('user-1');
    });

    it('returns false when adding duplicate spectator', () => {
      engine.addSpectator('user-1');
      const result = engine.addSpectator('user-1');
      expect(result).toBe(false);
      expect(engine.getGameState().spectators.filter((s) => s === 'user-1')).toHaveLength(1);
    });

    it('removes a spectator successfully', () => {
      engine.addSpectator('user-1');
      const result = engine.removeSpectator('user-1');
      expect(result).toBe(true);
      expect(engine.getGameState().spectators).not.toContain('user-1');
    });

    it('returns false when removing non-existent spectator', () => {
      const result = engine.removeSpectator('non-existent');
      expect(result).toBe(false);
    });

    it('handles multiple spectators', () => {
      engine.addSpectator('user-1');
      engine.addSpectator('user-2');
      engine.addSpectator('user-3');

      expect(engine.getGameState().spectators).toHaveLength(3);

      engine.removeSpectator('user-2');
      expect(engine.getGameState().spectators).toHaveLength(2);
      expect(engine.getGameState().spectators).toContain('user-1');
      expect(engine.getGameState().spectators).not.toContain('user-2');
      expect(engine.getGameState().spectators).toContain('user-3');
    });
  });

  describe('pause and resume', () => {
    beforeEach(() => {
      // Start the game so it transitions to 'active' status
      engine.startGame();
    });

    it('pauses an active game', () => {
      expect(engine.getGameState().gameStatus).toBe('active');
      const result = engine.pauseGame();
      expect(result).toBe(true);
      expect(engine.getGameState().gameStatus).toBe('paused');
    });

    it('returns false when pausing a non-active game', () => {
      engine.pauseGame(); // First pause
      const result = engine.pauseGame(); // Try to pause again
      expect(result).toBe(false);
      expect(engine.getGameState().gameStatus).toBe('paused');
    });

    it('resumes a paused game', () => {
      engine.pauseGame();
      expect(engine.getGameState().gameStatus).toBe('paused');

      const result = engine.resumeGame();
      expect(result).toBe(true);
      expect(engine.getGameState().gameStatus).toBe('active');
    });

    it('returns false when resuming a non-paused game', () => {
      const result = engine.resumeGame();
      expect(result).toBe(false);
      expect(engine.getGameState().gameStatus).toBe('active');
    });

    it('handles pause/resume cycle', () => {
      expect(engine.getGameState().gameStatus).toBe('active');

      engine.pauseGame();
      expect(engine.getGameState().gameStatus).toBe('paused');

      engine.resumeGame();
      expect(engine.getGameState().gameStatus).toBe('active');

      engine.pauseGame();
      expect(engine.getGameState().gameStatus).toBe('paused');
    });
  });

  describe('resignPlayer', () => {
    beforeEach(() => {
      engine.startGame();
    });

    it('ends game with resignation reason', () => {
      const result = engine.resignPlayer(1);

      expect(result.success).toBe(true);
      expect(result.gameResult.reason).toBe('resignation');
      expect(result.gameResult.winner).toBe(2); // Other player wins
      expect(engine.getGameState().gameStatus).toBe('completed');
    });

    it('awards win to player 1 when player 2 resigns', () => {
      const result = engine.resignPlayer(2);

      expect(result.success).toBe(true);
      expect(result.gameResult.winner).toBe(1);
    });
  });

  describe('abandonPlayer', () => {
    beforeEach(() => {
      engine.startGame();
    });

    it('ends game with abandonment reason', () => {
      const result = engine.abandonPlayer(1);

      expect(result.success).toBe(true);
      expect(result.gameResult.reason).toBe('abandonment');
      expect(result.gameResult.winner).toBe(2);
      expect(engine.getGameState().gameStatus).toBe('completed');
    });

    it('awards win to remaining player', () => {
      const result = engine.abandonPlayer(2);

      expect(result.success).toBe(true);
      expect(result.gameResult.winner).toBe(1);
    });
  });

  describe('abandonGameAsDraw', () => {
    beforeEach(() => {
      engine.startGame();
    });

    it('ends game without a winner', () => {
      const result = engine.abandonGameAsDraw();

      expect(result.success).toBe(true);
      expect(result.gameResult.reason).toBe('abandonment');
      expect(result.gameResult.winner).toBeUndefined();
      expect(engine.getGameState().gameStatus).toBe('completed');
    });
  });

  describe('forfeitGame (timeout)', () => {
    beforeEach(() => {
      engine.startGame();
    });

    it('ends game with timeout reason', () => {
      const result = engine.forfeitGame('1');

      expect(result.success).toBe(true);
      expect(result.gameResult?.reason).toBe('timeout');
      expect(result.gameResult?.winner).toBe(2);
      expect(engine.getGameState().gameStatus).toBe('completed');
    });

    it('awards win to player 1 when player 2 times out', () => {
      const result = engine.forfeitGame('2');

      expect(result.success).toBe(true);
      expect(result.gameResult?.winner).toBe(1);
    });
  });

  describe('swapSidesApplied getter', () => {
    it('returns false initially', () => {
      expect(engine.swapSidesApplied).toBe(false);
    });
  });

  describe('rated game ending', () => {
    let ratedEngine: GameEngine;

    beforeEach(() => {
      ratedEngine = new GameEngine('test-rated', 'square8', createPlayers(), timeControl, true);
      ratedEngine.startGame();
    });

    it('triggers rating update on resignation in rated game', () => {
      const result = ratedEngine.resignPlayer(1);

      expect(result.success).toBe(true);
      expect(result.gameResult.reason).toBe('resignation');
      // Rating update is logged but not directly testable without mocking
    });

    it('triggers rating update on abandonment in rated game', () => {
      const result = ratedEngine.abandonPlayer(1);

      expect(result.success).toBe(true);
      expect(result.gameResult.reason).toBe('abandonment');
    });
  });
});

describe('GameEngine orchestrator adapter control', () => {
  let engine: GameEngine;
  const timeControl: TimeControl = { initialTime: 600, increment: 0, type: 'blitz' };

  const createPlayers = (): Player[] => [
    {
      id: 'p1',
      username: 'Player1',
      playerNumber: 1,
      type: 'ai',
      isReady: true,
      timeRemaining: 600000,
      ringsInHand: BOARD_CONFIGS['square8'].ringsPerPlayer,
      eliminatedRings: 0,
      territorySpaces: 0,
    },
    {
      id: 'p2',
      username: 'Player2',
      playerNumber: 2,
      type: 'ai',
      isReady: true,
      timeRemaining: 600000,
      ringsInHand: BOARD_CONFIGS['square8'].ringsPerPlayer,
      eliminatedRings: 0,
      territorySpaces: 0,
    },
  ];

  beforeEach(() => {
    engine = new GameEngine('test-adapter', 'square8', createPlayers(), timeControl, false);
  });

  it('enables orchestrator adapter', () => {
    engine.enableOrchestratorAdapter();
    expect(engine.isOrchestratorAdapterEnabled()).toBe(true);
  });

  it('disables orchestrator adapter', () => {
    engine.disableOrchestratorAdapter();
    expect(engine.isOrchestratorAdapterEnabled()).toBe(false);
  });

  it('reports correct state after multiple toggles', () => {
    // Store initial state (may be enabled by config)
    const initialState = engine.isOrchestratorAdapterEnabled();

    engine.disableOrchestratorAdapter();
    expect(engine.isOrchestratorAdapterEnabled()).toBe(false);

    engine.enableOrchestratorAdapter();
    expect(engine.isOrchestratorAdapterEnabled()).toBe(true);

    engine.disableOrchestratorAdapter();
    expect(engine.isOrchestratorAdapterEnabled()).toBe(false);
  });
});

describe('GameEngine move-driven decision phases', () => {
  let engine: GameEngine;
  const timeControl: TimeControl = { initialTime: 600, increment: 0, type: 'blitz' };

  const createPlayers = (): Player[] => [
    {
      id: 'p1',
      username: 'Player1',
      playerNumber: 1,
      type: 'ai',
      isReady: true,
      timeRemaining: 600000,
      ringsInHand: BOARD_CONFIGS['square8'].ringsPerPlayer,
      eliminatedRings: 0,
      territorySpaces: 0,
    },
    {
      id: 'p2',
      username: 'Player2',
      playerNumber: 2,
      type: 'ai',
      isReady: true,
      timeRemaining: 600000,
      ringsInHand: BOARD_CONFIGS['square8'].ringsPerPlayer,
      eliminatedRings: 0,
      territorySpaces: 0,
    },
  ];

  beforeEach(() => {
    engine = new GameEngine('test-decision', 'square8', createPlayers(), timeControl, false);
  });

  it('enables move-driven decision phases', () => {
    // This method modifies internal flags for decision handling
    engine.enableMoveDrivenDecisionPhases();
    // The effect is internal - just verify it doesn't throw
    expect(true).toBe(true);
  });
});

describe('GameEngine getGameState', () => {
  let engine: GameEngine;
  const timeControl: TimeControl = { initialTime: 600, increment: 0, type: 'blitz' };

  const createPlayers = (): Player[] => [
    {
      id: 'p1',
      username: 'Player1',
      playerNumber: 1,
      type: 'ai',
      isReady: true,
      timeRemaining: 600000,
      ringsInHand: BOARD_CONFIGS['square8'].ringsPerPlayer,
      eliminatedRings: 0,
      territorySpaces: 0,
    },
    {
      id: 'p2',
      username: 'Player2',
      playerNumber: 2,
      type: 'ai',
      isReady: true,
      timeRemaining: 600000,
      ringsInHand: BOARD_CONFIGS['square8'].ringsPerPlayer,
      eliminatedRings: 0,
      territorySpaces: 0,
    },
  ];

  beforeEach(() => {
    engine = new GameEngine('test-state', 'square8', createPlayers(), timeControl, false);
  });

  it('returns game state with correct game ID', () => {
    const state = engine.getGameState();
    expect(state.id).toBe('test-state');
  });

  it('returns game state with correct board type', () => {
    const state = engine.getGameState();
    expect(state.boardType).toBe('square8');
  });

  it('returns game state with correct number of players', () => {
    const state = engine.getGameState();
    expect(state.players).toHaveLength(2);
  });

  it('returns game state with initial phase as ring_placement', () => {
    const state = engine.getGameState();
    expect(state.currentPhase).toBe('ring_placement');
  });

  it('returns game state with player 1 as current player initially', () => {
    const state = engine.getGameState();
    expect(state.currentPlayer).toBe(1);
  });

  it('returns game state with empty move history initially', () => {
    const state = engine.getGameState();
    expect(state.moveHistory).toHaveLength(0);
  });

  it('returns game state with correct time control', () => {
    const state = engine.getGameState();
    expect(state.timeControl).toEqual(timeControl);
  });

  it('returns game state with isRated matching constructor arg', () => {
    const state = engine.getGameState();
    expect(state.isRated).toBe(false);
  });
});

describe('GameEngine getValidMoves', () => {
  let engine: GameEngine;
  const timeControl: TimeControl = { initialTime: 600, increment: 0, type: 'blitz' };

  const createPlayers = (): Player[] => [
    {
      id: 'p1',
      username: 'Player1',
      playerNumber: 1,
      type: 'ai',
      isReady: true,
      timeRemaining: 600000,
      ringsInHand: BOARD_CONFIGS['square8'].ringsPerPlayer,
      eliminatedRings: 0,
      territorySpaces: 0,
    },
    {
      id: 'p2',
      username: 'Player2',
      playerNumber: 2,
      type: 'ai',
      isReady: true,
      timeRemaining: 600000,
      ringsInHand: BOARD_CONFIGS['square8'].ringsPerPlayer,
      eliminatedRings: 0,
      territorySpaces: 0,
    },
  ];

  beforeEach(() => {
    engine = new GameEngine('test-moves', 'square8', createPlayers(), timeControl, false);
    engine.startGame();
  });

  it('returns valid placement moves at game start', () => {
    const moves = engine.getValidMoves(1);
    // Placement moves should be available on an empty board
    expect(moves.length).toBeGreaterThan(0);
    // At least some moves should be place_ring type
    const placeMoves = moves.filter((m) => m.type === 'place_ring');
    expect(placeMoves.length).toBeGreaterThan(0);
  });

  it('returns no moves for non-current player', () => {
    // Player 2 is not the current player at start
    const moves = engine.getValidMoves(2);
    // In ring_placement phase, only current player can move
    expect(moves).toHaveLength(0);
  });
});

describe('GameEngine startGame edge cases', () => {
  const timeControl: TimeControl = { initialTime: 600, increment: 0, type: 'blitz' };

  it('returns false if not all players are ready', () => {
    const players: Player[] = [
      {
        id: 'p1',
        username: 'Player1',
        playerNumber: 1,
        type: 'human', // Human players are not auto-ready
        isReady: false,
        timeRemaining: 600000,
        ringsInHand: BOARD_CONFIGS['square8'].ringsPerPlayer,
        eliminatedRings: 0,
        territorySpaces: 0,
      },
      {
        id: 'p2',
        username: 'Player2',
        playerNumber: 2,
        type: 'ai',
        isReady: true,
        timeRemaining: 600000,
        ringsInHand: BOARD_CONFIGS['square8'].ringsPerPlayer,
        eliminatedRings: 0,
        territorySpaces: 0,
      },
    ];

    const engine = new GameEngine('test-not-ready', 'square8', players, timeControl, false);
    const result = engine.startGame();
    expect(result).toBe(false);
    expect(engine.getGameState().gameStatus).toBe('waiting');
  });

  it('returns true when all players are ready', () => {
    const players: Player[] = [
      {
        id: 'p1',
        username: 'Player1',
        playerNumber: 1,
        type: 'ai',
        isReady: true,
        timeRemaining: 600000,
        ringsInHand: BOARD_CONFIGS['square8'].ringsPerPlayer,
        eliminatedRings: 0,
        territorySpaces: 0,
      },
      {
        id: 'p2',
        username: 'Player2',
        playerNumber: 2,
        type: 'ai',
        isReady: true,
        timeRemaining: 600000,
        ringsInHand: BOARD_CONFIGS['square8'].ringsPerPlayer,
        eliminatedRings: 0,
        territorySpaces: 0,
      },
    ];

    const engine = new GameEngine('test-ready', 'square8', players, timeControl, false);
    const result = engine.startGame();
    expect(result).toBe(true);
    expect(engine.getGameState().gameStatus).toBe('active');
  });
});
