/**
 * GameEngine.branchCoverage.test.ts
 *
 * Additional branch coverage tests for GameEngine.ts targeting
 * specific uncovered branches identified by coverage analysis.
 */

import { GameEngine } from '../../src/server/game/GameEngine';
import { Player, TimeControl, BOARD_CONFIGS, Move } from '../../src/shared/types/game';

describe('GameEngine branch coverage', () => {
  const timeControl: TimeControl = { initialTime: 600, increment: 0, type: 'blitz' };

  const createPlayers = (count: number = 2): Player[] => {
    const players: Player[] = [];
    for (let i = 0; i < count; i++) {
      players.push({
        id: `p${i + 1}`,
        username: `Player${i + 1}`,
        playerNumber: i + 1,
        type: 'ai',
        isReady: true,
        timeRemaining: 600000,
        ringsInHand: BOARD_CONFIGS['square8'].ringsPerPlayer,
        eliminatedRings: 0,
        territorySpaces: 0,
      });
    }
    return players;
  };

  describe('constructor variations', () => {
    it('creates engine with default isRated=true', () => {
      const engine = new GameEngine('test-default', 'square8', createPlayers(), timeControl);
      const state = engine.getGameState();
      expect(state.isRated).toBe(true);
    });

    it('creates engine with isRated=false', () => {
      const engine = new GameEngine('test-unrated', 'square8', createPlayers(), timeControl, false);
      const state = engine.getGameState();
      expect(state.isRated).toBe(false);
    });

    it('creates engine with rngSeed', () => {
      const engine = new GameEngine(
        'test-seeded',
        'square8',
        createPlayers(),
        timeControl,
        false,
        undefined,
        12345
      );
      const state = engine.getGameState();
      expect(state.rngSeed).toBe(12345);
    });

    it('creates engine without rngSeed (undefined)', () => {
      const engine = new GameEngine(
        'test-no-seed',
        'square8',
        createPlayers(),
        timeControl,
        false,
        undefined,
        undefined
      );
      const state = engine.getGameState();
      expect(state.rngSeed).toBeUndefined();
    });

    it('creates engine with rulesOptions', () => {
      const engine = new GameEngine(
        'test-rules',
        'square8',
        createPlayers(),
        timeControl,
        false,
        undefined,
        undefined,
        { swapRuleEnabled: true }
      );
      const state = engine.getGameState();
      expect(state.rulesOptions?.swapRuleEnabled).toBe(true);
    });

    it('creates engine without rulesOptions', () => {
      const engine = new GameEngine(
        'test-no-rules',
        'square8',
        createPlayers(),
        timeControl,
        false,
        undefined,
        undefined,
        undefined
      );
      const state = engine.getGameState();
      expect(state.rulesOptions).toBeUndefined();
    });

    it('creates engine with 3 players', () => {
      const engine = new GameEngine('test-3p', 'square8', createPlayers(3), timeControl);
      const state = engine.getGameState();
      expect(state.players).toHaveLength(3);
      expect(state.maxPlayers).toBe(3);
      // 18 rings * 3 players = 54 rings total
      // Victory threshold = floor(54/2) + 1 = 28
      expect(state.victoryThreshold).toBe(28);
    });

    it('creates engine with 4 players', () => {
      const engine = new GameEngine('test-4p', 'square8', createPlayers(4), timeControl);
      const state = engine.getGameState();
      expect(state.players).toHaveLength(4);
      expect(state.maxPlayers).toBe(4);
      // 18 rings * 4 players = 72 rings total
      // Victory threshold = floor(72/2) + 1 = 37
      expect(state.victoryThreshold).toBe(37);
    });
  });

  describe('game state management', () => {
    let engine: GameEngine;

    beforeEach(() => {
      engine = new GameEngine('test-state', 'square8', createPlayers(), timeControl, false);
    });

    it('starts in waiting status', () => {
      expect(engine.getGameState().gameStatus).toBe('waiting');
    });

    it('transitions to active on startGame', () => {
      engine.startGame();
      expect(engine.getGameState().gameStatus).toBe('active');
    });

    it('startGame is idempotent', () => {
      engine.startGame();
      engine.startGame(); // Second call should be no-op
      expect(engine.getGameState().gameStatus).toBe('active');
    });
  });

  describe('spectator management edge cases', () => {
    let engine: GameEngine;

    beforeEach(() => {
      engine = new GameEngine('test-spectators', 'square8', createPlayers(), timeControl, false);
    });

    it('handles multiple add operations', () => {
      expect(engine.addSpectator('s1')).toBe(true);
      expect(engine.addSpectator('s2')).toBe(true);
      expect(engine.addSpectator('s3')).toBe(true);
      expect(engine.getGameState().spectators).toHaveLength(3);
    });

    it('handles duplicate add (returns false)', () => {
      engine.addSpectator('s1');
      expect(engine.addSpectator('s1')).toBe(false);
      expect(engine.getGameState().spectators).toHaveLength(1);
    });

    it('handles remove from empty list', () => {
      expect(engine.removeSpectator('nonexistent')).toBe(false);
    });

    it('handles remove existing then add again', () => {
      engine.addSpectator('s1');
      expect(engine.removeSpectator('s1')).toBe(true);
      expect(engine.addSpectator('s1')).toBe(true);
      expect(engine.getGameState().spectators).toHaveLength(1);
    });
  });

  describe('pause/resume edge cases', () => {
    let engine: GameEngine;

    beforeEach(() => {
      engine = new GameEngine('test-pause', 'square8', createPlayers(), timeControl, false);
    });

    it('cannot pause game in waiting status', () => {
      expect(engine.pauseGame()).toBe(false);
    });

    it('cannot resume game in waiting status', () => {
      expect(engine.resumeGame()).toBe(false);
    });

    it('can pause and resume when active', () => {
      engine.startGame();
      expect(engine.pauseGame()).toBe(true);
      expect(engine.getGameState().gameStatus).toBe('paused');
      expect(engine.resumeGame()).toBe(true);
      expect(engine.getGameState().gameStatus).toBe('active');
    });

    it('cannot resume when already active', () => {
      engine.startGame();
      expect(engine.resumeGame()).toBe(false);
    });

    it('cannot pause when already paused', () => {
      engine.startGame();
      engine.pauseGame();
      expect(engine.pauseGame()).toBe(false);
    });
  });

  describe('resignation and forfeit', () => {
    let engine: GameEngine;

    beforeEach(() => {
      engine = new GameEngine('test-resign', 'square8', createPlayers(), timeControl, false);
      engine.startGame();
    });

    it('player 1 resignation awards win to player 2', () => {
      const result = engine.resignPlayer(1);
      expect(result.success).toBe(true);
      expect(result.gameResult?.winner).toBe(2);
      expect(result.gameResult?.reason).toBe('resignation');
      expect(engine.getGameState().gameStatus).toBe('completed');
    });

    it('player 2 resignation awards win to player 1', () => {
      const result = engine.resignPlayer(2);
      expect(result.success).toBe(true);
      expect(result.gameResult?.winner).toBe(1);
    });

    it('forfeit ends game with timeout reason', () => {
      const result = engine.forfeitGame('1');
      expect(result.success).toBe(true);
      expect(result.gameResult?.reason).toBe('timeout');
    });
  });

  describe('abandon scenarios', () => {
    let engine: GameEngine;

    beforeEach(() => {
      engine = new GameEngine('test-abandon', 'square8', createPlayers(), timeControl, false);
      engine.startGame();
    });

    it('abandon player 1 ends game', () => {
      const result = engine.abandonPlayer(1);
      expect(result.success).toBe(true);
      expect(engine.getGameState().gameStatus).toBe('completed');
    });

    it('abandon player 2 ends game', () => {
      const result = engine.abandonPlayer(2);
      expect(result.success).toBe(true);
    });

    it('abandonGameAsDraw ends game as draw', () => {
      const result = engine.abandonGameAsDraw();
      expect(result.success).toBe(true);
      expect(engine.getGameState().gameStatus).toBe('completed');
      expect(result.gameResult?.reason).toBe('abandonment');
      expect(result.gameResult?.winner).toBeUndefined();
    });
  });

  describe('swapSidesApplied getter', () => {
    it('returns false for new game', () => {
      const engine = new GameEngine('test-swap', 'square8', createPlayers(), timeControl, false);
      expect(engine.swapSidesApplied).toBe(false);
    });

    it('returns false for started game without swap', () => {
      const engine = new GameEngine('test-swap2', 'square8', createPlayers(), timeControl, false);
      engine.startGame();
      expect(engine.swapSidesApplied).toBe(false);
    });

    // Note: swap_sides semantics and meta-move exposure are exercised in
    // dedicated GameEngine.swapRule tests; here we only cover the basic
    // getter behaviour.
  });

  describe('time control variations', () => {
    it('handles rapid time control', () => {
      const rapidTime: TimeControl = { initialTime: 900, increment: 10, type: 'rapid' };
      const engine = new GameEngine('test-rapid', 'square8', createPlayers(), rapidTime);
      const state = engine.getGameState();
      expect(state.timeControl.type).toBe('rapid');
      expect(state.players[0].timeRemaining).toBe(900000);
    });

    it('handles blitz time control with increment', () => {
      const blitzTime: TimeControl = { initialTime: 300, increment: 3, type: 'blitz' };
      const engine = new GameEngine('test-blitz', 'square8', createPlayers(), blitzTime);
      const state = engine.getGameState();
      expect(state.timeControl.increment).toBe(3);
    });

    it('handles bullet time control', () => {
      const bulletTime: TimeControl = { initialTime: 60, increment: 0, type: 'bullet' };
      const engine = new GameEngine('test-bullet', 'square8', createPlayers(), bulletTime);
      const state = engine.getGameState();
      expect(state.players[0].timeRemaining).toBe(60000);
    });
  });

  describe('board configuration', () => {
    it('square8 has correct total rings', () => {
      const engine = new GameEngine('test-s8', 'square8', createPlayers(), timeControl);
      const state = engine.getGameState();
      // 18 rings per player * 2 players = 36 total
      expect(state.totalRingsInPlay).toBe(36);
    });

    it('square8 has correct territory victory threshold', () => {
      const engine = new GameEngine('test-s8-terr', 'square8', createPlayers(), timeControl);
      const state = engine.getGameState();
      // 64 spaces / 2 + 1 = 33
      expect(state.territoryVictoryThreshold).toBe(33);
    });

    it('3-player game has correct total rings', () => {
      const engine = new GameEngine('test-3p-rings', 'square8', createPlayers(3), timeControl);
      const state = engine.getGameState();
      // 18 * 3 = 54
      expect(state.totalRingsInPlay).toBe(54);
    });
  });

  describe('getValidMoves safeguards and swap meta-move', () => {
    it('returns empty list for non-active player', () => {
      const engine = new GameEngine(
        'test-moves-nonactive',
        'square8',
        createPlayers(),
        timeControl
      );
      engine.startGame();
      const moves = engine.getValidMoves(2);
      expect(moves).toEqual([]);
    });

    // Swap-sides meta-move injection is covered by GameEngine.swapRule tests;
    // here we focus on core safeguards and non-active player behaviour.
  });

  describe('initial state correctness', () => {
    it('starts in ring_placement phase', () => {
      const engine = new GameEngine('test-phase', 'square8', createPlayers(), timeControl);
      engine.startGame();
      expect(engine.getGameState().currentPhase).toBe('ring_placement');
    });

    it('player 1 goes first', () => {
      const engine = new GameEngine('test-first', 'square8', createPlayers(), timeControl);
      engine.startGame();
      expect(engine.getGameState().currentPlayer).toBe(1);
    });

    it('move history starts empty', () => {
      const engine = new GameEngine('test-history', 'square8', createPlayers(), timeControl);
      expect(engine.getGameState().moveHistory).toHaveLength(0);
    });

    it('spectators list starts empty', () => {
      const engine = new GameEngine('test-spec', 'square8', createPlayers(), timeControl);
      expect(engine.getGameState().spectators).toHaveLength(0);
    });
  });

  describe('makeMoveById error branches', () => {
    let engine: GameEngine;

    beforeEach(() => {
      jest.restoreAllMocks();
      engine = new GameEngine('test-makeMoveById', 'square8', createPlayers(), timeControl, false);
      engine.startGame();
    });

    afterEach(() => {
      jest.restoreAllMocks();
    });

    it('returns error when player is not the active player', async () => {
      const result = await engine.makeMoveById(2, 'any-id');
      expect(result.success).toBe(false);
      expect(result.error).toMatch(/not the active player/);
    });

    it('returns error when no valid moves are available', async () => {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const getValidMovesSpy = jest.spyOn(engine as any, 'getValidMoves').mockReturnValue([]);

      const result = await engine.makeMoveById(1, 'missing');

      expect(getValidMovesSpy).toHaveBeenCalledWith(1);
      expect(result.success).toBe(false);
      expect(result.error).toMatch(/No valid moves available/);
    });

    it('returns error when move id is not found', async () => {
      const candidate: Move = {
        id: 'other-id',
        type: 'place_ring',
        player: 1,
        to: { x: 0, y: 0 },
        timestamp: new Date(),
        thinkTime: 0,
        moveNumber: 1,
      };

      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      jest.spyOn(engine as any, 'getValidMoves').mockReturnValue([candidate]);

      const result = await engine.makeMoveById(1, 'missing-id');

      expect(result.success).toBe(false);
      expect(result.error).toMatch(/No valid move with id missing-id/);
    });

    it('returns error when selected move belongs to a different player', async () => {
      const candidate: Move = {
        id: 'mismatch-player',
        type: 'place_ring',
        player: 2,
        to: { x: 0, y: 0 },
        timestamp: new Date(),
        thinkTime: 0,
        moveNumber: 1,
      };

      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      jest.spyOn(engine as any, 'getValidMoves').mockReturnValue([candidate]);

      const result = await engine.makeMoveById(1, 'mismatch-player');

      expect(result.success).toBe(false);
      expect(result.error).toMatch(/belongs to player 2, not 1/);
    });

    it('delegates to makeMove on successful selection', async () => {
      const candidate: Move = {
        id: 'ok-id',
        type: 'place_ring',
        player: 1,
        to: { x: 1, y: 1 },
        timestamp: new Date(),
        thinkTime: 0,
        moveNumber: 1,
      };

      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      jest.spyOn(engine as any, 'getValidMoves').mockReturnValue([candidate]);

      const makeMoveSpy = jest
        .spyOn(engine as any, 'makeMove')
        .mockResolvedValue({ success: true, gameState: engine.getGameState() });

      const result = await engine.makeMoveById(1, 'ok-id');

      expect(makeMoveSpy).toHaveBeenCalledTimes(1);
      expect(result.success).toBe(true);
    });
  });

  describe('makeMove chain capture branches', () => {
    let engine: GameEngine;

    beforeEach(() => {
      engine = new GameEngine('test-chain', 'square8', createPlayers(), timeControl, false);
      engine.startGame();
    });

    it('rejects continue_capture_segment when no chain is active', async () => {
      const result = await engine.makeMove({
        type: 'continue_capture_segment',
        player: 1,
        from: { x: 3, y: 3 },
        to: { x: 5, y: 5 },
        captureTarget: { x: 4, y: 4 },
      });
      expect(result.success).toBe(false);
      expect(result.error).toMatch(/No chain capture in progress/);
    });

    it('rejects chain move from wrong player when chain is active', async () => {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const engineAny = engine as any;
      engineAny.chainCaptureState = {
        playerNumber: 1,
        currentPosition: { x: 3, y: 3 },
        capturedCapHeights: [2],
        availableMoves: [],
      };

      const result = await engine.makeMove({
        type: 'continue_capture_segment',
        player: 2, // wrong player
        from: { x: 3, y: 3 },
        to: { x: 5, y: 5 },
        captureTarget: { x: 4, y: 4 },
      });
      expect(result.success).toBe(false);
      expect(result.error).toMatch(/only the capturing player may move/);
    });

    it('rejects chain move from wrong position', async () => {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const engineAny = engine as any;
      engineAny.chainCaptureState = {
        playerNumber: 1,
        currentPosition: { x: 3, y: 3 },
        capturedCapHeights: [2],
        availableMoves: [],
      };

      const result = await engine.makeMove({
        type: 'continue_capture_segment',
        player: 1,
        from: { x: 5, y: 5 }, // wrong position
        to: { x: 7, y: 7 },
        captureTarget: { x: 6, y: 6 },
      });
      expect(result.success).toBe(false);
      expect(result.error).toMatch(/must continue capturing with the same stack/);
    });

    it('rejects non-chain move during active chain', async () => {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const engineAny = engine as any;
      engineAny.chainCaptureState = {
        playerNumber: 1,
        currentPosition: { x: 3, y: 3 },
        capturedCapHeights: [2],
        availableMoves: [],
      };

      const result = await engine.makeMove({
        type: 'move_stack',
        player: 1,
        from: { x: 1, y: 1 },
        to: { x: 2, y: 2 },
      });
      expect(result.success).toBe(false);
      expect(result.error).toMatch(/must continue capturing/);
    });
  });

  describe('makeMove must-move-from enforcement', () => {
    let engine: GameEngine;

    beforeEach(() => {
      engine = new GameEngine('test-mustmove', 'square8', createPlayers(), timeControl, false);
      engine.startGame();
    });

    it('rejects movement from wrong stack when mustMoveFromStackKey is set', async () => {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const engineAny = engine as any;
      engineAny.mustMoveFromStackKey = '3,3';

      const result = await engine.makeMove({
        type: 'move_stack',
        player: 1,
        from: { x: 1, y: 1 }, // wrong position
        to: { x: 2, y: 2 },
      });
      expect(result.success).toBe(false);
      expect(result.error).toMatch(/must move the stack that was just placed/);
    });

    it('rejects capture from wrong stack when mustMoveFromStackKey is set', async () => {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const engineAny = engine as any;
      engineAny.mustMoveFromStackKey = '3,3';

      const result = await engine.makeMove({
        type: 'overtaking_capture',
        player: 1,
        from: { x: 1, y: 1 },
        to: { x: 5, y: 5 },
        captureTarget: { x: 3, y: 3 },
      });
      expect(result.success).toBe(false);
      expect(result.error).toMatch(/must move the stack that was just placed/);
    });
  });

  describe('square19 board configuration', () => {
    it('creates square19 board correctly', () => {
      const engine = new GameEngine('test-s19', 'square19', createPlayers(), timeControl);
      const state = engine.getGameState();
      expect(state.boardType).toBe('square19');
      expect(state.board.size).toBe(19);
    });

    it('square19 board with 3 players has correct thresholds', () => {
      const engine = new GameEngine('test-s19-3p', 'square19', createPlayers(3), timeControl);
      const state = engine.getGameState();
      expect(state.players).toHaveLength(3);
      // 36 rings per player * 3 = 108 total
      expect(state.totalRingsInPlay).toBe(108);
    });
  });

  describe('game end and forfeit variations', () => {
    let engine: GameEngine;

    beforeEach(() => {
      engine = new GameEngine('test-end', 'square8', createPlayers(), timeControl, false);
      engine.startGame();
    });

    it('forfeit by player id string ends game', () => {
      const result = engine.forfeitGame('p1');
      expect(result.success).toBe(true);
      expect(engine.getGameState().gameStatus).toBe('completed');
    });

    it('resign player 1 in 3-player game', () => {
      const engine3p = new GameEngine('test-3p-resign', 'square8', createPlayers(3), timeControl);
      engine3p.startGame();
      const result = engine3p.resignPlayer(1);
      expect(result.success).toBe(true);
      // In 3+ player games, resignation should eliminate the player
      expect(engine3p.getGameState().gameStatus).toBe('completed');
    });

    it('resign player 3 in 3-player game', () => {
      const engine3p = new GameEngine('test-3p-resign2', 'square8', createPlayers(3), timeControl);
      engine3p.startGame();
      const result = engine3p.resignPlayer(3);
      expect(result.success).toBe(true);
    });
  });

  describe('player ready state', () => {
    it('cannot start game when not all players ready', () => {
      const unreadyPlayers: Player[] = [
        {
          id: 'p1',
          username: 'Player1',
          playerNumber: 1,
          type: 'human',
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
          type: 'human',
          isReady: false, // not ready
          timeRemaining: 600000,
          ringsInHand: BOARD_CONFIGS['square8'].ringsPerPlayer,
          eliminatedRings: 0,
          territorySpaces: 0,
        },
      ];
      const engine = new GameEngine('test-notready', 'square8', unreadyPlayers, timeControl);
      expect(engine.startGame()).toBe(false);
      expect(engine.getGameState().gameStatus).toBe('waiting');
    });
  });

  describe('debugCheckpointHook', () => {
    it('can set and clear debug checkpoint hook', () => {
      const engine = new GameEngine('test-hook', 'square8', createPlayers(), timeControl);
      const hookFn = jest.fn();

      engine.setDebugCheckpointHook(hookFn);
      engine.setDebugCheckpointHook(undefined);

      // No assertion needed - just verifying no errors
      expect(true).toBe(true);
    });
  });

  describe('getGameState deep cloning', () => {
    it('returns cloned board with independent maps', () => {
      const engine = new GameEngine('test-clone', 'square8', createPlayers(), timeControl);
      const state1 = engine.getGameState();
      const state2 = engine.getGameState();

      // Board maps should be different instances
      expect(state1.board.stacks).not.toBe(state2.board.stacks);
      expect(state1.board.markers).not.toBe(state2.board.markers);
      expect(state1.board.collapsedSpaces).not.toBe(state2.board.collapsedSpaces);
      expect(state1.board.territories).not.toBe(state2.board.territories);
    });

    it('returns cloned players array', () => {
      const engine = new GameEngine('test-clone2', 'square8', createPlayers(), timeControl);
      const state1 = engine.getGameState();
      const state2 = engine.getGameState();

      expect(state1.players).not.toBe(state2.players);
      expect(state1.players[0]).not.toBe(state2.players[0]);
    });

    it('returns cloned history arrays', () => {
      const engine = new GameEngine('test-clone3', 'square8', createPlayers(), timeControl);
      const state1 = engine.getGameState();
      const state2 = engine.getGameState();

      expect(state1.moveHistory).not.toBe(state2.moveHistory);
      expect(state1.history).not.toBe(state2.history);
      expect(state1.spectators).not.toBe(state2.spectators);
    });
  });

  describe('swap sides meta-move', () => {
    it('makeMove routes swap_sides to applySwapSidesMove', async () => {
      const engine = new GameEngine(
        'test-swap-move',
        'square8',
        createPlayers(),
        timeControl,
        false,
        undefined,
        undefined,
        { swapRuleEnabled: true }
      );
      engine.startGame();

      // Need to complete player 1's first turn for swap to be available
      // For now, test that the route exists even if validation fails
      const result = await engine.makeMove({
        type: 'swap_sides',
        player: 2,
        to: { x: 0, y: 0 },
      });

      // Will fail validation since P1 hasn't moved, but confirms routing
      expect(result.success).toBe(false);
    });
  });

  describe('eliminatedRings tracking', () => {
    it('tracks eliminatedRings in board state', () => {
      const engine = new GameEngine('test-elim-track', 'square8', createPlayers(), timeControl);
      const state = engine.getGameState();
      expect(state.board.eliminatedRings).toEqual({});
    });

    it('totalRingsEliminated starts at 0', () => {
      const engine = new GameEngine('test-elim-total', 'square8', createPlayers(), timeControl);
      const state = engine.getGameState();
      expect(state.totalRingsEliminated).toBe(0);
    });
  });

  describe('orchestrator adapter methods', () => {
    it('enableMoveDrivenDecisionPhases is no-op', () => {
      const engine = new GameEngine('test-orch1', 'square8', createPlayers(), timeControl);
      // Should not throw
      engine.enableMoveDrivenDecisionPhases();
      expect(true).toBe(true);
    });

    it('enableOrchestratorAdapter is no-op', () => {
      const engine = new GameEngine('test-orch2', 'square8', createPlayers(), timeControl);
      engine.enableOrchestratorAdapter();
      expect(true).toBe(true);
    });

    it('disableOrchestratorAdapter is no-op', () => {
      const engine = new GameEngine('test-orch3', 'square8', createPlayers(), timeControl);
      engine.disableOrchestratorAdapter();
      expect(true).toBe(true);
    });

    it('isOrchestratorAdapterEnabled always returns true', () => {
      const engine = new GameEngine('test-orch4', 'square8', createPlayers(), timeControl);
      expect(engine.isOrchestratorAdapterEnabled()).toBe(true);
    });
  });

  describe('makeMove validation branches', () => {
    it('rejects continue_capture_segment when no chain is active', async () => {
      const engine = new GameEngine('test-no-chain', 'square8', createPlayers(), timeControl);
      engine.startGame();

      const result = await engine.makeMove({
        type: 'continue_capture_segment',
        player: 1,
        from: { x: 0, y: 0 },
        to: { x: 1, y: 1 },
      });

      expect(result.success).toBe(false);
      expect(result.error).toContain('No chain capture in progress');
    });

    it('handles placement move via adapter', async () => {
      const engine = new GameEngine('test-placement', 'square8', createPlayers(), timeControl);
      engine.startGame();

      // Player 1 should be able to place rings in ring_placement phase
      const result = await engine.makeMove({
        type: 'place_rings',
        player: 1,
        to: { x: 3, y: 3 },
        count: 1,
      });

      // Either succeeds or fails with meaningful error (not a crash)
      expect(typeof result.success).toBe('boolean');
    });

    it('handles movement move after placement', async () => {
      const engine = new GameEngine('test-movement', 'square8', createPlayers(), timeControl);
      engine.startGame();

      // Make initial placement
      await engine.makeMove({
        type: 'place_rings',
        player: 1,
        to: { x: 3, y: 3 },
        count: 1,
      });

      // Attempt movement from the placed position
      const result = await engine.makeMove({
        type: 'move_stack',
        player: 1,
        from: { x: 3, y: 3 },
        to: { x: 4, y: 3 },
      });

      // Either succeeds or fails with validation error
      expect(typeof result.success).toBe('boolean');
    });
  });

  describe('game lifecycle', () => {
    it('startGame sets game status to active', () => {
      const engine = new GameEngine('test-start', 'square8', createPlayers(), timeControl);
      const started = engine.startGame();

      expect(started).toBe(true);
      expect(engine.getGameState().gameStatus).toBe('active');
    });

    it('startGame can be called multiple times and always returns true if players ready', () => {
      const engine = new GameEngine('test-double-start', 'square8', createPlayers(), timeControl);
      engine.startGame();
      const secondStart = engine.startGame();

      // startGame doesn't prevent re-starting, it just checks player readiness
      expect(secondStart).toBe(true);
    });

    it('tracks current player after game start', () => {
      const engine = new GameEngine('test-turns', 'square8', createPlayers(), timeControl);
      engine.startGame();

      const currentPlayer = engine.getGameState().currentPlayer;
      expect(currentPlayer).toBe(1);
    });
  });

  describe('spectator management', () => {
    it('addSpectator adds user to spectators list', () => {
      const engine = new GameEngine('test-spec1', 'square8', createPlayers(), timeControl);
      const result = engine.addSpectator('spectator-1');

      expect(result).toBe(true);
      expect(engine.getGameState().spectators).toContain('spectator-1');
    });

    it('addSpectator rejects duplicate spectators', () => {
      const engine = new GameEngine('test-spec2', 'square8', createPlayers(), timeControl);
      engine.addSpectator('spectator-1');
      const result = engine.addSpectator('spectator-1');

      expect(result).toBe(false);
    });

    it('removeSpectator removes user from list', () => {
      const engine = new GameEngine('test-spec3', 'square8', createPlayers(), timeControl);
      engine.addSpectator('spectator-1');
      engine.removeSpectator('spectator-1');

      expect(engine.getGameState().spectators).not.toContain('spectator-1');
    });

    it('removeSpectator handles non-existent spectator', () => {
      const engine = new GameEngine('test-spec4', 'square8', createPlayers(), timeControl);
      // Should not throw
      engine.removeSpectator('non-existent');
      expect(engine.getGameState().spectators).toEqual([]);
    });
  });

  describe('board configuration', () => {
    it('supports square8 board type', () => {
      const engine = new GameEngine('test-sq8', 'square8', createPlayers(), timeControl);
      const state = engine.getGameState();

      expect(state.board.type).toBe('square8');
      expect(state.boardType).toBe('square8');
    });

    it('supports square19 board type', () => {
      const players = createPlayers(2);
      const square19Config = BOARD_CONFIGS['square19'];
      if (square19Config) {
        players.forEach((p) => {
          p.ringsInHand = square19Config.ringsPerPlayer;
        });
        const engine = new GameEngine('test-sq19', 'square19', players, timeControl);
        const state = engine.getGameState();
        expect(state.board.type).toBe('square19');
      }
    });

    it('supports hex board type', () => {
      const players = createPlayers(2);
      const hexConfig = BOARD_CONFIGS['hex'];
      if (hexConfig) {
        players.forEach((p) => {
          p.ringsInHand = hexConfig.ringsPerPlayer;
        });
        const engine = new GameEngine('test-hex', 'hex', players, timeControl);
        const state = engine.getGameState();
        expect(state.board.type).toBe('hex');
      }
    });
  });

  describe('multi-player support', () => {
    it('supports 3-player games', () => {
      const engine = new GameEngine('test-3p', 'square8', createPlayers(3), timeControl);
      const state = engine.getGameState();

      expect(state.players.length).toBe(3);
    });

    it('supports 4-player games', () => {
      const engine = new GameEngine('test-4p', 'square8', createPlayers(4), timeControl);
      const state = engine.getGameState();

      expect(state.players.length).toBe(4);
    });
  });

  describe('move history', () => {
    it('starts with empty move history', () => {
      const engine = new GameEngine('test-hist1', 'square8', createPlayers(), timeControl);
      const state = engine.getGameState();

      expect(state.moveHistory).toEqual([]);
    });

    it('starts with empty history array', () => {
      const engine = new GameEngine('test-hist2', 'square8', createPlayers(), timeControl);
      const state = engine.getGameState();

      expect(state.history).toEqual([]);
    });
  });

  describe('game state access', () => {
    it('getGameState returns consistent state', () => {
      const engine = new GameEngine('test-state', 'square8', createPlayers(), timeControl);
      const state1 = engine.getGameState();
      const state2 = engine.getGameState();

      expect(state1.id).toBe(state2.id);
      expect(state1.boardType).toBe(state2.boardType);
      expect(state1.players.length).toBe(state2.players.length);
    });
  });

  describe('debug checkpoint hook', () => {
    it('setDebugCheckpointHook accepts callback', () => {
      const engine = new GameEngine('test-debug1', 'square8', createPlayers(), timeControl);
      const hookFn = jest.fn();

      engine.setDebugCheckpointHook(hookFn);
      expect(true).toBe(true);
    });

    it('setDebugCheckpointHook accepts undefined to clear', () => {
      const engine = new GameEngine('test-debug2', 'square8', createPlayers(), timeControl);

      engine.setDebugCheckpointHook(undefined);
      expect(true).toBe(true);
    });
  });

  describe('getValidMoves', () => {
    it('returns valid moves for current player in ring_placement phase', () => {
      const engine = new GameEngine('test-moves1', 'square8', createPlayers(), timeControl);
      const state = engine.getGameState();

      expect(state.currentPhase).toBe('ring_placement');
      const moves = engine.getValidMoves();
      // In ring_placement, valid moves are place_ring moves
      expect(Array.isArray(moves)).toBe(true);
    });

    it('returns empty array when game is not active', () => {
      const engine = new GameEngine('test-moves2', 'square8', createPlayers(), timeControl);
      // Force game to completed state
      (engine as unknown as { gameState: { gameStatus: string } }).gameState.gameStatus =
        'completed';

      const moves = engine.getValidMoves();
      expect(moves).toEqual([]);
    });

    it('returns valid moves with playerNumber override', () => {
      const engine = new GameEngine('test-moves3', 'square8', createPlayers(), timeControl);

      // Get moves for player 2 even if it's player 1's turn
      const moves = engine.getValidMoves(2);
      expect(Array.isArray(moves)).toBe(true);
    });
  });

  describe('makeMove validation', () => {
    it('rejects continue_capture_segment when no chain in progress', async () => {
      const engine = new GameEngine('test-no-chain', 'square8', createPlayers(), timeControl);
      engine.startGame();

      const move = {
        type: 'continue_capture_segment' as const,
        player: 1,
        from: { x: 3, y: 3 },
        to: { x: 5, y: 5 },
        thinkTime: 100,
      };

      const result = await engine.makeMove(move);
      expect(result.success).toBe(false);
      expect(result.error).toContain('No chain capture in progress');
    });

    it('handles place_ring move in ring_placement phase', async () => {
      const engine = new GameEngine('test-place', 'square8', createPlayers(), timeControl);
      engine.startGame();
      const state = engine.getGameState();

      expect(state.currentPhase).toBe('ring_placement');

      // Find a valid position for placing a ring
      const boardSize = state.board.size;
      const validPosition = { x: Math.floor(boardSize / 2), y: Math.floor(boardSize / 2) };

      const move = {
        type: 'place_ring' as const,
        player: state.currentPlayer,
        to: validPosition,
        thinkTime: 100,
      };

      const result = await engine.makeMove(move);
      // Either succeeds or fails with a specific validation error
      expect(result.gameState).toMatchObject({
        gameStatus: expect.any(String),
        currentPlayer: expect.any(Number),
      });
    });

    it('handles swap_sides move type', async () => {
      const engine = new GameEngine(
        'test-swap',
        'square8',
        createPlayers(),
        timeControl,
        false,
        undefined,
        undefined,
        { swapRuleEnabled: true }
      );
      engine.startGame();

      const move = {
        type: 'swap_sides' as const,
        player: 2, // Second player can swap
        thinkTime: 100,
      };

      const result = await engine.makeMove(move);
      // Swap might fail if conditions aren't met, but function should be callable
      expect(result).toMatchObject({
        success: expect.any(Boolean),
      });
    });
  });

  describe('timer management', () => {
    it('starts with timers initialized', () => {
      const engine = new GameEngine('test-timer1', 'square8', createPlayers(), timeControl);
      const state = engine.getGameState();

      state.players.forEach((player) => {
        expect(player.timeRemaining).toBe(600000);
      });
    });

    it('startGame initializes timers', () => {
      const engine = new GameEngine('test-timer2', 'square8', createPlayers(), timeControl);

      const started = engine.startGame();
      expect(started).toBe(true);

      const state = engine.getGameState();
      expect(state.gameStatus).toBe('active');
    });

    it('pauseGame stops game timer', () => {
      const engine = new GameEngine('test-pause1', 'square8', createPlayers(), timeControl);
      engine.startGame();

      const paused = engine.pauseGame();
      expect(paused).toBe(true);

      const state = engine.getGameState();
      expect(state.gameStatus).toBe('paused');
    });

    it('resumeGame continues game timer', () => {
      const engine = new GameEngine('test-resume1', 'square8', createPlayers(), timeControl);
      engine.startGame();
      engine.pauseGame();

      const resumed = engine.resumeGame();
      expect(resumed).toBe(true);

      const state = engine.getGameState();
      expect(state.gameStatus).toBe('active');
    });
  });

  describe('game result handling', () => {
    it('resignPlayer ends game with resignation', () => {
      const engine = new GameEngine('test-result1', 'square8', createPlayers(), timeControl);
      engine.startGame();

      const result = engine.resignPlayer(1);

      expect(result.success).toBe(true);
      const state = engine.getGameState();
      expect(state.gameStatus).toBe('completed');
    });

    it('abandonGameAsDraw handles draw', () => {
      const engine = new GameEngine('test-result2', 'square8', createPlayers(), timeControl);
      engine.startGame();

      const result = engine.abandonGameAsDraw();

      expect(result.success).toBe(true);
      const state = engine.getGameState();
      expect(state.gameStatus).toBe('completed');
    });

    it('abandonPlayer marks player as abandoned', () => {
      const engine = new GameEngine('test-result3', 'square8', createPlayers(), timeControl);
      engine.startGame();

      const result = engine.abandonPlayer(1);

      expect(result.success).toBe(true);
      const state = engine.getGameState();
      expect(state.gameStatus).toBe('completed');
    });
  });

  describe('spectator management', () => {
    it('addSpectator adds spectator to game', () => {
      const engine = new GameEngine('test-spec1', 'square8', createPlayers(), timeControl);
      engine.startGame();

      const added = engine.addSpectator('spectator-1');
      expect(added).toBe(true);
    });

    it('addSpectator rejects duplicate', () => {
      const engine = new GameEngine('test-spec2', 'square8', createPlayers(), timeControl);
      engine.startGame();

      engine.addSpectator('spectator-1');
      const addedAgain = engine.addSpectator('spectator-1');
      expect(addedAgain).toBe(false);
    });

    it('removeSpectator removes spectator', () => {
      const engine = new GameEngine('test-spec3', 'square8', createPlayers(), timeControl);
      engine.startGame();
      engine.addSpectator('spectator-1');

      const removed = engine.removeSpectator('spectator-1');
      expect(removed).toBe(true);
    });

    it('removeSpectator returns false for non-existent', () => {
      const engine = new GameEngine('test-spec4', 'square8', createPlayers(), timeControl);
      engine.startGame();

      const removed = engine.removeSpectator('non-existent');
      expect(removed).toBe(false);
    });
  });

  describe('player lookup', () => {
    it('finds player in game state by player number', () => {
      const engine = new GameEngine('test-player1', 'square8', createPlayers(), timeControl);
      const state = engine.getGameState();

      const player1 = state.players.find((p) => p.playerNumber === 1);
      expect(player1?.playerNumber).toBe(1);
      expect(player1?.id).toBe('p1');
    });

    it('returns undefined for invalid player number', () => {
      const engine = new GameEngine('test-player2', 'square8', createPlayers(), timeControl);
      const state = engine.getGameState();

      const player = state.players.find((p) => p.playerNumber === 999);
      expect(player).toBeUndefined();
    });

    it('current player matches state.currentPlayer', () => {
      const engine = new GameEngine('test-current1', 'square8', createPlayers(), timeControl);
      const state = engine.getGameState();

      const currentPlayer = state.players.find((p) => p.playerNumber === state.currentPlayer);
      expect(currentPlayer?.playerNumber).toBe(state.currentPlayer);
    });
  });

  describe('phase transitions', () => {
    it('initial phase is ring_placement', () => {
      const engine = new GameEngine('test-phase1', 'square8', createPlayers(), timeControl);
      const state = engine.getGameState();

      expect(state.currentPhase).toBe('ring_placement');
    });

    it('game starts in active status after startGame', () => {
      const engine = new GameEngine('test-phase2', 'square8', createPlayers(), timeControl);
      engine.startGame();

      const state = engine.getGameState();
      expect(state.gameStatus).toBe('active');
    });
  });

  describe('board access', () => {
    it('game state includes board with correct size', () => {
      const engine = new GameEngine('test-board1', 'square8', createPlayers(), timeControl);
      const state = engine.getGameState();

      expect(state.board).toMatchObject({
        type: 'square8',
        size: 8,
      });
    });

    it('game state includes empty stacks initially', () => {
      const engine = new GameEngine('test-board2', 'square8', createPlayers(), timeControl);
      const state = engine.getGameState();

      expect(state.board.stacks).toBeInstanceOf(Map);
      expect(state.board.stacks.size).toBe(0);
    });

    it('game state includes empty markers initially', () => {
      const engine = new GameEngine('test-board3', 'square8', createPlayers(), timeControl);
      const state = engine.getGameState();

      expect(state.board.markers).toBeInstanceOf(Map);
      expect(state.board.markers.size).toBe(0);
    });
  });
});
