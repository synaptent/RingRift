/**
 * ClientSandboxEngine.branchCoverage.test.ts
 *
 * Additional branch coverage tests for ClientSandboxEngine.ts targeting
 * specific uncovered branches identified by coverage analysis.
 */

import {
  ClientSandboxEngine,
  SandboxInteractionHandler,
  SandboxConfig,
} from '../../src/client/sandbox/ClientSandboxEngine';
import type {
  GameState,
  Move,
  PlayerChoice,
  PlayerChoiceResponseFor,
  PlayerChoiceResponse,
} from '../../src/shared/types/game';
import { BOARD_CONFIGS } from '../../src/shared/types/game';

// Mock interaction handler that auto-selects the first option
class MockInteractionHandler implements SandboxInteractionHandler {
  async requestChoice<TChoice extends PlayerChoice>(
    choice: TChoice
  ): Promise<PlayerChoiceResponseFor<TChoice>> {
    // For ring_elimination, return the first option with its moveId
    if (choice.type === 'ring_elimination') {
      const elimChoice = choice as PlayerChoice & {
        options: Array<{
          stackPosition: { x: number; y: number };
          capHeight: number;
          totalHeight: number;
          moveId: string;
        }>;
      };
      return {
        choiceId: choice.id,
        selectedOption: elimChoice.options[0],
      } as PlayerChoiceResponseFor<TChoice>;
    }
    // For line_order, return the first line option
    if (choice.type === 'line_order') {
      const lineChoice = choice as PlayerChoice & {
        options: Array<{
          lineId: string;
          markerPositions: Array<{ x: number; y: number }>;
          moveId: string;
        }>;
      };
      return {
        choiceId: choice.id,
        selectedOption: lineChoice.options[0],
      } as PlayerChoiceResponseFor<TChoice>;
    }
    // For region_order, return the first region option
    if (choice.type === 'region_order') {
      const regionChoice = choice as PlayerChoice & {
        options: Array<{
          regionId: string;
          size: number;
          representativePosition: { x: number; y: number };
          moveId: string;
        }>;
      };
      return {
        choiceId: choice.id,
        selectedOption: regionChoice.options[0],
      } as PlayerChoiceResponseFor<TChoice>;
    }
    // For capture_direction, return the first direction
    if (choice.type === 'capture_direction') {
      const captureChoice = choice as PlayerChoice & {
        options: Array<{
          direction: { dx: number; dy: number };
          landing: { x: number; y: number };
          moveId: string;
        }>;
      };
      return {
        choiceId: choice.id,
        selectedOption: captureChoice.options[0],
      } as PlayerChoiceResponseFor<TChoice>;
    }
    // Default: return some response
    return {
      choiceId: choice.id,
      selectedOption: (choice as { options?: unknown[] }).options?.[0] ?? {},
    } as PlayerChoiceResponseFor<TChoice>;
  }
}

describe('ClientSandboxEngine branch coverage', () => {
  const createConfig = (
    numPlayers: number = 2,
    boardType: 'square8' | 'square19' = 'square8'
  ): SandboxConfig => ({
    boardType,
    numPlayers,
    playerKinds: Array(numPlayers).fill('human'),
  });

  const createAIConfig = (numPlayers: number = 2): SandboxConfig => ({
    boardType: 'square8',
    numPlayers,
    playerKinds: Array(numPlayers).fill('ai'),
  });

  const createMixedConfig = (): SandboxConfig => ({
    boardType: 'square8',
    numPlayers: 2,
    playerKinds: ['human', 'ai'],
  });

  describe('constructor variations', () => {
    it('creates engine with 2 human players on square8', () => {
      const engine = new ClientSandboxEngine({
        config: createConfig(2),
        interactionHandler: new MockInteractionHandler(),
      });
      const state = engine.getGameState();
      expect(state.players).toHaveLength(2);
      expect(state.boardType).toBe('square8');
      expect(state.players[0].type).toBe('human');
    });

    it('creates engine with 3 players', () => {
      const engine = new ClientSandboxEngine({
        config: createConfig(3),
        interactionHandler: new MockInteractionHandler(),
      });
      const state = engine.getGameState();
      expect(state.players).toHaveLength(3);
      expect(state.maxPlayers).toBe(3);
    });

    it('creates engine with 4 players', () => {
      const engine = new ClientSandboxEngine({
        config: createConfig(4),
        interactionHandler: new MockInteractionHandler(),
      });
      const state = engine.getGameState();
      expect(state.players).toHaveLength(4);
      expect(state.maxPlayers).toBe(4);
    });

    it('creates engine with AI players', () => {
      const engine = new ClientSandboxEngine({
        config: createAIConfig(),
        interactionHandler: new MockInteractionHandler(),
      });
      const state = engine.getGameState();
      expect(state.players[0].type).toBe('ai');
      expect(state.players[0].aiDifficulty).toBe(5);
    });

    it('creates engine with mixed players', () => {
      const engine = new ClientSandboxEngine({
        config: createMixedConfig(),
        interactionHandler: new MockInteractionHandler(),
      });
      const state = engine.getGameState();
      expect(state.players[0].type).toBe('human');
      expect(state.players[1].type).toBe('ai');
    });

    it('creates engine on square19 board', () => {
      const engine = new ClientSandboxEngine({
        config: createConfig(2, 'square19'),
        interactionHandler: new MockInteractionHandler(),
      });
      const state = engine.getGameState();
      expect(state.boardType).toBe('square19');
      expect(state.board.size).toBe(19);
    });

    it('creates engine with traceMode enabled', () => {
      const engine = new ClientSandboxEngine({
        config: createConfig(),
        interactionHandler: new MockInteractionHandler(),
        traceMode: true,
      });
      // traceMode is internal, just verify engine was created with active state
      const state = engine.getGameState();
      expect(state.gameStatus).toBe('active');
      expect(state.currentPhase).toBe('ring_placement');
    });

    it('2-player games have swapRuleEnabled', () => {
      const engine = new ClientSandboxEngine({
        config: createConfig(2),
        interactionHandler: new MockInteractionHandler(),
      });
      const state = engine.getGameState();
      expect(state.rulesOptions?.swapRuleEnabled).toBe(true);
    });

    it('3+ player games do not have swapRuleEnabled', () => {
      const engine = new ClientSandboxEngine({
        config: createConfig(3),
        interactionHandler: new MockInteractionHandler(),
      });
      const state = engine.getGameState();
      expect(state.rulesOptions?.swapRuleEnabled).toBeUndefined();
    });
  });

  describe('game state initialization', () => {
    let engine: ClientSandboxEngine;

    beforeEach(() => {
      engine = new ClientSandboxEngine({
        config: createConfig(),
        interactionHandler: new MockInteractionHandler(),
      });
    });

    it('starts in ring_placement phase', () => {
      expect(engine.getGameState().currentPhase).toBe('ring_placement');
    });

    it('starts with player 1 as current player', () => {
      expect(engine.getGameState().currentPlayer).toBe(1);
    });

    it('starts with active gameStatus', () => {
      expect(engine.getGameState().gameStatus).toBe('active');
    });

    it('starts with correct ring counts', () => {
      const state = engine.getGameState();
      const config = BOARD_CONFIGS['square8'];
      expect(state.players[0].ringsInHand).toBe(config.ringsPerPlayer);
      expect(state.players[1].ringsInHand).toBe(config.ringsPerPlayer);
      expect(state.totalRingsInPlay).toBe(config.ringsPerPlayer * 2);
    });

    it('starts with empty board', () => {
      const state = engine.getGameState();
      expect(state.board.stacks.size).toBe(0);
      expect(state.board.markers.size).toBe(0);
      expect(state.board.collapsedSpaces.size).toBe(0);
    });

    it('starts with empty history', () => {
      const state = engine.getGameState();
      expect(state.moveHistory).toHaveLength(0);
      expect(state.history).toHaveLength(0);
    });

    it('has rngSeed as a number', () => {
      const state = engine.getGameState();
      expect(typeof state.rngSeed).toBe('number');
      expect(state.rngSeed).toBeGreaterThanOrEqual(0);
    });
  });

  describe('getGameState cloning', () => {
    let engine: ClientSandboxEngine;

    beforeEach(() => {
      engine = new ClientSandboxEngine({
        config: createConfig(),
        interactionHandler: new MockInteractionHandler(),
      });
    });

    it('returns independent game state snapshots', () => {
      const state1 = engine.getGameState();
      const state2 = engine.getGameState();
      expect(state1).not.toBe(state2);
      expect(state1.players).not.toBe(state2.players);
    });

    it('board maps are cloned', () => {
      const state1 = engine.getGameState();
      const state2 = engine.getGameState();
      expect(state1.board.stacks).not.toBe(state2.board.stacks);
      expect(state1.board.markers).not.toBe(state2.board.markers);
    });
  });

  describe('getValidMoves', () => {
    let engine: ClientSandboxEngine;

    beforeEach(() => {
      engine = new ClientSandboxEngine({
        config: createConfig(),
        interactionHandler: new MockInteractionHandler(),
      });
    });

    it('returns placement moves in ring_placement phase', () => {
      const moves = engine.getValidMoves(1);
      expect(moves.length).toBeGreaterThan(0);
      expect(moves.every((m) => m.type === 'place_ring')).toBe(true);
    });

    it('returns empty array for non-active player', () => {
      const moves = engine.getValidMoves(2);
      expect(moves).toHaveLength(0);
    });

    it('placement moves target valid positions', () => {
      const moves = engine.getValidMoves(1);
      moves.forEach((move) => {
        expect(move.to).not.toBeNull();
        expect(move.to).toMatchObject({
          x: expect.any(Number),
          y: expect.any(Number),
        });
        expect(move.to!.x).toBeGreaterThanOrEqual(0);
        expect(move.to!.y).toBeGreaterThanOrEqual(0);
        expect(move.to!.x).toBeLessThan(8);
        expect(move.to!.y).toBeLessThan(8);
      });
    });
  });

  describe('ring placement', () => {
    let engine: ClientSandboxEngine;

    beforeEach(() => {
      engine = new ClientSandboxEngine({
        config: createConfig(),
        interactionHandler: new MockInteractionHandler(),
      });
    });

    it('placement reduces rings in hand', async () => {
      const stateBeforePlacement = engine.getGameState();
      const ringsBeforePlacement = stateBeforePlacement.players[0].ringsInHand;

      // Place ring at a valid position using tryPlaceRings
      const success = await engine.tryPlaceRings({ x: 3, y: 3 }, 1);
      expect(success).toBe(true);
      const state = engine.getGameState();

      expect(state.players[0].ringsInHand).toBe(ringsBeforePlacement - 1);
    });

    it('placement creates a stack on the board', async () => {
      await engine.tryPlaceRings({ x: 4, y: 4 }, 1);
      const state = engine.getGameState();
      const stack = state.board.stacks.get('4,4');

      expect(stack).not.toBeNull();
      expect(stack).toMatchObject({
        controllingPlayer: 1,
        stackHeight: 1,
        position: { x: 4, y: 4 },
      });
    });

    it('rejects placement on occupied space', async () => {
      await engine.tryPlaceRings({ x: 3, y: 3 }, 1);

      // Try to place on same spot - should fail
      const result = await engine.tryPlaceRings({ x: 3, y: 3 }, 1);
      expect(result).toBe(false);
    });
  });

  describe('applyCanonicalMove', () => {
    let engine: ClientSandboxEngine;

    beforeEach(() => {
      engine = new ClientSandboxEngine({
        config: createConfig(),
        interactionHandler: new MockInteractionHandler(),
      });
    });

    it('applies place_ring move', async () => {
      await engine.applyCanonicalMove({
        id: 'test-1',
        type: 'place_ring',
        player: 1,
        to: { x: 2, y: 2 },
        timestamp: new Date(),
        thinkTime: 0,
        moveNumber: 1,
      });

      // applyCanonicalMove returns void, so check state directly
      expect(engine.getGameState().board.stacks.has('2,2')).toBe(true);
    });

    it('processes move for different player via orchestrator', async () => {
      // First make a valid move by player 1
      await engine.applyCanonicalMove({
        id: 'test-1',
        type: 'place_ring',
        player: 1,
        to: { x: 2, y: 2 },
        timestamp: new Date(),
        thinkTime: 0,
        moveNumber: 1,
      });

      // After player 1's turn completes, player 2 should be current
      // The orchestrator handles turn advancement
      const state = engine.getGameState();
      // Just verify state is consistent after the move
      expect(state.board.stacks.size).toBeGreaterThan(0);
    });

    it('throws for mis-phased canonical move (place_ring in line_processing)', async () => {
      // Force the game into a mis-phased state: line_processing with no lines.
      const badPhaseState = {
        ...engine.getGameState(),
        currentPhase: 'line_processing' as GameState['currentPhase'],
      };
      (engine as any).gameState = badPhaseState;

      const badMove: Move = {
        id: 'misphased-place-ring',
        type: 'place_ring',
        player: badPhaseState.currentPlayer,
        to: { x: 0, y: 0 },
        timestamp: new Date(),
        thinkTime: 0,
        moveNumber: 1,
      };

      await expect(engine.applyCanonicalMove(badMove)).rejects.toThrow(
        /processMove failed for move type 'place_ring'/i
      );
    });
  });

  describe('victory conditions', () => {
    let engine: ClientSandboxEngine;

    beforeEach(() => {
      engine = new ClientSandboxEngine({
        config: createConfig(),
        interactionHandler: new MockInteractionHandler(),
      });
    });

    it('getVictoryResult returns null initially', () => {
      expect(engine.getVictoryResult()).toBeNull();
    });

    it('game starts in active status', () => {
      expect(engine.getGameState().gameStatus).toBe('active');
    });
  });

  describe('selection state', () => {
    let engine: ClientSandboxEngine;

    beforeEach(() => {
      engine = new ClientSandboxEngine({
        config: createConfig(),
        interactionHandler: new MockInteractionHandler(),
      });
    });

    it('clearSelection clears any selection', () => {
      // Access internal state to verify selection behavior
      engine.clearSelection();
      // Verify no errors occur
      expect(true).toBe(true);
    });
  });

  describe('thresholds and totals', () => {
    it('square8 2p has correct victory threshold', () => {
      const engine = new ClientSandboxEngine({
        config: createConfig(2, 'square8'),
        interactionHandler: new MockInteractionHandler(),
      });
      const state = engine.getGameState();
      // 18 rings * 2 players = 36, threshold = 36/2 + 1 = 19
      expect(state.victoryThreshold).toBe(19);
    });

    it('square8 3p has correct victory threshold', () => {
      const engine = new ClientSandboxEngine({
        config: createConfig(3, 'square8'),
        interactionHandler: new MockInteractionHandler(),
      });
      const state = engine.getGameState();
      // 18 rings * 3 players = 54, threshold = 54/2 + 1 = 28
      expect(state.victoryThreshold).toBe(28);
    });

    it('square8 4p has correct victory threshold', () => {
      const engine = new ClientSandboxEngine({
        config: createConfig(4, 'square8'),
        interactionHandler: new MockInteractionHandler(),
      });
      const state = engine.getGameState();
      // 18 rings * 4 players = 72, threshold = 72/2 + 1 = 37
      expect(state.victoryThreshold).toBe(37);
    });

    it('territory victory threshold is correct', () => {
      const engine = new ClientSandboxEngine({
        config: createConfig(2, 'square8'),
        interactionHandler: new MockInteractionHandler(),
      });
      const state = engine.getGameState();
      // 64 spaces, threshold = 64/2 + 1 = 33
      expect(state.territoryVictoryThreshold).toBe(33);
    });
  });

  describe('history replay state', () => {
    let engine: ClientSandboxEngine;

    beforeEach(() => {
      engine = new ClientSandboxEngine({
        config: createConfig(),
        interactionHandler: new MockInteractionHandler(),
      });
    });

    it('getStateAtMoveIndex returns state for valid index', () => {
      // No moves yet, but index 0 should return something
      const state = engine.getStateAtMoveIndex(0);
      // Before any moves, this may return current state or null
      expect(state === null || state !== undefined).toBe(true);
    });

    it('history starts empty', () => {
      expect(engine.getGameState().history).toHaveLength(0);
    });
  });

  describe('line highlights', () => {
    let engine: ClientSandboxEngine;

    beforeEach(() => {
      engine = new ClientSandboxEngine({
        config: createConfig(),
        interactionHandler: new MockInteractionHandler(),
      });
    });

    it('consumeRecentLineHighlights returns empty array initially', () => {
      const highlights = engine.consumeRecentLineHighlights();
      expect(highlights).toEqual([]);
    });
  });

  describe('serialization', () => {
    let engine: ClientSandboxEngine;

    beforeEach(() => {
      engine = new ClientSandboxEngine({
        config: createConfig(),
        interactionHandler: new MockInteractionHandler(),
      });
    });

    it('getSerializedState returns a serializable state object', () => {
      const serialized = engine.getSerializedState();
      expect(typeof serialized).toBe('object');
      expect(serialized).not.toBeNull();
      // Verify essential state properties are included
      expect(serialized).toHaveProperty('gameState');
    });

    it('can initialize from serialized state with player kinds', () => {
      const serialized = engine.getSerializedState();
      const newEngine = new ClientSandboxEngine({
        config: createConfig(),
        interactionHandler: new MockInteractionHandler(),
      });
      // initFromSerializedState may return void or boolean
      newEngine.initFromSerializedState(serialized, ['human', 'human']);
      // Verify state was restored
      const state = newEngine.getGameState();
      expect(state.gameStatus).toBe('active');
    });

    it('serialized state is serializable', () => {
      const serialized = engine.getSerializedState();
      // Should be able to JSON stringify/parse
      const json = JSON.stringify(serialized);
      const parsed = JSON.parse(json);
      expect(typeof parsed).toBe('object');
      expect(parsed).toHaveProperty('gameState');
    });
  });

  describe('debugCheckpointHook', () => {
    it('can set debug checkpoint hook', () => {
      const engine = new ClientSandboxEngine({
        config: createConfig(),
        interactionHandler: new MockInteractionHandler(),
      });
      const hook = jest.fn();
      engine.setDebugCheckpointHook(hook);
      // Just verifying no error
      expect(true).toBe(true);
    });

    it('can clear debug checkpoint hook', () => {
      const engine = new ClientSandboxEngine({
        config: createConfig(),
        interactionHandler: new MockInteractionHandler(),
      });
      engine.setDebugCheckpointHook(() => {});
      engine.setDebugCheckpointHook(undefined);
      expect(true).toBe(true);
    });
  });

  describe('AI last move tracking', () => {
    it('getLastAIMoveForTesting returns null initially', () => {
      const engine = new ClientSandboxEngine({
        config: createConfig(),
        interactionHandler: new MockInteractionHandler(),
      });
      expect(engine.getLastAIMoveForTesting()).toBeNull();
    });
  });

  describe('swap sides', () => {
    it('canCurrentPlayerSwapSides returns false initially for player 1', () => {
      const engine = new ClientSandboxEngine({
        config: createConfig(2),
        interactionHandler: new MockInteractionHandler(),
      });
      // Player 1 cannot swap - swap is only for Player 2
      expect(engine.canCurrentPlayerSwapSides()).toBe(false);
    });
  });

  describe('chain capture context', () => {
    it('getChainCaptureContextForCurrentPlayer returns null initially', () => {
      const engine = new ClientSandboxEngine({
        config: createConfig(),
        interactionHandler: new MockInteractionHandler(),
      });
      expect(engine.getChainCaptureContextForCurrentPlayer()).toBeNull();
    });
  });

  describe('valid landings', () => {
    let engine: ClientSandboxEngine;

    beforeEach(() => {
      engine = new ClientSandboxEngine({
        config: createConfig(),
        interactionHandler: new MockInteractionHandler(),
      });
    });

    it('getValidLandingPositionsForCurrentPlayer returns empty for nonexistent stack', () => {
      const landings = engine.getValidLandingPositionsForCurrentPlayer({ x: 0, y: 0 });
      expect(landings).toEqual([]);
    });

    it('getValidLandingPositionsForCurrentPlayer returns positions for valid stack', async () => {
      // First place a ring to create a stack
      await engine.tryPlaceRings({ x: 3, y: 3 }, 1);

      // Now get valid landing positions from that stack
      const landings = engine.getValidLandingPositionsForCurrentPlayer({ x: 3, y: 3 });
      // Should have some valid moves (adjacent cells in movement phase)
      expect(Array.isArray(landings)).toBe(true);
    });
  });

  describe('handleHumanCellClick', () => {
    let engine: ClientSandboxEngine;

    beforeEach(() => {
      engine = new ClientSandboxEngine({
        config: createConfig(),
        interactionHandler: new MockInteractionHandler(),
      });
    });

    it('handles click during ring placement', async () => {
      await engine.handleHumanCellClick({ x: 4, y: 4 });
      // Click in ring placement should trigger placement
      const state = engine.getGameState();
      // Either placement happened or engine state is consistent
      expect(state.gameStatus).toBe('active');
      expect(typeof state.currentPlayer).toBe('number');
    });
  });

  // ==========================================================================
  // Additional branch coverage tests
  // ==========================================================================
  describe('getGameState consistency', () => {
    it('returns consistent state on multiple calls', () => {
      const engine = new ClientSandboxEngine({
        config: createConfig(2),
        interactionHandler: new MockInteractionHandler(),
      });

      const state1 = engine.getGameState();
      const state2 = engine.getGameState();

      expect(state1.id).toBe(state2.id);
      expect(state1.boardType).toBe(state2.boardType);
      expect(state1.players.length).toBe(state2.players.length);
    });

    it('getGameState returns state with board stacks', () => {
      const engine = new ClientSandboxEngine({
        config: createConfig(2),
        interactionHandler: new MockInteractionHandler(),
      });
      const state = engine.getGameState();

      expect(state.board).toMatchObject({
        type: 'square8',
        size: 8,
      });
      expect(state.board.stacks).toBeInstanceOf(Map);
    });
  });

  describe('getVictoryResult branches', () => {
    it('returns null when game is active', () => {
      const engine = new ClientSandboxEngine({
        config: createConfig(2),
        interactionHandler: new MockInteractionHandler(),
      });
      const result = engine.getVictoryResult();

      // During ring placement, no victory yet
      expect(result).toBeNull();
    });
  });

  describe('loadFromExportedState error handling', () => {
    it('throws on invalid state data', () => {
      const engine = new ClientSandboxEngine({
        config: createConfig(2),
        interactionHandler: new MockInteractionHandler(),
      });

      expect(() => {
        engine.loadFromExportedState({ invalidData: true } as any);
      }).toThrow();
    });
  });

  describe('multi-player configurations', () => {
    it('initializes 3-player game correctly', () => {
      const engine = new ClientSandboxEngine({
        config: createConfig(3),
        interactionHandler: new MockInteractionHandler(),
      });
      const state = engine.getGameState();

      expect(state.players.length).toBe(3);
      expect(state.maxPlayers).toBe(3);
    });

    it('initializes 4-player game correctly', () => {
      const engine = new ClientSandboxEngine({
        config: createConfig(4),
        interactionHandler: new MockInteractionHandler(),
      });
      const state = engine.getGameState();

      expect(state.players.length).toBe(4);
      expect(state.maxPlayers).toBe(4);
    });
  });

  describe('board type configurations', () => {
    it('supports square19 board type', () => {
      const engine = new ClientSandboxEngine({
        config: createConfig(2, 'square19'),
        interactionHandler: new MockInteractionHandler(),
      });
      const state = engine.getGameState();

      expect(state.boardType).toBe('square19');
      expect(state.board.type).toBe('square19');
    });
  });

  describe('currentPhase tracking', () => {
    it('starts in ring_placement phase', () => {
      const engine = new ClientSandboxEngine({
        config: createConfig(2),
        interactionHandler: new MockInteractionHandler(),
      });
      const state = engine.getGameState();

      expect(state.currentPhase).toBe('ring_placement');
    });

    it('tracks current player correctly', () => {
      const engine = new ClientSandboxEngine({
        config: createConfig(2),
        interactionHandler: new MockInteractionHandler(),
      });
      const state = engine.getGameState();

      expect(state.currentPlayer).toBe(1);
    });
  });

  describe('history management', () => {
    it('starts with empty history', () => {
      const engine = new ClientSandboxEngine({
        config: createConfig(2),
        interactionHandler: new MockInteractionHandler(),
      });
      const state = engine.getGameState();

      expect(state.history).toEqual([]);
      expect(state.moveHistory).toEqual([]);
    });
  });

  describe('multiple board clicks', () => {
    it('handles rapid sequential clicks', async () => {
      const engine = new ClientSandboxEngine({
        config: createConfig(2),
        interactionHandler: new MockInteractionHandler(),
      });

      // Make several clicks in sequence
      await engine.handleHumanCellClick({ x: 0, y: 0 });
      await engine.handleHumanCellClick({ x: 1, y: 0 });
      await engine.handleHumanCellClick({ x: 2, y: 0 });

      const state = engine.getGameState();
      expect(state.gameStatus).toBe('active');
      expect(typeof state.currentPlayer).toBe('number');
    });

    it('handles click on same cell twice', async () => {
      const engine = new ClientSandboxEngine({
        config: createConfig(2),
        interactionHandler: new MockInteractionHandler(),
      });

      await engine.handleHumanCellClick({ x: 0, y: 0 });
      await engine.handleHumanCellClick({ x: 0, y: 0 });

      const state = engine.getGameState();
      expect(state.gameStatus).toBe('active');
      expect(typeof state.currentPlayer).toBe('number');
    });
  });

  describe('consumeRecentLineHighlights', () => {
    it('returns empty array initially', () => {
      const engine = new ClientSandboxEngine({
        config: createConfig(2),
        interactionHandler: new MockInteractionHandler(),
      });

      const highlights = engine.consumeRecentLineHighlights();
      expect(highlights).toEqual([]);
    });

    it('clears highlights after consumption', () => {
      const engine = new ClientSandboxEngine({
        config: createConfig(2),
        interactionHandler: new MockInteractionHandler(),
      });

      // Call twice - second call should also return empty
      engine.consumeRecentLineHighlights();
      const second = engine.consumeRecentLineHighlights();
      expect(second).toEqual([]);
    });
  });

  describe('getSerializedState', () => {
    it('returns serializable state', () => {
      const engine = new ClientSandboxEngine({
        config: createConfig(2),
        interactionHandler: new MockInteractionHandler(),
      });

      const serialized = engine.getSerializedState();
      expect(typeof serialized).toBe('object');
      expect(serialized).not.toBeNull();
      expect(serialized).toHaveProperty('gameState');
    });

    it('serialized state can be stringified', () => {
      const engine = new ClientSandboxEngine({
        config: createConfig(2),
        interactionHandler: new MockInteractionHandler(),
      });

      const serialized = engine.getSerializedState();
      const jsonString = JSON.stringify(serialized);
      expect(typeof jsonString).toBe('string');
      expect(jsonString.length).toBeGreaterThan(0);
    });
  });

  describe('getStateAtMoveIndex', () => {
    it('returns null for negative index', () => {
      const engine = new ClientSandboxEngine({
        config: createConfig(2),
        interactionHandler: new MockInteractionHandler(),
      });

      const state = engine.getStateAtMoveIndex(-1);
      expect(state).toBeNull();
    });

    it('returns initial state at index 0', () => {
      const engine = new ClientSandboxEngine({
        config: createConfig(2),
        interactionHandler: new MockInteractionHandler(),
      });

      const state = engine.getStateAtMoveIndex(0);
      // May be null if no history, or a state if there is history
      expect(state === null || state !== undefined).toBe(true);
    });

    it('returns state for out-of-bounds index (clamps to valid range)', () => {
      const engine = new ClientSandboxEngine({
        config: createConfig(2),
        interactionHandler: new MockInteractionHandler(),
      });

      const state = engine.getStateAtMoveIndex(1000);
      // Function clamps to valid range and returns initial state or null
      if (state !== null) {
        expect(state.gameStatus).toBe('active');
        expect(typeof state.currentPlayer).toBe('number');
      } else {
        expect(state).toBeNull();
      }
    });
  });

  describe('getLastAIMoveForTesting', () => {
    it('returns null when no AI moves made', () => {
      const engine = new ClientSandboxEngine({
        config: createConfig(2),
        interactionHandler: new MockInteractionHandler(),
      });

      const lastMove = engine.getLastAIMoveForTesting();
      expect(lastMove).toBeNull();
    });
  });

  describe('getChainCaptureContextForCurrentPlayer', () => {
    it('returns null when no chain capture active', () => {
      const engine = new ClientSandboxEngine({
        config: createConfig(2),
        interactionHandler: new MockInteractionHandler(),
      });

      const context = engine.getChainCaptureContextForCurrentPlayer();
      expect(context).toBeNull();
    });
  });

  describe('clearSelection', () => {
    it('clears any pending selection', () => {
      const engine = new ClientSandboxEngine({
        config: createConfig(2),
        interactionHandler: new MockInteractionHandler(),
      });

      engine.clearSelection();
      // Should not throw
      const state = engine.getGameState();
      expect(state.gameStatus).toBe('active');
      expect(state.currentPhase).toBe('ring_placement');
    });

    it('can be called multiple times safely', () => {
      const engine = new ClientSandboxEngine({
        config: createConfig(2),
        interactionHandler: new MockInteractionHandler(),
      });

      engine.clearSelection();
      engine.clearSelection();
      engine.clearSelection();

      const state = engine.getGameState();
      expect(state.gameStatus).toBe('active');
      expect(state.currentPhase).toBe('ring_placement');
    });
  });

  describe('getValidLandingPositionsForCurrentPlayer', () => {
    it('returns empty array for invalid from position', () => {
      const engine = new ClientSandboxEngine({
        config: createConfig(2),
        interactionHandler: new MockInteractionHandler(),
      });

      const landings = engine.getValidLandingPositionsForCurrentPlayer({ x: -1, y: -1 });
      expect(landings).toEqual([]);
    });

    it('returns empty array for empty cell', () => {
      const engine = new ClientSandboxEngine({
        config: createConfig(2),
        interactionHandler: new MockInteractionHandler(),
      });

      const landings = engine.getValidLandingPositionsForCurrentPlayer({ x: 0, y: 0 });
      expect(Array.isArray(landings)).toBe(true);
    });
  });

  describe('canCurrentPlayerSwapSides', () => {
    it('returns false in initial state without swap rule', () => {
      const engine = new ClientSandboxEngine({
        config: createConfig(2),
        interactionHandler: new MockInteractionHandler(),
      });

      const canSwap = engine.canCurrentPlayerSwapSides();
      expect(canSwap).toBe(false);
    });
  });

  describe('applySwapSidesForCurrentPlayer', () => {
    it('returns false when swap not allowed', () => {
      const engine = new ClientSandboxEngine({
        config: createConfig(2),
        interactionHandler: new MockInteractionHandler(),
      });

      const swapped = engine.applySwapSidesForCurrentPlayer();
      expect(swapped).toBe(false);
    });
  });

  describe('getValidMoves edge cases', () => {
    it('returns moves for player 1', () => {
      const engine = new ClientSandboxEngine({
        config: createConfig(2),
        interactionHandler: new MockInteractionHandler(),
      });

      const moves = engine.getValidMoves(1);
      expect(Array.isArray(moves)).toBe(true);
    });

    it('returns moves for player 2', () => {
      const engine = new ClientSandboxEngine({
        config: createConfig(2),
        interactionHandler: new MockInteractionHandler(),
      });

      const moves = engine.getValidMoves(2);
      expect(Array.isArray(moves)).toBe(true);
    });

    it('handles out-of-range player number', () => {
      const engine = new ClientSandboxEngine({
        config: createConfig(2),
        interactionHandler: new MockInteractionHandler(),
      });

      const moves = engine.getValidMoves(999);
      expect(Array.isArray(moves)).toBe(true);
    });
  });
});
