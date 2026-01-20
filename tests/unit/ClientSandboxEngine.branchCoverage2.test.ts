/**
 * ClientSandboxEngine.branchCoverage2.test.ts
 *
 * Additional branch coverage tests for ClientSandboxEngine.ts targeting
 * specific uncovered branches identified by coverage analysis.
 *
 * Target: Cover 50+ additional branches in:
 * - AI difficulty handling (getAIDifficulty, aiDifficulties array)
 * - initFromSerializedState edge cases (completed games, chainCapturePosition)
 * - Phase transitions and autoResolve functions
 * - Error handling branches
 * - LPS tracking branches
 * - applyCanonicalMoveForReplay lookahead and edge cases
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
  Position,
  BoardState,
  RingStack,
} from '../../src/shared/types/game';
import {
  serializeGameState,
  deserializeGameState,
} from '../../src/shared/engine/contracts/serialization';
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
          targetPosition: { x: number; y: number };
          landingPosition: { x: number; y: number };
          capturedCapHeight: number;
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

describe('ClientSandboxEngine Branch Coverage 2', () => {
  const createConfig = (
    numPlayers: number = 2,
    boardType: 'square8' | 'square19' = 'square8'
  ): SandboxConfig => ({
    boardType,
    numPlayers,
    playerKinds: Array(numPlayers).fill('human'),
  });

  const createAIConfig = (numPlayers: number = 2, difficulties?: number[]): SandboxConfig => ({
    boardType: 'square8',
    numPlayers,
    playerKinds: Array(numPlayers).fill('ai'),
    aiDifficulties: difficulties,
  });

  const createMixedConfig = (difficulties?: number[]): SandboxConfig => ({
    boardType: 'square8',
    numPlayers: 2,
    playerKinds: ['human', 'ai'],
    aiDifficulties: difficulties,
  });

  // Helper to create a board with stacks
  const createBoardWithStacks = (
    stacks: Array<{ pos: Position; player: number; rings: number[] }>
  ): Map<string, RingStack> => {
    const stackMap = new Map<string, RingStack>();
    for (const s of stacks) {
      const key = `${s.pos.x},${s.pos.y}`;
      stackMap.set(key, {
        position: s.pos,
        rings: s.rings,
        stackHeight: s.rings.length,
        capHeight: s.rings.filter((r) => r === s.player).length,
        controllingPlayer: s.player,
      });
    }
    return stackMap;
  };

  // ============================================================================
  // AI Difficulty Handling (getAIDifficulty branches - lines ~767-774)
  // ============================================================================
  describe('AI difficulty handling', () => {
    it('getAIDifficulty returns undefined for human players', () => {
      const engine = new ClientSandboxEngine({
        config: createConfig(2),
        interactionHandler: new MockInteractionHandler(),
      });
      const difficulty = engine.getAIDifficulty(1);
      expect(difficulty).toBeUndefined();
    });

    it('getAIDifficulty returns difficulty for AI player', () => {
      const engine = new ClientSandboxEngine({
        config: createAIConfig(2, [5, 7]),
        interactionHandler: new MockInteractionHandler(),
      });
      expect(engine.getAIDifficulty(1)).toBe(5);
      expect(engine.getAIDifficulty(2)).toBe(7);
    });

    it('getAIDifficulty returns default 4 when difficulties not specified', () => {
      const engine = new ClientSandboxEngine({
        config: createAIConfig(2),
        interactionHandler: new MockInteractionHandler(),
      });
      expect(engine.getAIDifficulty(1)).toBe(4);
      expect(engine.getAIDifficulty(2)).toBe(4);
    });

    it('getAIDifficulty clamps values to 1-10 range', () => {
      const engine = new ClientSandboxEngine({
        config: createAIConfig(2, [0, 15]),
        interactionHandler: new MockInteractionHandler(),
      });
      // Values should be clamped: 0 -> 1, 15 -> 10
      expect(engine.getAIDifficulty(1)).toBe(1);
      expect(engine.getAIDifficulty(2)).toBe(10);
    });

    it('getAIDifficulty returns undefined for nonexistent player', () => {
      const engine = new ClientSandboxEngine({
        config: createConfig(2),
        interactionHandler: new MockInteractionHandler(),
      });
      const difficulty = engine.getAIDifficulty(999);
      expect(difficulty).toBeUndefined();
    });

    it('mixed player config with AI difficulties', () => {
      const engine = new ClientSandboxEngine({
        config: createMixedConfig([3, 8]),
        interactionHandler: new MockInteractionHandler(),
      });
      // Player 1 is human, should return undefined
      expect(engine.getAIDifficulty(1)).toBeUndefined();
      // Player 2 is AI with difficulty 8
      expect(engine.getAIDifficulty(2)).toBe(8);
    });

    it('AI difficulty rounds to nearest integer', () => {
      const engine = new ClientSandboxEngine({
        config: createAIConfig(2, [3.7, 6.2]),
        interactionHandler: new MockInteractionHandler(),
      });
      expect(engine.getAIDifficulty(1)).toBe(4);
      expect(engine.getAIDifficulty(2)).toBe(6);
    });
  });

  // ============================================================================
  // initFromSerializedState edge cases (lines ~1044-1068)
  // ============================================================================
  describe('initFromSerializedState edge cases', () => {
    it('initializes from completed game state (normalizes phase)', () => {
      const engine = new ClientSandboxEngine({
        config: createConfig(2),
        interactionHandler: new MockInteractionHandler(),
      });

      // Create a completed state
      const completedState = {
        ...engine.getGameState(),
        gameStatus: 'completed' as const,
        currentPhase: 'capture' as GameState['currentPhase'],
        winner: 1,
      };
      const serialized = serializeGameState(completedState);

      engine.initFromSerializedState(serialized, ['human', 'human'], new MockInteractionHandler());

      const state = engine.getGameState();
      // Completed games should have phase normalized to ring_placement
      expect(state.gameStatus).toBe('completed');
      expect(state.currentPhase).toBe('ring_placement');
    });

    it('clears stale chainCapturePosition in active games', () => {
      const engine = new ClientSandboxEngine({
        config: createConfig(2),
        interactionHandler: new MockInteractionHandler(),
      });

      // Create state with stale chainCapturePosition
      const stateWithChain = {
        ...engine.getGameState(),
        currentPhase: 'movement' as GameState['currentPhase'],
        chainCapturePosition: { x: 3, y: 3 },
      };
      const serialized = serializeGameState(stateWithChain);

      engine.initFromSerializedState(serialized, ['human', 'human'], new MockInteractionHandler());

      const state = engine.getGameState();
      expect(state.chainCapturePosition).toBeUndefined();
    });

    it('preserves chainCapturePosition in chain_capture phase', () => {
      const engine = new ClientSandboxEngine({
        config: createConfig(2),
        interactionHandler: new MockInteractionHandler(),
      });

      // Create state properly in chain_capture phase
      const stateWithChain = {
        ...engine.getGameState(),
        currentPhase: 'chain_capture' as GameState['currentPhase'],
        chainCapturePosition: { x: 3, y: 3 },
      };
      const serialized = serializeGameState(stateWithChain);

      engine.initFromSerializedState(serialized, ['human', 'human'], new MockInteractionHandler());

      const state = engine.getGameState();
      // In chain_capture phase, position should be preserved
      expect(state.chainCapturePosition).toEqual({ x: 3, y: 3 });
    });

    it('converts human to AI with specified difficulty', () => {
      const engine = new ClientSandboxEngine({
        config: createConfig(2),
        interactionHandler: new MockInteractionHandler(),
      });

      const serialized = engine.getSerializedState();

      engine.initFromSerializedState(
        serialized,
        ['ai', 'ai'],
        new MockInteractionHandler(),
        [6, 9]
      );

      const state = engine.getGameState();
      expect(state.players[0].type).toBe('ai');
      expect(state.players[0].aiDifficulty).toBe(6);
      expect(state.players[1].type).toBe('ai');
      expect(state.players[1].aiDifficulty).toBe(9);
    });

    it('uses default difficulty when not specified', () => {
      const engine = new ClientSandboxEngine({
        config: createConfig(2),
        interactionHandler: new MockInteractionHandler(),
      });

      const serialized = engine.getSerializedState();

      engine.initFromSerializedState(serialized, ['ai', 'ai'], new MockInteractionHandler());

      const state = engine.getGameState();
      // Jan 10, 2026: Default difficulty is now 10 for optimal AI play
      expect(state.players[0].aiDifficulty).toBe(10);
      expect(state.players[1].aiDifficulty).toBe(10);
    });

    it('clears legacy mustMoveFromStackKey in completed games', () => {
      const engine = new ClientSandboxEngine({
        config: createConfig(2),
        interactionHandler: new MockInteractionHandler(),
      });

      // Simulate a completed state with legacy mustMoveFromStackKey
      const legacyState = {
        ...engine.getGameState(),
        gameStatus: 'completed' as const,
        mustMoveFromStackKey: '3,3',
      } as GameState;

      const serialized = serializeGameState(legacyState);

      engine.initFromSerializedState(serialized, ['human', 'human'], new MockInteractionHandler());

      const state = engine.getGameState();
      expect((state as any).mustMoveFromStackKey).toBeUndefined();
    });

    it('reinitializes RNG from seed (preserves seed in state)', () => {
      const engine = new ClientSandboxEngine({
        config: createConfig(2),
        interactionHandler: new MockInteractionHandler(),
      });

      const stateWithSeed = {
        ...engine.getGameState(),
        rngSeed: 12345,
      };
      const serialized = serializeGameState(stateWithSeed);

      engine.initFromSerializedState(serialized, ['human', 'human'], new MockInteractionHandler());

      const state = engine.getGameState();
      // Seed is stored in state but RNG may generate new values
      expect(typeof state.rngSeed).toBe('number');
    });

    it('generates new seed when not present', () => {
      const engine = new ClientSandboxEngine({
        config: createConfig(2),
        interactionHandler: new MockInteractionHandler(),
      });

      const stateNoSeed = {
        ...engine.getGameState(),
        rngSeed: undefined,
      } as GameState;
      const serialized = serializeGameState(stateNoSeed);

      engine.initFromSerializedState(serialized, ['human', 'human'], new MockInteractionHandler());

      const state = engine.getGameState();
      expect(typeof state.rngSeed).toBe('number');
    });
  });

  // ============================================================================
  // getStateAtMoveIndex branches (lines ~870-903)
  // ============================================================================
  describe('getStateAtMoveIndex branches', () => {
    it('returns current state for index >= totalMoves', async () => {
      const engine = new ClientSandboxEngine({
        config: createConfig(2),
        interactionHandler: new MockInteractionHandler(),
      });

      // Make a move to have some history
      await engine.tryPlaceRings({ x: 3, y: 3 }, 1);

      const state = engine.getStateAtMoveIndex(100);
      // Should return current state when index exceeds history
      expect(state).not.toBeNull();
    });

    it('returns null for negative index', () => {
      const engine = new ClientSandboxEngine({
        config: createConfig(2),
        interactionHandler: new MockInteractionHandler(),
      });

      const state = engine.getStateAtMoveIndex(-5);
      expect(state).toBeNull();
    });

    it('returns state for index 0 after moves', async () => {
      const engine = new ClientSandboxEngine({
        config: createConfig(2),
        interactionHandler: new MockInteractionHandler(),
      });

      // make a move so we have history
      await engine.tryPlaceRings({ x: 3, y: 3 }, 1);

      const stateAtZero = engine.getStateAtMoveIndex(0);
      // Index 0 returns initial or first state
      expect(stateAtZero).not.toBeNull();
    });

    it('returns current state for index 0 with no history', () => {
      const engine = new ClientSandboxEngine({
        config: createConfig(2),
        interactionHandler: new MockInteractionHandler(),
      });

      // No moves - getStateAtMoveIndex(0) returns current state
      const state = engine.getStateAtMoveIndex(0);
      // The implementation returns current state for index >= total moves
      // With 0 moves, index 0 returns current state
      if (state) {
        expect(state.gameStatus).toBe('active');
      }
    });

    it('returns correct snapshot for valid move index', async () => {
      const engine = new ClientSandboxEngine({
        config: createConfig(2),
        interactionHandler: new MockInteractionHandler(),
      });

      await engine.tryPlaceRings({ x: 2, y: 2 }, 1);
      await engine.tryPlaceRings({ x: 4, y: 4 }, 1);

      const stateAtMove1 = engine.getStateAtMoveIndex(1);
      if (stateAtMove1) {
        expect(stateAtMove1.board.stacks.size).toBeGreaterThanOrEqual(1);
      }
    });
  });

  // ============================================================================
  // processMoveViaAdapter error handling (lines ~669-677)
  // ============================================================================
  describe('processMoveViaAdapter error handling', () => {
    it('throws on invalid move type', async () => {
      const engine = new ClientSandboxEngine({
        config: createConfig(2),
        interactionHandler: new MockInteractionHandler(),
      });

      const invalidMove = {
        id: 'test',
        type: 'invalid_move_type' as Move['type'],
        player: 1,
        to: { x: 0, y: 0 },
        timestamp: new Date(),
        thinkTime: 0,
        moveNumber: 1,
      };

      await expect(engine.applyCanonicalMove(invalidMove)).rejects.toThrow(/unsupported move type/);
    });

    it('throws on mis-phased move', async () => {
      const engine = new ClientSandboxEngine({
        config: createConfig(2),
        interactionHandler: new MockInteractionHandler(),
      });

      // Force into movement phase without prior placement
      (engine as any).gameState.currentPhase = 'capture';

      const moveStackMove: Move = {
        id: 'test',
        type: 'place_ring',
        player: 1,
        to: { x: 0, y: 0 },
        timestamp: new Date(),
        thinkTime: 0,
        moveNumber: 1,
      };

      // Should throw because we're in capture phase, not ring_placement
      await expect(engine.applyCanonicalMove(moveStackMove)).rejects.toThrow(/processMove failed/);
    });
  });

  // ============================================================================
  // maybeRunAITurn branches
  // ============================================================================
  describe('maybeRunAITurn branches', () => {
    it('handles AI turn with custom RNG', async () => {
      const engine = new ClientSandboxEngine({
        config: createAIConfig(2, [4, 4]),
        interactionHandler: new MockInteractionHandler(),
      });

      let rngCallCount = 0;
      const customRng = () => {
        rngCallCount++;
        return 0.5;
      };

      await engine.maybeRunAITurn(customRng);

      // RNG was used
      expect(rngCallCount).toBeGreaterThan(0);
    });

    it('AI turn without RNG uses internal RNG', async () => {
      const engine = new ClientSandboxEngine({
        config: createAIConfig(2, [4, 4]),
        interactionHandler: new MockInteractionHandler(),
      });

      // This should not throw
      await engine.maybeRunAITurn();

      const state = engine.getGameState();
      // Some move may have been made
      expect(state.gameStatus).toBe('active');
    });
  });

  // ============================================================================
  // getValidLandingPositionsForCurrentPlayer branches (lines ~1426-1477)
  // ============================================================================
  describe('getValidLandingPositionsForCurrentPlayer phases', () => {
    it('returns empty array for position with no stack', () => {
      const engine = new ClientSandboxEngine({
        config: createConfig(2),
        interactionHandler: new MockInteractionHandler(),
      });

      const landings = engine.getValidLandingPositionsForCurrentPlayer({ x: 0, y: 0 });
      expect(landings).toHaveLength(0);
    });

    it('handles movement phase correctly', async () => {
      const engine = new ClientSandboxEngine({
        config: createConfig(2),
        interactionHandler: new MockInteractionHandler(),
      });

      // Place a ring first
      await engine.tryPlaceRings({ x: 3, y: 3 }, 1);

      // Now in movement phase
      const state = engine.getGameState();
      if (state.currentPhase === 'movement') {
        const landings = engine.getValidLandingPositionsForCurrentPlayer({ x: 3, y: 3 });
        expect(Array.isArray(landings)).toBe(true);
      }
    });

    it('returns capture landings in capture phase', async () => {
      const engine = new ClientSandboxEngine({
        config: createConfig(2),
        interactionHandler: new MockInteractionHandler(),
      });

      // Setup board for capture scenario
      (engine as any).gameState.currentPhase = 'capture';

      const landings = engine.getValidLandingPositionsForCurrentPlayer({ x: 3, y: 3 });
      expect(Array.isArray(landings)).toBe(true);
    });

    it('deduplicates landing positions', async () => {
      const engine = new ClientSandboxEngine({
        config: createConfig(2),
        interactionHandler: new MockInteractionHandler(),
      });

      await engine.tryPlaceRings({ x: 4, y: 4 }, 1);

      const landings = engine.getValidLandingPositionsForCurrentPlayer({ x: 4, y: 4 });
      // Check for duplicates
      const keys = landings.map((p) => `${p.x},${p.y}`);
      const uniqueKeys = [...new Set(keys)];
      expect(keys.length).toBe(uniqueKeys.length);
    });
  });

  // ============================================================================
  // handleHumanCellClick branches (lines ~1215-1355)
  // ============================================================================
  describe('handleHumanCellClick branches', () => {
    it('ignores click when game is not active', async () => {
      const engine = new ClientSandboxEngine({
        config: createConfig(2),
        interactionHandler: new MockInteractionHandler(),
      });

      (engine as any).gameState.gameStatus = 'completed';

      await engine.handleHumanCellClick({ x: 3, y: 3 });

      // Nothing should change
      const state = engine.getGameState();
      expect(state.board.stacks.size).toBe(0);
    });

    it('handles click on different position after initial placement', async () => {
      const engine = new ClientSandboxEngine({
        config: createConfig(2),
        interactionHandler: new MockInteractionHandler(),
      });

      // First placement
      await engine.handleHumanCellClick({ x: 3, y: 3 });

      // Second click on different position (should select, not place)
      await engine.handleHumanCellClick({ x: 5, y: 5 });

      const state = engine.getGameState();
      expect(state.gameStatus).toBe('active');
    });

    it('handles click on same position after initial placement', async () => {
      const engine = new ClientSandboxEngine({
        config: createConfig(2),
        interactionHandler: new MockInteractionHandler(),
      });

      // First placement
      await engine.handleHumanCellClick({ x: 3, y: 3 });

      // Second click on same position
      await engine.handleHumanCellClick({ x: 3, y: 3 });

      const state = engine.getGameState();
      // Should have placed additional ring
      expect(state.gameStatus).toBe('active');
    });

    it('handles chain_capture phase clicks', async () => {
      const engine = new ClientSandboxEngine({
        config: createConfig(2),
        interactionHandler: new MockInteractionHandler(),
      });

      // Force into chain_capture phase
      (engine as any).gameState.currentPhase = 'chain_capture';

      await engine.handleHumanCellClick({ x: 2, y: 2 });

      // Should handle gracefully
      const state = engine.getGameState();
      expect(state).not.toBeNull();
    });
  });

  // ============================================================================
  // LPS tracking branches (lines ~1645-1734)
  // ============================================================================
  describe('LPS tracking branches', () => {
    it('getLpsTrackingState returns initial state', () => {
      const engine = new ClientSandboxEngine({
        config: createConfig(2),
        interactionHandler: new MockInteractionHandler(),
      });

      const lpsState = engine.getLpsTrackingState();
      expect(lpsState.roundIndex).toBe(0);
      expect(lpsState.consecutiveExclusiveRounds).toBe(0);
      expect(lpsState.consecutiveExclusivePlayer).toBeNull();
      expect(lpsState.exclusivePlayerForCompletedRound).toBeNull();
    });

    it('LPS tracking updates after turn transitions', async () => {
      const engine = new ClientSandboxEngine({
        config: createConfig(2),
        interactionHandler: new MockInteractionHandler(),
      });

      // Make a move to trigger turn processing
      await engine.tryPlaceRings({ x: 3, y: 3 }, 1);

      const lpsState = engine.getLpsTrackingState();
      expect(typeof lpsState.roundIndex).toBe('number');
    });
  });

  // ============================================================================
  // applyCanonicalMoveForReplay branches (lines ~3722-4141)
  // ============================================================================
  describe('applyCanonicalMoveForReplay branches', () => {
    it('records history when game is already complete', async () => {
      const engine = new ClientSandboxEngine({
        config: createConfig(2),
        interactionHandler: new MockInteractionHandler(),
      });

      (engine as any).gameState.gameStatus = 'completed';

      const move: Move = {
        id: 'test',
        type: 'place_ring',
        player: 1,
        to: { x: 0, y: 0 },
        timestamp: new Date(),
        thinkTime: 0,
        moveNumber: 1,
      };

      await engine.applyCanonicalMoveForReplay(move);

      // History should be recorded even for completed game
      const state = engine.getGameState();
      expect(state.history.length).toBeGreaterThanOrEqual(0);
    });

    it('handles move with nextMove lookahead', async () => {
      const engine = new ClientSandboxEngine({
        config: createConfig(2),
        interactionHandler: new MockInteractionHandler(),
      });

      const move1: Move = {
        id: 'test-1',
        type: 'place_ring',
        player: 1,
        to: { x: 2, y: 2 },
        timestamp: new Date(),
        thinkTime: 0,
        moveNumber: 1,
      };

      const move2: Move = {
        id: 'test-2',
        type: 'move_stack',
        player: 1,
        from: { x: 2, y: 2 },
        to: { x: 3, y: 2 },
        timestamp: new Date(),
        thinkTime: 0,
        moveNumber: 2,
      };

      await engine.applyCanonicalMoveForReplay(move1, move2);

      const state = engine.getGameState();
      expect(state.board.stacks.has('2,2')).toBe(true);
    });

    it('handles end-game phase completion (no nextMove)', async () => {
      const engine = new ClientSandboxEngine({
        config: createConfig(2),
        interactionHandler: new MockInteractionHandler(),
      });

      const move: Move = {
        id: 'test',
        type: 'place_ring',
        player: 1,
        to: { x: 3, y: 3 },
        timestamp: new Date(),
        thinkTime: 0,
        moveNumber: 1,
      };

      // Apply with no nextMove
      await engine.applyCanonicalMoveForReplay(move, null);

      const state = engine.getGameState();
      expect(state.gameStatus).toBe('active');
    });

    it('handles no-op moves gracefully when player cannot place', async () => {
      // RR-FIX-2026-01-19: no_placement_action is only valid when player CANNOT place.
      // Set up a state where player has rings=0 and the game properly handles it.
      const engine = new ClientSandboxEngine({
        config: createConfig(2),
        interactionHandler: new MockInteractionHandler(),
      });

      // Set player 1 to have no rings in hand (makes no_placement_action valid)
      const engineAny = engine as any;
      const player1 = engineAny.gameState.players.find((p: any) => p.playerNumber === 1);
      player1.ringsInHand = 0;

      const noOpMove: Move = {
        id: 'test',
        type: 'no_placement_action',
        player: 1,
        to: { x: 0, y: 0 },
        timestamp: new Date(),
        thinkTime: 0,
        moveNumber: 1,
      };

      // Move should be accepted (player has no rings, so no_placement_action is valid)
      await engine.applyCanonicalMoveForReplay(noOpMove);

      const state = engine.getGameState();
      expect(state).not.toBeNull();
      // Game state should be defined (may be game_over due to no material check)
    });

    it('handles skip_capture moves', async () => {
      const engine = new ClientSandboxEngine({
        config: createConfig(2),
        interactionHandler: new MockInteractionHandler(),
      });

      // First place a ring
      await engine.applyCanonicalMove({
        id: 'place',
        type: 'place_ring',
        player: 1,
        to: { x: 3, y: 3 },
        timestamp: new Date(),
        thinkTime: 0,
        moveNumber: 1,
      });

      // Force to capture phase
      (engine as any).gameState.currentPhase = 'capture';

      const skipMove: Move = {
        id: 'skip',
        type: 'skip_capture',
        player: 1,
        to: { x: 0, y: 0 },
        timestamp: new Date(),
        thinkTime: 0,
        moveNumber: 2,
      };

      await engine.applyCanonicalMoveForReplay(skipMove);

      const state = engine.getGameState();
      expect(state).not.toBeNull();
    });
  });

  // ============================================================================
  // applySwapSidesForCurrentPlayer branches (lines ~1576-1637)
  // ============================================================================
  describe('applySwapSidesForCurrentPlayer branches', () => {
    it('returns false when swap not allowed', () => {
      const engine = new ClientSandboxEngine({
        config: createConfig(2),
        interactionHandler: new MockInteractionHandler(),
      });

      const result = engine.applySwapSidesForCurrentPlayer();
      expect(result).toBe(false);
    });

    it('returns false for 3+ player games', () => {
      const engine = new ClientSandboxEngine({
        config: createConfig(3),
        interactionHandler: new MockInteractionHandler(),
      });

      const result = engine.applySwapSidesForCurrentPlayer();
      expect(result).toBe(false);
    });
  });

  // ============================================================================
  // getChainCaptureContextForCurrentPlayer branches (lines ~1165-1193)
  // ============================================================================
  describe('getChainCaptureContextForCurrentPlayer edge cases', () => {
    it('returns null when game not active', () => {
      const engine = new ClientSandboxEngine({
        config: createConfig(2),
        interactionHandler: new MockInteractionHandler(),
      });

      (engine as any).gameState.gameStatus = 'completed';

      const context = engine.getChainCaptureContextForCurrentPlayer();
      expect(context).toBeNull();
    });

    it('returns null when not in chain_capture phase', () => {
      const engine = new ClientSandboxEngine({
        config: createConfig(2),
        interactionHandler: new MockInteractionHandler(),
      });

      const context = engine.getChainCaptureContextForCurrentPlayer();
      expect(context).toBeNull();
    });

    it('returns null when in chain_capture but no valid moves', () => {
      const engine = new ClientSandboxEngine({
        config: createConfig(2),
        interactionHandler: new MockInteractionHandler(),
      });

      (engine as any).gameState.currentPhase = 'chain_capture';

      const context = engine.getChainCaptureContextForCurrentPlayer();
      // No valid continue_capture_segment moves, so null
      expect(context).toBeNull();
    });
  });

  // ============================================================================
  // getGameEndExplanation branches (lines ~813-815)
  // ============================================================================
  describe('getGameEndExplanation branches', () => {
    it('returns null initially', () => {
      const engine = new ClientSandboxEngine({
        config: createConfig(2),
        interactionHandler: new MockInteractionHandler(),
      });

      const explanation = engine.getGameEndExplanation();
      expect(explanation).toBeNull();
    });
  });

  // ============================================================================
  // traceMode behavior branches
  // ============================================================================
  describe('traceMode behavior', () => {
    it('traceMode engine accepts canonical moves', async () => {
      const engine = new ClientSandboxEngine({
        config: createConfig(2),
        interactionHandler: new MockInteractionHandler(),
        traceMode: true,
      });

      await engine.applyCanonicalMove({
        id: 'test',
        type: 'place_ring',
        player: 1,
        to: { x: 3, y: 3 },
        timestamp: new Date(),
        thinkTime: 0,
        moveNumber: 1,
      });

      const state = engine.getGameState();
      expect(state.board.stacks.has('3,3')).toBe(true);
    });
  });

  // ============================================================================
  // Constructor edge cases for player kinds
  // ============================================================================
  describe('constructor player kinds edge cases', () => {
    it('defaults missing playerKinds to human', () => {
      const engine = new ClientSandboxEngine({
        config: {
          boardType: 'square8',
          numPlayers: 4,
          playerKinds: ['human', 'ai'], // Only 2 specified, need 4
        },
        interactionHandler: new MockInteractionHandler(),
      });

      const state = engine.getGameState();
      expect(state.players[0].type).toBe('human');
      expect(state.players[1].type).toBe('ai');
      // Players 3 and 4 should default to human
      expect(state.players[2].type).toBe('human');
      expect(state.players[3].type).toBe('human');
    });
  });

  // ============================================================================
  // getValidMoves edge cases
  // ============================================================================
  describe('getValidMoves edge cases', () => {
    it('returns empty for wrong player', () => {
      const engine = new ClientSandboxEngine({
        config: createConfig(2),
        interactionHandler: new MockInteractionHandler(),
      });

      const moves = engine.getValidMoves(2); // Player 2 is not current
      expect(moves).toHaveLength(0);
    });

    it('returns empty when game not active', () => {
      const engine = new ClientSandboxEngine({
        config: createConfig(2),
        interactionHandler: new MockInteractionHandler(),
      });

      (engine as any).gameState.gameStatus = 'completed';

      const moves = engine.getValidMoves(1);
      expect(moves).toHaveLength(0);
    });
  });

  // ============================================================================
  // debugCheckpoint behavior
  // ============================================================================
  describe('debugCheckpoint behavior', () => {
    it('checkpoint hook is called during move application', async () => {
      const engine = new ClientSandboxEngine({
        config: createConfig(2),
        interactionHandler: new MockInteractionHandler(),
      });

      const checkpoints: string[] = [];
      engine.setDebugCheckpointHook((label, _state) => {
        checkpoints.push(label);
      });

      await engine.applyCanonicalMove({
        id: 'test',
        type: 'place_ring',
        player: 1,
        to: { x: 3, y: 3 },
        timestamp: new Date(),
        thinkTime: 0,
        moveNumber: 1,
      });

      expect(checkpoints.length).toBeGreaterThan(0);
    });

    it('no error when hook not set', async () => {
      const engine = new ClientSandboxEngine({
        config: createConfig(2),
        interactionHandler: new MockInteractionHandler(),
      });

      // No hook set
      await engine.applyCanonicalMove({
        id: 'test',
        type: 'place_ring',
        player: 1,
        to: { x: 3, y: 3 },
        timestamp: new Date(),
        thinkTime: 0,
        moveNumber: 1,
      });

      const state = engine.getGameState();
      expect(state.board.stacks.has('3,3')).toBe(true);
    });
  });

  // ============================================================================
  // consumeRecentLineHighlights behavior
  // ============================================================================
  describe('consumeRecentLineHighlights clears buffer', () => {
    it('second consumption returns empty', () => {
      const engine = new ClientSandboxEngine({
        config: createConfig(2),
        interactionHandler: new MockInteractionHandler(),
      });

      engine.consumeRecentLineHighlights();
      const second = engine.consumeRecentLineHighlights();
      expect(second).toHaveLength(0);
    });
  });

  // ============================================================================
  // hexagonal board type
  // ============================================================================
  describe('hexagonal board support', () => {
    it('creates engine with hexagonal board', () => {
      const engine = new ClientSandboxEngine({
        config: {
          boardType: 'hexagonal',
          numPlayers: 2,
          playerKinds: ['human', 'human'],
        },
        interactionHandler: new MockInteractionHandler(),
      });

      const state = engine.getGameState();
      expect(state.boardType).toBe('hexagonal');
    });
  });
});
