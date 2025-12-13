/**
 * LPS Cross-Interaction Scenarios: Lines & Territory (Sandbox)
 *
 * These tests exercise Last-Player-Standing (LPS) victory (R172) in the
 * ClientSandboxEngine in the presence of active line and territory mechanics,
 * verifying that:
 * 1. Lines and territory resolve fully before LPS triggers.
 * 2. The sandbox engine correctly sequences phase processing.
 *
 * This mirrors the backend tests in GameEngine.victory.LPS.crossInteraction.test.ts
 * to ensure both engines agree on LPS + Line/Territory cross-interaction semantics.
 */

import {
  ClientSandboxEngine,
  SandboxConfig,
  SandboxInteractionHandler,
} from '../../src/client/sandbox/ClientSandboxEngine';
import {
  BoardType,
  GameResult,
  GameState,
  PlayerChoice,
  PlayerChoiceResponseFor,
  Position,
  BOARD_CONFIGS,
  positionToString,
} from '../../src/shared/types/game';
import { pos, addStack, addMarker } from '../utils/fixtures';

describe('ClientSandboxEngine LPS + Line/Territory Cross-Interaction Scenarios', () => {
  const boardType: BoardType = 'square8';
  const requiredLineLength = BOARD_CONFIGS[boardType].lineLength;

  function createThreePlayerEngine(): ClientSandboxEngine {
    const config: SandboxConfig = {
      boardType,
      numPlayers: 3,
      playerKinds: ['human', 'human', 'human'],
    };

    const handler: SandboxInteractionHandler = {
      async requestChoice<TChoice extends PlayerChoice>(
        choice: TChoice
      ): Promise<PlayerChoiceResponseFor<TChoice>> {
        const anyChoice = choice as any;
        const options: any[] = (anyChoice.options as any[]) ?? [];
        const selectedOption = options.length > 0 ? options[0] : undefined;

        return {
          choiceId: anyChoice.id,
          playerNumber: anyChoice.playerNumber,
          choiceType: anyChoice.type,
          selectedOption,
        } as PlayerChoiceResponseFor<TChoice>;
      },
    };

    return new ClientSandboxEngine({ config, interactionHandler: handler });
  }

  function startInteractiveTurn(
    engineAny: any,
    state: GameState,
    playerNumber: number
  ): GameResult | null {
    state.currentPlayer = playerNumber;
    state.currentPhase = 'ring_placement';
    engineAny.handleStartOfInteractiveTurn();
    return engineAny.victoryResult;
  }

  describe('LPS + Lines Cross-Interaction', () => {
    /**
     * Scenario: After game state changes (simulating line processing effects),
     * LPS plateau is detected when only one player has real actions.
     *
     * Setup:
     * - 3-player game on square8
     * - P1 has stacks and can make moves (has real actions)
     * - P2 and P3 have no stacks, no rings in hand (no real actions)
     *
     * Expected:
     * - After TWO full rounds of P1 being the only active player, LPS triggers
     * - This tests the LPS + line phase ordering conceptually
     */
    it('LPS_triggers_only_after_line_processing_completes', async () => {
      const engine = createThreePlayerEngine();
      const engineAny: any = engine;
      const state: GameState = engineAny.gameState as GameState;

      // Reset victory state
      engineAny.victoryResult = null;
      state.gameStatus = 'active';

      // Clear initial rings in hand to simulate mid-game state
      for (const player of state.players) {
        player.ringsInHand = 0;
      }

      // Setup: P1 has a stack (can act), P2 and P3 have nothing
      const p1StackPos: Position = pos(7, 7);
      addStack(state.board, p1StackPos, 1, 2);

      // Simulate line processing has happened by directly collapsing line positions
      for (let i = 0; i < requiredLineLength; i++) {
        const key = positionToString(pos(i, 1));
        state.board.collapsedSpaces.set(key, 1);
      }

      // Mock hasAnyRealActionForPlayer: P1 has real actions, P2/P3 don't
      const realActionByPlayer: Record<number, boolean> = { 1: true, 2: false, 3: false };
      engineAny.hasAnyRealActionForPlayer = jest.fn(
        (playerNumber: number) => !!realActionByPlayer[playerNumber]
      );

      // Game should be active before LPS tracking
      expect(state.gameStatus).toBe('active');

      // Now simulate THREE full rounds of LPS tracking (per RR-CANON-R172)
      // Round 1: P1 -> P2 -> P3
      let result = startInteractiveTurn(engineAny, state, 1);
      expect(result).toBeNull();
      result = startInteractiveTurn(engineAny, state, 2);
      expect(result).toBeNull();
      result = startInteractiveTurn(engineAny, state, 3);
      expect(result).toBeNull();

      // Start of round 2: P1's turn completes round 1
      result = startInteractiveTurn(engineAny, state, 1);
      expect(result).toBeNull(); // Still no LPS - only 1 consecutive round

      // Round 2: P2 -> P3
      result = startInteractiveTurn(engineAny, state, 2);
      expect(result).toBeNull();
      result = startInteractiveTurn(engineAny, state, 3);
      expect(result).toBeNull();

      // Start of round 3: P1's turn completes round 2
      result = startInteractiveTurn(engineAny, state, 1);
      expect(result).toBeNull(); // Still no LPS - only 2 consecutive rounds

      // Round 3: P2 -> P3
      result = startInteractiveTurn(engineAny, state, 2);
      expect(result).toBeNull();
      result = startInteractiveTurn(engineAny, state, 3);
      expect(result).toBeNull();

      // Start of round 4: P1's turn completes round 3 - NOW LPS triggers
      result = startInteractiveTurn(engineAny, state, 1);

      // LPS should trigger now (after 3 consecutive rounds per RR-CANON-R172)
      expect(result).not.toBeNull();
      expect(result!.winner).toBe(1);
      expect(result!.reason).toBe('last_player_standing');

      // Terminal state invariants
      const finalState = engine.getGameState();
      expect(finalState.gameStatus).toBe('completed');
      expect(finalState.winner).toBe(1);
    });

    /**
     * Scenario: After game state change simulating line processing,
     * LPS plateau is detected when only one player has real actions.
     */
    it('line_processing_can_create_LPS_plateau_by_changing_board_state', async () => {
      const engine = createThreePlayerEngine();
      const engineAny: any = engine;
      const state: GameState = engineAny.gameState as GameState;

      engineAny.victoryResult = null;
      state.gameStatus = 'active';

      // Clear initial rings in hand
      for (const player of state.players) {
        player.ringsInHand = 0;
      }

      // Setup: P1 has a stack (retained after line processing)
      const p1StackPos: Position = pos(7, 7);
      addStack(state.board, p1StackPos, 1, 2);

      // Simulate collapsed spaces from line processing
      for (let i = 0; i < requiredLineLength; i++) {
        const key = positionToString(pos(i, 1));
        state.board.collapsedSpaces.set(key, 1);
      }

      // After line processing: only P1 has real actions
      engineAny.hasAnyRealActionForPlayer = jest.fn((playerNumber: number) => playerNumber === 1);

      // THREE full rounds where only P1 has actions (per RR-CANON-R172)
      // Round 1
      let result = startInteractiveTurn(engineAny, state, 1);
      expect(result).toBeNull();
      result = startInteractiveTurn(engineAny, state, 2);
      expect(result).toBeNull();
      result = startInteractiveTurn(engineAny, state, 3);
      expect(result).toBeNull();

      // Start of round 2 (completes round 1)
      result = startInteractiveTurn(engineAny, state, 1);
      expect(result).toBeNull(); // Only 1 consecutive round - not enough

      // Round 2
      result = startInteractiveTurn(engineAny, state, 2);
      expect(result).toBeNull();
      result = startInteractiveTurn(engineAny, state, 3);
      expect(result).toBeNull();

      // Start of round 3 (completes round 2)
      result = startInteractiveTurn(engineAny, state, 1);
      expect(result).toBeNull(); // Only 2 consecutive rounds - not enough

      // Round 3
      result = startInteractiveTurn(engineAny, state, 2);
      expect(result).toBeNull();
      result = startInteractiveTurn(engineAny, state, 3);
      expect(result).toBeNull();

      // P1's turn again (start of round 4, completes round 3) - LPS should trigger now
      result = startInteractiveTurn(engineAny, state, 1);

      expect(result).not.toBeNull();
      expect(result!.winner).toBe(1);
      expect(result!.reason).toBe('last_player_standing');

      // Verify the board state reflects line processing
      for (let i = 0; i < requiredLineLength; i++) {
        const key = positionToString(pos(i, 1));
        expect(state.board.collapsedSpaces.get(key)).toBe(1);
      }
    });
  });

  describe('LPS + Territory Cross-Interaction', () => {
    /**
     * Scenario: After territory-like state changes, LPS plateau is detected
     *
     * This tests the phase ordering conceptually: territory processing
     * happens, then LPS is evaluated.
     */
    it('LPS_triggers_only_after_territory_processing_completes', async () => {
      const engine = createThreePlayerEngine();
      const engineAny: any = engine;
      const state: GameState = engineAny.gameState as GameState;

      engineAny.victoryResult = null;
      state.gameStatus = 'active';

      // Clear initial rings in hand
      for (const player of state.players) {
        player.ringsInHand = 0;
      }

      // P1 has a stack
      const p1OutsidePos: Position = pos(0, 0);
      addStack(state.board, p1OutsidePos, 1, 2);

      // Only P1 has real actions from the start
      const realActionByPlayer: Record<number, boolean> = { 1: true, 2: false, 3: false };

      engineAny.hasAnyRealActionForPlayer = jest.fn(
        (playerNumber: number) => !!realActionByPlayer[playerNumber]
      );

      // Verify P1 has material
      expect(engineAny.hasAnyRealActionForPlayer(1)).toBe(true);

      // THREE full rounds of LPS tracking where only P1 has actions (per RR-CANON-R172)
      // Round 1
      let result = startInteractiveTurn(engineAny, state, 1);
      expect(result).toBeNull();
      result = startInteractiveTurn(engineAny, state, 2);
      expect(result).toBeNull();
      result = startInteractiveTurn(engineAny, state, 3);
      expect(result).toBeNull();

      // Start of round 2 (completes round 1)
      result = startInteractiveTurn(engineAny, state, 1);
      expect(result).toBeNull(); // Only 1 consecutive round

      // Round 2
      result = startInteractiveTurn(engineAny, state, 2);
      expect(result).toBeNull();
      result = startInteractiveTurn(engineAny, state, 3);
      expect(result).toBeNull();

      // Start of round 3 (completes round 2)
      result = startInteractiveTurn(engineAny, state, 1);
      expect(result).toBeNull(); // Only 2 consecutive rounds

      // Round 3
      result = startInteractiveTurn(engineAny, state, 2);
      expect(result).toBeNull();
      result = startInteractiveTurn(engineAny, state, 3);
      expect(result).toBeNull();

      // P1's turn again (start of round 4) - LPS should trigger now (completed 3 rounds)
      result = startInteractiveTurn(engineAny, state, 1);

      expect(result).not.toBeNull();
      expect(result!.winner).toBe(1);
      expect(result!.reason).toBe('last_player_standing');

      const finalState = engine.getGameState();
      expect(finalState.gameStatus).toBe('completed');
      expect(finalState.winner).toBe(1);
    });

    /**
     * Scenario: Territory collapse removes all of P2's material, creating LPS
     */
    it('territory_collapse_can_create_LPS_plateau_by_eliminating_player_material', async () => {
      const engine = createThreePlayerEngine();
      const engineAny: any = engine;
      const state: GameState = engineAny.gameState as GameState;

      engineAny.victoryResult = null;
      state.gameStatus = 'active';

      // Clear initial rings in hand
      for (const player of state.players) {
        player.ringsInHand = 0;
      }

      // P1 has stacks outside any territory region
      const p1StackPos: Position = pos(0, 0);
      addStack(state.board, p1StackPos, 1, 3);

      // P2's only stack is inside the region - will be lost
      const p2StackPos: Position = pos(5, 5);
      addStack(state.board, p2StackPos, 2, 2);

      // Track whether territory collapse has happened
      let territoryProcessed = false;

      engineAny.hasAnyRealActionForPlayer = jest.fn((playerNumber: number) => {
        if (!territoryProcessed) {
          // Before territory: P1 and P2 have material
          return playerNumber === 1 || playerNumber === 2;
        } else {
          // After territory: only P1 has real actions
          return playerNumber === 1;
        }
      });

      // Verify both P1 and P2 have material before territory processing
      expect(engineAny.hasAnyRealActionForPlayer(1)).toBe(true);
      expect(engineAny.hasAnyRealActionForPlayer(2)).toBe(true);

      // Simulate territory collapse removing P2's stack
      state.board.stacks.delete(positionToString(p2StackPos));
      state.board.collapsedSpaces.set(positionToString(p2StackPos), 1);
      territoryProcessed = true;

      // THREE full rounds of LPS tracking (per RR-CANON-R172)
      // Round 1
      let result = startInteractiveTurn(engineAny, state, 1);
      expect(result).toBeNull();
      result = startInteractiveTurn(engineAny, state, 2);
      expect(result).toBeNull();
      result = startInteractiveTurn(engineAny, state, 3);
      expect(result).toBeNull();

      // Start of round 2 (completes round 1)
      result = startInteractiveTurn(engineAny, state, 1);
      expect(result).toBeNull(); // Only 1 consecutive round

      // Round 2
      result = startInteractiveTurn(engineAny, state, 2);
      expect(result).toBeNull();
      result = startInteractiveTurn(engineAny, state, 3);
      expect(result).toBeNull();

      // Start of round 3 (completes round 2)
      result = startInteractiveTurn(engineAny, state, 1);
      expect(result).toBeNull(); // Only 2 consecutive rounds

      // Round 3
      result = startInteractiveTurn(engineAny, state, 2);
      expect(result).toBeNull();
      result = startInteractiveTurn(engineAny, state, 3);
      expect(result).toBeNull();

      // LPS triggers (start of round 4, completing 3 consecutive rounds)
      result = startInteractiveTurn(engineAny, state, 1);
      expect(result).not.toBeNull();
      expect(result!.winner).toBe(1);
      expect(result!.reason).toBe('last_player_standing');
    });
  });

  describe('LPS + Lines + Territory Combined', () => {
    /**
     * Scenario: Combined line and territory processing effects
     *
     * This test simulates the effect of both line and territory processing
     * by directly manipulating board state, then verifies LPS triggers correctly.
     */
    it('LPS_evaluated_only_after_both_lines_and_territory_complete', async () => {
      const engine = createThreePlayerEngine();
      const engineAny: any = engine;
      const state: GameState = engineAny.gameState as GameState;

      engineAny.victoryResult = null;
      state.gameStatus = 'active';

      // Clear initial rings in hand
      for (const player of state.players) {
        player.ringsInHand = 0;
      }

      // P1 has a stack outside all regions
      const p1StackPos: Position = pos(7, 7);
      addStack(state.board, p1StackPos, 1, 3);

      // P2's stack will be affected by territory collapse
      const p2StackPos: Position = pos(5, 5);
      addStack(state.board, p2StackPos, 2, 1);

      // Track processing phases
      let allProcessingComplete = false;

      engineAny.hasAnyRealActionForPlayer = jest.fn((playerNumber: number) => {
        if (!allProcessingComplete) {
          // Before processing: P1 and P2 both have actions
          return playerNumber === 1 || playerNumber === 2;
        } else {
          // After processing: only P1 has real actions
          return playerNumber === 1;
        }
      });

      // Step 1: Simulate line processing effect (collapse line positions)
      for (let i = 0; i < requiredLineLength; i++) {
        const key = positionToString(pos(i, 0));
        state.board.collapsedSpaces.set(key, 1);
      }

      // Game should NOT have ended yet
      expect(state.gameStatus).toBe('active');

      // Step 2: Simulate territory collapse
      state.board.stacks.delete(positionToString(p2StackPos));
      state.board.collapsedSpaces.set(positionToString(p2StackPos), 1);

      // Mark all processing complete
      allProcessingComplete = true;

      // Game should still be active until LPS is properly tracked
      expect(state.gameStatus).toBe('active');

      // Step 3: THREE full rounds of LPS tracking (per RR-CANON-R172)
      // Round 1
      let result = startInteractiveTurn(engineAny, state, 1);
      expect(result).toBeNull();
      result = startInteractiveTurn(engineAny, state, 2);
      expect(result).toBeNull();
      result = startInteractiveTurn(engineAny, state, 3);
      expect(result).toBeNull();

      // Start of round 2 (completes round 1)
      result = startInteractiveTurn(engineAny, state, 1);
      expect(result).toBeNull(); // Only 1 consecutive round

      // Round 2
      result = startInteractiveTurn(engineAny, state, 2);
      expect(result).toBeNull();
      result = startInteractiveTurn(engineAny, state, 3);
      expect(result).toBeNull();

      // Start of round 3 (completes round 2)
      result = startInteractiveTurn(engineAny, state, 1);
      expect(result).toBeNull(); // Only 2 consecutive rounds

      // Round 3
      result = startInteractiveTurn(engineAny, state, 2);
      expect(result).toBeNull();
      result = startInteractiveTurn(engineAny, state, 3);
      expect(result).toBeNull();

      // Step 4: LPS triggers on P1's next turn (start of round 4, completes 3 consecutive rounds)
      result = startInteractiveTurn(engineAny, state, 1);

      expect(result).not.toBeNull();
      expect(result!.winner).toBe(1);
      expect(result!.reason).toBe('last_player_standing');

      const finalState = engine.getGameState();
      expect(finalState.gameStatus).toBe('completed');
      expect(finalState.winner).toBe(1);
    });
  });

  describe('Phase Order Verification', () => {
    /**
     * Verify that advanceAfterMovement processes phases in the correct order
     */
    it('advanceAfterMovement_processes_lines_before_territory_before_victory', async () => {
      const engine = createThreePlayerEngine();
      const engineAny: any = engine;
      const state: GameState = engineAny.gameState as GameState;

      // Track processing order
      const processingOrder: string[] = [];

      // Clear initial rings in hand
      for (const player of state.players) {
        player.ringsInHand = 0;
      }

      // P1 has material
      addStack(state.board, pos(7, 7), 1, 2);

      // Spy on the processing methods
      const originalProcessLines = engineAny.processLinesForCurrentPlayer.bind(engineAny);
      engineAny.processLinesForCurrentPlayer = jest.fn(async () => {
        processingOrder.push('line_processing');
        // Don't actually process to avoid state changes
      });

      const originalProcessTerritory =
        engineAny.processDisconnectedRegionsForCurrentPlayer.bind(engineAny);
      engineAny.processDisconnectedRegionsForCurrentPlayer = jest.fn(async () => {
        processingOrder.push('territory_processing');
        // Don't actually process to avoid state changes
      });

      const originalCheckVictory = engineAny.checkAndApplyVictory.bind(engineAny);
      engineAny.checkAndApplyVictory = jest.fn(() => {
        processingOrder.push('victory_check');
        // Don't actually check to avoid state changes
      });

      // Call advanceAfterMovement which should process in order
      state.currentPlayer = 1;
      state.currentPhase = 'movement';
      await engineAny.advanceAfterMovement();

      // Verify order: lines → territory → victory
      expect(processingOrder).toEqual(['line_processing', 'territory_processing', 'victory_check']);
    });
  });
});
