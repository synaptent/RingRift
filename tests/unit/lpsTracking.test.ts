/**
 * LPS (Last-Player-Standing) Tracking Module Unit Tests
 *
 * Tests for the LPS victory condition tracking (R172) including:
 * - Creating and resetting LPS tracking state
 * - Updating tracking as players take turns
 * - Finalizing completed rounds
 * - Evaluating victory conditions
 * - Phase checking and result building
 */

import {
  createLpsTrackingState,
  resetLpsTrackingState,
  updateLpsTracking,
  finalizeCompletedLpsRound,
  evaluateLpsVictory,
  isLpsActivePhase,
  buildLpsVictoryResult,
  LpsTrackingState,
  LPS_REQUIRED_CONSECUTIVE_ROUNDS,
} from '../../src/shared/engine/lpsTracking';
import type { GameState, GamePhase, Player } from '../../src/shared/types/game';

describe('lpsTracking module', () => {
  describe('createLpsTrackingState', () => {
    it('should create fresh state with roundIndex 0', () => {
      const state = createLpsTrackingState();
      expect(state.roundIndex).toBe(0);
    });

    it('should create empty currentRoundActorMask', () => {
      const state = createLpsTrackingState();
      expect(state.currentRoundActorMask.size).toBe(0);
    });

    it('should set currentRoundFirstPlayer to null', () => {
      const state = createLpsTrackingState();
      expect(state.currentRoundFirstPlayer).toBeNull();
    });

    it('should set exclusivePlayerForCompletedRound to null', () => {
      const state = createLpsTrackingState();
      expect(state.exclusivePlayerForCompletedRound).toBeNull();
    });

    it('should set consecutiveExclusiveRounds to 0', () => {
      const state = createLpsTrackingState();
      expect(state.consecutiveExclusiveRounds).toBe(0);
    });

    it('should set consecutiveExclusivePlayer to null', () => {
      const state = createLpsTrackingState();
      expect(state.consecutiveExclusivePlayer).toBeNull();
    });
  });

  describe('resetLpsTrackingState', () => {
    it('should reset roundIndex to 0', () => {
      const state = createLpsTrackingState();
      state.roundIndex = 5;
      resetLpsTrackingState(state);
      expect(state.roundIndex).toBe(0);
    });

    it('should clear currentRoundActorMask', () => {
      const state = createLpsTrackingState();
      state.currentRoundActorMask.set(1, true);
      state.currentRoundActorMask.set(2, false);
      resetLpsTrackingState(state);
      expect(state.currentRoundActorMask.size).toBe(0);
    });

    it('should reset currentRoundFirstPlayer to null', () => {
      const state = createLpsTrackingState();
      state.currentRoundFirstPlayer = 2;
      resetLpsTrackingState(state);
      expect(state.currentRoundFirstPlayer).toBeNull();
    });

    it('should reset exclusivePlayerForCompletedRound to null', () => {
      const state = createLpsTrackingState();
      state.exclusivePlayerForCompletedRound = 1;
      resetLpsTrackingState(state);
      expect(state.exclusivePlayerForCompletedRound).toBeNull();
    });

    it('should reset consecutiveExclusiveRounds to 0', () => {
      const state = createLpsTrackingState();
      state.consecutiveExclusiveRounds = 3;
      resetLpsTrackingState(state);
      expect(state.consecutiveExclusiveRounds).toBe(0);
    });

    it('should reset consecutiveExclusivePlayer to null', () => {
      const state = createLpsTrackingState();
      state.consecutiveExclusivePlayer = 2;
      resetLpsTrackingState(state);
      expect(state.consecutiveExclusivePlayer).toBeNull();
    });
  });

  describe('updateLpsTracking', () => {
    let lps: LpsTrackingState;

    beforeEach(() => {
      lps = createLpsTrackingState();
    });

    it('should do nothing when activePlayers is empty', () => {
      updateLpsTracking(lps, {
        currentPlayer: 1,
        activePlayers: [],
        hasRealAction: true,
      });

      expect(lps.roundIndex).toBe(0);
      expect(lps.currentRoundFirstPlayer).toBeNull();
      expect(lps.currentRoundActorMask.size).toBe(0);
    });

    it('should do nothing when currentPlayer has no material (not in activePlayers)', () => {
      updateLpsTracking(lps, {
        currentPlayer: 3,
        activePlayers: [1, 2],
        hasRealAction: true,
      });

      expect(lps.roundIndex).toBe(0);
      expect(lps.currentRoundActorMask.size).toBe(0);
    });

    it('should start new round on first call', () => {
      updateLpsTracking(lps, {
        currentPlayer: 1,
        activePlayers: [1, 2],
        hasRealAction: true,
      });

      expect(lps.roundIndex).toBe(1);
      expect(lps.currentRoundFirstPlayer).toBe(1);
      expect(lps.currentRoundActorMask.get(1)).toBe(true);
    });

    it('should record subsequent player turns without new round', () => {
      updateLpsTracking(lps, {
        currentPlayer: 1,
        activePlayers: [1, 2],
        hasRealAction: true,
      });

      updateLpsTracking(lps, {
        currentPlayer: 2,
        activePlayers: [1, 2],
        hasRealAction: false,
      });

      expect(lps.roundIndex).toBe(1);
      expect(lps.currentRoundActorMask.get(1)).toBe(true);
      expect(lps.currentRoundActorMask.get(2)).toBe(false);
    });

    it('should finalize round and start new one when cycling back to first player', () => {
      // First round
      updateLpsTracking(lps, {
        currentPlayer: 1,
        activePlayers: [1, 2],
        hasRealAction: true,
      });

      updateLpsTracking(lps, {
        currentPlayer: 2,
        activePlayers: [1, 2],
        hasRealAction: false,
      });

      // Cycle back to player 1
      updateLpsTracking(lps, {
        currentPlayer: 1,
        activePlayers: [1, 2],
        hasRealAction: true,
      });

      expect(lps.roundIndex).toBe(2);
      expect(lps.exclusivePlayerForCompletedRound).toBe(1);
      expect(lps.currentRoundFirstPlayer).toBe(1);
      // First round with exclusive player
      expect(lps.consecutiveExclusiveRounds).toBe(1);
      expect(lps.consecutiveExclusivePlayer).toBe(1);
    });

    it('should track consecutive exclusive rounds for the same player', () => {
      // Round 1: P1 exclusive
      updateLpsTracking(lps, { currentPlayer: 1, activePlayers: [1, 2], hasRealAction: true });
      updateLpsTracking(lps, { currentPlayer: 2, activePlayers: [1, 2], hasRealAction: false });
      // Cycle to round 2
      updateLpsTracking(lps, { currentPlayer: 1, activePlayers: [1, 2], hasRealAction: true });

      expect(lps.consecutiveExclusiveRounds).toBe(1);
      expect(lps.consecutiveExclusivePlayer).toBe(1);

      // Round 2: P1 exclusive again
      updateLpsTracking(lps, { currentPlayer: 2, activePlayers: [1, 2], hasRealAction: false });
      // Cycle to round 3
      updateLpsTracking(lps, { currentPlayer: 1, activePlayers: [1, 2], hasRealAction: true });

      expect(lps.consecutiveExclusiveRounds).toBe(2);
      expect(lps.consecutiveExclusivePlayer).toBe(1);
    });

    it('should reset consecutive count when different player becomes exclusive', () => {
      // Round 1: P1 exclusive
      updateLpsTracking(lps, { currentPlayer: 1, activePlayers: [1, 2], hasRealAction: true });
      updateLpsTracking(lps, { currentPlayer: 2, activePlayers: [1, 2], hasRealAction: false });
      // Cycle to round 2
      updateLpsTracking(lps, { currentPlayer: 1, activePlayers: [1, 2], hasRealAction: true });

      expect(lps.consecutiveExclusiveRounds).toBe(1);
      expect(lps.consecutiveExclusivePlayer).toBe(1);

      // Round 2: P2 exclusive now (different player).
      // New round starts with P1 (no real actions), then P2 (has real actions),
      // and finalises when we cycle back to P1.
      updateLpsTracking(lps, { currentPlayer: 1, activePlayers: [1, 2], hasRealAction: false });
      updateLpsTracking(lps, { currentPlayer: 2, activePlayers: [1, 2], hasRealAction: true });
      // Cycle back to P1 to complete Round 2 (actor mask: P1=false, P2=true).
      updateLpsTracking(lps, { currentPlayer: 1, activePlayers: [1, 2], hasRealAction: false });

      // P2 is now exclusive for the most recent round, count reset to 1
      expect(lps.consecutiveExclusiveRounds).toBe(1);
      expect(lps.consecutiveExclusivePlayer).toBe(2);
    });

    it('should reset consecutive count when no exclusive player', () => {
      // Round 1: P1 exclusive
      updateLpsTracking(lps, { currentPlayer: 1, activePlayers: [1, 2], hasRealAction: true });
      updateLpsTracking(lps, { currentPlayer: 2, activePlayers: [1, 2], hasRealAction: false });
      // Cycle to round 2
      updateLpsTracking(lps, { currentPlayer: 1, activePlayers: [1, 2], hasRealAction: true });

      expect(lps.consecutiveExclusiveRounds).toBe(1);

      // Round 2: Both players have actions (no exclusive)
      updateLpsTracking(lps, { currentPlayer: 2, activePlayers: [1, 2], hasRealAction: true });
      // Cycle to round 3
      updateLpsTracking(lps, { currentPlayer: 1, activePlayers: [1, 2], hasRealAction: true });

      // No exclusive player, count reset to 0
      expect(lps.consecutiveExclusiveRounds).toBe(0);
      expect(lps.consecutiveExclusivePlayer).toBeNull();
    });

    it('should handle 3-player cycle correctly', () => {
      updateLpsTracking(lps, {
        currentPlayer: 1,
        activePlayers: [1, 2, 3],
        hasRealAction: true,
      });

      updateLpsTracking(lps, {
        currentPlayer: 2,
        activePlayers: [1, 2, 3],
        hasRealAction: true,
      });

      updateLpsTracking(lps, {
        currentPlayer: 3,
        activePlayers: [1, 2, 3],
        hasRealAction: false,
      });

      // Still round 1
      expect(lps.roundIndex).toBe(1);

      // Back to player 1 starts round 2
      updateLpsTracking(lps, {
        currentPlayer: 1,
        activePlayers: [1, 2, 3],
        hasRealAction: true,
      });

      expect(lps.roundIndex).toBe(2);
      // Both 1 and 2 had actions, so no exclusive player
      expect(lps.exclusivePlayerForCompletedRound).toBeNull();
    });

    it('should start new cycle when first player drops out', () => {
      // Player 1 starts
      updateLpsTracking(lps, {
        currentPlayer: 1,
        activePlayers: [1, 2, 3],
        hasRealAction: true,
      });

      // Player 2 continues
      updateLpsTracking(lps, {
        currentPlayer: 2,
        activePlayers: [1, 2, 3],
        hasRealAction: false,
      });

      // Player 1 eliminated, player 3 becomes first
      updateLpsTracking(lps, {
        currentPlayer: 3,
        activePlayers: [2, 3], // Player 1 eliminated
        hasRealAction: false,
      });

      expect(lps.roundIndex).toBe(2);
      expect(lps.currentRoundFirstPlayer).toBe(3);
    });

    it('should record hasRealAction=false correctly', () => {
      updateLpsTracking(lps, {
        currentPlayer: 1,
        activePlayers: [1, 2],
        hasRealAction: false,
      });

      expect(lps.currentRoundActorMask.get(1)).toBe(false);
    });

    it('should clear exclusivePlayerForCompletedRound on new cycle start', () => {
      // Setup exclusive player in previous round
      lps.exclusivePlayerForCompletedRound = 2;
      lps.currentRoundFirstPlayer = null; // Force new cycle

      updateLpsTracking(lps, {
        currentPlayer: 1,
        activePlayers: [1, 2],
        hasRealAction: true,
      });

      expect(lps.exclusivePlayerForCompletedRound).toBeNull();
    });
  });

  describe('finalizeCompletedLpsRound', () => {
    it('should return exclusive player when only one has actions', () => {
      const actorMask = new Map<number, boolean>([
        [1, true],
        [2, false],
        [3, false],
      ]);

      const result = finalizeCompletedLpsRound([1, 2, 3], actorMask);
      expect(result).toBe(1);
    });

    it('should return null when no players have actions', () => {
      const actorMask = new Map<number, boolean>([
        [1, false],
        [2, false],
      ]);

      const result = finalizeCompletedLpsRound([1, 2], actorMask);
      expect(result).toBeNull();
    });

    it('should return null when multiple players have actions', () => {
      const actorMask = new Map<number, boolean>([
        [1, true],
        [2, true],
      ]);

      const result = finalizeCompletedLpsRound([1, 2], actorMask);
      expect(result).toBeNull();
    });

    it('should only consider active players', () => {
      const actorMask = new Map<number, boolean>([
        [1, true],
        [2, true],
        [3, false],
      ]);

      // Player 2 is not in activePlayers
      const result = finalizeCompletedLpsRound([1, 3], actorMask);
      expect(result).toBe(1);
    });

    it('should handle empty actorMask', () => {
      const actorMask = new Map<number, boolean>();
      const result = finalizeCompletedLpsRound([1, 2], actorMask);
      expect(result).toBeNull();
    });

    it('should handle player not in actorMask (undefined = false)', () => {
      const actorMask = new Map<number, boolean>([[1, true]]);

      // Player 2 not in mask, so treated as no actions
      const result = finalizeCompletedLpsRound([1, 2], actorMask);
      expect(result).toBe(1);
    });
  });

  describe('isLpsActivePhase', () => {
    it('should return true for ring_placement phase', () => {
      expect(isLpsActivePhase('ring_placement')).toBe(true);
    });

    it('should return true for movement phase', () => {
      expect(isLpsActivePhase('movement')).toBe(true);
    });

    it('should return true for capture phase', () => {
      expect(isLpsActivePhase('capture')).toBe(true);
    });

    it('should return true for chain_capture phase', () => {
      expect(isLpsActivePhase('chain_capture')).toBe(true);
    });

    it('should return false for line_formed phase', () => {
      expect(isLpsActivePhase('line_formed')).toBe(false);
    });

    it('should return false for territory_formed phase', () => {
      expect(isLpsActivePhase('territory_formed')).toBe(false);
    });

    it('should return false for game_over phase', () => {
      expect(isLpsActivePhase('game_over')).toBe(false);
    });
  });

  describe('evaluateLpsVictory', () => {
    const createMockGameState = (overrides: Partial<GameState> = {}): GameState =>
      ({
        gameStatus: 'active',
        currentPhase: 'movement' as GamePhase,
        currentPlayer: 1,
        players: [{ playerNumber: 1 } as Player, { playerNumber: 2 } as Player],
        ...overrides,
      }) as unknown as GameState;

    /**
     * Helper to set up LPS state for victory evaluation.
     * Sets both consecutiveExclusiveRounds and consecutiveExclusivePlayer
     * as required by the two-round LPS requirement.
     */
    function setupLpsForVictory(lps: LpsTrackingState, player: number): void {
      lps.consecutiveExclusiveRounds = 2;
      lps.consecutiveExclusivePlayer = player;
      lps.exclusivePlayerForCompletedRound = player;
    }

    it('should return isVictory=false when game is not active', () => {
      const gameState = createMockGameState({ gameStatus: 'completed' });
      const lps = createLpsTrackingState();
      setupLpsForVictory(lps, 1);

      const result = evaluateLpsVictory({
        gameState,
        lps,
        hasAnyRealAction: () => true,
        hasMaterial: () => true,
      });

      expect(result.isVictory).toBe(false);
      expect(result.reason).toBe('game_not_active');
    });

    it('should return isVictory=false when not in interactive phase', () => {
      const gameState = createMockGameState({ currentPhase: 'line_formed' as GamePhase });
      const lps = createLpsTrackingState();
      setupLpsForVictory(lps, 1);

      const result = evaluateLpsVictory({
        gameState,
        lps,
        hasAnyRealAction: () => true,
        hasMaterial: () => true,
      });

      expect(result.isVictory).toBe(false);
      expect(result.reason).toBe('not_interactive_phase');
    });

    it('should return isVictory=false when insufficient consecutive rounds', () => {
      const gameState = createMockGameState();
      const lps = createLpsTrackingState();
      // Only 1 consecutive round - not enough for LPS victory
      lps.consecutiveExclusiveRounds = 1;
      lps.consecutiveExclusivePlayer = 1;

      const result = evaluateLpsVictory({
        gameState,
        lps,
        hasAnyRealAction: () => true,
        hasMaterial: () => true,
      });

      expect(result.isVictory).toBe(false);
      expect(result.reason).toMatch(/insufficient_consecutive_rounds/);
    });

    it('should return isVictory=false when no exclusive candidate', () => {
      const gameState = createMockGameState();
      const lps = createLpsTrackingState();
      // Has 2 rounds but no consecutive player (e.g., different players were exclusive)
      lps.consecutiveExclusiveRounds = 0;
      lps.consecutiveExclusivePlayer = null;

      const result = evaluateLpsVictory({
        gameState,
        lps,
        hasAnyRealAction: () => true,
        hasMaterial: () => true,
      });

      expect(result.isVictory).toBe(false);
      expect(result.reason).toMatch(/insufficient_consecutive_rounds/);
    });

    it('should return isVictory=false when not candidates turn', () => {
      const gameState = createMockGameState({ currentPlayer: 2 });
      const lps = createLpsTrackingState();
      setupLpsForVictory(lps, 1); // P1 is candidate but it's P2's turn

      const result = evaluateLpsVictory({
        gameState,
        lps,
        hasAnyRealAction: () => true,
        hasMaterial: () => true,
      });

      expect(result.isVictory).toBe(false);
      expect(result.reason).toBe('not_candidate_turn');
    });

    it('should return isVictory=false when candidate has no actions', () => {
      const gameState = createMockGameState({ currentPlayer: 1 });
      const lps = createLpsTrackingState();
      setupLpsForVictory(lps, 1);

      const result = evaluateLpsVictory({
        gameState,
        lps,
        hasAnyRealAction: (pn) => pn !== 1, // Candidate has no actions
        hasMaterial: () => true,
      });

      expect(result.isVictory).toBe(false);
      expect(result.reason).toBe('candidate_no_actions');
    });

    it('should return isVictory=false when other player with material has actions', () => {
      const gameState = createMockGameState({ currentPlayer: 1 });
      const lps = createLpsTrackingState();
      setupLpsForVictory(lps, 1);

      const result = evaluateLpsVictory({
        gameState,
        lps,
        hasAnyRealAction: () => true, // All players have actions
        hasMaterial: () => true,
      });

      expect(result.isVictory).toBe(false);
      expect(result.reason).toBe('other_player_has_actions');
    });

    it('should return isVictory=true when all conditions met (2 consecutive rounds)', () => {
      const gameState = createMockGameState({ currentPlayer: 1 });
      const lps = createLpsTrackingState();
      setupLpsForVictory(lps, 1);

      const result = evaluateLpsVictory({
        gameState,
        lps,
        hasAnyRealAction: (pn) => pn === 1, // Only candidate has actions
        hasMaterial: () => true,
      });

      expect(result.isVictory).toBe(true);
      expect(result.winner).toBe(1);
    });

    it('should skip players without material when checking other player actions', () => {
      const gameState = createMockGameState({ currentPlayer: 1 });
      const lps = createLpsTrackingState();
      setupLpsForVictory(lps, 1);

      const result = evaluateLpsVictory({
        gameState,
        lps,
        hasAnyRealAction: () => true, // All players would have actions
        hasMaterial: (pn) => pn === 1, // But only player 1 has material
      });

      expect(result.isVictory).toBe(true);
      expect(result.winner).toBe(1);
    });

    it('should handle 3-player game correctly', () => {
      const gameState = createMockGameState({
        currentPlayer: 2,
        players: [
          { playerNumber: 1 } as Player,
          { playerNumber: 2 } as Player,
          { playerNumber: 3 } as Player,
        ],
      });
      const lps = createLpsTrackingState();
      setupLpsForVictory(lps, 2);

      const result = evaluateLpsVictory({
        gameState,
        lps,
        hasAnyRealAction: (pn) => pn === 2,
        hasMaterial: () => true,
      });

      expect(result.isVictory).toBe(true);
      expect(result.winner).toBe(2);
    });

    it('should check all LPS-active phases', () => {
      const phases: GamePhase[] = ['ring_placement', 'movement', 'capture', 'chain_capture'];

      for (const phase of phases) {
        const gameState = createMockGameState({ currentPhase: phase, currentPlayer: 1 });
        const lps = createLpsTrackingState();
        setupLpsForVictory(lps, 1);

        const result = evaluateLpsVictory({
          gameState,
          lps,
          hasAnyRealAction: (pn) => pn === 1,
          hasMaterial: () => true,
        });

        expect(result.isVictory).toBe(true);
      }
    });

    it('should trigger victory only after two-round exclusive real-action pattern', () => {
      const lps = createLpsTrackingState();
      const activePlayers = [1, 2];

      const makeState = (overrides: Partial<GameState> = {}): GameState =>
        ({
          gameStatus: 'active',
          currentPhase: 'movement' as GamePhase,
          currentPlayer: 1,
          players: [{ playerNumber: 1 } as Player, { playerNumber: 2 } as Player],
          ...overrides,
        }) as unknown as GameState;

      // Round 1: P1 has real actions; P2 has none.
      updateLpsTracking(lps, {
        currentPlayer: 1,
        activePlayers,
        hasRealAction: true,
      });
      updateLpsTracking(lps, {
        currentPlayer: 2,
        activePlayers,
        hasRealAction: false,
      });

      // Before cycling back to P1, no completed round exists and no LPS victory is possible.
      let state = makeState({ currentPlayer: 1 });
      let result = evaluateLpsVictory({
        gameState: state,
        lps,
        hasAnyRealAction: (pn) => pn === 1,
        hasMaterial: () => true,
      });
      expect(result.isVictory).toBe(false);
      expect(result.reason).toMatch(/insufficient_consecutive_rounds/);

      // Start of round 2: cycling back to P1 finalises Round 1 with P1 as exclusive actor.
      // consecutiveExclusiveRounds is now 1.
      updateLpsTracking(lps, {
        currentPlayer: 1,
        activePlayers,
        hasRealAction: true,
      });

      // After 1 round, still not enough for LPS victory (requires 2)
      state = makeState({ currentPlayer: 1 });
      result = evaluateLpsVictory({
        gameState: state,
        lps,
        hasAnyRealAction: (pn) => pn === 1,
        hasMaterial: () => true,
      });
      expect(result.isVictory).toBe(false);
      expect(result.reason).toMatch(/insufficient_consecutive_rounds_1/);
      expect(lps.consecutiveExclusiveRounds).toBe(1);

      // Round 2: P1 has real actions; P2 has none.
      updateLpsTracking(lps, {
        currentPlayer: 2,
        activePlayers,
        hasRealAction: false,
      });

      // Start of round 3: cycling back to P1 finalises Round 2 with P1 as exclusive actor.
      // consecutiveExclusiveRounds is now 2 - LPS condition is now met!
      updateLpsTracking(lps, {
        currentPlayer: 1,
        activePlayers,
        hasRealAction: true,
      });

      expect(lps.consecutiveExclusiveRounds).toBe(2);
      expect(lps.consecutiveExclusivePlayer).toBe(1);

      state = makeState({ currentPlayer: 1 });
      result = evaluateLpsVictory({
        gameState: state,
        lps,
        hasAnyRealAction: (pn) => pn === 1, // only P1 has real actions
        hasMaterial: () => true,
      });

      expect(result.isVictory).toBe(true);
      expect(result.winner).toBe(1);

      // If another player with material also has real actions at that moment, LPS must not fire.
      const blockedResult = evaluateLpsVictory({
        gameState: state,
        lps,
        hasAnyRealAction: () => true, // both players appear to have real actions
        hasMaterial: () => true,
      });

      expect(blockedResult.isVictory).toBe(false);
      expect(blockedResult.reason).toBe('other_player_has_actions');
    });
  });

  describe('buildLpsVictoryResult', () => {
    const createMockGameState = (): GameState =>
      ({
        players: [
          { playerNumber: 1, territorySpaces: 5, eliminatedRings: 2 } as Player,
          { playerNumber: 2, territorySpaces: 3, eliminatedRings: 1 } as Player,
        ],
        board: {
          stacks: new Map([
            ['0,0', { controllingPlayer: 1, stackHeight: 3 }],
            ['1,1', { controllingPlayer: 1, stackHeight: 2 }],
            ['2,2', { controllingPlayer: 2, stackHeight: 4 }],
          ]),
        },
      }) as unknown as GameState;

    it('should set winner correctly', () => {
      const gameState = createMockGameState();
      const result = buildLpsVictoryResult(gameState, 1);
      expect(result.winner).toBe(1);
    });

    it('should set reason to last_player_standing', () => {
      const gameState = createMockGameState();
      const result = buildLpsVictoryResult(gameState, 1);
      expect(result.reason).toBe('last_player_standing');
    });

    it('should compute ringsRemaining from stacks', () => {
      const gameState = createMockGameState();
      const result = buildLpsVictoryResult(gameState, 1);

      expect(result.finalScore.ringsRemaining[1]).toBe(5); // 3 + 2
      expect(result.finalScore.ringsRemaining[2]).toBe(4);
    });

    it('should use territorySpaces from player state', () => {
      const gameState = createMockGameState();
      const result = buildLpsVictoryResult(gameState, 1);

      expect(result.finalScore.territorySpaces[1]).toBe(5);
      expect(result.finalScore.territorySpaces[2]).toBe(3);
    });

    it('should use eliminatedRings from player state', () => {
      const gameState = createMockGameState();
      const result = buildLpsVictoryResult(gameState, 1);

      expect(result.finalScore.ringsEliminated[1]).toBe(2);
      expect(result.finalScore.ringsEliminated[2]).toBe(1);
    });

    it('should handle player with no stacks', () => {
      const gameState = {
        players: [
          { playerNumber: 1, territorySpaces: 0, eliminatedRings: 5 } as Player,
          { playerNumber: 2, territorySpaces: 0, eliminatedRings: 0 } as Player,
        ],
        board: {
          stacks: new Map([['0,0', { controllingPlayer: 2, stackHeight: 10 }]]),
        },
      } as unknown as GameState;

      const result = buildLpsVictoryResult(gameState, 2);

      expect(result.finalScore.ringsRemaining[1]).toBe(0);
      expect(result.finalScore.ringsRemaining[2]).toBe(10);
    });

    it('should handle undefined territorySpaces', () => {
      const gameState = {
        players: [{ playerNumber: 1 } as Player],
        board: { stacks: new Map() },
      } as unknown as GameState;

      const result = buildLpsVictoryResult(gameState, 1);
      expect(result.finalScore.territorySpaces[1]).toBe(0);
    });

    it('should handle undefined eliminatedRings', () => {
      const gameState = {
        players: [{ playerNumber: 1 } as Player],
        board: { stacks: new Map() },
      } as unknown as GameState;

      const result = buildLpsVictoryResult(gameState, 1);
      expect(result.finalScore.ringsEliminated[1]).toBe(0);
    });

    it('should handle 4-player game', () => {
      const gameState = {
        players: [
          { playerNumber: 1, territorySpaces: 1, eliminatedRings: 1 } as Player,
          { playerNumber: 2, territorySpaces: 2, eliminatedRings: 2 } as Player,
          { playerNumber: 3, territorySpaces: 3, eliminatedRings: 3 } as Player,
          { playerNumber: 4, territorySpaces: 4, eliminatedRings: 4 } as Player,
        ],
        board: { stacks: new Map() },
      } as unknown as GameState;

      const result = buildLpsVictoryResult(gameState, 3);

      expect(result.winner).toBe(3);
      expect(result.finalScore.territorySpaces[3]).toBe(3);
      expect(result.finalScore.ringsEliminated[4]).toBe(4);
    });
  });
});
