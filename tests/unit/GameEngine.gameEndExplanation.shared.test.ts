import { toVictoryState } from '../../src/shared/engine/orchestration/turnOrchestrator';
import type { GameState, BoardState, Position } from '../../src/shared/types/game';
import { createTestGameState, addMarker, addStack } from '../utils/fixtures';
import { positionToString } from '../../src/shared/types/game';

describe('shared engine – GameEndExplanation wiring', () => {
  it('attaches ring-majority GameEndExplanation on ring-elimination victory', () => {
    const state = createTestGameState();
    const p1 = state.players[0];

    state.victoryThreshold = 5;
    p1.eliminatedRings = 5;
    state.players[1].eliminatedRings = 0;

    // Victory is purely threshold-based here; phase is irrelevant for evaluateVictory.
    const victory = toVictoryState(state);

    expect(victory.isGameOver).toBe(true);
    expect(victory.reason).toBe('ring_elimination');
    expect(victory.gameEndExplanation).toBeDefined();

    const explanation = victory.gameEndExplanation!;

    expect(explanation.outcomeType).toBe('ring_elimination');
    expect(explanation.victoryReasonCode).toBe('victory_ring_majority');
    expect(explanation.winnerPlayerId).toBe('P1');
    expect(explanation.boardType).toBe(state.boardType);
    expect(explanation.numPlayers).toBe(state.players.length);

    // No weird-state or telemetry context for simple ring-majority endings.
    expect(explanation.weirdStateContext).toBeUndefined();
    expect(explanation.telemetry).toBeUndefined();
    expect(explanation.teaching).toBeUndefined();
  });

  it('attaches LPS GameEndExplanation with weird-state context for last_player_standing endings', () => {
    const state = createTestGameState();

    // Bare-board structural situation mirroring victory.shared marker tiebreak test.
    state.board.stacks.clear();
    state.board.markers.clear();
    state.players.forEach((p) => {
      p.ringsInHand = 0;
      p.territorySpaces = 0;
      p.eliminatedRings = 2;
    });

    state.victoryThreshold = 1000;
    state.territoryVictoryThreshold = 1000;

    // Two markers for Player 1, one for Player 2 -> marker tiebreak → last_player_standing.
    addMarker(state.board, { x: 0, y: 0 }, 1);
    addMarker(state.board, { x: 1, y: 0 }, 1);
    addMarker(state.board, { x: 0, y: 1 }, 2);

    // Structural bare-board LPS via marker tiebreak; call victory helper directly.
    const victory = toVictoryState(state);

    expect(victory.isGameOver).toBe(true);
    expect(victory.reason).toBe('last_player_standing');
    expect(victory.winner).toBe(1);
    expect(victory.gameEndExplanation).toBeDefined();

    const explanation = victory.gameEndExplanation!;

    expect(explanation.outcomeType).toBe('last_player_standing');
    expect(explanation.victoryReasonCode).toBe('victory_last_player_standing');
    expect(explanation.primaryConceptId).toBe('lps_real_actions');

    expect(explanation.weirdStateContext).toBeDefined();
    const ctx = explanation.weirdStateContext!;

    expect(ctx.reasonCodes).toContain('LAST_PLAYER_STANDING_EXCLUSIVE_REAL_ACTIONS');
    expect(ctx.primaryReasonCode).toBe('LAST_PLAYER_STANDING_EXCLUSIVE_REAL_ACTIONS');
    expect(ctx.rulesContextTags).toContain('last_player_standing');
    expect(ctx.teachingTopicIds).toContain('teaching.victory_stalemate');

    expect(explanation.telemetry).toBeDefined();
    const telemetry = explanation.telemetry!;

    expect(telemetry.weirdStateReasonCodes).toEqual(
      expect.arrayContaining(['LAST_PLAYER_STANDING_EXCLUSIVE_REAL_ACTIONS'])
    );
    expect(telemetry.rulesContextTags).toEqual(expect.arrayContaining(['last_player_standing']));
  });

  it('attaches structural-stalemate GameEndExplanation with territory tiebreak details', () => {
    const state = createTestGameState();

    // Bare-board global stalemate with differing territory counts, mirroring
    // the victory.shared territory tiebreak test.
    state.board.stacks.clear();
    state.players.forEach((p) => {
      p.ringsInHand = 0;
      p.territorySpaces = 0;
      p.eliminatedRings = 0;
    });

    state.victoryThreshold = 1000;
    state.territoryVictoryThreshold = 1000;

    state.players[0].territorySpaces = 3;
    state.players[1].territorySpaces = 1;

    // Bare-board territory tiebreak; invoke victory helper directly.
    const victory = toVictoryState(state);

    // Aggregate reason remains 'territory_control' but explanation should
    // classify this as a structural stalemate tiebreak.
    expect(victory.isGameOver).toBe(true);
    expect(victory.reason).toBe('territory_control');
    expect(victory.winner).toBe(1);
    expect(victory.gameEndExplanation).toBeDefined();

    const explanation = victory.gameEndExplanation!;

    expect(explanation.outcomeType).toBe('structural_stalemate');
    expect(explanation.victoryReasonCode).toBe('victory_structural_stalemate_tiebreak');
    expect(explanation.primaryConceptId).toBe('structural_stalemate');

    expect(explanation.tiebreakSteps).toBeDefined();
    const steps = explanation.tiebreakSteps!;
    expect(steps.length).toBeGreaterThanOrEqual(1);

    const firstStep = steps[0];
    expect(firstStep.kind).toBe('territory_spaces');
    expect(firstStep.winnerPlayerId).toBe('P1');
    expect(firstStep.valuesByPlayer.P1).toBe(3);
    expect(firstStep.valuesByPlayer.P2).toBe(1);

    expect(explanation.weirdStateContext).toBeDefined();
    const ctx = explanation.weirdStateContext!;
    expect(ctx.reasonCodes).toContain('STRUCTURAL_STALEMATE_TIEBREAK');
    expect(ctx.rulesContextTags).toContain('structural_stalemate');

    expect(explanation.telemetry).toBeDefined();
    const telemetry = explanation.telemetry!;
    expect(telemetry.rulesContextTags).toEqual(['structural_stalemate']);
    expect(telemetry.weirdStateReasonCodes).toEqual(['STRUCTURAL_STALEMATE_TIEBREAK']);
  });

  describe('territory mini-region detection', () => {
    /**
     * Helper to add a collapsed space to the board.
     */
    function addCollapsedSpace(board: BoardState, pos: Position, owner: number): void {
      const key = positionToString(pos);
      board.collapsedSpaces.set(key, owner);
    }

    it('identifies territory_mini_regions when winner has 2+ disconnected collapsed regions', () => {
      const state = createTestGameState();
      const p1 = state.players[0];

      // Set up territory victory threshold
      state.territoryVictoryThreshold = 8;
      state.victoryThreshold = 1000; // Prevent ring-elimination victory

      // Give P1 enough territory spaces to trigger victory
      p1.territorySpaces = 10;
      state.players[1].territorySpaces = 2;

      // Create two disconnected collapsed regions for P1:
      // Region 1: (0,0), (0,1), (1,0), (1,1) - 4 cells
      addCollapsedSpace(state.board, { x: 0, y: 0 }, 1);
      addCollapsedSpace(state.board, { x: 0, y: 1 }, 1);
      addCollapsedSpace(state.board, { x: 1, y: 0 }, 1);
      addCollapsedSpace(state.board, { x: 1, y: 1 }, 1);

      // Region 2: (5,5), (5,6), (6,5), (6,6) - 4 cells, disconnected from Region 1
      addCollapsedSpace(state.board, { x: 5, y: 5 }, 1);
      addCollapsedSpace(state.board, { x: 5, y: 6 }, 1);
      addCollapsedSpace(state.board, { x: 6, y: 5 }, 1);
      addCollapsedSpace(state.board, { x: 6, y: 6 }, 1);

      // Need a stack on board to prevent structural stalemate
      addStack(state.board, { x: 3, y: 3 }, 1, 2);

      const victory = toVictoryState(state);

      expect(victory.isGameOver).toBe(true);
      expect(victory.reason).toBe('territory_control');
      expect(victory.winner).toBe(1);
      expect(victory.gameEndExplanation).toBeDefined();

      const explanation = victory.gameEndExplanation!;

      expect(explanation.outcomeType).toBe('territory_control');
      expect(explanation.victoryReasonCode).toBe('victory_territory_majority');
      expect(explanation.primaryConceptId).toBe('territory_mini_regions');

      // Verify weird state context is set for mini-region
      expect(explanation.weirdStateContext).toBeDefined();
      const ctx = explanation.weirdStateContext!;
      expect(ctx.rulesContextTags).toContain('territory_mini_region');
      expect(ctx.teachingTopicIds).toContain('teaching.territory');

      // Verify telemetry tags
      expect(explanation.telemetry).toBeDefined();
      expect(explanation.telemetry!.rulesContextTags).toContain('territory_mini_region');

      // Verify UX copy key is mini-region specific
      expect(explanation.uxCopy.shortSummaryKey).toBe('game_end.territory_mini_region.short');
    });

    it('identifies territory_mini_regions when winner has a small isolated region (≤4 cells)', () => {
      const state = createTestGameState();
      const p1 = state.players[0];

      // Set up territory victory threshold
      state.territoryVictoryThreshold = 3;
      state.victoryThreshold = 1000;

      p1.territorySpaces = 4;
      state.players[1].territorySpaces = 1;

      // Create a single small (2-cell) collapsed region for P1
      // This is a "mini region" by size even though it's not disconnected
      addCollapsedSpace(state.board, { x: 2, y: 2 }, 1);
      addCollapsedSpace(state.board, { x: 2, y: 3 }, 1);

      // Need a stack on board to prevent structural stalemate
      addStack(state.board, { x: 4, y: 4 }, 1, 2);

      const victory = toVictoryState(state);

      expect(victory.isGameOver).toBe(true);
      expect(victory.reason).toBe('territory_control');
      expect(victory.gameEndExplanation).toBeDefined();

      const explanation = victory.gameEndExplanation!;

      // Should detect as mini-region because the single region is ≤4 cells
      expect(explanation.primaryConceptId).toBe('territory_mini_regions');
      expect(explanation.uxCopy.shortSummaryKey).toBe('game_end.territory_mini_region.short');
    });

    it('does not flag mini-region when winner has single large contiguous territory', () => {
      const state = createTestGameState();
      const p1 = state.players[0];

      // Set up territory victory threshold
      state.territoryVictoryThreshold = 5;
      state.victoryThreshold = 1000;

      p1.territorySpaces = 6;
      state.players[1].territorySpaces = 1;

      // Create a single 3x2 collapsed region (6 cells) - larger than threshold
      addCollapsedSpace(state.board, { x: 0, y: 0 }, 1);
      addCollapsedSpace(state.board, { x: 0, y: 1 }, 1);
      addCollapsedSpace(state.board, { x: 0, y: 2 }, 1);
      addCollapsedSpace(state.board, { x: 1, y: 0 }, 1);
      addCollapsedSpace(state.board, { x: 1, y: 1 }, 1);
      addCollapsedSpace(state.board, { x: 1, y: 2 }, 1);

      // Need a stack on board to prevent structural stalemate
      addStack(state.board, { x: 4, y: 4 }, 1, 2);

      const victory = toVictoryState(state);

      expect(victory.isGameOver).toBe(true);
      expect(victory.reason).toBe('territory_control');
      expect(victory.gameEndExplanation).toBeDefined();

      const explanation = victory.gameEndExplanation!;

      // Should NOT be flagged as mini-region - single large contiguous region
      expect(explanation.primaryConceptId).toBeUndefined();
      expect(explanation.uxCopy.shortSummaryKey).toBe('game_end.territory_control.short');
      expect(explanation.weirdStateContext).toBeUndefined();
    });

    it('does not flag mini-region when winner has no collapsed territories', () => {
      const state = createTestGameState();
      const p1 = state.players[0];

      // Set up territory victory threshold
      state.territoryVictoryThreshold = 5;
      state.victoryThreshold = 1000;

      // Territory spaces via player aggregate but no collapsed spaces
      p1.territorySpaces = 6;
      state.players[1].territorySpaces = 1;

      // Need a stack on board to prevent structural stalemate
      addStack(state.board, { x: 4, y: 4 }, 1, 2);

      const victory = toVictoryState(state);

      expect(victory.isGameOver).toBe(true);
      expect(victory.reason).toBe('territory_control');
      expect(victory.gameEndExplanation).toBeDefined();

      const explanation = victory.gameEndExplanation!;

      // No collapsed spaces = no mini-region detection
      expect(explanation.primaryConceptId).toBeUndefined();
      expect(explanation.uxCopy.shortSummaryKey).toBe('game_end.territory_control.short');
    });

    it('handles Q23-style scenario with self-elimination prerequisite correctly', () => {
      // This test mirrors the Q23 mini-region scenario from the rules
      const state = createTestGameState();
      const p1 = state.players[0];

      // Set up territory victory threshold
      state.territoryVictoryThreshold = 5;
      state.victoryThreshold = 1000;

      p1.territorySpaces = 8;
      state.players[1].territorySpaces = 2;

      // Create a 2×2 mini-region at (2,2)–(3,3)
      // This represents a Q23-style collapsed region
      addCollapsedSpace(state.board, { x: 2, y: 2 }, 1);
      addCollapsedSpace(state.board, { x: 2, y: 3 }, 1);
      addCollapsedSpace(state.board, { x: 3, y: 2 }, 1);
      addCollapsedSpace(state.board, { x: 3, y: 3 }, 1);

      // Add an "outside stack" at (0,0) per Q23 prerequisite
      addStack(state.board, { x: 0, y: 0 }, 1, 3);

      const victory = toVictoryState(state);

      expect(victory.isGameOver).toBe(true);
      expect(victory.reason).toBe('territory_control');
      expect(victory.winner).toBe(1);

      const explanation = victory.gameEndExplanation!;

      // Q23-style: small isolated region triggers mini-region detection
      expect(explanation.primaryConceptId).toBe('territory_mini_regions');
      expect(explanation.weirdStateContext).toBeDefined();
      expect(explanation.weirdStateContext!.rulesContextTags).toContain('territory_mini_region');
      expect(explanation.uxCopy.shortSummaryKey).toBe('game_end.territory_mini_region.short');
    });
  });
});
