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

  it('enriches LPS GameEndExplanation with ANM/FE context when forced_elimination moves occurred', () => {
    const state = createTestGameState();

    // Reuse the same bare-board LPS setup as the baseline test.
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

    // Record at least one forced_elimination move in structured history so the
    // explanation layer can classify this as an ANM/FE-heavy LPS sequence.
    (state.history as any).push({
      actor: 2,
      action: { type: 'forced_elimination' },
    } as any);

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

    expect(new Set(ctx.reasonCodes)).toEqual(
      new Set([
        'LAST_PLAYER_STANDING_EXCLUSIVE_REAL_ACTIONS',
        'ANM_MOVEMENT_FE_BLOCKED',
        'FE_SEQUENCE_CURRENT_PLAYER',
      ])
    );
    expect(ctx.primaryReasonCode).toBe('LAST_PLAYER_STANDING_EXCLUSIVE_REAL_ACTIONS');
    expect(new Set(ctx.rulesContextTags ?? [])).toEqual(
      new Set(['last_player_standing', 'anm_forced_elimination'])
    );
    expect(ctx.teachingTopicIds).toEqual(
      expect.arrayContaining([
        'teaching.victory_stalemate',
        'teaching.active_no_moves',
        'teaching.forced_elimination',
      ])
    );

    expect(explanation.telemetry).toBeDefined();
    const telemetry = explanation.telemetry!;

    expect(new Set(telemetry.weirdStateReasonCodes ?? [])).toEqual(
      new Set([
        'LAST_PLAYER_STANDING_EXCLUSIVE_REAL_ACTIONS',
        'ANM_MOVEMENT_FE_BLOCKED',
        'FE_SEQUENCE_CURRENT_PLAYER',
      ])
    );
    expect(new Set(telemetry.rulesContextTags ?? [])).toEqual(
      new Set(['last_player_standing', 'anm_forced_elimination'])
    );

    // LPS + ANM/FE should use FE-heavy LPS copy keys so the UI can explain
    // that forced elimination is recorded but does not count as a real move.
    expect(explanation.uxCopy.shortSummaryKey).toBe('game_end.lps.with_anm_fe.short');
    expect(explanation.uxCopy.detailedSummaryKey).toBe('game_end.lps.with_anm_fe.detailed');
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

    // Set territory via collapsedSpaces (the authoritative source for victory evaluation)
    // Player 1 controls 3 territory spaces, Player 2 controls 1
    state.board.collapsedSpaces.set('0,0', 1);
    state.board.collapsedSpaces.set('0,1', 1);
    state.board.collapsedSpaces.set('0,2', 1);
    state.board.collapsedSpaces.set('1,0', 2);

    // Also update player counters for consistency (though evaluation uses collapsedSpaces)
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

    // Structural stalemate tiebreak endings should use the dedicated copy keys.
    expect(explanation.uxCopy.shortSummaryKey).toBe('game_end.structural_stalemate.short');
    expect(explanation.uxCopy.detailedSummaryKey).toBe(
      'game_end.structural_stalemate.tiebreak.detailed'
    );

    expect(explanation.telemetry).toBeDefined();
    const telemetry = explanation.telemetry!;
    expect(telemetry.rulesContextTags).toEqual(['structural_stalemate']);
    expect(telemetry.weirdStateReasonCodes).toEqual(['STRUCTURAL_STALEMATE_TIEBREAK']);
  });

  it('enriches structural-stalemate explanation with ANM/FE context when forced_elimination moves occurred', () => {
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

    // Set territory via collapsedSpaces (the authoritative source for victory evaluation)
    // Player 1 controls 3 territory spaces, Player 2 controls 1
    state.board.collapsedSpaces.set('0,0', 1);
    state.board.collapsedSpaces.set('0,1', 1);
    state.board.collapsedSpaces.set('0,2', 1);
    state.board.collapsedSpaces.set('1,0', 2);

    // Also update player counters for consistency (though evaluation uses collapsedSpaces)
    state.players[0].territorySpaces = 3;
    state.players[1].territorySpaces = 1;

    // Mark that at least one forced_elimination move occurred earlier in the
    // game so the explanation layer can surface ANM/FE context alongside the
    // structural stalemate tiebreak.
    (state.history as any).push({
      actor: 1,
      action: { type: 'forced_elimination' },
    } as any);

    const victory = toVictoryState(state);

    // Aggregate reason remains 'territory_control' but explanation should
    // classify this as a structural stalemate tiebreak with ANM/FE details.
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
    expect(steps[0].kind).toBe('territory_spaces');

    expect(explanation.weirdStateContext).toBeDefined();
    const ctx = explanation.weirdStateContext!;

    expect(new Set(ctx.reasonCodes)).toEqual(
      new Set([
        'STRUCTURAL_STALEMATE_TIEBREAK',
        'ANM_MOVEMENT_FE_BLOCKED',
        'FE_SEQUENCE_CURRENT_PLAYER',
      ])
    );
    expect(ctx.primaryReasonCode).toBe('STRUCTURAL_STALEMATE_TIEBREAK');
    expect(new Set(ctx.rulesContextTags ?? [])).toEqual(
      new Set(['structural_stalemate', 'anm_forced_elimination'])
    );
    expect(ctx.teachingTopicIds).toEqual(
      expect.arrayContaining([
        'teaching.victory_stalemate',
        'teaching.active_no_moves',
        'teaching.forced_elimination',
      ])
    );

    expect(explanation.telemetry).toBeDefined();
    const telemetry = explanation.telemetry!;
    expect(new Set(telemetry.weirdStateReasonCodes ?? [])).toEqual(
      new Set([
        'STRUCTURAL_STALEMATE_TIEBREAK',
        'ANM_MOVEMENT_FE_BLOCKED',
        'FE_SEQUENCE_CURRENT_PLAYER',
      ])
    );
    expect(new Set(telemetry.rulesContextTags ?? [])).toEqual(
      new Set(['structural_stalemate', 'anm_forced_elimination'])
    );

    expect(explanation.uxCopy.shortSummaryKey).toBe('game_end.structural_stalemate.short');
    expect(explanation.uxCopy.detailedSummaryKey).toBe(
      'game_end.structural_stalemate.tiebreak.detailed'
    );
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
      expect(explanation.uxCopy.detailedSummaryKey).toBe('game_end.territory_mini_region.detailed');
    });

    it('identifies territory_mini_regions when winner has a small isolated region (≤4 cells)', () => {
      const state = createTestGameState();
      const p1 = state.players[0];

      // Set up territory victory threshold
      state.territoryVictoryThreshold = 3;
      state.victoryThreshold = 1000;

      p1.territorySpaces = 4;
      state.players[1].territorySpaces = 1;

      // Create a single small (3-cell) collapsed region for P1 to meet threshold
      // This is a "mini region" by size even though it's not disconnected
      // Victory evaluation uses board.collapsedSpaces as the authoritative source
      addCollapsedSpace(state.board, { x: 2, y: 2 }, 1);
      addCollapsedSpace(state.board, { x: 2, y: 3 }, 1);
      addCollapsedSpace(state.board, { x: 3, y: 2 }, 1); // 3rd space to meet threshold

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
      expect(explanation.uxCopy.detailedSummaryKey).toBe('game_end.territory_mini_region.detailed');
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

    it('does not trigger territory victory when winner has no collapsed territories', () => {
      const state = createTestGameState();
      const p1 = state.players[0];

      // Set up territory victory threshold
      state.territoryVictoryThreshold = 5;
      state.victoryThreshold = 1000;

      // Territory spaces via player aggregate but no collapsed spaces on board
      // Victory evaluation uses board.collapsedSpaces as the authoritative source,
      // so this should NOT trigger territory victory
      p1.territorySpaces = 6;
      state.players[1].territorySpaces = 1;

      // Need a stack on board to prevent structural stalemate
      addStack(state.board, { x: 4, y: 4 }, 1, 2);

      const victory = toVictoryState(state);

      // No collapsed spaces means territory threshold not met
      // Game continues (not over)
      expect(victory.isGameOver).toBe(false);
    });

    it('handles Q23-style scenario with self-elimination prerequisite correctly', () => {
      // This test mirrors the Q23 scenario from the rules where a player achieves
      // territory victory via a contiguous 5-cell region (meeting threshold of 5).
      // A 5-cell contiguous region is NOT a mini-region (only regions ≤4 cells are),
      // so this is a standard territory victory without mini-region flagging.
      const state = createTestGameState();
      const p1 = state.players[0];

      // Set up territory victory threshold
      state.territoryVictoryThreshold = 5;
      state.victoryThreshold = 1000;

      p1.territorySpaces = 8;
      state.players[1].territorySpaces = 2;

      // Create a 2×2 region at (2,2)–(3,3) plus one additional space
      // This represents a Q23-style collapsed region with 5 spaces (meeting threshold)
      // Victory evaluation uses board.collapsedSpaces as the authoritative source
      addCollapsedSpace(state.board, { x: 2, y: 2 }, 1);
      addCollapsedSpace(state.board, { x: 2, y: 3 }, 1);
      addCollapsedSpace(state.board, { x: 3, y: 2 }, 1);
      addCollapsedSpace(state.board, { x: 3, y: 3 }, 1);
      addCollapsedSpace(state.board, { x: 4, y: 2 }, 1); // 5th space to meet threshold

      // Add an "outside stack" at (0,0) per Q23 prerequisite
      addStack(state.board, { x: 0, y: 0 }, 1, 3);

      const victory = toVictoryState(state);

      expect(victory.isGameOver).toBe(true);
      expect(victory.reason).toBe('territory_control');
      expect(victory.winner).toBe(1);

      const explanation = victory.gameEndExplanation!;

      // 5-cell contiguous region exceeds mini-region threshold (≤4 cells),
      // so this is a normal territory victory without weird-state context
      expect(explanation.outcomeType).toBe('territory_control');
      expect(explanation.victoryReasonCode).toBe('victory_territory_majority');
      expect(explanation.primaryConceptId).toBeUndefined();
      expect(explanation.weirdStateContext).toBeUndefined();
      expect(explanation.uxCopy.shortSummaryKey).toBe('game_end.territory_control.short');
    });
  });
});
