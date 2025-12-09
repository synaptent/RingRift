import { buildGameEndExplanation } from '../../src/shared/engine/gameEndExplanation';
import type { GameEndExplanationSource } from '../../src/shared/engine/gameEndExplanation';

describe('buildGameEndExplanation', () => {
  it('builds ring-majority explanation without weird state or telemetry', () => {
    const source: GameEndExplanationSource = {
      gameId: 'g_square8_2p_001',
      boardType: 'square8',
      numPlayers: 2,
      winnerPlayerId: 'P1',
      outcomeType: 'ring_elimination',
      victoryReasonCode: 'victory_ring_majority',
      scoreBreakdown: {
        P1: {
          playerId: 'P1',
          eliminatedRings: 19,
          territorySpaces: 4,
          markers: 12,
        },
        P2: {
          playerId: 'P2',
          eliminatedRings: 6,
          territorySpaces: 2,
          markers: 10,
        },
      },
      uxCopy: {
        shortSummaryKey: 'game_end.ring_elimination.short',
      },
    };

    const explanation = buildGameEndExplanation(source);

    expect(explanation.gameId).toBe(source.gameId);
    expect(explanation.boardType).toBe(source.boardType);
    expect(explanation.numPlayers).toBe(source.numPlayers);
    expect(explanation.winnerPlayerId).toBe(source.winnerPlayerId);
    expect(explanation.outcomeType).toBe(source.outcomeType);
    expect(explanation.victoryReasonCode).toBe(source.victoryReasonCode);
    expect(explanation.scoreBreakdown).toEqual(source.scoreBreakdown);
    expect(explanation.weirdStateContext).toBeUndefined();
    expect(explanation.teaching).toBeUndefined();
    expect(explanation.telemetry).toBeUndefined();
  });

  it('includes weird-state context and telemetry for ANM/FE-involved LPS ending', () => {
    const source: GameEndExplanationSource = {
      gameId: 'g_square8_3p_lps_anm_fe',
      boardType: 'square8',
      numPlayers: 3,
      winnerPlayerId: 'P2',
      outcomeType: 'last_player_standing',
      victoryReasonCode: 'victory_last_player_standing',
      primaryConceptId: 'lps_real_actions',
      scoreBreakdown: {
        P1: {
          playerId: 'P1',
          eliminatedRings: 8,
          territorySpaces: 0,
          markers: 5,
        },
        P2: {
          playerId: 'P2',
          eliminatedRings: 10,
          territorySpaces: 3,
          markers: 7,
        },
        P3: {
          playerId: 'P3',
          eliminatedRings: 4,
          territorySpaces: 0,
          markers: 2,
        },
      },
      weirdStateContext: {
        reasonCodes: [
          'ANM_MOVEMENT_FE_BLOCKED',
          'FE_SEQUENCE_CURRENT_PLAYER',
          'LAST_PLAYER_STANDING_EXCLUSIVE_REAL_ACTIONS',
        ],
        primaryReasonCode: 'LAST_PLAYER_STANDING_EXCLUSIVE_REAL_ACTIONS',
        rulesContextTags: ['anm_forced_elimination', 'last_player_standing'],
        teachingTopicIds: [
          'teaching.active_no_moves',
          'teaching.forced_elimination',
          'teaching.victory_stalemate',
        ],
      },
      telemetryTags: ['anm_forced_elimination'],
      uxCopy: {
        shortSummaryKey: 'game_end.lps.short',
        detailedSummaryKey: 'game_end.lps.with_anm_fe.detailed',
      },
    };

    const explanation = buildGameEndExplanation(source);

    expect(explanation.weirdStateContext).toEqual(source.weirdStateContext);
    expect(explanation.telemetry).toBeDefined();

    const telemetry = explanation.telemetry!;
    expect(telemetry.weirdStateReasonCodes).toBeDefined();
    expect(telemetry.weirdStateReasonCodes).toEqual(
      expect.arrayContaining(source.weirdStateContext!.reasonCodes)
    );
    expect(new Set(telemetry.weirdStateReasonCodes)).toEqual(
      new Set(source.weirdStateContext!.reasonCodes)
    );

    expect(telemetry.rulesContextTags).toBeDefined();
    // rulesContextTags come from both weirdStateContext.rulesContextTags and telemetryTags
    const expectedTags = ['anm_forced_elimination', 'last_player_standing'];
    expect(new Set(telemetry.rulesContextTags)).toEqual(new Set(expectedTags));
  });

  it('preserves tiebreak steps and telemetry tags for structural stalemate ending', () => {
    const tiebreakSteps = [
      {
        kind: 'territory_spaces' as const,
        winnerPlayerId: 'P3',
        valuesByPlayer: { P1: 70, P2: 65, P3: 72 },
      },
    ];

    const source: GameEndExplanationSource = {
      gameId: 'g_square19_3p_stalemate',
      boardType: 'square19',
      numPlayers: 3,
      winnerPlayerId: 'P3',
      outcomeType: 'structural_stalemate',
      victoryReasonCode: 'victory_structural_stalemate_tiebreak',
      primaryConceptId: 'structural_stalemate',
      scoreBreakdown: {
        P1: {
          playerId: 'P1',
          eliminatedRings: 30,
          territorySpaces: 70,
          markers: 5,
        },
        P2: {
          playerId: 'P2',
          eliminatedRings: 28,
          territorySpaces: 65,
          markers: 4,
        },
        P3: {
          playerId: 'P3',
          eliminatedRings: 29,
          territorySpaces: 72,
          markers: 6,
        },
      },
      tiebreakSteps,
      telemetryTags: ['structural_stalemate'],
      uxCopy: {
        shortSummaryKey: 'game_end.structural_stalemate.short',
        detailedSummaryKey: 'game_end.structural_stalemate.tiebreak.detailed',
      },
    };

    const explanation = buildGameEndExplanation(source);

    expect(explanation.outcomeType).toBe('structural_stalemate');
    expect(explanation.victoryReasonCode).toBe('victory_structural_stalemate_tiebreak');
    expect(explanation.tiebreakSteps).toEqual(tiebreakSteps);
    expect(explanation.telemetry).toBeDefined();
    expect(explanation.telemetry!.rulesContextTags).toEqual(['structural_stalemate']);
    expect(explanation.telemetry!.weirdStateReasonCodes).toBeUndefined();
  });

  it('builds territory-control explanation and applies telemetry tags', () => {
    const source: GameEndExplanationSource = {
      gameId: 'g_square19_2p_territory_majority',
      boardType: 'square19',
      numPlayers: 2,
      winnerPlayerId: 'P1',
      outcomeType: 'territory_control',
      victoryReasonCode: 'victory_territory_majority',
      scoreBreakdown: {
        P1: {
          playerId: 'P1',
          eliminatedRings: 10,
          territorySpaces: 190,
          markers: 8,
        },
        P2: {
          playerId: 'P2',
          eliminatedRings: 4,
          territorySpaces: 170,
          markers: 6,
        },
      },
      telemetryTags: ['territory_control'],
      uxCopy: {
        shortSummaryKey: 'game_end.territory_control.short',
        detailedSummaryKey: 'game_end.territory_control.detailed',
      },
    };

    const explanation = buildGameEndExplanation(source);

    expect(explanation.outcomeType).toBe('territory_control');
    expect(explanation.victoryReasonCode).toBe('victory_territory_majority');
    expect(explanation.boardType).toBe('square19');
    expect(explanation.scoreBreakdown).toEqual(source.scoreBreakdown);
    expect(explanation.telemetry?.rulesContextTags).toEqual(['territory_control']);
  });
});
