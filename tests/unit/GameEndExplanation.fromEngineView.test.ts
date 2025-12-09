import { buildGameEndExplanationFromEngineView } from '../../src/shared/engine/gameEndExplanation';
import type { GameEndEngineView } from '../../src/shared/engine/gameEndExplanation';

describe('buildGameEndExplanationFromEngineView', () => {
  it('builds ring-majority explanation from minimal engine view', () => {
    const view: GameEndEngineView = {
      gameId: 'g_square8_2p_ring_majority',
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
    };

    const explanation = buildGameEndExplanationFromEngineView(view, {
      uxCopy: {
        shortSummaryKey: 'game_end.ring_majority.short',
      },
    });

    expect(explanation.gameId).toBe(view.gameId);
    expect(explanation.boardType).toBe(view.boardType);
    expect(explanation.numPlayers).toBe(view.numPlayers);
    expect(explanation.winnerPlayerId).toBe(view.winnerPlayerId);
    expect(explanation.outcomeType).toBe(view.outcomeType);
    expect(explanation.victoryReasonCode).toBe(view.victoryReasonCode);
    expect(explanation.scoreBreakdown).toEqual(view.scoreBreakdown);

    expect(explanation.primaryConceptId).toBeUndefined();
    expect(explanation.weirdStateContext).toBeUndefined();
    expect(explanation.teaching).toBeUndefined();
    expect(explanation.telemetry).toBeUndefined();
  });

  it('carries ANM/FE LPS weird-state context and merges telemetry tags', () => {
    const view: GameEndEngineView = {
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
      },
    };

    const explanation = buildGameEndExplanationFromEngineView(view, {
      teaching: {
        teachingTopics: ['lps_real_actions'],
        recommendedFlows: ['fe_loop_intro'],
      },
      telemetryTags: ['anm_forced_elimination'],
      uxCopy: {
        shortSummaryKey: 'game_end.lps.with_anm_fe.short',
        detailedSummaryKey: 'game_end.lps.with_anm_fe.detailed',
      },
    });

    expect(explanation.outcomeType).toBe('last_player_standing');
    expect(explanation.victoryReasonCode).toBe('victory_last_player_standing');
    expect(explanation.primaryConceptId).toBe('lps_real_actions');
    expect(explanation.weirdStateContext).toEqual(view.weirdStateContext);
    expect(explanation.teaching).toEqual({
      teachingTopics: ['lps_real_actions'],
      recommendedFlows: ['fe_loop_intro'],
    });

    expect(explanation.telemetry).toBeDefined();
    const telemetry = explanation.telemetry!;

    expect(telemetry.weirdStateReasonCodes).toBeDefined();
    expect(new Set(telemetry.weirdStateReasonCodes)).toEqual(
      new Set(view.weirdStateContext!.reasonCodes)
    );

    expect(telemetry.rulesContextTags).toBeDefined();
    const expectedTags = ['anm_forced_elimination', 'last_player_standing'];
    expect(new Set(telemetry.rulesContextTags)).toEqual(new Set(expectedTags));
  });

  it('preserves tiebreak steps and structural stalemate telemetry', () => {
    const tiebreakSteps = [
      {
        kind: 'territory_spaces' as const,
        winnerPlayerId: 'P3',
        valuesByPlayer: { P1: 70, P2: 65, P3: 72 },
      },
    ];

    const view: GameEndEngineView = {
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
      weirdStateContext: {
        reasonCodes: ['STRUCTURAL_STALEMATE_TIEBREAK'],
        primaryReasonCode: 'STRUCTURAL_STALEMATE_TIEBREAK',
        rulesContextTags: ['structural_stalemate'],
      },
    };

    const explanation = buildGameEndExplanationFromEngineView(view, {
      telemetryTags: ['structural_stalemate'],
      uxCopy: {
        shortSummaryKey: 'game_end.structural_stalemate.short',
        detailedSummaryKey: 'game_end.structural_stalemate.tiebreak.detailed',
      },
    });

    expect(explanation.outcomeType).toBe('structural_stalemate');
    expect(explanation.victoryReasonCode).toBe('victory_structural_stalemate_tiebreak');
    expect(explanation.primaryConceptId).toBe('structural_stalemate');
    expect(explanation.tiebreakSteps).toEqual(tiebreakSteps);

    expect(explanation.telemetry).toBeDefined();
    const telemetry = explanation.telemetry!;

    expect(telemetry.rulesContextTags).toBeDefined();
    expect(telemetry.rulesContextTags).toEqual(['structural_stalemate']);
    expect(telemetry.weirdStateReasonCodes).toEqual(['STRUCTURAL_STALEMATE_TIEBREAK']);
  });

  it('wires through territory mini-region concept and telemetry', () => {
    const view: GameEndEngineView = {
      gameId: 'g_square8_2p_territory_mini_region',
      boardType: 'square8',
      numPlayers: 2,
      winnerPlayerId: 'P1',
      outcomeType: 'territory_control',
      victoryReasonCode: 'victory_territory_majority',
      primaryConceptId: 'territory_mini_regions',
      scoreBreakdown: {
        P1: {
          playerId: 'P1',
          eliminatedRings: 12,
          territorySpaces: 20,
          markers: 9,
        },
        P2: {
          playerId: 'P2',
          eliminatedRings: 15,
          territorySpaces: 12,
          markers: 7,
        },
      },
      weirdStateContext: {
        reasonCodes: ['ANM_TERRITORY_NO_ACTIONS'],
        primaryReasonCode: 'ANM_TERRITORY_NO_ACTIONS',
        rulesContextTags: ['territory_mini_region'],
      },
    };

    const explanation = buildGameEndExplanationFromEngineView(view, {
      telemetryTags: ['territory_mini_region'],
      uxCopy: {
        shortSummaryKey: 'game_end.territory_mini_region.short',
        detailedSummaryKey: 'game_end.territory_mini_region.detailed',
      },
    });

    expect(explanation.outcomeType).toBe('territory_control');
    expect(explanation.victoryReasonCode).toBe('victory_territory_majority');
    expect(explanation.primaryConceptId).toBe('territory_mini_regions');
    expect(explanation.weirdStateContext).toEqual(view.weirdStateContext);

    expect(explanation.telemetry).toBeDefined();
    const telemetry = explanation.telemetry!;

    expect(new Set(telemetry.rulesContextTags)).toEqual(new Set(['territory_mini_region']));
    expect(telemetry.weirdStateReasonCodes).toEqual(['ANM_TERRITORY_NO_ACTIONS']);
  });

  it('builds territory-control explanation with provided telemetry tags', () => {
    const view: GameEndEngineView = {
      gameId: 'g_square19_2p_territory_control',
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
    };

    const explanation = buildGameEndExplanationFromEngineView(view, {
      telemetryTags: ['territory_control'],
      uxCopy: {
        shortSummaryKey: 'game_end.territory_control.short',
        detailedSummaryKey: 'game_end.territory_control.detailed',
      },
    });

    expect(explanation.outcomeType).toBe('territory_control');
    expect(explanation.victoryReasonCode).toBe('victory_territory_majority');
    expect(explanation.winnerPlayerId).toBe('P1');
    expect(explanation.boardType).toBe('square19');
    expect(explanation.scoreBreakdown).toEqual(view.scoreBreakdown);
    expect(explanation.telemetry?.rulesContextTags).toEqual(['territory_control']);
  });
});
