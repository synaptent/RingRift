/**
 * Branch coverage tests for weirdStateReasons.ts
 *
 * This file provides comprehensive test coverage for the weird state reason
 * mapping functions used for UX and telemetry, aligned with
 * docs/UX_RULES_WEIRD_STATES_SPEC.md.
 *
 * Functions tested:
 * - getWeirdStateReasonForType: Maps RulesUxWeirdStateType to WeirdStateReasonInfo
 * - getWeirdStateReasonForGameResult: Maps GameResult to WeirdStateReasonInfo
 * - getRulesContextForReason: Maps RulesWeirdStateReasonCode to RulesUxContext
 * - getTeachingTopicForReason: Maps RulesWeirdStateReasonCode to WeirdStateTeachingTopicId
 */

import {
  getWeirdStateReasonForType,
  getWeirdStateReasonForGameResult,
  getRulesContextForReason,
  getWeirdStateTypeForReason,
  getTeachingTopicForReason,
  type RulesWeirdStateReasonCode,
  type WeirdStateReasonInfo,
  type WeirdStateTeachingTopicId,
} from '../../src/shared/engine/weirdStateReasons';

import type {
  RulesUxWeirdStateType,
  RulesUxContext,
} from '../../src/shared/telemetry/rulesUxEvents';
import type { GameResult } from '../../src/shared/types/game';

describe('weirdStateReasons', () => {
  // =========================================================================
  // getWeirdStateReasonForType
  // =========================================================================

  describe('getWeirdStateReasonForType', () => {
    it('maps active-no-moves-movement to ANM_MOVEMENT_FE_BLOCKED', () => {
      const result = getWeirdStateReasonForType('active-no-moves-movement');

      expect(result).toEqual({
        reasonCode: 'ANM_MOVEMENT_FE_BLOCKED',
        rulesContext: 'anm_forced_elimination',
        weirdStateType: 'active-no-moves-movement',
      });
    });

    it('maps active-no-moves-line to ANM_LINE_NO_ACTIONS', () => {
      const result = getWeirdStateReasonForType('active-no-moves-line');

      expect(result).toEqual({
        reasonCode: 'ANM_LINE_NO_ACTIONS',
        rulesContext: 'line_reward_exact',
        weirdStateType: 'active-no-moves-line',
      });
    });

    it('maps active-no-moves-territory to ANM_TERRITORY_NO_ACTIONS', () => {
      const result = getWeirdStateReasonForType('active-no-moves-territory');

      expect(result).toEqual({
        reasonCode: 'ANM_TERRITORY_NO_ACTIONS',
        rulesContext: 'territory_mini_region',
        weirdStateType: 'active-no-moves-territory',
      });
    });

    it('maps forced-elimination to FE_SEQUENCE_CURRENT_PLAYER', () => {
      const result = getWeirdStateReasonForType('forced-elimination');

      expect(result).toEqual({
        reasonCode: 'FE_SEQUENCE_CURRENT_PLAYER',
        rulesContext: 'anm_forced_elimination',
        weirdStateType: 'forced-elimination',
      });
    });

    it('maps structural-stalemate to STRUCTURAL_STALEMATE_TIEBREAK', () => {
      const result = getWeirdStateReasonForType('structural-stalemate');

      expect(result).toEqual({
        reasonCode: 'STRUCTURAL_STALEMATE_TIEBREAK',
        rulesContext: 'structural_stalemate',
        weirdStateType: 'structural-stalemate',
      });
    });

    it('maps last-player-standing to LAST_PLAYER_STANDING_EXCLUSIVE_REAL_ACTIONS', () => {
      const result = getWeirdStateReasonForType('last-player-standing');

      expect(result).toEqual({
        reasonCode: 'LAST_PLAYER_STANDING_EXCLUSIVE_REAL_ACTIONS',
        rulesContext: 'last_player_standing',
        weirdStateType: 'last-player-standing',
      });
    });

    it('handles unknown type via default case (returns structural-stalemate)', () => {
      // Force an unknown type through the function to test default branch
      const unknownType = 'unknown-type' as RulesUxWeirdStateType;
      const result = getWeirdStateReasonForType(unknownType);

      expect(result).toEqual({
        reasonCode: 'STRUCTURAL_STALEMATE_TIEBREAK',
        rulesContext: 'structural_stalemate',
        weirdStateType: 'structural-stalemate',
      });
    });

    it('returns consistent weirdStateType for each input', () => {
      const types: RulesUxWeirdStateType[] = [
        'active-no-moves-movement',
        'active-no-moves-line',
        'active-no-moves-territory',
        'last-player-standing',
        'forced-elimination',
        'structural-stalemate',
      ];

      for (const type of types) {
        const result = getWeirdStateReasonForType(type);
        // All should have the input type echoed back, except default case
        if (type !== 'structural-stalemate') {
          expect(result.weirdStateType).toBe(type);
        }
      }
    });
  });

  // =========================================================================
  // getWeirdStateReasonForGameResult
  // =========================================================================

  describe('getWeirdStateReasonForGameResult', () => {
    it('returns null for null input', () => {
      const result = getWeirdStateReasonForGameResult(null);
      expect(result).toBeNull();
    });

    it('returns null for undefined input', () => {
      const result = getWeirdStateReasonForGameResult(undefined);
      expect(result).toBeNull();
    });

    it('maps game_completed reason to structural-stalemate info', () => {
      const gameResult: GameResult = {
        reason: 'game_completed',
        winner: 1,
      };

      const result = getWeirdStateReasonForGameResult(gameResult);

      expect(result).toEqual({
        reasonCode: 'STRUCTURAL_STALEMATE_TIEBREAK',
        rulesContext: 'structural_stalemate',
        weirdStateType: 'structural-stalemate',
      });
    });

    it('maps last_player_standing reason to LPS info', () => {
      const gameResult: GameResult = {
        reason: 'last_player_standing',
        winner: 2,
      };

      const result = getWeirdStateReasonForGameResult(gameResult);

      expect(result).toEqual({
        reasonCode: 'LAST_PLAYER_STANDING_EXCLUSIVE_REAL_ACTIONS',
        rulesContext: 'last_player_standing',
        weirdStateType: 'last-player-standing',
      });
    });

    it('returns null for elimination reason (standard victory)', () => {
      const gameResult: GameResult = {
        reason: 'elimination',
        winner: 1,
      };

      const result = getWeirdStateReasonForGameResult(gameResult);
      expect(result).toBeNull();
    });

    it('returns null for territory_threshold reason (standard victory)', () => {
      const gameResult: GameResult = {
        reason: 'territory_threshold',
        winner: 2,
      };

      const result = getWeirdStateReasonForGameResult(gameResult);
      expect(result).toBeNull();
    });

    it('returns null for resignation reason (standard victory)', () => {
      const gameResult: GameResult = {
        reason: 'resignation',
        winner: 1,
      };

      const result = getWeirdStateReasonForGameResult(gameResult);
      expect(result).toBeNull();
    });

    it('returns null for timeout reason (standard victory)', () => {
      const gameResult: GameResult = {
        reason: 'timeout',
        winner: 2,
      };

      const result = getWeirdStateReasonForGameResult(gameResult);
      expect(result).toBeNull();
    });

    it('returns null for draw reason', () => {
      const gameResult: GameResult = {
        reason: 'draw',
        winner: null,
      };

      const result = getWeirdStateReasonForGameResult(gameResult);
      expect(result).toBeNull();
    });
  });

  // =========================================================================
  // getRulesContextForReason
  // =========================================================================

  describe('getRulesContextForReason', () => {
    it('maps ANM_MOVEMENT_FE_BLOCKED to anm_forced_elimination', () => {
      const result = getRulesContextForReason('ANM_MOVEMENT_FE_BLOCKED');
      expect(result).toBe('anm_forced_elimination');
    });

    it('maps FE_SEQUENCE_CURRENT_PLAYER to anm_forced_elimination', () => {
      const result = getRulesContextForReason('FE_SEQUENCE_CURRENT_PLAYER');
      expect(result).toBe('anm_forced_elimination');
    });

    it('maps ANM_LINE_NO_ACTIONS to line_reward_exact', () => {
      const result = getRulesContextForReason('ANM_LINE_NO_ACTIONS');
      expect(result).toBe('line_reward_exact');
    });

    it('maps ANM_TERRITORY_NO_ACTIONS to territory_mini_region', () => {
      const result = getRulesContextForReason('ANM_TERRITORY_NO_ACTIONS');
      expect(result).toBe('territory_mini_region');
    });

    it('maps STRUCTURAL_STALEMATE_TIEBREAK to structural_stalemate', () => {
      const result = getRulesContextForReason('STRUCTURAL_STALEMATE_TIEBREAK');
      expect(result).toBe('structural_stalemate');
    });

    it('maps LAST_PLAYER_STANDING_EXCLUSIVE_REAL_ACTIONS to last_player_standing', () => {
      const result = getRulesContextForReason('LAST_PLAYER_STANDING_EXCLUSIVE_REAL_ACTIONS');
      expect(result).toBe('last_player_standing');
    });

    it('handles unknown reason code via default case', () => {
      // Force an unknown code through the function to test default branch
      const unknownCode = 'UNKNOWN_CODE' as RulesWeirdStateReasonCode;
      const result = getRulesContextForReason(unknownCode);
      expect(result).toBe('structural_stalemate');
    });

    it('returns valid RulesUxContext for all known codes', () => {
      const codes: RulesWeirdStateReasonCode[] = [
        'ANM_MOVEMENT_FE_BLOCKED',
        'ANM_LINE_NO_ACTIONS',
        'ANM_TERRITORY_NO_ACTIONS',
        'FE_SEQUENCE_CURRENT_PLAYER',
        'STRUCTURAL_STALEMATE_TIEBREAK',
        'LAST_PLAYER_STANDING_EXCLUSIVE_REAL_ACTIONS',
      ];

      const validContexts: RulesUxContext[] = [
        'anm_forced_elimination',
        'structural_stalemate',
        'last_player_standing',
        'territory_mini_region',
        'territory_multi_region',
        'line_reward_exact',
        'line_reward_overlength',
        'line_vs_territory_multi_phase',
        'capture_chain_mandatory',
        'landing_on_own_marker',
        'pie_rule_swap',
        'placement_cap',
      ];

      for (const code of codes) {
        const result = getRulesContextForReason(code);
        expect(validContexts).toContain(result);
      }
    });
  });

  // =========================================================================
  // getWeirdStateTypeForReason
  // =========================================================================

  describe('getWeirdStateTypeForReason', () => {
    it('maps ANM_MOVEMENT_FE_BLOCKED to active-no-moves-movement', () => {
      const result = getWeirdStateTypeForReason('ANM_MOVEMENT_FE_BLOCKED');
      expect(result).toBe('active-no-moves-movement');
    });

    it('maps ANM_LINE_NO_ACTIONS to active-no-moves-line', () => {
      const result = getWeirdStateTypeForReason('ANM_LINE_NO_ACTIONS');
      expect(result).toBe('active-no-moves-line');
    });

    it('maps ANM_TERRITORY_NO_ACTIONS to active-no-moves-territory', () => {
      const result = getWeirdStateTypeForReason('ANM_TERRITORY_NO_ACTIONS');
      expect(result).toBe('active-no-moves-territory');
    });

    it('maps FE_SEQUENCE_CURRENT_PLAYER to forced-elimination', () => {
      const result = getWeirdStateTypeForReason('FE_SEQUENCE_CURRENT_PLAYER');
      expect(result).toBe('forced-elimination');
    });

    it('maps STRUCTURAL_STALEMATE_TIEBREAK to structural-stalemate', () => {
      const result = getWeirdStateTypeForReason('STRUCTURAL_STALEMATE_TIEBREAK');
      expect(result).toBe('structural-stalemate');
    });

    it('maps LAST_PLAYER_STANDING_EXCLUSIVE_REAL_ACTIONS to last-player-standing', () => {
      const result = getWeirdStateTypeForReason('LAST_PLAYER_STANDING_EXCLUSIVE_REAL_ACTIONS');
      expect(result).toBe('last-player-standing');
    });

    it('handles unknown reason code via default case', () => {
      const unknownCode = 'UNKNOWN_CODE' as RulesWeirdStateReasonCode;
      const result = getWeirdStateTypeForReason(unknownCode);
      expect(result).toBe('structural-stalemate');
    });
  });

  // =========================================================================
  // getTeachingTopicForReason
  // =========================================================================

  describe('getTeachingTopicForReason', () => {
    it('maps ANM_MOVEMENT_FE_BLOCKED to teaching.active_no_moves', () => {
      const result = getTeachingTopicForReason('ANM_MOVEMENT_FE_BLOCKED');
      expect(result).toBe('teaching.active_no_moves');
    });

    it('maps ANM_LINE_NO_ACTIONS to teaching.line_bonus', () => {
      const result = getTeachingTopicForReason('ANM_LINE_NO_ACTIONS');
      expect(result).toBe('teaching.line_bonus');
    });

    it('maps ANM_TERRITORY_NO_ACTIONS to teaching.territory', () => {
      const result = getTeachingTopicForReason('ANM_TERRITORY_NO_ACTIONS');
      expect(result).toBe('teaching.territory');
    });

    it('maps FE_SEQUENCE_CURRENT_PLAYER to teaching.forced_elimination', () => {
      const result = getTeachingTopicForReason('FE_SEQUENCE_CURRENT_PLAYER');
      expect(result).toBe('teaching.forced_elimination');
    });

    it('maps STRUCTURAL_STALEMATE_TIEBREAK to teaching.victory_stalemate', () => {
      const result = getTeachingTopicForReason('STRUCTURAL_STALEMATE_TIEBREAK');
      expect(result).toBe('teaching.victory_stalemate');
    });

    it('maps LAST_PLAYER_STANDING_EXCLUSIVE_REAL_ACTIONS to teaching.victory_stalemate', () => {
      const result = getTeachingTopicForReason('LAST_PLAYER_STANDING_EXCLUSIVE_REAL_ACTIONS');
      expect(result).toBe('teaching.victory_stalemate');
    });

    it('handles unknown reason code via default case', () => {
      // Force an unknown code through the function to test default branch
      const unknownCode = 'UNKNOWN_CODE' as RulesWeirdStateReasonCode;
      const result = getTeachingTopicForReason(unknownCode);
      expect(result).toBe('teaching.victory_stalemate');
    });

    it('returns valid WeirdStateTeachingTopicId for all known codes', () => {
      const codes: RulesWeirdStateReasonCode[] = [
        'ANM_MOVEMENT_FE_BLOCKED',
        'ANM_LINE_NO_ACTIONS',
        'ANM_TERRITORY_NO_ACTIONS',
        'FE_SEQUENCE_CURRENT_PLAYER',
        'STRUCTURAL_STALEMATE_TIEBREAK',
        'LAST_PLAYER_STANDING_EXCLUSIVE_REAL_ACTIONS',
      ];

      const validTopics: WeirdStateTeachingTopicId[] = [
        'teaching.active_no_moves',
        'teaching.forced_elimination',
        'teaching.line_bonus',
        'teaching.territory',
        'teaching.victory_stalemate',
      ];

      for (const code of codes) {
        const result = getTeachingTopicForReason(code);
        expect(validTopics).toContain(result);
      }
    });
  });

  // =========================================================================
  // Integration / Cross-function consistency tests
  // =========================================================================

  describe('cross-function consistency', () => {
    it('getWeirdStateReasonForType output is consistent with getRulesContextForReason', () => {
      const types: RulesUxWeirdStateType[] = [
        'active-no-moves-movement',
        'active-no-moves-line',
        'active-no-moves-territory',
        'forced-elimination',
        'structural-stalemate',
      ];

      for (const type of types) {
        const info = getWeirdStateReasonForType(type);
        const contextFromReason = getRulesContextForReason(info.reasonCode);
        expect(info.rulesContext).toBe(contextFromReason);
      }
    });

    it('all reason codes have valid teaching topics', () => {
      const codes: RulesWeirdStateReasonCode[] = [
        'ANM_MOVEMENT_FE_BLOCKED',
        'ANM_LINE_NO_ACTIONS',
        'ANM_TERRITORY_NO_ACTIONS',
        'FE_SEQUENCE_CURRENT_PLAYER',
        'STRUCTURAL_STALEMATE_TIEBREAK',
        'LAST_PLAYER_STANDING_EXCLUSIVE_REAL_ACTIONS',
      ];

      for (const code of codes) {
        const topic = getTeachingTopicForReason(code);
        expect(topic).toMatch(/^teaching\./);
      }
    });

    it('game_completed and last_player_standing are the only weird game results', () => {
      const weirdReasons: GameResult['reason'][] = ['game_completed', 'last_player_standing'];
      const normalReasons: GameResult['reason'][] = [
        'elimination',
        'territory_threshold',
        'resignation',
        'timeout',
        'draw',
      ];

      for (const reason of weirdReasons) {
        const result = getWeirdStateReasonForGameResult({ reason, winner: 1 });
        expect(result).not.toBeNull();
      }

      for (const reason of normalReasons) {
        const result = getWeirdStateReasonForGameResult({ reason, winner: 1 });
        expect(result).toBeNull();
      }
    });
  });
});
