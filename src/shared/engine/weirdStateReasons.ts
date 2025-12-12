import type { GameResult } from '../types/game';
import type { RulesUxContext, RulesUxWeirdStateType } from '../telemetry/rulesUxEvents';

/**
 * Stable weird-state reason codes for UX and telemetry, aligned with
 * docs/UX_RULES_WEIRD_STATES_SPEC.md §2.1.
 */
export type RulesWeirdStateReasonCode =
  | 'ANM_MOVEMENT_FE_BLOCKED'
  | 'ANM_LINE_NO_ACTIONS'
  | 'ANM_TERRITORY_NO_ACTIONS'
  | 'FE_SEQUENCE_CURRENT_PLAYER'
  | 'STRUCTURAL_STALEMATE_TIEBREAK'
  | 'LAST_PLAYER_STANDING_EXCLUSIVE_REAL_ACTIONS';

export interface WeirdStateReasonInfo {
  reasonCode: RulesWeirdStateReasonCode;
  rulesContext: RulesUxContext;
  weirdStateType: RulesUxWeirdStateType;
}

/**
 * Runtime guard for whether a weird-state type should be surfaced to players
 * via HUD / Teaching overlays. Line/territory no-action cases are now treated
 * as routine and should not render banners, but may still be tracked via
 * telemetry when present.
 */
export function isSurfaceableWeirdStateType(type: RulesUxWeirdStateType): boolean {
  return (
    type === 'active-no-moves-movement' ||
    type === 'last-player-standing' ||
    type === 'forced-elimination' ||
    type === 'structural-stalemate'
  );
}

/**
 * Guard for reason codes that should be surfaced in player-facing overlays.
 * ANM line/territory no-actions are routine and should not trigger weird-state
 * overlays, even if they remain available for telemetry.
 */
export function isSurfaceableWeirdStateReason(reasonCode: RulesWeirdStateReasonCode): boolean {
  return reasonCode !== 'ANM_LINE_NO_ACTIONS' && reasonCode !== 'ANM_TERRITORY_NO_ACTIONS';
}

/**
 * Map a coarse weird state type (as exposed in HUD and telemetry) to a
 * canonical reason code and rules_context.
 *
 * This is a pure mapping layer; it does not attempt to inspect full board
 * geometry. Where the spec distinguishes between, e.g., mini- vs multi-region
 * cases, callers may refine rulesContext further based on local context.
 */
export function getWeirdStateReasonForType(
  weirdStateType: RulesUxWeirdStateType
): WeirdStateReasonInfo {
  switch (weirdStateType) {
    case 'active-no-moves-movement':
      return {
        reasonCode: 'ANM_MOVEMENT_FE_BLOCKED',
        rulesContext: 'anm_forced_elimination',
        weirdStateType,
      };
    case 'active-no-moves-line':
      return {
        reasonCode: 'ANM_LINE_NO_ACTIONS',
        rulesContext: 'line_reward_exact',
        weirdStateType,
      };
    case 'active-no-moves-territory':
      return {
        reasonCode: 'ANM_TERRITORY_NO_ACTIONS',
        rulesContext: 'territory_mini_region',
        weirdStateType,
      };
    case 'last-player-standing':
      return {
        reasonCode: 'LAST_PLAYER_STANDING_EXCLUSIVE_REAL_ACTIONS',
        rulesContext: 'last_player_standing',
        weirdStateType,
      };
    case 'forced-elimination':
      return {
        reasonCode: 'FE_SEQUENCE_CURRENT_PLAYER',
        rulesContext: 'anm_forced_elimination',
        weirdStateType,
      };
    case 'structural-stalemate':
    default:
      return {
        reasonCode: 'STRUCTURAL_STALEMATE_TIEBREAK',
        rulesContext: 'structural_stalemate',
        weirdStateType: 'structural-stalemate',
      };
  }
}

/**
 * Map a terminal GameResult reason into a weird-state reason when applicable.
 * Returns null for standard victories (simple elimination / territory / resign).
 */
export function getWeirdStateReasonForGameResult(
  result: GameResult | null | undefined
): WeirdStateReasonInfo | null {
  if (!result) return null;

  if (result.reason === 'game_completed') {
    const info = getWeirdStateReasonForType('structural-stalemate');
    return info;
  }

  if (result.reason === 'last_player_standing') {
    return {
      reasonCode: 'LAST_PLAYER_STANDING_EXCLUSIVE_REAL_ACTIONS',
      rulesContext: 'last_player_standing',
      weirdStateType: 'last-player-standing',
    };
  }

  return null;
}

/**
 * Convenience helper to derive the canonical RulesUxContext from a reason code.
 * Useful when only the reason_code is available (e.g. from telemetry logs).
 */
export function getRulesContextForReason(reasonCode: RulesWeirdStateReasonCode): RulesUxContext {
  switch (reasonCode) {
    case 'ANM_MOVEMENT_FE_BLOCKED':
    case 'FE_SEQUENCE_CURRENT_PLAYER':
      return 'anm_forced_elimination';
    case 'ANM_LINE_NO_ACTIONS':
      return 'line_reward_exact';
    case 'ANM_TERRITORY_NO_ACTIONS':
      return 'territory_mini_region';
    case 'STRUCTURAL_STALEMATE_TIEBREAK':
      return 'structural_stalemate';
    case 'LAST_PLAYER_STANDING_EXCLUSIVE_REAL_ACTIONS':
      return 'last_player_standing';
    default:
      // Type system should make this unreachable, but keep a safe fallback.
      return 'structural_stalemate';
  }
}

/**
 * Derive the coarse weird-state type from a stable reason code.
 *
 * This is useful for telemetry payloads that carry reason codes but need a
 * consistent `weirdStateType` label for aggregation.
 */
export function getWeirdStateTypeForReason(
  reasonCode: RulesWeirdStateReasonCode
): RulesUxWeirdStateType {
  switch (reasonCode) {
    case 'ANM_MOVEMENT_FE_BLOCKED':
      return 'active-no-moves-movement';
    case 'ANM_LINE_NO_ACTIONS':
      return 'active-no-moves-line';
    case 'ANM_TERRITORY_NO_ACTIONS':
      return 'active-no-moves-territory';
    case 'FE_SEQUENCE_CURRENT_PLAYER':
      return 'forced-elimination';
    case 'STRUCTURAL_STALEMATE_TIEBREAK':
      return 'structural-stalemate';
    case 'LAST_PLAYER_STANDING_EXCLUSIVE_REAL_ACTIONS':
      return 'last-player-standing';
    default:
      // Type system should make this unreachable, but keep a safe fallback.
      return 'structural-stalemate';
  }
}

/**
 * Canonical TeachingOverlay topic ids for weird-state reasons.
 *
 * These map the stable RWS-* reason codes from docs/UX_RULES_WEIRD_STATES_SPEC.md §4.3
 * onto abstract teaching topic identifiers. Client surfaces (HUD, VictoryModal,
 * TeachingOverlay) can translate these into concrete TeachingOverlay topics
 * (e.g. "teaching.active_no_moves" → "active_no_moves") without duplicating
 * the mapping logic.
 */
export type WeirdStateTeachingTopicId =
  | 'teaching.active_no_moves'
  | 'teaching.forced_elimination'
  | 'teaching.line_bonus'
  | 'teaching.territory'
  | 'teaching.victory_stalemate';

/**
 * Map a weird-state reason code to its primary TeachingOverlay topic id.
 *
 * This follows the mapping defined in docs/UX_RULES_WEIRD_STATES_SPEC.md §4.3:
 *
 * - ANM_MOVEMENT_FE_BLOCKED → teaching.active_no_moves (with FE as a related topic)
 * - ANM_LINE_NO_ACTIONS → teaching.line_bonus
 * - ANM_TERRITORY_NO_ACTIONS → teaching.territory
 * - FE_SEQUENCE_CURRENT_PLAYER → teaching.forced_elimination
 * - STRUCTURAL_STALEMATE_TIEBREAK → teaching.victory_stalemate
 * - LAST_PLAYER_STANDING_EXCLUSIVE_REAL_ACTIONS → teaching.victory_stalemate (LPS section)
 */
export function getTeachingTopicForReason(
  reasonCode: RulesWeirdStateReasonCode
): WeirdStateTeachingTopicId {
  switch (reasonCode) {
    case 'ANM_MOVEMENT_FE_BLOCKED':
      return 'teaching.active_no_moves';
    case 'ANM_LINE_NO_ACTIONS':
      return 'teaching.line_bonus';
    case 'ANM_TERRITORY_NO_ACTIONS':
      return 'teaching.territory';
    case 'FE_SEQUENCE_CURRENT_PLAYER':
      return 'teaching.forced_elimination';
    case 'STRUCTURAL_STALEMATE_TIEBREAK':
    case 'LAST_PLAYER_STANDING_EXCLUSIVE_REAL_ACTIONS':
      return 'teaching.victory_stalemate';
    default:
      // Type system should make this unreachable, but keep a safe fallback.
      return 'teaching.victory_stalemate';
  }
}
