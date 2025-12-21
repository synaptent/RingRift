import type { MoveType, LegacyMoveType } from '../../shared/types/game';
import {
  isLegacyMoveType,
  normalizeLegacyMoveType,
} from '../../shared/engine/legacy/legacyMoveTypes';

const CANONICAL_LABELS: Record<string, string> = {
  place_ring: 'Place Ring',
  skip_placement: 'Skip Placement',
  no_placement_action: 'No Placement Action',
  move_stack: 'Move Stack',
  overtaking_capture: 'Capture',
  continue_capture_segment: 'Continue Capture',
  chain_capture: 'Chain Capture',
  skip_capture: 'Skip Capture',
  no_movement_action: 'No Movement Action',
  process_line: 'Process Line',
  choose_line_option: 'Line Option',
  no_line_action: 'No Line Action',
  choose_territory_option: 'Territory Option',
  skip_territory_processing: 'Skip Territory',
  no_territory_action: 'No Territory Action',
  eliminate_rings_from_stack: 'Eliminate Rings',
  forced_elimination: 'Forced Elimination',
  swap_sides: 'Swap Sides',
  recovery_slide: 'Recovery',
  skip_recovery: 'Skip Recovery',
};

const LEGACY_LABELS: Record<LegacyMoveType, string> = {
  move_ring: 'Move Ring',
  build_stack: 'Build Stack',
  choose_line_reward: 'Line Reward',
  process_territory_region: 'Territory Process',
  line_formation: 'Line Formation',
  territory_claim: 'Territory Claim',
};

const LEGACY_CATEGORY_LABELS: Partial<Record<LegacyMoveType, string>> = {
  line_formation: 'Line Processing',
  territory_claim: 'Territory',
};

export function formatMoveTypeLabel(moveType: MoveType): string {
  if (isLegacyMoveType(moveType)) {
    const legacyLabel = LEGACY_LABELS[moveType] ?? moveType.replace(/_/g, ' ');
    return `${legacyLabel} (legacy)`;
  }

  const canonicalType = normalizeLegacyMoveType(moveType);
  return CANONICAL_LABELS[canonicalType] ?? canonicalType.replace(/_/g, ' ');
}

export function getMoveCategoryLabel(moveType: MoveType): string {
  if (isLegacyMoveType(moveType)) {
    const legacyCategory = LEGACY_CATEGORY_LABELS[moveType];
    if (legacyCategory) {
      return legacyCategory;
    }
  }

  const canonicalType = normalizeLegacyMoveType(moveType);
  switch (canonicalType) {
    case 'place_ring':
      return 'Placement';
    case 'skip_placement':
    case 'no_placement_action':
      return 'Placement (no action)';
    case 'move_stack':
    case 'no_movement_action':
      return 'Movement';
    case 'overtaking_capture':
    case 'continue_capture_segment':
      return 'Capture';
    case 'skip_capture':
      return 'Skip Capture';
    case 'process_line':
    case 'choose_line_option':
    case 'no_line_action':
      return 'Line Processing';
    case 'eliminate_rings_from_stack':
      return 'Eliminate Rings';
    case 'choose_territory_option':
    case 'skip_territory_processing':
    case 'no_territory_action':
      return 'Territory';
    case 'forced_elimination':
      return 'Forced Elimination';
    case 'swap_sides':
      return 'Swap Sides';
    case 'recovery_slide':
    case 'skip_recovery':
      return 'Recovery';
    default:
      return canonicalType.replace(/_/g, ' ');
  }
}
