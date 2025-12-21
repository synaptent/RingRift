import type { LegacyMoveType } from '../../types/game';

const LEGACY_MOVE_TYPE_SYMBOLS: Record<LegacyMoveType, string> = {
  move_ring: 'M',
  build_stack: 'M',
  choose_line_reward: 'LR',
  process_territory_region: 'TR',
  line_formation: 'L',
  territory_claim: 'T',
};

export function formatLegacyMoveTypeSymbol(moveType: LegacyMoveType): string {
  return LEGACY_MOVE_TYPE_SYMBOLS[moveType] ?? moveType;
}
