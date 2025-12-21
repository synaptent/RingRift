import type { Move, MoveType } from '../../types/game';

export type LegacyMoveType =
  | 'choose_line_reward'
  | 'process_territory_region'
  | 'line_formation'
  | 'territory_claim';

export const LEGACY_MOVE_TYPES: readonly LegacyMoveType[] = [
  'choose_line_reward',
  'process_territory_region',
  'line_formation',
  'territory_claim',
];

export function isLegacyMoveType(moveType: MoveType): moveType is LegacyMoveType {
  return (LEGACY_MOVE_TYPES as readonly string[]).includes(moveType);
}

export function normalizeLegacyMoveType(moveType: MoveType): MoveType {
  switch (moveType) {
    case 'choose_line_reward':
      return 'choose_line_option';
    case 'process_territory_region':
      return 'choose_territory_option';
    default:
      return moveType;
  }
}

export function normalizeLegacyMove(move: Move): Move {
  const normalizedType = normalizeLegacyMoveType(move.type);
  if (normalizedType === move.type) {
    return move;
  }
  return { ...move, type: normalizedType };
}

export function assertNoLegacyMoveType(move: Move, context: string): void {
  if (!isLegacyMoveType(move.type)) {
    return;
  }
  throw new Error(
    `[LegacyMoveType] ${context}: move.type=${move.type} is legacy-only. ` +
      'Canonical recordings must use the canonical MoveType names.'
  );
}
