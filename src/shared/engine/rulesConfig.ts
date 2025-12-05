import { BOARD_CONFIGS, BoardType, RulesOptions } from '../types/game';

/**
 * Compute the effective required line length for collapse / rewards for a
 * given board + player-count combination.
 *
 * Canonical semantics (RR-CANON-R120):
 * - Base threshold comes from BOARD_CONFIGS[boardType].lineLength.
 * - No special-casing by player count; for square8 this is always 3, and
 *   for square19 / hexagonal this is always 4.
 * - rulesOptions is reserved for future per-game overrides (for example,
 *   per-ruleset lineLength tweaks) and is currently unused.
 */
export function getEffectiveLineLengthThreshold(
  boardType: BoardType,
  _numPlayers: number,
  _rulesOptions?: RulesOptions
): number {
  const base = BOARD_CONFIGS[boardType].lineLength;

  return base;
}
