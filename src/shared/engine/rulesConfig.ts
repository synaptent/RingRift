import { BOARD_CONFIGS, BoardType, RulesOptions } from '../types/game';

/**
 * Compute the effective required line length for collapse / rewards for a
 * given board + player-count combination.
 *
 * Semantics:
 * - Base threshold comes from BOARD_CONFIGS[boardType].lineLength.
 * - For square8 with exactly two players, we elevate the threshold to
 *   4-in-a-row so that 3-in-a-row remains available for 3p/4p games and
 *   for heuristics, but does not trigger line_processing in 2p games.
 * - rulesOptions is reserved for future per-game overrides (for example,
 *   per-ruleset lineLength tweaks) and is currently unused.
 */
export function getEffectiveLineLengthThreshold(
  boardType: BoardType,
  numPlayers: number,
  rulesOptions?: RulesOptions
): number {
  const base = BOARD_CONFIGS[boardType].lineLength;

  if (boardType === 'square8' && numPlayers === 2) {
    return 4;
  }

  return base;
}