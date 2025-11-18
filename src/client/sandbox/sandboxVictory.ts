import { BOARD_CONFIGS, GameResult, GameState, positionToString } from '../../shared/types/game';

/**
 * Pure victory-condition helpers for the client sandbox.
 *
 * These intentionally mirror the compact rules while using the shared
 * GameState fields (victoryThreshold, territoryVictoryThreshold,
 * players[].eliminatedRings, players[].territorySpaces).
 */

interface PlayerVictoryStats {
  playerNumber: number;
  ringsRemaining: number;
  territorySpaces: number;
  ringsEliminated: number;
}

function computePlayerVictoryStats(state: GameState): PlayerVictoryStats[] {
  const board = state.board;
  const byPlayer: { [player: number]: PlayerVictoryStats } = {};

  for (const p of state.players) {
    byPlayer[p.playerNumber] = {
      playerNumber: p.playerNumber,
      ringsRemaining: 0,
      territorySpaces: p.territorySpaces,
      ringsEliminated: p.eliminatedRings
    };
  }

  // Count rings remaining from stacks on the board.
  for (const stack of board.stacks.values()) {
    const owner = stack.controllingPlayer;
    const entry = byPlayer[owner];
    if (entry) {
      entry.ringsRemaining += stack.stackHeight;
    }
  }

  return Object.values(byPlayer);
}

/**
 * Check victory conditions for the sandbox using the compact rules:
 * - Ring-elimination victory: player.eliminatedRings >= victoryThreshold.
 * - Territory-control victory: player.territorySpaces >= territoryVictoryThreshold.
 *
 * Returns a GameResult when the game should end, or null to continue.
 */
export function checkSandboxVictory(state: GameState): GameResult | null {
  const config = BOARD_CONFIGS[state.boardType];
  const stats = computePlayerVictoryStats(state);

  // 1. Ring-elimination victory (strictly more than 50% of total rings in play).
  const ringWinner = stats.find(s => s.ringsEliminated >= state.victoryThreshold);
  if (ringWinner) {
    return buildGameResult(state, stats, ringWinner.playerNumber, 'ring_elimination');
  }

  // 2. Territory-control victory (>50% of board spaces).
  const territoryWinner = stats.find(
    s => s.territorySpaces >= state.territoryVictoryThreshold
  );
  if (territoryWinner) {
    return buildGameResult(state, stats, territoryWinner.playerNumber, 'territory_control');
  }

  // 3. Fallback: no stacks and no rings in hand for any player. In this
  // situation, the game cannot progress further, even if nobody reached
  // the strict elimination/territory thresholds. Mirror the backend's
  // endGame final-score semantics by declaring the game completed using
  // territory (then eliminated rings) as tie-breakers, and allowing for
  // a draw when all scores are equal.
  const noStacksLeft = state.board.stacks.size === 0;
  const anyRingsInHand = state.players.some(p => p.ringsInHand > 0);

  if (noStacksLeft && !anyRingsInHand) {
    const maxTerritory = Math.max(...stats.map(s => s.territorySpaces));
    const territoryLeaders = stats.filter(s => s.territorySpaces === maxTerritory);

    if (territoryLeaders.length === 1 && maxTerritory > 0) {
      return buildGameResult(state, stats, territoryLeaders[0].playerNumber, 'territory_control');
    }

    const maxEliminated = Math.max(...stats.map(s => s.ringsEliminated));
    const eliminationLeaders = stats.filter(s => s.ringsEliminated === maxEliminated);

    if (eliminationLeaders.length === 1 && maxEliminated > 0) {
      return buildGameResult(state, stats, eliminationLeaders[0].playerNumber, 'ring_elimination');
    }

    // Perfect tie: mark the game as completed with no winner, leaving
    // finalScore populated for UI/debug consumers.
    return buildGameResult(state, stats, undefined, 'game_completed' as any);
  }

  return null;
}

function buildGameResult(
  state: GameState,
  stats: PlayerVictoryStats[],
  winner: number | undefined,
  reason: GameResult['reason']
): GameResult {
  const ringsRemaining: { [playerNumber: number]: number } = {};
  const territorySpaces: { [playerNumber: number]: number } = {};
  const ringsEliminated: { [playerNumber: number]: number } = {};

  for (const s of stats) {
    ringsRemaining[s.playerNumber] = s.ringsRemaining;
    territorySpaces[s.playerNumber] = s.territorySpaces;
    ringsEliminated[s.playerNumber] = s.ringsEliminated;
  }

  return {
    ...(winner !== undefined && { winner }),
    reason,
    finalScore: {
      ringsEliminated,
      territorySpaces,
      ringsRemaining
    }
  };
}
