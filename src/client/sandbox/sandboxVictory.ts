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
  markers: number;
}

function computePlayerVictoryStats(state: GameState): PlayerVictoryStats[] {
  const board = state.board;
  const byPlayer: { [player: number]: PlayerVictoryStats } = {};

  for (const p of state.players) {
    byPlayer[p.playerNumber] = {
      playerNumber: p.playerNumber,
      ringsRemaining: 0,
      territorySpaces: p.territorySpaces,
      ringsEliminated: p.eliminatedRings,
      markers: 0,
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

  // Count markers remaining on the board for each player.
  for (const marker of board.markers.values()) {
    const owner = marker.player;
    const entry = byPlayer[owner];
    if (entry) {
      entry.markers += 1;
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
  // the strict elimination/territory thresholds. Apply the same stalemate
  // ladder as the backend: territory → eliminated rings → markers →
  // last-actor. This ensures there is always a definitive winner under
  // normal RingRift rules.
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

    // Third rung: remaining markers on the board.
    const maxMarkers = Math.max(...stats.map(s => s.markers));
    const markerLeaders = stats.filter(s => s.markers === maxMarkers);

    if (markerLeaders.length === 1 && maxMarkers > 0) {
      return buildGameResult(state, stats, markerLeaders[0].playerNumber, 'last_player_standing');
    }

    // Final rung: last player to complete a valid turn action.
    const lastActor = getLastActorFromState(state);
    if (lastActor !== undefined) {
      return buildGameResult(state, stats, lastActor, 'last_player_standing');
    }

    // Safety fallback: in degenerate cases where no last actor can be
    // determined, mark the game as completed without a specific winner.
    return buildGameResult(state, stats, undefined, 'game_completed' as any);
  }

  return null;
}

function getLastActorFromState(state: GameState): number | undefined {
  if (state.history && state.history.length > 0) {
    const lastEntry = state.history[state.history.length - 1];
    if (lastEntry && typeof lastEntry.actor === 'number') {
      return lastEntry.actor;
    }
  }

  if (state.moveHistory && state.moveHistory.length > 0) {
    const lastMove = state.moveHistory[state.moveHistory.length - 1];
    if (lastMove && typeof lastMove.player === 'number') {
      return lastMove.player;
    }
  }

  const players = state.players;
  if (!players || players.length === 0) {
    return undefined;
  }

  const currentIdx = players.findIndex(p => p.playerNumber === state.currentPlayer);
  if (currentIdx === -1) {
    return players[0].playerNumber;
  }

  const lastIdx = (currentIdx - 1 + players.length) % players.length;
  return players[lastIdx].playerNumber;
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
