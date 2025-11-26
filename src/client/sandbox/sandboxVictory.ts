import type { GameResult, GameState } from '../../shared/engine';
import { evaluateVictory } from '../../shared/engine';

/**
 * Pure victory-condition helpers for the client sandbox.
 *
 * These now delegate winner/reason detection to the shared
 * {@link evaluateVictory} helper so that backend RuleEngine,
 * shared GameEngine, and sandbox all share a single source of
 * truth for ring-elimination, territory-control, and stalemate
 * ladder semantics. Sandbox-specific helpers are responsible
 * only for deriving UI-facing GameResult statistics.
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
 * Check victory conditions for the sandbox using the shared engine
 * victory helper. This keeps backend RuleEngine, shared GameEngine,
 * and sandbox aligned on:
 *
 * - Ring-elimination victory: players[].eliminatedRings >= victoryThreshold.
 * - Territory-control victory: players[].territorySpaces >= territoryVictoryThreshold.
 * - Bare-board global stalemate: territory → eliminated rings (including
 *   conceptual hand→eliminated when appropriate) → markers → last actor.
 *
 * Returns a GameResult when the game should end, or null to continue.
 */
export function checkSandboxVictory(state: GameState): GameResult | null {
  const verdict = evaluateVictory(state);

  if (!verdict.isGameOver) {
    return null;
  }

  const stats = computePlayerVictoryStats(state);

  // Map shared VictoryReason to the GameResult.reason union used by the UI.
  let reason: GameResult['reason'];
  switch (verdict.reason) {
    case 'ring_elimination':
    case 'territory_control':
    case 'last_player_standing':
    case 'game_completed':
      reason = verdict.reason;
      break;
    default:
      reason = 'game_completed';
  }

  return buildGameResult(state, stats, verdict.winner, reason);
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
      ringsRemaining,
    },
  };
}
