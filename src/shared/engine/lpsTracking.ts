/**
 * Shared Last-Player-Standing (LPS) tracking helpers for RingRift engine.
 *
 * The LPS victory condition (R172) fires when exactly one player has any real
 * actions available for THREE CONSECUTIVE full rounds while all other players
 * with material are blocked. After the second round completes, the exclusive
 * player wins by LPS.
 *
 * Real actions are: ring placement, non-capture movement, and overtaking capture.
 * Non-real actions (do not count for LPS): recovery_slide, forced_elimination,
 * and bookkeeping/post-processing decisions (skip_*, no_*, line/territory processing).
 *
 * @module lpsTracking
 *
 * Rule Reference: RR-CANON R172 (Last Player Standing Victory)
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * ARCHITECTURAL NOTE: Separation from VictoryAggregate
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * LPS tracking is intentionally separate from VictoryAggregate.evaluateVictory()
 * because it requires **stateful round tracking** that persists across turns:
 *
 * - evaluateVictory() is stateless: takes GameState, returns verdict
 * - LPS tracking is stateful: maintains round counters, exclusive player history
 *
 * The LpsTrackingState must be stored and managed by the host engine (GameEngine,
 * ClientSandboxEngine) since it's not part of the wire-serializable GameState.
 *
 * Usage pattern:
 * 1. Host engine maintains LpsTrackingState instance
 * 2. At each interactive turn start, call updateLpsTracking()
 * 3. After updating, call evaluateLpsVictory() to check for LPS win
 * 4. Combine with evaluateVictory() for complete victory detection
 *
 * See: VictoryAggregate.ts for immediate (stateless) victory detection
 * See: scripts/selfplay-db-ts-replay.ts for evaluateVictoryWithLps() helper
 */

import type { GameState, GamePhase, GameResult } from '../types/game';

/**
 * Phases where LPS tracking is active (interactive phases only).
 */
const LPS_ACTIVE_PHASES: GamePhase[] = ['ring_placement', 'movement', 'capture', 'chain_capture'];

/**
 * Internal state for tracking last-player-standing rounds.
 *
 * This is kept separate from GameState so it can be managed internally by
 * host engines without affecting wire formats or snapshots.
 */
export interface LpsTrackingState {
  /**
   * Current round index. Incremented when a new round begins.
   */
  roundIndex: number;

  /**
   * Map from player number to whether they had any real actions
   * available at the start of their most recent turn in this round.
   */
  currentRoundActorMask: Map<number, boolean>;

  /**
   * The first active player seen in the current round.
   * Used to detect when we've cycled back to start a new round.
   */
  currentRoundFirstPlayer: number | null;

  /**
   * If exactly one player had real actions in the most recently completed
   * round, their player number is stored here. Otherwise null.
   */
  exclusivePlayerForCompletedRound: number | null;

  /**
   * Number of consecutive completed rounds where the same player was the
   * exclusive real-action holder. LPS victory requires 3 consecutive rounds.
   */
  consecutiveExclusiveRounds: number;

  /**
   * The player who has been exclusive for consecutive rounds. Tracked
   * separately to detect when the exclusive player changes (resetting count).
   */
  consecutiveExclusivePlayer: number | null;
}

/**
 * Create a fresh LPS tracking state.
 */
export function createLpsTrackingState(): LpsTrackingState {
  return {
    roundIndex: 0,
    currentRoundActorMask: new Map(),
    currentRoundFirstPlayer: null,
    exclusivePlayerForCompletedRound: null,
    consecutiveExclusiveRounds: 0,
    consecutiveExclusivePlayer: null,
  };
}

/**
 * Reset LPS tracking state (e.g., after game ends).
 */
export function resetLpsTrackingState(state: LpsTrackingState): void {
  state.roundIndex = 0;
  state.currentRoundActorMask.clear();
  state.currentRoundFirstPlayer = null;
  state.exclusivePlayerForCompletedRound = null;
  state.consecutiveExclusiveRounds = 0;
  state.consecutiveExclusivePlayer = null;
}

/**
 * Options for updating LPS tracking.
 */
export interface LpsUpdateOptions {
  /**
   * Current player taking their turn.
   */
  currentPlayer: number;

  /**
   * List of player numbers who still have material (rings on board or in hand).
   * Players without material are excluded from LPS round tracking.
   */
  activePlayers: number[];

  /**
   * Whether the current player has any real actions available.
   * Real actions: placement, non-capture movement, overtaking capture.
   */
  hasRealAction: boolean;
}

/**
 * Update LPS round tracking for the current player's turn.
 *
 * This should be called at the start of each interactive turn (after the
 * shared turn engine has selected the next active player and phase).
 *
 * The function handles:
 * - Starting new rounds when the first player changes or cycles back
 * - Recording whether the current player has real actions
 * - Finalizing completed rounds to detect exclusive players
 *
 * @param lps - Current LPS tracking state (mutated in place)
 * @param options - Update options including current player and action availability
 *
 * @example
 * ```typescript
 * const activePlayers = state.players
 *   .filter(p => playerHasMaterial(p.playerNumber))
 *   .map(p => p.playerNumber);
 *
 * updateLpsTracking(lpsState, {
 *   currentPlayer: state.currentPlayer,
 *   activePlayers,
 *   hasRealAction: hasAnyRealActionForPlayer(state.currentPlayer),
 * });
 * ```
 */
export function updateLpsTracking(lps: LpsTrackingState, options: LpsUpdateOptions): void {
  const { currentPlayer, activePlayers, hasRealAction } = options;

  if (activePlayers.length === 0) {
    return;
  }

  const activeSet = new Set(activePlayers);
  if (!activeSet.has(currentPlayer)) {
    // Current player has no material; they're not part of LPS tracking
    return;
  }

  const first = lps.currentRoundFirstPlayer;
  const startingNewCycle = first === null || !activeSet.has(first);

  if (startingNewCycle) {
    // Either first round or previous round leader dropped out
    lps.roundIndex += 1;
    lps.currentRoundFirstPlayer = currentPlayer;
    lps.currentRoundActorMask.clear();
    lps.exclusivePlayerForCompletedRound = null;
    // Only reset consecutive tracking if the exclusive player also dropped out.
    // If the exclusive player is still active, they should continue counting
    // toward LPS victory even though the round structure changed (e.g., opponent
    // lost all material).
    const excl = lps.consecutiveExclusivePlayer;
    if (excl === null || !activeSet.has(excl)) {
      lps.consecutiveExclusiveRounds = 0;
      lps.consecutiveExclusivePlayer = null;
    }
  } else if (currentPlayer === first && lps.currentRoundActorMask.size > 0) {
    // Cycled back to first player - finalize previous round
    const exclusivePlayer = finalizeCompletedLpsRound(activePlayers, lps.currentRoundActorMask);
    lps.exclusivePlayerForCompletedRound = exclusivePlayer;

    // Track consecutive exclusive rounds for the same player
    if (exclusivePlayer !== null) {
      if (exclusivePlayer === lps.consecutiveExclusivePlayer) {
        // Same player remains exclusive - increment count
        lps.consecutiveExclusiveRounds += 1;
      } else {
        // Different player is now exclusive - reset and start counting
        lps.consecutiveExclusivePlayer = exclusivePlayer;
        lps.consecutiveExclusiveRounds = 1;
      }
    } else {
      // No exclusive player this round - reset consecutive tracking
      lps.consecutiveExclusiveRounds = 0;
      lps.consecutiveExclusivePlayer = null;
    }

    lps.roundIndex += 1;
    lps.currentRoundActorMask.clear();
    lps.currentRoundFirstPlayer = currentPlayer;
  }

  lps.currentRoundActorMask.set(currentPlayer, hasRealAction);
}

/**
 * Finalize a completed round by determining if exactly one player had real actions.
 *
 * @param activePlayers - List of players who have material
 * @param actorMask - Map of player number to whether they had real actions
 * @returns The exclusive player number, or null if 0 or 2+ players had actions
 */
export function finalizeCompletedLpsRound(
  activePlayers: number[],
  actorMask: Map<number, boolean>
): number | null {
  const truePlayers: number[] = [];

  for (const pid of activePlayers) {
    if (actorMask.get(pid)) {
      truePlayers.push(pid);
    }
  }

  return truePlayers.length === 1 ? truePlayers[0] : null;
}

/**
 * Options for evaluating LPS victory condition.
 */
export interface LpsEvaluationOptions {
  /**
   * Current game state.
   */
  gameState: GameState;

  /**
   * Current LPS tracking state.
   */
  lps: LpsTrackingState;

  /**
   * Callback to check if a player has any real actions.
   * Called for the candidate and potentially other players.
   */
  hasAnyRealAction: (playerNumber: number) => boolean;

  /**
   * Callback to check if a player has any material (rings on board or in hand).
   */
  hasMaterial: (playerNumber: number) => boolean;
}

/**
 * Result of LPS victory evaluation.
 */
export interface LpsEvaluationResult {
  /**
   * Whether LPS victory condition is satisfied.
   */
  isVictory: boolean;

  /**
   * The winning player number, if isVictory is true.
   */
  winner?: number;

  /**
   * Reason for non-victory (for debugging).
   */
  reason?: string;
}

/**
 * Default number of consecutive exclusive rounds required for LPS victory.
 * Per RR-CANON-R172, a player must be the exclusive real-action holder
 * for this many consecutive full rounds before winning by LPS.
 */
export const LPS_DEFAULT_REQUIRED_ROUNDS = 3;

/** @deprecated Use LPS_DEFAULT_REQUIRED_ROUNDS instead. */
export const LPS_REQUIRED_CONSECUTIVE_ROUNDS = LPS_DEFAULT_REQUIRED_ROUNDS;

/**
 * Evaluate whether the Last-Player-Standing victory condition (R172) is satisfied.
 *
 * This should be called at the start of each interactive turn after updating
 * LPS tracking. The condition is satisfied when:
 * 1. The same player has been the exclusive real-action holder for 3 consecutive rounds
 * 2. It's that player's turn (after completing round 3)
 * 3. They still have real actions
 * 4. All other players with material have no real actions
 *
 * @param options - Evaluation options with callbacks for action/material checks
 * @returns Evaluation result indicating whether LPS victory is achieved
 *
 * @example
 * ```typescript
 * const result = evaluateLpsVictory({
 *   gameState: state,
 *   lps: lpsState,
 *   hasAnyRealAction: (pn) => hasAnyRealActionForPlayer(state, pn),
 *   hasMaterial: (pn) => playerHasMaterial(pn),
 * });
 *
 * if (result.isVictory) {
 *   endGame(result.winner, 'last_player_standing');
 * }
 * ```
 */
export function evaluateLpsVictory(options: LpsEvaluationOptions): LpsEvaluationResult {
  const { gameState, lps, hasAnyRealAction, hasMaterial } = options;

  // Only evaluate in active games during interactive phases
  if (gameState.gameStatus !== 'active') {
    return { isVictory: false, reason: 'game_not_active' };
  }

  if (!LPS_ACTIVE_PHASES.includes(gameState.currentPhase)) {
    return { isVictory: false, reason: 'not_interactive_phase' };
  }

  const requiredRounds =
    gameState.lpsRoundsRequired ??
    gameState.rulesOptions?.lpsRoundsRequired ??
    LPS_DEFAULT_REQUIRED_ROUNDS;

  // Must have completed at least N consecutive rounds with the same exclusive player
  if (lps.consecutiveExclusiveRounds < requiredRounds) {
    return {
      isVictory: false,
      reason: `insufficient_consecutive_rounds_${lps.consecutiveExclusiveRounds}_of_${requiredRounds}`,
    };
  }

  // Must have a candidate who has been exclusive for consecutive rounds
  const candidate = lps.consecutiveExclusivePlayer;
  if (candidate === null || candidate === undefined) {
    return { isVictory: false, reason: 'no_exclusive_candidate' };
  }

  // Must be the candidate's turn (start of the round after the 2 qualifying rounds)
  if (gameState.currentPlayer !== candidate) {
    return { isVictory: false, reason: 'not_candidate_turn' };
  }

  // Candidate must still have real actions at the start of this turn
  if (!hasAnyRealAction(candidate)) {
    return { isVictory: false, reason: 'candidate_no_actions' };
  }

  // All other players with material must have no real actions
  for (const player of gameState.players) {
    if (player.playerNumber === candidate) {
      continue;
    }
    if (!hasMaterial(player.playerNumber)) {
      continue;
    }
    if (hasAnyRealAction(player.playerNumber)) {
      return { isVictory: false, reason: 'other_player_has_actions' };
    }
  }

  return { isVictory: true, winner: candidate };
}

/**
 * Check if a phase is an interactive phase for LPS tracking.
 */
export function isLpsActivePhase(phase: GamePhase): boolean {
  return LPS_ACTIVE_PHASES.includes(phase);
}

/**
 * Build a GameResult for an LPS victory.
 *
 * This creates a standard GameResult with 'last_player_standing' reason
 * and computed final scores.
 *
 * @param gameState - Current game state
 * @param winner - The winning player number
 * @returns GameResult with LPS victory details
 */
export function buildLpsVictoryResult(gameState: GameState, winner: number): GameResult {
  const board = gameState.board;

  // Compute per-player ring counts
  const perPlayer: Map<
    number,
    { ringsRemaining: number; territorySpaces: number; ringsEliminated: number }
  > = new Map();

  for (const p of gameState.players) {
    perPlayer.set(p.playerNumber, {
      ringsRemaining: 0,
      territorySpaces: p.territorySpaces ?? 0,
      ringsEliminated: p.eliminatedRings ?? 0,
    });
  }

  for (const stack of board.stacks.values()) {
    const entry = perPlayer.get(stack.controllingPlayer);
    if (entry) {
      entry.ringsRemaining += stack.stackHeight;
    }
  }

  // Build final score objects
  const ringsRemaining: { [playerNumber: number]: number } = {};
  const territorySpaces: { [playerNumber: number]: number } = {};
  const ringsEliminated: { [playerNumber: number]: number } = {};

  for (const p of gameState.players) {
    const entry = perPlayer.get(p.playerNumber);
    ringsRemaining[p.playerNumber] = entry?.ringsRemaining ?? 0;
    territorySpaces[p.playerNumber] = entry?.territorySpaces ?? 0;
    ringsEliminated[p.playerNumber] = entry?.ringsEliminated ?? 0;
  }

  return {
    winner,
    reason: 'last_player_standing',
    finalScore: {
      ringsEliminated,
      territorySpaces,
      ringsRemaining,
    },
  };
}
