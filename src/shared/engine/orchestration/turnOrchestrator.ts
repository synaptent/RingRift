/**
 * ═══════════════════════════════════════════════════════════════════════════
 * Turn Orchestrator
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * The canonical processTurn function that orchestrates all turn phases.
 * This is the single entry point for turn processing, delegating to
 * domain aggregates for actual logic.
 */

import type {
  GameState,
  GamePhase,
  Move,
  Territory,
  Position,
  MoveType,
  GameResult,
  BoardState,
} from '../../types/game';
import { positionToString, BOARD_CONFIGS } from '../../types/game';
import type {
  GameEndExplanation,
  GameEndEngineView,
  GameEndOutcomeType,
  GameEndPlayerScoreBreakdown,
  GameEndRulesContextTag,
  GameEndTeachingLink,
  GameEndTiebreakStep,
  GameEndUxCopyKeys,
  GameEndVictoryReasonCode,
  GameEndWeirdStateContext,
  GameEndRulesWeirdStateReasonCode,
} from '../gameEndExplanation';
import { buildGameEndExplanationFromEngineView } from '../gameEndExplanation';
import {
  getWeirdStateReasonForType,
  getWeirdStateReasonForGameResult,
  getTeachingTopicForReason,
} from '../weirdStateReasons';
import type { VictoryResult as AggregateVictoryResult } from '../aggregates/VictoryAggregate';

import { hashGameState, computeProgressSnapshot } from '../core';
import {
  isANMState,
  applyForcedEliminationForPlayer,
  computeGlobalLegalActionsSummary,
  enumerateForcedEliminationOptions,
  playerHasAnyRings,
  hasAnyGlobalMovementOrCapture,
} from '../globalActions';

import type {
  ProcessTurnResult,
  PendingDecision,
  TurnProcessingDelegates,
  ProcessingMetadata,
  VictoryState,
  DetectedLineInfo,
  PlayerScore,
  FSMDecisionSurface,
} from './types';
import type { TurnProcessingState, PerTurnFlags } from './types';

import { flagEnabled, debugLog, fsmTraceLog } from '../../utils/envFlags';
import { applySwapSidesIdentitySwap, validateSwapSidesMove } from '../swapSidesHelpers';
import { createLegacyCoercionError } from '../errors/CanonicalRecordError';
import {
  detectPhaseCoercion,
  applyPhaseCoercion,
  logPhaseCoercion,
} from '../legacy/legacyReplayHelper';
import {
  assertNoLegacyMoveType,
  isLegacyMoveType,
  normalizeLegacyMove,
} from '../legacy/legacyMoveTypes';

// FSM imports
import {
  validateMoveWithFSM,
  isMoveTypeValidForPhase,
  onLineProcessingComplete,
  onTerritoryProcessingComplete,
  computeFSMOrchestration,
  type FSMOrchestrationResult,
} from '../fsm';

// Import from domain aggregates
import {
  validatePlacement,
  mutatePlacement,
  enumeratePlacementPositions,
  evaluateSkipPlacementEligibility,
} from '../aggregates/PlacementAggregate';

import {
  validateMovement,
  enumerateSimpleMovesForPlayer,
  applySimpleMovement,
} from '../aggregates/MovementAggregate';

import {
  validateCapture,
  enumerateAllCaptureMoves,
  applyCaptureSegment,
  getChainCaptureContinuationInfo,
} from '../aggregates/CaptureAggregate';

import {
  findAllLines,
  enumerateProcessLineMoves,
  applyProcessLineDecision,
  applyChooseLineRewardDecision,
} from '../aggregates/LineAggregate';
import { getEffectiveLineLengthThreshold } from '../rulesConfig';

import {
  getProcessableTerritoryRegions,
  enumerateProcessTerritoryRegionMoves,
  enumerateTerritoryEliminationMoves,
  applyProcessTerritoryRegionDecision,
  applyEliminateRingsFromStackDecision,
} from '../aggregates/TerritoryAggregate';
import type { TerritoryEliminationScope } from '../aggregates/TerritoryAggregate';

import { evaluateVictory } from '../aggregates/VictoryAggregate';

import {
  enumerateExpandedRecoverySlideTargets,
  validateRecoverySlide,
  applyRecoverySlide,
} from '../aggregates/RecoveryAggregate';
import type { RecoverySlideMove } from '../aggregates/RecoveryAggregate';

/**
 * Extended Move interface for cross-system interop (e.g., Python AI service).
 * Python may use snake_case or different property names for recovery moves.
 */
interface ExtendedMoveProperties {
  recoveryOption?: 1 | 2;
  option?: 1 | 2;
  collapsePositions?: Position[];
  collapse_positions?: Position[];
  recoveryMode?: 'line' | 'fallback';
  extractionStacks?: string[];
  extraction_stacks?: string[];
}

import { isEligibleForRecovery } from '../playerStateHelpers';

import { countRingsOnBoardForPlayer } from '../core';

// ═══════════════════════════════════════════════════════════════════════════
// Inline Processing State Container
// ═══════════════════════════════════════════════════════════════════════════
// This inline class provides mutable state container functionality for turn
// processing. The legacy PhaseStateMachine was fully removed in PASS30-R1;
// phase transition logic is now handled by the FSM (TurnStateMachine).

/**
 * Mutable container for turn processing state.
 *
 * This is a lightweight state container that holds TurnProcessingState and
 * provides convenient getters/setters for use during processTurn and
 * processPostMovePhases. It does NOT implement phase transition logic -
 * that is handled by the FSM (TurnStateMachine).
 *
 * @internal Used only within turnOrchestrator.ts
 */
class ProcessingStateContainer {
  private state: TurnProcessingState;

  constructor(initialState: TurnProcessingState) {
    this.state = initialState;
  }

  /** Get current phase from game state. */
  get currentPhase(): GamePhase {
    return this.state.gameState.currentPhase;
  }

  /** Get current game state. */
  get gameState(): GameState {
    return this.state.gameState;
  }

  /** Get current processing state. */
  get processingState(): TurnProcessingState {
    return this.state;
  }

  /** Update the game state. */
  updateGameState(newState: GameState): void {
    this.state.gameState = newState;
    this.state.phasesTraversed.push(newState.currentPhase);
  }

  /** Update per-turn flags. */
  updateFlags(flags: Partial<PerTurnFlags>): void {
    this.state.perTurnFlags = { ...this.state.perTurnFlags, ...flags };
  }

  /** Set chain capture state. */
  setChainCapture(inProgress: boolean, position?: Position): void {
    this.state.chainCaptureInProgress = inProgress;
    this.state.chainCapturePosition = position;
  }

  /** Transition to a specific phase. */
  transitionTo(phase: GamePhase): void {
    this.state.gameState = {
      ...this.state.gameState,
      currentPhase: phase,
    };
    this.state.phasesTraversed.push(phase);
  }

  /** Set pending lines for processing. */
  setPendingLines(lines: TurnProcessingState['pendingLines']): void {
    this.state.pendingLines = lines;
  }

  /** Set pending regions for processing. */
  setPendingRegions(regions: TurnProcessingState['pendingRegions']): void {
    this.state.pendingRegions = regions;
  }

  /** Get the number of pending lines. */
  get pendingLineCount(): number {
    return this.state.pendingLines.length;
  }

  /** Get the number of pending regions. */
  get pendingRegionCount(): number {
    return this.state.pendingRegions.length;
  }
}

/**
 * Create a fresh turn processing state.
 * @internal Used only within turnOrchestrator.ts
 */
function createProcessingState(gameState: GameState, move: Move): TurnProcessingState {
  return {
    gameState,
    originalMove: move,
    perTurnFlags: {
      hasPlacedThisTurn: false,
      mustMoveFromStackKey: undefined,
      eliminationRewardPending: false,
      eliminationRewardCount: 0,
      hadActionThisTurn: false,
    },
    pendingLines: [],
    pendingRegions: [],
    chainCaptureInProgress: false,
    chainCapturePosition: undefined,
    events: [],
    phasesTraversed: [gameState.currentPhase],
    startTime: Date.now(),
  };
}

// ═══════════════════════════════════════════════════════════════════════════
// Turn Rotation Helper
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Compute the next player after the given player, skipping permanently eliminated
 * players (RR-CANON-R201).
 *
 * A player is permanently eliminated if they have no rings anywhere:
 * - No controlled stacks (top ring)
 * - No buried rings (their rings inside stacks controlled by others)
 * - No rings in hand
 *
 * Such players are removed from turn rotation entirely.
 *
 * @param state - The current game state (used to check player elimination status)
 * @param currentPlayerIndex - The index of the player whose turn just ended
 * @param players - Array of player states
 * @returns Object with nextPlayerIndex and nextPlayer number
 */
function computeNextNonEliminatedPlayer(
  state: GameState,
  currentPlayerIndex: number,
  players: GameState['players']
): { nextPlayerIndex: number; nextPlayer: number } {
  const numPlayers = players.length;
  let nextPlayerIndex = (currentPlayerIndex + 1) % numPlayers;
  let skips = 0;

  // Skip up to numPlayers times to find a non-eliminated player
  while (skips < numPlayers) {
    const candidate = players[nextPlayerIndex];
    if (playerHasAnyRings(state, candidate.playerNumber)) {
      return { nextPlayerIndex, nextPlayer: candidate.playerNumber };
    }
    // Player has no rings anywhere - permanently eliminated, skip
    nextPlayerIndex = (nextPlayerIndex + 1) % numPlayers;
    skips += 1;
  }

  // All players eliminated - return the simple rotation (shouldn't happen in valid games)
  const fallbackIndex = (currentPlayerIndex + 1) % numPlayers;
  return { nextPlayerIndex: fallbackIndex, nextPlayer: players[fallbackIndex].playerNumber };
}

// ═══════════════════════════════════════════════════════════════════════════
// S-Invariant Computation
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Compute the S-invariant for the current game state.
 * S = markers + collapsedSpaces + eliminatedRings (should be non-decreasing)
 */
function computeSInvariant(state: GameState): number {
  return computeProgressSnapshot(state).S;
}

/**
 * Resolve an ANM(state, currentPlayer) situation by applying forced elimination
 * and/or re-running victory evaluation until either:
 *
 * - The game becomes terminal, or
 * - The active player has at least one global legal action and ANM no longer holds.
 *
 * This encodes the RR-CANON R202/R203 semantics used by
 * INV-ACTIVE-NO-MOVES / INV-ANM-TURN-MATERIAL-SKIP: no externally visible
 * ACTIVE state may satisfy isANMState(state).
 */
function resolveANMForCurrentPlayer(state: GameState): {
  nextState: GameState;
  victoryResult?: VictoryState;
} {
  let workingState = state;

  // Conservative safety bound: one step per ring currently recorded in play
  // plus a small constant. Forced elimination is monotone in E (and S), so
  // ANM-resolution chains are finite (INV-ELIMINATION-MONOTONIC / INV-TERMINATION).
  const totalRingsInPlay =
    (workingState as GameState & { totalRingsInPlay?: number }).totalRingsInPlay ??
    workingState.players.reduce((sum, p) => sum + p.ringsInHand, 0) +
      Array.from(workingState.board.stacks.values()).reduce(
        (sum, stack) => sum + stack.stackHeight,
        0
      );
  const maxSteps = Math.max(4, totalRingsInPlay + 4);

  for (let i = 0; i < maxSteps; i++) {
    if (workingState.gameStatus !== 'active') {
      return { nextState: workingState };
    }

    if (!isANMState(workingState)) {
      return { nextState: workingState };
    }

    const player = workingState.currentPlayer;
    const forced = applyForcedEliminationForPlayer(workingState, player);

    if (!forced) {
      // No forced-elimination action available; fall back to a victory /
      // stalemate evaluation (R170–R173, R203).
      const victory = toVictoryState(workingState);
      if (victory.isGameOver) {
        const terminalState: GameState = {
          ...workingState,
          gameStatus: 'completed',
          winner: victory.winner,
          currentPhase: 'game_over',
        };
        return { nextState: terminalState, victoryResult: victory };
      }
      // If victory logic cannot classify the position as terminal we stop
      // resolving here and return the ANM state to the caller; this should
      // not happen for canonical states and will be surfaced by invariants.
      return { nextState: workingState };
    }

    workingState = forced.nextState;

    const victoryAfter = toVictoryState(workingState);
    if (victoryAfter.isGameOver) {
      const terminalState: GameState = {
        ...workingState,
        gameStatus: 'completed',
        winner: victoryAfter.winner,
        currentPhase: 'game_over',
      };
      return { nextState: terminalState, victoryResult: victoryAfter };
    }
  }

  // Safety fallback: after exhausting the bound, return the latest state.
  return { nextState: workingState };
}

// ═══════════════════════════════════════════════════════════════════════════
// Victory State Conversion
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Convert evaluateVictory result to VictoryState for the orchestrator and, when
 * the game has ended, attach a structured GameEndExplanation.
 *
 * Note: Exported primarily for internal/testing use; hosts should continue to
 * prefer processTurn / processTurnAsync as the main entrypoints.
 */
export function toVictoryState(state: GameState): VictoryState {
  const result: AggregateVictoryResult = evaluateVictory(state);

  const scores: PlayerScore[] = state.players.map((p) => ({
    player: p.playerNumber,
    eliminatedRings: p.eliminatedRings,
    territorySpaces: p.territorySpaces,
    ringsOnBoard: countRingsOnBoardForPlayer(state.board, p.playerNumber),
    ringsInHand: p.ringsInHand,
    markerCount: 0, // Will be computed below
    isEliminated: false, // Will be computed below
  }));

  // Compute marker counts
  for (const marker of state.board.markers.values()) {
    const scoreEntry = scores.find((s) => s.player === marker.player);
    if (scoreEntry) {
      scoreEntry.markerCount++;
    }
  }

  // Compute elimination status
  for (const score of scores) {
    score.isEliminated = score.ringsOnBoard === 0 && score.ringsInHand === 0;
  }

  const victoryState: VictoryState = {
    isGameOver: result.isGameOver,
    winner: result.winner,
    reason: result.reason as VictoryState['reason'],
    scores,
    tieBreaker: undefined,
  };

  if (victoryState.isGameOver && result.reason) {
    const explanation = buildGameEndExplanationForVictory(state, victoryState, result, scores);
    if (explanation) {
      victoryState.gameEndExplanation = explanation;
    }
  }

  return victoryState;
}

/**
 * Test-only wrapper that exposes the internal victory conversion logic without
 * requiring a full processTurn invocation. This is not part of the public
 * engine API and should only be used in unit tests.
 */
export function __test_only_toVictoryState(state: GameState): VictoryState {
  return toVictoryState(state);
}

/**
 * Build a minimal GameEndExplanation from the current GameState and VictoryState.
 *
 * This keeps the explanation logic close to the canonical victory evaluation
 * while remaining side-effect free.
 */
function buildGameEndExplanationForVictory(
  state: GameState,
  victoryState: VictoryState,
  aggregate: AggregateVictoryResult,
  scores: PlayerScore[]
): GameEndExplanation | undefined {
  const players = state.players;
  if (!players || players.length === 0) {
    return undefined;
  }

  const winnerNumber = victoryState.winner;
  const winnerPlayerId = typeof winnerNumber === 'number' ? toPlayerId(winnerNumber) : null;

  let outcomeType: GameEndOutcomeType | undefined;
  let victoryReasonCode: GameEndVictoryReasonCode | undefined;
  let primaryConceptId: string | undefined;
  let tiebreakSteps: GameEndTiebreakStep[] | undefined;
  let weirdStateContext: GameEndWeirdStateContext | undefined;
  let telemetryTags: GameEndRulesContextTag[] | undefined;
  let teaching: GameEndTeachingLink | undefined;

  const primaryRingWinner = players.find((p) => p.eliminatedRings >= state.victoryThreshold);
  const primaryTerritoryWinner = players.find(
    (p) => p.territorySpaces >= state.territoryVictoryThreshold
  );
  const noStacksLeft = state.board.stacks.size === 0;

  const aggReason = aggregate.reason;
  if (!aggReason) {
    return undefined;
  }

  const hadForcedEliminationSequence = hasForcedEliminationMove(state);

  if (aggReason === 'ring_elimination') {
    if (primaryRingWinner && !noStacksLeft) {
      // Standard ring-majority threshold victory.
      outcomeType = 'ring_elimination';
      victoryReasonCode = 'victory_ring_majority';
    } else if (noStacksLeft && !primaryRingWinner) {
      // Bare-board stalemate ladder resolved via eliminated-rings tiebreak.
      outcomeType = 'structural_stalemate';
      victoryReasonCode = 'victory_structural_stalemate_tiebreak';
      primaryConceptId = 'structural_stalemate';

      const treatHandAsEliminated = !!aggregate.handCountsAsEliminated;
      const valuesByPlayer: Record<string, number> = {};
      for (const p of players) {
        const pid = toPlayerId(p.playerNumber);
        const eliminationScore = p.eliminatedRings + (treatHandAsEliminated ? p.ringsInHand : 0);
        valuesByPlayer[pid] = eliminationScore;
      }
      tiebreakSteps = [
        {
          kind: 'eliminated_rings',
          winnerPlayerId,
          valuesByPlayer,
        },
      ];

      const info = getWeirdStateReasonForType('structural-stalemate');
      const reasonCodes: GameEndRulesWeirdStateReasonCode[] = [info.reasonCode];
      const rulesContextTags: GameEndRulesContextTag[] = [info.rulesContext];
      const teachingTopicIds: string[] = [getTeachingTopicForReason(info.reasonCode)];

      if (hadForcedEliminationSequence) {
        const anmInfo = getWeirdStateReasonForType('active-no-moves-movement');
        const feInfo = getWeirdStateReasonForType('forced-elimination');
        reasonCodes.push(anmInfo.reasonCode, feInfo.reasonCode);
        rulesContextTags.push(anmInfo.rulesContext, feInfo.rulesContext);
        teachingTopicIds.push(
          getTeachingTopicForReason(anmInfo.reasonCode),
          getTeachingTopicForReason(feInfo.reasonCode)
        );
      }

      const dedupRulesContexts = Array.from(new Set(rulesContextTags));
      const dedupTeaching = Array.from(new Set(teachingTopicIds));

      weirdStateContext = {
        reasonCodes,
        primaryReasonCode: info.reasonCode,
        rulesContextTags: dedupRulesContexts,
        teachingTopicIds: dedupTeaching,
      };
      teaching = {
        teachingTopics: dedupTeaching,
      };
      telemetryTags = dedupRulesContexts;
    } else {
      // Fallback: treat as standard ring-elimination.
      outcomeType = 'ring_elimination';
      victoryReasonCode = 'victory_ring_majority';
    }
  } else if (aggReason === 'territory_control') {
    if (primaryTerritoryWinner && !noStacksLeft) {
      outcomeType = 'territory_control';
      victoryReasonCode = 'victory_territory_majority';

      // Detect if this is a mini-region territory victory (Q23-style scenario)
      const miniRegionInfo = detectTerritoryMiniRegions(state.board, winnerNumber);
      if (miniRegionInfo.isMiniRegionVictory) {
        primaryConceptId = 'territory_mini_regions';
        weirdStateContext = {
          reasonCodes: ['ANM_TERRITORY_NO_ACTIONS'],
          primaryReasonCode: 'ANM_TERRITORY_NO_ACTIONS',
          rulesContextTags: ['territory_mini_region'],
          teachingTopicIds: ['teaching.territory'],
        };
        telemetryTags = ['territory_mini_region'];
        teaching = {
          teachingTopics: ['teaching.territory'],
        };
      }
    } else if (noStacksLeft && !primaryTerritoryWinner) {
      // Bare-board structural stalemate resolved via territory tiebreak.
      outcomeType = 'structural_stalemate';
      victoryReasonCode = 'victory_structural_stalemate_tiebreak';
      primaryConceptId = 'structural_stalemate';

      const valuesByPlayer: Record<string, number> = {};
      for (const p of players) {
        const pid = toPlayerId(p.playerNumber);
        valuesByPlayer[pid] = p.territorySpaces;
      }
      tiebreakSteps = [
        {
          kind: 'territory_spaces',
          winnerPlayerId,
          valuesByPlayer,
        },
      ];

      const info = getWeirdStateReasonForType('structural-stalemate');
      const reasonCodes: GameEndRulesWeirdStateReasonCode[] = [info.reasonCode];
      const rulesContextTags: GameEndRulesContextTag[] = [info.rulesContext];
      const teachingTopicIds: string[] = [getTeachingTopicForReason(info.reasonCode)];

      if (hadForcedEliminationSequence) {
        const anmInfo = getWeirdStateReasonForType('active-no-moves-movement');
        const feInfo = getWeirdStateReasonForType('forced-elimination');
        reasonCodes.push(anmInfo.reasonCode, feInfo.reasonCode);
        rulesContextTags.push(anmInfo.rulesContext, feInfo.rulesContext);
        teachingTopicIds.push(
          getTeachingTopicForReason(anmInfo.reasonCode),
          getTeachingTopicForReason(feInfo.reasonCode)
        );
      }

      const dedupRulesContexts = Array.from(new Set(rulesContextTags));
      const dedupTeaching = Array.from(new Set(teachingTopicIds));

      weirdStateContext = {
        reasonCodes,
        primaryReasonCode: info.reasonCode,
        rulesContextTags: dedupRulesContexts,
        teachingTopicIds: dedupTeaching,
      };
      teaching = {
        teachingTopics: dedupTeaching,
      };
      telemetryTags = dedupRulesContexts;
    } else {
      outcomeType = 'territory_control';
      victoryReasonCode = 'victory_territory_majority';
    }
  } else if (aggReason === 'last_player_standing') {
    outcomeType = 'last_player_standing';
    victoryReasonCode = 'victory_last_player_standing';
    primaryConceptId = 'lps_real_actions';

    const reasonCodes: GameEndRulesWeirdStateReasonCode[] = [];
    const rulesContextTags: GameEndRulesContextTag[] = [];
    const teachingTopicIds: string[] = [];

    // Base LPS reason: exclusive real actions over full rounds.
    const lpsWeird = getWeirdStateReasonForGameResult({
      winner: aggregate.winner,
      reason: 'last_player_standing',
      finalScore: {
        ringsEliminated: {},
        territorySpaces: {},
        ringsRemaining: {},
      },
    } as GameResult);

    if (lpsWeird) {
      reasonCodes.push(lpsWeird.reasonCode);
      rulesContextTags.push(lpsWeird.rulesContext);
      teachingTopicIds.push(getTeachingTopicForReason(lpsWeird.reasonCode));
    }

    // Enrich LPS endings that involved ANM/FE sequences with additional
    // weird-state reason codes so VictoryModal / TeachingOverlay can
    // explain both the exclusive-real-actions condition and the role of
    // forced elimination.
    if (hadForcedEliminationSequence) {
      const anmInfo = getWeirdStateReasonForType('active-no-moves-movement');
      const feInfo = getWeirdStateReasonForType('forced-elimination');

      reasonCodes.push(anmInfo.reasonCode, feInfo.reasonCode);
      rulesContextTags.push(anmInfo.rulesContext, feInfo.rulesContext);
      teachingTopicIds.push(
        getTeachingTopicForReason(anmInfo.reasonCode),
        getTeachingTopicForReason(feInfo.reasonCode)
      );
    }

    if (reasonCodes.length > 0 || rulesContextTags.length > 0 || teachingTopicIds.length > 0) {
      const dedupReasonCodes = Array.from(new Set(reasonCodes));
      const dedupRulesContexts = Array.from(new Set(rulesContextTags));
      const dedupTeaching = Array.from(new Set(teachingTopicIds));

      // Choose a canonical primary reason code. Prefer the explicit LPS reason
      // (exclusive real actions) when available, otherwise fall back to the
      // first deduplicated reason code. This avoids assigning `undefined`
      // under exactOptionalPropertyTypes while still reflecting the most
      // important concept.
      const primaryReasonCode = (lpsWeird && lpsWeird.reasonCode) || dedupReasonCodes[0];

      if (primaryReasonCode) {
        weirdStateContext = {
          reasonCodes: dedupReasonCodes,
          primaryReasonCode,
          rulesContextTags: dedupRulesContexts,
          teachingTopicIds: dedupTeaching,
        };
        telemetryTags = dedupRulesContexts;
      }
    }
  } else if (aggReason === 'game_completed') {
    // Fallback structural stalemate with no clear winner.
    outcomeType = 'structural_stalemate';
    victoryReasonCode = 'victory_structural_stalemate_tiebreak';
    primaryConceptId = 'structural_stalemate';

    const info = getWeirdStateReasonForType('structural-stalemate');
    const reasonCodes: GameEndRulesWeirdStateReasonCode[] = [info.reasonCode];
    const rulesContextTags: GameEndRulesContextTag[] = [info.rulesContext];
    const teachingTopicIds: string[] = [getTeachingTopicForReason(info.reasonCode)];

    if (hadForcedEliminationSequence) {
      const anmInfo = getWeirdStateReasonForType('active-no-moves-movement');
      const feInfo = getWeirdStateReasonForType('forced-elimination');
      reasonCodes.push(anmInfo.reasonCode, feInfo.reasonCode);
      rulesContextTags.push(anmInfo.rulesContext, feInfo.rulesContext);
      teachingTopicIds.push(
        getTeachingTopicForReason(anmInfo.reasonCode),
        getTeachingTopicForReason(feInfo.reasonCode)
      );
    }

    const dedupRulesContexts = Array.from(new Set(rulesContextTags));
    const dedupTeaching = Array.from(new Set(teachingTopicIds));

    weirdStateContext = {
      reasonCodes,
      primaryReasonCode: info.reasonCode,
      rulesContextTags: dedupRulesContexts,
      teachingTopicIds: dedupTeaching,
    };
    teaching = {
      teachingTopics: dedupTeaching,
    };
    telemetryTags = dedupRulesContexts;
  } else {
    return undefined;
  }

  if (!outcomeType || !victoryReasonCode) {
    return undefined;
  }

  const scoreBreakdown = createScoreBreakdown(scores);

  const view: GameEndEngineView = {
    gameId: state.id,
    boardType: state.boardType,
    numPlayers: players.length,
    winnerPlayerId,
    outcomeType,
    victoryReasonCode,
  };

  if (Object.keys(scoreBreakdown).length > 0) {
    view.scoreBreakdown = scoreBreakdown;
  }
  if (tiebreakSteps && tiebreakSteps.length > 0) {
    view.tiebreakSteps = tiebreakSteps;
  }
  if (primaryConceptId) {
    view.primaryConceptId = primaryConceptId;
  }
  if (weirdStateContext) {
    view.weirdStateContext = weirdStateContext;
  }

  const uxCopy: GameEndUxCopyKeys = deriveUxCopyKeys(
    outcomeType,
    primaryConceptId,
    aggReason,
    hadForcedEliminationSequence
  );

  const extra: {
    teaching?: GameEndTeachingLink;
    telemetryTags?: GameEndRulesContextTag[];
    uxCopy: GameEndUxCopyKeys;
  } = { uxCopy };

  if (telemetryTags && telemetryTags.length > 0) {
    extra.telemetryTags = telemetryTags;
  }
  if (teaching) {
    extra.teaching = teaching;
  }

  const explanation = buildGameEndExplanationFromEngineView(view, extra);

  return explanation;
}

function toPlayerId(playerNumber: number): string {
  return `P${playerNumber}`;
}

function createScoreBreakdown(scores: PlayerScore[]): Record<string, GameEndPlayerScoreBreakdown> {
  const breakdown: Record<string, GameEndPlayerScoreBreakdown> = {};
  for (const score of scores) {
    const playerId = toPlayerId(score.player);
    breakdown[playerId] = {
      playerId,
      eliminatedRings: score.eliminatedRings,
      territorySpaces: score.territorySpaces,
      markers: score.markerCount,
    };
  }
  return breakdown;
}

function deriveShortSummaryKey(outcomeType: GameEndOutcomeType, primaryConceptId?: string): string {
  if (outcomeType === 'ring_elimination') {
    return 'game_end.ring_elimination.short';
  }
  if (outcomeType === 'territory_control') {
    if (primaryConceptId === 'territory_mini_regions') {
      return 'game_end.territory_mini_region.short';
    }
    return 'game_end.territory_control.short';
  }
  if (outcomeType === 'last_player_standing') {
    // Generic LPS key; FE-heavy variants use game_end.lps.with_anm_fe.* via deriveUxCopyKeys.
    return 'game_end.lps.short';
  }
  if (outcomeType === 'structural_stalemate') {
    return 'game_end.structural_stalemate.short';
  }
  return 'game_end.generic.short';
}

/**
 * Derive UX copy keys (short + detailed) for complex endings.
 *
 * The shortSummaryKey drives compact HUD banners; detailedSummaryKey is used by
 * VictoryModal / TeachingOverlay to select richer explanation copy. Keys are
 * aligned with docs/ux/UX_RULES_COPY_SPEC.md and
 * docs/ux/UX_RULES_EXPLANATION_MODEL_SPEC.md.
 */
function deriveUxCopyKeys(
  outcomeType: GameEndOutcomeType,
  primaryConceptId: string | undefined,
  aggregateReason: AggregateVictoryResult['reason'],
  hadForcedEliminationSequence: boolean
): GameEndUxCopyKeys {
  // Territory mini-region endings (Q23-style cascades / mini-regions).
  if (outcomeType === 'territory_control' && primaryConceptId === 'territory_mini_regions') {
    return {
      shortSummaryKey: 'game_end.territory_mini_region.short',
      detailedSummaryKey: 'game_end.territory_mini_region.detailed',
    };
  }

  // Last-Player-Standing endings. Distinguish FE-heavy ANM/FE sequences from
  // generic LPS endings via the presence of forced_elimination moves.
  if (outcomeType === 'last_player_standing') {
    if (hadForcedEliminationSequence) {
      return {
        shortSummaryKey: 'game_end.lps.with_anm_fe.short',
        detailedSummaryKey: 'game_end.lps.with_anm_fe.detailed',
      };
    }
    return {
      shortSummaryKey: 'game_end.lps.short',
      detailedSummaryKey: 'game_end.lps.detailed',
    };
  }

  // Structural stalemate and tiebreak ladder. When aggregateReason is
  // 'game_completed' we treat this as a plateau without a clear tiebreak
  // winner; otherwise the ladder selected a winner via territory / rings /
  // markers / last actor.
  if (outcomeType === 'structural_stalemate') {
    if (aggregateReason === 'game_completed') {
      return {
        shortSummaryKey: 'game_end.structural_stalemate.short',
        detailedSummaryKey: 'game_end.structural_stalemate.detailed',
      };
    }
    return {
      shortSummaryKey: 'game_end.structural_stalemate.short',
      detailedSummaryKey: 'game_end.structural_stalemate.tiebreak.detailed',
    };
  }

  // Default mapping for simpler endings (ring / territory / generic).
  return {
    shortSummaryKey: deriveShortSummaryKey(outcomeType, primaryConceptId),
  };
}

/**
 * Detect whether the completed game involved at least one explicit
 * forced_elimination move. This is a pure inspection helper used only for
 * explanation/UX enrichment; it does not affect rules semantics or victory
 * evaluation.
 *
 * NOTE: FSM is now canonical, so state.history is always populated.
 * The legacy moveHistory fallback has been removed.
 */
function hasForcedEliminationMove(state: GameState): boolean {
  // FSM canonical: structured history is always available
  if (state.history && state.history.length > 0) {
    return state.history.some(
      (entry) => entry.action && entry.action.type === 'forced_elimination'
    );
  }
  return false;
}

// ═══════════════════════════════════════════════════════════════════════════
// Mini-Region Detection
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Result of detecting territory mini-regions for explanation purposes.
 */
interface MiniRegionDetectionResult {
  /** Whether the victory qualifies as a mini-region scenario */
  isMiniRegionVictory: boolean;
  /** Number of disconnected territory regions controlled by the winner */
  regionCount: number;
  /** True if at least one region is considered "mini" (small isolated region) */
  hasMiniRegion: boolean;
}

/**
 * Detect if a territory victory involves mini-regions (Q23-style scenarios).
 *
 * A territory mini-region ending is detected when the winner's controlled
 * collapsed territory consists of 2+ disconnected regions, or when they
 * have collapsed multiple small (≤4 cells) isolated regions.
 *
 * This is a UX-level detection for explanation enrichment; it does not
 * affect game rules or victory semantics.
 *
 * @param board - The final board state
 * @param winnerPlayer - Player number of the winner (may be null/undefined for draws)
 * @returns Detection result with region information
 */
function detectTerritoryMiniRegions(
  board: BoardState,
  winnerPlayer: number | null | undefined
): MiniRegionDetectionResult {
  if (winnerPlayer == null) {
    return { isMiniRegionVictory: false, regionCount: 0, hasMiniRegion: false };
  }

  // Find all collapsed spaces owned by the winner
  const winnerCollapsedPositions: Position[] = [];
  for (const [key, owner] of board.collapsedSpaces.entries()) {
    if (owner === winnerPlayer) {
      winnerCollapsedPositions.push(parsePositionKey(key, board.type));
    }
  }

  if (winnerCollapsedPositions.length === 0) {
    return { isMiniRegionVictory: false, regionCount: 0, hasMiniRegion: false };
  }

  // Group collapsed spaces into connected regions using flood-fill
  const regions = groupIntoConnectedRegions(winnerCollapsedPositions, board);

  // Mini-region detection criteria:
  // 1. Winner has 2+ disconnected territory regions, OR
  // 2. Winner has at least one small (≤4 cells) isolated region
  const MINI_REGION_SIZE_THRESHOLD = 4;
  const hasMiniRegion = regions.some((region) => region.length <= MINI_REGION_SIZE_THRESHOLD);
  const isMiniRegionVictory = regions.length >= 2 || hasMiniRegion;

  return {
    isMiniRegionVictory,
    regionCount: regions.length,
    hasMiniRegion,
  };
}

/**
 * Parse a position key string back into a Position object.
 */
function parsePositionKey(key: string, boardType: string): Position {
  // Position keys are "x,y" for square boards or "x,y,z" for hexagonal
  const parts = key.split(',').map(Number);
  if (boardType === 'hexagonal' && parts.length >= 3) {
    return { x: parts[0], y: parts[1], z: parts[2] };
  }
  return { x: parts[0], y: parts[1] };
}

/**
 * Group positions into connected regions using flood-fill.
 *
 * @param positions - All positions to group
 * @param board - Board for adjacency calculation
 * @returns Array of connected region groups
 */
function groupIntoConnectedRegions(positions: Position[], board: BoardState): Position[][] {
  if (positions.length === 0) {
    return [];
  }

  const positionSet = new Set(positions.map((p) => positionToString(p)));
  const visited = new Set<string>();
  const regions: Position[][] = [];

  for (const pos of positions) {
    const key = positionToString(pos);
    if (visited.has(key)) {
      continue;
    }

    // Flood-fill to find all connected positions
    const region: Position[] = [];
    const queue: Position[] = [pos];
    visited.add(key);

    while (queue.length > 0) {
      const current = queue.shift();
      if (!current) break; // Type guard, should never happen due to loop condition
      region.push(current);

      // Get territory-adjacent neighbors
      const neighbors = getTerritoryNeighborsForMiniRegion(current, board);
      for (const neighbor of neighbors) {
        const neighborKey = positionToString(neighbor);
        if (!visited.has(neighborKey) && positionSet.has(neighborKey)) {
          visited.add(neighborKey);
          queue.push(neighbor);
        }
      }
    }

    if (region.length > 0) {
      regions.push(region);
    }
  }

  return regions;
}

/**
 * Get territory-adjacent neighbors for mini-region detection.
 *
 * Uses the board's configured territory adjacency type.
 */
function getTerritoryNeighborsForMiniRegion(pos: Position, board: BoardState): Position[] {
  const config = BOARD_CONFIGS[board.type];
  const adjacencyType = config.territoryAdjacency;
  const neighbors: Position[] = [];
  const { x, y, z } = pos;

  if (adjacencyType === 'hexagonal') {
    const directions = [
      { x: 1, y: 0, z: -1 },
      { x: 1, y: -1, z: 0 },
      { x: 0, y: -1, z: 1 },
      { x: -1, y: 0, z: 1 },
      { x: -1, y: 1, z: 0 },
      { x: 0, y: 1, z: -1 },
    ];
    for (const dir of directions) {
      neighbors.push({
        x: x + dir.x,
        y: y + dir.y,
        z: (z ?? 0) + dir.z,
      });
    }
  } else if (adjacencyType === 'von_neumann') {
    const directions = [
      { x: 0, y: 1 },
      { x: 1, y: 0 },
      { x: 0, y: -1 },
      { x: -1, y: 0 },
    ];
    for (const dir of directions) {
      neighbors.push({ x: x + dir.x, y: y + dir.y });
    }
  } else {
    // Moore adjacency (8-directional)
    for (let dx = -1; dx <= 1; dx++) {
      for (let dy = -1; dy <= 1; dy++) {
        if (dx === 0 && dy === 0) continue;
        neighbors.push({ x: x + dx, y: y + dy });
      }
    }
  }

  return neighbors;
}

// ═══════════════════════════════════════════════════════════════════════════
// Decision Creation Helpers
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Create a pending decision for line processing.
 */
function createLineOrderDecision(state: GameState, lines: DetectedLineInfo[]): PendingDecision {
  const moves = enumerateProcessLineMoves(state, state.currentPlayer);

  return {
    type: 'line_order',
    player: state.currentPlayer,
    options: moves,
    context: {
      description: `Choose which line to process first (${lines.length} lines available)`,
      relevantPositions: lines.flatMap((l) => l.positions),
    },
  };
}

/**
 * Create a pending decision for territory region processing.
 *
 * RR-FIX-2026-01-12: Build moves directly from the `regions` parameter instead
 * of calling `enumerateProcessTerritoryRegionMoves`. The latter uses moveHistory
 * to check for pending eliminations, but during the decision loop moveHistory
 * is not yet updated. This caused 0-option decisions after eliminate_rings_from_stack
 * moves, breaking visual feedback for sequential territory claims.
 */
function createRegionOrderDecision(state: GameState, regions: Territory[]): PendingDecision {
  const player = state.currentPlayer;
  const moveNumber = state.moveHistory.length + 1;

  // Build moves directly from the regions parameter.
  // This mirrors the logic in enumerateProcessTerritoryRegionMoves but without
  // the moveHistory-based pending elimination check that can be stale during
  // the decision loop.
  const regionMoves: Move[] = [];
  regions.forEach((region, index) => {
    if (!region.spaces || region.spaces.length === 0) {
      return;
    }
    const representative = region.spaces[0];
    const regionKey = representative ? positionToString(representative) : `region-${index}`;

    regionMoves.push({
      id: `process-region-${index}-${regionKey}`,
      type: 'choose_territory_option',
      player,
      to: representative ?? { x: 0, y: 0 },
      disconnectedRegions: [region],
      timestamp: new Date(),
      thinkTime: 0,
      moveNumber,
    } as Move);
  });

  const moves: Move[] = [...regionMoves];

  // When one or more regions are processable for this player and no
  // self-elimination decision is currently outstanding, include the
  // canonical skip_territory_processing Move alongside the explicit
  // choose_territory_option options. This keeps the PendingDecision
  // surface aligned with getValidMoves for territory_processing.
  // Use enumerateTerritoryEliminationMoves for the elimination check since
  // it's safe to call here (we already know regions exist).
  const elimMoves = enumerateTerritoryEliminationMoves(state, player);
  if (regionMoves.length > 0 && elimMoves.length === 0) {
    moves.push({
      id: `skip-territory-${moveNumber}`,
      type: 'skip_territory_processing',
      player,
      to: { x: 0, y: 0 },
      timestamp: new Date(),
      thinkTime: 0,
      moveNumber,
    } as Move);
  }

  return {
    type: 'region_order',
    player,
    options: moves,
    context: {
      description: `Choose which region to process (${regions.length} regions available)`,
      relevantPositions: regions.flatMap((r) => r.spaces),
    },
  };
}

/**
 * Create a pending decision for forced elimination when the current
 * player is blocked with stacks but has no legal placement, movement,
 * or capture actions (RR-CANON-R072/R100/R206).
 *
 * The options are `forced_elimination` Moves, one per eligible
 * stack/standalone ring controlled by the current player.
 * This is distinct from `eliminate_rings_from_stack` which is used
 * during line/territory processing.
 */
function createForcedEliminationDecision(state: GameState): PendingDecision | undefined {
  const player = state.currentPlayer;

  const options: Move[] = [];

  for (const stack of state.board.stacks.values()) {
    if (stack.controllingPlayer !== player || stack.stackHeight <= 0) {
      continue;
    }

    const capHeight: number =
      typeof stack.capHeight === 'number' && stack.capHeight > 0 ? stack.capHeight : 0;
    const count = Math.max(1, capHeight || 0);

    // Use 'forced_elimination' move type for the forced_elimination phase
    // per RR-CANON-R070/R100/R205.
    const move: Move = {
      id: `forced-elim-${positionToString(stack.position)}`,
      type: 'forced_elimination',
      player,
      to: stack.position,
      eliminatedRings: [{ player, count }],
      eliminationFromStack: {
        position: stack.position,
        capHeight,
        totalHeight: stack.stackHeight,
      },
      // Deterministic placeholders for orchestrator-driven decisions; hosts
      // are free to override timestamp/thinkTime/moveNumber as needed.
      timestamp: new Date(0),
      thinkTime: 0,
      moveNumber: state.moveHistory.length + 1,
    } as Move;

    options.push(move);
  }

  if (options.length === 0) {
    return undefined;
  }

  return {
    type: 'elimination_target',
    player,
    options,
    context: {
      description: 'Choose which stack to eliminate from (forced elimination)',
      relevantPositions: options.map((m) => m.to).filter((p): p is Position => !!p),
      extra: {
        reason: 'forced_elimination',
      },
    },
  };
}

/**
 * Create a pending decision for mandatory territory self-elimination.
 * Per RR-CANON-R145 / FAQ Q23, after processing a disconnected region,
 * the player must eliminate their entire cap from a controlled stack
 * outside the processed region.
 */
function createTerritoryEliminationDecision(
  _state: GameState,
  player: number,
  eliminationMoves: Move[]
): PendingDecision {
  return {
    type: 'elimination_target',
    player,
    options: eliminationMoves,
    context: {
      description: 'Territory claimed. You must eliminate your entire cap from an eligible stack.',
      relevantPositions: eliminationMoves.map((m) => m.to).filter((p): p is Position => !!p),
      extra: {
        reason: 'territory_self_elimination',
        eliminationContext: 'territory',
      },
    },
  };
}

/**
 * Create a pending decision for chain capture continuation.
 */
function createChainCaptureDecision(state: GameState, continuations: Move[]): PendingDecision {
  return {
    type: 'chain_capture',
    player: state.currentPlayer,
    options: continuations,
    context: {
      description: 'Continue the capture chain',
    },
  };
}

/**
 * Derive a PendingDecision from FSM orchestration result.
 *
 * This function converts the FSM's pendingDecisionType and decisionSurface
 * into the legacy PendingDecision structure used by hosts. This allows FSM
 * to be the sole authority for decision surfacing while maintaining backward
 * compatibility with existing host code.
 *
 * @param state - Current game state (used to enumerate concrete move options)
 * @param fsmResult - FSM orchestration result containing decision surface
 * @returns PendingDecision if FSM indicates a decision is needed, undefined otherwise
 */
function derivePendingDecisionFromFSM(
  state: GameState,
  fsmResult: FSMOrchestrationResult
): PendingDecision | undefined {
  if (!fsmResult.pendingDecisionType || !fsmResult.decisionSurface) {
    return undefined;
  }

  const player = fsmResult.nextPlayer;

  switch (fsmResult.pendingDecisionType) {
    case 'chain_capture': {
      // Enumerate chain capture continuation moves from current position
      const chainPos = state.chainCapturePosition;
      if (!chainPos) {
        return undefined;
      }
      const info = getChainCaptureContinuationInfo(state, player, chainPos);
      if (!info.mustContinue || info.availableContinuations.length === 0) {
        return undefined;
      }
      return createChainCaptureDecision(state, info.availableContinuations);
    }

    case 'line_order_required': {
      // Enumerate process_line moves for detected lines
      const lines = fsmResult.decisionSurface.pendingLines;
      if (lines.length === 0) {
        return undefined;
      }
      const detectedLines: DetectedLineInfo[] = lines.map((l) => {
        // Compute direction from first two positions if available
        const direction: Position =
          l.positions.length >= 2
            ? {
                x: l.positions[1].x - l.positions[0].x,
                y: l.positions[1].y - l.positions[0].y,
              }
            : { x: 1, y: 0 }; // Default direction
        return {
          positions: l.positions,
          player: 'player' in l ? (l as { player: number }).player : player,
          length: l.positions.length,
          direction,
          collapseOptions: [],
        };
      });
      return createLineOrderDecision(state, detectedLines);
    }

    case 'no_line_action_required': {
      return {
        type: 'no_line_action_required',
        player,
        options: [],
        context: {
          description: 'No lines to process - explicit no_line_action required per RR-CANON-R075',
        },
      };
    }

    case 'region_order_required': {
      // Enumerate choose_territory_option moves
      const territoryEliminationContext = didCurrentTurnIncludeRecoverySlide(state, player)
        ? 'recovery'
        : 'territory';
      const regions = getProcessableTerritoryRegions(state.board, {
        player,
        eliminationContext: territoryEliminationContext,
      });
      if (regions.length === 0) {
        return undefined;
      }
      return createRegionOrderDecision(state, regions);
    }

    case 'no_territory_action_required': {
      return {
        type: 'no_territory_action_required',
        player,
        options: [],
        context: {
          description:
            'No territory regions to process - explicit no_territory_action required per RR-CANON-R075',
        },
      };
    }

    case 'forced_elimination': {
      return createForcedEliminationDecision(state);
    }

    default:
      return undefined;
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// Process Turn (Synchronous Entry Point)
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Options for processTurn behavior.
 *
 * @see RULES_CANONICAL_SPEC.md RR-CANON-R073 (NO phase skipping)
 * @see RULES_CANONICAL_SPEC.md RR-CANON-R075 (Every phase transition recorded)
 */
export interface ProcessTurnOptions {
  /**
   * When true, even single territory regions will return a decision instead
   * of being auto-processed. This is used in replay contexts where explicit
   * choose_territory_option moves from recordings should be used. Legacy
   * aliases are only normalized in replay-compatibility paths.
   */
  skipSingleTerritoryAutoProcess?: boolean;

  /**
   * When true, line-processing is never auto-applied (even for a single
   * exact-length line). Instead, a `line_order` decision is surfaced and
   * the host is expected to apply an explicit `process_line` /
   * `choose_line_option` move. This is primarily used in replay/trace
   * contexts so that recorded move sequences remain the sole source of
   * truth for when lines are processed.
   */
  skipAutoLineProcessing?: boolean;

  /**
   * When true, processTurnAsync will return immediately when a decision is
   * required, instead of calling the delegate to resolve it. This is used
   * in replay contexts where the next move in the recording is the decision
   * resolution.
   */
  breakOnDecisionRequired?: boolean;

  /**
   * RR-PARITY-FIX-2025-12-21: Track the original non-bookkeeping move that
   * started this turn sequence. This is needed because when processing
   * intermediate bookkeeping moves (no_line_action, no_territory_action),
   * the move history hasn't been updated yet, so computeHadAnyActionThisTurn
   * would incorrectly return false.
   */
  turnSequenceRealMoves?: Move[];

  /**
   * Enable legacy/parity replay tolerance.
   *
   * @deprecated Use strict mode with explicit error handling for legacy records.
   * This option will be removed in a future version.
   * Legacy records should be validated via check_canonical_phase_history.py
   * and quarantined rather than silently coerced.
   *
   * When true, the orchestrator may coerce `currentPhase` and/or `currentPlayer`
   * to match the incoming recorded move type in order to replay legacy logs and
   * TS↔Python parity fixtures that were produced before strict canonical phase
   * bookkeeping was enforced.
   *
   * Canonical write paths should keep this **false** so that phase/move
   * mismatches fail fast instead of being silently corrected.
   *
   * @see RULES_CANONICAL_SPEC.md RR-CANON-R073 (NO phase skipping)
   * @see RULES_CANONICAL_SPEC.md RR-CANON-R075 (Every phase transition recorded)
   */
  replayCompatibility?: boolean;
}

/**
 * Process a single move synchronously.
 *
 * This is the main entry point when decisions can be made immediately
 * (e.g., single-line/single-region cases or AI auto-selection).
 *
 * For cases requiring async decision resolution (human player choices),
 * use processTurnAsync or check the 'awaiting_decision' status.
 *
 * @see RULES_CANONICAL_SPEC.md RR-CANON-R073 (NO phase skipping)
 * @see RULES_CANONICAL_SPEC.md RR-CANON-R075 (Every phase transition recorded)
 */
export function processTurn(
  state: GameState,
  rawMove: Move,
  options?: ProcessTurnOptions
): ProcessTurnResult {
  const replayCompatibility = options?.replayCompatibility ?? false;
  const move = replayCompatibility ? normalizeLegacyMove(rawMove) : rawMove;

  // Detect if phase coercion would be needed for this move
  const coercionResult = detectPhaseCoercion(state, move);

  if (coercionResult.needsCoercion) {
    if (options?.replayCompatibility) {
      // DEPRECATED: Legacy replay compatibility mode
      // This code path will be removed in a future version.
      // Use check_canonical_phase_history.py to validate records.

      console.warn(
        '[DEPRECATED] replayCompatibility is deprecated. ' +
          'Non-canonical records should be quarantined, not silently coerced. ' +
          'This option will be removed in a future version. ' +
          `Coercing: ${coercionResult.reason}`
      );

      // Log the coercion for analysis
      logPhaseCoercion(
        state.id,
        state.moveHistory.length + 1,
        coercionResult,
        state.currentPhase,
        state.currentPlayer
      );

      // Apply the coercion (quarantined in legacyReplayHelper.ts)
      state = applyPhaseCoercion(state, coercionResult);
    } else {
      // STRICT MODE (default): Reject non-canonical records with detailed error
      // Per RR-CANON-R073/R075: No phase skipping, every phase transition recorded
      throw createLegacyCoercionError({
        currentPhase: state.currentPhase,
        wouldCoerceTo: coercionResult.targetPhase ?? state.currentPhase,
        moveType: rawMove.type,
        gameId: state.id,
        moveNumber: state.moveHistory.length + 1,
        currentPlayer: state.currentPlayer,
        movePlayer: rawMove.player,
      });
    }
  }

  // Enforce canonical phase→MoveType mapping for ACTIVE states. This ensures
  // that every visited phase is represented by an explicit action, skip, or
  // no-action move per RR-CANON-R075.
  // Skip for legacy replay - FSM validation handles legacy move types correctly.
  if (!replayCompatibility) {
    assertNoLegacyMoveType(rawMove, 'processTurn');
    assertPhaseMoveInvariant(state, move);
  }

  // FSM Validation: FSM is now the canonical validator - always enforce validation.
  const fsmValidationResult = performFSMValidation(state, move, replayCompatibility);
  if (!fsmValidationResult.valid) {
    const reason = fsmValidationResult.reason || `FSM validation rejected ${move.type} move`;
    throw new Error(
      `[FSM] Invalid move: ${reason} (phase=${fsmValidationResult.currentPhase}, errorCode=${fsmValidationResult.errorCode})`
    );
  }

  const sInvariantBefore = computeSInvariant(state);
  const startTime = Date.now();
  const stateMachine = new ProcessingStateContainer(createProcessingState(state, move));

  // Compute hash before applying move to detect if the move actually changed state
  const hashBefore = hashGameState(stateMachine.gameState);

  // Apply the move based on type
  // RR-FIX-2024-12-21: Pass options to applyMoveWithChainInfo so handlers can access
  // turnSequenceRealMoves for accurate hadAnyActionThisTurn checks.
  const applyResult = applyMoveWithChainInfo(stateMachine.gameState, move, options);

  // DEBUG: Trace stacks after applyMoveWithChainInfo for choose_line_option
  if (process.env.RINGRIFT_TRACE_DEBUG === '1' && move.type === 'choose_line_option') {
    // eslint-disable-next-line no-console
    console.log('[processTurn] after applyMoveWithChainInfo, stacks:', {
      stackCount: applyResult.nextState.board.stacks.size,
      stackKeys: Array.from(applyResult.nextState.board.stacks.keys()),
    });
  }

  stateMachine.updateGameState(applyResult.nextState);
  if (applyResult.pendingLineRewardElimination !== undefined) {
    stateMachine.updateFlags({
      eliminationRewardPending: applyResult.pendingLineRewardElimination,
    });
  }

  // DEBUG: Log mustMoveFromStackKey propagation
  if (process.env.RINGRIFT_TRACE_DEBUG === '1') {
    // eslint-disable-next-line no-console
    console.log('[processTurn] after updateGameState, mustMoveFromStackKey:', {
      moveType: move.type,
      fromApplyResult: applyResult.nextState.mustMoveFromStackKey,
      inStateMachine: stateMachine.gameState.mustMoveFromStackKey,
    });
  }

  // Compute hash after applying move
  const hashAfter = hashGameState(stateMachine.gameState);
  const moveActuallyChangedState = hashBefore !== hashAfter;

  // DEBUG: Trace line-phase move processing
  if (process.env.RINGRIFT_TRACE_DEBUG === '1' && move.type === 'choose_line_option') {
    // eslint-disable-next-line no-console
    console.log('[processTurn] LINE_PHASE_MOVE_DEBUG:', {
      moveType: move.type,
      hashBefore,
      hashAfter,
      moveActuallyChangedState,
      currentPhase: stateMachine.gameState.currentPhase,
      currentPlayer: stateMachine.gameState.currentPlayer,
    });
  }

  // If this was a capture move and chain continuation is required, set the chain capture state
  if (applyResult.chainCaptureRequired && applyResult.chainCapturePosition) {
    stateMachine.setChainCapture(true, applyResult.chainCapturePosition);
  }

  // For placement moves, don't process post-move phases - the player still needs to move
  // Post-move phases (lines, territory) only happen after movement/capture
  const isPlacementMove =
    move.type === 'place_ring' ||
    move.type === 'skip_placement' ||
    move.type === 'no_placement_action' ||
    move.type === 'swap_sides';

  // For decision moves (choose_territory_option, eliminate_rings_from_stack, etc.),
  // if the move didn't actually change state (e.g., Q23 prerequisite not met), don't
  // process post-move phases - just return the unchanged state.
  //
  // EXCEPTION: Line-phase decision moves (process_line, choose_line_option)
  // MUST always trigger processPostMovePhases because the phase transition logic needs to run
  // regardless of whether the collapse changed state. After choose_line_option, we need to
  // transition to territory_processing even if the line was already processed. This matches
  // Python's behavior which always checks remaining lines and advances phases after any
  // line-phase move (RR-PARITY-FIX-2025-12-13).
  const isLinePhaseMoveType = move.type === 'process_line' || move.type === 'choose_line_option';

  const isTerritoryDecisionMove =
    move.type === 'choose_territory_option' || move.type === 'eliminate_rings_from_stack';

  // Territory decision moves should only trigger post-move phases if they changed state
  const isDecisionMove = isLinePhaseMoveType || isTerritoryDecisionMove;

  // For turn-ending territory moves, the turn is complete - no post-move processing needed.
  // The applyMoveWithChainInfo handler already rotates to the next player.
  // RR-PARITY-FIX-2025-12-20: no_territory_action IS turn-ending now that applyMoveWithChainInfo
  // handles forced elimination check and turn rotation inline. Adding it here prevents
  // processPostMovePhases from incorrectly transitioning to line_processing after the
  // no_territory_action handler has already rotated to the next player's ring_placement.
  const isTurnEndingTerritoryMove =
    move.type === 'skip_territory_processing' || move.type === 'no_territory_action';

  // RR-PARITY-FIX-2025-12-20: Capture pendingDecision from applyMoveWithChainInfo.
  // For no_territory_action, the handler may return a forced elimination decision
  // which needs to be surfaced even when needsPostMoveProcessing is false.
  // RR-PARITY-FIX-2025-12-21: Also capture victoryResult from applyMoveWithChainInfo.
  // Turn-ending territory moves now check victory inline and return it directly.
  let result: { pendingDecision?: PendingDecision; victoryResult?: VictoryState } = {
    ...(applyResult.pendingDecision && { pendingDecision: applyResult.pendingDecision }),
    ...(applyResult.victoryResult && { victoryResult: applyResult.victoryResult }),
  };
  // Line-phase moves ALWAYS need post-move processing for phase transitions (RR-PARITY-FIX-2025-12-13)
  const needsPostMoveProcessing =
    !isPlacementMove &&
    !isTurnEndingTerritoryMove &&
    (moveActuallyChangedState || !isDecisionMove || isLinePhaseMoveType);

  // DEBUG: Trace post-move processing decision for line-phase moves
  if (process.env.RINGRIFT_TRACE_DEBUG === '1' && move.type === 'choose_line_option') {
    // eslint-disable-next-line no-console
    console.log('[processTurn] POST_MOVE_DECISION:', {
      isPlacementMove,
      isTurnEndingTerritoryMove,
      moveActuallyChangedState,
      isDecisionMove,
      isLinePhaseMoveType,
      needsPostMoveProcessing,
    });
  }

  if (needsPostMoveProcessing) {
    // Process post-move phases only for movement/capture moves, or for decision moves
    // that actually changed state, or for line-phase moves (always need phase transition check).
    // RR-FIX-2026-01-10: Preserve pendingDecision from applyMoveWithChainInfo if one exists.
    // For choose_territory_option with pendingSelfElimination, we must surface the elimination
    // decision rather than letting processPostMovePhases overwrite it.
    const existingPendingDecision = result.pendingDecision;
    const postMoveResult = processPostMovePhases(stateMachine, options);
    // Merge results, preserving existing pendingDecision if set (e.g., territory self-elimination)
    const mergedPendingDecision = existingPendingDecision ?? postMoveResult.pendingDecision;
    result = {
      ...postMoveResult,
      ...(mergedPendingDecision && { pendingDecision: mergedPendingDecision }),
    };

    // DEBUG: Trace stacks after processPostMovePhases for choose_line_option
    if (process.env.RINGRIFT_TRACE_DEBUG === '1' && move.type === 'choose_line_option') {
      // eslint-disable-next-line no-console
      console.log('[processTurn] after processPostMovePhases, stacks:', {
        stackCount: stateMachine.gameState.board.stacks.size,
        stackKeys: Array.from(stateMachine.gameState.board.stacks.keys()),
      });
    }
  }

  // Finalize metadata
  const sInvariantAfter = computeSInvariant(stateMachine.gameState);
  const metadata: ProcessingMetadata = {
    processedMove: move,
    phasesTraversed: stateMachine.processingState.phasesTraversed,
    linesDetected: stateMachine.processingState.pendingLines.length,
    regionsProcessed: 0,
    durationMs: Date.now() - startTime,
    sInvariantBefore,
    sInvariantAfter,
  };

  // Under trace-debug runs, log any strict S-invariant decreases that occur wholly
  // inside the orchestrator. This helps distinguish shared-engine bookkeeping
  // issues from host/adapter integration drift.
  const TRACE_DEBUG_ENABLED = flagEnabled('RINGRIFT_TRACE_DEBUG');

  if (
    TRACE_DEBUG_ENABLED &&
    state.gameStatus === 'active' &&
    stateMachine.gameState.gameStatus === 'active' &&
    sInvariantAfter < sInvariantBefore
  ) {
    const progressBefore = computeProgressSnapshot(state);
    const progressAfter = computeProgressSnapshot(stateMachine.gameState);

    debugLog(
      flagEnabled('RINGRIFT_TRACE_DEBUG'),
      '[turnOrchestrator.processTurn] STRICT_S_INVARIANT_DECREASE',
      {
        moveType: move.type,
        player: move.player,
        phaseBefore: state.currentPhase,
        phaseAfter: stateMachine.gameState.currentPhase,
        statusBefore: state.gameStatus,
        statusAfter: stateMachine.gameState.gameStatus,
        progressBefore,
        progressAfter,
        stateHashBefore: hashGameState(state),
        stateHashAfter: hashGameState(stateMachine.gameState),
      }
    );
  }

  let finalState = stateMachine.gameState;
  let finalPendingDecision = result.pendingDecision;
  let finalVictory = result.victoryResult;
  let finalStatus: ProcessTurnResult['status'] = finalPendingDecision
    ? 'awaiting_decision'
    : 'complete';
  let finalFsmDecisionSurface: FSMDecisionSurface | undefined;

  // Territory bookkeeping is fully explicit; do not surface forced elimination
  // immediately after territory decisions. The next player must start in
  // ring_placement and emit no_* actions as needed.
  const suppressForcedEliminationForTerritory =
    move.type === 'choose_territory_option' || move.type === 'eliminate_rings_from_stack';

  // If the turn is otherwise complete but the current player is blocked
  // with stacks and only a forced-elimination action is available, surface
  // that action as an explicit decision rather than applying a hidden
  // host-level tie-breaker. This implements RR-CANON-R206 for hosts that
  // drive decisions via PendingDecision/Move.
  //
  // Per the 7-phase model (RR-CANON-R070), forced_elimination is a distinct
  // phase entered only when the player had no actions in prior phases but
  // still controls stacks.
  if (
    finalStatus === 'complete' &&
    finalState.gameStatus === 'active' &&
    !suppressForcedEliminationForTerritory
  ) {
    // Only surface forced elimination after territory processing or when
    // already in the forced_elimination phase. This avoids jumping straight
    // to forced elimination from ring_placement when no placements exist; we
    // still need to traverse movement/line/territory phases per RR-CANON-R075.
    const canSurfaceForcedElimination =
      finalState.currentPhase === 'territory_processing' ||
      finalState.currentPhase === 'forced_elimination';

    if (canSurfaceForcedElimination) {
      const player = finalState.currentPlayer;
      const summary = computeGlobalLegalActionsSummary(finalState, player);

      if (
        summary.hasTurnMaterial &&
        summary.hasForcedEliminationAction &&
        !summary.hasGlobalPlacementAction &&
        !summary.hasPhaseLocalInteractiveMove
      ) {
        const forcedDecision = createForcedEliminationDecision(finalState);
        if (forcedDecision && forcedDecision.options.length > 0) {
          // Transition to forced_elimination phase (7th phase in the state machine)
          finalState = {
            ...finalState,
            currentPhase: 'forced_elimination' as GamePhase,
          };
          finalPendingDecision = forcedDecision;
          finalStatus = 'awaiting_decision';
        }
      }
    }
  }

  // Enforce INV-ACTIVE-NO-MOVES / INV-ANM-TURN-MATERIAL-SKIP at the orchestrator
  // boundary: no externally visible ACTIVE state may satisfy ANM(state) for the
  // current player. When such a state is detected and there is no pending
  // decision, resolve it immediately via forced elimination and/or victory
  // evaluation (RR-CANON R200–R205).
  if (finalStatus === 'complete' && finalState.gameStatus === 'active' && isANMState(finalState)) {
    const resolved = resolveANMForCurrentPlayer(finalState);
    finalState = resolved.nextState;
    if (resolved.victoryResult) {
      finalVictory = resolved.victoryResult;
    }
  }

  // Additional terminal guard: if all players have zero rings in hand and no
  // global interactive moves remain (placements, movement, captures), treat the
  // position as terminal rather than advancing to another ring_placement loop.
  // This mirrors Python's ANM/LPS termination observed in parity bundles.
  // FSM orchestrator: compute and apply FSM-derived phase/player transitions.
  // FSM is now the canonical orchestrator - always apply FSM results.
  // Phase 1 of FSM Integration: FSM also drives pending decisions.
  try {
    // Compute FSM orchestration result using the post-move state for FSM
    // transition. This keeps phase/player orchestration aligned with the
    // canonical TS/Python phase machines and avoids re-interpreting moves
    // against a stale pre-move snapshot.
    const fsmOrchResult = computeFSMOrchestration(finalState, move, {
      postMoveStateForChainCheck: finalState,
    });

    if (!fsmOrchResult.success) {
      fsmTraceLog('[FSM_ORCHESTRATOR] ERROR', {
        moveType: move.type,
        movePlayer: move.player,
        error: fsmOrchResult.error,
        gameId: state.id,
        moveNumber: state.moveHistory.length + 1,
      });
    } else {
      // Territory-phase moves (choose_territory_option, eliminate_rings_from_stack,
      // no_territory_action, skip_territory_processing)
      // have their post-move phase/player semantics driven by the shared orchestrator +
      // TurnStateMachine helpers (onLineProcessingComplete / onTerritoryProcessingComplete),
      // which are already aligned with Python's phase_machine. To avoid double-advancing
      // the turn and prematurely rotating out of territory_processing while
      // Python remains in territory_processing, we **do not** apply FSM
      // nextPhase/nextPlayer for these moves. FSM is still used for validation
      // and for non-territory phases.
      // RR-FIX-2025-12-13: Added eliminate_rings_from_stack to isTerritoryPhaseMove.
      // After eliminate_rings_from_stack, processPostMovePhases already advances to the
      // next player's ring_placement. Without this inclusion, the FSM result would
      // override that state, causing the game to get stuck in territory_processing.
      // RR-FIX-2025-12-20: Also exclude line-phase moves. After no_line_action, if
      // forced_elimination is triggered, processPostMovePhases already transitions
      // to forced_elimination. Without this exclusion, FSM would override with
      // territory_processing, causing PHASE_MOVE_INVARIANT errors.
      const isLinePhaseMove =
        move.type === 'no_line_action' ||
        move.type === 'process_line' ||
        move.type === 'choose_line_option';
      const isTerritoryPhaseMove =
        move.type === 'choose_territory_option' ||
        move.type === 'eliminate_rings_from_stack' ||
        move.type === 'no_territory_action' ||
        move.type === 'skip_territory_processing';
      const isPhaseHandledByProcessPostMovePhases = isLinePhaseMove || isTerritoryPhaseMove;

      // Handle game_over transition (e.g., from resign)
      if (fsmOrchResult.nextPhase === 'game_over') {
        finalState = {
          ...finalState,
          currentPhase: 'game_over',
          gameStatus: 'completed',
          // Winner would be computed based on resignation - the other player(s)
          // For 2-player games, winner is the non-resigning player
          winner:
            finalState.players.length === 2
              ? finalState.players.find(
                  (p: { playerNumber: number }) => p.playerNumber !== move.player
                )?.playerNumber
              : undefined,
        };
      } else if (!isPhaseHandledByProcessPostMovePhases) {
        // Apply FSM-derived state for moves NOT handled by processPostMovePhases.
        //
        // Line-phase moves (no_line_action, process_line, choose_line_option) and
        // territory-phase moves (choose_territory_option, eliminate_rings_from_stack,
        // no_territory_action,
        // skip_territory_processing) have their phase transitions handled by
        // processPostMovePhases. This includes transitions to forced_elimination
        // when the player had no actions this turn. Do NOT override their phases
        // with FSM results, as that would cause PHASE_MOVE_INVARIANT errors.
        //
        // Note: computeFSMOrchestration already resolves 'turn_end' to the
        // actual starting phase (ring_placement / movement) for the **next**
        // player, so we can safely apply it for other moves.
        finalState = {
          ...finalState,
          currentPhase: fsmOrchResult.nextPhase as GamePhase,
          currentPlayer: fsmOrchResult.nextPlayer,
        };
      }

      // Phase 1: Use FSM decision surface as the canonical source for pending decisions.
      // Derive PendingDecision from FSM orchestration result. Line-phase and territory-phase
      // moves continue to use the legacy orchestrator surface for region/no_territory
      // decisions to keep TS/Python parity with canonical history.
      const fsmDerivedDecision = !isPhaseHandledByProcessPostMovePhases
        ? derivePendingDecisionFromFSM(finalState, fsmOrchResult)
        : undefined;

      // Compare FSM-derived decision with legacy decision for debugging.
      // This comparison logging helps verify FSM decision parity during the transition.
      const legacyDecisionType = finalPendingDecision?.type;
      const fsmDecisionType = fsmDerivedDecision?.type;
      if (TRACE_DEBUG_ENABLED && legacyDecisionType !== fsmDecisionType) {
        fsmTraceLog('[FSM_ORCHESTRATOR] DECISION_DIVERGENCE', {
          moveType: move.type,
          movePlayer: move.player,
          legacyDecisionType: legacyDecisionType ?? 'none',
          fsmDecisionType: fsmDecisionType ?? 'none',
          fsmPendingDecisionType: fsmOrchResult.pendingDecisionType ?? 'none',
          gameId: state.id,
          moveNumber: state.moveHistory.length + 1,
        });
      }

      // Phase 1 Decision Strategy:
      // The FSM doesn't have access to board state, so it can't always determine
      // if regions/lines exist. Use FSM decision when it provides concrete
      // options (e.g., chain_capture, forced_elimination), but prefer legacy
      // for region/line decisions where FSM says "no action" but legacy found data.
      //
      // IMPORTANT: During the transition period, we only use FSM decisions when
      // they match or improve upon legacy decisions. We don't introduce NEW
      // decisions that legacy didn't surface (like no_*_action_required when
      // legacy had no decision), to maintain backward compatibility.
      //
      // Specifically:
      // - If FSM says region_order_required/line_order_required with data → use FSM
      // - If FSM says no_*_action_required but legacy found data → use legacy
      // - If FSM says no_*_action_required and legacy had no decision → keep legacy (no decision)
      // - For chain_capture/forced_elimination → always use FSM (board-aware)
      const shouldUseFsmDecision = (() => {
        if (!fsmDerivedDecision) return false;

        // FSM-driven decisions that don't need board context
        if (fsmDecisionType === 'chain_capture' || fsmDecisionType === 'elimination_target') {
          return true;
        }

        // For no-action decisions, only use FSM if legacy ALSO surfaced this decision.
        // Don't introduce new decisions that legacy didn't have.
        if (
          fsmDecisionType === 'no_line_action_required' ||
          fsmDecisionType === 'no_territory_action_required'
        ) {
          // If legacy found actual data (region_order, line_order), prefer legacy
          if (legacyDecisionType === 'region_order' || legacyDecisionType === 'line_order') {
            return false;
          }
          // If legacy also had the same no-action decision, use FSM (they agree)
          if (legacyDecisionType === fsmDecisionType) {
            return true;
          }
          // Legacy had no decision - don't introduce a new one
          return false;
        }

        // For data-bearing decisions (line_order, region_order), use FSM if it has options
        if (fsmDerivedDecision.options.length > 0) {
          return true;
        }

        // FSM has no options, prefer legacy if it has any
        return !finalPendingDecision || (finalPendingDecision.options?.length ?? 0) === 0;
      })();

      if (shouldUseFsmDecision && fsmDerivedDecision) {
        finalPendingDecision = fsmDerivedDecision;
        finalStatus = 'awaiting_decision';
      } else if (fsmOrchResult.pendingDecisionType && !fsmDerivedDecision) {
        // FSM indicated a decision is needed but we couldn't derive concrete options.
        // This can happen for empty decision surfaces. Log for debugging.
        if (TRACE_DEBUG_ENABLED) {
          fsmTraceLog('[FSM_ORCHESTRATOR] EMPTY_DECISION_SURFACE', {
            moveType: move.type,
            fsmPendingDecisionType: fsmOrchResult.pendingDecisionType,
            fsmPhase: fsmOrchResult.nextPhase,
            gameId: state.id,
          });
        }
      }

      // Phase 2: Build FSMDecisionSurface for the result.
      // This exposes the raw FSM orchestration data to hosts.
      const fsmSurface = fsmOrchResult.decisionSurface;
      if (fsmOrchResult.pendingDecisionType || fsmSurface) {
        const surface: FSMDecisionSurface = {};
        if (fsmOrchResult.pendingDecisionType) {
          surface.pendingDecisionType = fsmOrchResult.pendingDecisionType;
        }
        if (fsmSurface?.pendingLines) {
          surface.pendingLines = fsmSurface.pendingLines.map((l) => {
            const lineEntry: { positions: Position[]; player?: number } = {
              positions: l.positions,
            };
            if ('player' in l && typeof (l as { player: number }).player === 'number') {
              lineEntry.player = (l as { player: number }).player;
            }
            return lineEntry;
          });
        }
        if (fsmSurface?.pendingRegions) {
          surface.pendingRegions = fsmSurface.pendingRegions.map((r) => ({
            positions: r.positions,
            eliminationsRequired: r.eliminationsRequired,
          }));
        }
        if (fsmSurface?.chainContinuations) {
          surface.chainContinuations = fsmSurface.chainContinuations;
        }
        if (fsmSurface?.forcedEliminationCount !== undefined) {
          surface.forcedEliminationCount = fsmSurface.forcedEliminationCount;
        }
        finalFsmDecisionSurface = surface;
      }
    }
  } catch (err) {
    fsmTraceLog('[FSM_ORCHESTRATOR] EXCEPTION', {
      moveType: move.type,
      movePlayer: move.player,
      error: err instanceof Error ? err.message : String(err),
      gameId: state.id,
      moveNumber: state.moveHistory.length + 1,
    });
  }

  const processTurnResult: ProcessTurnResult = {
    nextState: finalState,
    status: finalPendingDecision ? 'awaiting_decision' : 'complete',
    pendingDecision: finalPendingDecision,
    victoryResult: finalVictory,
    metadata,
  };

  // Add FSM decision surface if available (Phase 2)
  if (finalFsmDecisionSurface) {
    processTurnResult.fsmDecisionSurface = finalFsmDecisionSurface;
  }

  return processTurnResult;
}

/**
 * Result of applying a move, including chain capture info for capture moves.
 */
interface ApplyMoveResult {
  nextState: GameState;
  chainCaptureRequired?: boolean;
  chainCapturePosition?: Position;
  /** Pending decision for forced elimination or other deferred actions */
  pendingDecision?: PendingDecision;
  /** Line reward elimination pending after line collapse (RR-CANON-R123) */
  pendingLineRewardElimination?: boolean;
  /** Victory result for turn-ending moves that detect game over inline */
  victoryResult?: VictoryState;
}

/**
 * Apply a move to the game state based on move type.
 * For capture moves, also returns chain capture continuation info.
 */
function applyMoveWithChainInfo(
  state: GameState,
  move: Move,
  options?: ProcessTurnOptions
): ApplyMoveResult {
  switch (move.type) {
    case 'swap_sides': {
      const validation = validateSwapSidesMove(state, move.player);
      if (!validation.valid) {
        throw new Error(validation.reason);
      }

      const swappedPlayers = applySwapSidesIdentitySwap(state.players);

      return {
        nextState: {
          ...state,
          players: swappedPlayers,
        },
      };
    }

    case 'place_ring': {
      const action = {
        type: 'PLACE_RING' as const,
        playerId: move.player,
        position: move.to,
        count: move.placementCount ?? 1,
      };
      const newState = mutatePlacement(state, action);
      // Set mustMoveFromStackKey to enforce that only the updated stack
      // can move/capture this turn. This matches Python's must_move_from_stack_key
      // semantics for TS/Python parity.
      const placementKey = move.to ? positionToString(move.to) : undefined;
      // DEBUG: Log mustMoveFromStackKey assignment
      if (process.env.RINGRIFT_TRACE_DEBUG === '1') {
        // eslint-disable-next-line no-console
        console.log(
          '[applyMoveWithChainInfo] place_ring setting mustMoveFromStackKey:',
          placementKey
        );
      }

      // RR-PARITY-FIX-2025-12-13: After place_ring, check if the player has any valid
      // movements or captures. If not, advance directly to line_processing (skipping
      // movement phase). This matches Python's phase_machine.py behavior after PLACE_RING
      // where it checks _has_valid_movements and _has_valid_captures and advances directly
      // to line_processing when neither is available.
      const stateAfterPlacement: GameState = {
        ...newState,
        mustMoveFromStackKey: placementKey,
      };
      const playerHasMovesOrCaptures = hasAnyGlobalMovementOrCapture(
        stateAfterPlacement,
        move.player
      );

      if (process.env.RINGRIFT_TRACE_DEBUG === '1') {
        // eslint-disable-next-line no-console
        console.log(
          '[applyMoveWithChainInfo] place_ring playerHasMovesOrCaptures:',
          playerHasMovesOrCaptures
        );
      }

      if (playerHasMovesOrCaptures) {
        // Normal flow: enter movement phase
        return {
          nextState: {
            ...stateAfterPlacement,
            currentPhase: 'movement' as GamePhase,
          },
        };
      } else {
        // No valid movements or captures: skip movement and go directly to line_processing
        // Per RR-CANON-R075, all phases must be visited. The AI/player must select
        // NO_MOVEMENT_ACTION which will be synthesized by the orchestrator when
        // entering line_processing with no movement available.
        return {
          nextState: {
            ...stateAfterPlacement,
            currentPhase: 'line_processing' as GamePhase,
            mustMoveFromStackKey: undefined, // Clear since we're skipping movement
          },
        };
      }
    }

    case 'skip_placement': {
      // Skip placement is a no-op on state, just phase transition.
      // Update currentPlayer from move.player to handle turn boundary cases.
      return {
        nextState: {
          ...state,
          currentPlayer: move.player,
          currentPhase: 'movement' as GamePhase,
        },
      };
    }

    case 'no_placement_action': {
      // Explicit forced no-op in ring_placement when the player has no legal
      // placement anywhere (RR-CANON-R075). State is unchanged; advance to
      // movement so that the rest of the turn can proceed.
      // Also update currentPlayer from move.player to handle turn boundary cases
      // where the move's player may differ from state.currentPlayer.
      return {
        nextState: {
          ...state,
          currentPlayer: move.player,
          currentPhase: 'movement' as GamePhase,
        },
      };
    }

    case 'skip_capture': {
      // RR-CANON-R070: Skip optional capture after movement, proceed to line processing
      return {
        nextState: {
          ...state,
          currentPhase: 'line_processing' as GamePhase,
        },
      };
    }

    case 'move_stack': {
      if (!move.from) {
        throw new Error('Move.from is required for movement moves');
      }
      const outcome = applySimpleMovement(state, {
        from: move.from,
        to: move.to,
        player: move.player,
      });
      // Update mustMoveFromStackKey to track the landing position when the
      // move originated from the must-move stack. This matches Python's
      // must_move_from_stack_key semantics for TS/Python parity.
      const fromKey = positionToString(move.from);
      let nextMustMove = outcome.nextState.mustMoveFromStackKey;
      if (state.mustMoveFromStackKey && fromKey === state.mustMoveFromStackKey && move.to) {
        nextMustMove = positionToString(move.to);
      }
      return {
        nextState: {
          ...outcome.nextState,
          mustMoveFromStackKey: nextMustMove,
        },
      };
    }

    case 'no_movement_action': {
      // Explicit forced no-op in movement phase when the player has no legal
      // movement or capture anywhere (RR-CANON-R075). State is unchanged;
      // post-move phase logic will advance to line_processing.
      // Update currentPlayer from move.player to handle turn boundary cases.
      return {
        nextState: {
          ...state,
          currentPlayer: move.player,
        },
      };
    }

    case 'skip_recovery': {
      // RR-CANON-R115: Recovery-eligible player explicitly declines recovery.
      // This is a no-op on the board; post-move phase logic will advance to
      // line_processing / territory_processing bookkeeping as appropriate.
      // Update currentPlayer from move.player to handle turn boundary cases.
      return {
        nextState: {
          ...state,
          currentPlayer: move.player,
        },
      };
    }

    case 'recovery_slide': {
      // RR-CANON-R110–R115: Recovery action for temporarily eliminated players
      // Validates that player is eligible and applies the slide + line collapse
      if (!move.from) {
        throw new Error('Move.from is required for recovery slide');
      }

      // Cast to extended type for accessing Python-style properties
      const extendedMove = move as Move & ExtendedMoveProperties;

      // Determine the option and mode from the move's metadata
      const collapsePos = extendedMove.collapsePositions || extendedMove.collapse_positions;
      const moveExtractionStacks = extendedMove.extractionStacks || extendedMove.extraction_stacks;
      const recoveryMove: RecoverySlideMove = {
        id: move.id,
        type: 'recovery_slide',
        player: move.player,
        from: move.from,
        to: move.to,
        timestamp: move.timestamp,
        thinkTime: move.thinkTime,
        moveNumber: move.moveNumber,
        option: extendedMove.recoveryOption || extendedMove.option || 1,
        ...(collapsePos && { collapsePositions: collapsePos }),
        ...(extendedMove.recoveryMode && { recoveryMode: extendedMove.recoveryMode }),
        extractionStacks: moveExtractionStacks || [], // Use provided stacks or empty
      };

      // For Option 1 or fallback, need to select an extraction stack if not already provided
      // Only auto-select if no extraction stacks were provided in the move
      if (
        recoveryMove.extractionStacks.length === 0 &&
        (recoveryMove.option === 1 ||
          recoveryMove.recoveryMode === 'fallback' ||
          recoveryMove.recoveryMode === 'stack_strike')
      ) {
        for (const [stackKey, stack] of state.board.stacks) {
          const hasBuriedRing = stack.rings.slice(1).includes(move.player); // rings[0] is top, check all except top
          if (hasBuriedRing) {
            recoveryMove.extractionStacks = [stackKey];
            break;
          }
        }
      }

      // Validate the recovery slide (after filling extractionStacks for Option 1)
      const validationResult = validateRecoverySlide(state, recoveryMove);
      if (!validationResult.valid) {
        throw new Error(`Invalid recovery slide: ${validationResult.reason}`);
      }

      // Apply the recovery slide
      const outcome = applyRecoverySlide(state, recoveryMove);
      return { nextState: outcome.nextState };
    }

    case 'overtaking_capture':
    case 'continue_capture_segment': {
      if (!move.from || !move.captureTarget) {
        throw new Error('Move.from and Move.captureTarget are required for capture moves');
      }
      const outcome = applyCaptureSegment(state, {
        from: move.from,
        target: move.captureTarget,
        landing: move.to,
        player: move.player,
      });
      // Update mustMoveFromStackKey to track the landing position when the
      // move originated from the must-move stack. This matches Python's
      // must_move_from_stack_key semantics for TS/Python parity.
      const fromKey = positionToString(move.from);
      let nextMustMove = outcome.nextState.mustMoveFromStackKey;
      if (state.mustMoveFromStackKey && fromKey === state.mustMoveFromStackKey && move.to) {
        nextMustMove = positionToString(move.to);
      }
      const updatedState = {
        ...outcome.nextState,
        mustMoveFromStackKey: nextMustMove,
      };
      // Return chain capture info so the orchestrator can set state machine flags
      if (outcome.chainContinuationRequired) {
        return {
          nextState: updatedState,
          chainCaptureRequired: true,
          chainCapturePosition: move.to,
        };
      }
      return { nextState: updatedState };
    }

    case 'process_line': {
      const outcome = applyProcessLineDecision(state, move);
      // RR-CANON-R123: Set pendingLineRewardElimination on GameState for ANM parity
      const nextStateWithFlag: GameState = {
        ...outcome.nextState,
        pendingLineRewardElimination: outcome.pendingLineRewardElimination,
      };
      return {
        nextState: nextStateWithFlag,
        pendingLineRewardElimination: outcome.pendingLineRewardElimination,
      };
    }

    case 'choose_line_option': {
      const outcome = applyChooseLineRewardDecision(state, move);
      // RR-CANON-R123: Set pendingLineRewardElimination on GameState for ANM parity
      const nextStateWithFlag: GameState = {
        ...outcome.nextState,
        pendingLineRewardElimination: outcome.pendingLineRewardElimination,
      };
      return {
        nextState: nextStateWithFlag,
        pendingLineRewardElimination: outcome.pendingLineRewardElimination,
      };
    }

    case 'no_line_action': {
      // Explicit no-op in line_processing phase when no lines exist for the
      // current player (RR-CANON-R075). State is unchanged; advance to
      // territory_processing so that the rest of the turn can proceed.
      // Update currentPlayer from move.player to handle turn boundary cases.
      return {
        nextState: {
          ...state,
          currentPlayer: move.player,
          currentPhase: 'territory_processing' as GamePhase,
        },
      };
    }

    case 'choose_territory_option': {
      const outcome = applyProcessTerritoryRegionDecision(state, move);

      // RR-CANON-R145: After processing a territory region, the player MUST
      // perform a mandatory self-elimination from a stack OUTSIDE the region.
      if (outcome.pendingSelfElimination) {
        const eliminationMoves = enumerateTerritoryEliminationMoves(
          outcome.nextState,
          move.player,
          { processedRegion: outcome.processedRegion, eliminationContext: 'territory' }
        );

        if (eliminationMoves.length > 0) {
          // Surface elimination decision - player must choose which stack to eliminate from
          const pendingDecision = createTerritoryEliminationDecision(
            outcome.nextState,
            move.player,
            eliminationMoves
          );
          return {
            nextState: {
              ...outcome.nextState,
              currentPhase: 'territory_processing' as GamePhase,
            },
            pendingDecision,
          };
        }
        // No eligible stacks - fall through (hand elimination handled elsewhere)
      }

      return { nextState: outcome.nextState };
    }

    case 'eliminate_rings_from_stack': {
      const outcome = applyEliminateRingsFromStackDecision(state, move);
      // RR-CANON-R123: Clear pendingLineRewardElimination when line elimination is applied
      // TypeScript doesn't narrow the Move type to include eliminationContext after type check,
      // so we access it via type assertion on the object with eliminationContext property.
      const elimContext = (
        move as { eliminationContext?: 'line' | 'territory' | 'forced' | 'recovery' }
      ).eliminationContext;
      if (elimContext === 'line' && state.pendingLineRewardElimination) {
        return {
          nextState: {
            ...outcome.nextState,
            pendingLineRewardElimination: false,
          },
        };
      }
      return { nextState: outcome.nextState };
    }

    case 'no_territory_action': {
      // Explicit no-op in territory_processing phase when no regions exist (RR-CANON-R075).
      // RR-PARITY-FIX-2025-12-20: Must complete phase transition inline for replay parity.
      // applyMoveForReplay does NOT call processPostMovePhases, so we must handle
      // forced elimination check and turn rotation here.
      //
      // Per RR-CANON-R070: If player had NO actions this turn AND still has stacks,
      // go to forced_elimination. Otherwise, rotate to next player.
      //
      // RR-REPLAY-COMPAT: If already in forced_elimination, skip the check and rotate.
      // This handles replay scenarios where processPostMovePhases transitioned to
      // forced_elimination, but the recorded sequence has no_territory_action next.
      const alreadyInForcedElimination = state.currentPhase === 'forced_elimination';

      if (!alreadyInForcedElimination) {
        // RR-FIX-2024-12-21: Pass turnSequenceRealMoves for accurate action detection
        // when processing bookkeeping moves in a multi-phase turn.
        const hadAnyAction = computeHadAnyActionThisTurn(
          state,
          move,
          options?.turnSequenceRealMoves
        );
        const hasStacks = playerHasStacksOnBoard(state, move.player);

        if (!hadAnyAction && hasStacks) {
          // Transition to forced_elimination and surface the pending decision
          const nextState = {
            ...state,
            currentPlayer: move.player,
            currentPhase: 'forced_elimination' as GamePhase,
          };
          const forcedDecision = createForcedEliminationDecision(nextState);
          if (forcedDecision && forcedDecision.options.length > 0) {
            return {
              nextState,
              pendingDecision: forcedDecision,
            };
          }
          // No valid forced elimination options - fall through to turn rotation
        }
      }

      // No forced elimination needed (or already was in FE) - check victory then rotate
      // RR-PARITY-FIX-2025-12-21: Check victory BEFORE rotating to next player.
      // Since processPostMovePhases is not called for turn-ending territory moves,
      // we must check victory inline to ensure territory victories are detected.
      const victoryResult = toVictoryState(state);
      if (victoryResult.isGameOver) {
        return {
          nextState: {
            ...state,
            gameStatus: 'completed',
            winner: victoryResult.winner,
            currentPhase: 'game_over' as GamePhase,
          },
          victoryResult,
        };
      }

      // No victory - rotate to next player
      const noTerritoryPlayers = state.players;
      const noTerritoryPlayerIndex = noTerritoryPlayers.findIndex(
        (p) => p.playerNumber === move.player
      );
      const { nextPlayer: noTerritoryNextPlayer } = computeNextNonEliminatedPlayer(
        state,
        noTerritoryPlayerIndex,
        noTerritoryPlayers
      );

      return {
        nextState: {
          ...state,
          currentPlayer: noTerritoryNextPlayer,
          currentPhase: 'ring_placement' as GamePhase,
          mustMoveFromStackKey: undefined,
          chainCapturePosition: undefined,
        },
      };
    }

    case 'skip_territory_processing': {
      // Explicit skip in territory_processing phase when player opts out of
      // processing available regions.
      //
      // IMPORTANT (RR-CANON-R070/R075/R206): This may still require transitioning
      // to forced_elimination before the turn can end. In particular, if the
      // player had no real actions this turn but still controls stacks, we must:
      //   1) enter forced_elimination, and
      //   2) surface an explicit forced_elimination decision.
      //
      // Defer any victory evaluation until after forced_elimination is resolved
      // via an explicit forced_elimination move.

      // RR-REPLAY-COMPAT: If already in forced_elimination, skip the check and rotate.
      const alreadyInForcedElimination = state.currentPhase === 'forced_elimination';

      if (!alreadyInForcedElimination) {
        // Mirror the no_territory_action forced-elimination gating.
        const hadAnyAction = computeHadAnyActionThisTurn(
          state,
          move,
          options?.turnSequenceRealMoves
        );
        const hasStacks = playerHasStacksOnBoard(state, move.player);

        if (!hadAnyAction && hasStacks) {
          // Transition to forced_elimination and surface the pending decision.
          const nextState = {
            ...state,
            currentPlayer: move.player,
            currentPhase: 'forced_elimination' as GamePhase,
          };
          const forcedDecision = createForcedEliminationDecision(nextState);
          if (forcedDecision && forcedDecision.options.length > 0) {
            return {
              nextState,
              pendingDecision: forcedDecision,
            };
          }
          // No valid forced elimination options - fall through to victory/rotation.
        }
      }

      // No forced elimination needed (or already was in FE) - check victory then rotate.
      const skipTerritoryVictory = toVictoryState(state);
      if (skipTerritoryVictory.isGameOver) {
        return {
          nextState: {
            ...state,
            gameStatus: 'completed',
            winner: skipTerritoryVictory.winner,
            currentPhase: 'game_over' as GamePhase,
          },
          victoryResult: skipTerritoryVictory,
        };
      }

      // RR-CANON-R201: Skip permanently eliminated players (no rings anywhere).
      // CRITICAL: NO PHASE SKIPPING - players with ringsInHand == 0 will emit
      // no_placement_action which transitions to movement, but they MUST enter
      // ring_placement first.
      const players = state.players;
      const currentPlayerIndex = players.findIndex((p) => p.playerNumber === move.player);
      const { nextPlayer } = computeNextNonEliminatedPlayer(state, currentPlayerIndex, players);

      return {
        nextState: {
          ...state,
          currentPlayer: nextPlayer,
          currentPhase: 'ring_placement' as GamePhase, // Always ring_placement - NO PHASE SKIPPING
          mustMoveFromStackKey: undefined, // Clear for new turn
          chainCapturePosition: undefined, // Clear for new turn
        },
      };
    }

    case 'forced_elimination': {
      // Per RR-CANON-R070, forced_elimination is the 7th phase. The player
      // must eliminate a ring from one of their stacks when they have no
      // other valid actions. Uses the same elimination logic as territory
      // processing but recorded as a distinct move type for replay clarity.
      const outcome = applyEliminateRingsFromStackDecision(state, move);
      return { nextState: outcome.nextState };
    }

    default: {
      // For unsupported move types, return state unchanged
      return { nextState: state };
    }
  }
}

/**
 * Enforce the canonical phase→MoveType mapping for ACTIVE states.
 *
 * For any call to processTurn, the incoming Move must be appropriate
 * for the currentPhase of the provided state:
 *
 * - ring_placement:
 *     place_ring, skip_placement, no_placement_action, swap_sides
 * - movement:
 *     move_stack, overtaking_capture, recovery_slide, skip_recovery, no_movement_action
 * - capture:
 *     overtaking_capture, skip_capture
 * - chain_capture:
 *     continue_capture_segment
 * - line_processing:
 *     process_line, choose_line_option, eliminate_rings_from_stack, no_line_action
 * - territory_processing:
 *     choose_territory_option, eliminate_rings_from_stack, skip_territory_processing,
 *     no_territory_action
 * - forced_elimination:
 *     forced_elimination
 *
 * swap_sides is permitted only in ring_placement (pie rule). Legacy move
 * types (see legacy/legacyMoveTypes.ts) are only accepted in replay
 * compatibility mode and must be treated as non-canonical by hosts.
 */
function assertPhaseMoveInvariant(state: GameState, move: Move): void {
  const phase = state.currentPhase;
  const type = move.type;

  // Delegate to FSMAdapter's canonical phase ↔ MoveType mapping. Legacy
  // compatibility is handled by replayCompatibility checks elsewhere.
  if (!isMoveTypeValidForPhase(phase, type)) {
    throw new Error(`[PHASE_MOVE_INVARIANT] Cannot apply move type '${type}' in phase '${phase}'`);
  }
}

/**
 * FSM validation result type.
 */
interface FSMValidationInternalResult {
  valid: boolean;
  currentPhase?: string | undefined;
  errorCode?: string | undefined;
  reason?: string | undefined;
}

/**
 * Structured FSM validation event for production monitoring.
 * Emits JSON logs that can be easily parsed by log aggregators.
 *
 * FSM is now the canonical validator (RR-CANON compliance).
 */
interface FSMValidationEvent {
  event: 'fsm_validation';
  timestamp: string;
  gameId: string;
  moveNumber: number;
  moveType: string;
  movePlayer: number;
  currentPhase: string;
  fsmValid: boolean;
  errorCode?: string | undefined;
  reason?: string | undefined;
  durationMs?: number | undefined;
}

/**
 * Emit a structured FSM validation event for monitoring.
 * In production, these can be aggregated to track:
 * - FSM rejection rate
 * - Validation performance
 */
function emitFSMValidationEvent(event: FSMValidationEvent): void {
  // Only emit if FSM logging is enabled
  if (!flagEnabled('RINGRIFT_FSM_STRUCTURED_LOGGING')) {
    return;
  }

  // Emit as JSON line for log aggregation
  // eslint-disable-next-line no-console
  console.log(JSON.stringify(event));
}

/**
 * Perform FSM validation on a move.
 *
 * FSM is now the canonical validator (RR-CANON compliance).
 * Returns the FSM validation result for the caller to enforce.
 *
 * @param state - The current game state
 * @param move - The move to validate
 * @returns FSM validation result
 */
function performFSMValidation(
  state: GameState,
  move: Move,
  replayCompatibility = false
): FSMValidationInternalResult {
  const startTime = Date.now();
  const moveNumber = state.moveHistory.length + 1;

  try {
    // Run FSM validation
    const fsmResult = validateMoveWithFSM(state, move, false, { replayCompatibility });

    // Emit structured event for monitoring
    emitFSMValidationEvent({
      event: 'fsm_validation',
      timestamp: new Date().toISOString(),
      gameId: state.id,
      moveNumber,
      moveType: move.type,
      movePlayer: move.player,
      currentPhase: state.currentPhase,
      fsmValid: fsmResult.valid,
      errorCode: fsmResult.errorCode,
      reason: fsmResult.reason,
      durationMs: Date.now() - startTime,
    });

    return {
      valid: fsmResult.valid,
      currentPhase: fsmResult.currentPhase,
      errorCode: fsmResult.errorCode,
      reason: fsmResult.reason,
    };
  } catch (error) {
    // Log FSM validation errors
    const errorMessage = error instanceof Error ? error.message : String(error);
    fsmTraceLog('[FSM_VALIDATION] ERROR', {
      moveType: move.type,
      currentPhase: state.currentPhase,
      error: errorMessage,
      gameId: state.id,
    });

    // Emit error event
    emitFSMValidationEvent({
      event: 'fsm_validation',
      timestamp: new Date().toISOString(),
      gameId: state.id,
      moveNumber,
      moveType: move.type,
      movePlayer: move.player,
      currentPhase: state.currentPhase,
      fsmValid: false,
      errorCode: 'FSM_ERROR',
      reason: errorMessage,
      durationMs: Date.now() - startTime,
    });

    // Treat errors as validation failures (FSM is canonical)
    return {
      valid: false,
      currentPhase: state.currentPhase,
      errorCode: 'FSM_ERROR',
      reason: errorMessage,
    };
  }
}

/**
 * Process post-move phases (lines, territory, victory check).
 *
 * @deprecated This function is being replaced by FSM-driven orchestration.
 * FSM (`computeFSMOrchestration`) now handles:
 * - Phase transitions per RR-CANON-R070/R071
 * - Decision timing via `pendingDecisionType`
 * - Player rotation
 *
 * This function is retained for backward compatibility during the FSM
 * integration transition. The decision derivation in `processTurn` now
 * uses `derivePendingDecisionFromFSM()` with fallback to this function.
 *
 * Future work: Once FSM orchestration handles all state mutations
 * (currently performed here via stateMachine.updateGameState/transitionTo),
 * this function can be removed.
 *
 * @see computeFSMOrchestration in FSMAdapter.ts
 * @see derivePendingDecisionFromFSM in this file
 */
function processPostMovePhases(
  stateMachine: ProcessingStateContainer,
  options?: ProcessTurnOptions
): {
  pendingDecision?: PendingDecision;
  victoryResult?: VictoryState;
} {
  const state = stateMachine.gameState;
  const originalMoveType = stateMachine.processingState.originalMove.type as string;
  // Handle forced_elimination phase completion - skip straight to victory check
  // and turn rotation since the elimination is the final action in the player's turn.
  // Per RR-CANON-R070, forced_elimination is the 7th and final phase.
  //
  // Territory-driven self-elimination (eliminate_rings_from_stack) is handled via
  // the territory_processing phase below and may still lead to forced_elimination
  // depending on hadAnyActionThisTurn and the player's remaining material.
  if (originalMoveType === 'forced_elimination') {
    // Check victory first
    const victoryResult = toVictoryState(stateMachine.gameState);
    if (victoryResult.isGameOver) {
      stateMachine.updateGameState({
        ...stateMachine.gameState,
        gameStatus: 'completed',
        winner: victoryResult.winner,
        currentPhase: 'game_over',
      });
      return { victoryResult };
    }

    // Rotate to next player, skipping permanently eliminated players per RR-CANON-R201.
    // A player is permanently eliminated if they have no rings anywhere (no controlled
    // stacks, no buried rings, no rings in hand). Such players are removed from turn
    // rotation entirely. Players with buried rings but no stacks/hand are NOT skipped -
    // they may be recovery-eligible and must traverse phases.
    const currentState = stateMachine.gameState;
    const players = currentState.players;
    const currentPlayerIndex = players.findIndex(
      (p: { playerNumber: number }) => p.playerNumber === currentState.currentPlayer
    );

    // Skip permanently eliminated players (RR-CANON-R201)
    const { nextPlayer } = computeNextNonEliminatedPlayer(
      currentState,
      currentPlayerIndex,
      players
    );

    // Per RR-CANON-R073: ALL players start in ring_placement without exception.
    // CRITICAL: NO PHASE SKIPPING - players with ringsInHand == 0 will emit
    // no_placement_action which transitions to movement, but they MUST enter
    // ring_placement first. This ensures:
    // - Consistent move history recording
    // - Replay parity between TS and Python engines
    // - Proper FSM state tracking
    // - LPS round tracking accuracy

    // Clear mustMoveFromStackKey and chainCapturePosition for new turn.
    stateMachine.updateGameState({
      ...currentState,
      currentPlayer: nextPlayer,
      currentPhase: 'ring_placement', // Always ring_placement - NO PHASE SKIPPING
      mustMoveFromStackKey: undefined, // Clear for new turn
      chainCapturePosition: undefined, // Clear for new turn
    });

    return {};
  }

  // Check for chain capture continuation first
  if (stateMachine.processingState.chainCaptureInProgress) {
    const pos = stateMachine.processingState.chainCapturePosition;
    if (pos) {
      const info = getChainCaptureContinuationInfo(state, state.currentPlayer, pos);
      if (info.mustContinue) {
        // Transition phase to chain_capture and set chainCapturePosition in state
        // so that RuleEngine/AIEngine can enumerate valid continuations.
        stateMachine.transitionTo('chain_capture');
        stateMachine.updateGameState({
          ...stateMachine.gameState,
          chainCapturePosition: pos,
        });
        return {
          pendingDecision: createChainCaptureDecision(
            stateMachine.gameState,
            info.availableContinuations
          ),
        };
      }
    }
    // Chain capture complete, clear chainCapturePosition and continue to line processing
    stateMachine.setChainCapture(false, undefined);
    stateMachine.updateGameState({
      ...stateMachine.gameState,
      chainCapturePosition: undefined,
    });
  }

  // RR-CANON-R070 / Section 4.3: After a non-capturing movement (move_stack),
  // the player may opt to make an overtaking capture before proceeding to line processing.
  // Check if the original move was a simple movement and if capture opportunities exist
  // FROM THE LANDING POSITION (not from all stacks).
  const wasSimpleMovement = originalMoveType === 'move_stack';
  const moveLandingPos = stateMachine.processingState.originalMove.to;
  if (wasSimpleMovement && state.currentPhase === 'movement' && moveLandingPos) {
    // Enumerate all captures and filter to only those from the landing position
    const allCaptures = enumerateAllCaptureMoves(
      stateMachine.gameState,
      stateMachine.gameState.currentPlayer
    );
    const landingKey = positionToString(moveLandingPos);
    const capturesFromLanding = allCaptures.filter(
      (m) => m.from && positionToString(m.from) === landingKey
    );
    if (capturesFromLanding.length > 0) {
      // Stay in capture phase to allow optional overtaking capture from the moved stack
      stateMachine.transitionTo('capture');
      return {};
    }
  }

  // Transition to line processing phase
  // Note: Also transition from 'ring_placement' when a movement move was made
  // (this happens when ringsInHand == 0 per RR-CANON-R204)
  // Also transition from 'chain_capture' when the capture chain has ended
  // (no more continuation captures available per RR-CANON-R073).
  if (
    state.currentPhase === 'movement' ||
    state.currentPhase === 'capture' ||
    state.currentPhase === 'chain_capture' ||
    state.currentPhase === 'ring_placement'
  ) {
    // RR-FIX-2025-12-12: Clear chainCapturePosition and mustMoveFromStackKey when
    // transitioning to line_processing. These values are only relevant during
    // movement/capture phases and must be cleared to prevent stale values from
    // persisting into later phases (line_processing, territory_processing).
    const needsClear =
      stateMachine.gameState.chainCapturePosition !== undefined ||
      stateMachine.gameState.mustMoveFromStackKey !== undefined;
    if (needsClear) {
      stateMachine.updateGameState({
        ...stateMachine.gameState,
        chainCapturePosition: undefined,
        mustMoveFromStackKey: undefined,
      });
    }
    stateMachine.transitionTo('line_processing');
  }

  const isLinePhaseMove =
    originalMoveType === 'no_line_action' ||
    originalMoveType === 'process_line' ||
    originalMoveType === 'choose_line_option';
  const shouldCheckLines = stateMachine.currentPhase === 'line_processing' || isLinePhaseMove;

  // Process lines
  if (shouldCheckLines) {
    // RR-PARITY-FIX-2024-12-09: After ANY move in line_processing (including
    // process_line), re-check for remaining lines. This mirrors Python's
    // phase_machine.py which always checks remaining_lines after each
    // process_line and stays in line_processing if more exist.
    // IMPORTANT: Use stateMachine.gameState (updated after move application),
    // not the stale `state` snapshot from function start.
    const updatedStateForLines = stateMachine.gameState;
    const allLines = findAllLines(updatedStateForLines.board);
    // RR-PARITY-FIX-2025-12-13: Filter lines by effective line length threshold.
    // For 2-player square8, the minimum line length is 4 (not 3). The findAllLines
    // function from LineAggregate uses the base config lineLength (3 for square8),
    // so we must additionally filter by the effective threshold here to match
    // Python's _get_line_processing_moves which uses getEffectiveLineLengthThreshold.
    const effectiveLineLength = getEffectiveLineLengthThreshold(
      updatedStateForLines.board.type,
      updatedStateForLines.players.length,
      updatedStateForLines.rulesOptions
    );
    const lines = allLines.filter(
      (l) => l.player === updatedStateForLines.currentPlayer && l.length >= effectiveLineLength
    );

    // DEBUG: Trace line detection in processPostMovePhases
    if (process.env.RINGRIFT_TRACE_DEBUG === '1' && originalMoveType === 'choose_line_option') {
      const player2Markers = Array.from(updatedStateForLines.board.markers.entries())
        .filter(([, m]) => m.player === 2)
        .map(([k]) => k);
      // eslint-disable-next-line no-console
      console.log('[processPostMovePhases] LINE_DETECTION:', {
        originalMoveType,
        currentPlayer: updatedStateForLines.currentPlayer,
        allLinesCount: allLines.length,
        playerLinesCount: lines.length,
        allLinesPlayers: allLines.map((l) => ({ player: l.player, len: l.length })),
        player2Markers,
        linePositions: lines.length > 0 ? lines[0].positions.map((p) => `${p.x},${p.y}`) : [],
      });
    }

    if (lines.length > 0) {
      // Core rules: never auto-generate or auto-apply process_line moves.
      // Surface a line_order decision so hosts can construct and apply
      // explicit process_line / choose_line_option moves that will be
      // recorded in canonical history (RR-CANON-R075/R076).
      if (stateMachine.currentPhase !== 'line_processing') {
        stateMachine.transitionTo('line_processing');
      }
      const detectedLines = lines.map((l) => ({
        positions: l.positions,
        player: l.player,
        length: l.length,
        direction: l.direction,
        collapseOptions: [],
      }));
      stateMachine.setPendingLines(detectedLines);
      return {
        pendingDecision: createLineOrderDecision(updatedStateForLines, detectedLines),
      };
    }

    // No lines exist for the current player. If the original move was not
    // already a line-phase move, surface a required no_line_action decision
    // so hosts can emit an explicit no_line_action bookkeeping move.
    if (!isLinePhaseMove) {
      return {
        pendingDecision: {
          type: 'no_line_action_required',
          player: updatedStateForLines.currentPlayer,
          options: [],
          context: {
            description: 'No lines to process - explicit no_line_action required per RR-CANON-R075',
          },
        },
      };
    }

    // RR-CANON-R123: After choose_line_option with 'eliminate' choice, the player
    // must execute a separate eliminate_rings_from_stack move. Stay in line_processing
    // and surface a decision for the elimination.
    const lineEliminationPending =
      stateMachine.processingState.perTurnFlags.eliminationRewardPending;
    if (
      (originalMoveType === 'process_line' || originalMoveType === 'choose_line_option') &&
      lineEliminationPending
    ) {
      // RR-FIX-2026-01-10: Enumerate actual elimination moves for the decision options.
      // Per RR-CANON-R022/R122: Line reward eliminations remove exactly ONE ring from
      // the top of any stack the player controls (not the full cap).
      const lineEliminationMoves: Move[] = [];
      const linePlayer = updatedStateForLines.currentPlayer;
      const nextMoveNumber = updatedStateForLines.moveHistory.length + 1;

      for (const [key, stack] of updatedStateForLines.board.stacks.entries()) {
        if (stack.controllingPlayer !== linePlayer) {
          continue;
        }
        const capHeight = stack.capHeight ?? 0;
        if (capHeight <= 0) {
          continue;
        }

        lineEliminationMoves.push({
          id: `eliminate-line-${key}`,
          type: 'eliminate_rings_from_stack',
          player: linePlayer,
          to: stack.position,
          eliminatedRings: [{ player: linePlayer, count: 1 }], // Line cost is always 1 ring
          eliminationContext: 'line',
          eliminationFromStack: {
            position: stack.position,
            capHeight,
            totalHeight: stack.stackHeight,
          },
          timestamp: new Date(),
          thinkTime: 0,
          moveNumber: nextMoveNumber,
        } as Move);
      }

      // Stay in line_processing, do not transition to territory_processing
      return {
        pendingDecision: {
          type: 'line_elimination_required',
          player: linePlayer,
          options: lineEliminationMoves,
          linePosition: stateMachine.processingState.originalMove.to,
          context: {
            description:
              'Line option "eliminate" chosen - explicit eliminate_rings_from_stack required per RR-CANON-R123',
          },
        },
      };
    }

    // Line-phase move was applied and no more lines; decide the next phase using
    // the canonical FSM helper so TS and Python share the same semantics.
    const player = updatedStateForLines.currentPlayer;
    const territoryEliminationContext = didCurrentTurnIncludeRecoverySlide(
      updatedStateForLines,
      player
    )
      ? 'recovery'
      : 'territory';
    const regionsForPlayer = getProcessableTerritoryRegions(updatedStateForLines.board, {
      player,
      eliminationContext: territoryEliminationContext,
    });
    const hasTerritoryRegions = regionsForPlayer.length > 0;
    // Pass currentMove for parity with Python where move is in history during phase checks
    // RR-FIX-2024-12-21: Also pass turnSequenceRealMoves for multi-phase turns where
    // the original move hasn't been added to history yet (host adds after processTurn returns).
    const hadAnyActionThisTurn = computeHadAnyActionThisTurn(
      stateMachine.gameState,
      stateMachine.processingState.originalMove,
      options?.turnSequenceRealMoves
    );
    const playerHasStacks = playerHasStacksOnBoard(stateMachine.gameState, player);

    const phaseAfterLine = onLineProcessingComplete(
      hasTerritoryRegions,
      hadAnyActionThisTurn,
      playerHasStacks
    );

    if (phaseAfterLine === 'forced_elimination') {
      // Player had no actions in any phase this turn but still controls stacks.
      // Enter forced_elimination and surface explicit forced_elimination options.
      stateMachine.transitionTo('forced_elimination');
      const forcedDecision = createForcedEliminationDecision(stateMachine.gameState);
      if (forcedDecision && forcedDecision.options.length > 0) {
        return {
          pendingDecision: forcedDecision,
        };
      }
      // If, unexpectedly, no forced_elimination options could be constructed,
      // fall through to global ANM/terminal handling below.
    } else {
      // RR-PARITY-FIX-2024-12-09: Per RR-CANON-R075, every phase must be visited
      // and produce a recorded action. Always transition to territory_processing
      // regardless of whether there are territory regions. This matches Python's
      // _on_line_processing_complete which always advances to territory_processing.
      // The territory_processing block below will surface no_territory_action_required
      // when there are no regions, ensuring an explicit move is recorded.
      // This fixes parity divergence where TS would skip directly to turn rotation
      // while Python stays in territory_processing waiting for no_territory_action.
      stateMachine.transitionTo('territory_processing');
      // RR-FIX-2025-12-12: Clear stale chain capture state when entering territory_processing.
      // These values should have been cleared when the chain capture ended, but
      // in some edge cases (e.g., fixture loading, async state sync) they may persist.
      // Territory processing never needs these values.
      if (
        stateMachine.gameState.chainCapturePosition !== undefined ||
        stateMachine.gameState.mustMoveFromStackKey !== undefined
      ) {
        stateMachine.updateGameState({
          ...stateMachine.gameState,
          chainCapturePosition: undefined,
          mustMoveFromStackKey: undefined,
        });
      }
    }
  }

  // Process territory
  if (stateMachine.currentPhase === 'territory_processing') {
    // If the original move was an explicit skip_territory_processing,
    // treat territory processing as complete for this turn and proceed
    // directly to victory/turn advancement.
    if (originalMoveType !== 'skip_territory_processing') {
      // RR-PARITY-FIX-2024-12-09: After ANY move in territory_processing (including
      // choose_territory_option), re-check for remaining regions. This mirrors
      // Python's phase_machine.py which always checks remaining_regions after each
      // choose_territory_option and stays in territory_processing if more exist.
      // IMPORTANT: Use stateMachine.gameState (updated after move application),
      // not the stale `state` snapshot from function start.
      const updatedState = stateMachine.gameState;
      const territoryEliminationContext = didCurrentTurnIncludeRecoverySlide(
        updatedState,
        updatedState.currentPlayer
      )
        ? 'recovery'
        : 'territory';

      // Mandatory self-elimination after processing a region (RR-CANON-R145 / RR-CANON-R114).
      //
      // Important host/orchestrator detail: backend hosts append the applied
      // Move into moveHistory *after* processTurn returns. That means helper
      // functions that key off "last move in moveHistory" (like
      // enumerateTerritoryEliminationMoves) must be driven by the current
      // move type, not by the previous moveHistory tail.
      //
      // Concretely: we only surface elimination_target immediately after a
      // territory region decision move (choose_territory_option / legacy
      // choose_territory_option). After an eliminate_rings_from_stack move, the
      // self-elimination requirement has been satisfied and we must not re-open
      // an elimination_target decision based on the stale moveHistory tail.
      const shouldCheckEliminationAfterRegion = originalMoveType === 'choose_territory_option';
      if (shouldCheckEliminationAfterRegion) {
        // If this turn is in recovery context, the cost is a buried-ring extraction.
        const elimScope: TerritoryEliminationScope = {
          eliminationContext: territoryEliminationContext,
        };
        const processedRegion = stateMachine.processingState.originalMove.disconnectedRegions?.[0];
        if (processedRegion) {
          elimScope.processedRegion = processedRegion;
        }

        const eliminationMoves = enumerateTerritoryEliminationMoves(
          updatedState,
          updatedState.currentPlayer,
          elimScope
        );
        if (eliminationMoves.length > 0) {
          return {
            pendingDecision: {
              type: 'elimination_target',
              player: updatedState.currentPlayer,
              eliminationReason: 'territory_disconnection',
              options: eliminationMoves,
              context: {
                description:
                  territoryEliminationContext === 'recovery'
                    ? 'Choose which stack to extract a buried ring from (territory via recovery)'
                    : 'Choose which stack to self-eliminate from (territory disconnection)',
                relevantPositions: eliminationMoves
                  .map((m) => m.to)
                  .filter((p): p is Position => !!p),
                extra: {
                  reason: 'territory_disconnection',
                  eliminationContext: territoryEliminationContext,
                },
              },
            },
          };
        }
      }

      const regions = getProcessableTerritoryRegions(updatedState.board, {
        player: updatedState.currentPlayer,
        eliminationContext: territoryEliminationContext,
      });

      if (regions.length > 1) {
        // Multiple regions: player must choose order via explicit
        // choose_territory_option moves constructed by the host.
        return {
          pendingDecision: createRegionOrderDecision(updatedState, regions),
        };
      } else if (regions.length === 1) {
        // Single region: per RR-CANON-R075/R076, the core rules layer does
        // not auto-apply CHOOSE_TERRITORY_OPTION. Surface a region_order
        // decision even when there is only one region; hosts may auto-select
        // the only option for live UX but must still emit the explicit move.
        return {
          pendingDecision: createRegionOrderDecision(updatedState, regions),
        };
      } else {
        // regions.length === 0: No regions to process.
        // Per RR-CANON-R075/R076, return a pending decision requiring an explicit
        // no_territory_action move. The core rules layer does NOT auto-generate moves.
        // EXCEPTION: If the original move was already a territory-related move
        // (no_territory_action, choose_territory_option, or eliminate_rings_from_stack),
        // we don't need to return another pending decision - that move IS the territory phase action.
        const isTerritoryPhaseMove =
          originalMoveType === 'no_territory_action' ||
          originalMoveType === 'choose_territory_option' ||
          originalMoveType === 'eliminate_rings_from_stack';
        if (!isTerritoryPhaseMove) {
          return {
            pendingDecision: {
              type: 'no_territory_action_required',
              player: updatedState.currentPlayer,
              options: [],
              context: {
                description:
                  'No territory regions to process - explicit no_territory_action required per RR-CANON-R075',
              },
            },
          };
        }
        // Territory phase move was applied and no more regions, proceed to victory/turn advancement.
      }
    }

    // Territory processing is complete for this player (either because there were
    // no regions, all regions have been resolved, or the player explicitly
    // skipped). Decide whether to enter forced_elimination or end the turn using
    // the canonical FSM helper so TS and Python stay aligned.
    const currentPlayer = stateMachine.gameState.currentPlayer;
    // Pass currentMove for parity with Python where move is in history during phase checks
    // RR-FIX-2024-12-21: Also pass turnSequenceRealMoves for multi-phase turns where
    // the original move hasn't been added to history yet (host adds after processTurn returns).
    const hadAnyActionThisTurn = computeHadAnyActionThisTurn(
      stateMachine.gameState,
      stateMachine.processingState.originalMove,
      options?.turnSequenceRealMoves
    );
    const playerHasStacks = playerHasStacksOnBoard(stateMachine.gameState, currentPlayer);

    const phaseAfterTerritory = onTerritoryProcessingComplete(
      hadAnyActionThisTurn,
      playerHasStacks
    );

    if (phaseAfterTerritory === 'forced_elimination') {
      // Player had no actions in any phase this turn but still controls stacks.
      // RR-PARITY-FIX-2025-12-20: ALWAYS transition when phase logic dictates
      // forced_elimination. Previously the transition was inside the summary
      // condition, causing phase invariant violations when checks failed.
      stateMachine.transitionTo('forced_elimination');

      // Only surface forced_elimination decision when it is the sole interactive
      // option, matching the ANM/FE invariants used by Python.
      const summary = computeGlobalLegalActionsSummary(stateMachine.gameState, currentPlayer);
      if (
        summary.hasTurnMaterial &&
        summary.hasForcedEliminationAction &&
        !summary.hasGlobalPlacementAction &&
        !summary.hasPhaseLocalInteractiveMove
      ) {
        const forcedDecision = createForcedEliminationDecision(stateMachine.gameState);
        if (forcedDecision && forcedDecision.options.length > 0) {
          return {
            pendingDecision: forcedDecision,
          };
        }
        // If no concrete forced_elimination options could be constructed,
        // fall through to the turn rotation logic below. This ensures that
        // multi-player games (3-4 players) correctly rotate to the next player
        // even when FE was computed but no valid options exist.
        // RR-PARITY-FIX-2025-12-10: Fixes 4-player no_territory_action turn
        // rotation where TS would remain on the same player while Python rotated.
      }
      // Fallback: if summary indicates that forced_elimination is not actually
      // available, treat this as a normal turn-end and fall through to the
      // generic rotation logic below. We're now correctly in forced_elimination phase.
    }

    // phaseAfterTerritory === 'turn_end' – either the player took at least one
    // real action this turn or they have no stacks left.
    //
    // Before rotating to the next player, check for victory conditions.
    // This is critical for LPS (last_player_standing) detection after
    // eliminate_rings_from_stack removes the last stack for a player.
    const territoryVictoryResult = toVictoryState(stateMachine.gameState);
    if (territoryVictoryResult.isGameOver) {
      stateMachine.updateGameState({
        ...stateMachine.gameState,
        gameStatus: 'completed',
        winner: territoryVictoryResult.winner,
        currentPhase: 'game_over',
      });
      return { victoryResult: territoryVictoryResult };
    }

    // Advance explicitly to the next player's starting phase.
    // Per RR-CANON-R073: ring_placement if ringsInHand > 0, else movement.
    // Clear mustMoveFromStackKey as it only applies within a single turn.
    // RR-CANON-R201: Skip permanently eliminated players (no rings anywhere).
    const currentState = stateMachine.gameState;
    const players = currentState.players;
    const currentPlayerIndex = players.findIndex(
      (p: { playerNumber: number }) => p.playerNumber === currentState.currentPlayer
    );
    const { nextPlayer } = computeNextNonEliminatedPlayer(
      currentState,
      currentPlayerIndex,
      players
    );
    // Per RR-CANON-R073: ALL players start in ring_placement without exception.
    // NO PHASE SKIPPING - players with ringsInHand == 0 will emit no_placement_action.

    stateMachine.updateGameState({
      ...currentState,
      currentPlayer: nextPlayer,
      currentPhase: 'ring_placement', // Always ring_placement - NO PHASE SKIPPING
      mustMoveFromStackKey: undefined, // Clear for new turn
      chainCapturePosition: undefined, // Clear for new turn
    });
    return {};
  }

  // Check victory
  const victoryResult = toVictoryState(stateMachine.gameState);
  if (victoryResult.isGameOver) {
    stateMachine.updateGameState({
      ...stateMachine.gameState,
      gameStatus: 'completed',
      winner: victoryResult.winner,
      currentPhase: 'game_over',
    });
    return { victoryResult };
  }

  // All phases complete - advance to next player's turn
  // RR-CANON-R201: Skip permanently eliminated players (no rings anywhere).
  const currentState = stateMachine.gameState;
  const players = currentState.players;
  const currentPlayerIndex = players.findIndex(
    (p: { playerNumber: number }) => p.playerNumber === currentState.currentPlayer
  );
  const { nextPlayer } = computeNextNonEliminatedPlayer(currentState, currentPlayerIndex, players);
  // Per RR-CANON-R073: ALL players start in ring_placement without exception.
  // NO PHASE SKIPPING - players with ringsInHand == 0 will emit no_placement_action.
  // When no legal placements exist (ringsInHand == 0), hosts must emit an
  // explicit no_placement_action bookkeeping move; the core orchestrator no longer
  // fabricates this move itself.
  // Clear mustMoveFromStackKey for new turn.

  stateMachine.updateGameState({
    ...currentState,
    currentPlayer: nextPlayer,
    currentPhase: 'ring_placement', // Always ring_placement - NO PHASE SKIPPING
    mustMoveFromStackKey: undefined, // Clear for new turn
    chainCapturePosition: undefined, // Clear for new turn
  });

  return {};
}

function didCurrentTurnIncludeRecoverySlide(state: GameState, player: number): boolean {
  for (let i = state.moveHistory.length - 1; i >= 0; i--) {
    const move = state.moveHistory[i];
    if (move.player !== player) {
      break;
    }
    if (move.type === 'recovery_slide') {
      return true;
    }
  }
  return false;
}

// ═══════════════════════════════════════════════════════════════════════════
// Process Turn Async (For Human Decisions)
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Process a turn with async decision resolution.
 *
 * When a decision is required, the delegates.resolveDecision function
 * is called to get the player's choice.
 */
export async function processTurnAsync(
  state: GameState,
  move: Move,
  delegates: TurnProcessingDelegates,
  options?: ProcessTurnOptions
): Promise<ProcessTurnResult> {
  // RR-PARITY-FIX-2025-12-21: Track non-bookkeeping moves for this turn sequence.
  // This is needed because move history isn't updated until the host adds them
  // after processTurnAsync returns. When processing intermediate bookkeeping moves,
  // computeHadAnyActionThisTurn needs to know about the original real move.
  const turnSequenceRealMoves: Move[] = options?.turnSequenceRealMoves
    ? [...options.turnSequenceRealMoves]
    : [];

  // Track the initial move if it's a real action (not bookkeeping)
  if (!isNoActionBookkeepingMove(move.type)) {
    turnSequenceRealMoves.push(move);
  }

  const optionsWithRealMoves: ProcessTurnOptions = {
    ...options,
    turnSequenceRealMoves,
  };

  let result = processTurn(state, move, optionsWithRealMoves);

  // Process any pending decisions
  while (result.status === 'awaiting_decision' && result.pendingDecision) {
    const decision = result.pendingDecision;

    // Chain capture decisions should NOT be auto-resolved via delegates.
    // They must be handled via explicit continue_capture_segment moves
    // submitted through the normal makeMove() flow. Return with the pending
    // decision so GameEngine can set up chainCaptureState and allow the
    // caller to select a continuation move.
    if (decision.type === 'chain_capture') {
      return result;
    }

    // In replay mode (breakOnDecisionRequired), we stop here and return the
    // pending decision. The caller (replay driver) is responsible for
    // supplying the next move from the recording, which will resolve this
    // decision.
    if (options?.breakOnDecisionRequired) {
      return result;
    }

    // Emit decision event if handler provided
    delegates.onProcessingEvent?.({
      type: 'decision_required',
      timestamp: new Date(),
      payload: { decision },
    });

    // RR-FIX-2026-01-18: Attach the intermediate state to the decision so that
    // decision handlers can broadcast it for UI updates (e.g., showing the board
    // with the triggering move applied while the player is choosing territory).
    decision.intermediateState = result.nextState;

    // Resolve the decision
    const chosenMove = await delegates.resolveDecision(decision);

    // Emit decision resolved event
    delegates.onProcessingEvent?.({
      type: 'decision_resolved',
      timestamp: new Date(),
      payload: { decision, chosenMove },
    });

    // RR-PARITY-FIX-2025-12-21: Track the chosen move if it's a real action
    if (!isNoActionBookkeepingMove(chosenMove.type)) {
      turnSequenceRealMoves.push(chosenMove);
    }

    // Continue processing with the chosen move, passing accumulated real moves
    result = processTurn(result.nextState, chosenMove, {
      ...optionsWithRealMoves,
      turnSequenceRealMoves,
    });
  }

  return result;
}

// ═══════════════════════════════════════════════════════════════════════════
// Utility Functions
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Check if a move is valid for the current game state.
 *
 * @deprecated Use `validateMoveWithFSM` from `../fsm` instead. This legacy
 * validator is maintained for backward compatibility but FSM validation is
 * now the canonical validator used by `processTurn`. FSM validation provides:
 * - Phase-aware validation per RR-CANON-R070/R075
 * - Consistent behavior between TypeScript and Python
 * - Better error messages with FSM state context
 *
 * This function may be removed in a future release.
 */
export function validateMove(state: GameState, move: Move): { valid: boolean; reason?: string } {
  if (isLegacyMoveType(move.type)) {
    return {
      valid: false,
      reason:
        'Legacy move types are not valid for canonical validation. Use legacy replay adapters.',
    };
  }

  switch (move.type) {
    case 'place_ring': {
      const action = {
        type: 'PLACE_RING' as const,
        playerId: move.player,
        position: move.to,
        count: move.placementCount ?? 1,
      };
      return validatePlacement(state, action);
    }

    case 'skip_placement': {
      const result = evaluateSkipPlacementEligibility(state, move.player);
      if (!result.eligible && result.reason) {
        return { valid: false, reason: result.reason };
      }
      return { valid: result.eligible };
    }

    case 'no_placement_action': {
      // Forced no-op in ring_placement when the player has no legal
      // placement anywhere. Hosts should only emit this in GamePhase
      // 'ring_placement'; the engine treats it as always valid in
      // canonical recordings.
      return { valid: true };
    }

    case 'move_stack': {
      if (!move.from) {
        return { valid: false, reason: 'Move.from is required' };
      }
      const action = {
        type: 'MOVE_STACK' as const,
        playerId: move.player,
        from: move.from,
        to: move.to,
      };
      return validateMovement(state, action);
    }

    case 'overtaking_capture':
    case 'continue_capture_segment': {
      if (!move.from || !move.captureTarget) {
        return { valid: false, reason: 'Move.from and Move.captureTarget are required' };
      }
      const action = {
        type: 'OVERTAKING_CAPTURE' as const,
        playerId: move.player,
        from: move.from,
        captureTarget: move.captureTarget,
        to: move.to,
      };
      return validateCapture(state, action);
    }

    case 'no_movement_action': {
      // Forced no-op in movement phase when the player has no legal
      // movement or capture anywhere. Hosts are responsible for only
      // emitting this in GamePhase 'movement'; the engine treats it
      // as always valid for canonical recordings.
      return { valid: true };
    }

    default:
      return { valid: true };
  }
}

/**
 * Get all valid moves for the current player and phase.
 *
 * Per RR-CANON-R076, the core rules layer MUST NOT auto-generate moves,
 * including no-action bookkeeping moves. This helper therefore returns
 * only **interactive** moves:
 *
 * - ring_placement: place_ring, skip_placement (when eligible).
 * - movement: move_stack, overtaking_capture, recovery_slide, skip_recovery.
 * - capture: overtaking_capture, skip_capture.
 * - chain_capture: continue_capture_segment.
 * - line_processing: process_line / choose_line_option.
 * - territory_processing: choose_territory_option /
 *   eliminate_rings_from_stack (+ skip_territory_processing).
 * - forced_elimination: forced_elimination options.
 *
 * When a phase has no interactive moves, this function returns an empty
 * array. Hosts are responsible for:
 *
 * - Detecting no-move situations, and
 * - Constructing explicit no_*_action moves (or forced_elimination moves)
 *   via the public API so that every visited phase is still recorded in
 *   canonical history (RR-CANON-R075/R076).
 */
export function getValidMoves(state: GameState): Move[] {
  const player = state.currentPlayer;
  const phase = state.currentPhase;
  const moveNumber = state.moveHistory.length + 1;

  switch (phase) {
    case 'ring_placement': {
      const playerObj = state.players.find((p) => p.playerNumber === player);

      // No rings in hand → placement is forbidden for this player.
      // RR-FIX-2026-01-10: Return no_placement_action as a valid move so that
      // hosts can select it to advance to movement phase. This fixes the
      // ACTIVE_NO_MOVES soak test invariant violation.
      // Per RR-CANON-R073, all players MUST start in ring_placement, but if
      // they have 0 rings, they MUST emit no_placement_action to advance.
      if (!playerObj || playerObj.ringsInHand === 0) {
        return [
          {
            id: `no-placement-action-${moveNumber}`,
            type: 'no_placement_action',
            player,
            to: { x: 0, y: 0 }, // Sentinel value for bookkeeping move
            timestamp: new Date(),
            thinkTime: 0,
            moveNumber,
          } as Move,
        ];
      }

      // Interactive placement and skip_placement options only.
      const positions = enumeratePlacementPositions(state, player);
      const moves: Move[] = positions.map((pos) => ({
        id: `place-${positionToString(pos)}-${moveNumber}`,
        type: 'place_ring',
        player,
        to: pos,
        timestamp: new Date(),
        thinkTime: 0,
        moveNumber,
      }));

      // Add skip placement if eligible
      const skipResult = evaluateSkipPlacementEligibility(state, player);
      if (skipResult.eligible) {
        moves.push({
          id: `skip-placement-${moveNumber}`,
          type: 'skip_placement',
          player,
          to: { x: 0, y: 0 },
          timestamp: new Date(),
          thinkTime: 0,
          moveNumber,
        });
      }

      // RR-FIX-2026-01-12: When a player has rings in hand but cannot place them
      // anywhere (all positions blocked or violate no-dead-placement) AND cannot
      // skip to movement (no stacks to move), return no_placement_action to allow
      // the turn to advance. Without this, the game stalls with an empty valid
      // moves array. This can happen when the board is heavily collapsed and the
      // player has no stacks while another player still does.
      if (moves.length === 0) {
        return [
          {
            id: `no-placement-action-blocked-${moveNumber}`,
            type: 'no_placement_action',
            player,
            to: { x: 0, y: 0 }, // Sentinel value for bookkeeping move
            timestamp: new Date(),
            thinkTime: 0,
            moveNumber,
          } as Move,
        ];
      }

      return moves;
    }

    case 'movement': {
      let movements = enumerateSimpleMovesForPlayer(state, player);
      let captures = enumerateAllCaptureMoves(state, player);

      // When mustMoveFromStackKey is set (after place_ring), only moves from
      // that stack are valid. This matches Python's must_move_from_stack_key
      // semantics for TS/Python parity.
      if (state.mustMoveFromStackKey) {
        // DEBUG: Log filtering
        if (process.env.RINGRIFT_TRACE_DEBUG === '1') {
          // eslint-disable-next-line no-console
          console.log('[getValidMoves] filtering by mustMoveFromStackKey:', {
            mustMoveFromStackKey: state.mustMoveFromStackKey,
            movementsBefore: movements.length,
            capturesBefore: captures.length,
          });
        }
        movements = movements.filter(
          (m) => m.from && positionToString(m.from) === state.mustMoveFromStackKey
        );
        captures = captures.filter(
          (m) => m.from && positionToString(m.from) === state.mustMoveFromStackKey
        );
        // DEBUG: Log after filtering
        if (process.env.RINGRIFT_TRACE_DEBUG === '1') {
          // eslint-disable-next-line no-console
          console.log(
            '[getValidMoves] after filtering:',
            JSON.stringify({
              movementsAfter: movements.length,
              capturesAfter: captures.length,
              captureDetails: captures.map((c) => ({
                from: c.from,
                captureTarget: c.captureTarget,
                to: c.to,
                type: c.type,
              })),
            })
          );
        }
      }

      // RR-CANON-R110–R115: Recovery action for temporarily eliminated players
      // If player is eligible for recovery, add recovery slide moves
      const recoveryMoves: Move[] = [];
      if (isEligibleForRecovery(state, player)) {
        // RR-CANON-R115: recovery-eligible players may explicitly skip recovery.
        recoveryMoves.push({
          id: `skip-recovery-${moveNumber}`,
          type: 'skip_recovery',
          player,
          to: { x: 0, y: 0 }, // Sentinel
          timestamp: new Date(),
          thinkTime: 0,
          moveNumber,
        });

        const recoveryTargets = enumerateExpandedRecoverySlideTargets(state, player);
        const lineLength = getEffectiveLineLengthThreshold(state.board.type, state.players.length);
        for (const target of recoveryTargets) {
          if (target.recoveryMode === 'fallback') {
            // RR-CANON-R112(b): fallback repositioning when no line recovery exists.
            recoveryMoves.push({
              id: `recovery-fallback-${target.from.x},${target.from.y}-${target.to.x},${target.to.y}-${moveNumber}`,
              type: 'recovery_slide',
              player,
              from: target.from,
              to: target.to,
              timestamp: new Date(),
              thinkTime: 0,
              moveNumber,
              recoveryMode: 'fallback',
              extractionStacks: [],
            } as Move);
            continue;
          }
          if (target.recoveryMode === 'stack_strike') {
            // Experimental (v1) fallback-class recovery: marker sacrifice to strike adjacent stack.
            recoveryMoves.push({
              id: `recovery-stack-strike-${target.from.x},${target.from.y}-${target.to.x},${target.to.y}-${moveNumber}`,
              type: 'recovery_slide',
              player,
              from: target.from,
              to: target.to,
              timestamp: new Date(),
              thinkTime: 0,
              moveNumber,
              recoveryMode: 'stack_strike',
              extractionStacks: [],
            } as Move);
            continue;
          }

          // For each target, generate moves for available options
          // Option 1 (collapse all, cost 1) - requires buried rings (eligibility already ensured)
          // Option 2 (collapse lineLength, cost 0) - only for overlength lines
          const baseId = `recovery-${target.from.x},${target.from.y}-${target.to.x},${target.to.y}-${moveNumber}`;
          if (target.isOverlength) {
            // For overlength, offer both options
            // Option 2 (free)
            recoveryMoves.push({
              id: `${baseId}-opt2`,
              type: 'recovery_slide',
              player,
              from: target.from,
              to: target.to,
              timestamp: new Date(),
              thinkTime: 0,
              moveNumber,
              option: 2,
              recoveryMode: 'line',
              // Default collapse subset: first lineLength positions from the formed line
              collapsePositions: target.linePositions.slice(0, lineLength),
              extractionStacks: [],
            } as Move);

            // Option 1 (cost 1)
            recoveryMoves.push({
              id: `${baseId}-opt1`,
              type: 'recovery_slide',
              player,
              from: target.from,
              to: target.to,
              timestamp: new Date(),
              thinkTime: 0,
              moveNumber,
              option: 1,
              recoveryMode: 'line',
              extractionStacks: [],
            } as Move);
          } else {
            // Exact length: only Option 1
            recoveryMoves.push({
              id: `${baseId}`,
              type: 'recovery_slide',
              player,
              from: target.from,
              to: target.to,
              timestamp: new Date(),
              thinkTime: 0,
              moveNumber,
              option: 1,
              recoveryMode: 'line',
              extractionStacks: [],
            } as Move);
          }
        }
      }

      // Also filter recovery moves by mustMoveFromStackKey if set
      let filteredRecoveryMoves = recoveryMoves;
      if (state.mustMoveFromStackKey) {
        filteredRecoveryMoves = recoveryMoves.filter(
          (m) => m.from && positionToString(m.from) === state.mustMoveFromStackKey
        );
      }

      const allMoves = [...movements, ...captures, ...filteredRecoveryMoves];

      // RR-FIX-2026-01-12: When a player has no movement, capture, or recovery
      // options (e.g., they have no stacks on the board), return no_movement_action
      // to allow the turn to advance. Without this, the game stalls with an empty
      // valid moves array. This can happen when a player lost all their stacks but
      // still has rings in hand that they couldn't place.
      if (allMoves.length === 0) {
        return [
          {
            id: `no-movement-action-blocked-${moveNumber}`,
            type: 'no_movement_action',
            player,
            to: { x: 0, y: 0 }, // Sentinel value for bookkeeping move
            timestamp: new Date(),
            thinkTime: 0,
            moveNumber,
          } as Move,
        ];
      }

      return allMoves;
    }

    case 'capture': {
      // RR-CANON-R070 / Section 4.3: In capture phase (after non-capturing movement),
      // the player may optionally make overtaking captures or skip to line processing.
      // Only captures from the last moved stack are valid (not all stacks).
      const lastMove =
        state.moveHistory.length > 0 ? state.moveHistory[state.moveHistory.length - 1] : null;
      const attackerPos = lastMove?.to;

      if (!attackerPos) {
        // No last move position, can't determine attacker - return empty
        return [];
      }

      // Get all captures and filter to only those from the attacker position
      const allCaptures = enumerateAllCaptureMoves(state, player);
      const attackerKey = positionToString(attackerPos);
      const captures = allCaptures.filter(
        (m) => m.from && positionToString(m.from) === attackerKey
      );

      const moves: Move[] = [...captures];

      // Add skip option to proceed to line processing without capturing
      // (always available in capture phase since captures are optional)
      moves.push({
        id: `skip-capture-${moveNumber}`,
        type: 'skip_capture',
        player,
        to: { x: 0, y: 0 },
        timestamp: new Date(),
        thinkTime: 0,
        moveNumber,
      } as Move);

      return moves;
    }

    case 'chain_capture': {
      // Get capture continuations from the chain capture position.
      // Per RR-CANON-R084/R085, chain captures must continue from the landing
      // position of the previous capture segment, stored in chainCapturePosition.
      let chainPos = state.chainCapturePosition;
      if (!chainPos) {
        // No chain position set - fall back to last move's landing position
        // (this handles states loaded from fixtures that may not have chainCapturePosition)
        const lastMove =
          state.moveHistory.length > 0 ? state.moveHistory[state.moveHistory.length - 1] : null;
        if (!lastMove?.to) {
          return [];
        }
        chainPos = lastMove.to;
      }

      // Use getChainCaptureContinuationInfo which returns properly-typed
      // 'continue_capture_segment' moves, consistent with what the sandbox
      // and other hosts expect during chain_capture phase.
      const info = getChainCaptureContinuationInfo(state, player, chainPos);
      return info.availableContinuations;
    }

    case 'line_processing': {
      // Use 'detect_now' to ensure fresh line detection. The default
      // 'use_board_cache' mode may return no moves when board.formedLines
      // is empty or stale, causing a mismatch with hasPhaseLocalInteractiveMove
      // which always uses fresh detection.
      const lineMoves = enumerateProcessLineMoves(state, player, { detectionMode: 'detect_now' });

      // RR-FIX-2026-01-11: When line reward elimination is pending, include elimination moves.
      // After a line collapse that grants an elimination reward, the player must select a
      // stack to eliminate from. Without this, getValidMoves() returns only process_line
      // moves, causing the sandbox to appear stuck.
      if (state.pendingLineRewardElimination) {
        const eliminationMoves: Move[] = [];

        for (const [key, stack] of state.board.stacks.entries()) {
          if (stack.controllingPlayer !== player) continue;
          const capHeight = stack.capHeight ?? 0;
          if (capHeight <= 0) continue;

          eliminationMoves.push({
            id: `eliminate-line-${key}`,
            type: 'eliminate_rings_from_stack',
            player,
            to: stack.position,
            eliminatedRings: [{ player, count: 1 }], // Line cost is always 1 ring
            eliminationContext: 'line',
            timestamp: new Date(),
            thinkTime: 0,
            moveNumber,
          } as Move);
        }

        return [...lineMoves, ...eliminationMoves];
      }

      return lineMoves;
    }

    case 'territory_processing': {
      const regionMoves = enumerateProcessTerritoryRegionMoves(state, player);
      const elimMoves = enumerateTerritoryEliminationMoves(state, player);

      const moves: Move[] = [...regionMoves, ...elimMoves];

      // When one or more regions are processable for this player and no
      // self-elimination decision is currently outstanding, expose an
      // explicit "skip territory processing" meta-move. This lets humans
      // and AIs opt out of processing some or all eligible regions while
      // still recording that decision in canonical history.
      if (regionMoves.length > 0 && elimMoves.length === 0) {
        moves.push({
          id: `skip-territory-${moveNumber}`,
          type: 'skip_territory_processing',
          player,
          to: { x: 0, y: 0 },
          timestamp: new Date(),
          thinkTime: 0,
          moveNumber,
        } as Move);
      }

      return moves;
    }

    case 'forced_elimination': {
      // Enumerate canonical forced-elimination options for the blocked
      // player. Each option becomes a `forced_elimination` Move whose
      // underlying elimination semantics are handled by
      // applyEliminateRingsFromStackDecision in applyMoveWithChainInfo.
      const options = enumerateForcedEliminationOptions(state, player);
      const moves: Move[] = options.map((opt) => {
        const capHeight = typeof opt.capHeight === 'number' ? opt.capHeight : 0;
        const count = Math.max(1, capHeight || 0);
        return {
          id: opt.moveId,
          type: 'forced_elimination',
          player,
          to: opt.position,
          eliminatedRings: [{ player, count }],
          eliminationFromStack: {
            position: opt.position,
            capHeight,
            totalHeight: opt.stackHeight,
          },
          timestamp: new Date(),
          thinkTime: 0,
          moveNumber,
        } as Move;
      });
      return moves;
    }

    default:
      return [];
  }
}

/**
 * Check if the current player has any valid moves.
 */
export function hasValidMoves(state: GameState): boolean {
  return getValidMoves(state).length > 0;
}

/**
 * Determine whether the active player has performed any "real" actions during
 * the current turn, based on contiguous moveHistory entries for the current
 * player. Forced no-op bookkeeping moves (no_*_action) are ignored; voluntary
 * skips (skip_*) count as actions.
 */
function computeHadAnyActionThisTurn(
  state: GameState,
  currentMove?: Move,
  turnSequenceRealMoves?: Move[]
): boolean {
  const currentPlayer = state.currentPlayer;
  const history = state.moveHistory;

  // RR-PARITY-FIX-2025-12-21: Check turnSequenceRealMoves first.
  // This tracks non-bookkeeping moves that have been processed in the current
  // async turn sequence but haven't been added to history yet.
  if (turnSequenceRealMoves && turnSequenceRealMoves.length > 0) {
    for (const move of turnSequenceRealMoves) {
      if (move.player === currentPlayer && !isNoActionBookkeepingMove(move.type)) {
        return true;
      }
    }
  }

  // Include the current move being processed if provided.
  // This is needed for parity with Python, where the move is already
  // appended to move_history before phase transitions are computed.
  if (currentMove && currentMove.player === currentPlayer) {
    if (!isNoActionBookkeepingMove(currentMove.type)) {
      return true;
    }
  }

  for (let i = history.length - 1; i >= 0; i--) {
    const move = history[i];
    if (move.player !== currentPlayer) {
      break;
    }
    if (!isNoActionBookkeepingMove(move.type)) {
      return true;
    }
  }

  return false;
}

/**
 * Check whether the given player currently controls at least one stack on the
 * board (stackHeight > 0 and controllingPlayer === player).
 */
function playerHasStacksOnBoard(state: GameState, player: number): boolean {
  for (const stack of state.board.stacks.values()) {
    if (stack.controllingPlayer === player && stack.stackHeight > 0) {
      return true;
    }
  }
  return false;
}

/**
 * Identify move types that do NOT change the board state and thus do NOT
 * prevent forced elimination from triggering.
 *
 * Per RR-CANON-R072/R100: Forced elimination triggers when a player controls
 * stacks but made no "board state change" during their turn. This includes:
 *
 * 1. **Forced no-ops (NO_*_ACTION)**: Player entered a phase with no options.
 *    These are bookkeeping moves that record the phase was visited.
 *
 * 2. **Voluntary skips (SKIP_*)**: Player chose to skip an optional action.
 *    These don't change the board and thus don't prevent forced elimination.
 *    - skip_placement: Player has rings but chose not to place
 *    - skip_territory_processing: Player has territory options but chose to skip
 *    - skip_capture: Player has capture available but chose not to take it
 *    - skip_recovery: Player has recovery slide available but chose to skip
 *
 * Note: For LPS (Last Player Standing) purposes, the distinction between
 * voluntary skip vs forced no-op matters for determining if a player has
 * "real actions available". For forced elimination gating, the criterion is
 * simpler: did the player change the board state during their turn?
 */
function isNoActionBookkeepingMove(type: MoveType): boolean {
  return (
    // Forced no-ops (player had no choice)
    type === 'no_placement_action' ||
    type === 'no_movement_action' ||
    type === 'no_line_action' ||
    type === 'no_territory_action' ||
    // Voluntary skips (player chose not to act, but didn't change board)
    type === 'skip_placement' ||
    type === 'skip_territory_processing' ||
    type === 'skip_capture' ||
    type === 'skip_recovery'
  );
}

/**
 * Result of applying a single move for replay/snapshot purposes.
 */
export interface ApplyMoveForReplayResult {
  /** The game state after applying the move */
  nextState: GameState;
  /** For capture moves, whether a chain capture continuation is required */
  chainCaptureRequired?: boolean;
  /** For capture moves, the position from which chain capture must continue */
  chainCapturePosition?: Position;
}

/**
 * Apply a single move to a game state for replay/snapshot reconstruction purposes.
 *
 * This is a lower-level function than `processTurn` - it applies the move's effects
 * to the board but does NOT run full turn orchestration (victory checks, automatic
 * phase transitions, etc.). Use this when replaying a sequence of moves that were
 * already validated and processed, such as when reconstructing snapshots from
 * move history.
 *
 * @param state - The current game state
 * @param move - The move to apply
 * @returns The resulting state after applying the move
 */
export function applyMoveForReplay(state: GameState, move: Move): ApplyMoveForReplayResult {
  const result = applyMoveWithChainInfo(state, move);
  const replayResult: ApplyMoveForReplayResult = {
    nextState: result.nextState,
  };
  if (result.chainCaptureRequired !== undefined) {
    replayResult.chainCaptureRequired = result.chainCaptureRequired;
  }
  if (result.chainCapturePosition !== undefined) {
    replayResult.chainCapturePosition = result.chainCapturePosition;
  }
  return replayResult;
}
