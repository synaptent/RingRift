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
} from '../globalActions';

import type {
  ProcessTurnResult,
  PendingDecision,
  TurnProcessingDelegates,
  ProcessingMetadata,
  VictoryState,
  DetectedLineInfo,
  PlayerScore,
} from './types';

import { PhaseStateMachine, createTurnProcessingState } from './phaseStateMachine';
import { flagEnabled, debugLog } from '../../utils/envFlags';

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

import {
  getProcessableTerritoryRegions,
  enumerateProcessTerritoryRegionMoves,
  enumerateTerritoryEliminationMoves,
  applyProcessTerritoryRegionDecision,
  applyEliminateRingsFromStackDecision,
} from '../aggregates/TerritoryAggregate';

import { evaluateVictory } from '../aggregates/VictoryAggregate';

import { countRingsOnBoardForPlayer } from '../core';

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
 * aligned with docs/UX_RULES_COPY_SPEC.md and
 * docs/UX_RULES_EXPLANATION_MODEL_SPEC.md.
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
 */
function hasForcedEliminationMove(state: GameState): boolean {
  // Prefer structured history when available.
  if (state.history && state.history.length > 0) {
    if (state.history.some((entry) => entry.action && entry.action.type === 'forced_elimination')) {
      return true;
    }
  }

  // Fallback to legacy moveHistory.
  return state.moveHistory.some((move) => move.type === 'forced_elimination');
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
      const current = queue.shift()!;
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
 */
function createRegionOrderDecision(state: GameState, regions: Territory[]): PendingDecision {
  const player = state.currentPlayer;
  const regionMoves = enumerateProcessTerritoryRegionMoves(state, player);
  const elimMoves = enumerateTerritoryEliminationMoves(state, player);

  const moves: Move[] = [...regionMoves];

  // When one or more regions are processable for this player and no
  // self-elimination decision is currently outstanding, include the
  // canonical skip_territory_processing Move alongside the explicit
  // process_territory_region options. This keeps the PendingDecision
  // surface aligned with getValidMoves for territory_processing.
  if (regionMoves.length > 0 && elimMoves.length === 0) {
    const moveNumber = state.moveHistory.length + 1;
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

// ═══════════════════════════════════════════════════════════════════════════
// Process Turn (Synchronous Entry Point)
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Options for processTurn behavior.
 */
export interface ProcessTurnOptions {
  /**
   * When true, even single territory regions will return a decision instead
   * of being auto-processed. This is used in replay contexts where explicit
   * process_territory_region moves from recordings should be used.
   */
  skipSingleTerritoryAutoProcess?: boolean;

  /**
   * When true, line-processing is never auto-applied (even for a single
   * exact-length line). Instead, a `line_order` decision is surfaced and
   * the host is expected to apply an explicit `process_line` /
   * `choose_line_reward` move. This is primarily used in replay/trace
   * contexts so that recorded move sequences remain the sole source of
   * truth for when lines are processed.
   */
  skipAutoLineProcessing?: boolean;
}

/**
 * Process a single move synchronously.
 *
 * This is the main entry point when decisions can be made immediately
 * (e.g., single-line/single-region cases or AI auto-selection).
 *
 * For cases requiring async decision resolution (human player choices),
 * use processTurnAsync or check the 'awaiting_decision' status.
 */
export function processTurn(
  state: GameState,
  move: Move,
  options?: ProcessTurnOptions
): ProcessTurnResult {
  // Enforce canonical phase→MoveType mapping for ACTIVE states. This ensures
  // that every visited phase is represented by an explicit action, skip, or
  // no-action move per RR-CANON-R075.
  assertPhaseMoveInvariant(state, move);

  const sInvariantBefore = computeSInvariant(state);
  const startTime = Date.now();
  const stateMachine = new PhaseStateMachine(createTurnProcessingState(state, move));

  // Compute hash before applying move to detect if the move actually changed state
  const hashBefore = hashGameState(stateMachine.gameState);

  // Apply the move based on type
  const applyResult = applyMoveWithChainInfo(stateMachine.gameState, move);
  stateMachine.updateGameState(applyResult.nextState);

  // Compute hash after applying move
  const hashAfter = hashGameState(stateMachine.gameState);
  const moveActuallyChangedState = hashBefore !== hashAfter;

  // If this was a capture move and chain continuation is required, set the chain capture state
  if (applyResult.chainCaptureRequired && applyResult.chainCapturePosition) {
    stateMachine.setChainCapture(true, applyResult.chainCapturePosition);
  }

  // For placement moves, don't process post-move phases - the player still needs to move
  // Post-move phases (lines, territory) only happen after movement/capture
  const isPlacementMove =
    move.type === 'place_ring' ||
    move.type === 'skip_placement' ||
    move.type === 'no_placement_action';

  // For decision moves (process_territory_region, eliminate_rings_from_stack, etc.),
  // if the move didn't actually change state (e.g., Q23 prerequisite not met), don't
  // process post-move phases - just return the unchanged state.
  const isDecisionMove =
    move.type === 'process_territory_region' ||
    move.type === 'eliminate_rings_from_stack' ||
    move.type === 'process_line' ||
    move.type === 'choose_line_reward';

  // For turn-ending territory moves, the turn is complete - no post-move processing needed.
  // The applyMoveWithChainInfo handler already rotates to the next player.
  const isTurnEndingTerritoryMove =
    move.type === 'no_territory_action' || move.type === 'skip_territory_processing';

  let result: { pendingDecision?: PendingDecision; victoryResult?: VictoryState } = {};
  if (
    !isPlacementMove &&
    !isTurnEndingTerritoryMove &&
    (moveActuallyChangedState || !isDecisionMove)
  ) {
    // Process post-move phases only for movement/capture moves, or for decision moves
    // that actually changed state.
    result = processPostMovePhases(stateMachine, options);
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

  // If the turn is otherwise complete but the current player is blocked
  // with stacks and only a forced-elimination action is available, surface
  // that action as an explicit decision rather than applying a hidden
  // host-level tie-breaker. This implements RR-CANON-R206 for hosts that
  // drive decisions via PendingDecision/Move.
  //
  // Per the 7-phase model (RR-CANON-R070), forced_elimination is a distinct
  // phase entered only when the player had no actions in prior phases but
  // still controls stacks.
  if (finalStatus === 'complete' && finalState.gameStatus === 'active') {
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

  return {
    nextState: finalState,
    status: finalPendingDecision ? 'awaiting_decision' : 'complete',
    pendingDecision: finalPendingDecision,
    victoryResult: finalVictory,
    metadata,
  };
}

/**
 * Result of applying a move, including chain capture info for capture moves.
 */
interface ApplyMoveResult {
  nextState: GameState;
  chainCaptureRequired?: boolean;
  chainCapturePosition?: Position;
}

/**
 * Apply a move to the game state based on move type.
 * For capture moves, also returns chain capture continuation info.
 */
function applyMoveWithChainInfo(state: GameState, move: Move): ApplyMoveResult {
  switch (move.type) {
    case 'place_ring': {
      const action = {
        type: 'PLACE_RING' as const,
        playerId: move.player,
        position: move.to,
        count: move.placementCount ?? 1,
      };
      const newState = mutatePlacement(state, action);
      // After placement, advance to movement phase
      return {
        nextState: {
          ...newState,
          currentPhase: 'movement' as GamePhase,
        },
      };
    }

    case 'skip_placement': {
      // Skip placement is a no-op on state, just phase transition
      return {
        nextState: {
          ...state,
          currentPhase: 'movement' as GamePhase,
        },
      };
    }

    case 'no_placement_action': {
      // Explicit forced no-op in ring_placement when the player has no legal
      // placement anywhere (RR-CANON-R075). State is unchanged; advance to
      // movement so that the rest of the turn can proceed.
      return {
        nextState: {
          ...state,
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

    case 'move_stack':
    case 'move_ring': {
      if (!move.from) {
        throw new Error('Move.from is required for movement moves');
      }
      const outcome = applySimpleMovement(state, {
        from: move.from,
        to: move.to,
        player: move.player,
      });
      return { nextState: outcome.nextState };
    }

    case 'no_movement_action': {
      // Explicit forced no-op in movement phase when the player has no legal
      // movement or capture anywhere (RR-CANON-R075). State is unchanged;
      // post-move phase logic will advance to line_processing.
      return { nextState: state };
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
      // Return chain capture info so the orchestrator can set state machine flags
      if (outcome.chainContinuationRequired) {
        return {
          nextState: outcome.nextState,
          chainCaptureRequired: true,
          chainCapturePosition: move.to,
        };
      }
      return { nextState: outcome.nextState };
    }

    case 'process_line': {
      const outcome = applyProcessLineDecision(state, move);
      return { nextState: outcome.nextState };
    }

    case 'choose_line_reward': {
      const outcome = applyChooseLineRewardDecision(state, move);
      return { nextState: outcome.nextState };
    }

    case 'no_line_action': {
      // Explicit no-op in line_processing phase when no lines exist for the
      // current player (RR-CANON-R075). State is unchanged; advance to
      // territory_processing so that the rest of the turn can proceed.
      return {
        nextState: {
          ...state,
          currentPhase: 'territory_processing' as GamePhase,
        },
      };
    }

    case 'process_territory_region': {
      const outcome = applyProcessTerritoryRegionDecision(state, move);
      return { nextState: outcome.nextState };
    }

    case 'eliminate_rings_from_stack': {
      const outcome = applyEliminateRingsFromStackDecision(state, move);
      return { nextState: outcome.nextState };
    }

    case 'no_territory_action': {
      // Explicit no-op in territory_processing phase when no regions exist
      // for the current player (RR-CANON-R075). Rotate to next player and
      // start their turn in ring_placement phase.
      const players = state.players;
      const currentPlayerIndex = players.findIndex((p) => p.playerNumber === state.currentPlayer);
      const nextPlayerIndex = (currentPlayerIndex + 1) % players.length;
      const nextPlayer = players[nextPlayerIndex].playerNumber;

      return {
        nextState: {
          ...state,
          currentPlayer: nextPlayer,
          currentPhase: 'ring_placement' as GamePhase,
        },
      };
    }

    case 'skip_territory_processing': {
      // Explicit skip in territory_processing phase when player opts out of
      // processing available regions. Rotate to next player and start their
      // turn in ring_placement phase.
      const players = state.players;
      const currentPlayerIndex = players.findIndex((p) => p.playerNumber === state.currentPlayer);
      const nextPlayerIndex = (currentPlayerIndex + 1) % players.length;
      const nextPlayer = players[nextPlayerIndex].playerNumber;

      return {
        nextState: {
          ...state,
          currentPlayer: nextPlayer,
          currentPhase: 'ring_placement' as GamePhase,
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
 *     place_ring, skip_placement, no_placement_action
 * - movement:
 *     move_stack, move_ring, overtaking_capture,
 *     continue_capture_segment, no_movement_action
 * - capture:
 *     overtaking_capture, continue_capture_segment, skip_capture
 * - chain_capture:
 *     overtaking_capture, continue_capture_segment
 * - line_processing:
 *     process_line, choose_line_reward, no_line_action
 * - territory_processing:
 *     process_territory_region, eliminate_rings_from_stack,
 *     skip_territory_processing, no_territory_action
 * - forced_elimination:
 *     forced_elimination
 *
 * swap_sides is permitted in any phase as a meta-move. Legacy move
 * types (line_formation, territory_claim) are accepted for historical
 * recordings but should be treated as non-canonical by hosts.
 */
function assertPhaseMoveInvariant(state: GameState, move: Move): void {
  const phase = state.currentPhase;
  const type = move.type;

  // Meta-move allowed in any phase
  if (type === 'swap_sides') {
    return;
  }

  // Legacy / experimental – allow to avoid breaking historical logs,
  // but these recordings should not be considered canonical.
  if (type === 'line_formation' || type === 'territory_claim') {
    return;
  }

  let allowed: Set<MoveType>;

  switch (phase) {
    case 'ring_placement':
      allowed = new Set<MoveType>(['place_ring', 'skip_placement', 'no_placement_action']);
      break;
    case 'movement':
      allowed = new Set<MoveType>([
        'move_stack',
        'move_ring',
        'overtaking_capture',
        'continue_capture_segment',
        'no_movement_action',
      ]);
      break;
    case 'capture':
      allowed = new Set<MoveType>([
        'overtaking_capture',
        'continue_capture_segment',
        'skip_capture',
      ]);
      break;
    case 'chain_capture':
      allowed = new Set<MoveType>(['overtaking_capture', 'continue_capture_segment']);
      break;
    case 'line_processing':
      allowed = new Set<MoveType>(['process_line', 'choose_line_reward', 'no_line_action']);
      break;
    case 'territory_processing':
      allowed = new Set<MoveType>([
        'process_territory_region',
        'eliminate_rings_from_stack',
        'skip_territory_processing',
        'no_territory_action',
      ]);
      break;
    case 'forced_elimination':
      allowed = new Set<MoveType>(['forced_elimination']);
      break;
    default:
      // Unknown phase: do not enforce
      return;
  }

  if (!allowed.has(type as MoveType)) {
    throw new Error(`[PHASE_MOVE_INVARIANT] Cannot apply move type '${type}' in phase '${phase}'`);
  }
}

/**
 * Process post-move phases (lines, territory, victory check).
 */
function processPostMovePhases(
  stateMachine: PhaseStateMachine,
  _options?: ProcessTurnOptions
): {
  pendingDecision?: PendingDecision;
  victoryResult?: VictoryState;
} {
  const state = stateMachine.gameState;
  const originalMoveType = stateMachine.processingState.originalMove.type;

  // Handle elimination phase completion - skip straight to victory check
  // and turn rotation since the elimination is the final action in the player's turn.
  // Per RR-CANON-R070, forced_elimination is the 7th and final phase.
  // Also handle eliminate_rings_from_stack during territory_processing - this is
  // the elimination that follows territory collapse (Q23 precondition).
  if (
    originalMoveType === 'forced_elimination' ||
    originalMoveType === 'eliminate_rings_from_stack'
  ) {
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

    // Rotate to next player without skipping. This matches Python's _end_turn logic
    // which does NOT skip empty seats - players with no stacks and no rings in hand
    // must still traverse phases and record no-action/FE moves per RR-CANON-R075/LPS rules.
    const currentState = stateMachine.gameState;
    const players = currentState.players;
    const currentPlayerIndex = players.findIndex(
      (p) => p.playerNumber === currentState.currentPlayer
    );

    // Simply rotate to the next player in turn order (no skipping)
    const nextPlayerIndex = (currentPlayerIndex + 1) % players.length;
    const nextPlayer = players[nextPlayerIndex].playerNumber;

    // Always begin the next turn in ring_placement. When no legal placements exist
    // (including ringsInHand == 0), hosts must emit a NO_PLACEMENT_ACTION bookkeeping
    // move based on the phase requirements. This matches Python's _end_turn behavior.
    stateMachine.updateGameState({
      ...currentState,
      currentPlayer: nextPlayer,
      currentPhase: 'ring_placement',
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

  // RR-CANON-R070 / Section 4.3: After a non-capturing movement (move_stack/move_ring),
  // the player may opt to make an overtaking capture before proceeding to line processing.
  // Check if the original move was a simple movement and if capture opportunities exist
  // FROM THE LANDING POSITION (not from all stacks).
  const wasSimpleMovement = originalMoveType === 'move_stack' || originalMoveType === 'move_ring';
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
    stateMachine.transitionTo('line_processing');
  }

  // Process lines
  if (stateMachine.currentPhase === 'line_processing') {
    const lines = findAllLines(state.board).filter((l) => l.player === state.currentPlayer);

    if (lines.length > 0) {
      // Core rules: never auto-generate or auto-apply process_line moves.
      // Surface a line_order decision so hosts can construct and apply
      // explicit process_line / choose_line_reward moves that will be
      // recorded in canonical history (RR-CANON-R075/R076).
      const detectedLines = lines.map((l) => ({
        positions: l.positions,
        player: l.player,
        length: l.length,
        direction: l.direction,
        collapseOptions: [],
      }));
      stateMachine.setPendingLines(detectedLines);
      return {
        pendingDecision: createLineOrderDecision(state, detectedLines),
      };
    }

    // No lines exist for the current player. If the original move was not
    // already a line-phase move, surface a required no_line_action decision
    // so hosts can emit an explicit no_line_action bookkeeping move.
    const isLinePhaseMove =
      originalMoveType === 'no_line_action' ||
      originalMoveType === 'process_line' ||
      originalMoveType === 'choose_line_reward';
    if (!isLinePhaseMove) {
      return {
        pendingDecision: {
          type: 'no_line_action_required',
          player: state.currentPlayer,
          options: [],
          context: {
            description: 'No lines to process - explicit no_line_action required per RR-CANON-R075',
          },
        },
      };
    }

    // Line phase move was applied and no more lines, continue to territory.
    stateMachine.transitionTo('territory_processing');
  }

  // Process territory
  if (stateMachine.currentPhase === 'territory_processing') {
    // If the original move was an explicit skip_territory_processing,
    // treat territory processing as complete for this turn and proceed
    // directly to victory/turn advancement.
    if (originalMoveType !== 'skip_territory_processing') {
      const regions = getProcessableTerritoryRegions(state.board, {
        player: state.currentPlayer,
      });

      if (regions.length > 1) {
        // Multiple regions: player must choose order via explicit
        // process_territory_region moves constructed by the host.
        return {
          pendingDecision: createRegionOrderDecision(state, regions),
        };
      } else if (regions.length === 1) {
        // Single region: per RR-CANON-R075/R076, the core rules layer does
        // not auto-apply PROCESS_TERRITORY_REGION. Surface a region_order
        // decision even when there is only one region; hosts may auto-select
        // the only option for live UX but must still emit the explicit move.
        return {
          pendingDecision: createRegionOrderDecision(state, regions),
        };
      } else {
        // regions.length === 0: No regions to process.
        // Per RR-CANON-R075/R076, return a pending decision requiring an explicit
        // no_territory_action move. The core rules layer does NOT auto-generate moves.
        // EXCEPTION: If the original move was already a territory-related move
        // (no_territory_action or process_territory_region), we don't need to return
        // another pending decision - that move IS the territory phase action.
        const isTerritoryPhaseMove =
          originalMoveType === 'no_territory_action' ||
          originalMoveType === 'process_territory_region';
        if (!isTerritoryPhaseMove) {
          return {
            pendingDecision: {
              type: 'no_territory_action_required',
              player: state.currentPlayer,
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
  const currentState = stateMachine.gameState;
  const players = currentState.players;
  const currentPlayerIndex = players.findIndex(
    (p) => p.playerNumber === currentState.currentPlayer
  );
  const nextPlayerIndex = (currentPlayerIndex + 1) % players.length;
  const nextPlayer = players[nextPlayerIndex].playerNumber;
  // Always begin the next turn in ring_placement. When no legal placements
  // exist for that player, hosts must emit an explicit no_placement_action
  // bookkeeping move (or advance directly to movement via shared turnLogic);
  // the core orchestrator no longer fabricates this move itself.
  const nextPhase: GamePhase = 'ring_placement';

  stateMachine.updateGameState({
    ...currentState,
    currentPlayer: nextPlayer,
    currentPhase: nextPhase,
  });

  return {};
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
  delegates: TurnProcessingDelegates
): Promise<ProcessTurnResult> {
  let result = processTurn(state, move);

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

    // Emit decision event if handler provided
    delegates.onProcessingEvent?.({
      type: 'decision_required',
      timestamp: new Date(),
      payload: { decision },
    });

    // Resolve the decision
    const chosenMove = await delegates.resolveDecision(decision);

    // Emit decision resolved event
    delegates.onProcessingEvent?.({
      type: 'decision_resolved',
      timestamp: new Date(),
      payload: { decision, chosenMove },
    });

    // Continue processing with the chosen move
    result = processTurn(result.nextState, chosenMove);
  }

  return result;
}

// ═══════════════════════════════════════════════════════════════════════════
// Utility Functions
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Check if a move is valid for the current game state.
 */
export function validateMove(state: GameState, move: Move): { valid: boolean; reason?: string } {
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

    case 'move_stack':
    case 'move_ring': {
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
 * - movement: move_stack, move_ring, overtaking_capture,
 *   continue_capture_segment.
 * - capture / chain_capture: capture segments + skip_capture.
 * - line_processing: process_line / choose_line_reward.
 * - territory_processing: process_territory_region /
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
      // Per RR-CANON-R076, the core layer does not fabricate
      // no_placement_action moves here; hosts must either:
      // - Advance to movement via the shared turnLogic helpers, or
      // - Construct an explicit no_placement_action move when required.
      if (!playerObj || playerObj.ringsInHand === 0) {
        return [];
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

      return moves;
    }

    case 'movement': {
      const movements = enumerateSimpleMovesForPlayer(state, player);
      const captures = enumerateAllCaptureMoves(state, player);
      return [...movements, ...captures];
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
      // Get capture continuations from current position
      // Note: This requires knowing the chain position, which needs context
      const captures = enumerateAllCaptureMoves(state, player);
      return captures;
    }

    case 'line_processing': {
      // Use 'detect_now' to ensure fresh line detection. The default
      // 'use_board_cache' mode may return no moves when board.formedLines
      // is empty or stale, causing a mismatch with hasPhaseLocalInteractiveMove
      // which always uses fresh detection.
      return enumerateProcessLineMoves(state, player, { detectionMode: 'detect_now' });
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
