/**
 * ═══════════════════════════════════════════════════════════════════════════
 * Victory Explanation Module
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * This module provides functions for building game end explanations from
 * victory results. It handles the mapping of victory conditions to
 * user-facing explanations including:
 *
 * - Ring elimination victories
 * - Territory control victories (including mini-region scenarios)
 * - Last player standing victories
 * - Structural stalemate and tiebreak scenarios
 *
 * The module is designed to be side-effect free and focuses purely on
 * constructing explanation data structures from game state and victory
 * results.
 *
 * @see docs/ux/UX_RULES_EXPLANATION_MODEL_SPEC.md
 */

import type { GameState, Position, BoardState } from '../types/game';
import { positionToString, BOARD_CONFIGS } from '../types/game';
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
} from './gameEndExplanation';
import { buildGameEndExplanationFromEngineView } from './gameEndExplanation';
import {
  getWeirdStateReasonForType,
  getWeirdStateReasonForGameResult,
  getTeachingTopicForReason,
} from './weirdStateReasons';
import type { VictoryResult as AggregateVictoryResult } from './aggregates/VictoryAggregate';
import type { PlayerScore, VictoryState } from './orchestration/types';

// ═══════════════════════════════════════════════════════════════════════════
// Type Definitions
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Result of detecting territory mini-regions for explanation purposes.
 */
export interface MiniRegionDetectionResult {
  /** Whether the victory qualifies as a mini-region scenario */
  isMiniRegionVictory: boolean;
  /** Number of disconnected territory regions controlled by the winner */
  regionCount: number;
  /** True if at least one region is considered "mini" (small isolated region) */
  hasMiniRegion: boolean;
}

// ═══════════════════════════════════════════════════════════════════════════
// Helper Functions
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Convert a player number to a player ID string.
 */
export function toPlayerId(playerNumber: number): string {
  return `P${playerNumber}`;
}

/**
 * Create a score breakdown record from player scores.
 */
export function createScoreBreakdown(
  scores: PlayerScore[]
): Record<string, GameEndPlayerScoreBreakdown> {
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

/**
 * Derive a short summary key for the game end UX copy.
 */
export function deriveShortSummaryKey(
  outcomeType: GameEndOutcomeType,
  primaryConceptId?: string
): string {
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
export function deriveUxCopyKeys(
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
export function hasForcedEliminationMove(state: GameState): boolean {
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
export function detectTerritoryMiniRegions(
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
// Main Victory Explanation Builder
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Build a minimal GameEndExplanation from the current GameState and VictoryState.
 *
 * This keeps the explanation logic close to the canonical victory evaluation
 * while remaining side-effect free.
 */
export function buildGameEndExplanationForVictory(
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
    } as import('../types/game').GameResult);

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
